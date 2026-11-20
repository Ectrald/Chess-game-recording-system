# Проектирование и план реализации: ЭТАП 5 (STAGE/5)

**Тема этапа:** Подсистема долговременного хранения (Persistence), отказоустойчивости (Crash Recovery), криптографического аудита журналов событий и асинхронного шахматного анализа (Stockfish UCI Engine Integration).  
**Ветка реализации:** `STAGE/5`  
**Целевой статус:** От подтверждённых в оперативной памяти событий `MoveEvent` — к гарантированно сохранённому на диске аудируемому журналу (append-only JSONL с SHA-256 чексуммами), архиву доказательных кадров (`EvidenceWriter`), механизму холодного старта / восстановления после внезапных сбоев (`RecoveryManager`) и неблокирующему асинхронному аналитическому контуру с нормализацией оценок движка Stockfish и классификацией ходов (blunder/mistake/best).

---

## 1. Целевое состояние и обоснование архитектурных решений

### 1.1. Назначение и контекст этапа
В соответствии с концепцией распределения обязанностей ([ПЗ, подразд. 2.4.6, 2.4.7](file:///home/ectrald/Repos/Chess-game-recording-system/explanatory%20note/sections/sec_designing.tex#L465-L515), [05_REASONING_AND_ANALYSIS.md](../chess_game_recording_docs/05_REASONING_AND_ANALYSIS.md) и [09_ARCHITECTURE_DECISIONS.md](../chess_game_recording_docs/09_ARCHITECTURE_DECISIONS.md)):
1. **Регистрация партии не должна зависеть от стабильности внешних сервисов:** Шахматный движок Stockfish — это внешний ресурсоёмкий процесс, который может зависнуть, упасть из-за нехватки памяти или исчерпать вычислительный лимит. Никакой сбой, задержка или перезапуск Stockfish не имеют права задерживать видеопоток или приводить к отмене (rollback) уже совершённого игроком хода (ADR-010).
2. **Гарантия целостности данных при аварийном отключении питания:** Физические матчи за доской могут прерываться сбоем оборудования или отключением питания. Система обязана сохранять историю так, чтобы после перезапуска гарантированно восстановить официальное состояние партии до последнего подтвержденного полухода без дублирования ходов (ADR-011).
3. **Объективность шахматной оценки (Side-to-Move Normalization):** Оценки шахматных движков в протоколе UCI традиционно передаются либо с точки зрения белых, либо с точки зрения стороны, чей сейчас ход. Для корректного определения ошибки (blunder detection) потеря сантипешек ($\Delta \text{eval}$) обязана нормализоваться относительно игрока, совершившего ход.

---

### 1.2. Обоснование проектных решений

#### 1. Журнал событий Append-Only JSONL с криптографическим аудитом (ADR-011)
* **Проблема реляционных СУБД (SQLite/PostgreSQL):** Накладные расходы на поддержание схемы, блокировки транзакций и риск повреждения файла БД при внезапном сбое питания (`SIGKILL`).
* **Принятое решение:** Формат JSON Lines (`.jsonl`). Каждая строка — самодостаточный сериализованный JSON-объект `MoveEvent`. 
* **Цепочка целостности (Hash Chaining / SHA-256):** Каждая запись содержит криптографический хеш текущего события и хеш предыдущей записи (`prev_event_hash`). Это превращает журнал в неизменяемый аудируемый лог: любая ручная модификация нотации, FEN или временных меток мгновенно выявляется процедурой аудита при перезапуске.

#### 2. Атомарная запись доказательной базы (`EvidenceWriter`)
* **Разделение метаданных и бинарных данных:** Сами события хранятся в JSONL, а доказательные артефакты (кроп доски в момент завершения хода, тепловая карта движения, снимок ключевых полей хода) сохраняются в структурированном каталоге сессии `data/sessions/<session_id>/evidence/`.
* **Селективная политика хранения (Selective Retention):**
  * Для однозначных ходов ($\Delta \ge 0{,}30$, окклюзия $= 0$) сохраняется только сжатый JPEG-снимок доски в момент фиксации хода;
  * Для спорных ходов, потребовавших разрешения через `UNCERTAIN` или ручное подтверждение, сохраняется серия кадров окна верификации ($\pm 3$ кадра) и маски перекрытия рук.
  Это предотвращает лавинообразное переполнение диска при длительных многочасовых партиях.

#### 3. Восстановление после сбоев и идемпотентный Startup Replay (`RecoveryManager`)
* При инициализации сессии или старте системы `RecoveryManager` проверяет наличие незавершенных сессий:
  1. Читает журнал `events.jsonl` строка за строкой;
  2. Проверяет непрерывность FEN-цепочки и валидность SHA-256 контрольных сумм;
  3. В случае обнаружения оборванной строки на конце файла (эффект внезапного отключения питания в фазе записи) автоматически выполняет усечение (truncate) до последней валидной записи;
  4. Передает последовательность валидных `MoveEvent` в метод `GameStateManager.replay()`, восстанавливая доску в оперативной памяти с точностью до последнего полухода;
  5. Проверяет сохранность ключей идемпотентности, исключая дублирование уже учтённых ходов.

#### 4. Неблокирующая ограниченная очередь анализа (`Bounded Queue`)
* **Архитектура изоляции потоков:** Видеорегистрация и FSM работают в высокоприоритетном цикле реального времени (Capture Thread). Анализ Stockfish вынесен в отдельный рабочий поток (`AnalysisWorkerThread`), взаимодействующий с очередью ограниченного размера `queue.Queue(maxsize=10)`.
* **Политика переполнения (Backpressure Policy):** Если движок не успевает за быстрой серией ходов (например, блиц-цейтнот), очередь удерживает задачи до лимита. При переполнении старые задачи анализа не блокируют видеопоток; в журнал выставляется статус `AnalysisTaskStatus.SKIPPED_QUEUE_FULL`, и анализ продолжается для актуальных позиций.
* **Изоляция сбоев:** Никакое исключение в аналитическом потоке не затрагивает состояние `GameStateManager` и FSM (`engine failure не делает commit rollback`).

#### 5. Супервизор процесса Stockfish UCI с таймаутом и авторестартом
* **Управление процессом:** Запуск процесса движка через `chess.engine.popen_uci(path)`.
* **Таймаут анализа:** Жесткое ограничение времени расчета на один полуход (`analysis_timeout_sec = 2.0`). Если движок превышает таймаут:
  1. Запрос принудительно отменяется (`engine.ping()` / cancellation);
  2. Событие помечается статусом `TIMED_OUT`;
  3. Если процесс движка перестал отвечать на команды UCI (`isready`), супервизор завершает процесс (`SIGTERM`/`SIGKILL`) и перезапускает новый экземпляр движка (`restart()`).
* **Адаптивный Fallback при отсутствии бинарника (`MockStockfishEngine`):** Если исполняемый файл Stockfish не установлен в операционной системе, система автоматически переключается на встроенный легковесный анализатор `MockStockfishEngine` (материальный баланс + mobility score) либо переводит статус анализа в `DISABLED`. Это гарантирует 100% работоспособность системы в любых средах.

#### 6. Нормализация оценки и классификация ходов
* Оценка Stockfish преобразуется к перспективе игрока, сделавшего ход:
  $$\text{eval\_mover} = \begin{cases} +\text{eval\_white}, & \text{если ход сделали белые} \\ -\text{eval\_white}, & \text{если ход сделали чёрные} \end{cases}$$
* Потеря оценки (Centipawn Loss):
  $$\Delta \text{eval} = \text{eval}_{\text{after\_normalized}} - \text{eval}_{\text{before\_normalized}}$$
* Градация качества хода (по стандартам FIDE и Chess.com / Lichess):
  * **Best / Excellent:** $\Delta \text{eval} \ge -10$ cp (или лучший ход движка);
  * **Good:** $-10 > \Delta \text{eval} \ge -30$ cp;
  * **Inaccuracy:** $-30 > \Delta \text{eval} \ge -90$ cp;
  * **Mistake:** $-90 > \Delta \text{eval} \ge -200$ cp;
  * **Blunder:** $\Delta \text{eval} < -200$ cp (грубый зевок или упущенный мат);
  * **Mate:** специальная нормализация форсированного мата ($\pm 10000 \mp \text{moves}$).

---

## 2. Структура файлов и модулей этапа STAGE/5

```text
Chess-game-recording-system/
├── configs/
│   └── default_config.yaml                  # Обновление: секция persistence и расширение analysis
├── docs/stages/
│   └── STAGE_5.md                           # Настоящий документ спецификации
├── src/
│   ├── analysis/
│   │   ├── __init__.py                      # Экспорт EngineService, MoveEvaluator, StockfishWorker
│   │   ├── engine_service.py                # Асинхронный сервис с bounded queue и событиями
│   │   ├── move_evaluator.py                # Нормализация оценок, centipawn loss, классификация ходов
│   │   └── stockfish_worker.py              # UCI-клиент, таймауты, авторестарт, MockStockfishEngine
│   ├── core/
│   │   ├── config.py                        # Добавление PersistenceConfig и обновление AnalysisConfig
│   │   ├── datatypes.py                     # Аналитические контракты: MoveQuality, AnalysisResult, SessionManifest
│   │   └── __init__.py                      # Экспорт новых моделей
│   └── persistence/
│       ├── __init__.py                      # Экспорт EventRepository, EvidenceWriter, RecoveryManager
│       ├── event_repository.py              # JSONL append-only журнал, SHA-256 хеширование, fsync
│       ├── evidence_writer.py               # Селективное сохранение визуальных доказательств и кадров
│       └── recovery_manager.py              # Восстановление после сбоев, валидация цепочки, truncate
└── tests/
    ├── test_engine_service_queue.py         # Тесты очереди анализа, non-blocking video, backpressure
    ├── test_event_repository.py             # Тесты атомарной записи JSONL, SHA-256 чексумм, целостности
    ├── test_evidence_writer.py              # Тесты сохранения кадров, селективной политики хранения
    ├── test_move_evaluator.py               # Тесты нормализации перспективы, оценки мата и классов ошибок
    ├── test_persistence_analysis_e2e.py     # Сквозной тест: GameStateManager + Persistence + Analysis
    ├── test_recovery_manager.py             # Crash-injection тесты, восстановление оборванных файлов
    └── test_stockfish_worker.py             # Тесты UCI протокола, mock-движка, таймаутов и авторестарта
```

---

## 3. Спецификация контрактов данных (`src/core/datatypes.py` и `config.py`)

### 3.1. Аналитические контракты данных

```python
class MoveQuality(str, Enum):
    """Категория качества шахматного хода по классификации FIDE/компьютерного анализа."""
    BEST = "best"              # Наилучший ход по оценке движка
    EXCELLENT = "excellent"    # Практически оптимальный ход (потеря <= 10 cp)
    GOOD = "good"              # Надёжный позиционный ход (потеря <= 30 cp)
    INACCURACY = "inaccuracy"  # Неточность (потеря 30..90 cp)
    MISTAKE = "mistake"        # Ошибка (потеря 90..200 cp)
    BLUNDER = "blunder"        # Грубая ошибка / зевок (потеря > 200 cp или упущенный мат)
    BOOK = "book"              # Теоретический дебютный ход


class EngineTaskStatus(str, Enum):
    """Статус асинхронного аналитического задания."""
    QUEUED = "queued"          # В очереди на анализ
    RUNNING = "running"        # Вычисляется движком
    COMPLETED = "completed"    # Успешно завершён
    FAILED = "failed"          # Ошибка движка или сбой
    TIMED_OUT = "timed_out"    # Превышен лимит времени анализа
    SKIPPED = "skipped"        # Пропущен из-за переполнения очереди
    DISABLED = "disabled"      # Анализ отключён в конфигурации


class EngineEvaluation(BaseModel):
    """Единичная оценка шахматной позиции от Stockfish."""
    eval_cp: Optional[int] = Field(default=None, description="Оценка в сантипешках со стороны белых")
    mate_in: Optional[int] = Field(default=None, description="Число полуходов до мата (+N победа белых, -N чёрных)")
    best_move_uci: Optional[str] = Field(default=None, description="Рекомендуемый движком лучший ход")
    pv: List[str] = Field(default_factory=list, description="Основной вариант (Principal Variation) в UCI")
    depth: int = Field(default=0, ge=0, description="Достигнутая глубина расчёта")
    nodes: int = Field(default=0, ge=0, description="Количество просчитанных узлов дерева")
    time_sec: float = Field(default=0.0, ge=0.0, description="Затраченное время расчёта в секундах")


class AnalysisResult(BaseModel):
    """Полный результат сравнительного анализа совершенного хода."""
    event_id: str = Field(description="UUID анализируемого MoveEvent")
    ply: int = Field(ge=1, description="Номер полухода")
    move_uci: str = Field(description="Сыгранный ход в формате UCI")
    status: EngineTaskStatus = Field(description="Финальный статус анализа")
    eval_before: Optional[EngineEvaluation] = Field(default=None, description="Оценка позиции до хода")
    eval_after: Optional[EngineEvaluation] = Field(default=None, description="Оценка позиции после хода")
    eval_loss_cp: Optional[int] = Field(default=None, description="Потеря сантипешек с точки зрения ходившего игрока")
    quality: Optional[MoveQuality] = Field(default=None, description="Классификация качества хода")
    is_blunder: bool = Field(default=False, description="Флаг грубой ошибки (eval_loss > threshold)")
    error_message: Optional[str] = Field(default=None, description="Сообщение об ошибке при FAILED/TIMED_OUT")
```

### 3.2. Контракт паспорта игровой сессии (`SessionManifest`)

```python
class SessionManifest(BaseModel):
    """Метаданные и паспорт сессии записи шахматной партии."""
    session_id: str = Field(description="Уникальный строковый идентификатор сессии")
    created_at: str = Field(description="ISO 8601 временная метка начала записи партии")
    player_white: str = Field(default="White Player", description="Имя игрока белыми фигурами")
    player_black: str = Field(default="Black Player", description="Имя игрока чёрными фигурами")
    initial_fen: str = Field(default=chess.STARTING_FEN, description="Начальная расстановка фигур")
    current_fen: str = Field(description="Текущая / финальная позиция FEN")
    total_plies: int = Field(default=0, ge=0, description="Общее число зафиксированных полуходов")
    journal_path: str = Field(description="Относительный путь к файлу events.jsonl")
    pgn_path: str = Field(description="Относительный путь к экспортированному файлу game.pgn")
    journal_checksum: str = Field(default="", description="Финальная SHA-256 контрольная сумма журнала событий")
    is_completed: bool = Field(default=False, description="Признак штатного завершения матча")
```

### 3.3. Расширенный `MoveEvent`
В существующий класс `MoveEvent` добавляется поле контрольной суммы строки:
```python
class MoveEvent(BaseModel):
    ...
    prev_event_hash: Optional[str] = Field(default=None, description="SHA-256 хеш предшествующего события в журнале")
    event_hash: Optional[str] = Field(default=None, description="Собственный SHA-256 хеш события для аудита")
```

---

## 4. Спецификация алгоритмов и программных интерфейсов

### 4.1. Репозиторий событий (`src/persistence/event_repository.py`)

Класс `JSONLEventRepository`:
* **Атомарность записи:**
  1. Блокировка файлового рентабельного мьютекса (`threading.RLock`);
  2. Вычисление `event_hash = sha256(prev_event_hash + event_json_canonical)`;
  3. Сериализация в компактную JSON-строку;
  4. Запись в append-режиме (`"a"`, UTF-8);
  5. Принудительный сброс буферов операционной системы на физический диск: `file.flush()`, `os.fsync(file.fileno())`;
  6. Обновление внутреннего указателя последнего хеша.
* **Аудит целостности (`verify_integrity`):**
  - Последовательно читает файл с первой до последней строки;
  - Проверяет равенство вычисленных хешей и соответствие `event.ply == index + 1`;
  - Возвращает `Tuple[bool, Optional[str]]` (валиден ли файл, и описание ошибки в случае фальсификации или повреждения).

```python
class JSONLEventRepository:
    def __init__(self, session_dir: Path, filename: str = "events.jsonl") -> None: ...
    def append_event(self, event: MoveEvent) -> MoveEvent: ...
    def read_all_events(self) -> List[MoveEvent]: ...
    def verify_integrity(self) -> Tuple[bool, Optional[str]]: ...
    def close(self) -> None: ...
```

---

### 4.2. Доказательная база (`src/persistence/evidence_writer.py`)

Класс `EvidenceWriter`:
* Организует хранение визуальных доказательств в подкаталоге `evidence/`:
  * `evidence/ply_001_e2e4_frame_000120.jpg` — снимок доски в момент подтверждения хода;
  * `evidence/ply_001_e2e4_meta.json` — сохранённый срез наблюдения (уверенность, геометрия, окклюзия);
* **Селективная политика:**
  * Метод `save_evidence(event, observation, raw_board_image)`:
    - Проверяет `retention_policy` ("selective", "all", "none");
    - Если ход принят с высоким запасом уверенности (`margin >= 0.30` и `penalty_occlusion == 0`), сохраняет сжатый JPEG с качеством 85;
    - Если ход был неоднозначным (`UNCERTAIN`) или перекрыт рукой, сохраняет исходный PNG без потерь и метаданные поклеточных вероятностей.

```python
class EvidenceWriter:
    def __init__(self, evidence_dir: Path, policy: str = "selective") -> None: ...
    def save_evidence(
        self,
        event: MoveEvent,
        observation: BoardObservation,
        board_image: Optional[np.ndarray] = None,
    ) -> Optional[str]: ...
```

---

### 4.3. Менеджер восстановления (`src/persistence/recovery_manager.py`)

Класс `RecoveryManager`:
* **Процедура Crash Recovery при старте:**
  1. Проверяет наличие файла `events.jsonl` в каталоге сессии;
  2. Сканирует файл с конца в поиске поврежденных (неполных) строк, возникших из-за падения питания в момент записи;
  3. При обнаружении неполной строки обрезает файл до последнего валидного разделителя `\n` (`truncate`);
  4. Загружает список валидных `MoveEvent`;
  5. Передает события в `GameStateManager.replay(events)`;
  6. Если финальный FEN совпадает с ожидаемым, восстанавливает сессию и возвращает активный `GameStateManager`;
  7. При фатальном расхождении FEN генерирует диагностический отчет и переводит сессию в режим `RECOVERY_FAILED`.

```python
class RecoveryManager:
    def __init__(self, session_dir: Path) -> None: ...
    def recover_session(self) -> Tuple[GameStateManager, List[MoveEvent]]: ...
    def sanitize_corrupted_tail(self) -> int: ...
```

---

### 4.4. Клиент шахматного движка и Mock-эмулятор (`src/analysis/stockfish_worker.py`)

Класс `StockfishUCIWorker`:
* Инкапсулирует `chess.engine.SimpleEngine`;
* Автоматический поиск бинарника Stockfish:
  1. `config.stockfish_path` (если задан);
  2. Стандартные системные пути (`which stockfish`, `/usr/bin/stockfish`, `/usr/local/bin/stockfish`, `/usr/games/stockfish`);
  3. Если бинарник не обнаружен — прозрачный Fallback на `MockStockfishEngine`.
* **Метод `evaluate_position(board, depth, timeout_sec) -> EngineEvaluation`:**
  - Запускает анализ с ограничением `chess.engine.Limit(depth=depth, time=timeout_sec)`;
  - Оборачивает вызов в `asyncio.wait_for` / threading timeout;
  - При возникновении таймаута корректно прерывает расчет без блокировки потока;
  - При фатальном падении процесса движка автоматически перезапускает его через `_restart_engine()`.

Класс `MockStockfishEngine`:
* Легковесный встроенный аналитический движок, не требующий внешних бинарников;
* Рассчитывает материальный баланс фигур (пешка = 100 cp, конь/слон = 300 cp, ладья = 500 cp, ферзь = 900 cp) и мобильность короля;
* Генерирует полностью валидные структуры `EngineEvaluation`, позволяя тестировать всю систему без установки внешнего ПО.

```python
class BaseEngineWorker(ABC):
    @abstractmethod
    def evaluate(self, board: chess.Board, depth: int, timeout_sec: float) -> EngineEvaluation: ...
    @abstractmethod
    def restart(self) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
```

---

### 4.5. Нормализация оценок и классификатор ходов (`src/analysis/move_evaluator.py`)

Класс `MoveEvaluator`:
* **Нормализация перспективы:**
  - Извлекает оценку позиции до хода `eval_before` и после хода `eval_after`;
  - Если был ход белых:
    $$\Delta = \text{eval\_cp}_{\text{after}} - \text{eval\_cp}_{\text{before}}$$
  - Если был ход чёрных:
    $$\Delta = (-\text{eval\_cp}_{\text{after}}) - (-\text{eval\_cp}_{\text{before}}) = \text{eval\_cp}_{\text{before}} - \text{eval\_cp}_{\text{after}}$$
* **Обработка матовых позиций:**
  - Превращает `mate_in = +k` в эквивалент $+10000 - k \times 10$;
  - Превращает `mate_in = -k` в эквивалент $-10000 + k \times 10$.
* **Классификация качества хода:**
  - Если ход совпадает с `eval_before.best_move_uci` $\implies$ `MoveQuality.BEST`;
  - Иначе классифицирует потерю по шкале (Excellent, Good, Inaccuracy, Mistake, Blunder).

```python
class MoveEvaluator:
    def __init__(self, blunder_threshold_cp: int = 150) -> None: ...
    def evaluate_move(
        self,
        event: MoveEvent,
        eval_before: EngineEvaluation,
        eval_after: EngineEvaluation,
    ) -> AnalysisResult: ...
```

---

### 4.6. Асинхронная служба анализа (`src/analysis/engine_service.py`)

Класс `AsyncEngineService`:
* **Очередь задач:** `queue.Queue(maxsize=10)`;
* **Фоновый поток (`AnalysisWorkerThread`):**
  - Извлекает `AnalysisTask(event, fen_before, fen_after)`;
  - Запускает `engine_worker.evaluate()` для `fen_before` и `fen_after`;
  - Вычисляет `AnalysisResult` через `MoveEvaluator`;
  - Обновляет поля `eval_centipawns` и `is_blunder` в соответствующем `MoveEvent`;
  - Уведомляет подписчиков через колбэк `on_analysis_completed(result)` (UI Event Bus);
* **Защита от сбоев:**
  - При таймауте публикует `AnalysisResult(status=TIMED_OUT)`;
  - При падении движка публикует `AnalysisResult(status=FAILED)`;
  - Никогда не выбрасывает исключения в поток видеозахвата.

```python
class AsyncEngineService:
    def __init__(
        self,
        config: AnalysisConfig,
        on_analysis_completed: Optional[Callable[[AnalysisResult], None]] = None,
        on_analysis_failed: Optional[Callable[[str, str], None]] = None,
    ) -> None: ...
    def submit_move(self, event: MoveEvent) -> bool: ...
    def start(self) -> None: ...
    def stop(self, timeout_sec: float = 2.0) -> None: ...
```

---

## 5. Обновление конфигурации (`configs/default_config.yaml` и `src/core/config.py`)

```yaml
persistence:
  base_dir: "data/sessions"           # Корневой каталог сохранения игровых сессий
  enable_checksum: true               # Криптографический аудит целостности SHA-256
  evidence_format: "jpg"              # Формат сохранения кадров (jpg/png)
  evidence_retention: "selective"     # Политика хранения: all | selective | none
  save_full_frame_on_uncertain: true  # Сохранять несжатый кадр при статусе UNCERTAIN
  sync_to_disk: true                  # Принудительный os.fsync после каждого полухода

analysis:
  stockfish_enabled: true             # Включить фоновый шахматный анализ
  stockfish_depth: 15                 # Глубина перебора движка
  stockfish_path: null                # Путь к бинарнику (null = автопоиск в PATH + Mock fallback)
  blunder_threshold_cp: 150           # Порог классификации зевка (1.5 пешки)
  timeout_sec: 2.5                    # Максимальное время расчета на одну позицию
  max_queue_size: 10                  # Размер ограниченной очереди фоновых задач
  threads: 1                          # Число потоков CPU для Stockfish
  hash_mb: 32                         # Размер хэш-таблицы Stockfish в МБ
```

---

## 6. Комплексный план тестирования для STAGE/5

| Тестовый модуль | Проверяемый функционал | Критерий успешности |
| :--- | :--- | :--- |
| `tests/test_event_repository.py` | 1. Запись событий в append-only JSONL.<br>2. Формирование непрерывной цепочки SHA-256 хешей.<br>3. Верификация целостности: обнаружение модификации файла или искажения нотации.<br>4. Принудительный сброс через `os.fsync`. | Записи сериализуются без потерь; `verify_integrity()` подтверждает валидный файл и детектирует повреждения. |
| `tests/test_evidence_writer.py` | 1. Сохранение изображений доски в каталог сессии.<br>2. Селективное сохранение: сжатый JPEG для очевидных ходов, полный PNG для `UNCERTAIN`.<br>3. Корректная генерация путей к доказательствам. | Файлы артефактов создаются на диске; пути соответствуют `session_id/evidence/`. |
| `tests/test_recovery_manager.py` | 1. Crash-injection: симуляция внезапного обрыва последней строки файла (power cutoff).<br>2. Метод `sanitize_corrupted_tail()` обрезает оборванный хвост.<br>3. `recover_session()` восстанавливает авторитетный `GameStateManager` до последнего валидного полухода.<br>4. Replay совпадает с ожидаемым FEN. | Сессия успешно восстанавливается после симуляции аварии питания; дублирование ходов исключено. |
| `tests/test_stockfish_worker.py` | 1. Инициализация UCI-воркера и fallback на `MockStockfishEngine`.<br>2. Оценка позиций: material balance + mobility score в mock-режиме.<br>3. Обработка таймаута: прерывание расчета при превышении `timeout_sec`.<br>4. Перезапуск зависшего процесса движка (`restart()`). | Воркер корректно возвращает `EngineEvaluation`; таймауты и сбои обрабатываются без зависания процесса. |
| `tests/test_move_evaluator.py` | 1. Нормализация оценки с точки зрения ходившего игрока (White vs Black).<br>2. Расчет потери сантипешек $\Delta \text{eval}$.<br>3. Преобразование оценок мата в фиксированные очки.<br>4. Корректная классификация ходов (`BEST`, `GOOD`, `INACCURACY`, `MISTAKE`, `BLUNDER`). | Классы ходов строго соответствуют шкале потерь; зевки маркируются `is_blunder = True`. |
| `tests/test_engine_service_queue.py` | 1. Асинхронное выполнение анализа в фоновом потоке.<br>2. Bounded queue: защита от переполнения при высокой скорости подачи ходов.<br>3. Изоляция: сбой или таймаут анализа не блокирует поток видеорегистрации и не вызывает отката хода.<br>4. Доставка событий `on_analysis_completed` и `on_analysis_failed`. | Видеопоток не ждёт окончания анализа; результаты доставляются асинхронно через события. |
| `tests/test_persistence_analysis_e2e.py` | 1. Сквозная интеграция: совершение ходов в `GameStateManager` $\to$ запись в `EventRepository` $\to$ сохранение доказательств $\to$ постановка в `AsyncEngineService` $\to$ обогащение `MoveEvent` оценками.<br>2. Воспроизведение короткой партии `test_short_game.pgn`.<br>3. Проверка журнала сессии после завершения игры. | Полный цикл сохранения и анализа выполняется на 100% без ошибок; журнал готов к экспорту. |

---

## 7. Критерии приёмки (Definition of Done) этапа STAGE/5

Этап `STAGE/5` считается завершённым при выполнении следующих требований:

1. [x] Реализованы модели данных `MoveQuality`, `EngineTaskStatus`, `EngineEvaluation`, `AnalysisResult`, `SessionManifest` в `src/core/datatypes.py`.
2. [x] Добавлен конфигурационный класс `PersistenceConfig` и расширен `AnalysisConfig` в `src/core/config.py` и `configs/default_config.yaml`.
3. [x] Реализован репозиторий событий `JSONLEventRepository` (`src/persistence/event_repository.py`) с атомарной записью, сбросом `os.fsync`, криптографическими SHA-256 хешами и верификацией целостности.
4. [x] Реализован модуль сохранения доказательной базы `EvidenceWriter` (`src/persistence/evidence_writer.py`) с поддержкой селективной политики хранения доказательств.
5. [x] Реализован менеджер восстановления `RecoveryManager` (`src/persistence/recovery_manager.py`), успешно восстанавливающий состояние партии после сбоев и устраняющий неполные строки.
6. [x] Реализован UCI-воркер `StockfishUCIWorker` с автопоиском бинарника, строгими таймаутами, авторестартом и встроенным `MockStockfishEngine` для сред без предустановленного Stockfish.
7. [x] Реализован модуль нормализации оценок и классификации качества ходов `MoveEvaluator` (`src/analysis/move_evaluator.py`).
8. [x] Реализована асинхронная служба анализа `AsyncEngineService` (`src/analysis/engine_service.py`) с ограниченной очередью (`Bounded Queue`), потокобезопасностью и событиями для UI.
9. [x] Доказана изоляция: таймаут, зависание или падение шахматного движка не блокируют видеозахват и не приводят к rollback зафиксированных ходов.
10. [x] Разработан полный набор тестов (`test_event_repository.py`, `test_evidence_writer.py`, `test_recovery_manager.py`, `test_stockfish_worker.py`, `test_move_evaluator.py`, `test_engine_service_queue.py`, `test_persistence_analysis_e2e.py`).
11. [x] Crash-injection тесты подтверждают 100% корректное восстановление журнала и состояния доски после смоделированного отключения питания.
12. [x] Полный комплекс тестов `pytest tests/` (Этапы 0, 1, 2, 3, 4, 5) выполняется на 100% без единой ошибки.

