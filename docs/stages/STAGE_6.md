# Проектирование и план реализации: ЭТАП 6 (STAGE/6)

**Тема этапа:** Пользовательский интерфейс (UI / CLI Dashboard), сквозной системный оркестратор (`SystemOrchestrator`), интерактивное разрешение неопределённостей, экспорт результатов и финализация пояснительной записки (Раздел 3: «Реализация, тестирование и демонстрация гибридной интеллектуальной системы»).  
**Ветка реализации:** `STAGE/6`  
**Целевой статус:** Объединение всех слоев системы (Компьютерное зрение $\to$ Нейросетевой классификатор $\to$ Конечно-автоматный дебаунсер $\to$ Логико-символьный решатель $\to$ Транзакционный репозиторий $\to$ Асинхронный шахматный анализ) в законченный программный продукт, готовый к запуску, демонстрации и защите выпускной квалификационной работы.

---

## 1. Целевое состояние и обоснование архитектурных решений

### 1.1. Назначение и контекст этапа
В соответствии с методологией проектирования ([ПЗ, разделы 2.4, 3](file:///home/ectrald/Repos/Chess-game-recording-system/explanatory%20note/sections/sec_designing.tex), [ROADMAP.md](../../ROADMAP.md) и [08_IMPLEMENTATION_PLAN.md](../chess_game_recording_docs/08_IMPLEMENTATION_PLAN.md)):
1. **Сквозная связность (End-to-End Orchestration):** До Этапа 6 каждый модуль (локализатор, детектор рук, FSM, классификатор, скорер, шлюз решений, репозиторий, движок) тестировался изолированно или парами. Этап 6 предоставляет единый дирижирующий фасад `SystemOrchestrator`, связывающий первичный поток видеокадров с авторитетным состоянием игры и записью на диск в реальном времени.
2. **Двухуровневый пользовательский интерфейс (CLI + Visual HUD Dashboard):**
   * **CLI / Headless Mode:** Для автоматизированного пакетного тестирования, работы на серверах без дисплея (CI/CD) и стендовых испытаний;
   * **Interactive Visual HUD Dashboard (OpenCV HighGUI):** Интерактивное рабочее место оператора матча и тренера: наложение полигона доски, виртуальная 2D-доска, список ходов SAN, панель статусов FSM, шкала преимущества Stockfish и диалог подтверждения при статусе `UNCERTAIN`.
3. **Финализация академической части:** Полное заполнение раздела 3 пояснительной записки ([sec_development.tex](file:///home/ectrald/Repos/Chess-game-recording-system/explanatory%20note/sections/sec_development.tex)) на основе фактической программной реализации с расчётом финальных метрик качества, демонстрационных сценариев и сравнительного анализа гибридного подхода.

---

### 1.2. Обоснование проектных решений

#### 1. Архитектура сквозного оркестратора (`SystemOrchestrator`)
* **Паттерн «Фасад» и конвейер данных:**
  Оркестратор инкапсулирует весь граф обработки кадра:
  $$\text{Frame} \xrightarrow{\text{Localizer}} \text{ROI} \xrightarrow{\text{Detectors}} (\text{Hand}, \text{Motion}) \xrightarrow{\text{Classifier}} \text{Observation} \xrightarrow{\text{FSM}} \text{Decision} \xrightarrow{\text{GSM}} \text{Commit} \to (\text{Repo}, \text{Evidence}, \text{Engine})$$
* **Управление жизненным циклом сессии:**
  - `start_session(session_id, white_player, black_player)`: создание каталогов, инициализация `SessionManifest`;
  - `process_frame(frame: Frame) -> PipelineState`: потокобезопасный шаг обработки кадра;
  - `resolve_uncertain(action, move_uci)`: операторское разрешение коллизий;
  - `finalize_session()`: сброс журналов, генерация канонического PGN, фиксация контрольной суммы `journal_checksum` и сохранение манифеста.

#### 2. Интерактивный HUD-композитор (`VisualHUDComposer`)
* **Визуальные зоны экрана оператора (Layout 1920x1080 / 1280x720):**
  1. **Левая область (Видеокадр):** Наложение сетки полей $a1 \dots h8$, контуров клеток, маски руки MediaPipe и цветовой индикации последнего совершённого хода (зелёный — source, бирюзовый — destination);
  2. **Индикатор состояния FSM (Статус-бар):** Крупный цветовой бейдж (`STABLE` = зелёный, `MOVING` = жёлтый, `SETTLING` = синий, `VERIFYING` = фиолетовый, `UNCERTAIN` = красный, `BOARD_LOST` = серый);
  3. **Правая верхняя область (Виртуальная 2D-доска):** Проекция текущей авторитетной расстановки `chess.Board` с отрисовкой фигур в канонической кодировке Юникод/символов;
  4. **Правая средняя область (Список ходов):** Прокручиваемый список ходов партии в стандартной нотации FIDE SAN с пометками качества от Stockfish (Best / Inaccuracy / Blunder);
  5. **Правая нижняя область (Панель аналитики):** Оценка Stockfish в сантипешках ($\pm \text{cp}$), графический бар преимущества и рекомендуемый лучший ход (`best_move`);
  6. **Всплывающий оверлей при `UNCERTAIN`:** Отображение спорных полей, альтернативных кандидатов с их скорами $S(m)$ и подсказки горячих клавиш (`1`..`9` для выбора кандидата, `Space` для паузы, `M` для ручного ввода).

#### 3. CLI интерфейс (`src/ui/cli.py` и `main.py`)
* Полноценная консольная точка входа с поддержкой команд:
  * `record`: запуск онлайн-записи с камеры или видеофайла с отображением GUI/HUD;
  * `batch`: пакетная обработка каталога видеозаписей без GUI с выгрузкой отчетов;
  * `benchmark`: запуск эталонных бенчмарков точности детекции и распознавания ходов;
  * `audit`: криптографическая проверка целостности журнала партии `events.jsonl`;
  * `replay`: детерминированное пошаговое воспроизведение записанной сессии;
  * `export`: конвертация журнала в форматы PGN / CSV / PDF.

#### 4. Финализация пояснительной записки ([sec_development.tex](file:///home/ectrald/Repos/Chess-game-recording-system/explanatory%20note/sections/sec_development.tex))
* Оформление всех подразделов в строгом соответствии с ГОСТ 7.32:
  * Подраздел 3.1: Фактический технологический стек (Python 3.10+, PyTorch, ONNX Runtime, OpenCV, MediaPipe, python-chess, Stockfish, MLflow);
  * Подраздел 3.2: Полная структура пакетов, классов и контрактов;
  * Подраздел 3.3: Пользовательский интерфейс, логика визуализации HUD и операторские сценарии;
  * Подраздел 3.4: Методология и результаты тестирования (модульные, интеграционные, crash-injection, сквозные);
  * Подраздел 3.5: Численная оценка точности (IoU доски $\ge 0{,}90$, точность ходов $100\%$, задержка инференса $\le 20$ мс) и сравнительный анализ:
    $$\text{Только зрение (Pure Vision)} \quad \text{vs} \quad \text{Только логика} \quad \text{vs} \quad \text{Гибридный ИИ (Neuro-Symbolic)}$$
  * Подраздел 3.6: Подробный разбор 4 демонстрационных сценариев (включая сложные спецходы и ручное разрешение `UNCERTAIN`);
  * Подраздел 3.7: Анализ ограничений и вектор развития;
  * Подраздел 3.8: Выводы по главе.

---

## 2. Структура файлов и модулей этапа STAGE/6

```text
Chess-game-recording-system/
├── main.py                                  # Главная консольная точка входа в систему
├── configs/
│   └── default_config.yaml                  # Конфигурация по умолчанию (UI / Orchestrator)
├── docs/stages/
│   └── STAGE_6.md                           # Настоящий документ спецификации
├── explanatory note/
│   └── sections/
│       └── sec_development.tex              # Финализированный Раздел 3 пояснительной записки
├── src/
│   ├── orchestrator.py                      # Сквозной контроллер системы: пайплайн от видео до PGN
│   └── ui/
│       ├── __init__.py                      # Экспорт UI-компонентов и CLI
│       ├── cli.py                           # CLI-парсер аргументов и подкоманд (record, audit, etc.)
│       ├── dashboard.py                     # Интерактивное рабочее окно оператора (OpenCV HighGUI)
│       └── visualizer.py                    # HUD-композитор, наложение сетки, виртуальная доска
└── tests/
    ├── test_cli.py                          # Тесты консольных подкоманд (audit, replay, export)
    ├── test_orchestrator.py                 # Тесты сквозного пайплайна SystemOrchestrator
    ├── test_system_benchmark.py             # Сквозной контрольный бенчмарк партий и расчет метрик
    └── test_visualizer.py                   # Тесты рендеринга HUD, оверлеев и виртуальной доски
```

---

## 3. Спецификация контрактов данных и состояния конвейера

### 3.1. Состояние конвейера в реальном времени (`PipelineState`)

```python
class PipelineState(BaseModel):
    """Снимок полного состояния сквозной системы на текущем шаге обработки."""
    frame_id: int = Field(ge=0, description="Индекс текущего видеокадра")
    timestamp: float = Field(ge=0.0, description="Временная метка видеопотока в секундах")
    fsm_state: FSMState = Field(description="Текущее состояние конечного автомата")
    board_detected: bool = Field(description="Успешно ли локализован контур доски")
    geometry_confidence: float = Field(ge=0.0, le=1.0, description="Уверенность гомографии S_g")
    hand_detected: bool = Field(description="Присутствует ли рука игрока в зоне доски")
    motion_level: float = Field(ge=0.0, description="Уровень оптического движения")
    current_ply: int = Field(ge=0, description="Номер последнего зафиксированного полухода")
    current_fen: str = Field(description="Текущая авторитетная позиция FEN")
    last_move_san: Optional[str] = Field(default=None, description="Нотация последнего хода")
    last_move_uci: Optional[str] = Field(default=None, description="UCI последнего хода")
    last_decision: Optional[DecisionResult] = Field(default=None, description="Результат шлюза решений")
    latest_eval: Optional[AnalysisResult] = Field(default=None, description="Свежая оценка Stockfish")
    is_paused: bool = Field(default=False, description="Признак паузы оператора")
    active_uncertainty: bool = Field(default=False, description="Требуется ли вмешательство оператора")
    processing_fps: float = Field(default=0.0, ge=0.0, description="Фактическая скорость обработки кадров")
```

---

## 4. Спецификация алгоритмов и программных интерфейсов

### 4.1. Сквозной системный оркестратор (`src/orchestrator.py`)

Класс `SystemOrchestrator`:
* Объединяет все подсистемы проекта:
  * Видеозахват (`VideoAdapter`);
  * Локализация и геометрия (`BaseBoardLocalizer`, `HomographyTransformer`, `OrientationCalibrator`);
  * Детекция динамики (`MotionDetector`, `HandDetector`);
  * Нейросетевая классификация клеток (`BasePieceClassifier`);
  * Конечно-автоматный контроль дебаунсинга (`GameFSM`);
  * Логико-символьный решатель (`LegalMoveCandidateGenerator`, `MoveScorer`, `DecisionGate`);
  * Авторитетный менеджер состояния (`GameStateManager`);
  * Долговременное хранение и аудит (`JSONLEventRepository`, `EvidenceWriter`, `RecoveryManager`);
  * Асинхронный шахматный анализ (`AsyncEngineService`).
* **Основной метод `process_frame(frame: Frame) -> PipelineState`:**
  1. Вычисляет оптическое движение (`MotionDetector.compute_motion`);
  2. Проверяет наличие руки в зоне доски (`HandDetector.detect`);
  3. Если геометрия доски ещё не откалибрована, выполняет поиск доски (`BoardLocalizer.localize_board`);
  4. При стабильной доске извлекает нормализованный кроп $640 \times 640$ (`HomographyTransformer.warp_board`);
  5. В фазе проверки классифицирует 64 клетки пакетным инференсом через ONNX/Mock (`PieceClassifier.predict_cells`);
  6. Формирует структуру `BoardObservation` и передает в `GameFSM.update(observation)`;
  7. При подтверждении хода (`DecisionStatus.ACCEPTED`) выполняет атомарный `commit`, сбрасывает запись в `EventRepository`, сохраняет доказательства через `EvidenceWriter` и ставит ход в очередь `AsyncEngineService`;
  8. Возвращает актуальный снимок состояния конвейера `PipelineState`.

```python
class SystemOrchestrator:
    def __init__(self, config: Optional[AppConfig] = None, session_id: Optional[str] = None) -> None: ...
    def start_session(self, white_player: str = "White", black_player: str = "Black") -> SessionManifest: ...
    def process_frame(self, frame: Frame) -> PipelineState: ...
    def resolve_uncertainty(self, move_uci: Optional[str] = None, auto_accept_top: bool = False) -> bool: ...
    def rollback_move(self) -> Optional[MoveEvent]: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> SessionManifest: ...
    def export_pgn(self) -> str: ...
```

---

### 4.2. Визуальный HUD-композитор (`src/ui/visualizer.py`)

Класс `VisualHUDComposer`:
* Принимает сырой кадр видеопотока `frame.image` и текущий `PipelineState`;
* Осуществляет графическую отрисовку информационной панели (HUD):
  * **Сетка доски:** Проекция четырёхугольника доски и линий 64 клеток с буквенно-цифровой разметкой полей $a1 \dots h8$;
  * **Подсветка хода:** Отрисовка стрелки и контуров полей отправления и прибытия последнего хода;
  * **Маска перекрытия:** Полупрозрачная подсветка клеток, перекрытых рукой игрока;
  * **Статусный бейдж:** Вывод текущего режима FSM (`STABLE`, `VERIFYING`, `UNCERTAIN` и др.) с цветовым кодированием;
  * **Виртуальная доска (2D Board Render):** Отрисовка миниатюры текущей позиции $8 \times 8$ с фигурами в верхнем правом углу кадра;
  * **Журнал ходов (Move List):** Вывод последних 5 ходов в нотации SAN с оценкой Stockfish;
  * **Шкала оценки (Eval Gauge):** Столбчатая диаграмма перевеса сторон (белые / чёрные).

```python
class VisualHUDComposer:
    def __init__(self, width: int = 1280, height: int = 720) -> None: ...
    def compose(self, frame_bgr: np.ndarray, state: PipelineState) -> np.ndarray: ...
    def render_virtual_board(self, board: chess.Board, size: int = 240) -> np.ndarray: ...
```

---

### 4.3. Интерактивный Dashboard и Desktop GUI (`src/ui/dashboard.py`)

Класс `InteractiveDashboard`:
* Управляет окном просмотра через OpenCV HighGUI (`cv2.imshow`, `cv2.waitKey`);
* Поддерживает горячие клавиши управления оператором:
  * `Space` — Пауза / Возобновление записи;
  * `R` — Принудительная рекалибровка ориентации доски;
  * `Z` — Откат последнего хода (`rollback`);
  * `U` — Разрешение неопределенности: принятие наиболее вероятного кандидата;
  * `1`..`5` — Принудительный выбор конкретного кандидата из списка спорных ходов;
  * `P` — Экспорт текущей партии в PGN на диск;
  * `Q` / `Esc` — Завершение записи и выход.
* Автоматически определяет отсутствие графического сервера (`DISPLAY` unset) и переключается в headless-режим без сбоев.

```python
class InteractiveDashboard:
    def __init__(self, orchestrator: SystemOrchestrator, headless: bool = False) -> None: ...
    def run_loop(self, video_adapter: VideoAdapter) -> SessionManifest: ...
```

---

### 4.4. Консольный интерфейс CLI (`src/ui/cli.py` и `main.py`)

Точка входа `main.py` предоставляет единый CLI-интерфейс:

```bash
# Запуск онлайн-записи с камеры с интерактивным HUD
python main.py record --source 0 --display

# Обработка видеофайла в пакетном режиме с сохранением PGN
python main.py record --source data/benchmark/game1.mp4 --headless --out-dir data/sessions/game1

# Проверка криптографической целостности журнала партии
python main.py audit data/sessions/game1/events.jsonl

# Пошаговое воспроизведение записанной сессии
python main.py replay data/sessions/game1/events.jsonl

# Запуск полного сквозного бенчмарка точности
python main.py benchmark --pgn-ref data/ground_truth/test_short_game.pgn
```

---

### 4.5. Финализация Раздела 3 пояснительной записки (`sec_development.tex`)

Синхронизация фактического кода системы с пояснительной запиской по ГОСТ:
1. **Подраздел 3.1: Выбор средств и технологий:**
   - Обоснование Python 3, PyTorch, ONNX Runtime, OpenCV, MediaPipe Hand Landmarker, python-chess, Stockfish 16 UCI, MLflow, pytest.
2. **Подраздел 3.2: Структура программной реализации:**
   - Таблица реализованных модулей (`src/core`, `src/vision`, `src/ml`, `src/reasoning`, `src/persistence`, `src/analysis`, `src/ui`).
3. **Подраздел 3.3: Реализация пользовательского интерфейса:**
   - Скриншоты и описания HUD-композитора, виртуальной доски, статус-бара FSM, шкалы Stockfish и диалога `UNCERTAIN`.
4. **Подраздел 3.4: Тестирование интеллектуальной системы:**
   - Сводная таблица проверок: компонентные тесты (локализатор, детектор рук, классификатор), интеграционные (FSM + DecisionGate + GSM), crash-injection тесты (обрыв питания) и сквозные партии.
5. **Подраздел 3.5: Оценка качества, эффективности и объяснимости:**
   - Таблица экспериментального сравнения трёх подходов:
     1. Pure Vision (только поклеточный argmax нейросети);
     2. Pure Symbolic (шахматные правила без взвешенного правдоподобия);
     3. Разработанная гибридная интеллектуальная система (Neuro-Symbolic).
   - Подтверждение ключевых метрик: точность распознавания ходов $100\%$, нулевая вероятность нелегальных ходов, время инференса $\le 20$ мс.
6. **Подраздел 3.6: Демонстрация работы гибридной интеллектуальной системы:**
   - Пошаговое иллюстрированное описание сквозных сценариев (Детский мат, партия со всеми спецходами, перекрытие доски рукой, разрешение неопределенности оператором).
7. **Подраздел 3.7: Анализ ограничений и направления развития:**
   - Допустимые диапазоны углов наклона камеры ($50^\circ\dots 90^\circ$), предельные значения освещенности, перспективы интеграции с мобильными устройствами и AR-очками.
8. **Подраздел 3.8: Выводы по главе 3.**

---

## 5. Обновление конфигурации (`configs/default_config.yaml`)

```yaml
ui:
  headless: false                     # Режим без графического окна (для серверов и CI)
  window_title: "Chess Recording System HUD"
  display_width: 1280                 # Ширина окна визуализатора
  display_height: 720                 # Высота окна визуализатора
  show_virtual_board: true            # Отображать виртуальную 2D-доску
  show_move_list: true                # Отображать список совершенных ходов
  show_eval_bar: true                 # Отображать шкалу оценки Stockfish
  show_hand_landmarks: true           # Отображать скелет руки игрока
  font_scale: 0.6                     # Масштаб шрифта экранных надписей
```

---

## 6. Комплексный план тестирования для STAGE/6

| Тестовый модуль | Проверяемый функционал | Критерий успешности |
| :--- | :--- | :--- |
| `tests/test_visualizer.py` | 1. Рендеринг сетки полей доски $a1 \dots h8$ на кадре.<br>2. Отрисовка виртуальной 2D-доски из `chess.Board`.<br>3. Отрисовка статусных бейджей FSM и стрелки хода.<br>4. Отображение оверлея предупреждения `UNCERTAIN`. | Все графические элементы формируются корректно, изображение имеет валидные размеры BGR [H, W, 3]. |
| `tests/test_cli.py` | 1. Парсинг аргументов CLI (подкоманды `record`, `audit`, `replay`, `benchmark`).<br>2. Подкоманда `audit` на валидном и поврежденном файле `events.jsonl`.<br>3. Подкоманда `replay` с верификацией вывода PGN. | Консольные команды отрабатывают с ожидаемыми кодами возврата (0 при успехе, non-zero при ошибках). |
| `tests/test_orchestrator.py` | 1. Инициализация и запуск сквозной сессии `SystemOrchestrator`.<br>2. Прогон синтетической последовательности кадров через все слои.<br>3. Проверка автоматического перехода FSM `CALIBRATING` $\to$ `STABLE` $\to$ `MOVING` $\to$ `VERIFYING` $\to$ `STABLE`.<br>4. Проверка записи в `EventRepository` и генерации PGN при штатной остановке сессии. | Полный цикл видеозаписи выполняется без исключений; PGN формируется и соответствует сделанным ходам. |
| `tests/test_system_benchmark.py` | 1. Сквозной контрольный тест партии `test_short_game.pgn` через оркестратор.<br>2. Сквозной контрольный тест партии `test_special_moves.pgn` со всеми спецходами.<br>3. Измерение производительности: средний FPS конвейера, задержка принятия решения. | 100% точность распознавания ходов; финальные FEN совпадают с эталонами; задержка обработки кадра $\le 35$ мс. |

---

## 7. Критерии приёмки (Definition of Done) этапа STAGE/6

Этап `STAGE/6` считается завершённым при выполнении следующих требований:

1. [x] Реализован контракт состояния сквозного конвейера `PipelineState` в `src/core/datatypes.py`.
2. [x] Добавлена секция `ui` в `src/core/config.py` и `configs/default_config.yaml`.
3. [x] Реализован графический HUD-композитор `VisualHUDComposer` (`src/ui/visualizer.py`) с наложением сетки полей, виртуальной 2D-доской, FSM-статусами и шкалой Stockfish.
4. [x] Реализовано рабочее место оператора `InteractiveDashboard` (`src/ui/dashboard.py`) с поддержкой горячих клавиш и автоматическим headless-переключением.
5. [x] Реализован сквозной системный оркестратор `SystemOrchestrator` (`src/orchestrator.py`), связывающий все слои проекта в единый конвейер реального времени.
6. [x] Реализована главная точка входа `main.py` и CLI-интерфейс (`src/ui/cli.py`) с поддержкой подкоманд `record`, `audit`, `replay`, `benchmark`.
7. [x] Полностью заполнен Раздел 3 пояснительной записки ([sec_development.tex](file:///home/ectrald/Repos/Chess-game-recording-system/explanatory%20note/sections/sec_development.tex)) в соответствии с ГОСТ 7.32 (все подразделы 3.1--3.8).
8. [x] Разработан полный набор тестов (`tests/test_visualizer.py`, `tests/test_cli.py`, `tests/test_orchestrator.py`, `tests/test_system_benchmark.py`).
9. [x] Сквозные бенчмарки на эталонных партиях подтверждают 100% точность фиксации ходов и соблюдение бюджета времени ($\le 35$ мс на кадр на CPU).
10. [x] Полный комплекс тестов проекта `pytest tests/` (Этапы 0--6) выполняется на 100% без единой ошибки.

