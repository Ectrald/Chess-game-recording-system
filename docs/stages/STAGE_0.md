# Проектирование и план реализации: ЭТАП 0 (STAGE/0)

**Тема этапа:** Инфраструктура, строгие контракты данных, подсистема конфигурации и тестовый бенчмарк-стенд.  
**Ветка реализации:** `STAGE/0`  

---

## 1. Целевое состояние и обоснование архитектурных решений

### 1.1. Назначение и контекст этапа
В соответствии с методологией **Tracer Bullet** ([ROADMAP.md](../../ROADMAP.md), разд. 3) и принципом модульности гибридных интеллектуальных систем ([ПЗ, разд. 1.5](file:///home/ectrald/Repos/Chess-game-recording-system/explanatory%20note/sections/sec_analysis.tex#L90-L100)) реализация алгоритмов компьютерного зрения и шахматного вывода не может начинаться до фиксации:
1. Единой системы строгих интерфейсных типов и контрактов данных между контурами (Vision $\to$ Reasoning $\to$ Analysis);
2. Декларативной подсистемы валидации параметров (пороги нейросетей $\tau_o, \tau_c, \tau_t$, веса целевой функции $w_v, w_t, w_g, w_o$, окно дебаунсинга $\Delta t_{\text{settle}}$, зазор уверенности $\Delta$);
3. Независимого адаптера видеопотока (`FrameAdapter`), изолирующего физический источник (RTSP, файл, веб-камера, генератор синтетических кадров) от конвейера обработки;
4. Автоматизированного стенда тестирования и оценки качества (Benchmark Runner), сопоставляющего поток событий с эталонным PGN.

### 1.2. Обоснование проектных решений
* **Выбор Pydantic v2 для контрактов данных:**  
  В [ПЗ, подразд. 2.4.8](file:///home/ectrald/Repos/Chess-game-recording-system/explanatory%20note/sections/sec_designing.tex#L333-L357) (Таблица 2.2 «Основные структуры данных») зафиксированы сущности: `Frame`, `BoardGeometry`, `CellObservation`, `BoardObservation`, `MoveHypothesis`, `MoveEvent`. Использование Pydantic гарантирует строгую проверку типов в рантайме, валидацию диапазонов вероятностей ($p \in [0, 1]$), автоматическую сериализацию/десериализацию в JSON для сохранения доказательств и сетевого взаимодействия, а также высокую производительность за счет `pydantic-core`.
* **Выбор YAML + Pydantic Settings для конфигураций:**  
  Все математические пороги и веса, описанные в формулах [ПЗ, подразд. 2.4.2, 2.4.5](file:///home/ectrald/Repos/Chess-game-recording-system/explanatory%20note/sections/sec_designing.tex#L160-L172), должны настраиваться без изменения исходного кода. Валидация через Pydantic исключает запуск системы с некорректными весами ($\sum w_i \neq 1$) или отрицательными таймаутами.
* **Синтетический генератор доски для отладки без видео:**  
  До подключения обученных весов EfficientNet и YOLO (Этапы 1–3) необходим эталонный источник кадров, рендерящий 2D-доску и фигуры программно. Это позволит тестировать логику FSM и символьного решателя уже на Этапе 0 в изолированном окружении.

---

## 2. Структура файлов и модулей этапа STAGE/0

В ходе реализации ветки `STAGE/0` в репозитории формируется следующая архитектурная структура:

```text
Chess-game-recording-system/
├── docs/
│   └── stages/
│       └── STAGE_0.md                  # Настоящий документ проектирования
├── configs/
│   ├── default_config.yaml             # Значения параметров по умолчанию
│   └── camera_config.yaml              # Настройки источника видеоданных
├── data/
│   ├── benchmark_videos/
│   │   └── .gitkeep                    # Каталог для тестовых видеозаписей
│   └── ground_truth/
│       ├── test_short_game.pgn         # Эталонный PGN короткой партии
│       └── test_special_moves.pgn      # Эталон партии с рокировкой и en passant
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── datatypes.py                # Pydantic-контракты данных (Таблица 2.2 ПЗ)
│   │   ├── config.py                   # Модель конфигурации и загрузчик YAML
│   │   ├── exceptions.py               # Доменные исключения (BoardLost, AmbiguousMove)
│   │   └── benchmark.py                # Модуль оценки метрик (Accuracy, PGN diff)
│   └── vision/
│       ├── __init__.py
│       ├── video_adapter.py            # Абстракция видеопотока (OpenCV + Synthetic)
│       └── synthetic_board.py          # Генератор тестовых кадров доски
└── tests/
    ├── __init__.py
    ├── conftest.py                     # Фикстуры pytest (кадры, геометрия, конфиги)
    ├── test_datatypes.py               # Тесты контрактов, валидации полей, FEN/PGN
    ├── test_config.py                  # Тесты загрузки, валидации порогов и весов
    ├── test_video_adapter.py           # Тесты видеопотока, FPS, EOF, синтетики
    └── test_benchmark.py               # Тесты алгоритма сверки ходов и расчета метрик
```

---

## 3. Спецификация контрактов данных (`src/core/datatypes.py`)

Контракты строго соответствуют проектным структурам из [ПЗ, Таблица 2.2](file:///home/ectrald/Repos/Chess-game-recording-system/explanatory%20note/sections/sec_designing.tex#L337-L357) и [ROADMAP.md](../../ROADMAP.md):

### 3.1. Структура `Frame`
```python
class Frame(BaseModel):
    frame_id: int = Field(ge=0, description="Монотонно возрастающий счетчик кадров")
    timestamp: float = Field(ge=0.0, description="Метка времени в секундах от начала сеанса")
    image: np.ndarray = Field(description="RGB/BGR изображение кадра")
    width: int = Field(gt=0, description="Ширина исходного кадра в пикселях")
    height: int = Field(gt=0, description="Высота исходного кадра в пикселях")

    class Config:
        arbitrary_types_allowed = True
```

### 3.2. Структура `BoardGeometry`
```python
class BoardGeometry(BaseModel):
    corners: list[tuple[float, float]] = Field(
        min_length=4, max_length=4,
        description="4 вершины доски [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] в порядке TL, TR, BR, BL"
    )
    homography_matrix: Optional[list[list[float]]] = Field(
        default=None, description="Матрица гомографии 3x3 для трансформации в 640x640"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Мера качества четырехугольника S_g")
    is_valid: bool = Field(description="Флаг соответствия условиям выпуклости и площади")
    orientation_calibrated: bool = Field(default=False, description="Привязан ли угол a1-h8")
```

### 3.3. Структура `CellObservation`
```python
class CellObservation(BaseModel):
    square: str = Field(pattern=r"^[a-h][1-8]$", description="Координата клетки в алгебраической нотации")
    probabilities: dict[str, float] = Field(
        description="Распределение по 13 классам: empty, wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK"
    )
    predicted_piece: Optional[str] = Field(description="Класс с наибольшей вероятностью")
    confidence: float = Field(ge=0.0, le=1.0, description="Максимальная вероятность")
    is_occupied: bool = Field(description="Бинарное состояние занятости поля")
    is_occluded: bool = Field(default=False, description="Флаг перекрытия маской кисти MediaPipe")
```

### 3.4. Структура `BoardObservation`
```python
class BoardObservation(BaseModel):
    timestamp: float = Field(description="Временная отметка момента наблюдения")
    frame_id: int = Field(ge=0, description="Номер опорного кадра")
    cells: dict[str, CellObservation] = Field(
        description="Словарь из ровно 64 полей доски (ключи от 'a1' до 'h8')"
    )
    motion_detected: bool = Field(description="Флаг наличия движения на доске")
    motion_level: float = Field(ge=0.0, description="Численная мера межкадровой разности")
    hand_present: bool = Field(description="Обнаружена ли кисть руки в рабочей зоне доски")
    geometry_confidence: float = Field(ge=0.0, le=1.0, description="Оценка качества геометрии S_g")
    
    @field_validator("cells")
    def validate_cells_count(cls, v):
        assert len(v) == 64, "BoardObservation обязан содержать ровно 64 клетки"
        return v
```

### 3.5. Структуры `MoveHypothesis` и `MoveEvent`
```python
class MoveHypothesis(BaseModel):
    move_uci: str = Field(description="Ход в нотации UCI, например 'e2e4', 'e1g1'")
    move_san: Optional[str] = Field(default=None, description="Ход в нотации SAN, например 'e4', 'O-O'")
    changed_squares: list[str] = Field(description="Список изменившихся клеток delta S")
    score_visual: float = Field(description="Логарифмическое правдоподобие S_v(m)")
    score_temporal: float = Field(description="Стабильность во времени S_t(m)")
    penalty_occlusion: float = Field(description="Штраф за перекрытие рукой P_o(m)")
    total_score: float = Field(description="Комплексный балл S(m)")

class MoveEvent(BaseModel):
    ply: int = Field(ge=1, description="Порядковый номер полухода в партии")
    fen_before: str = Field(description="FEN состояние позиции до хода")
    move_uci: str = Field(description="Подтвержденный ход в UCI")
    move_san: str = Field(description="Подтвержденный ход в SAN")
    fen_after: str = Field(description="FEN состояние позиции после хода")
    confidence: float = Field(ge=0.0, le=1.0, description="Запас уверенности Delta")
    timestamp: float = Field(description="Время фиксации хода")
    evidence_frame_id: int = Field(description="Идентификатор доказательного кадра")
    eval_centipawns: Optional[int] = Field(default=None, description="Оценка Stockfish в сантипешках")
    is_blunder: Optional[bool] = Field(default=None, description="Флаг критической ошибки игрока")
```

---

## 4. Спецификация подсистемы конфигурации (`src/core/config.py`)

Конфигурация разбита на секции в соответствии с [ПЗ, разд. 2.4](file:///home/ectrald/Repos/Chess-game-recording-system/explanatory%20note/sections/sec_designing.tex#L135-L330):

```yaml
# configs/default_config.yaml
vision:
  target_board_size: 640          # Нормализованный размер доски (ПЗ 2.4.1)
  min_camera_angle: 60            # Минимальный допустимый угол съемки (градусы)
  parallax_height_alpha: 1.25     # Коэффициент вертикального расширения клетки (ПЗ 2.4.1)
  yolo_board_confidence: 0.50     # Порог уверенности детектора доски
  classifier_batch_size: 64       # Пакетная обработка всех клеток доски (ПЗ 2.4.2)
  threshold_occupied: 0.60        # tau_o: порог пустоты/занятости клетки
  threshold_color: 0.70           # tau_c: порог уверенности определения цвета
  threshold_piece: 0.65           # tau_t: порог уверенности типа фигуры

temporal:
  settling_time_sec: 0.8          # Delta t_settle: окно дебаунсинга теней и рук (ПЗ 2.4.4)
  motion_diff_threshold: 0.05     # Порог оптической разности для перехода в MOVING
  verifying_stable_frames: 10     # Число стабильных кадров подряд в VERIFYING

decision:
  weight_visual: 0.50             # w_v: вес нейросетевого правдоподобия S(m) (ПЗ 2.4.5)
  weight_temporal: 0.25           # w_t: вес временной стабильности
  weight_geometry: 0.15           # w_g: вес качества гомографии
  weight_occlusion: 0.10          # w_o: штраф за перекрытие кистью
  min_confidence_margin: 0.15     # Delta: минимальный зазор между топ-1 и топ-2 ходами

analysis:
  stockfish_enabled: true
  stockfish_depth: 15
  blunder_threshold_cp: 150       # Потеря >1.5 пешки считается грубой ошибкой
```

---

## 5. Спецификация видеоадаптера и синтетической доски

### 5.1. `VideoAdapter` (`src/vision/video_adapter.py`)
* Поддержка протоколов источников:
  * Локальный видеофайл (`.mp4`, `.avi`);
  * USB веб-камера (`/dev/video0`, индекс integer);
  * Сетевой RTSP-поток (`rtsp://...`);
  * Синтетический поток (`synthetic://game.pgn`).
* Методы интерфейса:
  * `read() -> Optional[Frame]` — потокобезопасное получение следующего кадра;
  * `get_fps() -> float`, `get_frame_count() -> int`, `seek(frame_id: int)`;
  * `release()` — освобождение дескрипторов устройств.

### 5.2. `SyntheticBoardGenerator` (`src/vision/synthetic_board.py`)
* Рендеринг шахматной доски $640 \times 640$ на базе `python-chess` и `OpenCV`.
* Программное воспроизведение партий из PGN со сменой позиций через заданные интервалы кадров.
* Наложение искусственных возмущений для проверки устойчивости:
  * Перспективное искажение (проективное преобразование углов);
  * Наложение искусственной тени/руки (эллиптическая маска с гауссовым размытием);
  * Гауссов шум и перепады яркости.

---

## 6. Тестовая стратегия и план тестирования для STAGE/0

Для приемки Этапа 0 разрабатывается комплекс модульных тестов с покрытием $\ge 90\%$:

| Тестовый модуль | Проверяемые сценарии |
| :--- | :--- |
| `tests/test_datatypes.py` | 1. Валидация 64 полей в `BoardObservation` (ошибка при <64 или >64).<br>2. Проверка корректности диапазонов вероятностей $p \in [0, 1]$.<br>3. Сериализация в JSON и обратная десериализация без потерь.<br>4. Корректность валидации FEN и шахматных ходов в `MoveEvent`. |
| `tests/test_config.py` | 1. Загрузка стандартного YAML-файла конфигурации.<br>2. Проверка валидатора суммы весов целевой функции $\sum w_i = 1.0$.<br>3. Реакция на недопустимые пороги ($\tau < 0$ или $\tau > 1$).<br>4. Переопределение параметров через переменные окружения. |
| `tests/test_video_adapter.py` | 1. Корректное чтение синтетического видеопотока.<br>2. Генерация монотонных меток `frame_id` и `timestamp`.<br>3. Обработка ситуации исчерпания кадров (EOF).<br>4. Проверка корректности возвращаемых размеров `Frame.width`, `Frame.height`. |
| `tests/test_benchmark.py` | 1. Загрузка эталонного PGN с таймкодами ходов.<br>2. Алгоритм вычисления FEN-дистанции между предсказанной и истинной позицией.<br>3. Расчет метрик Move Accuracy, Blunder Accuracy, Mean Latency.<br>4. Экспорт сводного JSON/Markdown отчета тестирования. |

---

## 7. Критерии приемки (Definition of Done) этапа STAGE/0

Этап 0 считается успешно завершенным, а ветка `STAGE/0` готовой к слиянию при выполнении следующих критериев:

1. [x] В репозитории развернута утвержденная иерархия каталогов (`src/core/`, `src/vision/`, `configs/`, `data/`, `tests/`).
2. [x] Реализованы и задокументированы модули `datatypes.py`, `config.py`, `exceptions.py`, `video_adapter.py`, `synthetic_board.py`, `benchmark.py`.
3. [x] Созданы файлы конфигураций по умолчанию `configs/default_config.yaml` и `configs/camera_config.yaml`.
4. [x] Подготовлены минимум 2 эталонных PGN-файла партий в `data/ground_truth/`.
5. [x] Набор тестов `pytest tests/` проходит на $100\%$ без ворнингов и сбоев (40/40 tests passed).
6. [x] Все зависимости зафиксированы в `requirements.txt`.
7. [x] В `README.md` и документации отражен статус завершения Этапа 0.
