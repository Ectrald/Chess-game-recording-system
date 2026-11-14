# 03. Контракты данных и межмодульные API

## 1. Общие соглашения

### 1.1. Каноническая система координат

- Шахматная координата клетки всегда записывается в алгебраической нотации: `a1`…`h8`.
- Внутренний список клеток формируется в фиксированном порядке: `a1,b1,…,h1,a2,…,h8` либо другой порядок, но он должен быть один и зафиксирован константой. Не использовать неявный порядок словаря.
- Ориентация доски отделена от расположения вершин. `TL/TR/BR/BL` описывает геометрию изображения; `a1`…`h8` описывает шахматную семантику. Это разные сущности.
- UCI используется как каноническая машинная запись хода (`e2e4`, `e7e8q`), SAN — как отображаемая запись (`e4`, `e8=Q+`, `O-O`).
- Все временные метки хранятся в секундах от начала конкретного сеанса и не должны смешиваться с wall-clock timestamp. Для устойчивого времени желательно хранить также UTC creation time.

### 1.2. Цветовые соглашения изображения

Видеокадры Stage 0 фактически поступают из OpenCV как BGR. Это необходимо зафиксировать в `Frame.color_space`, а не оставлять в комментарии «BGR/RGB». Внутри OpenCV-контура использовать BGR. Преобразование `BGR → RGB` выполнять непосредственно перед моделью, если её preprocessing ожидает RGB. После обработки не переинтерпретировать байты без преобразования.

### 1.3. Канонический алфавит классов

```python
PIECE_CLASSES = (
    "empty",
    "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK",
)
```

В будущем желательно оформить `PieceClass` как `Literal`/Enum. Вероятностное распределение должно содержать каждый класс ровно один раз. Допускается `UNKNOWN` как статус наблюдения, но `UNKNOWN` не является 14-м обучаемым классом: он означает отсутствие надёжного результата (например, закрытая клетка или ошибка preprocessing).

## 2. Контракт `Frame`

Назначение: перенос одного исходного кадра с метаданными между Video Adapter и Vision.

Рекомендуемая модель:

```python
class Frame(BaseModel):
    frame_id: int                 # >= 0, уникален и монотонен в сессии
    timestamp_sec: float          # монотонное время источника
    captured_at_utc: datetime | None
    image: np.ndarray             # H x W x C, uint8
    width: int                    # == image.shape[1]
    height: int                   # == image.shape[0]
    color_space: Literal["BGR", "RGB", "GRAY"]
    source_id: str
    session_id: UUID
    is_keyframe: bool = False
```

Инварианты:

- `width/height` согласованы с `image.shape`;
- кадр не пустой, shape допустима;
- `timestamp_sec` монотонен; если источник предоставляет PTS, использовать его;
- `frame_id` назначается адаптером, а пропуск кадра не приводит к перенумерации уже выданных кадров;
- не сериализовать ndarray через обычный JSON. Для evidence хранить изображение отдельным файлом, а в event — ссылку и checksum.

Возврат источника должен быть различим:

```python
FrameReadResult = FrameAvailable(frame) | EndOfStream | TemporaryReadFailure(error) | SourceLost(error)
```

Это рекомендуемое расширение текущего `Optional[Frame]`, где `None` может означать несколько разных событий.

## 3. Контракт `BoardGeometry`

Назначение: геометрия доски в исходном кадре и матрица преобразования в нормализованную плоскость.

```python
class BoardOrientation(str, Enum):
    UNKNOWN = "unknown"
    WHITE_BOTTOM = "white_bottom"
    BLACK_BOTTOM = "black_bottom"

class BoardGeometry(BaseModel):
    frame_id: int
    corners_image: list[Point2D]       # 4 точки, порядок TL/TR/BR/BL в изображении
    homography_image_to_board: Matrix3x3
    homography_board_to_image: Matrix3x3
    target_size: int                   # например, 640
    orientation: BoardOrientation
    detection_method: str              # hough_ransac/manual/yolo_refined/aruco
    reprojection_error_px: float
    polygon_area_px2: float
    geometry_quality: float            # [0,1], отдельная оценка геометрии
    orientation_quality: float         # [0,1]
    is_valid: bool
```

### 3.1. Проверки геометрии

1. Ровно четыре конечные точки.
2. Порядок вершин согласован; ребра не пересекаются.
3. Полигон выпуклый и имеет положительную площадь.
4. Площадь и aspect ratio находятся в допустимом диапазоне конфигурации.
5. Матрица гомографии конечна, имеет полный ранг и не имеет аномально плохой обусловленности.
6. Reprojection error измерен на найденных точках сетки/углах.
7. Обратное преобразование должно восстанавливать точки в пределах установленной ошибки.
8. При сильном искажении или недостатке инлаеров возвращать `is_valid=False`, а не плохую матрицу.

### 3.2. Важное ограничение гомографии

Гомография корректно выпрямляет плоскую поверхность доски. Верхняя часть высокой фигуры находится выше плоскости, поэтому её проекция не совпадает с проекцией основания на одной и той же плоскости. Это 3D-параллакс. Его нельзя «исправить» одной матрицей homography. Требуется корректный crop, позиционная модель, реальный training data с разными углами или оценка основания фигуры.

## 4. Контракт `CellObservation`

```python
class CellObservation(BaseModel):
    square: SquareName
    probabilities: dict[PieceClass, float]    # все 13 ключей, сумма ~= 1
    predicted_piece: PieceClass | None
    confidence: float                         # max probability или calibrated confidence
    entropy: float                            # мера неопределённости distribution
    is_occupied: bool | None
    occupancy_probability: float
    occlusion_ratio: float                    # [0,1]
    occlusion_sources: list[str]              # hand / unknown / board_edge / etc.
    is_reliable: bool
    reliability_reasons: list[str]
    source_frame_ids: list[int]
```

Инварианты:

- `set(probabilities.keys()) == set(PIECE_CLASSES)`;
- все вероятности конечны и в [0,1];
- сумма вероятностей приблизительно равна 1 с epsilon, например `1e-3`;
- `predicted_piece` равен argmax только если observation reliable; иначе `None`;
- `is_occupied` вычисляется из occupancy branch/вероятности и не противоречит `empty` distribution;
- occlusion не подменяется вероятностью `empty`;
- если клетка закрыта или preprocessing испорчен, она получает `is_reliable=False` и причину, а не низкую уверенность в empty.

## 5. Контракт `BoardObservation`

```python
class BoardObservation(BaseModel):
    session_id: UUID
    frame_id: int
    timestamp_sec: float
    geometry: BoardGeometry
    cells: dict[SquareName, CellObservation]  # ровно 64
    motion_score: float
    motion_detected: bool
    hand_detector_status: DetectorStatus
    hand_present: bool | None
    hand_regions: list[Polygon]
    observation_quality: float
    is_stable_candidate: bool
```

Инварианты:

- ровно 64 уникальных клетки;
- для каждой пары `key, cell` верно `key == cell.square`;
- geometry валидна для данного frame;
- нельзя трактовать `hand_present=None` как `False`;
- `is_stable_candidate` — сигнал для FSM, но не означает confirmed move.

## 6. Контракт временного окна

Для анализа движения лучше передавать не одно observation, а `ObservationWindow`:

```python
class ObservationWindow(BaseModel):
    session_id: UUID
    start_timestamp_sec: float
    end_timestamp_sec: float
    observations: list[BoardObservation]
    stable_duration_sec: float
    any_motion: bool
    any_hand_occlusion: bool
    geometry_consistency: float
```

Окно содержит последовательность observation за период до/после события. Для экономии памяти можно хранить все timestamps, probability vectors только для изменившихся клеток, а полные кадры — только evidence/keyframes.

## 7. Контракт `MoveHypothesis`

```python
class MoveHypothesis(BaseModel):
    candidate_id: UUID
    move_uci: str
    move_san: str | None
    fen_before: str
    expected_fen_after: str
    affected_squares: list[SquareName]
    visual_log_likelihood: float
    temporal_score: float
    geometry_score: float
    occlusion_penalty: float
    total_score: float
    rejected_reason: str | None
    evidence_frame_ids: list[int]
```

`affected_squares` вычисляется из семантики хода, а не из эвристики `abs(delta cells)`. Для castling задействованы четыре клетки, en-passant — три, promotion меняет тип фигуры на destination. Однако число изменившихся клеток используется только как быстрая сигнатура, а окончательная проверка выполняется сравнением ожидаемого `Board` после `push(move)` с вероятностным наблюдением.

## 8. Контракт `MoveEvent`

```python
class MoveEvent(BaseModel):
    event_id: UUID
    session_id: UUID
    ply: int
    move_uci: str
    move_san: str
    fen_before: str
    fen_after: str
    confidence_score: float
    confidence_margin: float
    decision_method: Literal["automatic", "operator_confirmed", "operator_corrected"]
    confirmed_at_utc: datetime
    timestamp_sec: float
    evidence: list[EvidenceRef]
    score_components: dict[str, float]
    analysis_status: Literal["not_requested", "queued", "running", "completed", "failed", "skipped"]
```

Перед созданием события:

1. `fen_before` парсится как полный валидный FEN;
2. `move_uci` парсится как `chess.Move`;
3. ход есть в `board.legal_moves`;
4. SAN вычисляется на исходной позиции;
5. move выполняется на копии позиции;
6. `fen_after` вычисляется из результата, а не берётся из ML;
7. sequence `ply` согласован с журналом;
8. `event_id` уникален и используется как idempotency key.

## 9. Контракт `AnalysisResult`

```python
class AnalysisResult(BaseModel):
    event_id: UUID
    fen_analyzed: str
    engine_name: str
    engine_version: str | None
    depth: int | None
    time_sec: float
    score_before: Score
    score_after: Score
    best_move_uci: str | None
    principal_variation: list[str]
    centipawn_loss: int | None
    classification: Literal["excellent", "good", "inaccuracy", "mistake", "blunder", "unknown"]
    status: Literal["completed", "timeout", "engine_unavailable", "failed"]
```

Значение оценки должно быть нормализовано относительно стороны, которая ходила. Не сравнивать сырой centipawn score без учёта perspective. Классификация `blunder` — эвристика конфигурации, не абсолютный факт.

## 10. Контракт журнала событий и evidence

Хранение первой версии:

```text
sessions/<session_id>/
  session.json
  events.jsonl
  game.pgn
  evidence/
    <event_id>/before.jpg
    <event_id>/after.jpg
    <event_id>/overlay.jpg
    <event_id>/observation.json
    <event_id>/decision.json
  analysis/
    <event_id>.json
```

`events.jsonl` — append-only. Каждая строка является одним событием с `schema_version`. Изображения хранятся файлами; JSON содержит относительные пути, размер и SHA-256. Файл пишется сначала во временный путь и атомарно переименовывается. PGN можно пересоздать из подтверждённых events.

## 11. Версионирование схем

- Каждая сериализуемая структура имеет `schema_version`.
- Изменение обязательного поля — новая версия схемы и миграция.
- Не хранить Python pickle для долгоживущих моделей событий.
- Не сериализовать `np.ndarray` как огромный JSON-массив.
- Логи должны оставаться читаемыми независимо от версии ML модели.

## 12. Конфигурационные контракты

Текущий `AppConfig` Stage0 уже разделён на `vision`, `temporal`, `decision`, `analysis`, `camera`. При реализации новых компонентов дополнить его строго типизированными разделами, не смешивая параметры разных контуров:

```yaml
vision:
  target_board_size: 640
  detector_method: "hough_ransac"
  board_detector_confidence: 0.50
  max_reprojection_error_px: 4.0
  parallax_height_alpha: 1.25
  crop_height_scale: 2.0
  crop_width_scale: 1.5

hand:
  enabled: true
  inference_fps: 10
  min_hand_confidence: 0.50
  polygon_dilation_px: 8
  unavailable_policy: "conservative"

classifier:
  model_path: "data/weights/piece_classifier.pt"
  manifest_path: "data/weights/model_manifest.json"
  batch_size: 64
  min_cell_confidence: 0.55
  min_distribution_sum_tolerance: 0.001

temporal:
  settling_time_sec: 0.8
  verifying_stable_frames: 10
  motion_diff_threshold: 0.05
  max_uncertain_retries: 3

decision:
  weight_visual: 0.50
  weight_temporal: 0.25
  weight_geometry: 0.15
  weight_occlusion: 0.10
  min_absolute_score: 0.60
  min_confidence_margin: 0.15

persistence:
  root_dir: "data/sessions"
  fsync_on_commit: true
  save_evidence_for_every_move: true

analysis:
  enabled: true
  stockfish_path: null
  max_queue_size: 64
  timeout_sec: 10
  depth: 15
```

Значения — примеры стартовой конфигурации. Они не должны копироваться в production как оптимальные без benchmark. Не допускать скрытого дублирования: `min_cell_confidence` управляет cell reliability; `min_absolute_score` — решением по move; `min_confidence_margin` — различием top-1/top-2. Эти три значения имеют разный смысл.

Для конфигурации обязательны:

- schema validation;
- отказ при неизвестных ключах (или warning, если нужен backward compatibility);
- проверка весов и thresholds;
- сохранение resolved config в session manifest;
- версия конфигурации.
