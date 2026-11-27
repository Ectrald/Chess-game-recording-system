# 08. План реализации и Definition of Done

## 1. Стратегия

Разработка продолжается после существующего `STAGE/0`. Использовать tracer-bullet/incremental подход: каждый этап должен оставлять проект в запускаемом состоянии и проходить тесты. Сначала обеспечить корректную геометрию и структуру наблюдений, затем piece classifier, потом reasoning и integration. Не обучать classifier на невалидных crops.

## 2. Этапы

| Этап | Содержание | Основной результат | DoD |
|---|---|---|---|
| `STAGE/0` | Инфраструктура, contracts, config, video adapter, synthetic generator, benchmark | Базовая среда разработки | Уже присутствует в ветке `STAGE/0`; требуется audit/fixes по документу 00 |
| `STAGE/1` | Board localization, geometry, homography, orientation, cell crops | `BoardGeometry` + 64 ROI из реального кадра | Углы размеченных тестовых кадров находятся в пределах установленной ошибки; ориентация однозначна или unknown; diagnostic overlay |
| `STAGE/2` | Motion, Hand Landmarker, occlusion mapping, FSM | Стабильные/unstable observation windows | Никакой MoveEvent не создаётся во время движения/критической окклюзии; unit tests на FSM |
| `STAGE/3` | Dataset pipeline, auto-labeling, classifier training | ML model artifact + manifest | Grouped split, unseen set test, macro/per-class metrics, reproducible run |
| `STAGE/4` | Candidate move generation, scorer, acceptance gate, commit | Автоматическое определение одного физического хода | Специальные ходы, ambiguous, illegal transitions, idempotent commit проходят E2E |
| `STAGE/5` | Persistence/evidence hardening, recovery, Stockfish UCI worker | Отказоустойчивый session lifecycle и post-game analysis | Потеря engine не блокирует игру; сбой storage не меняет game state; replay восстанавливает PGN |
| `STAGE/6` | UI/CLI, packaging, final benchmark | Пользовательский прототип | Сценарии старт/пауза/подтверждение/экспорт/анализ; тестовый отчёт |

## 3. STAGE/1 — Board localization и geometry

### Задачи

- Усилить `BoardGeometry` и `Frame` contracts до начала алгоритмов.
- Реализовать diagnostics для визуализации углов и нормализованной доски.
- Разработать двухстадийный контур локализации доски (Coarse-to-Fine):
  - coarse BBox ROI через легковесную модель YOLOv8n-board (отсечение фонового шума стола/комнаты);
  - субпиксельное уточнение углов решётки Canny/Hough/clustering/intersections/RANSAC внутри ROI (наследие `chesscog`).
- Реализовать режим прямого детерминированного поиска углов (для синтетики Stage 0) и ручной fallback углов.
- Добавить ArUco-калибровку для эталонного лабораторного замера геометрической погрешности ($\le 2$ px).
- Рассчитать forward/inverse homography (разрешение $640 \times 640$).
- Добавить трёхфакторный orientation resolver (тест правого нижнего белого поля, градиентная масса рядов 1-2 vs 7-8, явный выбор стороны) с возвратом `UNKNOWN` при сомнениях.
- Реализовать нарезку 64 ROI клеток с position-aware crop policy и учётом параллакса высоты фигур ($\alpha_{\text{height}}$).
- Добавить тесты на rotation/perspective/board edge clipping.

### DoD

- воспроизводимые углы и субпиксельная точность $\le 2$ px;
- валидатор geometry отсекает вырожденные и самопересекающиеся полигоны;
- 64 клетки корректно отображаются на алгебраические координаты `a1`–`h8`;
- неопределённость $0^\circ / 180^\circ$ надёжно разрешается до фиксации ходов;
- сохранён диагностический оверлей (BBox, углы, сетка, подписи полей).

## 4. STAGE/2 — Motion, hand/occlusion и FSM

### Задачи

- Реализовать motion detector на нормализованной доске;
- подключить MediaPipe Hand Landmarker как optional/advisory component;
- перевести hand polygon в board coordinates и пересечение на клетки;
- ввести `DetectorStatus` и `occlusion_ratio`;
- реализовать FSM с unit tests;
- хранить ObservationWindow и evidence around movement;
- настроить settle window на реальных коротких видео.

### DoD

- hand detector failure не равен no-hand;
- shadow alone не порождает move;
- moving piece не вызывает commit до settle;
- return-to-origin не создаёт ход;
- loss of board переходит в отдельное состояние.

## 5. STAGE/3 — Dataset и classifier

### Задачи

- Загрузить и интегрировать предобученные веса `chesscog` (Wölflein, 2021) через скрипт `download_model.py` в качестве немедленного Baseline 0 (Zero-Shot).
- Собрать и структурировать многоисточниковый датасет: синтетика 3D `chesscog` (~5 000 рендеров) + нарезка из видео партий через `AutoLabeller` + реальные фотосессии.
- Реализовать специализированные доменные аугментации: `AnchoredShear` (сдвиг с фиксацией основания клетки), `AnchoredScale` (масштабирование от основания), инъекцию теней и шум сенсора.
- Развернуть контур отслеживания экспериментов в **MLflow**: логирование параметров, кривых обучения, матрицы ошибок 13×13 и чекпоинтов.
- Обучить модели: ResNet-18 baseline и EfficientNet-B0 multi-head (occupancy, piece type, color) с transfer learning от ImageNet.
- Провести оценку калибровки вероятностей, per-class F1 и обобщающей способности на unseen chess sets.
- Экспортировать обученную PyTorch-модель в **ONNX** с валидацией паритета выходов ($\max |\Delta p| < 10^{-4}$) и верификацией latency $\le 20$ мс на CPU для батча из 64 клеток.

### DoD

- модель принимает 64 кропа из реального vision-пайплайна;
- идентичный препроцессинг при обучении и инференсе;
- строгий grouped split: тестовые комплекты фигур не пересекаются с обучающими;
- в MLflow зафиксирован run с артефактами (classification report, confusion matrix, `.onnx`);
- проверено соблюдение аппаратного бюджета инференса: batch-64 через ONNX Runtime выполняется за $\le 20$ мс на CPU;
- модель не допускается к использованию, если manifest несовместим с версией препроцессинга.

## 6. STAGE/4 — Reasoning и move commit

### Задачи

- реализовать `TemporalStateMachine`;
- реализовать `LegalMoveCandidateGenerator` через python-chess;
- реализовать `MoveScorer` по формуле/score components;
- реализовать acceptance threshold и confidence margin;
- `UNCERTAIN` + recovery;
- atomic `GameStateManager.commit`;
- PGN projection и event log;
- E2E tests all special moves.

### DoD

- нет незаконных committed transitions;
- no-candidate/close-candidates не меняют позицию;
- у каждого commit есть evidence и FEN before/after;
- повтор события не дублирует ход;
- позиция после replay журнала совпадает с последним FEN.

## 7. STAGE/5 — Persistence, recovery и engine analysis

### Задачи

- event repository JSONL + evidence file writer;
- atomic writes and checksum;
- startup replay;
- idempotency keys;
- bounded analysis queue;
- Stockfish worker via UCI with timeout and restart;
- result classification with side-to-move normalization;
- UI events for completed/failed analysis.

### DoD

- crash injection tests проходят;
- analysis timeout не тормозит video capture;
- engine failure не делает commit rollback;
- event log восстановим и audit-able.

## 8. STAGE/6 — User interface and release

### Задачи

- выбрать минимальный UI: CLI для test/debug, desktop UI если нужна работа с live camera;
- показать camera preview, board overlay, recognized position, PGN move list;
- показать status/fsm/uncertain reasons;
- разрешить orientation/starting position setup;
- ручная коррекция спорного хода;
- export PGN/evidence/benchmark report;
- проверить install/launch на чистом окружении.

### DoD

- сквозной сценарий записи партии выполняется без разработки вручную;
- пользователь различает accepted/uncertain/error;
- сохранённая партия открывается стандартным шахматным приложением;
- конфигурация и model manifest доступны пользователю.

## 8.1. STAGE/8 — Real-World Hardening (по результатам полной партии)

### Задачи

- Устранить скачки Hough на внешний пластиковый бортик и добавить анти-выбросовый трекер дрифта камеры/стола (`ADR-019`).
- Реализовать `base_core` и расширенный unclipped warp для краевых полей `a/h/1/8` (`ADR-021`).
- Внедрить факторизованный `occupancy + color` скоринг и нелинейный штраф провальной клетки (`ADR-020`), устраняющий ложные `UNCERTAIN` при `0.7 vs 0.6` и поддерживающий нестандартную фигурку на месте пешки.
- Добавить Sparse Changed-Cell Inference, прореживание MediaPipe на статичной доске и заготовку Static INT8 PTQ для ускорения обработки хода в ~10–15× на CPU (`ADR-018`).
- Реализовать Bounded 2-Ply Recovery (быстрые ходы), окно восстановления после падения фигур и физический takeback (`ADR-022`).

### DoD

- Сквозной прогон на видеозаписи реальной партии `data/benchmark_videos/2026-11-25 22-05-47.mp4` и нарезках с краёв доски устойчиво отслеживает наклон камеры (`t≈546..550s`), возврат ошибочного хода (`t≈784..811s`), падения и ход фигуркой-пешкой без ложных `UNCERTAIN`.
- Время верификации хода в `VERIFYING` снижено до `<= 150 мс` при динамическом батче изменённых клеток.

## 9. Приоритеты при ограниченном времени

**Обязательное ядро системы:** строгие data contracts, двухстадийная локализация (YOLO coarse BBox + Hough/RANSAC corners), baseline `chesscog`, MLOps-трекинг в MLflow, высокопроизводительный инференс в ONNX Runtime ($\le 20$ мс), grouped test split по комплектам фигур, обработка неопределённости `UNCERTAIN`, строгая валидация легальности ходов по правилам ФИДЕ, транзакционный event log, benchmark.  
**Можно отложить на фазу пост-MVP:** тяжёлые ViT/DINO бэкбоны, сегментация по маскам SAM, сложный многооконный GUI, распределённый кластер MLflow в облаке, внешняя база данных PostgreSQL/SQLite.  
**Категорически не делать:** end-to-end video $\to$ PGN монолитная нейросеть, обучение классификатора фигур с нуля без pretraining, многоагентная оркестровка поверх vision, детектирование всех посторонних объектов сцены.

## 10. Практическая контрольная точка

До обучения classifier необходимо получить из 20–50 реальных кадров:

- вручную проверенные углы;
- корректно rectified board images;
- 64 ROI с понятной выборкой фигур разных типов;
- известную ориентацию;
- оценку crop coverage для высоких фигур.

Если этот набор показывает систематическую ошибку геометрии или crop, обучение не начинать: ML будет пытаться компенсировать ошибку preprocessing и хуже обобщаться.
