# Полный технический аудит проекта (6 независимых агентов)

Дата: 2026-11-28  
Охват: 100% подсистем (`Web UI`, `Services/CLI/Docker`, `Orchestrator/Core`, `Vision`, `ML/Scripts`, `Reasoning/Stockfish/Persistence`).

---

## 1. Web UI & API (`src/ui/web/app.py`, `templates/index.html`, `static/js/app.js`, `static/css/style.css`)

- **[CRITICAL] Бесконечный спам уведомлениями после завершения транскрибации видео** (`app.js:457-459, 833-845`, `app.py:251-259, 335-341`): `transcriber.get_status()` остаётся в `status="completed"`, из-за чего каждые ~0.5с всплывает тост и перезатирает другие сообщения.
- **[CRITICAL] `uncertainModal` закрывается при ошибке разрешения** (`app.js:1081-1084, 1062-1079`, `app.py:517-528`): `resolveUncertainty()` вызывает `hideUncertainModal()`, не проверяя результат `postAction`.
- **[CRITICAL] Рассинхрон наложения углов и координат мыши под `object-fit: contain` и до загрузки кадра** (`style.css:238-256, 684-692`, `app.js:939-1060`): `getCanvasScale()` считает `rect.width/natW` и `rect.height/natH` раздельно вместо единого коэффициента с центрирующими отступами (`offsetX`, `offsetY`), `#cornersCanvas` растянут по `.video-wrapper` (`min-height: 380px`), а до прихода первого кадра `naturalWidth === 0` вызывает фолбэк в CSS-размер.
- **[HIGH] Гонка `cur_adapter.read()` против `old_adapter.release()` при смене видеоисточника** (`app.py:283-294, 609-617`).
- **[HIGH] Тихое игнорирование ошибок HTTP во всех вызовах `postAction`** (`app.js:154-250, 807-821, 1055-1079`).
- **[MEDIUM]** Отсутствие начальной синхронизации через `/api/state`; `onclose` старого WebSocket закрывает переподключённый сокет; синхронные V4L2-пробы и запись аплоадов блокируют event loop; `worker_thread` не стартует при `video_adapter=None`; отсутствие валидации формата UCI и учёта `illegal_attempts` в сигнатуре модалки; повторные клики на «Анализировать» во время выполнения задачи.
- **[LOW]** Чтение состояния оркестратора без `_lock` в `get_state_payload()`; таймер записи замирает на EOF; наследование имён игроков в «Новая сессия»; горячие клавиши прокрутки страницы срабатывают при открытых модалках.

---

## 2. Services, CLI, HUD & Docker (`video_services.py`, `streamer.py`, `ws_manager.py`, `cli.py`, `dashboard.py`, `visualizer.py`, `video_adapter.py`, `Dockerfile`, `docker-compose.yml`)

- **[HIGH] `OpenCVVideoAdapter` и CLI некорректно обрабатывают `/dev/video*` и `https://`** (`video_adapter.py:80-94`, `cli.py:72-79, 223-230`, `app.py:197-206`).
- **[HIGH] `cmd_web` аварийно завершается при отсутствии камеры `0`** (`cli.py:241, 300`, `docker-compose.yml:23-25`).
- **[HIGH] Гонка / несинхронизированный `discover_video_devices` для symlink-устройств `/dev/v4l/by-id/*` и несоответствие активного источника** (`video_services.py:44-87`, `app.py:578-586`).
- **[HIGH] Запись видео с номинальным FPS при низком реальном FPS и без авто-остановки на EOF** (`video_services.py:199`, `app.py:295-306, 666`).
- **[HIGH] `/api/videos/download` разрешает скачивание произвольных файлов из `data/sessions`** (`app.py:701-712`).
- **[HIGH] `VideoTranscriptionManager` ломается на VFR/WebM (`FPS=1000` или `NaN`), обрывается на 1 битом кадре, игнорирует `rotate` и завершается со статусом `completed` с 0 ходами при отказе калибровки** (`video_services.py:364-406`).
- **[HIGH] `VisualHUDComposer` и `MJPEGStreamer` падают (`cv2.error` / `TypeError` / `ValueError`) на пустой `hand_mask`, grayscale/BGRA/float32-кадрах или малых размерах HUD** (`visualizer.py:228-252, 309-355, 443-446, 612-624`, `streamer.py:36-49, 64-87`).
- **[HIGH] `cmd_replay` падает с `IllegalMoveError` при наличии событий `rollback` / `illegal_attempt`** (`cli.py:143-156`).
- **[MEDIUM]** Потеря `/app/data/recordings` при пересоздании контейнера; относительные пути от CWD; `InteractiveDashboard` читает несуществующий `_corners` и клавиша `R` не сбрасывает геометрию; `load_config` падает на пустой секции `analysis: null` при заданном `STOCKFISH_PATH`; `cmd_record` игнорирует `--white`/`--black`, `cmd_web` игнорирует `host`/`port` из конфига.

---

## 3. Orchestrator & Core (`src/orchestrator.py`, `src/core/config.py`, `src/core/datatypes.py`)

- **[HIGH] В zero-shot ветке в качестве `nn_occ_before` передаётся уже перезаписанный текущий `cell_obs`** (`orchestrator.py:662-664, 948-957`).
- **[HIGH] `rollback_move()` и `resolve_uncertainty()` сбрасывают паузу/FSM без проверки `_is_paused`, оставляют грязные буферы takeback/rapid-queue, возможен `AttributeError` при `_last_observation is None`** (`orchestrator.py:1577-1660`).
- **[HIGH] `_last_confirmed_orientation` не сбрасывается при `recalibrate`, `reset_session`, `new_session`, `resume_session`** (`orchestrator.py:202, 527-528, 1407-1460`).
- **[HIGH] `_refine_with_whole_board_hybrid` не обнуляет `_pending_physical_anomaly` на входе**, из-за чего старая аномалия может заблокировать следующий валидный ход (`orchestrator.py:287-297, 1078-1373`).
- **[MEDIUM]** Не очищаются эталонные кадры при `drift_status == "lost"`; takeback в `UNCERTAIN` недостижим; быстрые 2-пли с общей клеткой (взятие с перевзятием на одной клетке) ошибочно затираются; устаревшие коллбэки Stockfish перетирают `_latest_eval`; `Sequence` не импортирован в `orchestrator.py`; `BoardGeometry.polygon_area_px2` падает при ранней ошибке валидации.

---

## 4. Vision Pipeline (`src/vision/*.py`)

- **[CRITICAL] Перепутаны прямая и обратная гомографии в `SyntheticBoardGenerator.get_ground_truth_geometry`** (`synthetic_board.py:145-153`).
- **[HIGH] `compute_occlusion_from_board_mask` игнорирует ориентации, отличные от `WHITE_BOTTOM`, при `rois=None`** (`hand_detector.py:70-93, 389-394, 508-513`).
- **[HIGH] `ColorSilhouetteHandDetector` ложно определяет деревянные шахматные доски как руку (64/64 клеток перекрыто)** (`hand_detector.py:448-452, 577-585`).
- **[HIGH] `create_board_localizer` игнорирует параметры `LocalizerConfig`/`VisionConfig`**, а `CoarseToFineLocalizer` не откатывается на Hough при промахе YOLO (`board_localizer.py:258-260, 491-526`).
- **[HIGH] `BoardGeometryTracker.check_drift` даёт ложный `lost`/`updated` при перекрытии рукой и малых/наклонных досках** (`board_localizer.py:385, 404-489`).
- **[HIGH] `_select_grid_boundary_candidates` отбрасывает валидные доски с охватом `< 35%` по оси** (`corner_refinement.py:174`).
- **[HIGH] Падения на `float32` / grayscale / 4-ch в `MotionDetector` и `BoardDiagnosticVisualizer`** (`motion_detector.py:85, 109`, `diagnostics.py:152-162`).
- **[MEDIUM/LOW]** Относительный путь к `hand_landmarker.task`; пустые массивы в `perspectiveTransform` (`homography.py:256-281`); `NaN` в `occupancy_tracker.py`; обход `mass_distribution` при низком контрасте углов в `OrientationResolver`; `extract_occupancy_crops` без padding-логики.

---

## 5. ML Pipeline & Scripts (`src/ml/*.py`, `scripts/*.py`)

- **[CRITICAL] `AutoLabeller.run()` вызывает `localize()` вместо `locate()`, падает на `None` и вечно висит в `CALIBRATING`** (`auto_labeller.py:198-249`).
- **[HIGH] `DINOv2PatchChangeDetector._image_sig` вызывает коллизии кэша и не сбрасывается при смене сессии/паузе/рекалибровке** (`piece_classifier.py:598-636`).
- **[HIGH] `probabilities_to_cell_observation` может вернуть `predicted_piece="empty"` при `is_occupied=True`** (`piece_classifier.py:99-111`), а повторный softmax портит откалиброванные вероятности.
- **[HIGH] `classify_squares(..., squares=[])` и `predict_squares(..., squares=[])` классифицируют все 64 поля вместо `{}`** (`piece_classifier.py:265, 402`).
- **[HIGH] `export_model_to_onnx` сломан для `MultiHeadPieceClassifier`** (`export.py:32-77`, `models.py:156-176`).
- **[HIGH] `ensure_model_weights` и `download_model.py` не скачивают `chessq_vit_int8.onnx`, `dinov2_small_int8.onnx` и `hand_landmarker.task`; `download_file` пишет напрямую в целевой файл** (`cli.py:42-63`, `download_model.py:24-105`).
- **[HIGH] `PieceClassifierTrainer` и `run_training.py` не содержат цикла обучения** (`trainer.py:142-208`, `run_training.py:59-80`).
- **[MEDIUM/LOW]** Отсутствие блокировки и масштабирования `board_size` в `DINOv2PatchChangeDetector`; падение `detect_orientation` на `NaN`; утечка `group_id` между `AutoLabeller` и `dataset.py`; missing `Any`/`Dict` в `export.py`.

---

## 6. Reasoning, Stockfish & Persistence (`src/reasoning/*.py`, `src/analysis/*.py`, `src/persistence/*.py`)

- **[CRITICAL] Неатомарная перезапись `"wb"` без `fsync` в `RecoveryManager.sanitize_corrupted_tail()`** (`recovery_manager.py:74-75, 87-88`) + не удаляется битая строка перед хвостовыми пустыми строками (`lines 58-81`).
- **[HIGH] `StockfishUCIWorker.evaluate()` падает с `AttributeError` после `close()`**, а зависшие процессы Stockfish не принудительно закрываются (`stockfish_worker.py:146-158, 209-212, 264-271`).
- **[HIGH] Устаревшие оценки `AsyncEngineService` перетирают `_latest_eval` после отката/сброса, а `submit_move()` продолжает добавлять в очередь при мёртвом воркере** (`engine_service.py:85-125, 146-173`).
- **[MEDIUM]** Несуществующий метод `observation.get_occluded_squares()` и несбрасываемый `_uncertain_motion_streak` в `fsm.py`; игнорирование `has_geometry`/`orientation_ready` при `resume()` из `UNCERTAIN`; `manual_commit` дубликат при `frame_id=0`; `mate_in == 0` оценивается как `-10000`; `verify_integrity()` допускает rollback при пустом списке ходов; отсутствие блокировки и проверки `cv2.imwrite` в `evidence_writer.py`.
