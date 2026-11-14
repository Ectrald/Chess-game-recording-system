# 07. Тестирование, benchmark и критерии качества

## 1. Принципы

Тестирование должно оценивать не только классификатор клеток, а весь путь от кадра до подтверждённого хода. У системы есть несколько уровней качества: геометрия доски, распознавание клетки, восстановление позиции, точность конкретного хода и точность партии целиком.

Главный safety KPI — **false commit rate**. Лучше отправить несколько ходов в `UNCERTAIN`, чем автоматически записать неверный ход и испортить всю оставшуюся последовательность.

## 2. Уровни тестирования

### 2.1. Unit tests

Проверяют независимые функции и контракты:

- Frame shape/color/timestamp;
- BoardGeometry convexity/order/homography validity;
- полный набор 64 клеток и согласованность key/square;
- вероятности 13 классов и их нормализацию;
- корректность UCI/SAN/FEN;
- преобразование координат `pixel ↔ board ↔ square`;
- cell ROI bounds и отсутствие crop за пределами массива;
- motion score на идентичных/изменённых изображениях;
- FSM legal transitions;
- candidate scoring на заданных synthetic probabilities;
- idempotent commit;
- replay JSONL → identical board/PGN.

### 2.2. Component tests

- Hough/RANSAC localizer на наборе изображений с известными corners;
- orientation resolver на board rotations 0/90/180/270;
- hand mask to cell occlusion mapping;
- classifier inference на golden crops;
- Stockfish UCI adapter с реальным бинарником/мок-сервисом;
- persistence recovery после искусственного сбоя между write и state update.

### 2.3. Integration tests

Синтетический источник Stage0 предоставляет known frames/geometry/ground truth. Проверить интеграцию:

- `VideoAdapter → Frame`;
- `Frame → Geometry → 64 cells`;
- `BoardObservation → FSM`;
- `FSM → Candidate Generator → Move Scorer`;
- `MoveEvent → PGN + JSONL`;
- `MoveEvent → AnalysisQueue → AnalysisResult`.

SyntheticBoardGenerator хорош для воспроизводимости переходов и ошибок pipeline, но не для оценки качества на реальных силуэтах фигур.

### 2.4. End-to-end tests

Вход — полный видеофайл партии с эталонной временной разметкой. Выход — PGN и event log. Сопоставить ход-за-ходом с ground truth, включая специальные ходы.

## 3. Benchmark dataset protocol

Для каждого видео хранить:

- `video_id`, `session_id`, `camera_id`, `board_set_id`, lighting condition;
- reference PGN;
- timestamp начала/конца каждого полухода;
- timestamp стабильной позиции после каждого хода;
- camera angle/разрешение/FPS, если известно;
- режим ориентации;
- наличие/тип occlusions;
- expected board geometry labels для части кадров.

PGN сам по себе не содержит timestamps. Если timestamp отсутствует, нельзя достоверно считать move latency или обучать frame→board labels по точному времени.

## 4. Обязательные test scenarios

| ID | Сценарий | Ожидаемое поведение |
|---|---|---|
| T-01 | Статичная начальная позиция | Стабильная геометрия и правильная позиция |
| T-02 | Обычный ход пешкой на 1 клетку | Один корректный MoveEvent |
| T-03 | Длинный ход пешки | Один ход после стабилизации |
| T-04 | Взятие | Capture правильно отражён в FEN/PGN |
| T-05 | Королевская/ферзевская рокировка | Перемещаются король и ладья, один ход |
| T-06 | En passant | Обновляются три клетки, корректный UCI/SAN |
| T-07 | Promotion | Правильный promoted piece; если фигуру не видно, запросить уточнение |
| T-08 | Рука закрывает from/to square | Не фиксировать ход до достаточного evidence |
| T-09 | Тень проходит по доске без перемещения фигуры | Не создавать MoveEvent |
| T-10 | Игрок взял фигуру и вернул на ту же клетку | Не создавать MoveEvent |
| T-11 | Камера слегка сместилась | Перекалибровка или board lost; не создавать ложный ход |
| T-12 | Камера потеряла доску | `BOARD_LOST`, восстановление после появления доски |
| T-13 | Временный RTSP disconnect | `SOURCE_LOST`, retry, сохранить confirmed position |
| T-14 | Низкая classifier confidence | `UNCERTAIN`, не выбирать argmax вслепую |
| T-15 | Нелегальная визуальная позиция | Recovery, предыдущая position остаётся authoritative |
| T-16 | Stockfish недоступен | Регистрация партии продолжается |
| T-17 | Дубликат commit event | Нет второго одинакового хода |
| T-18 | Сбой persistence | Position state не обновляется до успешной фиксации |
| T-19 | Новая модель/preprocessing version | Golden test обнаруживает несовместимость артефакта |
| T-20 | Новая шахматная расстановка/произвольная стартовая позиция | Запросить side-to-move/orientation/castling rights, если они не наблюдаемы |

## 5. Метрики

### 5.1. Vision

- `corner_error_px` / normalized reprojection error;
- `board_localization_success_rate`;
- `orientation_accuracy`;
- `cell_macro_f1` и per-class F1;
- `full_board_exact_accuracy`;
- `occlusion_detection_recall/precision`;
- `unknown_cell_rate`;
- `classifier_latency_ms`.

### 5.2. Move registration

- `move_precision = correct committed / all committed`;
- `move_recall = correct committed / all expected moves`;
- `false_commit_rate`;
- `missed_move_rate`;
- `uncertain_rate`;
- `manual_correction_rate`;
- `p50/p95_move_latency`;
- `game_exact_match` — полное совпадение всей последовательности UCI;
- `first_error_ply`.

`game_exact_match` является строгой метрикой: одна ошибка делает партию несовпадающей. Считать её вместе с move accuracy, а не вместо неё.

### 5.3. Reasoning/recovery

- доля ошибок classifier, исправленных legal move filtering;
- доля неоднозначных состояний, разрешённых дополнительными кадрами;
- recovery success rate;
- число повторных inference calls на ход;
- доля событий, потребовавших оператора.

### 5.4. Engine analysis

- engine task latency;
- queue wait time;
- timeout/failure rate;
- average depth/time;
- reproducibility при одинаковых параметрах (учитывая engine version/threads/Hash).

## 6. Сравнительные эксперименты и абляционные исследования

Для академического обоснования каждого компонента системы (согласно требованиям пояснительной записки) проводятся два независимых цикла сравнительных экспериментов.

### 6.1. Абляция контура принятия решений (Reasoning & FSM)

Сравнение вариантов на одном и том же тестовом наборе видеозаписей:

| Вариант | Конфигурация контура | Исследуемый фактор |
|---|---|---|
| **A (Naive CV)** | Одиночный кадр $\to$ argmax по 64 клеткам $\to$ прямой diff | Базовый визуальный baseline без учёта времени и правил |
| **B (Temporal)** | Вероятностный вектор клеток + временное окно стабильности | Вклад подавления случайных флуктуаций освещения и шума |
| **C (Rules-aware)** | Вариант B + фильтрация легальных ходов (`python-chess`) | Вклад строгих шахматных правил в отсев артефактов |
| **D (Score-driven)** | Вариант C + взвешенный скоринг правдоподобия $S(m)$ и margin | Вклад мягкой вероятностной оценки надёжности хода |
| **E (Full Hybrid)** | Вариант D + детектор руки (MediaPipe) + recovery state | Вклад изоляции физических окклюзий и автокалибровки |

*Сравниваемые метрики:* `Move Precision`, `Move Recall`, `False Commit Rate`, `Uncertain Rate`, `Game Exact Match`, `P95 Latency`.

### 6.2. Сравнительный бенчмарк нейросетевых классификаторов клеток

Сравнение точности и скорости различных стратегий обучения на контрольной выборке реальных досок (unseen sets):

| Модель / Стратегия | Описание архитектуры | Обучающие данные | Целевой Macro F1 | Инференс CPU (Batch=64) |
|---|---|---|---|---|
| **Baseline 0 (`chesscog`)** | ResNet/Inception веса Wölflein (2021) | 5 000 3D-рендеров (Zero-shot) | $\sim 0.75\text{--}0.85$ (unseen physical) | $\sim 90$ мс (PyTorch) |
| **Baseline 1 (Standard TL)** | EfficientNet-B0 (ImageNet pretrain) | Real crops + станд. ColorJitter | $\sim 0.88\text{--}0.92$ | $\sim 85$ мс (PyTorch) |
| **Proposed (Domain Aug)** | EfficientNet-B0 multi-head | Real + 3D + `AnchoredShear/Scale` | $\mathbf{\ge 0.95}$ | $\mathbf{\le 1 8}$ **мс (ONNX Runtime)** |
| **Few-Shot Adapted** | Proposed + калибровка по 2 фото | Начальная расстановка комплекта | $\ge 0.98$ (на данном комплекте) | $\le 18$ мс (ONNX Runtime) |

### 6.3. Аппаратный бенчмарк среды выполнения (PyTorch CPU vs ONNX Runtime)

Оценка соответствия вычислительного бюджета требованиям реального времени ($< 100$ мс по ПЗ разд. 2.4.2):
- **PyTorch eager CPU (batch=64, float32):** замер базового времени прямого прохода;
- **ONNX Runtime CPU (batch=64, AVX2/AVX-512):** замер времени оптимизированного графа;
- **Критерий приёмки:** ускорение не менее чем в $3.5\text{--}5$ раз, latency батча из 64 клеток $\le 20$ мс на стандартном 4-ядерном CPU.

## 7. Проверка качества на unseen chess set

Отдельный test split должен содержать комплекты, которых не было в обучении. Необходимо указать:

- какие конкретно sets входят в train/val/test;
- какая камера использовалась;
- какие условия освещения;
- сколько изображений/кропов/позиций;
- есть ли пересечение сессий между split;
- как делались labels.

Если один физический комплект появляется в train и test под разным освещением, это не является полноценной проверкой unseen-set generalization.

## 8. Benchmark исправления Stage0

Текущий benchmark следует расширить:

1. сохранить piece-placement distance отдельно;
2. сравнивать side-to-move/castling/en-passant state отдельно;
3. учитывать лишние predicted moves после конца ground truth;
4. учитывать пропущенные и дублированные события;
5. latency считать по синхронизированным timestamps, а отрицательное время отмечать как alignment error;
6. экспортировать JSON machine-readable и Markdown summary;
7. benchmark должен возвращать per-game и aggregate metrics.

## 9. Критерии приёмки

Пороговые target значения зафиксировать после получения baseline. Не объявлять, например, `95% move accuracy` достигнутой до выполнения benchmark на held-out реальных партиях. Для пилота разумно задать сначала минимальное требование по безопасности: автоматические commits должны иметь высокую precision, а сомнительные ходы — переходить в uncertain. Число выбирается после первой размеченной выборки и профиля риска.
