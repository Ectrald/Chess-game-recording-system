# Проектирование и план реализации: ЭТАП 3 (STAGE/3)

**Тема этапа:** Контур машинного обучения: датасеты, специализированные доменные аугментации (`AnchoredShear`, `AnchoredScale`), архитектуры нейросетевых классификаторов клеток (EfficientNet-B0 multi-head, ResNet-18 baseline), трекинг экспериментов в MLflow и высокопроизводительный рантайм-инференс в ONNX Runtime ($\le 20$ мс на CPU для батча из 64 клеток).  
**Ветка реализации:** `STAGE/3`  
**Целевой статус:** От изолированных нарезанных снимков клеток — к обученной, калиброванной и аппаратно оптимизированной модели распознавания содержимого 64 полей доски, способной надежно обобщаться на не встречавшиеся ранее шахматные комплекты и ракурсы камеры.

---

## 1. Целевое состояние и обоснование архитектурных решений

### 1.1. Назначение и контекст этапа
В соответствии с архитектурной концепцией системы ([ПЗ, подразд. 2.4.2, 2.4.4](file:///home/ectrald/Repos/Chess-game-recording-system/explanatory%20note/sections/sec_designing.tex#L200-L260), [02_ARCHITECTURE.md](../chess_game_recording_docs/02_ARCHITECTURE.md) и [06_DATA_AND_MLOPS.md](../chess_game_recording_docs/06_DATA_AND_MLOPS.md)), нейросетевой классификатор клеток образует ядро нейробионического контура визуального восприятия.

Его задача — преобразовать 64 участка выпрямленной доски с поправкой на параллакс высоты $\alpha_{\text{height}}$ в матрицу апостериорных вероятностей $P(s) \in [0, 1]^{13}$ для каждого поля $s \in \{\text{a1}\dots\text{h8}\}$. Полученный вектор вероятностей затем поступает в логико-символьный контур (Этап 4) для взвешивания гипотез ходов в целевой функции правдоподобия:
$$S(m) = w_v S_v(m) + w_t S_t(m) + w_g S_g(m) - w_o P_o(m)$$

**Ключевой вызов Этапа 3:** в отличие от классических задач классификации (ImageNet), шахматные фигуры на доске обладают жесткими физико-геометрическими ограничениями и высокой степенью вариативности:
* **Невиданный инвентарь (Unseen Sets):** система должна работать в клубах и на турнирах с произвольными комплектами фигур (деревянные Staunton, пластиковые DGT, утяжеленные клубные);
* **Перспективный наклон фигур:** при углах наклона камеры $\theta \in [50^\circ, 85^\circ]$ верхняя часть фигуры смещается вдоль луча перспективы, тогда как основание всегда неподвижно зафиксировано на плоскости клетки;
* **Жесткий аппаратный бюджет инференса:** распознавание всех 64 полей должно выполняться за время $\le 20$ мс на стандартном CPU, чтобы весь цикл обработки кадра ($< 100$ мс) не приводил к деградации видеопотока.

---

### 1.2. Обоснование проектных решений

#### 1. Факторизованная multi-head архитектура против прямого single-head 13-class
* **Проблема прямого 13-классового softmax:** В стандартной 13-классовой классификации модель нередко путает цвет фигуры (например, белый конь классифицируется как черный конь из-за глубокой тени) или считает пустую клетку с тенью пешкой. При этом вероятность пустой клетки конкурирует с фигурами в общем знаменателе $\sum_{k=1}^{13} e^{z_k}$.
* **Принятое решение (ADR-006):** Использование факторизованной трехголовой архитектуры (концепция `chess-vision` на базе бэкбона **EfficientNet-B0**):
  $$\text{Crop } (112 \times 112) \xrightarrow{\text{Backbone}} \mathbf{h} \in \mathbb{R}^{1280} \begin{cases}
  \xrightarrow{\text{Occupancy Head}} p_{\text{occ}} \in [0, 1] & (2 \text{ логита}) \\
  \xrightarrow{\text{Color Head}} p_{\text{color}} \in \{\text{white}, \text{black}\} & (2 \text{ логита}) \\
  \xrightarrow{\text{Piece Type Head}} p_{\text{type}} \in \{\text{P}, \text{N}, \text{B}, \text{R}, \text{Q}, \text{K}\} & (6 \text{ логитов})
  \end{cases}$$
  Итоговое распределение вероятностей синтезируется детерминированно:
  $$p(\text{empty}) = 1 - p_{\text{occ}}$$
  $$p(\text{wX}) = p_{\text{occ}} \cdot p_{\text{color}}(\text{white}) \cdot p_{\text{type}}(\text{X})$$
  $$p(\text{bX}) = p_{\text{occ}} \cdot p_{\text{color}}(\text{black}) \cdot p_{\text{type}}(\text{X})$$
* **Преимущество:** Ошибки цвета не разрушают уверенность в типе фигуры, а пустые клетки с бликами или тенями оцениваются специализированной головой занятости с маскированием потерь типа и цвета:
  $$\mathcal{L} = w_{\text{occ}} \mathcal{L}_{\text{BCE}} + \mathbb{I}(y_{\text{occ}} = 1) \left[ w_{\text{color}} \mathcal{L}_{\text{color}} + w_{\text{type}} \mathcal{L}_{\text{type}} \right]$$

#### 2. Физико-ориентированные доменные аугментации с фиксацией основания (`AnchoredShear`, `AnchoredScale`)
* **Проблема стандартных аугментаций:** Стандартные преобразования (`RandomRotation`, случайный вертикальный сдвиг) отрывают основание фигуры от клетки, нарушая физический контакт с доской. Модель, обученная на оторванных фигурах, демонстрирует низкую точность на реальных ракурсах $60^\circ\text{--}80^\circ$.
* **Принятое решение:** Разработка специализированных доменных преобразований (`src/ml/transforms.py`):
  1. `AnchoredShear`: аффинный сдвиг по оси $X$, в котором нижняя кромка клетки ($y = y_{\max}$) остается строго неподвижной:
     $$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} 1 & s \\ 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y - y_{\max} \end{pmatrix} + \begin{pmatrix} 0 \\ y_{\max} \end{pmatrix},\quad s \in [-0{,}25; 0{,}25]$$
  2. `AnchoredScale`: вертикальное масштабирование относительно нижнего основания, варьирующее видимую высоту фигуры при сохранении точки физического контакта с полем;
  3. `ShadowInjection`: наложение полупрозрачных градиентных эллипсов, моделирующих тени от фигур и рук;
  4. `SensorNoiseAndBlur`: фильтр Гаусса и аддитивный сенсорный шум дешевых веб-камер.

#### 3. Строгий протокол группового разбиения (Grouped Split by Set)
* **Проблема утечки данных (Data Leakage):** Если кадры одной партии или фотосессии одного комплекта случайно распределяются между train и test, нейросеть запоминает текстуру конкретного дерева/пластика и освещение, демонстрируя 99% на тесте, но полностью отказывая на новом комплекте фигур.
* **Принятое решение:**
  - Разделение датасета выполняется **строго по идентификаторам комплектов и сессий** (`set_id` / `session_id`);
  - Все клетки одного комплекта фигур направляются либо целиком в Train, либо целиком в Test;
  - Test содержит исключительно невиданные ранее физические комплекты (Unseen Sets), гарантируя честную валидацию обобщающей способности.

#### 4. Высокопроизводительный рантайм-инференс в ONNX Runtime (Бюджет $\le 20$ мс на CPU)
* **Проблема прямого выполнения в PyTorch:** Инференс батча из 64 кропов через PyTorch на CPU занимает $80\text{--}150$ мс, создавая узкое горлышко и вызывая деградацию видеопотока.
* **Принятое решение:**
  - Экспорт обученной PyTorch-модели в оптимизированный формат ONNX с операторным сетом 17;
  - Пакетная передача всех 64 кропов в одном тензоре `(64, 3, 112, 112)`;
  - Оптимизация графа ONNX Runtime (constant folding, operator fusion, SIMD AVX2/AVX-512) сокращает латентность инференса до $\mathbf{\le 15\text{--}20}$ **мс на CPU**;
  - Паритетный тест: гарантируется численное соответствие между выходами PyTorch и ONNX Runtime с погрешностью $\max |\Delta p| < 10^{-4}$.

#### 5. MLOps и трекинг экспериментов (MLflow)
* **Управление воспроизводимостью:**
  - Фиксация параметров: бэкбон, гиперпараметры оптимизатора (AdamW, $\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$), конфигурация аугментаций, seed;
  - Логирование метрик по эпохам: Train/Val Loss, Macro F1, Per-Class F1, Occupancy Accuracy, Color Accuracy;
  - Сохранение артефактов каждого эксперимента: тепловая карта матрицы ошибок $13 \times 13$ (`confusion_matrix.png`), отчет `classification_report.json`, чекпоинт `best_model.pt` и оптимизированный `model.onnx`.

#### 6. Стратегия интеграции моделей: Pretrained Chesscog как основная рантайм-модель
* **Архитектурная фиксация:**
  - Для ускорения реализации и непрерывности проверок последующих этапов (Этап 4 Reasoning, Этап 5 Stockfish/хранилище, Этап 6 UI) в качестве **основной рабочей модели приложения** принимается предобученная модель `chesscog` (Wölflein & Arandjelović, 2021);
  - На данном этапе полностью реализуются все программные компоненты контура обучения и MLOps: датасетные абстракции, физические аугментации (`AnchoredShear`, `AnchoredScale`), архитектуры сетей (EfficientNet-B0 multi-head, ResNet-18), маскированные функции потерь, трейнер MLflow, утилиты конвертации и тесты;
  - Полномасштабное обучение собственных сетей на многотысячных выборках и их финальное экспериментальное сравнение с `chesscog` выносятся на завершающий этап проекта.

---

## 2. Структура файлов и модулей этапа STAGE/3

```text
Chess-game-recording-system/
├── configs/
│   └── default_config.yaml         # Обновление: секции piece_classifier и training
├── docs/
│   └── stages/
│       └── STAGE_3.md              # Настоящий документ спецификации
├── scripts/
│   ├── download_model.py           # Скрипт скачивания предобученных весов (Baseline 0)
│   └── run_training.py             # CLI точка запуска обучения и экспорта в ONNX
├── src/
│   ├── core/
│   │   ├── config.py               # Добавление ClassifierConfig и TrainingConfig
│   │   └── datatypes.py            # Добавление ModelManifest и EvaluationReport
│   └── ml/
│       ├── __init__.py             # Экспорт классификаторов, моделей и датасетов
│       ├── auto_labeller.py        # Модуль сбора данных (реализован на Этапе 2)
│       ├── dataset.py              # Загрузка датасетов, манифесты и Grouped Split
│       ├── export.py               # Утилита экспорта PyTorch -> ONNX и валидация паритета
│       ├── models.py               # EfficientNet-B0 Multi-Head, ResNet-18, Masked Loss
│       ├── piece_classifier.py     # BasePieceClassifier, ONNXPieceClassifier, Mock
│       ├── trainer.py              # Цикл обучения PyTorch, валидация, MLflow трекинг
│       └── transforms.py           # AnchoredShear, AnchoredScale, ShadowInjection
└── tests/
    ├── test_dataset_split.py       # Тесты группового сплита, манифестов и балансировки
    ├── test_piece_classifier.py    # Тесты инференса ONNX/Mock, batch-64, латентности
    ├── test_training_pipeline.py   # Тесты лосса, шага обучения, расчета метрик и F1
    └── test_transforms.py          # Тесты геометрии AnchoredShear, AnchoredScale, теней
```

---

## 3. Спецификация контрактов данных и типов (`src/core/datatypes.py` и `config.py`)

### 3.1. Конфигурация классификатора (`ClassifierConfig`)
В `src/core/config.py` вводится конфигурация рантайм-инференса классификатора:

```python
class ClassifierConfig(BaseModel):
    """Конфигурация нейросетевого классификатора содержимого клеток."""
    enabled: bool = Field(default=True, description="Включить нейросетевую классификацию клеток")
    model_path: str = Field(
        default="models/piece_classifier.onnx",
        description="Путь к скомпилированному ONNX-файлу модели"
    )
    architecture: str = Field(
        default="efficientnet_b0",
        description="Архитектура модели: 'efficientnet_b0', 'resnet18' или 'mock'"
    )
    backend: str = Field(
        default="auto",
        description="Движок инференса: 'onnx', 'torch', 'mock' или 'auto'"
    )
    batch_size: int = Field(default=64, ge=1, le=64, description="Размер батча (по умолчанию вся доска 64)")
    input_size: Tuple[int, int] = Field(default=(112, 112), description="Входное разрешение клетки (W, H)")
    occupancy_threshold: float = Field(
        default=0.50, ge=0.01, le=0.99,
        description="Порог вероятности занятости поля tau_o"
    )
    confidence_threshold: float = Field(
        default=0.60, ge=0.01, le=0.99,
        description="Минимальный порог уверенности в типе фигуры"
    )
    device: str = Field(default="cpu", description="Устройство инференса: 'cpu' или 'cuda'")
```

### 3.2. Конфигурация обучения (`TrainingConfig`)
```python
class TrainingConfig(BaseModel):
    """Параметры обучения нейросетевого классификатора и MLOps-трекинга."""
    dataset_dir: str = Field(default="data/dataset", description="Корневая директория датасета")
    output_dir: str = Field(default="models/checkpoints", description="Директория сохранения чекпоинтов")
    architecture: str = Field(default="efficientnet_b0", description="Бэкбон: 'efficientnet_b0' или 'resnet18'")
    batch_size: int = Field(default=32, ge=8, le=256, description="Размер мини-батча при обучении")
    epochs: int = Field(default=30, ge=1, le=200, description="Максимальное число эпох обучения")
    learning_rate: float = Field(default=1e-3, ge=1e-6, le=1e-1, description="Начальный шаг обучения (LR)")
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1e-2, description="L2-регуляризация")
    patience: int = Field(default=5, ge=1, le=20, description="Patience ранней остановки по Val Macro F1")
    val_split_ratio: float = Field(default=0.15, ge=0.05, le=0.30)
    test_split_ratio: float = Field(default=0.15, ge=0.05, le=0.30)
    seed: int = Field(default=42, description="Генератор случайных чисел для воспроизводимости")
    mlflow_tracking_uri: Optional[str] = Field(default=None, description="URI сервера MLflow (None = локально)")
    experiment_name: str = Field(default="chess_piece_classification", description="Имя эксперимента в MLflow")
```

### 3.3. Манифест обученной модели (`ModelManifest`)
```python
class ModelManifest(BaseModel):
    """Метаданные скомпилированного артефакта модели классификатора."""
    model_name: str = Field(description="Наименование модели (e.g. piece_classifier_v1)")
    architecture: str = Field(description="Архитектура бэкбона")
    format: str = Field(description="Формат артефакта: 'onnx' или 'pytorch'")
    input_shape: List[int] = Field(description="Форма входного тензора [B, C, H, W]")
    classes: List[str] = Field(default_factory=lambda: list(PIECE_CLASSES))
    val_macro_f1: float = Field(ge=0.0, le=1.0, description="Macro F1 на валидационной выборке")
    test_macro_f1: float = Field(ge=0.0, le=1.0, description="Macro F1 на контрольной выборке (unseen sets)")
    latency_cpu_ms: float = Field(description="Среднее время инференса 64 клеток на CPU в мс")
    trained_at: str = Field(description="ISO 8601 временная метка обучения")
```

### 3.4. Отчет об оценке качества (`EvaluationReport`)
```python
class EvaluationReport(BaseModel):
    """Комплексный отчет о метриках классификации по 13 классам."""
    overall_accuracy: float = Field(ge=0.0, le=1.0)
    occupancy_accuracy: float = Field(ge=0.0, le=1.0)
    color_accuracy: float = Field(ge=0.0, le=1.0)
    macro_precision: float = Field(ge=0.0, le=1.0)
    macro_recall: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    per_class_f1: Dict[str, float] = Field(description="Метрика F1 для каждого из 13 классов")
    confusion_matrix: List[List[int]] = Field(description="Матрица ошибок 13x13")
    expected_calibration_error: float = Field(ge=0.0, le=1.0, description="Калибровочная ошибка ECE")
```

---

## 4. Спецификация алгоритмов и программных интерфейсов

### 4.1. Доменные физические аугментации (`src/ml/transforms.py`)

Модуль реализует специализированные трансформации для одиночных изображений клеток (поддерживаются форматы PIL Image и NumPy uint8 BGR/RGB):

```python
class AnchoredShear:
    """Аффинный сдвиг по оси X с фиксацией нижней кромки основания клетки."""
    def __init__(self, max_shear: float = 0.25) -> None: ...
    def __call__(self, img: Image.Image) -> Image.Image: ...

class AnchoredScale:
    """Масштабирование по высоте с опорой на нижнюю кромку клетки."""
    def __init__(self, scale_range: Tuple[float, float] = (0.85, 1.20)) -> None: ...
    def __call__(self, img: Image.Image) -> Image.Image: ...

class ShadowInjection:
    """Синтез полупрозрачных направленных теней от фигур и рук."""
    def __init__(self, max_shadow_intensity: float = 0.40, p: float = 0.5) -> None: ...
    def __call__(self, img: Image.Image) -> Image.Image: ...

class SensorNoiseAndBlur:
    """Моделирование расфокусировки и шума матрицы веб-камеры."""
    def __init__(self, blur_prob: float = 0.3, noise_std: float = 4.0) -> None: ...
    def __call__(self, img: Image.Image) -> Image.Image: ...

def get_chess_train_transforms(target_size: Tuple[int, int] = (112, 112)) -> Any:
    """Фабрика цепочки обучающих аугментаций с сохранением физики поля."""
    ...

def get_chess_val_transforms(target_size: Tuple[int, int] = (112, 112)) -> Any:
    """Детерминированная цепочка валидационной предобработки (Resize + Normalization)."""
    ...
```

---

### 4.2. Управление датасетом и Grouped Split (`src/ml/dataset.py`)

Модуль отвечает за структурирование образцов, расчет весов балансировки классов и строгое групповое разбиение:

1. **Групповой сплиттер (`grouped_split_dataset`):**
   - Принимает список образцов `List[AutoLabellerSample]` или путь к манифесту;
   - Извлекает группу для каждого образца: `group_id = sample.set_id` (или префикс партии/видео);
   - Распределяет уникальные группы между `train` (70%), `val` (15%), `test` (15%);
   - Гарантирует, что ни один комплект фигур из `test` не встречается в `train`.

2. **Класс `ChessCropDataset`:**
   - Читает нарезанные изображения с диска по относительному пути `image_rel_path`;
   - Преобразует строковый класс (`wP`, `empty` и т.д.) в числовой индекс $0\dots 12$;
   - Раскладывает класс на мульти-головые метки:
     - `occ_label`: 0 (пусто) или 1 (занято);
     - `color_label`: 0 (белый), 1 (черный) или -1 (для пустых);
     - `type_label`: 0..5 (P, N, B, R, Q, K) или -1 (для пустых);
   - Применяет композицию аугментаций.

3. **Балансировка классов (`compute_class_weights`):**
   - Вычисляет сбалансированные веса классов $w_c = \frac{N}{13 \cdot N_c}$ для борьбы с преобладанием пустых полей.

---

### 4.3. Архитектуры нейросетевых моделей (`src/ml/models.py`)

1. **`MultiHeadPieceClassifier` (Основная модель):**
   - Бэкбон: `EfficientNet-B0` (или `ResNet-18`);
   - Замена классификационного слоя на три параллельные полносвязные головы:
     - `occupancy_fc: Linear(feat_dim, 2)`
     - `color_fc: Linear(feat_dim, 2)`
     - `type_fc: Linear(feat_dim, 6)`
   - Метод `forward(x)`: возвращает словарь логитов `{"occ": ..., "color": ..., "type": ...}`;
   - Метод `predict_probabilities(x)`: вычисляет объединенные 13-классовые вероятности.

2. **`SingleHeadPieceClassifier` (Контрольная модель):**
   - Стандартный ResNet-18 с выходом `Linear(512, 13)`.

3. **`MaskedMultiHeadLoss`:**
   - Комбинирует взвешенную Cross-Entropy для каждой головы;
   - Исключает потери по цвету и типу для пустых клеток ($y_{\text{occ}} = 0$).

---

### 4.4. Высокопроизводительный рантайм-инференс (`src/ml/piece_classifier.py`)

Предоставляет унифицированный программный фасад системы компьютерного зрения:

```python
class BasePieceClassifier(ABC):
    """Абстрактный базовый класс классификатора клеток."""
    @abstractmethod
    def classify_cell(self, cell_patch: np.ndarray, square: str) -> CellObservation: ...

    @abstractmethod
    def classify_board(
        self,
        cell_patches: Dict[str, np.ndarray],
        rois: Optional[Dict[str, CellROI]] = None,
    ) -> Dict[str, CellObservation]:
        """Пакетная классификация всех 64 полей за один проход."""
        ...

class ONNXPieceClassifier(BasePieceClassifier):
    """Высокопроизводительный классификатор на базе ONNX Runtime (CPU/CUDA)."""
    def __init__(self, config: Optional[ClassifierConfig] = None) -> None: ...
    ...

class TorchPieceClassifier(BasePieceClassifier):
    """Классификатор на базе чистой PyTorch-модели для экспериментов."""
    ...

class MockPieceClassifier(BasePieceClassifier):
    """Детерминированный классификатор для модульных тестов и CI."""
    ...

def create_piece_classifier(
    config: Optional[ClassifierConfig] = None,
    backend: str = "auto",
) -> BasePieceClassifier:
    """Фабрика с автоматическим выбором доступного бэкбонда."""
    ...
```

* **Инвариант пакетной обработки (`classify_board`):**
  Метод формирует из словаря 64 изображений клеток тензор размерности $(64, 3, 112, 112)$, нормализует средние и дисперсии ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$) и выполняет **ровно один** вызов `session.run()`, заполняя все поля `CellObservation` параллельно.

---

### 4.5. Контур обучения и трекинг MLflow (`src/ml/trainer.py`)

Класс `PieceClassifierTrainer`:
- Управляет эпохами обучения, прямым и обратным проходами;
- Вычисляет метрики по валидационной выборке: Macro F1, Per-class F1, Accuracy;
- Реализует раннюю остановку (Early Stopping) с сохранением весов наилучшей эпохи;
- Интегрируется с **MLflow**:
  - `mlflow.log_params(...)`: архитектура, lr, weight_decay, batch_size, split ratios;
  - `mlflow.log_metrics(...)`: эпохальные метрики качества;
  - `mlflow.log_artifact(...)`: матрица ошибок (`confusion_matrix.png`), `classification_report.json`, экспортный ONNX-файл.
- При недоступности сервера MLflow сохраняет структурированные JSON-логи локально в `output_dir`.

---

### 4.6. Экспорт в ONNX и загрузка Baseline 0 (`src/ml/export.py`, `scripts/download_model.py`)

1. **Модуль `src/ml/export.py`:**
   - Функция `export_to_onnx(model, output_path, input_size=(112, 112), batch_size=64, opset=17)`: выполняет трассировку графа и сериализацию;
   - Функция `verify_onnx_parity(torch_model, onnx_path, tolerance=1e-4)`: прогоняет тестовый батч и проверяет численное совпадение вероятностей.

2. **Скрипт `scripts/download_model.py`:**
   - Загружает предобученные веса моделей из открытого репозитория `chesscog` (Wölflein & Arandjelović, 2021) или локального зеркала;
   - Проверяет SHA-256 хеш-сумму загруженного файла;
   - Помещает артефакт в `models/piece_classifier.onnx`.

---

## 5. Обновление конфигурации (`configs/default_config.yaml` и `src/core/config.py`)

В `configs/default_config.yaml` добавляются секции `piece_classifier` и `training`:

```yaml
piece_classifier:
  enabled: true
  model_path: "models/piece_classifier.onnx"
  architecture: "efficientnet_b0"
  backend: "auto"
  batch_size: 64
  input_size: [112, 112]
  occupancy_threshold: 0.50
  confidence_threshold: 0.60
  device: "cpu"

training:
  dataset_dir: "data/dataset"
  output_dir: "models/checkpoints"
  architecture: "efficientnet_b0"
  batch_size: 32
  epochs: 30
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 5
  val_split_ratio: 0.15
  test_split_ratio: 0.15
  seed: 42
  experiment_name: "chess_piece_classification"
```

---

## 6. Комплексный план тестирования для STAGE/3

| Тестовый модуль | Проверяемый функционал | Критерий успешности |
| :--- | :--- | :--- |
| `tests/test_transforms.py` | 1. `AnchoredShear`: основание клетки ($y=y_{\max}$) сохраняет исходные координаты, верхняя часть смещается.<br>2. `AnchoredScale`: высота фигуры варьируется, контактная линия с полем неизменна.<br>3. `ShadowInjection`: яркость под маской снижается, форма теней физически правдоподобна.<br>4. Корректная обработка изображений PIL и NumPy массивов. | Преобразованные изображения имеют заданные размеры $112 \times 112$; нижняя кромка геометрически инвариантна. |
| `tests/test_dataset_split.py` | 1. Корректность парсинга манифеста `dataset_manifest.json`.<br>2. Строгий Grouped Split: множества `set_id` в train и test взаимно не пересекаются ($\text{Train} \cap \text{Test} = \emptyset$).<br>3. Расчет весов классов для балансировки выборки.<br>4. Преобразование строковых меток в мульти-головые индексы (occ, color, type). | 100% изоляция тестовых комплектов фигур; веса классов обратно пропорциональны частотам. |
| `tests/test_piece_classifier.py` | 1. Инференс `MockPieceClassifier`: корректная генерация 64 объектов `CellObservation`.<br>2. Инференс `ONNXPieceClassifier` (при наличии рантайма): пакетная обработка 64 клеток.<br>3. Замер латентности: batch-64 выполняется за $\le 20$ мс на CPU.<br>4. Корректность вероятностных полей `is_occupied`, `predicted_piece`, `confidence`. | Все 64 поля содержат валидные вероятности $p \in [0, 1]$, сумма $\sum p_i = 1$; латентность $\le 20$ мс. |
| `tests/test_training_pipeline.py` | 1. Функция потерь `MaskedMultiHeadLoss`: отсутствие штрафов за цвет/тип на пустых клетках.<br>2. Шаг оптимизатора и градиентный спуск (на синтетическом микробатче).<br>3. Расчет метрик: Overall Accuracy, Macro F1, Per-class F1, матрица 13x13.<br>4. Экспорт модели в ONNX и проверка числового паритета $\max \|\Delta p\| < 10^{-4}$. | Функция потерь убывает; метрики вычисляются строго математически; паритет выходов соблюден. |

---

## 7. Критерии приёмки (Definition of Done) этапа STAGE/3

Этап `STAGE/3` считается завершённым при выполнении следующих требований:

1. [x] Реализованы конфигурационные модели `ClassifierConfig`, `TrainingConfig` и контракты данных `ModelManifest`, `EvaluationReport` в `src/core/config.py` и `src/core/datatypes.py`.
2. [x] Реализован модуль специализированных доменных аугментаций (`src/ml/transforms.py`) с поддержкой `AnchoredShear`, `AnchoredScale`, инъекции теней и шума матрицы.
3. [x] Реализован модуль управления датасетом (`src/ml/dataset.py`) со строгим групповым разбиением по комплектам (Grouped Split by Set) и балансировкой весов 13 классов.
4. [x] Реализованы архитектуры нейросетевых моделей (`src/ml/models.py`): EfficientNet-B0 multi-head, ResNet-18 single-head и маскированная функция потерь `MaskedMultiHeadLoss`.
5. [x] Реализован высокопроизводительный фасад инференса (`src/ml/piece_classifier.py`) с поддержкой пакетной классификации 64 клеток, ONNX Runtime бэкбона и mock-режима.
6. [x] Реализован цикл обучения и валидации (`src/ml/trainer.py`) с расчетом матрицы ошибок $13 \times 13$, ECE, Macro F1 и логированием в MLflow.
7. [x] Реализованы утилиты экспорта в ONNX с проверкой численного паритета выходов ($\max |\Delta p| < 10^{-4}$) и скрипт загрузки эталонных весов Baseline 0 (`scripts/download_model.py`).
8. [x] Проверено соблюдение аппаратного бюджета: латентность инференса батча из 64 клеток на CPU составляет $\le 20$ мс.
9. [x] Разработан полный набор модульных тестов (`tests/test_transforms.py`, `tests/test_dataset_split.py`, `tests/test_piece_classifier.py`, `tests/test_training_pipeline.py`).
10. [x] Полный комплекс тестов `pytest tests/` (Этап 0 + Этап 1 + Этап 2 + Этап 3) выполняется на 100% без ошибок.
11. [x] Проектный документ этапа синхронизирован с документацией системы (`06_DATA_AND_MLOPS.md`, `08_IMPLEMENTATION_PLAN.md`) и пояснительной запиской.

