# Проектирование и план реализации: ЭТАП 4 (STAGE/4)

**Тема этапа:** Логико-символьный контур (Reasoning): генерация легальных кандидатов (`LegalMoveCandidateGenerator`), многокритериальный скоринг правдоподобия ($S(m)$), шлюз принятия решений (`DecisionGate`), разрешение неоднозначностей (`UNCERTAIN`), атомарная фиксация ходов (`GameStateManager.commit`), PGN/FEN проекции и сквозная верификация специальных ходов (рокировка, взятие на проходе, превращение пешки).  
**Ветка реализации:** `STAGE/4`  
**Целевой статус:** От вероятностных поклеточных наблюдений $P(s) \in [0, 1]^{13}$ — к надежной, математически верифицированной фиксации официального состояния шахматной партии ($B_t \to B_{t+1}$), исключающей регистрацию невозможных или недопустимых ходов при любых зрительных помехах и окклюзиях.

---

## 1. Целевое состояние и обоснование архитектурных решений

### 1.1. Назначение и контекст этапа
В соответствии с концепцией гибридного искусственного интеллекта ([ПЗ, подразд. 2.4.3, 2.4.4](file:///home/ectrald/Repos/Chess-game-recording-system/explanatory%20note/sections/sec_designing.tex#L320-L400), [02_ARCHITECTURE.md](../chess_game_recording_docs/02_ARCHITECTURE.md) и [05_REASONING_AND_ANALYSIS.md](../chess_game_recording_docs/05_REASONING_AND_ANALYSIS.md)), нейросетевой классификатор (Этап 3) оперирует непрерывными, зашумленными и неполными вероятностными данными. Отдельный снимок клетки может содержать ошибку из-за блика ламп, остаточной тени руки или схожести силуэтов (например, ферзя и слона под острым углом).

**Фундаментальный принцип надежности:**
> *Система никогда не изменяет официальную шахматную позицию на основе простого покоординатного argmax нейросети.* Регистрация хода выполняется исключительно логико-символьным контуром, сопоставляющим наблюдаемые зрительные изменения со строгой онтологией шахматных правил FIDE.

Если на доске зафиксировано физическое изменение полей, контур рассуждений определяет, какой именно **один легальный ход** из множества допустимых $M(B_t)$ перевел доску из состояния $B_t$ в наблюдаемое состояние $B_{t+1}$. Любая неопределенность, противоречие или отсутствие выраженного лидера среди кандидатов оставляет авторитетное состояние партии неизменным и переводит автомат в состояние `UNCERTAIN`.

---

### 1.2. Обоснование проектных решений

#### 1. Генерация кандидатов от авторитетной позиции (Legal Move Restriction)
* **Проблема прямого распознавания 64 клеток:** На доске $8 \times 8$ число возможных конфигураций 13 классов составляет $13^{64} \approx 10^{71}$. Если распознавать каждую клетку независимо, любая единичная ошибка создает невозможную конфигурацию (например, три белых короля или пешки на первой горизонтали).
* **Принятое решение (ADR-007):** Поиск гипотез сужается исключительно до множества легальных ходов текущей проверенной позиции:
  $$m \in M(B_t) = \text{board.legal\_moves}$$
  В любой типичной шахматной позиции мощность множества $|M(B_t)| \le 35\text{--}40$ ходов. Это гарантирует:
  1. Невозможность регистрации нелегального хода (хода сквозь фигуру, хода связанной фигурой, хода королем под шах, хода не в свою очередь);
  2. Вычислительную сложность $O(|M(B_t)|)$ вместо перебора $13^{64}$, что занимает $< 1$ мс на CPU.

#### 2. Топологические сигнатуры изменений клеток
* **Классификация геометрии ходов:** Каждый тип шахматного хода оставляет уникальную топологическую сигнатуру затронутых полей $A(m) \subset \{\text{a1}\dots\text{h8}\}$:
  * **Обычный ход (тихий или взятие):** ровно 2 поля: $A(m) = \{s_{\text{from}}, s_{\text{to}}\}$. Поле $s_{\text{from}}$ обязано стать пустым (`empty`), а поле $s_{\text{to}}$ — получить перемещенную фигуру;
  * **Рокировка (Kingside / Queenside):** ровно 4 поля: $A(m) = \{\text{e1}, \text{g1}, \text{f1}, \text{h1}\}$ (для белых $0\text{-}0$) или $\{\text{e1}, \text{c1}, \text{d1}, \text{a1}\}$ (для белых $0\text{-}0\text{-}0$). Освобождаются e1 и h1, заполняются g1 (король) и f1 (ладья);
  * **Взятие на проходе (En Passant):** ровно 3 поля: $A(m) = \{s_{\text{from}}, s_{\text{to}}, s_{\text{captured}}\}$. Пешка перемещается по диагонали, а сбитая пешка соперника на соседней горизонтали исчезает;
  * **Превращение пешки (Promotion):** 2 поля: пешка уходит с 7-й горизонтали и появляется на 8-й в виде одной из 4 фигур $\{Q, R, B, N\}$.
* **Роль сигнатур:** Сопоставление визуально изменившихся клеток $\Delta S = \text{diff}(B_t, O)$ с $A(m)$ позволяет мгновенно отфильтровать заведомо неподходящие ходы на противоположном конце доски. При этом окклюдированные поля не считаются противоречием, что обеспечивает устойчивость к перекрытию доски рукой.

#### 3. Комплексная целевая функция правдоподобия $S(m)$
Каждый легальный кандидат $m \in M(B_t)$ оценивается по многокритериальной формуле из Таблицы 2.2 и формулы 2.1 пояснительной записки:
$$S(m) = w_v S_v(m) + w_t S_t(m) + w_g S_g(m) - w_o P_o(m)$$

Где слагаемые приведены к единой шкале $[0, 1]$:
* **$S_v(m)$ — визуальное логарифмическое правдоподобие:**
  Согласованность ожидаемой расстановки фигур $B_m = \text{apply}(B_t, m)$ с матрицей вероятностей $P(s)$:
  $$S_v(m) = \alpha_{\text{aff}} S_{\text{affected}}(m) + (1 - \alpha_{\text{aff}}) S_{\text{global}}(m)$$
  $$S_{\text{affected}}(m) = \frac{1}{|A(m) \setminus \text{Occluded}|} \sum_{s \in A(m) \setminus \text{Occluded}} P_s(p_m(s))$$
  Для полей вне хода ($s \notin A(m)$) глобальный штраф начисляется только при высокой уверенности детектора в конфликте, игнорируя незначительные шумы;
* **$S_t(m)$ — временная стабильность:** мера затухания межкадрового движения в окрестности полей $A(m)$ на протяжении окна верификации;
* **$S_g(m)$ — геометрическое качество:** показатель устойчивости четырехугольника доски $S_g \in [0, 1]$ из гомографии;
* **$P_o(m)$ — штраф за окклюзию:** суммарная доля перекрытия маской руки ключевых полей хода $A(m)$:
  $$P_o(m) = \frac{1}{|A(m)|} \sum_{s \in A(m)} \text{occlusion\_ratio}(s)$$

#### 4. Двухфакторный шлюз принятия решения (Decision Gate)
Для фиксации хода требуется выполнение двух независимых условий:
1. **Абсолютный порог достоверности:** Лучший кандидат $m_1$ обязан набрать достаточный балл:
   $$S(m_1) \ge \tau_{\text{accept}} \quad (\text{по умолчанию } 0{,}55)$$
2. **Запас уверенности (Confidence Margin):** Разница баллов между лучшим ($m_1$) и вторым по качеству ($m_2$) кандидатом должна превышать установленный порог:
   $$\text{margin} = S(m_1) - S(m_2) \ge \Delta \quad (\text{по умолчанию } 0{,}15)$$
3. **Критическая окклюзия:** Ни одно из полей $s \in A(m_1)$ не должно быть перекрыто рукой более чем на $\tau_{\text{occ\_crit}} = 0{,}50$.

Если условие шлюза нарушено (например, два хода $e2\text{-}e4$ и $e2\text{-}e3$ имеют близкие баллы из-за неполного обзора), система **не угадывает ход**, а активирует статус `UNCERTAIN`.

#### 5. Атомарный менеджер состояния игры (`GameStateManager`)
* **Авторитетность:** `GameStateManager` — единственный компонент системы, имеющий право изменять официальное состояние партии $B_t \to B_{t+1}$.
* **Транзакционность и защита от сбоев:**
  1. Блокировка рентабельного мьютекса (Thread-Safety);
  2. Валидация легальности хода на изолированной копии `chess.Board`;
  3. Вычисление точной стандартной нотации SAN (с учетом знаков взятия `x`, шаха `+`, мата `#`);
  4. Атомарное применение хода `board.push(move)`;
  5. Формирование неизменяемой структуры `MoveEvent`;
  6. Добавление записи в append-only журнал событий;
  7. Идемпотентность: повторная подача того же хода или кадра игнорируется, предотвращая рассинхронизацию полуходов.

---

## 2. Структура файлов и модулей этапа STAGE/4

```text
Chess-game-recording-system/
├── configs/
│   └── default_config.yaml              # Обновление: параметры decision gate и acceptance margin
├── docs/
│   └── stages/
│       └── STAGE_4.md                   # Настоящий документ спецификации
├── src/
│   ├── core/
│   │   ├── config.py                    # Дополнение DecisionConfig порогами tau_accept, max_occlusion
│   │   └── datatypes.py                 # Дополнение MoveEvent и MoveHypothesis служебными полями
│   └── reasoning/
│       ├── __init__.py                  # Экспорт GameStateManager, MoveScorer, CandidateGenerator, Gate
│       ├── candidate_generator.py       # LegalMoveCandidateGenerator и топологические сигнатуры
│       ├── decision_gate.py             # DecisionGate, DecisionResult и UncertaintyHandler
│       ├── fsm.py                       # Интеграция шлюза решений в фазу VERIFYING конечного автомата
│       ├── game_state_manager.py        # Авторитетный GameStateManager, транзакции, FEN/PGN проекция
│       └── scorer.py                    # MoveScorer, S(m), S_v(m), визуальное правдоподобие
└── tests/
    ├── test_candidate_generator.py      # Тесты генерации ходов, рокировок, en passant, превращений
    ├── test_decision_gate.py            # Тесты шлюза принятия решений, margin, фильтрации неопределенности
    ├── test_game_state_manager.py       # Тесты commit, идемпотентности, PGN экспорта, FEN before/after
    ├── test_move_scorer.py              # Тесты расчета S(m), S_v, штрафов за окклюзию и весов
    └── test_reasoning_e2e.py            # Сквозное воспроизведение партий test_short_game и test_special_moves
```

---

## 3. Спецификация контрактов данных (`src/core/datatypes.py` и `config.py`)

### 3.1. Усиленный контракт `MoveHypothesis`
В `src/core/datatypes.py` структура кандидата расширяется полями декомпозиции визуальной оценки и флагом легальности:

```python
class MoveHypothesis(BaseModel):
    """Кандидат хода, оцененный по визуальным, временным и геометрическим доказательствам."""
    move_uci: str = Field(pattern=r"^[a-h][1-8][a-h][1-8][qrbn]?$", description="Ход в нотации UCI")
    move_san: Optional[str] = Field(default=None, description="Человекочитаемая нотация SAN (e.g. 'e4', 'Nf3')")
    changed_squares: List[str] = Field(description="Поля, затронутые ходом: A(m)")
    score_visual: float = Field(ge=0.0, le=1.0, description="Визуальное правдоподобие S_v(m)")
    score_temporal: float = Field(default=1.0, ge=0.0, le=1.0, description="Временная стабильность S_t(m)")
    score_geometry: float = Field(default=1.0, ge=0.0, le=1.0, description="Геометрическая стабильность S_g(m)")
    penalty_occlusion: float = Field(default=0.0, ge=0.0, description="Штраф за окклюзию полей хода P_o(m)")
    total_score: float = Field(ge=0.0, le=1.0, description="Итоговый интегральный балл S(m)")
    is_legal: bool = Field(default=True, description="Флаг легальности относительно текущей позиции")
```

### 3.2. Усиленный контракт `MoveEvent`
Структура подтвержденного хода дополняется уникальным идентификатором события и ключом идемпотентности:

```python
class MoveEvent(BaseModel):
    """Событие подтвержденного хода, зафиксированное авторитетным менеджером состояния."""
    event_id: str = Field(description="Уникальный UUID события фиксации хода")
    ply: int = Field(ge=1, description="Номер полухода партии (1 = 1. Белые, 2 = 1... Чёрные)")
    fen_before: str = Field(description="FEN-строка позиции до совершения хода")
    move_uci: str = Field(pattern=r"^[a-h][1-8][a-h][1-8][qrbn]?$", description="Канонический ход UCI")
    move_san: str = Field(description="Стандартная нотация FIDE (SAN)")
    fen_after: str = Field(description="FEN-строка позиции после совершения хода")
    confidence: float = Field(ge=0.0, le=1.0, description="Запас уверенности Delta: S(m1) - S(m2)")
    timestamp: float = Field(ge=0.0, description="Временная метка видеозаписи в секундах")
    evidence_frame_id: int = Field(ge=0, description="Индекс опорного кадра видеопотока")
    idempotency_key: Optional[str] = Field(default=None, description="Ключ предотвращения дублирования события")
    eval_centipawns: Optional[int] = Field(default=None, description="Оценка Stockfish в сантипешках (Этап 5)")
    is_blunder: Optional[bool] = Field(default=None, description="Флаг грубой ошибки (Этап 5)")
```

### 3.3. Результат работы шлюза решений (`DecisionResult`)
```python
class DecisionStatus(str, Enum):
    ACCEPTED = "accepted"          # Ход однозначно подтвержден и принят
    UNCERTAIN = "uncertain"        # Неоднозначность (малый margin или низкий score)
    REJECTED = "rejected"          # Ход не удовлетворяет базовым критериям достоверности
    NO_CANDIDATE = "no_candidate"  # Нет ни одного легального кандидата

class DecisionResult(BaseModel):
    """Результат сопоставления гипотез в шлюзе принятия решений."""
    status: DecisionStatus
    accepted_hypothesis: Optional[MoveHypothesis] = None
    margin: float = Field(default=0.0, ge=0.0, description="Разница S(m1) - S(m2)")
    rejection_reason: Optional[str] = None
    top_hypotheses: List[MoveHypothesis] = Field(default_factory=list)
    uncertain_squares: List[str] = Field(default_factory=list)
```

---

## 4. Спецификация алгоритмов и программных интерфейсов

### 4.1. Генератор кандидатов и топологические сигнатуры (`src/reasoning/candidate_generator.py`)

Класс `LegalMoveCandidateGenerator`:
1. **Детекция изменившихся полей (`detect_changed_squares`):**
   - Сравнивает текущую позицию `board: chess.Board` с наблюдением `obs: BoardObservation`;
   - Для каждого поля $s \in \text{CHESS\_SQUARES}$:
     - Ожидаемая фигура: `expected = board.piece_at(chess.parse_square(s))`;
     - Наблюдаемая фигура: `observed = obs.cells[s].predicted_piece`;
     - Если ожидаемое и наблюдаемое состояния не совпадают и поле не окклюдировано (`not obs.cells[s].is_occluded`), поле заносится в `changed_squares`.
2. **Топологическое картирование ходов (`get_affected_squares`):**
   - Для любого хода $m \in \text{board.legal\_moves}$ возвращает точный список затрагиваемых полей $A(m)$:
     - Обычный ход: `[from_square, to_square]`;
     - Рокировка белых $0\text{-}0$: `["e1", "g1", "f1", "h1"]`;
     - Рокировка белых $0\text{-}0\text{-}0$: `["e1", "c1", "d1", "a1"]`;
     - Рокировка чёрных $0\text{-}0$: `["e8", "g8", "f8", "h8"]`;
     - Рокировка чёрных $0\text{-}0\text{-}0$: `["e8", "c8", "d8", "a8"]`;
     - Взятие на проходе (`board.is_en_passant(m)`): `[from_square, to_square, captured_square]`, где `captured_square` — поле сбитой пешки.
3. **Генерация кандидатов (`generate_candidates`):**
   - Извлекает `list(board.legal_moves)`;
   - Если переданы `changed_squares`, ранжирует ходы: приоритет отдается кандидатам, у которых $A(m) \cap \text{changed\_squares} \neq \emptyset$;
   - Возвращает структурированный список пар `(move: chess.Move, affected_squares: List[str])`.

```python
class LegalMoveCandidateGenerator:
    def __init__(self) -> None: ...
    def detect_changed_squares(self, board: chess.Board, observation: BoardObservation) -> List[str]: ...
    def get_affected_squares(self, board: chess.Board, move: chess.Move) -> List[str]: ...
    def generate_candidates(
        self,
        board: chess.Board,
        observation: Optional[BoardObservation] = None,
    ) -> List[Tuple[chess.Move, List[str]]]: ...
```

---

### 4.2. Многокритериальный оценщик правдоподобия (`src/reasoning/scorer.py`)

Класс `MoveScorer` реализует расчет целевой функции правдоподобия:
1. **Визуальная правдоподобность $S_v(m)$:**
   - Для каждого кандидата моделируется гипотетическая доска: `board_hypo = board.copy()`, `board_hypo.push(move)`;
   - Для каждого поля $s \in A(m)$:
     - Если поле перекрыто рукой (`is_occluded == True`), оно исключается из штрафования;
     - Иначе из `obs.cells[s].probabilities` извлекается вероятность ожидаемой фигуры $p = P_s(p_{\text{expected}})$;
     - Балл затронутых полей: $S_{\text{aff}} = \frac{1}{|A_{\text{reliable}}|} \sum \max(\epsilon, p)$;
   - Балл глобальной непротиворечивости $S_{\text{glob}}$: проверка остальных полей доски (штраф начисляется только если наблюдается уверенная чужая фигура на свободном поле);
   - Итоговое $S_v(m) = 0{,}75 \cdot S_{\text{aff}} + 0{,}25 \cdot S_{\text{glob}}$.
2. **Временная стабильность $S_t(m)$:**
   - Оценивается локальное движение на затронутых полях: $S_t(m) = 1{,}0 - \max_{s \in A(m)} M(s)$.
3. **Штраф за окклюзию $P_o(m)$:**
   - Средняя доля перекрытия маской руки: $P_o(m) = \frac{1}{|A(m)|} \sum_{s \in A(m)} \text{occlusion\_ratio}(s)$.
4. **Итоговый балл $S(m)$:**
   $$S(m) = \text{clip}\left(w_v S_v(m) + w_t S_t(m) + w_g S_g - w_o P_o(m), 0{,}0, 1{,}0\right)$$

```python
class MoveScorer:
    def __init__(self, config: Optional[DecisionConfig] = None) -> None: ...
    def score_hypothesis(
        self,
        board: chess.Board,
        move: chess.Move,
        affected_squares: List[str],
        observation: BoardObservation,
    ) -> MoveHypothesis: ...
    def rank_candidates(
        self,
        board: chess.Board,
        candidates: List[Tuple[chess.Move, List[str]]],
        observation: BoardObservation,
    ) -> List[MoveHypothesis]: ...
```

---

### 4.3. Шлюз принятия решений (`src/reasoning/decision_gate.py`)

Класс `DecisionGate`:
- Принимает упорядоченный по убыванию $S(m)$ список гипотез `ranked_hypotheses`;
- Вычисляет запас уверенности $\text{margin} = S(m_1) - S(m_2)$ (при одном кандидате $\text{margin} = 1{,}0$);
- Проверяет критерии шлюза:
  1. $S(m_1) \ge \tau_{\text{accept}}$ (порог абсолютной достоверности, default $0{,}55$);
  2. $\text{margin} \ge \Delta$ (минимальный отрыв лидера от второго кандидата, default $0{,}15$);
  3. $S_g \ge 0{,}50$ (качество геометрии доски);
  4. $\max_{s \in A(m_1)} \text{occlusion\_ratio}(s) < \tau_{\text{occ\_crit}}$ (отсутствие критического перекрытия полей хода рукой).
- При успешном выполнении всех условий возвращает `DecisionResult(status=DecisionStatus.ACCEPTED, accepted_hypothesis=m1)`.
- При нарушении порога margin или score возвращает `status=DecisionStatus.UNCERTAIN` со списком спорных полей и причиной отказа.

```python
class DecisionGate:
    def __init__(self, config: Optional[DecisionConfig] = None) -> None: ...
    def evaluate(
        self,
        ranked_hypotheses: List[MoveHypothesis],
        observation: BoardObservation,
    ) -> DecisionResult: ...
```

---

### 4.4. Авторитетный менеджер состояния игры (`src/reasoning/game_state_manager.py`)

Класс `GameStateManager`:
- Инкапсулирует единственный авторитетный объект `chess.Board()`;
- Поддерживает рентабельный мьютекс (`threading.RLock`) для исключения гонок при параллельных запросах;
- **Метод `commit(hypothesis: MoveHypothesis, observation: BoardObservation) -> MoveEvent`:**
  1. Проверяет `board.is_legal(move)`;
  2. Генерирует SAN: `move_san = board.san(move)`;
  3. Запоминает `fen_before = board.fen()`;
  4. Применяет ход: `board.push(move)`;
  5. Запоминает `fen_after = board.fen()`;
  6. Формирует ключ идемпотентности: `idempotency_key = f"{ply}_{move.uci()}_{observation.frame_id}"`;
  7. Создает `MoveEvent` и сохраняет в историю `_history: List[MoveEvent]`;
  8. Уведомляет слушателей: `on_move_committed(event)`.
- **Проекция и экспорт PGN (`export_pgn`):**
  - Создает `chess.pgn.Game()` с заголовками (Event, Site, Date, Round, White, Black, Result) и экспортирует каноническую строку PGN.
- **Откат и воспроизведение (`rollback`, `replay`):**
  - `rollback()`: отменяет последний полуход `board.pop()` и удаляет событие из истории;
  - `replay(events)`: проверяет целостность журнала событий и восстанавливает позицию с нуля, гарантируя точное совпадение с финальным FEN.
- **Ручная коррекция оператором (`manual_commit`):**
  - Позволяет оператору принудительно подтвердить ход при разрешении состояния `UNCERTAIN`.

```python
class GameStateManager:
    def __init__(
        self,
        initial_fen: str = chess.STARTING_FEN,
        on_move_committed: Optional[Callable[[MoveEvent], None]] = None,
    ) -> None: ...
    @property
    def current_board(self) -> chess.Board: ...
    @property
    def current_fen(self) -> str: ...
    @property
    def ply(self) -> int: ...
    def commit(self, hypothesis: MoveHypothesis, observation: BoardObservation) -> MoveEvent: ...
    def manual_commit(self, move_uci: str, observation: Optional[BoardObservation] = None) -> MoveEvent: ...
    def rollback(self) -> Optional[MoveEvent]: ...
    def export_pgn(self, headers: Optional[Dict[str, str]] = None) -> str: ...
    def replay(self, events: List[MoveEvent]) -> chess.Board: ...
```

---

### 4.5. Интеграция со шлюзом решений в `GameFSM` (`src/reasoning/fsm.py`)

Конечный автомат расширяется связкой с `DecisionGate` и `GameStateManager`:
- При нахождении в состоянии `VERIFYING`:
  - Когда накоплена серия стабильных кадров $N_{\text{verify}}$, наблюдение $O$ передается в `CandidateGenerator` $\to$ `MoveScorer` $\to$ `DecisionGate`;
  - Если `decision.status == DecisionStatus.ACCEPTED`:
    - Выполняется `game_state_manager.commit(decision.accepted_hypothesis, observation)`;
    - Автомат переходит в `STABLE` и ожидает следующего хода;
  - Если `decision.status == DecisionStatus.UNCERTAIN`:
    - Автомат переходит в `UNCERTAIN`;
    - Ожидаются дополнительные стабильные кадры либо ручное подтверждение оператора;
  - Если `decision.status == DecisionStatus.NO_CANDIDATE`:
    - Позиция доски не изменилась (ложное срабатывание детектора движения);
    - Автомат возвращается в `STABLE` без инкремента полухода.

---

## 5. Обновление конфигурации (`configs/default_config.yaml` и `src/core/config.py`)

В секцию `decision` добавляются пороги абсолютной достоверности и допустимой окклюзии:

```yaml
decision:
  weight_visual: 0.50             # w_v: Вес визуального логарифмического правдоподобия
  weight_temporal: 0.25           # w_t: Вес временной стабильности
  weight_geometry: 0.15           # w_g: Вес устойчивости геометрии
  weight_occlusion: 0.10          # w_o: Вес штрафа за перекрытие рукой
  min_confidence_margin: 0.15     # Delta: Минимальный запас уверенности S(m1) - S(m2)
  min_acceptance_score: 0.55      # tau_accept: Минимальный абсолютный балл для шлюза
  max_occlusion_threshold: 0.50   # Предельно допустимая окклюзия полей хода
```

---

## 6. Комплексный план тестирования для STAGE/4

| Тестовый модуль | Проверяемый функционал | Критерий успешности |
| :--- | :--- | :--- |
| `tests/test_candidate_generator.py` | 1. Генерация ходов начальной позиции (20 легальных ходов).<br>2. Топологические сигнатуры: 2 поля для тихих ходов и взятий.<br>3. Специальные ходы: рокировка (4 поля), en passant (3 поля), превращение пешки (2 поля, 4 суффикса).<br>4. Приоритезация кандидатов по изменившимся полям. | Все легальные ходы строятся строго по `python-chess`; топологические сигнатуры соответствуют типам ходов. |
| `tests/test_move_scorer.py` | 1. Расчет $S_v(m)$ по матрице вероятностей 64 клеток.<br>2. Расчет $S(m)$ по формуле взвешенной суммы с нормализацией весов.<br>3. Штраф $P_o(m)$ при наличии окклюзии на полях хода.<br>4. Ранжирование кандидатов: истинный ход получает максимальный балл. | Истинный ход занимает top-1; окклюзия корректно снижает скор; итоговый балл строго в диапазоне $[0, 1]$. |
| `tests/test_decision_gate.py` | 1. Успешный пропуск истинного хода при $S(m_1) \ge 0{,}55$ и $\text{margin} \ge 0{,}15$.<br>2. Отклонение и переход в `UNCERTAIN` при малом запасе уверенности ($\text{margin} < 0{,}15$).<br>3. Отклонение при критической окклюзии полей хода ($\ge 0{,}50$).<br>4. Корректный статус `NO_CANDIDATE` при нулевых изменениях доски. | Шлюз пропускает только надежные ходы; любые коллизии переводят результат в `UNCERTAIN`. |
| `tests/test_game_state_manager.py` | 1. Атомарный `commit`: генерация SAN, обновление FEN до/после, инкремент ply.<br>2. Идемпотентность: повторный вызов с тем же ходом не создает дубликатов.<br>3. Экспорт партии в валидный формат PGN с заголовками FIDE.<br>4. Откат хода `rollback` и восстановление истории `replay`. | Ни один нелегальный ход не фиксируется; PGN проходит валидацию `chess.pgn.read_game`; replay совпадает с FEN. |
| `tests/test_reasoning_e2e.py` | 1. Сквозная симуляция короткой партии `test_short_game.pgn` (Мат Легаля / Детский мат).<br>2. Сквозная симуляция партии со всеми спецходами `test_special_moves.pgn` (рокировки, en passant, promotion).<br>3. Интеграция `GameFSM` + `CandidateGenerator` + `Scorer` + `Gate` + `GameStateManager`. | 100% распознанных ходов партии; финальный FEN совпадает с эталоном PGN; отсутствие сбоев на спецходах. |

---

## 7. Критерии приёмки (Definition of Done) этапа STAGE/4

Этап `STAGE/4` считается завершённым при выполнении следующих требований:

1. [x] Реализованы расширенные контракты `MoveHypothesis`, `MoveEvent`, `DecisionResult`, `DecisionStatus` в `src/core/datatypes.py`.
2. [x] Дополнен класс `DecisionConfig` параметрами порогов `min_acceptance_score` и `max_occlusion_threshold` в `src/core/config.py` и `configs/default_config.yaml`.
3. [x] Реализован генератор легальных кандидатов (`src/reasoning/candidate_generator.py`) с поддержкой топологических сигнатур обычных ходов, рокировок, взятий на проходе и превращений пешек.
4. [x] Реализован многокритериальный оценщик гипотез `MoveScorer` (`src/reasoning/scorer.py`) по формуле $S(m) = w_v S_v + w_t S_t + w_g S_g - w_o P_o$.
5. [x] Реализован шлюз принятия решений `DecisionGate` (`src/reasoning/decision_gate.py`) с двухфакторной проверкой достоверности и запаса уверенности $\text{margin} \ge \Delta$.
6. [x] Реализован авторитетный менеджер состояния игры `GameStateManager` (`src/reasoning/game_state_manager.py`) с потокобезопасным `commit`, генерацией SAN, расчетом FEN до/после, защитой идемпотентности, PGN-проекцией и функцией `rollback`.
7. [x] Реализована интеграция шлюза решений и менеджера состояния с конечным автоматом `GameFSM` в фазе `VERIFYING`.
8. [x] Разработан полный набор тестов (`tests/test_candidate_generator.py`, `tests/test_move_scorer.py`, `tests/test_decision_gate.py`, `tests/test_game_state_manager.py`, `tests/test_reasoning_e2e.py`).
9. [x] Сквозные тесты на эталонных партиях `test_short_game.pgn` и `test_special_moves.pgn` выполняются с результатом 100% корректных committed ходов без ложных переходов.
10. [x] Полный комплекс тестов `pytest tests/` (Этапы 0, 1, 2, 3, 4) выполняется на 100% без единой ошибки.
11. [x] Проектный документ этапа синхронизирован с проектной документацией (`05_REASONING_AND_ANALYSIS.md`, `08_IMPLEMENTATION_PLAN.md`) и пояснительной запиской.

