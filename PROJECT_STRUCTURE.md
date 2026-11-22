# TraceForge — структура проекта, соответствие критериям и план развития

Документ для человека, который впервые открыл репозиторий: что здесь лежит, как всё связано, чем доказан каждый критерий и что делать дальше.
Все числа получены реальными запусками (`FINAL_BENCHMARK.json`, `FINAL_ABLATION.json`, `SPEC_COMPLIANCE.md`); где что-то **не** проверялось, это сказано прямо.

## 1. Что это

TraceForge — офлайн-среда (только стандартная библиотека Python ≥ 3.11) для обучения и оценки ИИ-агентов, выполняющих финансовый аудит.
Генератор строит синтетическую компанию (журнал, счета, платежи с разнесением, курсы валют, политика периода), внедряет проверяемые ошибки и
законные «двойники» ошибок, кладёт публичные данные отдельно от закрытой истины. Агент расследует данные через 5 инструментов, отправляет
структурированные выводы; оценщик сопоставляет их один-к-одному с истиной и считает награду `clip(2·F1 − 1 − 0.002·шаги, −1, 1)`.

## 2. Дерево репозитория

```
Genesis1/
├── README.md                    вход: английская сводка + подробное описание (рус.), команды CLI, формат данных, контракты
├── PROJECT_STRUCTURE.md         этот файл
├── VIDEO_SCRIPT.md              сценарий демонстрационного видео (4 минуты, по секундам)
├── JUDGE_SUMMARY.md             короткая сводка для жюри: требования, результаты, ограничения
├── FINAL_AUDIT.md               аудит требований, безопасности, воспроизводимости (с доказательствами)
├── FINAL_ENGINEERING_REPORT.md  инженерный отчёт (бенчмарки по уровням, демо, производительность)
├── ENGINEERING_AUDIT.md         ранний аудит и найденные/исправленные дефекты
├── SPEC_COMPLIANCE.md           матрица «требование → реализация → тесты», генерируется из реального прогона pytest
├── ARCHITECTURE.md · DESIGN.md  диаграммы (Mermaid) и принятые решения
├── DEMO_SCRIPT.md               сценарий презентации на 3 минуты (терминал + HTML-отчёт)
├── FINAL_BENCHMARK.json/.csv    бенчмарк: 52 эпизода × 4 политики = 208 прогонов (+ _summary, _by_level)
├── FINAL_ABLATION.json/.csv     чувствительность награды: аудитор с одной убранной способностью
├── Dockerfile · .github/workflows/ci.yml   контейнер для недоверенных агентов; CI (3 ОС × Python 3.11/3.12)
├── pyproject.toml · requirements*.txt · LICENSE · configs/default.json
│
├── src/traceforge/
│   ├── cli.py · main.py · __main__.py     команды: generate, demo, evaluate, benchmark, ablation, pairs, paired-eval, generate-eval, verify, audit, inspect
│   ├── constants.py · config.py · schemas.py · money.py · episode_io.py · exceptions.py
│   │       schemas: Episode/PublicEpisode/PrivateEpisode, разбор submit (поле суммы: claimed_amount_minor | amount | amount_minor)
│   │       money: целые minor units + Decimal-курсы, ROUND_HALF_UP построчно
│   ├── generator/
│   │   ├── company.py · financial_world.py · documents.py · ledger.py     чистая компания → документы → журнал
│   │   ├── clean_cases.py                                                 законные «двойники» ошибок (возвраты, повторы сумм, split-платежи, граница периода)
│   │   ├── mutations.py                                                   4 независимых движка ошибок: duplicate, allocation, fx, cutoff
│   │   ├── compound_scenarios.py                                          2 взаимодействующих сценария
│   │   ├── provenance.py · validation.py                                  манифесты мутаций; независимый валидатор (пересчёт журнала, ссылки, доказательства, итоги)
│   │   ├── transformations.py                                             метаморфика: перенумерация id, перестановка, переименование, смена UTC-смещений документов
│   │   ├── pairs.py                                                       пары-двойники и скрытый набор из секрета организатора
│   │   └── builder.py                                                     оркестрация, пресеты сложности, наборы train/dev
│   ├── environment/
│   │   ├── audit_env.py · actions.py · observations.py · pagination.py    reset/step в стиле Gymnasium, лимиты 100 шагов / 50 строк
│   │   ├── calculator.py                                                  безопасный калькулятор (AST + Decimal)
│   │   └── isolation.py                                                   агент в отдельном процессе, очищенное окружение, ограниченный ввод-вывод
│   ├── evaluation/   matcher.py (максимальное паросочетание) · reward.py · metrics.py · evidence_validator.py · evaluator.py · oracle_policy.py (только сторона оценщика)
│   ├── policies/     random · reconciliation · mark_all · ablations.py (аудитор с убранной способностью; не в реестре лидерборда)
│   ├── runner/       episode_runner · benchmark · paired · ablation · agent_worker · agent_protocol · sandbox (audit-hook) · container (docker)
│   ├── trajectories/ recorder.py · reader.py   траектории без скрытой истины
│   └── reporting/    terminal · html_report · demo_report · audit · performance · spec_compliance · transformation_report · verification_report
│
├── datasets/
│   ├── train/ dev/            по 10 эпизодов; public/ (ledger.csv, invoices.jsonl, payments.jsonl, fx.csv, policy, manifest) и private/ (truth, provenance, clean_baseline)
│   └── pairs/train|dev/       по 10 пар-двойников (+ private/transformation.json с картой id)
├── tests/            32 файла, 1370+ тестов (контракт, изоляция, генератор, матчер, награда, метаморфика, песочница, ablation, пары, CI/контейнер)
├── examples/         gym_loop.py (reset/step), custom_agent.py + run_custom_agent.py (внешний агент по JSON-протоколу)
└── tools/            clean_env_check.sh (проверка в чистом venv), container_smoke.sh, mutation_check.py (проверка самих тестов), stress_gen.py, ...
```

`reports/`, `trajectories/*.jsonl`, `graphify-out/` создаются локально и в git не попадают.

## 3. Поток данных

```
seed + config ─► generator ─► чистая компания ─► мутации (≤ 1 на документ) ─► независимый validator
                                                         │
                         ┌───────────────────────────────┴────────────────────────┐
                    public/ (агент видит)                                  private/ (только оценщик)
                         │                                                  truth · provenance · baseline
                    AuditEnv.reset/step ◄── JSON-наблюдения ──► агент (отдельный процесс, песочница / контейнер)
                         │                                                          │
                         └──────────── submit(findings) ───────────────► Evaluator: one-to-one matcher ─► F1 ─► reward
                                                                                    │
                                                                   trajectories (вид агента, без истины)
```

## 4. Критерии оценивания → где и чем доказано

| Критерий (баллы) | Что проверяется | Где реализовано | Чем доказано (команда / тест) |
|---|---|---|---|
| **Contract & Isolation (25)** | `reset(seed, config)`, `step` → 5-кортеж, 5 инструментов, ≤ 50 строк, ≤ 100 шагов, read-only, нет утечек | `environment/`, `runner/isolation.py`, `sandbox.py`, `container.py` | `test_environment`, `test_pagination`, `test_isolation`, `test_adversarial`, `test_sandbox` (12 атак: без песочницы все проходят, с ней все блокированы), `test_container_ci`; LeakGuard на каждом сообщении |
| **Data & Ground Truth (25)** | реалистичные данные, 4 типа ошибок + 2 compound, чистые компании, decoys, достоверная истина | `generator/` | `test_generator*`, `test_mutations`, `test_compound_scenarios`, `test_validator_integrity` (независимый пересчёт), `test_property`; `python -m traceforge audit` |
| **Reward & Evaluation (25)** | one-to-one, TP/FP/FN, precision/recall/F1, официальная формула, 4 политики | `evaluation/`, `policies/` | `test_matcher*`, `test_reward`, `test_evidence`, `test_policies`; `FINAL_BENCHMARK.*`; **`FINAL_ABLATION.*`** — награда реагирует на каждую способность, текст объяснения не оценивается |
| **Reproducibility & Generalization (15)** | один seed → те же байты, пары-преобразования, скрытый набор | `generator/transformations.py`, `pairs.py`, `runner/paired.py` | `test_reproducibility` (в т.ч. между процессами и `PYTHONHASHSEED`), `test_transformations`, `test_metamorphic`, `test_pairs_eval`; `paired-eval`: устойчивый аудитор consistent = 1.000, читающий время «по настенным часам» — 0.000 |
| **Engineering & Explanation (10)** | структура, документация, демо, воспроизводимый запуск | весь репозиторий | `README.md` (EN-сводка + RU), `python -m traceforge demo` + HTML-отчёт, `verify` → `SPEC_COMPLIANCE.md` (75 требований), `tools/clean_env_check.sh` |

## 5. Шпаргалка команд

```powershell
pip install -e . ; pip install -r requirements-dev.txt
python -m traceforge demo                         # демо: расследование, оценка, benchmark, reports/demo_report.html
python -m traceforge benchmark --suite all --seeds 8 --out . --name FINAL_BENCHMARK
python -m traceforge ablation --out . --name FINAL_ABLATION
python -m traceforge pairs --split all            # пары-двойники для train/dev
python -m traceforge paired-eval --split dev      # устойчивость политик к преобразованиям
python -m traceforge generate-eval --out eval_hidden --count 20 --secret-out secret.txt   # скрытый набор организатора
python -m pytest                                  # все тесты
python -m traceforge verify                       # тесты + матрица требований + аудит + отчёты
bash tools/clean_env_check.sh                     # чистое окружение: install → tests → benchmark → demo
```

## 6. Честные ограничения (не скрываем)

* Эталонный аудитор `reconciliation` решает все эпизоды (F1 = 1.000): это следствие того, что каждая ошибка по построению выводима из публичных данных. Бенчмарк из 4 политик
  поэтому не различает сильных агентов; различие награды по способностям показано в `FINAL_ABLATION.*`. **LLM-агент не запускался**, выводов о качестве моделей нет.
* Песочница — audit-hook только для Python-агентов (скоростная неровность, не граница безопасности). Контейнерная граница (`Dockerfile`, `container.py`) проверена только на уровне
  сформированной команды: Docker в среде разработки не было.
* CI (`.github/workflows/ci.yml`) описан, но его результат не наблюдался из среды разработки. Python 3.11, Linux и macOS локально не проверялись.
* Допущения генератора: четыре семейства ошибок, фиксированные смещения UTC без DST, ежедневные курсы включая выходные; допуск суммы ±1 minor unit и определение `inspection_steps`
  (действия до `submit`) — опубликованные решения реализации.
* Знание кода и угадываемого seed позволяет восстановить истину: скрытый набор нужно генерировать из секрета (`generate-eval`), а не из малых seed.

## 7. Что делать дальше

**До дедлайна (22 ноября, 23:59 по Астане):**
1. Открыть вкладку **Actions** репозитория и убедиться, что CI зелёный на всех 6 комбинациях; при красном — прислать лог, исправлю.
2. Записать видео по `VIDEO_SCRIPT.md` (4 минуты) и загрузить так, как требуют организаторы (платформа сдачи мне неизвестна — следуйте их инструкции).
3. Убедиться, что в репозитории последний коммит (`git log -1`) и ссылка на репозиторий вписаны в форму сдачи.

**Развитие после хакатона (по убыванию пользы):**
1. **Адаптер LLM-агента** (локальная модель, без платных API) поверх JSON-протокола — первый честный замер того, чего не может простой сверщик; сложность добавлять через `difficulty`, число валют и бюджет шагов.
2. **Более трудные задачи, где простая сверка не хватает:** ошибки, выводимые только из нескольких документов и политики; шумные описания; частично наблюдаемые данные.
3. **Новые семейства ошибок и DST/часовые пояса с переходом времени;** отдельная таблица `allocations` как самостоятельная сущность.
4. **Проверка на Linux/macOS/Python 3.11** (CI) и запуск `container_smoke.sh` на машине с Docker; затем перевести статус изоляции из «PARTIAL» в «PASS».
5. **Обёртка Gymnasium** (`gym.register`) и цикл обучения с траекториями (`trajectories/`) для RL/SFT.
6. **Публичный лидерборд** скрытого набора: `generate-eval` + `paired-eval` на стороне организатора, секрет не хранится в репозитории.
