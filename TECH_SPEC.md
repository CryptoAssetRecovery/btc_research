
```
btc_research/
│
├── pyproject.toml
├── README.md
├── btc_research/
│   ├── __init__.py
│   ├── config/           # *.yaml templates
│   ├── data/             # cached OHLCV
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base_indicator.py
│   │   ├── registry.py
│   │   ├── datafeed.py
│   │   ├── engine.py
│   │   ├── backtester.py
│   │   ├── broker.py
│   │   ├── optimiser.py         # (grid / bayes search – phase 7)
│   │   └── schema.py            # json‑schema for configs
│   ├── indicators/
│   │   ├── __init__.py
│   │   └── <indicator>.py …
│   └── cli/
│       ├── download.py
│       ├── backtest.py
│       └── optimise.py
└── tests/
    ├── fixtures/
    └── test_*.py
```

---

## 2 Technical implementation roadmap

> **Legend**: **M** = mandatory for MVP, *O* = optional / nice‑to‑have.
> Estimates are in **net engineering hours** assuming one senior Python dev. Buffer for meetings etc. not included.

| #      | Phase & Outcomes               | Key Tasks                                                                                                                                                               | Deliverables                                             | Hours   |
| ------ | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------- |
| **0**  | **Dev‑ops bootstrap**          | ‑ Initialise git repo, enable GitHub Actions<br>‑ Create `pyproject.toml` with Poetry (Python 3.11)<br>‑ Add dev‑deps: `black`, `ruff`, `pytest`, `pre‑commit`, `mypy`  | Reproducible env & CI job that runs formatting + tests   | **2 h** |
| **1**  | **Skeleton & Registry (M)**    | ‑ Scaffold directories<br>‑ Implement `core/registry.py` with `register()` / `get()`<br>‑ Unit test registry round‑trip                                                 | Green test: indicator class registers and is retrievable | **2 h** |
| **2**  | **BaseIndicator contract (M)** | ‑ Create `base_indicator.py` (ABC)<br>‑ Docstring with param conventions<br>‑ Add mypy type hints                                                                       | Pass mypy & tests (instantiation + compute stub)         | **1 h** |
| **3**  | **DataFeed service (M)**       | ‑ Wrap existing fetcher into `class DataFeed`<br>‑ Expose `get(symbol, tf, start, end, source)`<br>‑ Ensure caching & resampling<br>‑ Write unit tests with fixture CSV | Cached minute‑level data loads in <200 ms                | **4 h** |
| **4**  | **Reference indicators (M)**   | ‑ Port EMA & RSI using `ta` lib<br>‑ Register with decorator<br>‑ Provide `params()` defaults                                                                           | Two indicators return correct columns vs `ta` baseline   | **2 h** |
| **5**  | **YAML schema & loader (M)**   | ‑ Draft JSON‑schema in `schema.py`<br>‑ Validate configs at CLI entry<br>‑ Support env var overrides (`${VAR}`)                                                         | Invalid YAML fails fast with clear message               | **3 h** |
| **6**  | **Confluence Engine v1 (M)**   | ‑ Implement `Engine.run(cfg)`<br>‑ Multi‑TF caching & forward‑fill alignment<br>‑ Returned DataFrame + debug logging                                                    | Smoke script demonstrating 1 h + 5 m merge               | **6 h** |
| **7**  | **Backtester wrapper (M)**     | ‑ Create `Backtester` class<br>‑ Dynamically build Backtrader strategy from `logic` expressions (safe `numexpr`/`pandas.eval`)<br>‑ Expose `.run(df) → dict`            | Backtest of EMA‑bias + RSI config prints equity & Sharpe | **6 h** |
| **8**  | **CLI utilities (M)**          | ‑ `cli/download.py`: just calls DataFeed.get()<br>‑ `cli/backtest.py`: load YAML → Engine → Backtester<br>‑ Register `btc-download`, `btc-backtest` in Poetry           | One‑command demo from README                             | **2 h** |
| **9**  | **Test suite & coverage (M)**  | ‑ pytest + fixtures for fetch, indicator, engine, backtester<br>‑ Minimum 80 % coverage gate in CI                                                                      | Bad refactor trips CI                                    | **4 h** |
| **10** | **Grid optimiser (O)**         | ‑ `core/optimiser.py` with Cartesian product or Optuna<br>‑ Summarise runs to CSV                                                                                       | `btc-optimise` explores 12 combos overnight              | 5 h     |
| **11** | **Docs & Examples (O)**        | ‑ Expand README<br>‑ Autogen API docs with `pdoc`<br>‑ Add `examples/` notebooks                                                                                        | New dev onboards in <30 min                              | 3 h     |
| **12** | **Continuous deployment (O)**  | ‑ Build GitHub Action to publish versioned wheel to internal PyPI or GitHub Packages on tag                                                                             | `pip install btc‑research==0.1.0`                        | 2 h     |

**MVP cut:** phases 0–9 → **28 h (\~3 full days)**
Full vision incl. optimiser & docs → **38 h (\~5 days)**

---

## 3 Critical path & sequencing

1. **Registry + BaseIndicator** are prerequisites for everything.
2. **DataFeed** must land before Engine (to supply data).
3. **Engine** must finish before Backtester.
4. Tests & CI should be wired as soon as DataFeed is deterministic (phase 3) to avoid regressions.
5. Optimiser can be parallel‑developed later because it only orchestrates CLI calls.

---

## 4 Technology choices (justification)

| Component               | Library                   | Reasoning                                    |
| ----------------------- | ------------------------- | -------------------------------------------- |
| Data handling           | **pandas** 2.x            | ubiquitous, already in your prototype        |
| Indicator maths         | **ta‑lib or `ta`**        | wide coverage; avoids re‑coding formulas     |
| Back‑testing            | **Backtrader**            | you already use it; mature; multi‑TF support |
| Config                  | **YAML + json‑schema**    | human‑readable, but statically validated     |
| Expression evaluation   | **numexpr / pandas.eval** | safe, vectorised boolean logic               |
| Optimisation (optional) | **Optuna**                | distributed hyper‑param grids / Bayesian     |
| Packaging               | **Poetry**                | lockfile + scripts + publish in one tool     |

---

## 5 Definition of “done” for MVP

* `poetry install && btc-backtest config/demo.yaml`
  **produces** a JSON block with trades, PnL, Sharpe and draws a matplotlib equity curve (optional `--plot`).
* Adding a new indicator requires **only**:
  `indicators/my_new.py` with the decorator **+** YAML edit – *no* core code changes.
* CI workflow green on main branch: format, type‑check, 80 % tests.
* README shows <15‑line quick‑start and lists available CLI commands.