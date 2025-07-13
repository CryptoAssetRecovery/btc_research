Below is a **practical blueprint** you can follow to turn the two ad‑hoc scripts into a **modular research harness** that:

* lets you register indicators once and reuse them everywhere
* supports any mix of multi‑time‑frame logic you describe in a single YAML file
* allows one‑click back‑tests (or walk‑forward tests later) against data pulled and cached with CCXT.

---

## 1 High–level layout

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

*Everything inside `btc_research/core` is deliberately **framework code** – it should hardly ever change.
Everything under `indicators/` or `config/` is **strategy‑specific** and may change daily.*

---

## 2 Core building blocks

| File                   | Responsibility                                                                                                                                                                              | Key public methods                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **core/registry.py**   | Global plug‑in registry. Keeps mapping `{name: class}`.                                                                                                                                     | `register(name)(cls) → cls`, `get(name)`                  |
| **core/datafeed.py**   | **Class wrapper** around your `fetch_ohlcv` (now a private `_fetch_batch`). Handles caching, gap filling, and returning pandas DataFrames in *any* timeframe.                               | `get(symbol, tf, start, end, source="binanceus")`         |
| **core/engine.py**     | Reads a YAML “experiment”, loads requested timeframes, instantiates indicators via registry, calls their `compute()` to obtain columns, and combines them into one multi‑indexed DataFrame. | `run(config_path) → pandas.DataFrame with signal columns` |
| **core/backtester.py** | Very thin wrapper around Backtrader; only converts the DataFrame produced by the engine into feeds and injects the *dynamically* assembled strategy class.                                  | `run(df, cash, slippage, commission) → stats`             |
| **core/broker.py**     | (Optional) later‑on place to unify live‑trading executors (ccxt) and broker simulation; start as a façade around Backtrader’s broker.                                                       |                                                           |

All these files know **nothing** about RSI, FVG, etc. – they only depend on the registry.

---

## 3 Indicator interface

```python
# btc_research/core/base_indicator.py
from abc import ABC, abstractmethod
import pandas as pd

class Indicator(ABC):
    """Contract every custom indicator must fulfil."""

    @classmethod
    @abstractmethod
    def params(cls) -> dict:
        """Return default hyper‑parameters for YAML auto‑documentation."""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given OHLCV DataFrame in this indicator's timeframe,
        return **one or more** columns indexed exactly like `df`.
        Column names should be prefixed with the indicator's short‑name.
        """
```

Each concrete indicator is as small as:

```python
# btc_research/indicators/rsi.py
import pandas as pd
import ta.momentum as ta_mom
from btc_research.core.registry import register
from btc_research.core.base_indicator import Indicator

@register("RSI")
class RSI(Indicator):
    @classmethod
    def params(cls):
        return {"length": 14}

    def __init__(self, length=14):
        self.length = length

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.Series(
            ta_mom.RSIIndicator(close=df["close"], window=self.length).rsi(),
            name=f"RSI_{self.length}"
        )
        return out.to_frame()
```

The decorator:

```python
# btc_research/core/registry.py
_registry = {}

def register(name):
    def _decorator(cls):
        _registry[name] = cls
        return cls
    return _decorator

def get(name):
    return _registry[name]
```

Now adding a brand‑new indicator is literally **one file** with that decorator – no changes elsewhere.

---

## 4 YAML‑driven experiments

```yaml
# config/ema‑bias‑rsi‑entry.yaml
name: "EMA bias + RSI entry"
symbol: "BTC/USDC"
exchange: "binanceus"
timeframes:
  bias:   "1h"
  entry:  "5m"

indicators:
  # any kwargs are passed into the Indicator constructor
  - id: "EMA_200"
    type: "EMA"
    timeframe: "1h"
    length: 200

  - id: "RSI_14"
    type: "RSI"
    timeframe: "5m"
    length: 14

logic:
  entry_long:
    - "EMA_200_trend == 'bull'"         # auto‑generated by EMA indicator
    - "RSI_14 < 30"
  exit_long:
    - "RSI_14 > 70"
  # leaving exit_short/entry_short empty disables shorts

backtest:
  cash: 10000
  commission: 0.0004
  slippage: 0.0
  from: "2024-01-01"
  to:   "2024-06-30"
```

### Why a tiny DSL instead of free‑form Python?

* Re‑test **hundreds of combos** by just editing YAML (great for grid search).
* Configs double as documentation of every idea you ever tried.
* Can be validated by JSON‑schema to catch typos.

---

## 5 Inside `core/engine.py` (confluence engine)

Pseudocode for clarity:

```python
class Engine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.indicator_objects = []

    def _load_timeframes(self) -> dict[str, pd.DataFrame]:
        data = {}
        for tf in set(ind["timeframe"] for ind in self.cfg["indicators"]):
            data[tf] = DataFeed.get(
                symbol=self.cfg["symbol"],
                tf=tf,
                start=self.cfg["backtest"]["from"],
                end=self.cfg["backtest"]["to"],
                source=self.cfg.get("exchange", "binanceus")
            )
        return data

    def _instantiate_indicators(self, data_by_tf):
        for spec in self.cfg["indicators"]:
            klass = registry.get(spec["type"])
            obj   = klass(**{k:v for k,v in spec.items() if k not in ("id","type","timeframe")})
            out   = obj.compute(data_by_tf[spec["timeframe"]])
            out.columns = [f"{spec['id']}_{c}" if c != spec["id"] else spec["id"] for c in out.columns]
            data_by_tf[spec["timeframe"]] = data_by_tf[spec["timeframe"]].join(out)
            self.indicator_objects.append(obj)

    def run(self) -> pd.DataFrame:
        data_by_tf = self._load_timeframes()
        self._instantiate_indicators(data_by_tf)

        # combine MTFs by forward‑filling higher‑TF cols onto entry timeframe index
        entry_tf = self.cfg["timeframes"]["entry"]
        base = data_by_tf[entry_tf].copy()
        for tf, df in data_by_tf.items():
            if tf == entry_tf: continue
            # reindex to entry tf
            joined = df.reindex(base.index, method="ffill")
            base = base.join(joined, rsuffix=f"_{tf}")
        return base
```

The output is one tidy DataFrame ready for:

1. vectorised research (quick `df.query()` to count signals) **or**
2. handing over to Backtrader in `core/backtester.py`.

Backtester just builds a throw‑away Strategy class whose `next()` calls `self._df.iloc[self.baridx]` to see columns and fire trades when `eval(expr)` for every rule returns `True`. This keeps indicator logic 100 % outside BT.

---

## 6 Command‑line entry points

```bash
poetry run btc-download   config/ema-bias-rsi-entry.yaml
poetry run btc-backtest   config/ema-bias-rsi-entry.yaml --plot
poetry run btc-optimise   config/ema-bias-rsi-entry.yaml --grid rsi_length=10,14,21 ema_length=100,200
```

Each simply:

```python
def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    df  = Engine(cfg).run()
    stats = Backtester(cfg).run(df)
    print(json.dumps(stats, indent=2))
```

---

## 7 Refactoring your existing scripts

| Current script    | Drop‑in replacement                                                                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datafeed.py`     | code moves into `core/datafeed.py` as `class DataFeed:` with an internal `_fetch_batch` identical to your function. All public interaction via `DataFeed.get()` |
| `run_backtest.py` | logic split across `Engine`, `Backtester`, and small `cli/backtest.py` harness (≈20 LOC).                                                                       |

No behaviour changes – only **responsibilities set in stone and discoverable imports**.

---

## 8 Clean code & project hygiene from day 1

* **Poetry** (or `pip-tools`) + lockfile → same env on every dev box / CI runner.
* **Pre‑commit** with `ruff`, `black`, `isort`.
* **pytest**: unit‑test each indicator’s `compute()` against a 50‑row fixture.
* **mypy** optional; makes YAML → dict → object mapping safer.
* **doq / pdoc** for auto‑generated docs (indicators automatically documented via `params()`).

---

## 9 Next steps in order

| Step                                                                                      | Outcome                | Est. effort |
| ----------------------------------------------------------------------------------------- | ---------------------- | ----------- |
| 1    Scaffold repo & Poetry, create `core/registry.py`, move your fetcher into `DataFeed` | Solid skeleton         | 1 h         |
| 2    Write two reference indicators (`EMA`, `RSI`) and register them                      | Proof registry works   | 1 h         |
| 3    Implement `Engine.run()` (focus on single‑TF first, then add MTF ffill)              | Get combined DataFrame | 2 h         |
| 4    Port minimal Backtester that trades on a *single boolean column* (e.g. `go_long`)    | End‑to‑end smoke test  | 1 h         |
| 5    Add YAML schema + CLI + logging                                                      | Usable by non‑coders   | 2 h         |
| 6    Gradually port your ICT strategy logic into YAML rules, then start adding FVG etc.   | Real research begins   | continuous  |

After step 4 you already have a repeatable pipeline; every later commit is incremental.

---

### Real‑world analogy

Think of **`core/`** as the chassis and engine of a track‑car that rarely changes.
Each **indicator** is just a different set of tyres or aero‑kit you can bolt on for a lap, defined in YAML.
The **confluence engine** is the pit‑crew: it selects the right parts from the garage and mounts them; the **backtester** is your timing system that tells whether the new setup shaved off seconds.

Keep that separation clear and the project will grow pain‑free.

---

### Short answer summary

* **Package structure:** keep stable framework code in `core/`, strategy‑specific code in `indicators/`, configs in YAML.
* **Indicator plug‑ins:** one decorator registers any new indicator; zero framework edits required.
* **Confluence engine:** loads multi‑TF data, computes indicators, merges into a single DataFrame.
* **Backtester:** minimal wrapper around Backtrader; strategy rules come from YAML expressions.
* **CLI tools:** `download`, `backtest`, `optimise`.
* **Hygiene:** Poetry, pre‑commit, pytest, docs from day one.

With this skeleton in place you can start **researching edges immediately** and extend forever without tangled code.
