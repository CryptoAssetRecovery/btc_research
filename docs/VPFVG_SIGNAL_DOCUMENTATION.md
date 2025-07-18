# VPFVGSignal Confluence Indicator

## Overview

The VPFVGSignal indicator is a sophisticated confluence indicator that combines Volume Profile and Fair Value Gap (FVG) analysis to identify high-probability trading opportunities. This indicator implements the core trading logic from the VP-FVG strategy, focusing on two primary signal types:

1. **Reversal Setup (vf_long)**: Bullish FVG near Low Volume Node (LVN)
2. **Continuation Setup (vf_short)**: Bearish FVG overlapping High Volume Node (HVN) with POC shift

## Key Features

- **Multi-indicator confluence**: Combines Volume Profile and FVG analysis
- **ATR-based distance calculations**: Uses adaptive thresholds based on market volatility
- **Look-ahead bias prevention**: All signals are properly shifted to prevent future data leakage
- **Configurable parameters**: Fully customizable tolerance levels and thresholds
- **Comprehensive diagnostics**: Provides detailed signal analysis data

## Signal Logic

### Reversal Setup (vf_long)

**Conditions:**
- Active bullish FVG signal present
- FVG midpoint within 0.25 × ATR(14) of a Low Volume Node (LVN)
- At least one active bullish gap

**Interpretation:**
This signal indicates a potential bounce from a low-volume support level where price has created an imbalance (FVG) that may act as support.

### Continuation Setup (vf_short)

**Conditions:**
- Active bearish FVG signal present
- FVG overlaps with High Volume Node (HVN) area
- POC shift > 0.5 × ATR(14) indicating strong momentum
- At least one active bearish gap

**Interpretation:**
This signal indicates strong downward momentum where price is breaking through a high-volume resistance area with significant POC displacement.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atr_period` | 14 | ATR calculation period for distance thresholds |
| `lvn_dist_multiplier` | 0.25 | ATR multiplier for LVN proximity (0.25 × ATR) |
| `poc_shift_multiplier` | 0.5 | ATR multiplier for POC shift threshold (0.5 × ATR) |
| `hvn_overlap_pct` | 0.7 | Minimum overlap percentage for HVN confluence |
| `min_fvg_size` | 1.0 | Minimum FVG size to consider valid |
| `lookback_validation` | 5 | Bars to look back for validation |

## Output Columns

| Column | Type | Description |
|--------|------|-------------|
| `vf_long` | bool | Bullish reversal signal |
| `vf_short` | bool | Bearish continuation signal |
| `vf_atr` | float | ATR values used for distance calculations |
| `vf_poc_shift` | float | POC shift magnitude |
| `vf_lvn_distance` | float | Distance from FVG mid to nearest LVN |
| `vf_hvn_overlap` | float | HVN overlap percentage |

## Dependencies

The VPFVGSignal indicator requires the following input columns from other indicators:

### Volume Profile columns:
- `VolumeProfile_poc_price`: Point of Control price
- `VolumeProfile_is_lvn`: Low Volume Node indicator
- `VolumeProfile_is_hvn`: High Volume Node indicator

### FVG columns:
- `FVG_bullish_signal`: Bullish FVG signal
- `FVG_bearish_signal`: Bearish FVG signal
- `FVG_nearest_support_mid`: Midpoint of nearest support FVG
- `FVG_nearest_resistance_mid`: Midpoint of nearest resistance FVG
- `FVG_active_bullish_gaps`: Count of active bullish gaps
- `FVG_active_bearish_gaps`: Count of active bearish gaps

## Usage Example

### Python Code

```python
from btc_research.indicators import VPFVGSignal, VolumeProfile, FVG

# Create indicators
vp_indicator = VolumeProfile(lookback=200, price_bins=30)
fvg_indicator = FVG(min_gap_pips=5.0)
vpfvg_indicator = VPFVGSignal(
    atr_period=14,
    lvn_dist_multiplier=0.25,
    poc_shift_multiplier=0.5
)

# Compute indicators in sequence
vp_result = vp_indicator.compute(ohlcv_data)
fvg_result = fvg_indicator.compute(ohlcv_data)

# Combine data
combined_data = ohlcv_data.copy()
for col in vp_result.columns:
    combined_data[f'VolumeProfile_{col}'] = vp_result[col]
for col in fvg_result.columns:
    combined_data[col] = fvg_result[col]

# Generate confluence signals
signals = vpfvg_indicator.compute(combined_data)
```

### YAML Configuration

```yaml
indicators:
  - id: "VolumeProfile"
    type: "VolumeProfile"
    timeframe: "15m"
    lookback: 200
    price_bins: 30
    hvn_threshold: 1.5
    lvn_threshold: 0.5
    
  - id: "FVG"
    type: "FVG"
    timeframe: "15m"
    min_gap_pips: 5.0
    max_lookback: 500
    
  - id: "VPFVGSignal"
    type: "VPFVGSignal"
    timeframe: "15m"
    atr_period: 14
    lvn_dist_multiplier: 0.25
    poc_shift_multiplier: 0.5
    hvn_overlap_pct: 0.7

strategies:
  - id: "VP_FVG_Reversal"
    type: "simple"
    entry_conditions:
      - "VPFVGSignal_vf_long == True"
    exit_conditions:
      - "VPFVGSignal_vf_long == False"
      
  - id: "VP_FVG_Continuation"
    type: "simple"
    entry_conditions:
      - "VPFVGSignal_vf_short == True"
    exit_conditions:
      - "VPFVGSignal_vf_short == False"
```

## Signal Interpretation

### Reversal Signals (vf_long)

**When to Act:**
- `vf_long == True`: Consider long entry
- Look for additional confluence factors
- Set stop loss below the LVN level
- Take profit at next resistance level

**Risk Management:**
- Stop loss: Below LVN - 1 × ATR
- Take profit: Next HVN or resistance level
- Position size: Based on ATR for volatility adjustment

### Continuation Signals (vf_short)

**When to Act:**
- `vf_short == True`: Consider short entry
- Confirm with volume spike
- Set stop loss above the HVN level
- Take profit at next support level

**Risk Management:**
- Stop loss: Above HVN + 1 × ATR
- Take profit: Next LVN or support level
- Position size: Based on ATR for volatility adjustment

## Signal Frequency

The VPFVGSignal indicator is designed to be highly selective, typically generating signals on less than 1-2% of bars. This low frequency is intentional and indicates:

- **High-probability setups**: Only triggers when multiple confluence factors align
- **Quality over quantity**: Focuses on the best trading opportunities
- **Risk management**: Reduces false signals and overtrading

## Performance Considerations

### Computational Efficiency

- **ATR calculation**: O(n) time complexity
- **Signal detection**: O(n × lookback) for each bar
- **Memory usage**: Minimal additional memory beyond input data

### Optimization Tips

1. **Adjust lookback_validation**: Smaller values = faster computation
2. **Pre-filter data**: Remove periods with low volatility
3. **Batch processing**: Process multiple timeframes together
4. **Parameter tuning**: Optimize thresholds for your specific market

## Testing and Validation

The indicator includes comprehensive test suites:

- **Unit tests**: Core functionality and edge cases
- **Integration tests**: Full pipeline with other indicators
- **Performance tests**: Speed and memory benchmarks
- **Validation tests**: Look-ahead bias prevention

Run tests with:
```bash
pytest tests/test_vpfvg_signal.py -v
pytest tests/test_vpfvg_integration.py -v
```

## Common Issues and Solutions

### No Signals Generated

**Possible causes:**
- Missing required input columns
- Thresholds too strict
- Insufficient data for calculation
- No VP/FVG signals in the data

**Solutions:**
- Verify all dependencies are computed
- Reduce threshold multipliers
- Increase data sample size
- Check VP/FVG parameters

### Too Many Signals

**Possible causes:**
- Thresholds too loose
- Noisy data
- Incorrect parameter settings

**Solutions:**
- Increase threshold multipliers
- Apply additional filters
- Tune parameters for your market

### Performance Issues

**Possible causes:**
- Large datasets
- Frequent recalculation
- Memory constraints

**Solutions:**
- Increase update_frequency
- Use data sampling
- Process in batches

## Advanced Usage

### Custom Threshold Calculation

```python
# Dynamic threshold based on market conditions
def calculate_dynamic_threshold(data, base_multiplier=0.25):
    volatility = data['close'].pct_change().std()
    return base_multiplier * (1 + volatility * 2)

# Apply custom threshold
indicator = VPFVGSignal(
    lvn_dist_multiplier=calculate_dynamic_threshold(data)
)
```

### Signal Filtering

```python
# Filter signals based on additional criteria
def filter_signals(signals, data):
    # Only take signals during high volume periods
    high_volume = data['volume'] > data['volume'].rolling(20).mean()
    
    # Only take signals with strong trend
    strong_trend = abs(data['close'].pct_change(20)) > 0.05
    
    filtered_long = signals['vf_long'] & high_volume & strong_trend
    filtered_short = signals['vf_short'] & high_volume & strong_trend
    
    return filtered_long, filtered_short
```

## Related Indicators

- **VolumeProfile**: Provides HVN/LVN classification
- **FVG**: Identifies imbalance areas
- **ATR**: Used for adaptive thresholds
- **EMA**: Can be used for trend filtering

## References

- Volume Profile analysis methodology
- Fair Value Gap (ICT) concepts
- ATR-based risk management
- Confluence trading strategies

## Version History

- **v1.0**: Initial implementation with core confluence logic
- **v1.1**: Added comprehensive testing and documentation
- **v1.2**: Performance optimizations and parameter tuning