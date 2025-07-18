# Volume Profile Implementation Documentation

## Overview

This document outlines the implementation plan for Volume Profile indicators in the BTC Research Engine. The implementation follows a two-phase approach: starting with rolling Volume Profile for immediate testing, then extending to a pre-calculation engine for institutional-grade analysis.

## Architecture Overview

### Phase 1: Rolling Volume Profile (Current Implementation)
- Rolling window approach using existing BaseIndicator framework
- Real-time adaptive calculations suitable for live trading
- 24-48 hour lookback periods for short-term support/resistance

### Phase 2: Pre-Calculation Engine (Future Enhancement)
- Separate calculation engine for historical Volume Profile analysis
- Multiple time periods: 1D, 1W, 1M, 3M, 1Y, All-Time
- Cache-based storage for performance optimization

## Phase 1: Rolling Volume Profile Implementation

### 1. Indicator Structure

**File**: `btc_research/indicators/volume_profile.py`

**Class**: `VolumeProfile(BaseIndicator)`

**Registration**: `@register("VolumeProfile")`

### 2. Parameters

```python
@classmethod
def params(cls) -> dict[str, Any]:
    return {
        "lookback": 96,             # Number of candles (24h for 15m timeframe)
        "price_bins": 50,           # Number of price levels to create
        "value_area_pct": 70,       # Percentage for Value Area calculation
        "update_frequency": 1,       # Recalculate every N candles
        "poc_sensitivity": 0.1,      # POC change threshold (percentage)
        "min_volume_threshold": 0.01 # Minimum volume per bin (percentage of total)
    }
```

### 3. Algorithm Design

#### Core Calculation Steps:
1. **Rolling Window**: Extract last N candles based on lookback parameter
2. **Price Range**: Calculate min/max from High/Low over the window
3. **Price Binning**: Divide price range into equal-sized bins
4. **Volume Distribution**: Distribute each candle's volume across bins using OHLC
5. **POC Calculation**: Find price bin with highest total volume
6. **Value Area**: Calculate price range containing 70% of total volume around POC

#### Volume Distribution Method:
```python
def distribute_volume(self, candle_data, price_bins):
    """
    Distribute candle volume across price bins based on OHLC.
    Uses triangular distribution weighted toward close price.
    """
    # Simplified approach: distribute volume evenly across OHLC range
    # Advanced: weight distribution toward close price
```

### 4. Output Columns

Generated columns with indicator ID prefix (`VP_{id}_`):

**Core Metrics:**
- `poc_price` - Point of Control price level
- `vah_price` - Value Area High price  
- `val_price` - Value Area Low price
- `total_volume` - Total volume in lookback period
- `poc_volume` - Volume at POC level
- `value_area_volume` - Total volume in Value Area

**Trading Signals:**
- `price_above_poc` - Boolean: current price > POC
- `price_below_poc` - Boolean: current price < POC  
- `price_in_value_area` - Boolean: price within Value Area
- `poc_breakout` - Boolean: significant POC level change
- `volume_spike` - Boolean: current volume > average volume

**Additional Analytics:**
- `poc_strength` - Relative volume concentration at POC
- `value_area_width` - Price width of Value Area
- `profile_balance` - Symmetry of volume distribution

### 5. Performance Considerations

#### Optimization Strategies:
- **Vectorized Operations**: Use numpy for bulk calculations
- **Incremental Updates**: Only recalculate when update_frequency threshold met
- **Memory Management**: Limit stored historical data to prevent memory bloat
- **Cache Binning**: Reuse price bins when price range hasn't changed significantly

#### Performance Targets:
- Calculation time: <50ms per update for 96-candle lookback
- Memory usage: <10MB per indicator instance
- Compatible with existing <200ms cache load requirement

### 6. Testing Strategy

#### Unit Tests:
- Test volume distribution across price bins
- Validate POC and Value Area calculations
- Test edge cases (insufficient data, zero volume)
- Performance benchmarks

#### Integration Tests:
- End-to-end backtesting with simple strategy
- Multi-timeframe compatibility
- YAML configuration validation

#### Test Data:
- Use existing sample data from fixtures
- Generate synthetic data for edge case testing
- Historical BTC data for realistic scenarios

## Phase 1 Implementation Plan

### Step 1: Core Indicator Implementation
1. Create `btc_research/indicators/volume_profile.py`
2. Implement `VolumeProfile` class with BaseIndicator interface
3. Add core calculation methods (binning, POC, Value Area)
4. Register indicator with `@register("VolumeProfile")`

### Step 2: Test Configuration
1. Create `btc_research/config/simple-volume-profile.yaml`
2. Configure basic VP strategy for backtesting
3. Use 15m timeframe with 24-hour lookback

### Step 3: Unit Testing
1. Create `tests/test_volume_profile.py`
2. Test core calculation methods
3. Validate output column generation
4. Performance benchmarking

### Step 4: Integration Testing  
1. Test with Engine integration
2. Run backtest with simple strategy
3. Validate signal generation
4. Performance validation

### Step 5: Documentation & Examples
1. Update indicator documentation
2. Create usage examples
3. Performance optimization notes

## Test Configuration

### Simple Volume Profile Strategy

**File**: `btc_research/config/simple-volume-profile.yaml`

```yaml
version: "1.0"
name: "Simple Volume Profile Strategy"
symbol: "BTC/USDC"
exchange: "binanceus"

timeframes:
  entry: "15m"

indicators:
  - id: "VP_24h"
    type: "VolumeProfile"
    timeframe: "15m"
    lookback: 96          # 24 hours
    price_bins: 40        # 40 price levels
    value_area_pct: 70    # Standard 70%
    update_frequency: 4   # Update every hour

logic:
  entry_long:
    - "close > VP_24h_poc_price"
    - "VP_24h_price_in_value_area == False"
    - "volume > VP_24h_total_volume * 1.5"

  exit_long:
    - "close <= VP_24h_poc_price"
    - "VP_24h_price_in_value_area == True"

  entry_short:
    - "close < VP_24h_poc_price"  
    - "VP_24h_price_in_value_area == False"
    - "volume > VP_24h_total_volume * 1.5"

  exit_short:
    - "close >= VP_24h_poc_price"
    - "VP_24h_price_in_value_area == True"

backtest:
  cash: 10000
  commission: 0.0004
  from: "2024-01-01"
  to: "2024-02-01"
```

## Success Criteria for Phase 1

### Functional Requirements:
- [ ] VolumeProfile indicator correctly implements BaseIndicator interface
- [ ] Generates all specified output columns
- [ ] POC and Value Area calculations are mathematically correct
- [ ] Integration with Engine works without errors
- [ ] Backtest runs successfully with test configuration

### Performance Requirements:
- [ ] Indicator calculation completes in <50ms per update
- [ ] Memory usage stays within reasonable bounds (<10MB per instance)
- [ ] Compatible with existing caching infrastructure

### Quality Requirements:
- [ ] Unit test coverage >90%
- [ ] Integration tests pass
- [ ] No breaking changes to existing functionality
- [ ] Code follows project style guidelines

## Phase 2: Pre-Calculation Engine (Future)

### Architecture Overview:
```
btc_research/
├── core/
│   ├── engine.py           # Current engine
│   └── precalc_engine.py   # New pre-calculation engine
├── indicators/
│   ├── volume_profile.py   # Phase 1 rolling VP
│   └── precalc_volume_profile.py  # Phase 2 pre-calculated VP
└── data/
    ├── ohlcv/             # Raw OHLCV cache  
    └── precalc/           # Pre-calculated indicator cache
        └── volume_profile/
            ├── daily/
            ├── weekly/
            ├── monthly/
            └── all_time/
```

### Future Enhancements:
- Multiple VP periods (1D, 1W, 1M, 3M, 1Y, All-Time)
- Incremental update mechanism
- Institutional-grade support/resistance levels
- Advanced volume distribution models
- Performance optimization with pre-calculation

## Notes

### Dependencies:
- numpy (already available) - for vectorized calculations
- pandas (already available) - for DataFrame operations
- Existing BaseIndicator framework
- Current DataFeed and caching infrastructure

### Compatibility:
- Works with existing CCXT OHLCV data
- Compatible with current backtesting framework
- No external API dependencies required
- Maintains <200ms performance requirement

### Limitations:
- Volume distribution limited by OHLC granularity (not tick-level)
- Rolling window provides limited historical context
- No buy/sell volume classification (would need tick data)

---

**Implementation Status**: Phase 1 - Ready for Implementation
**Next Steps**: Begin Step 1 - Core Indicator Implementation