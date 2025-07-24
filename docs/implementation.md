# Paper Trading System - Technical Implementation Plan

## Architecture Decisions

### Core Framework Extensions

#### Engine Modifications (`btc_research/core/engine.py`)
```python
class Engine:
    def __init__(self, live_mode=False, stream_manager=None):
        self.live_mode = live_mode
        self.stream_manager = stream_manager
        # Existing backtesting logic preserved
```

**Key Decision**: Extend existing Engine class rather than create new one
- **Rationale**: Preserves all indicator calculations and multi-timeframe logic
- **Caveat**: Must maintain backward compatibility for backtesting
- **Implementation**: Use composition pattern for live-specific functionality

#### Data Flow Architecture
```
WebSocket Data → StreamManager → Engine → Indicators → StrategyRunner → PaperTrader
                                    ↓
                               Redis Buffer ← API Server ← Flask Dashboard
```

### New Component Design

#### 1. StreamManager (`btc_research/live/stream_manager.py`)

**Technical Decisions**:
- **WebSocket Library**: `websocket-client` (simple, reliable)
- **Data Buffering**: Redis for cross-process access
- **Reconnection Strategy**: Exponential backoff with circuit breaker
- **Data Alignment**: Maintain existing forward-fill logic

**Critical Implementation Details**:
```python
class StreamManager:
    def __init__(self, symbols, timeframes):
        self.buffers = {tf: deque(maxlen=1000) for tf in timeframes}
        self.ws_connections = {}
        self.redis_client = redis.Redis()
    
    async def process_tick(self, symbol, price, timestamp):
        # Update all timeframe buffers
        # Trigger indicator recalculation
        # Maintain DataFrame compatibility with Engine
```

**Performance Requirements**:
- Buffer 1000 candles per timeframe (memory: ~50MB)
- Process ticks within 10ms
- Maintain data alignment across timeframes

#### 2. PaperTrader (`btc_research/live/paper_trader.py`)

**Technical Decisions**:
- **Order Execution**: Market orders with realistic slippage (0.05%)
- **Commission Model**: 0.1% per trade (Binance standard)
- **Position Tracking**: Net position with average entry price
- **Balance Management**: Separate available/total balance tracking

**Critical Caveats**:
- **Fill Prices**: Use bid/ask spread simulation for realistic fills
- **Partial Fills**: Support for large orders split across multiple prices
- **Latency Simulation**: 50-100ms delay for order acknowledgment

```python
class PaperTrader:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.positions = {}  # symbol -> Position
        self.orders = {}     # order_id -> Order
        self.trades = []     # Historical trade records
    
    def submit_order(self, symbol, side, size, order_type='market'):
        # Realistic order simulation with commissions
        # Position size validation
        # Balance checks
```

#### 3. StrategyRunner (`btc_research/live/strategy_runner.py`)

**Technical Decisions**:
- **Execution Model**: Event-driven with async/await
- **Configuration**: Direct YAML loading (no changes to existing configs)
- **Error Handling**: Strategy-level isolation (one failure doesn't stop others)
- **State Management**: Persistent strategy state across restarts

**Implementation Strategy**:
```python
class StrategyRunner:
    def __init__(self, config_path, paper_trader, stream_manager):
        self.config = self.load_config(config_path)
        self.engine = Engine(live_mode=True, stream_manager=stream_manager)
        self.strategy_state = {}
    
    async def on_data_update(self, symbol, timeframe):
        # Recalculate indicators
        # Check strategy conditions
        # Submit orders if signals present
```

### API Design (`btc_research/api/`)

#### Framework Choice: FastAPI
**Rationale**: 
- Automatic OpenAPI documentation
- Built-in async support
- Type hints integration
- Performance advantages over Flask

#### Endpoint Specification
```python
# Strategy Management
POST   /api/v1/strategies/start          # Start strategy from config
POST   /api/v1/strategies/{id}/stop      # Stop running strategy
GET    /api/v1/strategies/active         # List active strategies
GET    /api/v1/strategies/{id}/status    # Strategy health/performance

# Statistics & Data
GET    /api/v1/strategies/{id}/stats     # Real-time performance metrics
GET    /api/v1/strategies/{id}/trades    # Trade history with pagination
GET    /api/v1/strategies/{id}/positions # Current positions
GET    /api/v1/market/data/{symbol}      # Current market data

# Configuration
GET    /api/v1/configs/available         # List available strategy configs
POST   /api/v1/configs/validate          # Validate strategy configuration
```

**Authentication**: JWT tokens for production, API keys for development
**Rate Limiting**: 100 requests/minute per client
**CORS**: Enabled for local development

### Database Design

#### Development: JSON File Storage
```
data/
├── strategies/
│   ├── {strategy_id}/
│   │   ├── config.yaml
│   │   ├── state.json
│   │   ├── trades.jsonl
│   │   └── positions.json
└── market_data/
    ├── realtime_buffer.json
    └── historical_cache/
```

#### Production: PostgreSQL Schema
```sql
-- Strategy management
CREATE TABLE strategies (
    id UUID PRIMARY KEY,
    config_path TEXT NOT NULL,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Trade history
CREATE TABLE trades (
    id UUID PRIMARY KEY,
    strategy_id UUID REFERENCES strategies(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL,
    size DECIMAL(18,8) NOT NULL,
    price DECIMAL(18,8) NOT NULL,
    commission DECIMAL(18,8) NOT NULL,
    timestamp TIMESTAMP NOT NULL
);

-- Position tracking
CREATE TABLE positions (
    strategy_id UUID REFERENCES strategies(id),
    symbol VARCHAR(20) NOT NULL,
    size DECIMAL(18,8) NOT NULL,
    average_price DECIMAL(18,8) NOT NULL,
    unrealized_pnl DECIMAL(18,8) NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (strategy_id, symbol)
);
```

### Docker Architecture

#### Service Composition
```yaml
# docker-compose.yml
services:
  paper-trader:
    build: .
    command: python -m btc_research.live.main
    volumes:
      - ./btc_research/config:/app/config:ro
      - ./data:/app/data
    depends_on: [redis]
    
  api-server:
    build: .
    command: uvicorn btc_research.api.main:app --host 0.0.0.0
    ports: ["8000:8000"]
    depends_on: [paper-trader, redis]
    
  dashboard:
    build: .
    command: python -m btc_research.web.app
    ports: ["5000:5000"]
    depends_on: [api-server]
    
  redis:
    image: redis:alpine
    ports: ["6379:6379"]
    
  # Optional for production
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: paper_trading
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

#### Container Requirements
- **paper-trader**: 2 CPU cores, 4GB RAM
- **api-server**: 1 CPU core, 2GB RAM  
- **dashboard**: 1 CPU core, 1GB RAM
- **redis**: 1 CPU core, 512MB RAM

### Frontend Architecture (`btc_research/web/`)

#### Framework Choice: Flask + Jinja2
**Rationale**:
- Simplicity over React/Vue complexity
- Server-side rendering reduces client-side dependencies
- Easy integration with existing Python ecosystem
- Minimal JavaScript requirements

#### Dashboard Components
```
templates/
├── base.html           # Common layout
├── dashboard.html      # Main strategy overview
├── strategy.html       # Individual strategy details  
├── trades.html         # Trade history table
└── charts.html         # Basic price/PnL charts

static/
├── css/
│   └── dashboard.css   # Simple styling
├── js/
│   ├── charts.js       # Chart.js integration
│   └── api.js          # API communication
└── assets/
    └── favicon.ico
```

#### Real-time Updates
- **WebSocket connection** for live data updates
- **Server-Sent Events** as fallback
- **5-second polling** as final fallback

### Critical Implementation Caveats  

#### 1. Data Consistency
**Issue**: Real-time calculations must match backtest results exactly
**Solution**: Validation test suite comparing live vs historical calculations
**Implementation**: 
```python
def validate_indicator_consistency(historical_data, live_data):
    # Compare indicator values within 0.01% tolerance
    # Log discrepancies for debugging
    # Alert if deviation exceeds threshold
```

#### 2. Memory Management
**Issue**: Continuous operation leads to memory accumulation  
**Solution**: Circular buffers with configurable limits
**Implementation**: 
- 1000 candles per timeframe (max 50MB)
- Periodic garbage collection
- Memory usage monitoring

#### 3. Error Recovery
**Issue**: WebSocket disconnections or exchange API failures
**Solution**: Multi-level fallback mechanisms
**Implementation**:
```python
class DataFeedManager:
    async def get_data(self, symbol, timeframe):
        try:
            return await self.websocket_feed.get_data(symbol, timeframe)
        except ConnectionError:
            return await self.rest_fallback.get_data(symbol, timeframe)
        except Exception:
            return self.cached_data.get_latest(symbol, timeframe)
```

#### 4. Strategy State Persistence
**Issue**: System restarts must not lose strategy state
**Solution**: Periodic state snapshots to disk
**Implementation**:
- JSON state files updated every 10 seconds
- Recovery logic on startup
- Checksum validation for corruption detection

#### 5. Performance Bottlenecks
**Critical Points**:
- **Indicator calculations**: Must complete within 200ms
- **WebSocket processing**: Queue management under high-frequency updates
- **Database writes**: Async operations to prevent blocking
- **API responses**: Caching for frequently requested data

### Testing Strategy

#### Unit Tests
- **StreamManager**: Mock WebSocket data feeds
- **PaperTrader**: Verify order execution logic
- **StrategyRunner**: Test strategy signal generation
- **API endpoints**: Request/response validation

#### Integration Tests  
- **End-to-end**: Config → Strategy → Trades → API → Dashboard
- **Data consistency**: Live vs backtest result comparison
- **Error scenarios**: Network failures, invalid data, system restarts

#### Performance Tests
- **Load testing**: Multiple concurrent strategies
- **Memory testing**: 24+ hour continuous operation
- **Latency testing**: Data processing within SLA requirements

### Deployment Strategy

#### Development Environment
```bash
# Single command setup
docker-compose -f docker-compose.dev.yml up

# Includes:
# - Hot reload for code changes
# - Debug logging enabled
# - Mock data generators for testing
```

#### Production Environment  
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Includes:
# - Resource limits and health checks
# - Log aggregation and monitoring
# - Backup and recovery procedures
```

### Migration Path

#### Phase 1: Core Implementation (Week 1-2)
- Extend Engine class for live mode
- Implement StreamManager with WebSocket connections
- Build PaperTrader with order simulation
- Create StrategyRunner for real-time execution

#### Phase 2: API Development (Week 3)
- FastAPI server with strategy management endpoints
- Real-time statistics and trade history APIs
- Basic authentication and rate limiting

#### Phase 3: Frontend & Deployment (Week 4)
- Flask dashboard with strategy monitoring
- Docker Compose setup with all services
- Basic charts and trade visualization

#### Phase 4: Testing & Validation (Week 5)
- Comprehensive test suite
- Performance validation
- Data consistency verification
- Documentation and deployment guides