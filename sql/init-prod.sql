-- Production Database initialization script for BTC Research Paper Trading System
-- This script sets up the PostgreSQL database schema with production optimizations

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Set production-specific configurations
ALTER DATABASE paper_trading SET log_statement = 'mod';
ALTER DATABASE paper_trading SET log_min_duration_statement = 1000;
ALTER DATABASE paper_trading SET shared_preload_libraries = 'pg_stat_statements';

-- Create enum types
CREATE TYPE strategy_status AS ENUM ('starting', 'running', 'stopping', 'stopped', 'error');
CREATE TYPE order_side AS ENUM ('buy', 'sell');
CREATE TYPE order_status AS ENUM ('pending', 'filled', 'cancelled', 'rejected');

-- Create strategies table with production optimizations
CREATE TABLE IF NOT EXISTS strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    config_path TEXT NOT NULL,
    status strategy_status NOT NULL DEFAULT 'stopped',
    initial_balance DECIMAL(18,8) NOT NULL DEFAULT 10000.0,
    current_balance DECIMAL(18,8) NOT NULL DEFAULT 10000.0,
    total_pnl DECIMAL(18,8) NOT NULL DEFAULT 0.0,
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    stopped_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for strategies with production optimizations
CREATE INDEX IF NOT EXISTS idx_strategies_user_id ON strategies(user_id);
CREATE INDEX IF NOT EXISTS idx_strategies_status ON strategies(status);
CREATE INDEX IF NOT EXISTS idx_strategies_created_at ON strategies(created_at);
CREATE INDEX IF NOT EXISTS idx_strategies_user_status ON strategies(user_id, status);

-- Create trades table with partitioning for large datasets
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    side order_side NOT NULL,
    quantity DECIMAL(18,8) NOT NULL,
    price DECIMAL(18,8) NOT NULL,
    commission DECIMAL(18,8) NOT NULL DEFAULT 0.0,
    pnl DECIMAL(18,8) DEFAULT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (timestamp);

-- Create trade partitions for current and next month
CREATE TABLE IF NOT EXISTS trades_current PARTITION OF trades 
FOR VALUES FROM (DATE_TRUNC('month', CURRENT_DATE)) 
TO (DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month'));

CREATE TABLE IF NOT EXISTS trades_next PARTITION OF trades 
FOR VALUES FROM (DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month')) 
TO (DATE_TRUNC('month', CURRENT_DATE + INTERVAL '2 months'));

-- Create indexes for trades
CREATE INDEX IF NOT EXISTS idx_trades_strategy_id ON trades(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(side);
CREATE INDEX IF NOT EXISTS idx_trades_strategy_timestamp ON trades(strategy_id, timestamp);

-- Create orders table
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    side order_side NOT NULL,
    quantity DECIMAL(18,8) NOT NULL,
    price DECIMAL(18,8),
    filled_quantity DECIMAL(18,8) NOT NULL DEFAULT 0.0,
    status order_status NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    filled_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for orders
CREATE INDEX IF NOT EXISTS idx_orders_strategy_id ON orders(strategy_id);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
CREATE INDEX IF NOT EXISTS idx_orders_strategy_status ON orders(strategy_id, status);

-- Create positions table
CREATE TABLE IF NOT EXISTS positions (
    strategy_id UUID NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(18,8) NOT NULL DEFAULT 0.0,
    average_price DECIMAL(18,8) NOT NULL DEFAULT 0.0,
    unrealized_pnl DECIMAL(18,8) NOT NULL DEFAULT 0.0,
    realized_pnl DECIMAL(18,8) NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (strategy_id, symbol)
);

-- Create indexes for positions
CREATE INDEX IF NOT EXISTS idx_positions_strategy_id ON positions(strategy_id);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);

-- Create market_data table with partitioning for time-series data
CREATE TABLE IF NOT EXISTS market_data (
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open_price DECIMAL(18,8) NOT NULL,
    high_price DECIMAL(18,8) NOT NULL,
    low_price DECIMAL(18,8) NOT NULL,
    close_price DECIMAL(18,8) NOT NULL,
    volume DECIMAL(18,8) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create market data partitions
CREATE TABLE IF NOT EXISTS market_data_current PARTITION OF market_data 
FOR VALUES FROM (DATE_TRUNC('month', CURRENT_DATE)) 
TO (DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month'));

CREATE TABLE IF NOT EXISTS market_data_next PARTITION OF market_data 
FOR VALUES FROM (DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month')) 
TO (DATE_TRUNC('month', CURRENT_DATE + INTERVAL '2 months'));

-- Create indexes for market_data
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe ON market_data(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);

-- Create performance_stats table for caching calculated metrics
CREATE TABLE IF NOT EXISTS performance_stats (
    strategy_id UUID NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
    calculated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    total_return DECIMAL(10,4),
    annualized_return DECIMAL(10,4),
    volatility DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    sortino_ratio DECIMAL(10,4),
    calmar_ratio DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    win_rate DECIMAL(10,4),
    profit_factor DECIMAL(10,4),
    expectancy DECIMAL(10,4),
    PRIMARY KEY (strategy_id, calculated_at)
);

-- Create indexes for performance_stats
CREATE INDEX IF NOT EXISTS idx_performance_stats_strategy_id ON performance_stats(strategy_id);
CREATE INDEX IF NOT EXISTS idx_performance_stats_calculated_at ON performance_stats(calculated_at);

-- Create audit log table for security and compliance
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for audit_log
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_strategies_updated_at BEFORE UPDATE ON strategies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function to calculate strategy performance
CREATE OR REPLACE FUNCTION calculate_strategy_performance(strategy_uuid UUID)
RETURNS TABLE (
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    total_pnl DECIMAL(18,8),
    win_rate DECIMAL(10,4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::INTEGER as total_trades,
        COUNT(CASE WHEN t.pnl > 0 THEN 1 END)::INTEGER as winning_trades,
        COUNT(CASE WHEN t.pnl < 0 THEN 1 END)::INTEGER as losing_trades,
        COALESCE(SUM(t.pnl), 0) as total_pnl,
        CASE 
            WHEN COUNT(*) > 0 THEN 
                ROUND((COUNT(CASE WHEN t.pnl > 0 THEN 1 END)::DECIMAL / COUNT(*)::DECIMAL) * 100, 2)
            ELSE 0 
        END as win_rate
    FROM trades t
    WHERE t.strategy_id = strategy_uuid
    AND t.pnl IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- Create view for strategy overview
CREATE OR REPLACE VIEW strategy_overview AS
SELECT 
    s.id,
    s.user_id,
    s.name,
    s.status,
    s.initial_balance,
    s.current_balance,
    s.total_pnl,
    s.created_at,
    s.started_at,
    s.stopped_at,
    COALESCE(perf.total_trades, 0) as total_trades,
    COALESCE(perf.winning_trades, 0) as winning_trades,
    COALESCE(perf.losing_trades, 0) as losing_trades,
    COALESCE(perf.win_rate, 0) as win_rate
FROM strategies s
LEFT JOIN LATERAL calculate_strategy_performance(s.id) as perf ON true;

-- Create partition maintenance function
CREATE OR REPLACE FUNCTION maintain_partitions()
RETURNS void AS $$
DECLARE
    next_month_start DATE;
    next_month_end DATE;
    partition_name_trades TEXT;
    partition_name_market_data TEXT;
BEGIN
    -- Calculate next month dates
    next_month_start := DATE_TRUNC('month', CURRENT_DATE + INTERVAL '2 months');
    next_month_end := DATE_TRUNC('month', CURRENT_DATE + INTERVAL '3 months');
    
    -- Create partition names
    partition_name_trades := 'trades_' || TO_CHAR(next_month_start, 'YYYY_MM');
    partition_name_market_data := 'market_data_' || TO_CHAR(next_month_start, 'YYYY_MM');
    
    -- Create new partitions if they don't exist
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF trades FOR VALUES FROM (%L) TO (%L)',
                   partition_name_trades, next_month_start, next_month_end);
    
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF market_data FOR VALUES FROM (%L) TO (%L)',
                   partition_name_market_data, next_month_start, next_month_end);
END;
$$ LANGUAGE plpgsql;

-- Create database maintenance procedures
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Clean up old performance stats (keep last 30 days)
    DELETE FROM performance_stats 
    WHERE calculated_at < NOW() - INTERVAL '30 days';
    
    -- Clean up old audit logs (keep last 90 days)  
    DELETE FROM audit_log 
    WHERE timestamp < NOW() - INTERVAL '90 days';
    
    -- Update statistics
    ANALYZE;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions for the trader user
GRANT ALL PRIVILEGES ON DATABASE paper_trading TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trader;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO trader;

-- Set up row-level security for multi-tenant isolation
ALTER TABLE strategies ENABLE ROW LEVEL SECURITY;
CREATE POLICY strategies_user_isolation ON strategies FOR ALL TO trader USING (user_id = current_setting('app.current_user', true));

ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
CREATE POLICY trades_user_isolation ON trades FOR ALL TO trader USING (
    strategy_id IN (SELECT id FROM strategies WHERE user_id = current_setting('app.current_user', true))
);

-- Production settings
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Create maintenance schedule comment
COMMENT ON FUNCTION maintain_partitions() IS 'Run monthly to create new partitions';
COMMENT ON FUNCTION cleanup_old_data() IS 'Run daily to clean up old data';