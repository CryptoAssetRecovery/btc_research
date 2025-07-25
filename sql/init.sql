-- Database initialization script for BTC Research Paper Trading System
-- This script sets up the PostgreSQL database schema

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create enum types
CREATE TYPE strategy_status AS ENUM ('starting', 'running', 'stopping', 'stopped', 'error');
CREATE TYPE order_side AS ENUM ('buy', 'sell');
CREATE TYPE order_status AS ENUM ('pending', 'filled', 'cancelled', 'rejected');

-- Create strategies table
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

-- Create indexes for strategies
CREATE INDEX IF NOT EXISTS idx_strategies_user_id ON strategies(user_id);
CREATE INDEX IF NOT EXISTS idx_strategies_status ON strategies(status);
CREATE INDEX IF NOT EXISTS idx_strategies_created_at ON strategies(created_at);

-- Create trades table
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
);

-- Create indexes for trades
CREATE INDEX IF NOT EXISTS idx_trades_strategy_id ON trades(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(side);

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

-- Create market_data table for caching
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
);

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

-- Grant permissions for the trader user
GRANT ALL PRIVILEGES ON DATABASE paper_trading TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trader;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO trader;

-- Create some sample data for development/testing
INSERT INTO strategies (user_id, name, config_path, status, initial_balance, current_balance)
VALUES 
    ('dev-user', 'Sample Volume Profile Strategy', 'config/volume-profile-breakout.yaml', 'stopped', 10000.0, 10000.0)
ON CONFLICT DO NOTHING;