#!/bin/bash
# Docker entrypoint script for BTC Research Paper Trading System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    log "Waiting for $service_name at $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" >/dev/null 2>&1; then
            log_success "$service_name is ready!"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: $service_name not ready, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "$service_name is not available after $max_attempts attempts"
    return 1
}

# Function to check Redis connectivity
check_redis() {
    local redis_url=${REDIS_URL:-"redis://localhost:6379/0"}
    log "Checking Redis connectivity: $redis_url"
    
    python3 -c "
import redis
import sys
from urllib.parse import urlparse

try:
    parsed = urlparse('$redis_url')
    r = redis.Redis(
        host=parsed.hostname or 'localhost',
        port=parsed.port or 6379,
        db=int(parsed.path.lstrip('/')) if parsed.path else 0,
        password=parsed.password,
        socket_timeout=5
    )
    r.ping()
    print('Redis connection successful')
    sys.exit(0)
except Exception as e:
    print(f'Redis connection failed: {e}')
    sys.exit(1)
"
    return $?
}

# Function to initialize application data directories
init_directories() {
    log "Initializing application directories..."
    
    mkdir -p /app/data/strategies
    mkdir -p /app/data/market_data
    mkdir -p /app/logs
    mkdir -p /app/config
    
    # Set proper permissions
    chmod 755 /app/data
    chmod 755 /app/logs
    chmod 755 /app/config
    
    log_success "Directories initialized"
}

# Function to check configuration files
check_config() {
    log "Checking configuration files..."
    
    local config_files=("volume-profile-breakout.yaml")
    local missing_configs=()
    
    for config in "${config_files[@]}"; do
        if [ ! -f "/app/config/$config" ]; then
            missing_configs+=("$config")
        fi
    done
    
    if [ ${#missing_configs[@]} -gt 0 ]; then
        log_warning "Missing configuration files: ${missing_configs[*]}"
        log_warning "Creating default configurations..."
        
        # Create minimal default config if none exists
        if [ ! -f "/app/config/volume-profile-breakout.yaml" ]; then
            cat > "/app/config/volume-profile-breakout.yaml" << 'EOF'
# Default Volume Profile Breakout Strategy Configuration
symbol: "BTC/USDT"
timeframes:
  - 1m
  - 5m
  - 15m
  - 1h

indicators:
  volume_profile:
    enabled: true
    lookback_periods: 100
  
  rsi:
    enabled: true
    period: 14
    
  ema:
    enabled: true
    fast_period: 12
    slow_period: 26

strategy:
  name: "volume_profile_breakout"
  risk_percentage: 0.02
  max_position_size: 0.1
  
entry_conditions:
  - volume_profile_breakout
  - rsi_oversold_bounce
  
exit_conditions:
  - profit_target: 0.03
  - stop_loss: 0.02
EOF
        fi
    else
        log_success "All configuration files found"
    fi
}

# Function to run health checks
run_health_check() {
    log "Running health checks..."
    
    # Check Python imports
    python3 -c "
import sys
try:
    import btc_research
    print('✓ BTC Research module imports successfully')
except ImportError as e:
    print(f'✗ Failed to import btc_research: {e}')
    sys.exit(1)

try:
    import redis
    import fastapi
    import uvicorn
    print('✓ All required dependencies available')
except ImportError as e:
    print(f'✗ Missing dependencies: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "Health checks passed"
    else
        log_error "Health checks failed"
        return 1
    fi
}

# Main entrypoint logic
main() {
    log "Starting BTC Research Paper Trading System..."
    log "Environment: ${ENVIRONMENT:-development}"
    log "Command: $*"
    
    # Initialize directories
    init_directories
    
    # Check configuration
    check_config
    
    # Wait for Redis if needed
    if [[ "$*" != *"redis-server"* ]]; then
        local redis_host=${REDIS_HOST:-redis}
        local redis_port=${REDIS_PORT:-6379}
        
        if ! wait_for_service "$redis_host" "$redis_port" "Redis"; then
            log_error "Cannot start without Redis"
            exit 1
        fi
        
        if ! check_redis; then
            log_error "Redis connectivity check failed"
            exit 1
        fi
    fi
    
    # Run health checks
    if ! run_health_check; then
        log_error "Health checks failed, exiting"
        exit 1
    fi
    
    # Handle special commands
    case "$1" in
        "bash"|"sh")
            log "Starting interactive shell..."
            exec "$@"
            ;;
        "python")
            log "Running Python command: ${*:2}"
            exec "$@"
            ;;
        "uvicorn")
            log "Starting API server with uvicorn..."
            exec "$@"
            ;;
        "gunicorn")
            log "Starting API server with gunicorn..."
            exec "$@"
            ;;
        *)
            log "Starting application: $*"
            exec "$@"
            ;;
    esac
}

# Trap signals for graceful shutdown
trap 'log "Received shutdown signal, gracefully stopping..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@"