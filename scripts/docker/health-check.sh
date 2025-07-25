#!/bin/bash
# Health check script for Docker containers

set -e

# Configuration
SERVICE_TYPE=${1:-"api"}
TIMEOUT=${HEALTH_CHECK_TIMEOUT:-10}
REDIS_URL=${REDIS_URL:-"redis://localhost:6379/0"}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >&2
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $1${NC}" >&2
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗ $1${NC}" >&2
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}" >&2
}

# Function to check Redis connectivity
check_redis() {
    log "Checking Redis connectivity..."
    
    python3 -c "
import redis
import sys
from urllib.parse import urlparse

try:
    parsed = urlparse('$REDIS_URL')
    r = redis.Redis(
        host=parsed.hostname or 'localhost',
        port=parsed.port or 6379,
        db=int(parsed.path.lstrip('/')) if parsed.path else 0,
        password=parsed.password,
        socket_timeout=$TIMEOUT
    )
    
    # Test basic operations
    r.ping()
    r.set('health_check', 'ok', ex=60)
    result = r.get('health_check')
    
    if result == b'ok':
        print('Redis health check passed')
        sys.exit(0)
    else:
        print('Redis health check failed: could not read test value')
        sys.exit(1)
        
except Exception as e:
    print(f'Redis health check failed: {e}')
    sys.exit(1)
" 2>/dev/null
    
    return $?
}

# Function to check API server health
check_api() {
    log "Checking API server health..."
    
    local api_url="http://localhost:8000/health"
    local response
    
    # Try to get health endpoint
    if command -v curl >/dev/null; then
        response=$(curl -s -f --max-time $TIMEOUT "$api_url" 2>/dev/null)
        local curl_exit=$?
        
        if [ $curl_exit -eq 0 ]; then
            log_success "API server health check passed"
            return 0
        else
            log_error "API server health check failed (curl exit: $curl_exit)"
            return 1
        fi
    else
        log_error "curl not available for API health check"
        return 1
    fi
}

# Function to check paper trader health
check_paper_trader() {
    log "Checking paper trader health..."
    
    # Check if Redis has trading data
    python3 -c "
import redis
import sys
import json
from urllib.parse import urlparse

try:
    parsed = urlparse('$REDIS_URL')
    r = redis.Redis(
        host=parsed.hostname or 'localhost',
        port=parsed.port or 6379,
        db=int(parsed.path.lstrip('/')) if parsed.path else 0,
        password=parsed.password,
        socket_timeout=$TIMEOUT
    )
    
    # Check if we can access trading data structures
    r.ping()
    
    # Check for strategy registry
    strategies = r.keys('strategy:*')
    print(f'Found {len(strategies)} strategy entries in Redis')
    
    # Check for market data
    market_data = r.keys('market_data:*')
    print(f'Found {len(market_data)} market data entries in Redis')
    
    print('Paper trader health check passed')
    sys.exit(0)
    
except Exception as e:
    print(f'Paper trader health check failed: {e}')
    sys.exit(1)
" 2>/dev/null
    
    return $?
}

# Function to check dashboard health
check_dashboard() {
    log "Checking dashboard health..."
    
    local dashboard_url="http://localhost:5000"
    
    if command -v curl >/dev/null; then
        curl -s -f --max-time $TIMEOUT "$dashboard_url" >/dev/null 2>&1
        local curl_exit=$?
        
        if [ $curl_exit -eq 0 ]; then
            log_success "Dashboard health check passed"
            return 0
        else
            log_error "Dashboard health check failed (curl exit: $curl_exit)"
            return 1
        fi
    else
        log_error "curl not available for dashboard health check"
        return 1
    fi
}

# Function to check system resources
check_resources() {
    log "Checking system resources..."
    
    # Check memory usage
    if command -v free >/dev/null; then
        local mem_usage=$(free | awk '/Mem:/ {printf "%.1f", $3/$2 * 100.0}')
        log "Memory usage: ${mem_usage}%"
        
        if (( $(echo "$mem_usage > 90" | bc -l) )); then
            log_warning "High memory usage: ${mem_usage}%"
        fi
    fi
    
    # Check disk space
    if command -v df >/dev/null; then
        local disk_usage=$(df /app | awk 'NR==2 {print $5}' | sed 's/%//')
        log "Disk usage: ${disk_usage}%"
        
        if [ "$disk_usage" -gt 90 ]; then
            log_warning "High disk usage: ${disk_usage}%"
        fi
    fi
    
    return 0
}

# Main health check function
main() {
    log "Starting health check for service: $SERVICE_TYPE"
    
    local checks_passed=0
    local total_checks=0
    
    # Always check Redis first
    total_checks=$((total_checks + 1))
    if check_redis; then
        log_success "Redis check passed"
        checks_passed=$((checks_passed + 1))
    else
        log_error "Redis check failed"
    fi
    
    # Check specific service type
    case "$SERVICE_TYPE" in
        "api"|"api-server")
            total_checks=$((total_checks + 1))
            if check_api; then
                checks_passed=$((checks_passed + 1))
            fi
            ;;
        "paper-trader"|"trader")
            total_checks=$((total_checks + 1))
            if check_paper_trader; then
                checks_passed=$((checks_passed + 1))
            fi
            ;;
        "dashboard"|"web")
            total_checks=$((total_checks + 1))
            if check_dashboard; then
                checks_passed=$((checks_passed + 1))
            fi
            ;;
        "all")
            total_checks=$((total_checks + 3))
            check_api && checks_passed=$((checks_passed + 1))
            check_paper_trader && checks_passed=$((checks_passed + 1))
            check_dashboard && checks_passed=$((checks_passed + 1))
            ;;
    esac
    
    # Check system resources (non-critical)
    check_resources
    
    # Report results
    log "Health check results: $checks_passed/$total_checks checks passed"
    
    if [ $checks_passed -eq $total_checks ]; then
        log_success "All health checks passed"
        exit 0
    else
        log_error "Some health checks failed ($checks_passed/$total_checks)"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-api}" in
    "redis")
        check_redis
        exit $?
        ;;
    "api"|"api-server")
        SERVICE_TYPE="api"
        ;;
    "paper-trader"|"trader")
        SERVICE_TYPE="paper-trader"
        ;;
    "dashboard"|"web")
        SERVICE_TYPE="dashboard"
        ;;
    "all")
        SERVICE_TYPE="all"
        ;;
    *)
        SERVICE_TYPE="api"
        ;;
esac

# Run main health check
main