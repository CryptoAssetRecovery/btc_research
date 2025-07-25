#!/bin/bash
# Test script for Docker Compose setup

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✓${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ✗${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠${NC} $1"
}

# Function to test Docker Compose file syntax
test_compose_syntax() {
    local compose_file=$1
    log "Testing Docker Compose file syntax: $compose_file"
    
    if docker compose -f "$compose_file" config >/dev/null 2>&1; then
        log_success "Docker Compose syntax valid: $compose_file"
        return 0
    else
        log_error "Docker Compose syntax invalid: $compose_file"
        docker compose -f "$compose_file" config
        return 1
    fi
}

# Function to test Dockerfile syntax
test_dockerfile_syntax() {
    log "Testing Dockerfile syntax..."
    
    if docker build --no-cache -t btc-research-test . >/dev/null 2>&1; then
        log_success "Dockerfile builds successfully"
        
        # Clean up test image
        docker rmi btc-research-test >/dev/null 2>&1 || true
        return 0
    else
        log_error "Dockerfile build failed"
        return 1
    fi
}

# Function to test Redis standalone
test_redis() {
    log "Testing Redis container..."
    
    # Start Redis container
    docker run -d --name test-redis -p 6380:6379 redis:7-alpine >/dev/null
    
    # Wait for Redis to start
    sleep 3
    
    # Test Redis connectivity
    if docker exec test-redis redis-cli ping | grep -q PONG; then
        log_success "Redis container working"
        docker stop test-redis >/dev/null && docker rm test-redis >/dev/null
        return 0
    else
        log_error "Redis container failed"
        docker stop test-redis >/dev/null && docker rm test-redis >/dev/null
        return 1
    fi
}

# Function to test development setup
test_dev_setup() {
    log "Testing development setup..."
    
    # Copy environment file
    if [ ! -f .env ]; then
        cp .env.development .env
        log "Created .env from .env.development"
    fi
    
    # Test basic service startup
    log "Starting Redis for development test..."
    docker compose -f docker-compose.dev.yml up -d redis-dev
    
    # Wait for Redis
    sleep 5
    
    # Test Redis connectivity
    if docker compose -f docker-compose.dev.yml exec -T redis-dev redis-cli ping | grep -q PONG; then
        log_success "Development Redis working"
        
        # Stop services
        docker compose -f docker-compose.dev.yml down >/dev/null
        return 0
    else
        log_error "Development Redis failed"
        docker compose -f docker-compose.dev.yml down >/dev/null
        return 1
    fi
}

# Function to check required files
check_required_files() {
    log "Checking required files..."
    
    local required_files=(
        "Dockerfile"
        "docker-compose.yml"
        "docker-compose.dev.yml"
        "docker-compose.prod.yml"
        ".dockerignore"
        ".env.example"
        ".env.development"
        "config/redis.conf"
        "config/redis-prod.conf"
        "scripts/docker/entrypoint.sh"
        "scripts/docker/health-check.sh"
        "sql/init.sql"
    )
    
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        log_success "All required files present"
        return 0
    else
        log_error "Missing files: ${missing_files[*]}"
        return 1
    fi
}

# Function to validate environment variables
validate_env_vars() {
    log "Validating environment configuration..."
    
    # Check .env.example has all necessary variables
    local required_vars=(
        "ENVIRONMENT"
        "REDIS_URL"
        "API_HOST"
        "API_PORT"
        "SECRET_KEY"
        "JWT_SECRET_KEY"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "^$var=" .env.example; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -eq 0 ]; then
        log_success "Environment variables validated"
        return 0
    else
        log_error "Missing environment variables: ${missing_vars[*]}"
        return 1
    fi
}

# Main test function
main() {
    log "Starting Docker setup validation..."
    
    local tests_passed=0
    local total_tests=0
    
    # Test 1: Check required files
    total_tests=$((total_tests + 1))
    if check_required_files; then
        tests_passed=$((tests_passed + 1))
    fi
    
    # Test 2: Validate environment variables
    total_tests=$((total_tests + 1))
    if validate_env_vars; then
        tests_passed=$((tests_passed + 1))
    fi
    
    # Test 3: Test Dockerfile syntax
    total_tests=$((total_tests + 1))
    if test_dockerfile_syntax; then
        tests_passed=$((tests_passed + 1))
    fi
    
    # Test 4: Test Docker Compose files
    local compose_files=("docker-compose.yml" "docker-compose.dev.yml" "docker-compose.prod.yml")
    for compose_file in "${compose_files[@]}"; do
        total_tests=$((total_tests + 1))
        if test_compose_syntax "$compose_file"; then
            tests_passed=$((tests_passed + 1))
        fi
    done
    
    # Test 5: Test Redis standalone
    total_tests=$((total_tests + 1))
    if test_redis; then
        tests_passed=$((tests_passed + 1))
    fi
    
    # Test 6: Test development setup
    total_tests=$((total_tests + 1))
    if test_dev_setup; then
        tests_passed=$((tests_passed + 1))
    fi
    
    # Report results
    echo
    log "Test results: $tests_passed/$total_tests tests passed"
    
    if [ $tests_passed -eq $total_tests ]; then
        log_success "All Docker setup tests passed! 🎉"
        echo
        log "Next steps:"
        log "1. Start development environment: docker compose -f docker-compose.dev.yml up"
        log "2. Start production environment: docker compose -f docker-compose.prod.yml up -d"
        log "3. View API docs: http://localhost:8000/docs"
        echo
        exit 0
    else
        log_error "Some tests failed ($tests_passed/$total_tests)"
        echo
        log "Please fix the issues above before proceeding"
        exit 1
    fi
}

# Clean up function
cleanup() {
    log "Cleaning up test containers..."
    docker stop test-redis >/dev/null 2>&1 || true
    docker rm test-redis >/dev/null 2>&1 || true
    docker compose -f docker-compose.dev.yml down >/dev/null 2>&1 || true
}

# Set up signal handlers
trap cleanup EXIT SIGINT SIGTERM

# Change to project directory
cd "$(dirname "$0")/../.."

# Run main test
main "$@"