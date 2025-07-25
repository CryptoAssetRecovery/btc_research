#!/bin/bash
# Backup script for BTC Research Paper Trading System

set -e

# Configuration
BACKUP_DIR="/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
POSTGRES_HOST=${POSTGRES_HOST:-postgres-prod}
POSTGRES_DB=${POSTGRES_DB:-paper_trading}
POSTGRES_USER=${POSTGRES_USER:-trader}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

# Create backup directory
create_backup_dir() {
    local backup_path="$BACKUP_DIR/$TIMESTAMP"
    mkdir -p "$backup_path"
    echo "$backup_path"
}

# Backup PostgreSQL database
backup_database() {
    local backup_path=$1
    log "Starting database backup..."
    
    local db_backup_file="$backup_path/database_${TIMESTAMP}.sql"
    
    if pg_dump -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
               --no-password --verbose --format=custom \
               --file="$db_backup_file.custom" 2>/dev/null; then
        
        # Also create plain SQL backup
        pg_dump -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
                --no-password --format=plain \
                --file="$db_backup_file" 2>/dev/null
        
        # Compress the backups
        gzip "$db_backup_file"
        
        log_success "Database backup completed: $(basename "$db_backup_file.gz")"
        return 0
    else
        log_error "Database backup failed"
        return 1
    fi
}

# Backup Redis data
backup_redis() {
    local backup_path=$1
    log "Starting Redis backup..."
    
    # Create Redis dump
    local redis_backup_file="$backup_path/redis_${TIMESTAMP}.rdb"
    
    if docker exec btc-redis-prod redis-cli BGSAVE >/dev/null 2>&1; then
        sleep 5  # Wait for background save to complete
        
        if docker cp btc-redis-prod:/data/dump.rdb "$redis_backup_file"; then
            gzip "$redis_backup_file"
            log_success "Redis backup completed: $(basename "$redis_backup_file.gz")"
            return 0
        fi
    fi
    
    log_error "Redis backup failed"
    return 1
}

# Backup configuration files
backup_configs() {
    local backup_path=$1
    log "Starting configuration backup..."
    
    local config_backup_dir="$backup_path/configs"
    mkdir -p "$config_backup_dir"
    
    # Copy important configuration files
    cp -r config/ "$config_backup_dir/" 2>/dev/null || true
    cp docker-compose*.yml "$config_backup_dir/" 2>/dev/null || true
    cp .env.example "$config_backup_dir/" 2>/dev/null || true
    cp Dockerfile "$config_backup_dir/" 2>/dev/null || true
    
    # Copy strategy configurations
    if [ -d "btc_research/config" ]; then
        cp -r btc_research/config/ "$config_backup_dir/strategies/" 2>/dev/null || true
    fi
    
    log_success "Configuration backup completed"
}

# Backup application data
backup_app_data() {
    local backup_path=$1
    log "Starting application data backup..."
    
    local data_backup_dir="$backup_path/app_data"
    mkdir -p "$data_backup_dir"
    
    # Copy important data directories
    if [ -d "data" ]; then
        cp -r data/ "$data_backup_dir/" 2>/dev/null || true
    fi
    
    if [ -d "logs" ]; then
        # Only backup recent logs (last 7 days)
        find logs/ -name "*.log" -mtime -7 -exec cp {} "$data_backup_dir/" \; 2>/dev/null || true
    fi
    
    log_success "Application data backup completed"
}

# Create backup metadata
create_metadata() {
    local backup_path=$1
    local metadata_file="$backup_path/backup_metadata.json"
    
    cat > "$metadata_file" << EOF
{
    "timestamp": "$TIMESTAMP",
    "date": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "postgres_version": "$(docker exec btc-postgres-prod psql -U $POSTGRES_USER -d $POSTGRES_DB -c 'SELECT version();' -t 2>/dev/null | head -1 | xargs || echo 'unknown')",
    "redis_version": "$(docker exec btc-redis-prod redis-cli INFO server | grep redis_version | cut -d: -f2 | tr -d '\r' || echo 'unknown')",
    "system_info": {
        "disk_usage": "$(df -h $BACKUP_DIR | tail -1 | awk '{print $5}')",
        "memory_usage": "$(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')",
        "uptime": "$(uptime -p)"
    }
}
EOF
    
    log_success "Backup metadata created"
}

# Compress entire backup
compress_backup() {
    local backup_path=$1
    log "Compressing backup archive..."
    
    local archive_name="btc_research_backup_${TIMESTAMP}.tar.gz"
    local archive_path="$BACKUP_DIR/$archive_name"
    
    if tar -czf "$archive_path" -C "$BACKUP_DIR" "$(basename "$backup_path")"; then
        # Remove uncompressed directory
        rm -rf "$backup_path"
        log_success "Backup compressed: $archive_name"
        echo "$archive_path"
    else
        log_error "Backup compression failed"
        return 1
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    local retention_days=${BACKUP_RETENTION_DAYS:-30}
    log "Cleaning up backups older than $retention_days days..."
    
    find "$BACKUP_DIR" -name "btc_research_backup_*.tar.gz" -mtime +$retention_days -delete 2>/dev/null || true
    
    local remaining=$(find "$BACKUP_DIR" -name "btc_research_backup_*.tar.gz" | wc -l)
    log_success "Cleanup completed. $remaining backup(s) remaining."
}

# Verify backup integrity
verify_backup() {
    local archive_path=$1
    log "Verifying backup integrity..."
    
    if tar -tzf "$archive_path" >/dev/null 2>&1; then
        log_success "Backup integrity verified"
        return 0
    else
        log_error "Backup integrity check failed"
        return 1
    fi
}

# Main backup function
main() {
    log "Starting BTC Research backup process..."
    
    # Check if required tools are available
    if ! command -v pg_dump >/dev/null 2>&1; then
        log_error "pg_dump not available"
        exit 1
    fi
    
    # Create backup directory
    local backup_path
    backup_path=$(create_backup_dir)
    
    local success=true
    
    # Perform backups
    backup_database "$backup_path" || success=false
    backup_redis "$backup_path" || success=false
    backup_configs "$backup_path"
    backup_app_data "$backup_path"
    create_metadata "$backup_path"
    
    if [ "$success" = true ]; then
        # Compress and verify
        local archive_path
        archive_path=$(compress_backup "$backup_path")
        
        if verify_backup "$archive_path"; then
            cleanup_old_backups
            
            log_success "Backup process completed successfully"
            log "Archive: $(basename "$archive_path")"
            log "Size: $(du -h "$archive_path" | cut -f1)"
            exit 0
        else
            log_error "Backup verification failed"
            exit 1
        fi
    else
        log_error "Backup process completed with errors"
        exit 1
    fi
}

# Handle signals
trap 'log_error "Backup interrupted"; exit 1' SIGINT SIGTERM

# Run main function
main "$@"