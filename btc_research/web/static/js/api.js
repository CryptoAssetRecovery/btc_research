/**
 * API Client for Paper Trading Dashboard
 * Handles communication with FastAPI backend
 */

class APIClient {
    constructor(baseUrl = '', apiKey = 'dev-key-123') {
        this.baseUrl = baseUrl;
        this.apiKey = apiKey;
        this.cache = new Map();
        this.cacheTimeout = 30000; // 30 seconds
    }

    /**
     * Make authenticated GET request to API
     * @param {string} endpoint - API endpoint
     * @param {Object} params - Query parameters
     * @param {boolean} useCache - Whether to use caching
     * @returns {Promise<Object>} API response
     */
    async get(endpoint, params = {}, useCache = true) {
        const url = this.buildUrl(endpoint, params);
        const cacheKey = url;

        // Check cache first
        if (useCache && this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                return cached.data;
            }
        }

        try {
            const response = await fetch(url, {
                method: 'GET',
                headers: this.getHeaders()
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            // Cache successful responses
            if (useCache) {
                this.cache.set(cacheKey, {
                    data: data,
                    timestamp: Date.now()
                });
            }

            return data;

        } catch (error) {
            console.error(`API GET error for ${endpoint}:`, error);
            throw error;
        }
    }

    /**
     * Make authenticated POST request to API
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body data
     * @returns {Promise<Object>} API response
     */
    async post(endpoint, data = {}) {
        const url = this.buildUrl(endpoint);

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: this.getHeaders(),
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            const responseData = await response.json();

            // Invalidate related cache entries
            this.invalidateCache(endpoint);

            return responseData;

        } catch (error) {
            console.error(`API POST error for ${endpoint}:`, error);
            throw error;
        }
    }

    /**
     * Get system health status
     * @returns {Promise<Object>} Health status
     */
    async getHealth() {
        return this.get('/health', {}, false); // Don't cache health checks
    }

    /**
     * Get active strategies
     * @returns {Promise<Object>} Active strategies
     */
    async getActiveStrategies() {
        return this.get('/api/v1/strategies/active');
    }

    /**
     * Get strategy details
     * @param {string} strategyId - Strategy ID
     * @returns {Promise<Object>} Strategy details
     */
    async getStrategy(strategyId) {
        return this.get(`/api/v1/strategies/${strategyId}/status`);
    }

    /**
     * Get strategy statistics
     * @param {string} strategyId - Strategy ID
     * @returns {Promise<Object>} Strategy statistics
     */
    async getStrategyStats(strategyId) {
        return this.get(`/api/v1/strategies/${strategyId}/stats`);
    }

    /**
     * Get strategy trades
     * @param {string} strategyId - Strategy ID
     * @param {number} limit - Number of trades to retrieve
     * @param {number} skip - Number of trades to skip
     * @returns {Promise<Object>} Trade history
     */
    async getStrategyTrades(strategyId, limit = 20, skip = 0) {
        return this.get(`/api/v1/statistics/strategies/${strategyId}/trades`, {
            limit,
            skip
        });
    }

    /**
     * Get strategy positions
     * @param {string} strategyId - Strategy ID
     * @returns {Promise<Object>} Current positions
     */
    async getStrategyPositions(strategyId) {
        return this.get(`/api/v1/statistics/strategies/${strategyId}/positions`);
    }

    /**
     * Get market data
     * @param {string} symbol - Trading symbol
     * @param {string} timeframe - Data timeframe
     * @param {number} limit - Number of candles
     * @returns {Promise<Object>} Market data
     */
    async getMarketData(symbol, timeframe = '1m', limit = 100) {
        return this.get(`/api/v1/market/data/${symbol}`, {
            timeframe,
            limit
        });
    }

    /**
     * Start new strategy
     * @param {string} configPath - Strategy configuration path
     * @param {number} initialBalance - Initial balance
     * @returns {Promise<Object>} Start strategy response
     */
    async startStrategy(configPath, initialBalance) {
        return this.post('/api/v1/strategies/start', {
            config_path: configPath,
            initial_balance: initialBalance
        });
    }

    /**
     * Stop strategy
     * @param {string} strategyId - Strategy ID
     * @returns {Promise<Object>} Stop strategy response
     */
    async stopStrategy(strategyId) {
        return this.post(`/api/v1/strategies/${strategyId}/stop`);
    }

    /**
     * Get available configurations
     * @returns {Promise<Object>} Available configurations
     */
    async getConfigurations() {
        return this.get('/api/v1/configs/available');
    }

    /**
     * Validate configuration
     * @param {Object} config - Configuration to validate
     * @returns {Promise<Object>} Validation result
     */
    async validateConfiguration(config) {
        return this.post('/api/v1/configs/validate', config);
    }

    /**
     * Build full URL with parameters
     * @private
     */
    buildUrl(endpoint, params = {}) {
        const url = new URL(endpoint, this.baseUrl || window.location.origin);
        
        Object.entries(params).forEach(([key, value]) => {
            if (value !== null && value !== undefined) {
                url.searchParams.append(key, value);
            }
        });

        return url.toString();
    }

    /**
     * Get request headers
     * @private
     */
    getHeaders() {
        return {
            'Content-Type': 'application/json',
            'X-API-Key': this.apiKey,
            'Accept': 'application/json'
        };
    }

    /**
     * Invalidate cache entries related to endpoint
     * @private
     */
    invalidateCache(endpoint) {
        // Remove cache entries that might be affected by this change
        const keysToDelete = [];
        
        for (const key of this.cache.keys()) {
            if (key.includes('/strategies') || key.includes('/statistics')) {
                keysToDelete.push(key);
            }
        }

        keysToDelete.forEach(key => this.cache.delete(key));
    }

    /**
     * Clear all cached data
     */
    clearCache() {
        this.cache.clear();
    }

    /**
     * Get cache statistics
     */
    getCacheStats() {
        return {
            size: this.cache.size,
            keys: Array.from(this.cache.keys())
        };
    }
}

/**
 * Utility Functions
 */

/**
 * Format currency values
 * @param {number} value - Numeric value
 * @param {string} currency - Currency code
 * @returns {string} Formatted currency string
 */
function formatCurrency(value, currency = 'USD') {
    if (value === null || value === undefined || isNaN(value)) {
        return '$0.00';
    }
    
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency,
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

/**
 * Format percentage values
 * @param {number} value - Numeric value
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted percentage string
 */
function formatPercentage(value, decimals = 2) {
    if (value === null || value === undefined || isNaN(value)) {
        return '0.00%';
    }
    
    return `${parseFloat(value).toFixed(decimals)}%`;
}

/**
 * Format date/time values
 * @param {string|Date} timestamp - Timestamp to format
 * @param {boolean} includeTime - Whether to include time
 * @returns {string} Formatted date string
 */
function formatDateTime(timestamp, includeTime = true) {
    if (!timestamp) return 'N/A';
    
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return 'Invalid Date';
    
    const options = {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    };
    
    if (includeTime) {
        options.hour = '2-digit';
        options.minute = '2-digit';
        options.second = '2-digit';
    }
    
    return date.toLocaleDateString('en-US', options);
}

/**
 * Format large numbers with suffixes
 * @param {number} value - Numeric value
 * @returns {string} Formatted number string
 */
function formatNumber(value) {
    if (value === null || value === undefined || isNaN(value)) {
        return '0';
    }
    
    const absValue = Math.abs(value);
    
    if (absValue >= 1e9) {
        return (value / 1e9).toFixed(1) + 'B';
    } else if (absValue >= 1e6) {
        return (value / 1e6).toFixed(1) + 'M';
    } else if (absValue >= 1e3) {
        return (value / 1e3).toFixed(1) + 'K';
    }
    
    return value.toLocaleString();
}

/**
 * Debounce function calls
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function calls
 * @param {Function} func - Function to throttle
 * @param {number} limit - Time limit in milliseconds
 * @returns {Function} Throttled function
 */
function throttle(func, limit) {
    let inThrottle;
    
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Show toast notification
 * @param {string} message - Message to display
 * @param {string} type - Toast type (success, error, warning, info)
 * @param {number} duration - Display duration in milliseconds
 */
function showToast(message, type = 'info', duration = 5000) {
    // Create toast container if it doesn't exist
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '1055';
        document.body.appendChild(container);
    }
    
    // Create toast element
    const toastId = 'toast-' + Date.now();
    const bgClass = {
        success: 'bg-success',
        error: 'bg-danger',
        warning: 'bg-warning',
        info: 'bg-info'
    }[type] || 'bg-info';
    
    const iconClass = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-triangle',
        warning: 'fa-exclamation-circle',
        info: 'fa-info-circle'
    }[type] || 'fa-info-circle';
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white ${bgClass} border-0`;
    toast.id = toastId;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas ${iconClass} me-2"></i>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                    data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    container.appendChild(toast);
    
    // Initialize and show toast
    const bsToast = new bootstrap.Toast(toast, {
        autohide: true,
        delay: duration
    });
    
    bsToast.show();
    
    // Remove from DOM after hiding
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
    
    return bsToast;
}

/**
 * Show loading overlay
 * @param {boolean} show - Whether to show or hide
 * @param {string} message - Loading message
 */
function showLoading(show = true, message = 'Loading...') {
    let overlay = document.getElementById('loadingOverlay');
    
    if (show) {
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'loadingOverlay';
            overlay.className = 'loading-overlay';
            overlay.innerHTML = `
                <div class="text-center">
                    <div class="loading-spinner mb-3"></div>
                    <h5 class="text-white">${message}</h5>
                </div>
            `;
            document.body.appendChild(overlay);
        }
        overlay.style.display = 'flex';
    } else {
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
}

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 * @returns {Promise<boolean>} Success status
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard!', 'success', 2000);
        return true;
    } catch (error) {
        console.error('Failed to copy to clipboard:', error);
        showToast('Failed to copy to clipboard', 'error');
        return false;
    }
}

/**
 * Download data as file
 * @param {string} filename - File name
 * @param {string} content - File content
 * @param {string} contentType - MIME type
 */
function downloadFile(filename, content, contentType = 'text/plain') {
    const blob = new Blob([content], { type: contentType });
    const url = window.URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.style.display = 'none';
    
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    window.URL.revokeObjectURL(url);
}

// Initialize global API client with configuration from backend
console.log('API_CONFIG:', window.API_CONFIG);
const baseUrl = window.API_CONFIG?.baseUrl || 'http://localhost:8002';
const apiKey = window.API_CONFIG?.apiKey || 'dev-key-123';
console.log('Initializing API client with baseUrl:', baseUrl, 'apiKey:', apiKey);

window.apiClient = new APIClient(baseUrl, apiKey);
console.log('API client initialized with baseUrl:', window.apiClient.baseUrl);

// Export utilities for global use
window.formatCurrency = formatCurrency;
window.formatPercentage = formatPercentage;
window.formatDateTime = formatDateTime;
window.formatNumber = formatNumber;
window.showToast = showToast;
window.showLoading = showLoading;
window.copyToClipboard = copyToClipboard;
window.downloadFile = downloadFile;
window.debounce = debounce;
window.throttle = throttle;