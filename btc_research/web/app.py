"""
Flask web application for paper trading dashboard.
Provides web interface for monitoring and controlling the paper trading system.
"""

import os
import json
import asyncio
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_socketio import SocketIO, emit
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-123')

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:8002')
API_KEY = os.environ.get('API_KEY', 'dev-key-123')

# Frontend API URL (for browser-side JavaScript)
# In Docker, internal services use container names, but frontend needs external URLs
FRONTEND_API_URL = os.environ.get('FRONTEND_API_URL', 'http://localhost:8002')


class APIClient:
    """Client for communicating with FastAPI backend."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        })
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request to API."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API GET error for {endpoint}: {e}")
            return {"error": str(e)}
    
    def post(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make POST request to API."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API POST error for {endpoint}: {e}")
            return {"error": str(e)}


# Initialize API client
api_client = APIClient(API_BASE_URL, API_KEY)


def generate_mock_candles(limit: int, period: str) -> List[Dict]:
    """Generate mock OHLCV data for demonstration purposes."""
    try:
        # Base BTC price around $43,000
        base_price = 43000.0
        current_price = base_price
        candles = []
        
        # Calculate time interval based on period
        time_intervals = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '1d': timedelta(days=1)
        }
        
        interval = time_intervals.get(period, timedelta(minutes=1))
        start_time = datetime.utcnow() - (interval * limit)
        
        for i in range(limit):
            timestamp = start_time + (interval * i)
            
            # Generate realistic price movement
            volatility = 0.002  # 0.2% volatility
            change = random.gauss(0, volatility)
            current_price *= (1 + change)
            
            # Ensure price doesn't go too far from base
            current_price = max(base_price * 0.8, min(base_price * 1.2, current_price))
            
            # Generate OHLC from current price
            open_price = current_price
            high_offset = random.uniform(0, 0.005)  # Up to 0.5% higher
            low_offset = random.uniform(0, 0.005)   # Up to 0.5% lower
            close_offset = random.uniform(-0.003, 0.003)  # +/- 0.3%
            
            high = open_price * (1 + high_offset)
            low = open_price * (1 - low_offset)
            close = open_price * (1 + close_offset)
            
            # Ensure high >= low and OHLC consistency
            high = max(open_price, close, high)
            low = min(open_price, close, low)
            
            # Generate volume (higher volume on larger price movements)
            base_volume = random.uniform(10, 50)  # Base BTC volume
            volume_multiplier = 1 + abs(change) * 10
            volume = base_volume * volume_multiplier
            
            current_price = close  # Update current price for next candle
            
            candles.append({
                'timestamp': timestamp.isoformat() + 'Z',
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': round(volume, 4)
            })
        
        return candles
        
    except Exception as e:
        logger.error(f"Error generating mock candles: {e}")
        return []


def generate_mock_indicators(candles: List[Dict], trades: List[Dict]) -> Dict:
    """Generate mock technical indicators for demonstration."""
    try:
        if not candles:
            return {}
        
        prices = [float(candle['close']) for candle in candles]
        volumes = [float(candle['volume']) for candle in candles]
        
        indicators = {}
        
        # EMA calculations
        if len(prices) >= 20:
            indicators['ema'] = {
                '20': calculate_ema(prices, 20),
                '50': calculate_ema(prices, 50) if len(prices) >= 50 else []
            }
        
        # Volume Profile (simplified)
        if volumes:
            vp_levels = generate_volume_profile(prices, volumes)
            indicators['volumeProfile'] = vp_levels
        
        # FVG zones (Fair Value Gaps) - simplified mock
        fvg_zones = []
        for i in range(2, len(candles) - 1):
            prev_candle = candles[i-1]
            curr_candle = candles[i]
            next_candle = candles[i+1]
            
            # Simple gap detection
            gap_up = float(curr_candle['low']) > float(prev_candle['high'])
            gap_down = float(curr_candle['high']) < float(prev_candle['low'])
            
            if gap_up or gap_down:
                fvg_zones.append({
                    'start': curr_candle['timestamp'],
                    'end': next_candle['timestamp'],
                    'top': max(float(prev_candle['high']), float(curr_candle['high'])),
                    'bottom': min(float(prev_candle['low']), float(curr_candle['low'])),
                    'type': 'bullish' if gap_up else 'bearish'
                })
        
        indicators['fvg'] = fvg_zones[:5]  # Limit to 5 zones
        
        # ADX trend strength (simplified)
        if len(prices) >= 14:
            adx_values = calculate_mock_adx(prices, 14)
            indicators['adx'] = adx_values
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error generating mock indicators: {e}")
        return {}


def calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return []
    
    multiplier = 2 / (period + 1)
    ema_values = []
    
    # Start with SMA for first value
    sma = sum(prices[:period]) / period
    ema_values.append(sma)
    
    # Calculate EMA for remaining values
    for i in range(period, len(prices)):
        ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
        ema_values.append(ema)
    
    # Pad with None values for the initial period
    return [None] * (period - 1) + ema_values


def generate_volume_profile(prices: List[float], volumes: List[float]) -> List[Dict]:
    """Generate simplified volume profile levels."""
    if not prices or not volumes:
        return []
    
    min_price = min(prices) * 0.995
    max_price = max(prices) * 1.005
    num_levels = 20
    
    price_levels = []
    level_height = (max_price - min_price) / num_levels
    
    for i in range(num_levels):
        level_price = min_price + (i * level_height)
        
        # Calculate volume at this price level
        level_volume = 0
        for j, price in enumerate(prices):
            if level_price <= price < level_price + level_height:
                level_volume += volumes[j]
        
        price_levels.append({
            'price': round(level_price, 2),
            'volume': round(level_volume, 4),
            'maxVolume': max(volumes) if volumes else 1
        })
    
    return sorted(price_levels, key=lambda x: x['volume'], reverse=True)[:10]


def calculate_mock_adx(prices: List[float], period: int) -> List[float]:
    """Calculate simplified ADX (Average Directional Index) for trend strength."""
    if len(prices) < period * 2:
        return []
    
    # Simplified ADX calculation - just trend strength indicator
    adx_values = []
    
    for i in range(period, len(prices)):
        # Look at price changes over the period
        recent_prices = prices[i-period:i]
        price_changes = [abs(recent_prices[j] - recent_prices[j-1]) 
                        for j in range(1, len(recent_prices))]
        
        # Simple trend strength (0-100)
        avg_change = sum(price_changes) / len(price_changes)
        trend_strength = min(100, (avg_change / recent_prices[-1]) * 10000)
        
        adx_values.append(round(trend_strength, 2))
    
    return [None] * period + adx_values


@app.route('/')
def dashboard():
    """Main dashboard showing strategy overview."""
    # Get active strategies
    strategies_data = api_client.get('/api/v1/strategies/active')
    
    if 'error' in strategies_data:
        flash(f"Error loading strategies: {strategies_data['error']}", 'error')
        strategies = []
        condition_metrics_data = {}
    else:
        strategies = strategies_data.get('strategies', [])
        condition_metrics_data = {}
        
        # Fetch statistics and condition metrics for each strategy
        for strategy in strategies:
            try:
                # Get basic stats (includes condition_metrics_summary)
                stats_data = api_client.get(f'/api/v1/strategies/{strategy["id"]}/stats')
                if 'error' not in stats_data and 'stats' in stats_data:
                    strategy['stats'] = stats_data['stats']
                else:
                    logger.warning(f"Failed to load stats for strategy {strategy['id']}: {stats_data}")
                    strategy['stats'] = None
                
                # Get detailed condition metrics for dashboard transparency
                conditions_data = api_client.get(f'/api/v1/statistics/strategies/{strategy["id"]}/conditions')
                if 'error' not in conditions_data and 'data' in conditions_data:
                    condition_metrics_data[strategy['id']] = conditions_data['data']
                else:
                    logger.warning(f"Failed to load condition metrics for strategy {strategy['id']}: {conditions_data}")
                    condition_metrics_data[strategy['id']] = None
                    
            except Exception as e:
                logger.error(f"Error fetching data for strategy {strategy['id']}: {e}")
                strategy['stats'] = None
                condition_metrics_data[strategy['id']] = None
    
    return render_template('dashboard.html', 
                         strategies=strategies,
                         condition_metrics=condition_metrics_data,
                         page_title="Dashboard")


@app.route('/strategy/<strategy_id>')
def strategy_detail(strategy_id: str):
    """Individual strategy detail page."""
    # Get strategy status
    strategy_data = api_client.get(f'/api/v1/strategies/{strategy_id}/status')
    
    if 'error' in strategy_data:
        flash(f"Error loading strategy: {strategy_data['error']}", 'error')
        return redirect(url_for('dashboard'))
    
    # Get strategy statistics
    stats_data = api_client.get(f'/api/v1/strategies/{strategy_id}/stats')
    
    # Get recent trades
    trades_data = api_client.get(f'/api/v1/statistics/strategies/{strategy_id}/trades', 
                                params={'limit': 10})
    
    # Get current positions
    positions_data = api_client.get(f'/api/v1/statistics/strategies/{strategy_id}/positions')
    
    # Get detailed condition metrics for this strategy
    condition_metrics_data = api_client.get(f'/api/v1/statistics/strategies/{strategy_id}/conditions')
    
    # Process condition metrics for template
    condition_metrics = None
    if 'error' not in condition_metrics_data and 'data' in condition_metrics_data:
        condition_metrics = condition_metrics_data['data']
        logger.info(f"Retrieved condition metrics for strategy {strategy_id}")
    else:
        logger.warning(f"Failed to load condition metrics for strategy {strategy_id}: {condition_metrics_data}")
    
    return render_template('strategy.html',
                         strategy=strategy_data.get('strategy', strategy_data),
                         stats=stats_data,
                         trades=trades_data.get('trades', []),
                         positions=positions_data.get('positions', []),
                         condition_metrics=condition_metrics,
                         strategy_id=strategy_id,
                         page_title=f"Strategy {strategy_id}")


@app.route('/trades')
def trades():
    """Trade history page with pagination."""
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)
    strategy_id = request.args.get('strategy_id')
    
    params = {
        'limit': limit,
        'skip': (page - 1) * limit
    }
    
    if strategy_id:
        endpoint = f'/api/v1/statistics/strategies/{strategy_id}/trades'
        trades_data = api_client.get(endpoint, params=params)
    else:
        # Get all trades from all strategies
        strategies_data = api_client.get('/api/v1/strategies/active')
        all_trades = []
        
        if 'strategies' in strategies_data:
            for strategy in strategies_data['strategies']:
                trades_data_temp = api_client.get(f'/api/v1/statistics/strategies/{strategy["id"]}/trades', 
                                                params={'limit': 100})  # Get more trades for aggregation
                if 'trades' in trades_data_temp:
                    for trade in trades_data_temp['trades']:
                        trade['strategy_id'] = strategy['id']
                        trade['strategy_name'] = strategy.get('name', strategy['id'])
                    all_trades.extend(trades_data_temp['trades'])
        
        # Sort by timestamp and paginate
        all_trades.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        trades_data = {
            'trades': all_trades[start_idx:end_idx],
            'total': len(all_trades)
        }
    
    if 'error' in trades_data:
        flash(f"Error loading trades: {trades_data['error']}", 'error')
        trades_data = {'trades': [], 'total': 0}
    
    # Calculate pagination info
    total_trades = trades_data.get('total', 0)
    total_pages = (total_trades + limit - 1) // limit
    
    return render_template('trades.html',
                         trades=trades_data.get('trades', []),
                         page=page,
                         total_pages=total_pages,
                         total_trades=total_trades,
                         limit=limit,
                         strategy_id=strategy_id,
                         page_title="Trade History")


@app.route('/charts')
def charts():
    """Charts page with basic price/PnL charts."""
    # Get available strategies for chart selection
    strategies_data = api_client.get('/api/v1/strategies/active')
    strategies = strategies_data.get('strategies', []) if 'error' not in strategies_data else []
    
    return render_template('charts.html',
                         strategies=strategies,
                         page_title="Charts")


@app.route('/api/chart-data/<strategy_id>')
def chart_data(strategy_id: str):
    """API endpoint for comprehensive chart data."""
    try:  
        # Get query parameters
        period = request.args.get('period', '1m')
        limit = min(int(request.args.get('limit', 100)), 1000)  # Cap at 1000 candles
        
        logger.info(f"Loading chart data for strategy {strategy_id}, period {period}, limit {limit}")
        
        # Get trades for the strategy
        trades_data = api_client.get(f'/api/v1/statistics/strategies/{strategy_id}/trades', 
                                    params={'limit': limit})
        
        # Get market data with proper timeframe
        market_data = api_client.get('/api/v1/market/data/BTC/USDT', 
                                    params={'timeframe': period, 'limit': limit})
        
        # Generate mock OHLCV data if API doesn't provide it
        mock_candles = generate_mock_candles(limit, period)
        
        # Format trades data for charts
        formatted_trades = []
        if 'trades' in trades_data and trades_data['trades']:
            for trade in trades_data['trades']:
                formatted_trades.append({
                    'id': trade.get('id', ''),
                    'timestamp': trade.get('timestamp', ''),
                    'symbol': trade.get('symbol', 'BTC/USDT'),
                    'side': trade.get('side', 'buy'),
                    'size': float(trade.get('size', 0)),
                    'price': float(trade.get('price', 0)),
                    'pnl': float(trade.get('realized_pnl', 0))
                })
        
        # Use mock candles if real market data is not available
        formatted_candles = mock_candles
        if 'data' in market_data and market_data['data'] and 'candles' in market_data['data']:
            # Use real market data if available
            candles = market_data['data']['candles']
            formatted_candles = [{
                'timestamp': candle.get('timestamp', ''),
                'open': float(candle.get('open', 0)),
                'high': float(candle.get('high', 0)),
                'low': float(candle.get('low', 0)),
                'close': float(candle.get('close', 0)),
                'volume': float(candle.get('volume', 0))
            } for candle in candles]
        
        # Generate mock indicators data
        indicators = generate_mock_indicators(formatted_candles, formatted_trades)
        
        chart_response = {
            'trades': formatted_trades,
            'market_data': formatted_candles,
            'indicators': indicators,
            'metadata': {
                'strategy_id': strategy_id,
                'period': period,
                'candles_count': len(formatted_candles),
                'trades_count': len(formatted_trades),
                'generated_at': datetime.utcnow().isoformat()
            },
            'error': None
        }
        
        logger.info(f"Chart data loaded: {len(formatted_candles)} candles, {len(formatted_trades)} trades")
        return jsonify(chart_response)
        
    except Exception as e:
        logger.error(f"Error loading chart data for strategy {strategy_id}: {e}")
        return jsonify({
            'trades': [],
            'market_data': [],
            'indicators': {},
            'error': f'Error loading chart data: {str(e)}'
        })


@app.route('/start-strategy', methods=['GET', 'POST'])
def start_strategy():
    """Start a new strategy."""
    if request.method == 'GET':
        # Get available configurations
        configs_data = api_client.get('/api/v1/configs/available')
        configs = configs_data.get('configs', []) if 'error' not in configs_data else []
        
        return render_template('start_strategy.html',
                             configs=configs,
                             page_title="Start Strategy")
    
    # Handle POST request to start strategy
    config_path = request.form.get('config_path')
    initial_balance = float(request.form.get('initial_balance', 10000))
    
    if not config_path:
        flash('Please select a configuration', 'error')
        return redirect(url_for('start_strategy'))
    
    # Start the strategy
    result = api_client.post('/api/v1/strategies/start', {
        'config_path': config_path,
        'initial_balance': initial_balance
    })
    
    if 'error' in result:
        flash(f"Error starting strategy: {result['error']}", 'error')
    else:
        flash(f"Strategy started successfully! ID: {result.get('strategy_id', 'Unknown')}", 'success')
        return redirect(url_for('dashboard'))
    
    return redirect(url_for('start_strategy'))


@app.route('/stop-strategy/<strategy_id>', methods=['POST'])
def stop_strategy(strategy_id: str):
    """Stop a running strategy."""
    result = api_client.post(f'/api/v1/strategies/{strategy_id}/stop')
    
    if 'error' in result:
        flash(f"Error stopping strategy: {result['error']}", 'error')
    else:
        flash('Strategy stopped successfully!', 'success')
    
    return redirect(url_for('dashboard'))


# WebSocket events for real-time updates
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info('Client connected to WebSocket')
    emit('status', {'message': 'Connected to live updates'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info('Client disconnected from WebSocket')


@socketio.on('subscribe_strategy')
def handle_subscribe_strategy(data):
    """Subscribe to strategy updates."""
    strategy_id = data.get('strategy_id')
    if strategy_id:
        logger.info(f'Client subscribed to strategy {strategy_id}')
        # Join strategy-specific room for targeted updates
        from flask_socketio import join_room
        join_room(f'strategy_{strategy_id}')


def emit_strategy_update(strategy_id: str, data: Dict):
    """Emit strategy update to subscribed clients."""
    socketio.emit('strategy_update', data, room=f'strategy_{strategy_id}')


# Background task for real-time updates (runs in development mode)
def background_updates():
    """Background task to emit real-time updates."""
    import time
    import threading
    
    def update_loop():
        while True:
            try:
                # Get active strategies
                strategies_data = api_client.get('/api/v1/strategies/active')
                
                if 'strategies' in strategies_data:
                    for strategy in strategies_data['strategies']:
                        strategy_id = strategy['id']
                        
                        # Get updated stats
                        stats_data = api_client.get(f'/api/v1/strategies/{strategy_id}/stats')
                        
                        if 'error' not in stats_data:
                            # Emit update to clients
                            socketio.emit('strategy_update', {
                                'strategy_id': strategy_id,
                                'stats': stats_data
                            }, room=f'strategy_{strategy_id}')
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Background update error: {e}")
                time.sleep(60)  # Wait longer on error
    
    # Start background thread
    thread = threading.Thread(target=update_loop)
    thread.daemon = True
    thread.start()


# Template context processors
def get_system_health():
    """Get system health status."""
    try:
        health_data = api_client.get('/health')
        return 'healthy' if health_data.get('status') == 'healthy' else 'degraded'
    except Exception as e:
        logger.error(f"Error fetching system health: {e}")
        return 'degraded'


@app.context_processor
def inject_system_status():
    """Make system status available to all templates."""
    return dict(system_status=get_system_health())


@app.context_processor
def inject_strategies():
    """Make strategies available to all templates."""
    def get_strategies():
        strategies_data = api_client.get('/api/v1/strategies/active')
        return strategies_data.get('strategies', []) if 'error' not in strategies_data else []
    
    return dict(get_strategies=get_strategies)

@app.context_processor
def inject_config():
    """Make configuration available to all templates."""
    from flask import request
    
    # Get the current host from the request
    current_host = request.host.split(':')[0]  # Remove port if present
    
    return dict(config={
        'API_BASE_URL': FRONTEND_API_URL,  # Use frontend URL for browser JavaScript
        'API_KEY': API_KEY,
        'CURRENT_HOST': current_host
    })

# Template filters
@app.template_filter('datetime')
def datetime_filter(timestamp):
    """Format timestamp for display."""
    if not timestamp:
        return 'N/A'
    
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = timestamp
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(timestamp)


@app.template_filter('currency')
def currency_filter(value):
    """Format currency values."""
    if value is None:
        return '$0.00'
    
    try:
        return f'${float(value):,.2f}'
    except:
        return str(value)


@app.template_filter('percentage')
def percentage_filter(value):
    """Format percentage values."""
    if value is None:
        return '0.00%'
    
    try:
        return f'{float(value):.2f}%'
    except:
        return str(value)


if __name__ == '__main__':
    # Start background updates in development
    if app.debug:
        background_updates()
    
    # Run the Flask app with SocketIO support
    socketio.run(app, 
                host='0.0.0.0', 
                port=int(os.environ.get('FLASK_PORT', 5000)),
                debug=True)