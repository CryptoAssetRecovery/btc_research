/**
 * Advanced Financial Charts for Trading Dashboard
 * Handles OHLC candlestick charts, volume profiles, trade markers, and real-time updates
 */

class AdvancedTradingChart {
    constructor(canvasId, options = {}) {
        this.canvasId = canvasId;
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.chart = null;
        this.options = {
            theme: 'light',
            showVolume: true,
            showTradeMarkers: true,
            showIndicators: true,
            realTimeUpdates: true,
            ...options
        };
        
        this.data = {
            ohlc: [],
            volume: [],
            trades: [],
            indicators: {}
        };
        
        this.plugins = [];
        this.initializePlugins();
        this.createChart();
    }

    initializePlugins() {
        // Register Chart.js plugins
        Chart.register(...ChartjsFinancial.getChartComponents());
        
        // Custom trade markers plugin
        this.tradeMarkersPlugin = {
            id: 'tradeMarkers',
            afterDatasetsDraw: (chart) => {
                if (!this.options.showTradeMarkers || !this.data.trades.length) return;
                this.drawTradeMarkers(chart);
            }
        };
        
        // Custom volume profile plugin
        this.volumeProfilePlugin = {
            id: 'volumeProfile',
            afterDatasetsDraw: (chart) => {
                if (!this.options.showVolume || !this.data.volume.length) return;
                this.drawVolumeProfile(chart);
            }
        };
        
        // Custom crosshair plugin
        this.crosshairPlugin = {
            id: 'crosshair',
            afterDraw: (chart) => {
                this.drawCrosshair(chart);
            }
        };

        this.plugins = [
            this.tradeMarkersPlugin,
            this.volumeProfilePlugin,
            this.crosshairPlugin
        ];
    }

    createChart() {
        const config = {
            type: 'candlestick',
            data: {
                datasets: [
                    {
                        label: 'Price',
                        data: this.data.ohlc,
                        borderColor: {
                            up: '#22c55e',
                            down: '#ef4444',
                            unchanged: '#6b7280'
                        },
                        backgroundColor: {
                            up: 'rgba(34, 197, 94, 0.3)',
                            down: 'rgba(239, 68, 68, 0.3)',
                            unchanged: 'rgba(107, 114, 128, 0.3)'
                        }
                    },
                    {
                        label: 'Volume',
                        type: 'bar',
                        data: this.data.volume,
                        backgroundColor: 'rgba(59, 130, 246, 0.5)',
                        borderColor: '#3b82f6',
                        borderWidth: 1,
                        yAxisID: 'volume',
                        order: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Price Chart with Trade Markers',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            title: (context) => {
                                return new Date(context[0].parsed.x).toLocaleString();
                            },
                            label: (context) => {
                                const data = context.parsed;
                                if (context.dataset.type === 'bar') {
                                    return `Volume: ${formatNumber(data.y)}`;
                                }
                                return [
                                    `Open: ${formatCurrency(data.o)}`,
                                    `High: ${formatCurrency(data.h)}`,
                                    `Low: ${formatCurrency(data.l)}`,
                                    `Close: ${formatCurrency(data.c)}`
                                ];
                            }
                        }
                    },
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'x'
                        },
                        zoom: {
                            wheel: {
                                enabled: true,
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'x'
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            displayFormats: {
                                minute: 'HH:mm',
                                hour: 'MMM DD HH:mm',
                                day: 'MMM DD'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        type: 'linear',
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Price (USD)'
                        },
                        ticks: {
                            callback: function(value) {
                                return formatCurrency(value);
                            }
                        }
                    },
                    volume: {
                        type: 'linear',
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Volume'
                        },
                        max: (context) => {
                            if (!this.data.volume.length) return 1000;
                            const maxVolume = Math.max(...this.data.volume.map(v => v.y));
                            return maxVolume * 4; // Show volume in lower 25% of chart
                        },
                        ticks: {
                            callback: function(value) {
                                return formatNumber(value);
                            }
                        },
                        grid: {
                            display: false
                        }
                    }
                },
                onHover: (event, elements) => {
                    this.canvas.style.cursor = elements.length > 0 ? 'crosshair' : 'default';
                }
            },
            plugins: this.plugins
        };

        this.chart = new Chart(this.ctx, config);
        
        // Add mouse move handler for crosshair
        this.canvas.addEventListener('mousemove', (e) => {
            this.handleMouseMove(e);
        });
    }

    updateData(marketData, trades = [], indicators = {}) {
        if (!marketData || !marketData.length) return;

        // Update OHLC data
        this.data.ohlc = marketData.map(candle => ({
            x: new Date(candle.timestamp).getTime(),
            o: parseFloat(candle.open),
            h: parseFloat(candle.high),
            l: parseFloat(candle.low),
            c: parseFloat(candle.close)
        }));

        // Update volume data
        this.data.volume = marketData.map(candle => ({
            x: new Date(candle.timestamp).getTime(),
            y: parseFloat(candle.volume || 0)
        }));

        // Update trades data
        this.data.trades = trades.map(trade => ({
            x: new Date(trade.timestamp).getTime(),
            y: parseFloat(trade.price),
            size: parseFloat(trade.size || 0),
            side: trade.side,
            pnl: parseFloat(trade.pnl || 0),
            id: trade.id
        }));

        // Update indicators
        this.data.indicators = indicators;

        // Update chart datasets
        this.chart.data.datasets[0].data = this.data.ohlc;
        this.chart.data.datasets[1].data = this.data.volume;

        // Add indicator datasets
        this.addIndicatorDatasets();

        this.chart.update('none'); // Fast update without animation
    }

    addIndicatorDatasets() {
        // Remove existing indicator datasets
        this.chart.data.datasets = this.chart.data.datasets.filter(dataset => 
            !dataset.isIndicator
        );

        if (!this.options.showIndicators) return;

        // Add EMA lines
        if (this.data.indicators.ema) {
            Object.entries(this.data.indicators.ema).forEach(([period, values], index) => {
                this.chart.data.datasets.push({
                    label: `EMA ${period}`,
                    type: 'line',
                    data: values.map((value, i) => ({
                        x: this.data.ohlc[i]?.x,
                        y: value
                    })).filter(point => point.x && point.y),
                    borderColor: this.getIndicatorColor(index),
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1,
                    isIndicator: true
                });
            });
        }

        // Add Volume Profile levels
        if (this.data.indicators.volumeProfile) {
            this.chart.data.datasets.push({
                label: 'Volume Profile',
                type: 'line',
                data: this.data.indicators.volumeProfile.map(level => ({
                    x: level.timestamp,
                    y: level.price
                })),
                borderColor: 'rgba(147, 51, 234, 0.8)',
                backgroundColor: 'rgba(147, 51, 234, 0.1)',
                borderWidth: 3,
                pointRadius: 0,
                stepped: true,
                isIndicator: true
            });
        }

        // Add FVG zones
        if (this.data.indicators.fvg) {
            this.data.indicators.fvg.forEach((zone, index) => {
                this.chart.data.datasets.push({
                    label: `FVG ${index + 1}`,
                    type: 'line',
                    data: [
                        { x: zone.start, y: zone.top },
                        { x: zone.end, y: zone.top },
                        { x: zone.end, y: zone.bottom },
                        { x: zone.start, y: zone.bottom },
                        { x: zone.start, y: zone.top }
                    ],
                    borderColor: 'rgba(249, 115, 22, 0.6)',
                    backgroundColor: 'rgba(249, 115, 22, 0.1)',
                    borderWidth: 1,
                    fill: true,
                    pointRadius: 0,
                    isIndicator: true
                });
            });
        }
    }

    drawTradeMarkers(chart) {
        const ctx = chart.ctx;
        const xScale = chart.scales.x;
        const yScale = chart.scales.y;

        this.data.trades.forEach(trade => {
            const x = xScale.getPixelForValue(trade.x);
            const y = yScale.getPixelForValue(trade.y);

            if (x < xScale.left || x > xScale.right || y < yScale.top || y > yScale.bottom) {
                return; // Trade marker outside visible area
            }

            // Draw trade marker
            ctx.save();
            
            const isBuy = trade.side.toLowerCase() === 'buy';
            const color = isBuy ? '#22c55e' : '#ef4444';
            const symbol = isBuy ? '▲' : '▼';
            
            // Draw marker shape
            ctx.fillStyle = color;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            
            if (isBuy) {
                // Draw triangle pointing up
                this.drawTriangle(ctx, x, y + 15, 8, true);
            } else {
                // Draw triangle pointing down
                this.drawTriangle(ctx, x, y - 15, 8, false);
            }
            
            ctx.fill();
            ctx.stroke();

            // Draw P&L indicator
            if (trade.pnl !== 0) {
                const pnlColor = trade.pnl > 0 ? '#22c55e' : '#ef4444';
                const pnlText = `${trade.pnl > 0 ? '+' : ''}${formatCurrency(trade.pnl)}`;
                
                ctx.fillStyle = pnlColor;
                ctx.font = '10px Arial';
                ctx.textAlign = 'center';
                
                const textY = isBuy ? y + 30 : y - 25;
                ctx.fillText(pnlText, x, textY);
            }

            ctx.restore();
        });
    }

    drawTriangle(ctx, x, y, size, pointUp) {
        ctx.beginPath();
        if (pointUp) {
            ctx.moveTo(x, y - size);
            ctx.lineTo(x - size, y + size);
            ctx.lineTo(x + size, y + size);
        } else {
            ctx.moveTo(x, y + size);
            ctx.lineTo(x - size, y - size);
            ctx.lineTo(x + size, y - size);
        }
        ctx.closePath();
    }

    drawVolumeProfile(chart) {
        if (!this.data.indicators.volumeProfile) return;

        const ctx = chart.ctx;
        const xScale = chart.scales.x;
        const yScale = chart.scales.y;

        ctx.save();
        ctx.fillStyle = 'rgba(147, 51, 234, 0.2)';
        ctx.strokeStyle = 'rgba(147, 51, 234, 0.8)';
        ctx.lineWidth = 1;

        this.data.indicators.volumeProfile.forEach(level => {
            const y = yScale.getPixelForValue(level.price);
            const width = (level.volume / level.maxVolume) * 100; // Scale to max 100px
            
            // Draw volume bar from right edge
            const x = xScale.right - width;
            ctx.fillRect(x, y - 2, width, 4);
            ctx.strokeRect(x, y - 2, width, 4);
        });

        ctx.restore();
    }

    drawCrosshair(chart) {
        if (!this.mousePosition) return;

        const ctx = chart.ctx;
        const canvasPosition = Chart.helpers.getRelativePosition(this.mousePosition, chart);
        
        if (canvasPosition.x < chart.chartArea.left || 
            canvasPosition.x > chart.chartArea.right ||
            canvasPosition.y < chart.chartArea.top || 
            canvasPosition.y > chart.chartArea.bottom) {
            return;
        }

        ctx.save();
        ctx.strokeStyle = 'rgba(107, 114, 128, 0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);

        // Vertical line
        ctx.beginPath();
        ctx.moveTo(canvasPosition.x, chart.chartArea.top);
        ctx.lineTo(canvasPosition.x, chart.chartArea.bottom);
        ctx.stroke();

        // Horizontal line
        ctx.beginPath();
        ctx.moveTo(chart.chartArea.left, canvasPosition.y);
        ctx.lineTo(chart.chartArea.right, canvasPosition.y);
        ctx.stroke();

        ctx.restore();
    }

    handleMouseMove(event) {
        this.mousePosition = event;
        this.chart.update('none');
    }

    getIndicatorColor(index) {
        const colors = [
            '#3b82f6', // Blue
            '#f59e0b', // Amber
            '#10b981', // Emerald
            '#8b5cf6', // Violet
            '#f97316', // Orange
            '#06b6d4', // Cyan
            '#84cc16', // Lime
            '#ec4899'  // Pink
        ];
        return colors[index % colors.length];
    }

    resetZoom() {
        this.chart.resetZoom();
    }

    exportChart(filename = 'chart.png') {
        const link = document.createElement('a');
        link.download = filename;
        link.href = this.canvas.toDataURL();
        link.click();
    }

    setTheme(theme) {
        this.options.theme = theme;
        // Update chart colors based on theme
        this.updateThemeColors();
        this.chart.update();
    }

    updateThemeColors() {
        const isDark = this.options.theme === 'dark';
        
        // Update chart options for theme
        this.chart.options.plugins.title.color = isDark ? '#ffffff' : '#000000';
        this.chart.options.plugins.legend.labels.color = isDark ? '#ffffff' : '#000000';
        this.chart.options.scales.x.title.color = isDark ? '#ffffff' : '#000000';
        this.chart.options.scales.y.title.color = isDark ? '#ffffff' : '#000000';
        this.chart.options.scales.x.ticks.color = isDark ? '#9ca3af' : '#6b7280';
        this.chart.options.scales.y.ticks.color = isDark ? '#9ca3af' : '#6b7280';
    }

    destroy() {
        if (this.chart) {
            this.chart.destroy();
        }
    }
}

// Performance Chart for P&L and metrics
class PerformanceChart {
    constructor(canvasId, options = {}) {
        this.canvasId = canvasId;
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.chart = null;
        this.options = {
            showDrawdown: true,
            showBenchmark: false,
            ...options
        };
        
        this.createChart();
    }

    createChart() {
        const config = {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Cumulative P&L',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1
                    },
                    {
                        label: 'Drawdown',
                        data: [],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1,
                        yAxisID: 'drawdown',
                        hidden: !this.options.showDrawdown
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Performance Analysis',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            title: (context) => {
                                return new Date(context[0].parsed.x).toLocaleString();
                            },
                            label: (context) => {
                                const value = context.parsed.y;
                                if (context.dataset.label === 'Drawdown') {
                                    return `${context.dataset.label}: ${formatPercentage(value)}`;
                                }
                                return `${context.dataset.label}: ${formatCurrency(value)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            displayFormats: {
                                hour: 'MMM DD HH:mm',
                                day: 'MMM DD'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        title: {
                            display: true,
                            text: 'P&L (USD)'
                        },
                        ticks: {
                            callback: function(value) {
                                return formatCurrency(value);
                            }
                        }
                    },
                    drawdown: {
                        type: 'linear',
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Drawdown (%)'
                        },
                        min: -100,
                        max: 0,
                        ticks: {
                            callback: function(value) {
                                return formatPercentage(value);
                            }
                        },
                        grid: {
                            display: false
                        }
                    }
                }
            }
        };

        this.chart = new Chart(this.ctx, config);
    }

    updateData(performanceData) {
        if (!performanceData || !performanceData.length) return;

        // Update P&L data
        const pnlData = performanceData.map(point => ({
            x: new Date(point.timestamp).getTime(),
            y: parseFloat(point.cumulative_pnl || 0)
        }));

        // Calculate drawdown data
        const drawdownData = this.calculateDrawdown(performanceData);

        this.chart.data.datasets[0].data = pnlData;
        this.chart.data.datasets[1].data = drawdownData;

        this.chart.update('none');
    }

    calculateDrawdown(performanceData) {
        let peak = 0;
        return performanceData.map(point => {
            const value = parseFloat(point.cumulative_pnl || 0);
            peak = Math.max(peak, value);
            const drawdown = peak > 0 ? ((value - peak) / peak) * 100 : 0;
            
            return {
                x: new Date(point.timestamp).getTime(),
                y: drawdown
            };
        });
    }

    destroy() {
        if (this.chart) {
            this.chart.destroy();
        }
    }
}

// Export classes for global use
window.AdvancedTradingChart = AdvancedTradingChart;
window.PerformanceChart = PerformanceChart;