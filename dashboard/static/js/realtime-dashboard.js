/**
 * Realtime API Dashboard Extension for Sovereign 4.0
 * 
 * Extends the existing dashboard with Realtime API specific metrics:
 * - Voice-to-voice latency monitoring with P50/P95/P99 percentiles
 * - Connection stability visualization and health monitoring
 * - Real-time cost tracking with hourly projections and alerts
 * - Audio quality metrics and session analytics
 * - WebSocket integration for live updates
 */

class RealtimeDashboardExtension {
    constructor(sovereignDashboard) {
        this.dashboard = sovereignDashboard;
        this.socket = sovereignDashboard.socket;
        
        // Configuration for Realtime API monitoring
        this.config = {
            updateInterval: 1000,  // 1 second for real-time feel
            chartRefreshRate: 500,
            maxDataPoints: 200,    // More data points for Realtime API
            costAlertThreshold: 10.0,  // $10/hour
            latencyAlertThreshold: 500,  // 500ms
            connectionQualityThreshold: 60  // 60/100
        };
        
        // Real-time data storage
        this.realtimeData = {
            latency: [],
            connection: [],
            audio: [],
            cost: [],
            stability: []
        };
        
        // Chart instances for Realtime API metrics
        this.realtimeCharts = {};
        
        // Alert state
        this.activeAlerts = new Map();
        this.lastAlertTime = 0;
        
        // Subscription state
        this.realtimeSubscriptions = [
            'realtime_latency',
            'realtime_connection', 
            'realtime_audio',
            'realtime_cost',
            'connection_stability'
        ];
        
        // Initialize the extension
        this.init();
    }
    
    init() {
        console.log('🚀 Initializing Realtime API Dashboard Extension...');
        
        try {
            // Setup UI components
            this.createRealtimeUI();
            
            // Initialize charts
            this.initializeRealtimeCharts();
            
            // Setup WebSocket handlers
            this.setupRealtimeSocketHandlers();
            
            // Subscribe to Realtime API metrics
            this.subscribeToRealtimeMetrics();
            
            // Start periodic updates
            this.startPeriodicUpdates();
            
            console.log('✅ Realtime API Dashboard Extension initialized');
        } catch (error) {
            console.error('❌ Failed to initialize Realtime API extension:', error);
        }
    }
    
    createRealtimeUI() {
        """Create UI components for Realtime API metrics"""
        
        // Create Realtime API section in the dashboard
        const dashboardContainer = document.querySelector('.dashboard-container') || document.body;
        
        const realtimeSection = document.createElement('div');
        realtimeSection.className = 'realtime-section dashboard-section';
        realtimeSection.innerHTML = `
            <div class="section-header">
                <h2>🎙️ Realtime API Performance</h2>
                <div class="realtime-status">
                    <span class="status-indicator" id="realtime-status">●</span>
                    <span id="realtime-status-text">Connecting...</span>
                </div>
            </div>
            
            <div class="realtime-metrics-grid">
                <!-- Voice-to-Voice Latency Card -->
                <div class="metric-card realtime-latency-card">
                    <div class="card-header">
                        <h3>🗣️ Voice Latency</h3>
                        <div class="card-controls">
                            <span class="current-value" id="current-latency">-</span>
                            <span class="unit">ms</span>
                        </div>
                    </div>
                    <div class="chart-container">
                        <canvas id="realtime-latency-chart"></canvas>
                    </div>
                    <div class="metric-summary">
                        <div class="summary-item">
                            <span class="label">P50:</span>
                            <span class="value" id="latency-p50">-</span>
                        </div>
                        <div class="summary-item">
                            <span class="label">P95:</span>
                            <span class="value" id="latency-p95">-</span>
                        </div>
                        <div class="summary-item">
                            <span class="label">P99:</span>
                            <span class="value" id="latency-p99">-</span>
                        </div>
                    </div>
                </div>
                
                <!-- Connection Health Card -->
                <div class="metric-card connection-health-card">
                    <div class="card-header">
                        <h3>🔗 Connection Health</h3>
                        <div class="card-controls">
                            <span class="quality-badge" id="connection-quality">Unknown</span>
                        </div>
                    </div>
                    <div class="chart-container">
                        <canvas id="connection-stability-chart"></canvas>
                    </div>
                    <div class="metric-summary">
                        <div class="summary-item">
                            <span class="label">Stability:</span>
                            <span class="value" id="stability-score">-</span>
                        </div>
                        <div class="summary-item">
                            <span class="label">Reliability:</span>
                            <span class="value" id="reliability-score">-</span>
                        </div>
                        <div class="summary-item">
                            <span class="label">Network:</span>
                            <span class="value" id="network-score">-</span>
                        </div>
                    </div>
                </div>
                
                <!-- Cost Tracking Card -->
                <div class="metric-card cost-tracking-card">
                    <div class="card-header">
                        <h3>💰 Cost Tracking</h3>
                        <div class="card-controls">
                            <span class="current-value" id="current-cost">$0.00</span>
                            <span class="unit">/session</span>
                        </div>
                    </div>
                    <div class="chart-container">
                        <canvas id="cost-tracking-chart"></canvas>
                    </div>
                    <div class="metric-summary">
                        <div class="summary-item">
                            <span class="label">Hourly:</span>
                            <span class="value" id="projected-hourly-cost">$0.00</span>
                        </div>
                        <div class="summary-item">
                            <span class="label">Tokens:</span>
                            <span class="value" id="total-tokens">0</span>
                        </div>
                        <div class="summary-item">
                            <span class="label">Ratio:</span>
                            <span class="value" id="token-ratio">0:0</span>
                        </div>
                    </div>
                </div>
                
                <!-- Audio Quality Card -->
                <div class="metric-card audio-quality-card">
                    <div class="card-header">
                        <h3>🎵 Audio Quality</h3>
                        <div class="card-controls">
                            <span class="current-value" id="audio-quality">-</span>
                            <span class="unit">%</span>
                        </div>
                    </div>
                    <div class="chart-container">
                        <canvas id="audio-quality-chart"></canvas>
                    </div>
                    <div class="metric-summary">
                        <div class="summary-item">
                            <span class="label">Samples:</span>
                            <span class="value" id="samples-processed">0</span>
                        </div>
                        <div class="summary-item">
                            <span class="label">Processing:</span>
                            <span class="value" id="processing-time">-</span>
                        </div>
                        <div class="summary-item">
                            <span class="label">Buffer:</span>
                            <span class="value" id="buffer-health">-</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Realtime Alerts Section -->
            <div class="realtime-alerts" id="realtime-alerts">
                <h3>🚨 Realtime API Alerts</h3>
                <div class="alerts-container" id="alerts-container">
                    <div class="no-alerts">No active alerts</div>
                </div>
            </div>
        `;
        
        // Insert the realtime section after existing content
        dashboardContainer.appendChild(realtimeSection);
        
        // Add Realtime API specific styles
        this.addRealtimeStyles();
    }
    
    addRealtimeStyles() {
        """Add CSS styles for Realtime API components"""
        
        const style = document.createElement('style');
        style.textContent = `
            .realtime-section {
                margin-top: 2rem;
                padding: 1.5rem;
                background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
                border-radius: 12px;
                border: 1px solid #3e3e5e;
            }
            
            .section-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1.5rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid #3e3e5e;
            }
            
            .realtime-status {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 0.9rem;
            }
            
            .status-indicator {
                font-size: 1.2rem;
                transition: color 0.3s ease;
            }
            
            .status-indicator.connected { color: #10b981; }
            .status-indicator.connecting { color: #f59e0b; animation: pulse 2s infinite; }
            .status-indicator.disconnected { color: #ef4444; }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .realtime-metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2rem;
            }
            
            .metric-card {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                padding: 1.5rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            }
            
            .card-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            }
            
            .card-header h3 {
                margin: 0;
                font-size: 1.1rem;
                color: #e5e7eb;
            }
            
            .card-controls {
                display: flex;
                align-items: baseline;
                gap: 0.25rem;
            }
            
            .current-value {
                font-size: 1.5rem;
                font-weight: bold;
                color: #3b82f6;
            }
            
            .unit {
                font-size: 0.9rem;
                color: #9ca3af;
            }
            
            .quality-badge {
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: bold;
                text-transform: uppercase;
            }
            
            .quality-badge.excellent { background: #10b981; color: white; }
            .quality-badge.good { background: #3b82f6; color: white; }
            .quality-badge.fair { background: #f59e0b; color: white; }
            .quality-badge.poor { background: #ef4444; color: white; }
            .quality-badge.unknown { background: #6b7280; color: white; }
            
            .chart-container {
                height: 200px;
                margin: 1rem 0;
                position: relative;
            }
            
            .metric-summary {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding-top: 1rem;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .summary-item {
                text-align: center;
                flex: 1;
            }
            
            .summary-item .label {
                display: block;
                font-size: 0.8rem;
                color: #9ca3af;
                margin-bottom: 0.25rem;
            }
            
            .summary-item .value {
                display: block;
                font-weight: bold;
                color: #e5e7eb;
            }
            
            .realtime-alerts {
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                border-radius: 8px;
                padding: 1.5rem;
            }
            
            .realtime-alerts h3 {
                margin: 0 0 1rem 0;
                color: #fca5a5;
            }
            
            .alerts-container {
                max-height: 200px;
                overflow-y: auto;
            }
            
            .alert-item {
                background: rgba(239, 68, 68, 0.2);
                border: 1px solid rgba(239, 68, 68, 0.4);
                border-radius: 6px;
                padding: 0.75rem;
                margin-bottom: 0.5rem;
                animation: slideIn 0.3s ease;
            }
            
            .alert-item.warning {
                background: rgba(245, 158, 11, 0.2);
                border-color: rgba(245, 158, 11, 0.4);
            }
            
            .alert-item.info {
                background: rgba(59, 130, 246, 0.2);
                border-color: rgba(59, 130, 246, 0.4);
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateX(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            .no-alerts {
                text-align: center;
                color: #9ca3af;
                font-style: italic;
                padding: 2rem;
            }
            
            /* Mobile responsiveness */
            @media (max-width: 768px) {
                .realtime-metrics-grid {
                    grid-template-columns: 1fr;
                }
                
                .metric-summary {
                    flex-direction: column;
                    gap: 0.5rem;
                }
                
                .summary-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    width: 100%;
                }
                
                .summary-item .label,
                .summary-item .value {
                    display: inline;
                }
            }
        `;
        
        document.head.appendChild(style);
    }
    
    initializeRealtimeCharts() {
        """Initialize Chart.js charts for Realtime API metrics"""
        
        const chartDefaults = {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    type: 'realtime',
                    realtime: {
                        duration: 60000,  // 1 minute window
                        refresh: this.config.chartRefreshRate,
                        delay: 500,
                        onRefresh: (chart) => this.onChartRefresh(chart)
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#9ca3af' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#e5e7eb' }
                }
            }
        };
        
        // Voice-to-Voice Latency Chart
        const latencyCtx = document.getElementById('realtime-latency-chart').getContext('2d');
        this.realtimeCharts.latency = new Chart(latencyCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Voice-to-Voice',
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        data: []
                    },
                    {
                        label: 'P95',
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        data: []
                    }
                ]
            },
            options: {
                ...chartDefaults,
                scales: {
                    ...chartDefaults.scales,
                    y: {
                        ...chartDefaults.scales.y,
                        title: { display: true, text: 'Latency (ms)', color: '#9ca3af' },
                        min: 0,
                        max: 1000
                    }
                }
            }
        });
        
        // Connection Stability Chart
        const stabilityCtx = document.getElementById('connection-stability-chart').getContext('2d');
        this.realtimeCharts.stability = new Chart(stabilityCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Stability Score',
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        data: []
                    },
                    {
                        label: 'Network Score',
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        data: []
                    }
                ]
            },
            options: {
                ...chartDefaults,
                scales: {
                    ...chartDefaults.scales,
                    y: {
                        ...chartDefaults.scales.y,
                        title: { display: true, text: 'Score (0-100)', color: '#9ca3af' },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
        
        // Cost Tracking Chart
        const costCtx = document.getElementById('cost-tracking-chart').getContext('2d');
        this.realtimeCharts.cost = new Chart(costCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Session Cost',
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        data: []
                    },
                    {
                        label: 'Projected Hourly',
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        data: []
                    }
                ]
            },
            options: {
                ...chartDefaults,
                scales: {
                    ...chartDefaults.scales,
                    y: {
                        ...chartDefaults.scales.y,
                        title: { display: true, text: 'Cost (USD)', color: '#9ca3af' },
                        min: 0
                    }
                }
            }
        });
        
        // Audio Quality Chart
        const audioCtx = document.getElementById('audio-quality-chart').getContext('2d');
        this.realtimeCharts.audio = new Chart(audioCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Quality Score',
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        data: []
                    },
                    {
                        label: 'Buffer Health',
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        data: []
                    }
                ]
            },
            options: {
                ...chartDefaults,
                scales: {
                    ...chartDefaults.scales,
                    y: {
                        ...chartDefaults.scales.y,
                        title: { display: true, text: 'Percentage (%)', color: '#9ca3af' },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }
    
    setupRealtimeSocketHandlers() {
        """Setup WebSocket event handlers for Realtime API metrics"""
        
        // Handle Realtime API metric updates
        this.socket.on('realtime_metrics_update', (data) => {
            this.handleRealtimeMetricUpdate(data);
        });
        
        // Handle Realtime API alerts
        this.socket.on('realtime_alert', (alert) => {
            this.handleRealtimeAlert(alert);
        });
        
        // Handle initial metrics snapshot
        this.socket.on('realtime_metrics_snapshot', (data) => {
            this.handleRealtimeSnapshot(data);
        });
        
        // Handle subscription confirmations
        this.socket.on('realtime_subscription_confirmed', (data) => {
            console.log('✅ Realtime metrics subscription confirmed:', data.subscribed_metrics);
            this.updateConnectionStatus('connected');
        });
        
        // Handle connection events
        this.socket.on('connect', () => {
            this.subscribeToRealtimeMetrics();
        });
        
        this.socket.on('disconnect', () => {
            this.updateConnectionStatus('disconnected');
        });
    }
    
    subscribeToRealtimeMetrics() {
        """Subscribe to Realtime API metrics via WebSocket"""
        
        this.socket.emit('subscribe_realtime_metrics', {
            metrics: this.realtimeSubscriptions
        });
        
        this.updateConnectionStatus('connecting');
    }
    
    handleRealtimeMetricUpdate(update) {
        """Handle incoming Realtime API metric updates"""
        
        const { type, data } = update;
        const timestamp = new Date(data.timestamp);
        
        switch (type) {
            case 'realtime_latency':
                this.updateLatencyMetrics(timestamp, data);
                break;
            case 'realtime_connection':
                this.updateConnectionMetrics(timestamp, data);
                break;
            case 'realtime_audio':
                this.updateAudioMetrics(timestamp, data);
                break;
            case 'realtime_cost':
                this.updateCostMetrics(timestamp, data);
                break;
            case 'connection_stability':
                this.updateStabilityMetrics(timestamp, data);
                break;
        }
        
        // Update last update time
        this.updateLastUpdateTime();
    }
    
    updateLatencyMetrics(timestamp, data) {
        """Update latency metrics and charts"""
        
        // Update current value display
        document.getElementById('current-latency').textContent = `${data.voice_to_voice_ms.toFixed(1)}`;
        document.getElementById('latency-p50').textContent = `${data.p50.toFixed(1)}ms`;
        document.getElementById('latency-p95').textContent = `${data.p95.toFixed(1)}ms`;
        document.getElementById('latency-p99').textContent = `${data.p99.toFixed(1)}ms`;
        
        // Add data to chart
        if (this.realtimeCharts.latency) {
            this.realtimeCharts.latency.data.datasets[0].data.push({
                x: timestamp,
                y: data.voice_to_voice_ms
            });
            this.realtimeCharts.latency.data.datasets[1].data.push({
                x: timestamp,
                y: data.p95
            });
        }
        
        // Check for alerts
        if (data.voice_to_voice_ms > this.config.latencyAlertThreshold) {
            this.addAlert('high_latency', 'warning', 
                `High latency detected: ${data.voice_to_voice_ms.toFixed(1)}ms`);
        }
    }
    
    updateConnectionMetrics(timestamp, data) {
        """Update connection metrics"""
        
        // Update connection state display
        const statusBadge = document.getElementById('connection-quality');
        const connectionState = data.connection_state;
        
        if (statusBadge) {
            statusBadge.textContent = connectionState;
            statusBadge.className = `quality-badge ${connectionState.toLowerCase()}`;
        }
    }
    
    updateAudioMetrics(timestamp, data) {
        """Update audio quality metrics"""
        
        // Update display values
        document.getElementById('audio-quality').textContent = `${data.quality_score.toFixed(1)}`;
        document.getElementById('samples-processed').textContent = data.samples_processed.toLocaleString();
        document.getElementById('processing-time').textContent = `${data.processing_time_ms.toFixed(1)}ms`;
        document.getElementById('buffer-health').textContent = `${data.buffer_health.toFixed(1)}%`;
        
        // Add data to chart
        if (this.realtimeCharts.audio) {
            this.realtimeCharts.audio.data.datasets[0].data.push({
                x: timestamp,
                y: data.quality_score
            });
            this.realtimeCharts.audio.data.datasets[1].data.push({
                x: timestamp,
                y: data.buffer_health
            });
        }
    }
    
    updateCostMetrics(timestamp, data) {
        """Update cost tracking metrics"""
        
        // Update display values
        document.getElementById('current-cost').textContent = `$${data.session_cost_usd.toFixed(4)}`;
        document.getElementById('projected-hourly-cost').textContent = `$${data.projected_hourly_cost.toFixed(2)}`;
        document.getElementById('total-tokens').textContent = data.total_tokens.toLocaleString();
        document.getElementById('token-ratio').textContent = `${data.input_tokens}:${data.output_tokens}`;
        
        // Add data to chart
        if (this.realtimeCharts.cost) {
            this.realtimeCharts.cost.data.datasets[0].data.push({
                x: timestamp,
                y: data.session_cost_usd
            });
            this.realtimeCharts.cost.data.datasets[1].data.push({
                x: timestamp,
                y: data.projected_hourly_cost
            });
        }
        
        // Check for cost alerts
        if (data.projected_hourly_cost > this.config.costAlertThreshold) {
            this.addAlert('high_cost', 'critical', 
                `High cost projection: $${data.projected_hourly_cost.toFixed(2)}/hour`);
        }
    }
    
    updateStabilityMetrics(timestamp, data) {
        """Update connection stability metrics"""
        
        // Update display values
        document.getElementById('stability-score').textContent = `${data.stability_score.toFixed(1)}`;
        document.getElementById('reliability-score').textContent = `${data.reliability_score.toFixed(1)}`;
        document.getElementById('network-score').textContent = `${data.network_score.toFixed(1)}`;
        
        // Update quality badge
        const qualityBadge = document.getElementById('connection-quality');
        if (qualityBadge && data.connection_quality) {
            qualityBadge.textContent = data.connection_quality;
            qualityBadge.className = `quality-badge ${data.connection_quality.toLowerCase()}`;
        }
        
        // Add data to chart
        if (this.realtimeCharts.stability) {
            this.realtimeCharts.stability.data.datasets[0].data.push({
                x: timestamp,
                y: data.stability_score
            });
            this.realtimeCharts.stability.data.datasets[1].data.push({
                x: timestamp,
                y: data.network_score
            });
        }
        
        // Check for connection quality alerts
        if (data.stability_score < this.config.connectionQualityThreshold) {
            this.addAlert('low_connection_quality', 'warning', 
                `Low connection quality: ${data.stability_score.toFixed(1)}/100`);
        }
    }
    
    handleRealtimeAlert(alert) {
        """Handle incoming Realtime API alerts"""
        
        console.warn('🚨 Realtime Alert:', alert);
        this.addAlert(alert.type, alert.severity, alert.message, alert.timestamp);
    }
    
    addAlert(type, severity, message, timestamp = null) {
        """Add an alert to the alerts container"""
        
        const alertsContainer = document.getElementById('alerts-container');
        const noAlertsMsg = alertsContainer.querySelector('.no-alerts');
        
        // Remove "no alerts" message
        if (noAlertsMsg) {
            noAlertsMsg.style.display = 'none';
        }
        
        // Create alert element
        const alertElement = document.createElement('div');
        alertElement.className = `alert-item ${severity}`;
        alertElement.innerHTML = `
            <div class="alert-header">
                <strong>${type.replace(/_/g, ' ').toUpperCase()}</strong>
                <span class="alert-time">${timestamp || new Date().toLocaleTimeString()}</span>
            </div>
            <div class="alert-message">${message}</div>
        `;
        
        // Add to container
        alertsContainer.appendChild(alertElement);
        
        // Store alert
        this.activeAlerts.set(type, {
            severity,
            message,
            timestamp: timestamp || new Date().toISOString(),
            element: alertElement
        });
        
        // Auto-remove after 30 seconds for non-critical alerts
        if (severity !== 'critical') {
            setTimeout(() => {
                if (alertElement.parentNode) {
                    alertElement.remove();
                    this.activeAlerts.delete(type);
                    
                    // Show "no alerts" message if no alerts remain
                    if (this.activeAlerts.size === 0 && noAlertsMsg) {
                        noAlertsMsg.style.display = 'block';
                    }
                }
            }, 30000);
        }
    }
    
    updateConnectionStatus(status) {
        """Update the connection status indicator"""
        
        const indicator = document.getElementById('realtime-status');
        const statusText = document.getElementById('realtime-status-text');
        
        if (indicator) {
            indicator.className = `status-indicator ${status}`;
        }
        
        if (statusText) {
            const statusTexts = {
                connected: 'Connected',
                connecting: 'Connecting...',
                disconnected: 'Disconnected'
            };
            statusText.textContent = statusTexts[status] || status;
        }
    }
    
    updateLastUpdateTime() {
        """Update the last update timestamp"""
        this.lastAlertTime = Date.now();
    }
    
    startPeriodicUpdates() {
        """Start periodic updates for non-real-time data"""
        
        setInterval(() => {
            // Request configuration updates
            this.socket.emit('get_realtime_config');
            
            // Clean up old chart data
            this.cleanupChartData();
            
        }, 30000);  // Every 30 seconds
    }
    
    cleanupChartData() {
        """Remove old data points from charts to prevent memory bloat"""
        
        const cutoffTime = Date.now() - (60 * 1000 * 5);  // 5 minutes ago
        
        Object.values(this.realtimeCharts).forEach(chart => {
            if (!chart || !chart.data || !chart.data.datasets) return;
            
            chart.data.datasets.forEach(dataset => {
                if (dataset.data) {
                    dataset.data = dataset.data.filter(point => 
                        new Date(point.x).getTime() > cutoffTime
                    );
                }
            });
        });
    }
    
    onChartRefresh(chart) {
        """Called when charts refresh - can be used for custom logic"""
        // This method can be extended for custom chart refresh behavior
    }
}

// Auto-initialize when the main dashboard is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for the main dashboard to initialize
    const checkForDashboard = setInterval(() => {
        if (window.sovereignDashboard && window.sovereignDashboard.socket) {
            clearInterval(checkForDashboard);
            
            // Initialize Realtime API extension
            window.realtimeDashboardExtension = new RealtimeDashboardExtension(window.sovereignDashboard);
            
            console.log('🎙️ Realtime API Dashboard Extension ready');
        }
    }, 100);
});

// Export for use in other modules
window.RealtimeDashboardExtension = RealtimeDashboardExtension; 