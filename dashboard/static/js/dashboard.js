/**
 * Sovereign 4.0 Dashboard - Interactive JavaScript
 * 
 * Features:
 * - Chart.js integration for real-time metrics visualization
 * - Socket.IO WebSocket communication for live updates
 * - Responsive UI management and mobile navigation
 * - Performance metrics tracking and display
 * - Anomaly detection and alerting
 * - Progressive disclosure for mobile devices
 */

class SovereignDashboard {
    constructor() {
        // Configuration
        this.config = {
            chartRefreshRate: 500, // ms
            reconnectAttempts: 5,
            reconnectDelay: 3000,
            maxDataPoints: 100,
            alertDisplayTime: 10000
        };

        // State management
        this.isConnected = false;
        this.reconnectCount = 0;
        this.charts = {};
        this.expandedCards = new Set();
        this.metricSubscriptions = ['latency', 'accuracy', 'resource', 'throughput'];
        this.timeRange = 3600; // 1 hour default
        
        // Data storage for aggregation
        this.metricsData = {
            latency: [],
            accuracy: [],
            resource: [],
            throughput: []
        };
        
        // Socket.IO instance
        this.socket = null;
        
        // Chart color palette
        this.chartColors = {
            primary: '#3b82f6',
            success: '#10b981',
            warning: '#f59e0b',
            danger: '#ef4444',
            info: '#06b6d4',
            purple: '#8b5cf6'
        };

        // Initialize dashboard
        this.init();
    }

    async init() {
        console.log('🎤 Initializing Sovereign 4.0 Dashboard...');
        
        try {
            // Setup UI components
            this.setupEventListeners();
            this.updateTimeDisplay();
            this.startTimeDisplayUpdater();
            
            // Initialize charts
            this.initializeCharts();
            
            // Connect to server
            await this.connectWebSocket();
            
            // Hide loading overlay
            this.hideLoading();
            
            console.log('✅ Dashboard initialization complete');
        } catch (error) {
            console.error('❌ Dashboard initialization failed:', error);
            this.showError('Failed to initialize dashboard');
        }
    }

    // ================================
    // WebSocket Communication
    // ================================
    
    async connectWebSocket() {
        console.log('🔌 Connecting to WebSocket...');
        
        try {
            // Initialize Socket.IO connection
            this.socket = io({
                autoConnect: true,
                reconnection: true,
                reconnectionAttempts: this.config.reconnectAttempts,
                reconnectionDelay: this.config.reconnectDelay,
                timeout: 10000
            });

            // Setup event handlers
            this.setupWebSocketHandlers();
            
        } catch (error) {
            console.error('❌ WebSocket connection failed:', error);
            this.updateConnectionStatus('disconnected');
            throw error;
        }
    }

    setupWebSocketHandlers() {
        // Connection events
        this.socket.on('connect', () => {
            console.log('✅ WebSocket connected');
            this.isConnected = true;
            this.reconnectCount = 0;
            this.updateConnectionStatus('connected');
            
            // Subscribe to metrics
            this.socket.emit('subscribe_metrics', {
                metrics: this.metricSubscriptions
            });
        });

        this.socket.on('disconnect', (reason) => {
            console.log('🔌 WebSocket disconnected:', reason);
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');
        });

        this.socket.on('connect_error', (error) => {
            console.error('❌ WebSocket connection error:', error);
            this.reconnectCount++;
            this.updateConnectionStatus('connecting');
        });

        // Data events
        this.socket.on('connection_status', (data) => {
            console.log('📊 Connection status:', data);
        });

        this.socket.on('metrics_snapshot', (data) => {
            console.log('📸 Initial metrics snapshot received');
            this.processMetricsSnapshot(data);
        });

        this.socket.on('metrics_update', (data) => {
            this.processMetricsUpdate(data);
        });

        this.socket.on('anomaly_alert', (alert) => {
            console.log('🚨 Anomaly alert received:', alert);
            this.displayAlert(alert);
        });

        this.socket.on('subscription_confirmed', (data) => {
            console.log('✅ Metrics subscription confirmed:', data.subscribed_metrics);
        });

        // Heartbeat
        setInterval(() => {
            if (this.isConnected) {
                this.socket.emit('ping');
            }
        }, 30000);

        this.socket.on('pong', (data) => {
            this.updateLastPingTime(data.timestamp);
        });
    }

    // ================================
    // Chart Initialization
    // ================================
    
    initializeCharts() {
        console.log('📊 Initializing charts...');
        
        // Register Chart.js plugins
        Chart.register(
            Chart.LinearScale,
            Chart.TimeScale,
            Chart.LineElement,
            Chart.PointElement,
            Chart.LineController,
            Chart.Title,
            Chart.Tooltip,
            Chart.Legend
        );

        // Initialize latency chart
        this.initLatencyChart();
        
        // Initialize accuracy chart
        this.initAccuracyChart();
        
        // Initialize resource chart
        this.initResourceChart();
        
        // Initialize throughput chart
        this.initThroughputChart();
    }

    initLatencyChart() {
        const ctx = document.getElementById('latencyChart');
        if (!ctx) return;

        this.charts.latency = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'P50 Latency',
                        borderColor: this.chartColors.success,
                        backgroundColor: this.chartColors.success + '20',
                        data: [],
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    },
                    {
                        label: 'P95 Latency',
                        borderColor: this.chartColors.warning,
                        backgroundColor: this.chartColors.warning + '20',
                        data: [],
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    },
                    {
                        label: 'P99 Latency',
                        borderColor: this.chartColors.danger,
                        backgroundColor: this.chartColors.danger + '20',
                        data: [],
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Pipeline Latency Over Time',
                        color: '#f8fafc'
                    },
                    legend: {
                        labels: {
                            color: '#cbd5e1'
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: '#1e293b',
                        titleColor: '#f8fafc',
                        bodyColor: '#cbd5e1',
                        borderColor: '#475569',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            displayFormats: {
                                second: 'HH:mm:ss',
                                minute: 'HH:mm',
                                hour: 'MMM dd HH:mm'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Time',
                            color: '#cbd5e1'
                        },
                        ticks: {
                            color: '#64748b'
                        },
                        grid: {
                            color: '#334155'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Latency (ms)',
                            color: '#cbd5e1'
                        },
                        ticks: {
                            color: '#64748b'
                        },
                        grid: {
                            color: '#334155'
                        }
                    }
                },
                interaction: {
                    mode: 'index',
                    intersect: false
                }
            }
        });
    }

    initAccuracyChart() {
        const ctx = document.getElementById('accuracyChart');
        if (!ctx) return;

        this.charts.accuracy = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'STT Accuracy',
                        borderColor: this.chartColors.primary,
                        backgroundColor: this.chartColors.primary + '20',
                        data: [],
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0
                    },
                    {
                        label: 'Memory Recall',
                        borderColor: this.chartColors.purple,
                        backgroundColor: this.chartColors.purple + '20',
                        data: [],
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0
                    },
                    {
                        label: 'LLM Quality',
                        borderColor: this.chartColors.info,
                        backgroundColor: this.chartColors.info + '20',
                        data: [],
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Accuracy Metrics Over Time',
                        color: '#f8fafc'
                    },
                    legend: {
                        labels: {
                            color: '#cbd5e1'
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        ticks: { color: '#64748b' },
                        grid: { color: '#334155' }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Accuracy (%)',
                            color: '#cbd5e1'
                        },
                        min: 0,
                        max: 100,
                        ticks: { color: '#64748b' },
                        grid: { color: '#334155' }
                    }
                }
            }
        });
    }

    initResourceChart() {
        const ctx = document.getElementById('resourceChart');
        if (!ctx) return;

        this.charts.resource = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'CPU %',
                        borderColor: this.chartColors.danger,
                        backgroundColor: this.chartColors.danger + '20',
                        data: [],
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0
                    },
                    {
                        label: 'Memory %',
                        borderColor: this.chartColors.warning,
                        backgroundColor: this.chartColors.warning + '20',
                        data: [],
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0
                    },
                    {
                        label: 'GPU %',
                        borderColor: this.chartColors.success,
                        backgroundColor: this.chartColors.success + '20',
                        data: [],
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'System Resource Usage',
                        color: '#f8fafc'
                    },
                    legend: {
                        labels: {
                            color: '#cbd5e1'
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        ticks: { color: '#64748b' },
                        grid: { color: '#334155' }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Usage (%)',
                            color: '#cbd5e1'
                        },
                        min: 0,
                        max: 100,
                        ticks: { color: '#64748b' },
                        grid: { color: '#334155' }
                    }
                }
            }
        });
    }

    initThroughputChart() {
        const ctx = document.getElementById('throughputChart');
        if (!ctx) return;

        this.charts.throughput = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Requests/sec',
                        borderColor: this.chartColors.primary,
                        backgroundColor: this.chartColors.primary + '20',
                        data: [],
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0
                    },
                    {
                        label: 'Success Rate %',
                        borderColor: this.chartColors.success,
                        backgroundColor: this.chartColors.success + '20',
                        data: [],
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'System Throughput',
                        color: '#f8fafc'
                    },
                    legend: {
                        labels: {
                            color: '#cbd5e1'
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        ticks: { color: '#64748b' },
                        grid: { color: '#334155' }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Requests/sec',
                            color: '#cbd5e1'
                        },
                        ticks: { color: '#64748b' },
                        grid: { color: '#334155' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Success Rate (%)',
                            color: '#cbd5e1'
                        },
                        min: 0,
                        max: 100,
                        ticks: { color: '#64748b' },
                        grid: {
                            drawOnChartArea: false,
                            color: '#334155'
                        }
                    }
                }
            }
        });
    }

    // ================================
    // Data Processing
    // ================================
    
    processMetricsSnapshot(data) {
        console.log('📸 Processing metrics snapshot');
        
        if (data.metrics) {
            this.updateDashboardSummary(data.metrics);
            this.updateComponentMetrics(data.metrics);
        }
    }

    processMetricsUpdate(update) {
        const timestamp = new Date(update.data.timestamp);
        
        // Update charts based on type
        switch (update.type) {
            case 'latency':
                this.updateLatencyChart(timestamp, update.data.latency);
                this.updateLatencySummary(update.data.latency);
                break;
            case 'accuracy':
                this.updateAccuracyChart(timestamp, update.data.accuracy);
                this.updateAccuracySummary(update.data.accuracy);
                break;
            case 'resource':
                this.updateResourceChart(timestamp, update.data.resource);
                this.updateResourceSummary(update.data.resource);
                break;
            case 'throughput':
                this.updateThroughputChart(timestamp, update.data.throughput);
                this.updateThroughputSummary(update.data.throughput);
                break;
        }
        
        // Update component metrics
        this.updateComponentMetrics(update.data);
        
        // Update last update time
        this.updateLastUpdateTime();
    }

    updateLatencyChart(timestamp, latencyData) {
        if (!this.charts.latency || !latencyData) return;
        
        // Calculate overall pipeline latency (average of all components)
        const components = Object.keys(latencyData);
        if (components.length === 0) return;
        
        const p50Values = components.map(comp => latencyData[comp].p50).filter(v => v > 0);
        const p95Values = components.map(comp => latencyData[comp].p95).filter(v => v > 0);
        const p99Values = components.map(comp => latencyData[comp].p99).filter(v => v > 0);
        
        if (p50Values.length > 0) {
            const avgP50 = p50Values.reduce((a, b) => a + b, 0) / p50Values.length;
            const avgP95 = p95Values.reduce((a, b) => a + b, 0) / p95Values.length;
            const avgP99 = p99Values.reduce((a, b) => a + b, 0) / p99Values.length;
            
            this.addDataPoint(this.charts.latency, timestamp, [avgP50, avgP95, avgP99]);
        }
    }

    updateAccuracyChart(timestamp, accuracyData) {
        if (!this.charts.accuracy || !accuracyData) return;
        
        const sttAccuracy = accuracyData.stt_transcription?.mean_score || 0;
        const memoryRecall = accuracyData.memory_recall?.mean_score || 0;
        const llmQuality = accuracyData.llm_response_quality?.mean_score || 0;
        
        this.addDataPoint(this.charts.accuracy, timestamp, [sttAccuracy, memoryRecall, llmQuality]);
    }

    updateResourceChart(timestamp, resourceData) {
        if (!this.charts.resource || !resourceData) return;
        
        const cpu = resourceData.cpu_percent || 0;
        const memory = resourceData.memory_percent || 0;
        const gpu = resourceData.gpu_percent || 0;
        
        this.addDataPoint(this.charts.resource, timestamp, [cpu, memory, gpu]);
    }

    updateThroughputChart(timestamp, throughputData) {
        if (!this.charts.throughput || !throughputData) return;
        
        // Get overall throughput metrics
        const components = Object.keys(throughputData);
        if (components.length === 0) return;
        
        const totalRps = components.reduce((sum, comp) => 
            sum + (throughputData[comp].requests_per_second || 0), 0);
        const avgSuccessRate = components.reduce((sum, comp) => 
            sum + (throughputData[comp].success_rate || 0), 0) / components.length;
        
        this.addDataPoint(this.charts.throughput, timestamp, [totalRps, avgSuccessRate]);
    }

    addDataPoint(chart, timestamp, values) {
        values.forEach((value, index) => {
            if (chart.data.datasets[index]) {
                chart.data.datasets[index].data.push({
                    x: timestamp,
                    y: value
                });
                
                // Limit data points
                if (chart.data.datasets[index].data.length > this.config.maxDataPoints) {
                    chart.data.datasets[index].data.shift();
                }
            }
        });
        
        chart.update('none');
    }

    // ================================
    // UI Updates
    // ================================
    
    updateLatencySummary(latencyData) {
        if (!latencyData) return;
        
        // Calculate overall pipeline latency
        const overallLatency = latencyData.overall_pipeline || 
            Object.values(latencyData)[0] || { p50: 0, p95: 0, p99: 0 };
        
        this.updateElement('latencyPrimary', `${overallLatency.p50?.toFixed(0) || '--'}ms`);
        this.updateElement('latencyP95', overallLatency.p95?.toFixed(0) || '--');
        this.updateElement('latencyP99', overallLatency.p99?.toFixed(0) || '--');
    }

    updateAccuracySummary(accuracyData) {
        if (!accuracyData) return;
        
        const sttAccuracy = accuracyData.stt_transcription?.mean_score || 0;
        const memoryAccuracy = accuracyData.memory_recall?.mean_score || 0;
        
        // Calculate overall accuracy as weighted average
        const overallAccuracy = (sttAccuracy + memoryAccuracy) / 2;
        
        this.updateElement('accuracyPrimary', `${overallAccuracy.toFixed(0)}%`);
        this.updateElement('sttAccuracy', sttAccuracy.toFixed(0));
        this.updateElement('memoryAccuracy', memoryAccuracy.toFixed(0));
    }

    updateResourceSummary(resourceData) {
        if (!resourceData) return;
        
        const cpu = resourceData.cpu_percent || 0;
        const memory = resourceData.memory_percent || 0;
        const gpu = resourceData.gpu_percent || 0;
        
        // Update gauge displays
        this.updateGauge('cpu', cpu);
        this.updateGauge('memory', memory);
        this.updateGauge('gpu', gpu);
    }

    updateThroughputSummary(throughputData) {
        if (!throughputData) return;
        
        // Calculate totals across all components
        const components = Object.values(throughputData);
        const totalRps = components.reduce((sum, comp) => sum + (comp.requests_per_second || 0), 0);
        const avgSuccessRate = components.reduce((sum, comp) => sum + (comp.success_rate || 0), 0) / components.length;
        const totalRequests = components.reduce((sum, comp) => sum + (comp.total_requests || 0), 0);
        
        this.updateElement('throughputPrimary', `${totalRps.toFixed(1)} req/s`);
        this.updateElement('successRate', avgSuccessRate.toFixed(1));
        this.updateElement('totalRequests', totalRequests.toString());
    }

    updateGauge(type, percentage) {
        const gauge = document.getElementById(`${type}Gauge`);
        const value = document.getElementById(`${type}Value`);
        
        if (gauge) {
            gauge.style.width = `${Math.min(percentage, 100)}%`;
            
            // Color coding
            if (percentage > 80) {
                gauge.style.background = 'var(--danger-color)';
            } else if (percentage > 60) {
                gauge.style.background = 'var(--warning-color)';
            } else {
                gauge.style.background = 'var(--success-color)';
            }
        }
        
        if (value) {
            value.textContent = `${percentage.toFixed(1)}%`;
        }
    }

    updateComponentMetrics(data) {
        // Update individual component metrics
        if (data.latency) {
            Object.entries(data.latency).forEach(([component, metrics]) => {
                this.updateComponentLatency(component, metrics);
            });
        }
        
        if (data.accuracy) {
            Object.entries(data.accuracy).forEach(([metric, data]) => {
                this.updateComponentAccuracy(metric, data);
            });
        }
    }

    updateComponentLatency(component, metrics) {
        const elementMap = {
            'audio_capture': 'audioCaptureLatency',
            'stt_processing': 'sttLatency',
            'llm_inference': 'llmLatency',
            'tts_generation': 'ttsLatency',
            'memory_retrieval': 'memoryLatency',
            'ocr_processing': 'ocrLatency'
        };
        
        const elementId = elementMap[component];
        if (elementId) {
            this.updateElement(elementId, `${metrics.p50?.toFixed(0) || '--'}ms`);
        }
    }

    updateComponentAccuracy(metric, data) {
        const elementMap = {
            'stt_transcription': 'sttConfidence',
            'memory_recall': 'memoryBleu',
            'llm_response_quality': 'llmQuality',
            'tts_generation': 'ttsQuality',
            'ocr_processing': 'ocrAccuracy'
        };
        
        const elementId = elementMap[metric];
        if (elementId) {
            this.updateElement(elementId, `${data.mean_score?.toFixed(1) || '--'}%`);
        }
    }

    updateDashboardSummary(metricsData) {
        // This would be called with the comprehensive metrics data
        // Implementation depends on the structure of metricsData
        console.log('📊 Updating dashboard summary with:', metricsData);
    }

    updateConnectionStatus(status) {
        const indicator = document.getElementById('statusIndicator');
        const text = document.getElementById('statusText');
        
        if (indicator) {
            indicator.className = `status-indicator ${status}`;
        }
        
        if (text) {
            const statusTexts = {
                connected: 'Connected',
                disconnected: 'Disconnected',
                connecting: 'Connecting...'
            };
            text.textContent = statusTexts[status] || status;
        }
    }

    updateSystemHealth(status, issues = []) {
        const indicator = document.getElementById('healthIndicator');
        const text = document.getElementById('healthText');
        
        if (indicator) {
            indicator.className = `health-indicator ${status}`;
        }
        
        if (text) {
            const statusTexts = {
                healthy: 'Healthy',
                warning: 'Warning',
                error: 'Error',
                unknown: 'Unknown'
            };
            text.textContent = statusTexts[status] || status;
            
            if (issues.length > 0) {
                text.title = issues.join('\n');
            }
        }
    }

    updateTimeDisplay() {
        const timeElement = document.getElementById('timeDisplay');
        if (timeElement) {
            const now = new Date();
            timeElement.textContent = now.toLocaleTimeString();
        }
    }

    updateLastUpdateTime() {
        const lastUpdateElement = document.getElementById('lastUpdate');
        if (lastUpdateElement) {
            const now = new Date();
            lastUpdateElement.textContent = now.toLocaleTimeString();
        }
    }

    updateLastPingTime(timestamp) {
        // Could show network latency here
        console.log('📡 Ping received at:', timestamp);
    }

    // ================================
    // Alert Management
    // ================================
    
    displayAlert(alert) {
        const alertsContainer = document.getElementById('alertsContainer');
        const noAlertsMessage = document.getElementById('noAlertsMessage');
        
        if (!alertsContainer) return;
        
        // Hide "no alerts" message
        if (noAlertsMessage) {
            noAlertsMessage.style.display = 'none';
        }
        
        // Create alert element
        const alertElement = document.createElement('div');
        alertElement.className = `alert-item ${alert.severity}`;
        alertElement.innerHTML = `
            <div class="alert-header">
                <span class="alert-type">${alert.type.replace('_', ' ')}</span>
                <span class="alert-timestamp">${new Date(alert.timestamp).toLocaleTimeString()}</span>
            </div>
            <div class="alert-message">
                ${this.formatAlertMessage(alert)}
            </div>
        `;
        
        // Add to container
        alertsContainer.insertBefore(alertElement, alertsContainer.firstChild);
        
        // Auto-remove after timeout
        setTimeout(() => {
            if (alertElement.parentNode) {
                alertElement.remove();
                
                // Show "no alerts" message if no alerts remain
                if (alertsContainer.children.length === 0 || 
                    (alertsContainer.children.length === 1 && noAlertsMessage)) {
                    if (noAlertsMessage) {
                        noAlertsMessage.style.display = 'block';
                    }
                }
            }
        }, this.config.alertDisplayTime);
    }

    formatAlertMessage(alert) {
        const data = alert.data;
        
        switch (alert.type) {
            case 'latency_anomaly':
                return `High latency detected in ${data.component}: P95 = ${data.p95_latency?.toFixed(1)}ms (threshold: ${(data.mean_latency * data.threshold_multiplier)?.toFixed(1)}ms)`;
            
            case 'accuracy_degradation':
                return `Accuracy drop in ${data.metric}: ${data.recent_accuracy?.toFixed(1)}% (down ${data.drop_amount?.toFixed(1)}% from baseline)`;
            
            case 'high_cpu_usage':
                return `High CPU usage: ${data.cpu_percent?.toFixed(1)}% (threshold: ${data.threshold}%)`;
            
            case 'high_memory_usage':
                return `High memory usage: ${data.memory_percent?.toFixed(1)}% (${data.memory_used_gb?.toFixed(1)}GB used)`;
            
            default:
                return JSON.stringify(data, null, 2);
        }
    }

    clearAlerts() {
        const alertsContainer = document.getElementById('alertsContainer');
        const noAlertsMessage = document.getElementById('noAlertsMessage');
        
        if (alertsContainer) {
            // Remove all alert items
            const alertItems = alertsContainer.querySelectorAll('.alert-item');
            alertItems.forEach(item => item.remove());
            
            // Show "no alerts" message
            if (noAlertsMessage) {
                noAlertsMessage.style.display = 'block';
            }
        }
    }

    // ================================
    // UI Interaction Handlers
    // ================================
    
    setupEventListeners() {
        // Card expansion/collapse
        window.toggleCard = (cardId) => {
            const card = document.getElementById(cardId);
            if (card) {
                if (this.expandedCards.has(cardId)) {
                    card.classList.remove('expanded');
                    this.expandedCards.delete(cardId);
                } else {
                    card.classList.add('expanded');
                    this.expandedCards.add(cardId);
                }
            }
        };

        // Section toggle
        window.toggleSection = (sectionId) => {
            const section = document.getElementById(sectionId);
            if (section) {
                const content = section.querySelector('.component-grid');
                if (content) {
                    content.style.display = content.style.display === 'none' ? 'grid' : 'none';
                }
            }
        };

        // Time range update
        window.updateTimeRange = () => {
            const select = document.getElementById('timeRange');
            if (select) {
                this.timeRange = parseInt(select.value);
                this.refreshChartData();
            }
        };

        // Metric display update
        window.updateMetricDisplay = () => {
            const checkboxes = {
                latency: document.getElementById('showLatency'),
                accuracy: document.getElementById('showAccuracy'),
                resource: document.getElementById('showResource'),
                throughput: document.getElementById('showThroughput')
            };

            Object.entries(checkboxes).forEach(([metric, checkbox]) => {
                if (checkbox) {
                    const card = document.getElementById(`${metric}Card`);
                    if (card) {
                        card.style.display = checkbox.checked ? 'block' : 'none';
                    }
                }
            });
        };

        // Mobile navigation
        window.openMobileNav = () => {
            const overlay = document.getElementById('mobileNavOverlay');
            if (overlay) {
                overlay.classList.add('active');
            }
        };

        window.closeMobileNav = () => {
            const overlay = document.getElementById('mobileNavOverlay');
            if (overlay) {
                overlay.classList.remove('active');
            }
        };

        window.scrollToSection = (sectionId) => {
            const section = document.querySelector(`.${sectionId}`);
            if (section) {
                section.scrollIntoView({ behavior: 'smooth' });
                this.closeMobileNav();
            }
        };

        // Clear alerts
        window.clearAlerts = () => {
            this.clearAlerts();
        };

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeMobileNav();
            }
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            Object.values(this.charts).forEach(chart => {
                if (chart) {
                    chart.resize();
                }
            });
        });
    }

    // ================================
    // Utility Methods
    // ================================
    
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.add('hidden');
            setTimeout(() => {
                overlay.remove();
            }, 500);
        }
    }

    showError(message) {
        console.error('📊 Dashboard error:', message);
        // Could implement toast notifications here
    }

    startTimeDisplayUpdater() {
        setInterval(() => {
            this.updateTimeDisplay();
        }, 1000);
    }

    refreshChartData() {
        // Could fetch historical data based on new time range
        console.log('📊 Refreshing chart data for time range:', this.timeRange);
    }

    async fetchSystemStatus() {
        try {
            const response = await fetch('/api/system/status');
            const data = await response.json();
            
            this.updateSystemHealth(data.status, data.issues || []);
            this.updateElement('connectedClients', data.connected_clients || 0);
            
        } catch (error) {
            console.error('❌ Failed to fetch system status:', error);
        }
    }
}

// ================================
// Initialize Dashboard
// ================================

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎤 DOM ready, initializing dashboard...');
    window.dashboard = new SovereignDashboard();
});

// Export for global access
window.SovereignDashboard = SovereignDashboard; 