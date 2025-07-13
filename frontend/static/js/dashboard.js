/**
 * Main Dashboard JavaScript
 * Manages overall application state, WebSocket connections, and UI coordination
 */

class Dashboard {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.startTime = new Date();
        this.imagesProcessed = 0;
        this.detectionsCount = 0;
        this.isProcessing = false;
        
        this.init();
    }

    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.updateUptime();
        this.updateSystemStatus();
        
        // Update uptime every second
        setInterval(() => this.updateUptime(), 1000);
        
        // Update system status every 5 seconds
        setInterval(() => this.updateSystemStatus(), 5000);
    }

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.socket = new WebSocket(wsUrl);
        
        this.socket.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.updateConnectionStatus(true);
        };
        
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.socket.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus(false);
            
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.setupWebSocket(), 3000);
        };
        
        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showAlert('WebSocket connection error', 'danger');
        };
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'detection_result':
                this.handleDetectionResult(data.data);
                break;
            case 'system_status':
                this.handleSystemStatus(data.data);
                break;
            case 'camera_status':
                this.handleCameraStatus(data.data);
                break;
            case 'error':
                this.showAlert(data.message, 'danger');
                break;
            case 'info':
                this.showAlert(data.message, 'info');
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }

    handleDetectionResult(result) {
        this.imagesProcessed++;
        this.detectionsCount += result.detections.length;
        
        // Update detection display
        if (window.detectionManager) {
            window.detectionManager.addDetection(result);
        }
        
        // Update statistics
        if (window.statisticsManager) {
            window.statisticsManager.updateStats({
                imagesProcessed: this.imagesProcessed,
                detectionsCount: this.detectionsCount,
                detectionRate: this.calculateDetectionRate()
            });
        }
        
        // Update camera feed with annotated image
        if (window.cameraManager && result.annotated_image) {
            window.cameraManager.updateFeed(result.annotated_image);
        }
    }

    handleSystemStatus(status) {
        document.getElementById('cpu-usage').textContent = `${status.cpu_usage}%`;
        document.getElementById('memory-usage').textContent = `${status.memory_usage}%`;
        document.getElementById('gpu-usage').textContent = `${status.gpu_usage}%`;
        
        if (status.gpu_memory) {
            document.getElementById('gpu-memory').textContent = `${status.gpu_memory}%`;
        }
    }

    handleCameraStatus(status) {
        if (window.cameraManager) {
            window.cameraManager.updateStatus(status);
        }
    }

    setupEventListeners() {
        // Start/Stop Detection button
        const startStopBtn = document.getElementById('start-stop-detection');
        if (startStopBtn) {
            startStopBtn.addEventListener('click', () => this.toggleDetection());
        }
        
        // Clear Results button
        const clearBtn = document.getElementById('clear-results');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearResults());
        }
        
        // Settings form
        const settingsForm = document.getElementById('settings-form');
        if (settingsForm) {
            settingsForm.addEventListener('submit', (e) => this.saveSettings(e));
        }
        
        // Export Results button
        const exportBtn = document.getElementById('export-results');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportResults());
        }
    }

    toggleDetection() {
        const button = document.getElementById('start-stop-detection');
        
        if (this.isProcessing) {
            this.stopDetection();
            button.textContent = 'Start Detection';
            button.classList.remove('btn-danger');
            button.classList.add('btn-success');
        } else {
            this.startDetection();
            button.textContent = 'Stop Detection';
            button.classList.remove('btn-success');
            button.classList.add('btn-danger');
        }
    }

    startDetection() {
        if (!this.isConnected) {
            this.showAlert('Not connected to server', 'danger');
            return;
        }
        
        const settings = this.getDetectionSettings();
        this.socket.send(JSON.stringify({
            type: 'start_detection',
            data: settings
        }));
        
        this.isProcessing = true;
        this.showLoadingIndicator(true);
        this.showAlert('Detection started', 'success');
    }

    stopDetection() {
        if (!this.isConnected) {
            return;
        }
        
        this.socket.send(JSON.stringify({
            type: 'stop_detection'
        }));
        
        this.isProcessing = false;
        this.showLoadingIndicator(false);
        this.showAlert('Detection stopped', 'info');
    }

    getDetectionSettings() {
        return {
            camera_id: document.getElementById('camera-select').value,
            confidence_threshold: parseFloat(document.getElementById('confidence-threshold').value),
            detection_mode: document.getElementById('detection-mode').value,
            save_results: document.getElementById('save-results').checked,
            show_labels: document.getElementById('show-labels').checked,
            show_confidence: document.getElementById('show-confidence').checked
        };
    }

    saveSettings(event) {
        event.preventDefault();
        
        const settings = this.getDetectionSettings();
        
        // Save to localStorage
        localStorage.setItem('detection_settings', JSON.stringify(settings));
        
        // Send to server
        if (this.isConnected) {
            this.socket.send(JSON.stringify({
                type: 'update_settings',
                data: settings
            }));
        }
        
        this.showAlert('Settings saved', 'success');
    }

    loadSettings() {
        const savedSettings = localStorage.getItem('detection_settings');
        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            
            // Apply settings to form
            if (settings.camera_id) {
                document.getElementById('camera-select').value = settings.camera_id;
            }
            if (settings.confidence_threshold) {
                document.getElementById('confidence-threshold').value = settings.confidence_threshold;
            }
            if (settings.detection_mode) {
                document.getElementById('detection-mode').value = settings.detection_mode;
            }
            if (settings.save_results !== undefined) {
                document.getElementById('save-results').checked = settings.save_results;
            }
            if (settings.show_labels !== undefined) {
                document.getElementById('show-labels').checked = settings.show_labels;
            }
            if (settings.show_confidence !== undefined) {
                document.getElementById('show-confidence').checked = settings.show_confidence;
            }
        }
    }

    clearResults() {
        if (window.detectionManager) {
            window.detectionManager.clearDetections();
        }
        
        if (window.statisticsManager) {
            window.statisticsManager.resetStats();
        }
        
        this.imagesProcessed = 0;
        this.detectionsCount = 0;
        
        this.showAlert('Results cleared', 'info');
    }

    exportResults() {
        if (!this.isConnected) {
            this.showAlert('Not connected to server', 'danger');
            return;
        }
        
        this.socket.send(JSON.stringify({
            type: 'export_results',
            data: { format: 'csv' }
        }));
        
        this.showAlert('Export requested', 'info');
    }

    updateUptime() {
        const now = new Date();
        const uptimeMs = now - this.startTime;
        const uptimeStr = this.formatUptime(uptimeMs);
        
        const uptimeElement = document.getElementById('uptime');
        if (uptimeElement) {
            uptimeElement.textContent = uptimeStr;
        }
    }

    formatUptime(ms) {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${seconds % 60}s`;
        } else {
            return `${seconds}s`;
        }
    }

    calculateDetectionRate() {
        if (this.imagesProcessed === 0) return 0;
        return (this.detectionsCount / this.imagesProcessed * 100).toFixed(1);
    }

    updateSystemStatus() {
        if (this.isConnected) {
            this.socket.send(JSON.stringify({
                type: 'get_system_status'
            }));
        }
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = connected ? 'Connected' : 'Disconnected';
            statusElement.className = connected ? 'status-indicator online' : 'status-indicator offline';
        }
    }

    showAlert(message, type = 'info') {
        const alertsContainer = document.getElementById('alerts-container');
        if (!alertsContainer) return;
        
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.innerHTML = `
            <span>${message}</span>
            <button type="button" class="close" onclick="this.parentElement.remove()">×</button>
        `;
        
        alertsContainer.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentElement) {
                alert.remove();
            }
        }, 5000);
    }

    showLoadingIndicator(show) {
        const loadingElement = document.getElementById('loading-indicator');
        if (loadingElement) {
            loadingElement.style.display = show ? 'block' : 'none';
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
    window.dashboard.loadSettings();
});
