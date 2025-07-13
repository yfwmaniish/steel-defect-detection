// Detection functionality for steel defect detection dashboard
class DetectionController {
    constructor() {
        this.isDetecting = false;
        this.detectionInterval = null;
        this.detectionResults = document.getElementById('detectionResults');
        this.startDetectionBtn = document.getElementById('startDetection');
        this.stopDetectionBtn = document.getElementById('stopDetection');
        this.captureBtn = document.getElementById('captureBtn');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.uploadInput = document.getElementById('uploadInput');
        this.currentResults = null;
        
        this.init();
    }

    init() {
        // Add event listeners
        if (this.startDetectionBtn) {
            this.startDetectionBtn.addEventListener('click', () => this.startDetection());
        }
        if (this.stopDetectionBtn) {
            this.stopDetectionBtn.addEventListener('click', () => this.stopDetection());
        }
        if (this.captureBtn) {
            this.captureBtn.addEventListener('click', () => this.captureAndDetect());
        }
        if (this.uploadBtn) {
            this.uploadBtn.addEventListener('click', () => this.uploadInput.click());
        }
        if (this.uploadInput) {
            this.uploadInput.addEventListener('change', (e) => this.handleImageUpload(e));
        }
        
        // Initialize results display
        this.showNoResults();
        
        // Listen for socket events
        this.initSocketListeners();
    }

    initSocketListeners() {
        if (window.socket) {
            window.socket.on('detection_result', (data) => {
                this.displayResults(data);
            });
            
            window.socket.on('detection_error', (data) => {
                this.showError(data.error);
            });
        }
    }

    startDetection() {
        if (!window.cameraController || !window.cameraController.isActive()) {
            this.showError('Please start camera first');
            return;
        }

        this.isDetecting = true;
        this.startDetectionBtn.disabled = true;
        this.startDetectionBtn.innerHTML = '<div class="loading"></div> Starting...';
        this.stopDetectionBtn.disabled = false;
        this.stopDetectionBtn.style.display = 'inline-block';
        this.startDetectionBtn.style.display = 'none';
        
        // Start continuous detection
        this.detectionInterval = setInterval(() => {
            this.performDetection();
        }, 2000); // Detect every 2 seconds
        
        // Emit detection start event
        if (window.socket) {
            window.socket.emit('detection_start');
        }
        
        console.log('Detection started');
    }

    stopDetection() {
        this.isDetecting = false;
        
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
        
        this.startDetectionBtn.disabled = false;
        this.startDetectionBtn.innerHTML = '<i class="fas fa-play"></i> Start Detection';
        this.stopDetectionBtn.disabled = true;
        this.startDetectionBtn.style.display = 'inline-block';
        this.stopDetectionBtn.style.display = 'none';
        
        // Emit detection stop event
        if (window.socket) {
            window.socket.emit('detection_stop');
        }
        
        console.log('Detection stopped');
    }

    performDetection() {
        if (!this.isDetecting || !window.cameraController) {
            return;
        }

        const frameData = window.cameraController.captureFrame();
        if (!frameData) {
            return;
        }

        // Send frame to server for detection
        if (window.socket) {
            window.socket.emit('detect_frame', { image: frameData });
        }
    }

    captureAndDetect() {
        if (!window.cameraController || !window.cameraController.isActive()) {
            this.showError('Please start camera first');
            return;
        }

        this.captureBtn.disabled = true;
        this.captureBtn.innerHTML = '<div class="loading"></div> Processing...';
        
        const frameData = window.cameraController.captureFrame();
        if (frameData) {
            // Send single frame for detection
            if (window.socket) {
                window.socket.emit('detect_single', { image: frameData });
            }
        }
        
        setTimeout(() => {
            this.captureBtn.disabled = false;
            this.captureBtn.innerHTML = '<i class="fas fa-camera"></i> Capture';
        }, 2000);
    }

    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            this.showError('Please select an image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            this.uploadBtn.disabled = true;
            this.uploadBtn.innerHTML = '<div class="loading"></div> Processing...';
            
            // Send image to server for detection
            if (window.socket) {
                window.socket.emit('detect_upload', { 
                    image: e.target.result,
                    filename: file.name
                });
            }
            
            setTimeout(() => {
                this.uploadBtn.disabled = false;
                this.uploadBtn.innerHTML = '<i class="fas fa-upload"></i> Upload';
            }, 2000);
        };
        
        reader.readAsDataURL(file);
    }

    displayResults(data) {
        this.currentResults = data;
        
        if (!data.detections || data.detections.length === 0) {
            this.showNoDefects();
            return;
        }

        const resultsHtml = `
            <div class="results-header">
                <h3>Detection Results</h3>
                <span class="timestamp">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="defects-grid">
                ${data.detections.map(detection => `
                    <div class="defect-item ${this.getDefectSeverity(detection.confidence)}">
                        <div class="defect-info">
                            <div class="defect-type">${detection.class}</div>
                            <div class="defect-confidence">${(detection.confidence * 100).toFixed(1)}%</div>
                        </div>
                        <div class="defect-details">
                            <div class="defect-location">
                                Position: (${detection.bbox[0]}, ${detection.bbox[1]})
                            </div>
                            <div class="defect-size">
                                Size: ${detection.bbox[2]}×${detection.bbox[3]}
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
            <div class="results-summary">
                <div class="summary-item">
                    <span class="label">Total Defects:</span>
                    <span class="value">${data.detections.length}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Processing Time:</span>
                    <span class="value">${data.processing_time || 'N/A'}</span>
                </div>
            </div>
        `;
        
        this.detectionResults.innerHTML = resultsHtml;
        this.animateResults();
    }

    showNoDefects() {
        this.detectionResults.innerHTML = `
            <div class="no-results">
                <i class="fas fa-check-circle" style="color: #48bb78;"></i>
                <div>No Defects Detected</div>
                <div style="font-size: 0.9em; color: #999; margin-top: 5px;">
                    Steel surface appears clean
                </div>
            </div>
        `;
    }

    showNoResults() {
        this.detectionResults.innerHTML = `
            <div class="no-results">
                <i class="fas fa-search"></i>
                <div>No Detection Results</div>
                <div style="font-size: 0.9em; color: #999; margin-top: 5px;">
                    Start detection or upload an image
                </div>
            </div>
        `;
    }

    showError(message) {
        this.detectionResults.innerHTML = `
            <div class="no-results">
                <i class="fas fa-exclamation-triangle" style="color: #e53e3e;"></i>
                <div style="color: #e53e3e;">Error</div>
                <div style="font-size: 0.9em; color: #999; margin-top: 5px;">
                    ${message}
                </div>
            </div>
        `;
    }

    getDefectSeverity(confidence) {
        if (confidence >= 0.8) return 'high';
        if (confidence >= 0.6) return 'medium';
        return 'low';
    }

    animateResults() {
        const results = this.detectionResults.querySelector('.defects-grid');
        if (results) {
            results.style.opacity = '0';
            results.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                results.style.transition = 'all 0.3s ease';
                results.style.opacity = '1';
                results.style.transform = 'translateY(0)';
            }, 100);
        }
    }

    // Export results to CSV
    exportResults() {
        if (!this.currentResults || !this.currentResults.detections.length) {
            alert('No results to export');
            return;
        }

        const csvContent = this.convertToCSV(this.currentResults.detections);
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `detection_results_${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
        window.URL.revokeObjectURL(url);
    }

    convertToCSV(detections) {
        const headers = ['Class', 'Confidence', 'X', 'Y', 'Width', 'Height', 'Timestamp'];
        const rows = detections.map(detection => [
            detection.class,
            detection.confidence.toFixed(3),
            detection.bbox[0],
            detection.bbox[1],
            detection.bbox[2],
            detection.bbox[3],
            new Date().toISOString()
        ]);
        
        return [headers, ...rows].map(row => row.join(',')).join('\n');
    }
}

// Initialize detection controller when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.detectionController = new DetectionController();
});
