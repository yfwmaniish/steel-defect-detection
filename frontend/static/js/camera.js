// Camera functionality for steel defect detection dashboard
class CameraController {
    constructor() {
        this.stream = null;
        this.isStreaming = false;
        this.cameraFeed = document.getElementById('cameraFeed');
        this.startBtn = document.getElementById('startCamera');
        this.stopBtn = document.getElementById('stopCamera');
        this.videoElement = null;
        
        this.init();
    }

    init() {
        // Add event listeners
        if (this.startBtn) {
            this.startBtn.addEventListener('click', () => this.startCamera());
        }
        if (this.stopBtn) {
            this.stopBtn.addEventListener('click', () => this.stopCamera());
        }
        
        // Initialize camera feed placeholder
        this.showPlaceholder();
    }

    async startCamera() {
        try {
            this.startBtn.disabled = true;
            this.startBtn.innerHTML = '<div class="loading"></div> Starting...';
            
            // Get user media stream
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });

            // Create video element if not exists
            if (!this.videoElement) {
                this.videoElement = document.createElement('video');
                this.videoElement.autoplay = true;
                this.videoElement.muted = true;
                this.videoElement.playsInline = true;
                this.videoElement.style.width = '100%';
                this.videoElement.style.height = '100%';
                this.videoElement.style.objectFit = 'cover';
                this.videoElement.style.borderRadius = '8px';
            }

            // Set video source and add to DOM
            this.videoElement.srcObject = this.stream;
            this.cameraFeed.innerHTML = '';
            this.cameraFeed.appendChild(this.videoElement);

            // Update UI state
            this.isStreaming = true;
            this.startBtn.disabled = false;
            this.startBtn.innerHTML = '<i class="fas fa-video"></i> Start Camera';
            this.stopBtn.disabled = false;
            this.startBtn.style.display = 'none';
            this.stopBtn.style.display = 'inline-block';
            
            // Update status indicator
            this.updateStatus('online');
            
            // Emit camera start event to server
            if (window.socket) {
                window.socket.emit('camera_start');
            }
            
            console.log('Camera started successfully');
            
        } catch (error) {
            console.error('Error starting camera:', error);
            this.showError('Failed to start camera. Please check permissions.');
            this.startBtn.disabled = false;
            this.startBtn.innerHTML = '<i class="fas fa-video"></i> Start Camera';
        }
    }

    stopCamera() {
        if (this.stream) {
            // Stop all tracks
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        // Remove video element
        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }

        // Update UI state
        this.isStreaming = false;
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.startBtn.style.display = 'inline-block';
        this.stopBtn.style.display = 'none';
        
        // Show placeholder
        this.showPlaceholder();
        
        // Update status indicator
        this.updateStatus('offline');
        
        // Emit camera stop event to server
        if (window.socket) {
            window.socket.emit('camera_stop');
        }
        
        console.log('Camera stopped');
    }

    showPlaceholder() {
        this.cameraFeed.innerHTML = `
            <div class="placeholder">
                <i class="fas fa-camera"></i>
                <div>Camera Feed</div>
                <div style="font-size: 0.9em; color: #999; margin-top: 5px;">
                    Click "Start Camera" to begin
                </div>
            </div>
        `;
    }

    showError(message) {
        this.cameraFeed.innerHTML = `
            <div class="placeholder">
                <i class="fas fa-exclamation-triangle" style="color: #e53e3e;"></i>
                <div style="color: #e53e3e;">${message}</div>
            </div>
        `;
    }

    updateStatus(status) {
        const statusIndicator = document.querySelector('.status-indicator');
        if (statusIndicator) {
            statusIndicator.className = `status-indicator ${status}`;
        }
    }

    // Capture current frame for processing
    captureFrame() {
        if (!this.videoElement || !this.isStreaming) {
            return null;
        }

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = this.videoElement.videoWidth;
        canvas.height = this.videoElement.videoHeight;
        
        ctx.drawImage(this.videoElement, 0, 0);
        
        return canvas.toDataURL('image/jpeg', 0.8);
    }

    // Get camera stream status
    isActive() {
        return this.isStreaming && this.stream !== null;
    }
}

// Initialize camera controller when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.cameraController = new CameraController();
});
