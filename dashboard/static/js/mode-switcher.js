/**
 * Sovereign 4.0 Mode Switcher
 * 
 * Handles operation mode switching UI and API communication:
 * - Real-time mode status display
 * - Mode validation and switching
 * - User-friendly mode selection interface
 * - Integration with dashboard Socket.IO
 */

class ModeSwitcher {
    constructor() {
        this.currentMode = null;
        this.availableModes = [];
        this.selectedMode = null;
        this.validationIssues = [];
        this.isLoading = false;
        
        // UI elements
        this.elements = {
            modeIcon: document.getElementById('modeIcon'),
            modeCurrent: document.getElementById('modeCurrent'),
            modeStatus: document.getElementById('modeStatus'),
            modeSwitchBtn: document.getElementById('modeSwitchBtn'),
            
            // Modal elements
            overlay: document.getElementById('modeSelectorOverlay'),
            currentModeCard: document.getElementById('currentModeCard'),
            currentModeName: document.getElementById('currentModeName'),
            currentModeDescription: document.getElementById('currentModeDescription'),
            currentModeMetrics: document.getElementById('currentModeMetrics'),
            currentModeSessions: document.getElementById('currentModeSessions'),
            currentModeSuccessRate: document.getElementById('currentModeSuccessRate'),
            currentModeResponseTime: document.getElementById('currentModeResponseTime'),
            
            modesGrid: document.getElementById('modesGrid'),
            validationIssues: document.getElementById('validationIssues'),
            issuesList: document.getElementById('issuesList'),
            confirmButton: document.getElementById('confirmModeSwitch')
        };
        
        // Initialize
        this.init();
    }
    
    async init() {
        console.log('🎛️ Initializing Mode Switcher...');
        
        // Load initial mode status
        await this.loadModeStatus();
        await this.loadAvailableModes();
        
        // Set up periodic updates
        setInterval(() => {
            if (!this.isLoading) {
                this.loadModeStatus();
            }
        }, 5000); // Update every 5 seconds
        
        // Listen for mode change events from Socket.IO
        if (window.socket) {
            window.socket.on('mode_changed', (data) => {
                this.handleModeChanged(data);
            });
        }
        
        console.log('✅ Mode Switcher initialized');
    }
    
    async loadModeStatus() {
        try {
            const response = await fetch('/api/mode/status');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.updateModeDisplay(data);
            
        } catch (error) {
            console.error('❌ Error loading mode status:', error);
            this.updateModeDisplay({
                current_mode: null,
                capabilities: {},
                metrics: {},
                health: {},
                status: { initialized: false }
            });
        }
    }
    
    async loadAvailableModes() {
        try {
            const response = await fetch('/api/mode/available');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.availableModes = data.modes;
            
        } catch (error) {
            console.error('❌ Error loading available modes:', error);
            this.availableModes = [];
        }
    }
    
    updateModeDisplay(data) {
        const mode = data.current_mode;
        const metrics = data.metrics || {};
        const status = data.status || {};
        
        // Update header display
        if (mode) {
            this.currentMode = mode;
            this.elements.modeCurrent.textContent = this.formatModeName(mode);
            this.elements.modeIcon.textContent = this.getModeIcon(mode);
            
            // Update status based on health and metrics
            const successRate = metrics.success_rate || 0;
            if (!status.initialized) {
                this.elements.modeStatus.textContent = 'Initializing...';
            } else if (successRate >= 0.95) {
                this.elements.modeStatus.textContent = 'Healthy';
            } else if (successRate >= 0.8) {
                this.elements.modeStatus.textContent = 'Warning';
            } else {
                this.elements.modeStatus.textContent = 'Issues';
            }
        } else {
            this.elements.modeCurrent.textContent = 'Unknown';
            this.elements.modeStatus.textContent = 'Not initialized';
            this.elements.modeIcon.textContent = '❓';
        }
        
        // Update modal current mode display
        if (this.elements.currentModeName) {
            this.elements.currentModeName.textContent = mode ? this.formatModeName(mode) : 'Unknown';
            this.elements.currentModeDescription.textContent = mode ? this.getModeDescription(mode) : 'Mode not detected';
            
            // Update metrics
            this.elements.currentModeSessions.textContent = metrics.total_sessions || 0;
            this.elements.currentModeSuccessRate.textContent = 
                metrics.success_rate ? `${(metrics.success_rate * 100).toFixed(1)}%` : '--';
            this.elements.currentModeResponseTime.textContent = 
                metrics.average_response_time ? `${(metrics.average_response_time * 1000).toFixed(0)}ms` : '--';
        }
    }
    
    formatModeName(mode) {
        return mode.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    getModeIcon(mode) {
        switch (mode) {
            case 'realtime_only': return '⚡';
            case 'traditional_only': return '🔄';
            case 'hybrid_auto': return '🤖';
            default: return '❓';
        }
    }
    
    getModeDescription(mode) {
        switch (mode) {
            case 'realtime_only': return 'Using OpenAI Realtime API for fastest responses';
            case 'traditional_only': return 'Using traditional STT → LLM → TTS pipeline';
            case 'hybrid_auto': return 'Automatically switching between realtime and traditional';
            default: return 'Unknown operation mode';
        }
    }
    
    showModeSelector() {
        this.isLoading = true;
        this.elements.overlay.style.display = 'flex';
        this.renderAvailableModes();
        this.loadModeStatus(); // Refresh current mode data
        this.isLoading = false;
    }
    
    hideModeSelector() {
        this.elements.overlay.style.display = 'none';
        this.selectedMode = null;
        this.validationIssues = [];
        this.elements.confirmButton.disabled = true;
        this.elements.validationIssues.style.display = 'none';
    }
    
    renderAvailableModes() {
        const grid = this.elements.modesGrid;
        grid.innerHTML = '';
        
        this.availableModes.forEach(mode => {
            const card = this.createModeCard(mode);
            grid.appendChild(card);
        });
    }
    
    createModeCard(mode) {
        const card = document.createElement('div');
        card.className = `mode-card ${mode.is_available ? 'available' : 'unavailable'}`;
        if (mode.value === this.currentMode) {
            card.classList.add('current');
        }
        
        card.innerHTML = `
            <div class="mode-card-header">
                <span class="mode-card-icon">${this.getModeIcon(mode.value)}</span>
                <span class="mode-card-name">${mode.name}</span>
                ${!mode.is_available ? '<span class="unavailable-badge">Unavailable</span>' : ''}
            </div>
            <div class="mode-card-description">${mode.description}</div>
            <div class="mode-card-status">${mode.validation_summary}</div>
        `;
        
        if (mode.is_available && mode.value !== this.currentMode) {
            card.style.cursor = 'pointer';
            card.addEventListener('click', () => this.selectMode(mode.value));
        }
        
        return card;
    }
    
    async selectMode(modeValue) {
        if (modeValue === this.currentMode) return;
        
        this.selectedMode = modeValue;
        
        // Clear previous selection
        document.querySelectorAll('.mode-card').forEach(card => {
            card.classList.remove('selected');
        });
        
        // Highlight selected card
        event.currentTarget.classList.add('selected');
        
        // Validate the selected mode
        await this.validateSelectedMode();
    }
    
    async validateSelectedMode() {
        if (!this.selectedMode) return;
        
        this.isLoading = true;
        this.elements.confirmButton.disabled = true;
        this.elements.confirmButton.textContent = 'Validating...';
        
        try {
            const response = await fetch('/api/mode/validate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: this.selectedMode })
            });
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.validationIssues = data.issues || [];
            
            // Show validation results
            this.renderValidationIssues();
            
            // Enable switch button if validation passed
            const criticalIssues = this.validationIssues.filter(issue => issue.severity === 'critical');
            if (criticalIssues.length === 0) {
                this.elements.confirmButton.disabled = false;
                this.elements.confirmButton.textContent = 'Switch Mode';
            } else {
                this.elements.confirmButton.disabled = true;
                this.elements.confirmButton.textContent = 'Cannot Switch (Critical Issues)';
            }
            
        } catch (error) {
            console.error('❌ Error validating mode:', error);
            this.elements.confirmButton.disabled = true;
            this.elements.confirmButton.textContent = 'Validation Failed';
        }
        
        this.isLoading = false;
    }
    
    renderValidationIssues() {
        const issuesContainer = this.elements.validationIssues;
        const issuesList = this.elements.issuesList;
        
        if (this.validationIssues.length === 0) {
            issuesContainer.style.display = 'none';
            return;
        }
        
        issuesContainer.style.display = 'block';
        issuesList.innerHTML = '';
        
        this.validationIssues.forEach(issue => {
            const issueElement = document.createElement('div');
            issueElement.className = `issue-item ${issue.severity}`;
            
            const icon = this.getSeverityIcon(issue.severity);
            
            issueElement.innerHTML = `
                <span class="issue-icon">${icon}</span>
                <div class="issue-content">
                    <div class="issue-message">${issue.message}</div>
                    ${issue.suggestion ? `<div class="issue-suggestion">${issue.suggestion}</div>` : ''}
                </div>
            `;
            
            issuesList.appendChild(issueElement);
        });
    }
    
    getSeverityIcon(severity) {
        switch (severity) {
            case 'critical': return '🚨';
            case 'error': return '❌';
            case 'warning': return '⚠️';
            case 'info': return 'ℹ️';
            default: return '●';
        }
    }
    
    async confirmModeSwitch() {
        if (!this.selectedMode || this.isLoading) return;
        
        this.isLoading = true;
        this.elements.confirmButton.disabled = true;
        this.elements.confirmButton.textContent = 'Switching...';
        
        try {
            const response = await fetch('/api/mode/switch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    mode: this.selectedMode,
                    reason: 'User requested mode switch via dashboard'
                })
            });
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            
            if (data.success) {
                console.log(`✅ Mode switched to ${data.mode}`);
                
                // Show success message
                this.showSuccessMessage(`Successfully switched to ${this.formatModeName(data.mode)}`);
                
                // Update UI
                this.currentMode = data.mode;
                this.hideModeSelector();
                await this.loadModeStatus();
                
            } else {
                throw new Error(data.error || 'Mode switch failed');
            }
            
        } catch (error) {
            console.error('❌ Error switching mode:', error);
            this.showErrorMessage(`Failed to switch mode: ${error.message}`);
        }
        
        this.isLoading = false;
        this.elements.confirmButton.textContent = 'Switch Mode';
    }
    
    handleModeChanged(data) {
        console.log('🔄 Mode changed event received:', data);
        
        // Update current mode
        this.currentMode = data.mode;
        
        // Refresh UI
        this.loadModeStatus();
        
        // Show notification
        this.showSuccessMessage(`Mode switched to ${this.formatModeName(data.mode)}`);
        
        // Close modal if open
        if (this.elements.overlay.style.display !== 'none') {
            this.hideModeSelector();
        }
    }
    
    showSuccessMessage(message) {
        this.showNotification(message, 'success');
    }
    
    showErrorMessage(message) {
        this.showNotification(message, 'error');
    }
    
    showNotification(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `mode-notification ${type}`;
        notification.innerHTML = `
            <span class="notification-icon">${type === 'success' ? '✅' : '❌'}</span>
            <span class="notification-message">${message}</span>
        `;
        
        // Add to DOM
        document.body.appendChild(notification);
        
        // Show animation
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Auto remove
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => document.body.removeChild(notification), 300);
        }, 3000);
    }
}

// Global functions for HTML onclick handlers
window.showModeSelector = () => {
    if (window.modeSwitcher) {
        window.modeSwitcher.showModeSelector();
    }
};

window.hideModeSelector = () => {
    if (window.modeSwitcher) {
        window.modeSwitcher.hideModeSelector();
    }
};

window.confirmModeSwitch = () => {
    if (window.modeSwitcher) {
        window.modeSwitcher.confirmModeSwitch();
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Wait for Socket.IO to be ready
    setTimeout(() => {
        window.modeSwitcher = new ModeSwitcher();
    }, 1000);
}); 