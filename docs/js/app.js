/**
 * ===== APP.JS =====
 * Main application entry point
 * Initializes all modules and handles global functionality
 */

class PyMultiWFNApp {
    constructor() {
        this.version = '1.0.0';
        this.modules = new Map();
        this.isInitialized = false;
        this.config = {
            enableAnalytics: true,
            enableAnimations: true,
            enableSocialSharing: true,
            githubRepo: 'https://github.com/yourusername/PyMultiWFN',
            apiEndpoint: 'https://api.github.com/repos/yourusername/PyMultiWFN'
        };

        this.init();
    }

    async init() {
        try {
            console.log('🚀 Initializing PyMultiWFN Website v' + this.version);

            // Direct show content immediately - no loading screen
            this.showContentImmediately();

            // Wait for DOM to be ready, then initialize modules
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', () => this.start());
            } else {
                this.start();
            }
        } catch (error) {
            console.error('❌ Failed to initialize app:', error);
            // Don't show error to user, just log it
            console.log('Some features may be limited, but core functionality works.');
        }
    }

    async start() {
        try {
            // Initialize core modules asynchronously (non-blocking)
            this.initializeModulesAsync();

            // Load external data asynchronously (non-blocking)
            this.loadExternalDataAsync();

            // Setup global event listeners
            this.setupGlobalEventListeners();

            // Initialize performance monitoring (lightweight)
            this.initializePerformanceMonitoring();

            this.isInitialized = true;
            console.log('✅ PyMultiWFN Website initialized successfully');

            // Trigger ready event
            document.dispatchEvent(new CustomEvent('appReady', {
                detail: { version: this.version, modules: Array.from(this.modules.keys()) }
            }));

        } catch (error) {
            console.error('❌ Failed to start app:', error);
            // Don't show error to user - continue with basic functionality
        }
    }

    /**
     * Show content immediately without loading screen
     */
    showContentImmediately() {
        // Hide any existing loading overlays
        const loadingElement = document.getElementById('loading');
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }

        // Show main content
        const mainContent = document.getElementById('app');
        if (mainContent) {
            mainContent.style.display = 'block';
            mainContent.style.opacity = '1';
            mainContent.classList.add('loaded');
        }

        // Add minimal inline styles to ensure content is visible
        document.body.style.visibility = 'visible';
        document.body.style.overflow = 'auto';
    }

    /**
     * Initialize all modules asynchronously (non-blocking)
     */
    initializeModulesAsync() {
        const modules = [
            { name: 'analytics', class: AnalyticsManager, condition: () => this.config.enableAnalytics },
            { name: 'animations', class: AnimationManager, condition: () => this.config.enableAnimations },
            { name: 'socialShare', class: SocialShareManager, condition: () => this.config.enableSocialSharing },
            { name: 'ui', class: UIManager, condition: () => true },
            { name: 'github', class: GitHubManager, condition: () => true }
        ];

        // Initialize modules in background without blocking UI
        modules.forEach(moduleConfig => {
            if (moduleConfig.condition()) {
                setTimeout(() => {
                    try {
                        const moduleInstance = new moduleConfig.class(this.config);
                        this.modules.set(moduleConfig.name, moduleInstance);
                        console.log(`📦 Module '${moduleConfig.name}' loaded`);
                    } catch (error) {
                        console.warn(`⚠️ Failed to load module '${moduleConfig.name}':`, error);
                    }
                }, 0); // Schedule for next event loop tick
            }
        });
    }

    /**
     * Load external data from APIs asynchronously
     */
    loadExternalDataAsync() {
        // Load GitHub data in background
        setTimeout(() => {
            this.loadGitHubData().catch(error => {
                console.warn('⚠️ Failed to load GitHub data:', error);
            });
        }, 100);

        // Load user analytics in background
        setTimeout(() => {
            try {
                this.loadUserAnalytics();
            } catch (error) {
                console.warn('⚠️ Failed to load user analytics:', error);
            }
        }, 200);
    }

    /**
     * Load GitHub repository data
     */
    async loadGitHubData() {
        try {
            const githubManager = this.modules.get('github');
            if (githubManager) {
                await githubManager.fetchRepoData();
            }
        } catch (error) {
            console.warn('⚠️ Failed to load GitHub data:', error);
        }
    }

    /**
     * Load user analytics data
     */
    loadUserAnalytics() {
        const analyticsManager = this.modules.get('analytics');
        if (analyticsManager) {
            analyticsManager.loadUserData();
        }
    }

    /**
     * Setup global event listeners
     */
    setupGlobalEventListeners() {
        // Handle visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseAnimations();
            } else {
                this.resumeAnimations();
            }
        });

        // Handle online/offline status
        window.addEventListener('online', () => {
            this.showNotification('Connection restored', 'success');
            this.retryFailedRequests();
        });

        window.addEventListener('offline', () => {
            this.showNotification('Connection lost. Some features may be limited.', 'warning');
        });

        // Handle window resize with debounce
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.handleResize();
            }, 250);
        });

        // Handle keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });

        // Handle error reporting
        window.addEventListener('error', (e) => {
            this.reportError(e);
        });

        // Handle unhandled promise rejections
        window.addEventListener('unhandledrejection', (e) => {
            this.reportError(new Error(e.reason));
        });
    }

    /**
     * Initialize performance monitoring
     */
    initializePerformanceMonitoring() {
        // Monitor page load performance
        if ('performance' in window) {
            window.addEventListener('load', () => {
                setTimeout(() => {
                    const perfData = performance.getEntriesByType('navigation')[0];
                    const loadTime = perfData.loadEventEnd - perfData.loadEventStart;
                    console.log(`📊 Page load time: ${loadTime}ms`);

                    // Track performance event
                    const analyticsManager = this.modules.get('analytics');
                    if (analyticsManager) {
                        analyticsManager.trackEvent('page_load_complete', {
                            loadTime: loadTime,
                            userAgent: navigator.userAgent
                        });
                    }
                }, 0);
            });
        }
    }

    /**
     * Show loading state
     */
    showLoadingState() {
        const loadingElement = document.getElementById('loading');
        if (loadingElement) {
            loadingElement.style.display = 'flex';
        }

        // Create custom loading overlay if needed
        if (!loadingElement) {
            const loadingOverlay = document.createElement('div');
            loadingOverlay.id = 'appLoading';
            loadingOverlay.innerHTML = `
                <div class="loading-content">
                    <div class="loading-spinner"></div>
                    <p>Initializing PyMultiWFN...</p>
                </div>
            `;
            loadingOverlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(255, 255, 255, 0.95);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 99999;
                backdrop-filter: blur(5px);
            `;
            document.body.appendChild(loadingOverlay);
        }
    }

    /**
     * Hide loading state
     */
    hideLoadingState() {
        const loadingElement = document.getElementById('loading');
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }

        const appLoading = document.getElementById('appLoading');
        if (appLoading) {
            appLoading.style.display = 'none';
            setTimeout(() => appLoading.remove(), 300);
        }
    }

    /**
     * Show ready state
     */
    showReadyState() {
        // Add ready class to body
        document.body.classList.add('app-ready');

        // Show welcome notification for first-time visitors
        if (!this.getCookie('pymultiwfn_visited')) {
            setTimeout(() => {
                this.showWelcomeMessage();
                this.setCookie('pymultiwfn_visited', 'true', 30);
            }, 1000);
        }
    }

    /**
     * Show welcome message
     */
    showWelcomeMessage() {
        this.showNotification(
            '🎉 Welcome to PyMultiWFN! Discover the future of quantum chemistry analysis.',
            'info',
            5000
        );
    }

    /**
     * Show notification
     */
    showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `app-notification app-notification--${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        // Add to page
        document.body.appendChild(notification);

        // Auto remove
        setTimeout(() => {
            if (notification.parentElement) {
                notification.classList.add('app-notification--hide');
                setTimeout(() => notification.remove(), 300);
            }
        }, duration);
    }

    /**
     * Get notification icon based on type
     */
    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    /**
     * Show error message
     */
    showErrorMessage(message) {
        this.showNotification(message, 'error', 0); // Don't auto-hide error messages
    }

    /**
     * Pause animations
     */
    pauseAnimations() {
        const animationManager = this.modules.get('animations');
        if (animationManager && typeof animationManager.pause === 'function') {
            animationManager.pause();
        }
    }

    /**
     * Resume animations
     */
    resumeAnimations() {
        const animationManager = this.modules.get('animations');
        if (animationManager && typeof animationManager.resume === 'function') {
            animationManager.resume();
        }
    }

    /**
     * Handle window resize
     */
    handleResize() {
        // Notify modules of resize
        this.modules.forEach((module, name) => {
            if (typeof module.handleResize === 'function') {
                module.handleResize();
            }
        });
    }

    /**
     * Handle keyboard shortcuts
     */
    handleKeyboardShortcuts(e) {
        // Ctrl/Cmd + K for search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            this.openSearch();
        }

        // Escape to close modals
        if (e.key === 'Escape') {
            this.closeModals();
        }

        // Ctrl/Cmd + / for keyboard shortcuts help
        if ((e.ctrlKey || e.metaKey) && e.key === '/') {
            e.preventDefault();
            this.showKeyboardShortcuts();
        }
    }

    /**
     * Open search
     */
    openSearch() {
        // Implementation depends on search module
        console.log('🔍 Search requested');
    }

    /**
     * Close all modals
     */
    closeModals() {
        // Close social widget
        if (window.socialShareManager) {
            window.socialShareManager.closeWidget();
        }

        // Close other modals
        document.querySelectorAll('.modal-overlay').forEach(modal => {
            modal.style.display = 'none';
        });
    }

    /**
     * Show keyboard shortcuts help
     */
    showKeyboardShortcuts() {
        this.showNotification('⌨️ Ctrl+K: Search, Ctrl+/: Shortcuts, Escape: Close modals', 'info', 4000);
    }

    /**
     * Report errors
     */
    reportError(error) {
        console.error('🐛 Error reported:', error);

        const analyticsManager = this.modules.get('analytics');
        if (analyticsManager) {
            analyticsManager.trackEvent('error', {
                message: error.message,
                stack: error.stack,
                url: window.location.href
            });
        }
    }

    /**
     * Retry failed requests
     */
    retryFailedRequests() {
        const githubManager = this.modules.get('github');
        if (githubManager && typeof githubManager.retry === 'function') {
            githubManager.retry();
        }
    }

    /**
     * Cookie utilities
     */
    setCookie(name, value, days) {
        const expires = new Date();
        expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
        document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/`;
    }

    getCookie(name) {
        const nameEQ = name + "=";
        const ca = document.cookie.split(';');
        for (let i = 0; i < ca.length; i++) {
            let c = ca[i];
            while (c.charAt(0) === ' ') c = c.substring(1, c.length);
            if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
        }
        return null;
    }

    /**
     * Get module by name
     */
    getModule(name) {
        return this.modules.get(name);
    }

    /**
     * Check if app is initialized
     */
    ready() {
        return this.isInitialized;
    }

    /**
     * Cleanup and destroy app
     */
    destroy() {
        this.modules.forEach((module, name) => {
            if (typeof module.cleanup === 'function') {
                module.cleanup();
            }
        });
        this.modules.clear();
        this.isInitialized = false;
    }
}

// Initialize app when DOM is ready
const app = new PyMultiWFNApp();

// Make app globally available
window.PyMultiWFNApp = app;
window.app = app;

// Export for ES6 modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PyMultiWFNApp;
}