/**
 * ===== ANALYTICS MODULE =====
 * Handles user analytics and behavior tracking
 * Provides insights into website usage and conversion optimization
 */

class AnalyticsManager {
    constructor(config = {}) {
        this.config = {
            trackPageViews: true,
            trackUserInteractions: true,
            trackPerformance: true,
            enableHeatmap: false,
            enableSessionRecording: false,
            ...config
        };

        this.sessionData = {
            startTime: Date.now(),
            pageViews: 0,
            interactions: 0,
            timeOnPage: 0,
            scrollDepth: 0,
            maxScrollDepth: 0
        };

        this.events = [];
        this.isInitialized = false;
        this.init();
    }

    init() {
        this.sessionId = this.generateSessionId();
        this.userId = this.getUserId();
        this.setupPageTracking();
        this.setupInteractionTracking();
        this.setupPerformanceTracking();
        this.loadStoredData();
        this.isInitialized = true;

        console.log('📊 Analytics Manager initialized');
    }

    /**
     * Generate unique session ID
     */
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    /**
     * Get or generate user ID
     */
    getUserId() {
        let userId = localStorage.getItem('pymultiwfn_user_id');
        if (!userId) {
            userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('pymultiwfn_user_id', userId);
        }
        return userId;
    }

    /**
     * Setup page view tracking
     */
    setupPageTracking() {
        if (!this.config.trackPageViews) return;

        // Track initial page view
        this.trackPageView();

        // Track page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.trackPageLeave();
            } else {
                this.trackPageReturn();
            }
        });

        // Track page unload
        window.addEventListener('beforeunload', () => {
            this.trackPageLeave();
            this.flushEvents();
        });

        // Track SPA navigation
        window.addEventListener('popstate', () => {
            this.trackPageView();
        });
    }

    /**
     * Setup interaction tracking
     */
    setupInteractionTracking() {
        if (!this.config.trackUserInteractions) return;

        // Track clicks
        document.addEventListener('click', (e) => {
            this.trackClick(e);
        });

        // Track form interactions
        document.addEventListener('submit', (e) => {
            this.trackFormSubmit(e);
        });

        // Track scroll depth
        this.setupScrollTracking();

        // Track time on page
        this.setupTimeTracking();

        // Track custom events
        document.addEventListener('pymultiwfn_analytics', (e) => {
            this.trackEvent(e.detail.eventName, e.detail.data);
        });
    }

    /**
     * Setup performance tracking
     */
    setupPerformanceTracking() {
        if (!this.config.trackPerformance || !('performance' in window)) return;

        window.addEventListener('load', () => {
            setTimeout(() => {
                this.trackPerformanceMetrics();
            }, 0);
        });
    }

    /**
     * Track page view
     */
    trackPageView(path = window.location.pathname) {
        this.sessionData.pageViews++;

        const pageViewEvent = {
            type: 'pageview',
            timestamp: Date.now(),
            sessionId: this.sessionId,
            userId: this.userId,
            path: path,
            referrer: document.referrer,
            userAgent: navigator.userAgent,
            screenSize: `${screen.width}x${screen.height}`,
            viewport: `${window.innerWidth}x${window.innerHeight}`
        };

        this.addEvent(pageViewEvent);

        // Send to external analytics (Google Analytics, etc.)
        this.sendToExternalAnalytics('pageview', { path });
    }

    /**
     * Track page leave
     */
    trackPageLeave() {
        this.sessionData.timeOnPage += Date.now() - this.sessionData.lastPageView || 0;
    }

    /**
     * Track page return
     */
    trackPageReturn() {
        this.sessionData.lastPageView = Date.now();
    }

    /**
     * Track click events
     */
    trackClick(e) {
        const target = e.target.closest('a, button, .btn, [data-track]');
        if (!target) return;

        this.sessionData.interactions++;

        const clickEvent = {
            type: 'click',
            timestamp: Date.now(),
            sessionId: this.sessionId,
            userId: this.userId,
            target: {
                tagName: target.tagName,
                className: target.className,
                id: target.id,
                href: target.href,
                text: target.textContent?.trim().substring(0, 100),
                trackData: target.dataset.track
            },
            position: {
                x: e.clientX,
                y: e.clientY
            },
            page: {
                scrollY: window.pageYOffset,
                scrollX: window.pageXOffset
            }
        };

        this.addEvent(clickEvent);

        // Track specific interactions
        this.trackSpecificInteraction(target);
    }

    /**
     * Track specific interactions like CTA clicks
     */
    trackSpecificInteraction(target) {
        const trackData = target.dataset.track;
        const href = target.href;

        if (trackData) {
            this.trackEvent('custom_interaction', { action: trackData });
        }

        if (href && href.includes('github.com')) {
            this.trackEvent('github_visit', { url: href });
        }

        if (target.classList.contains('btn-primary') || target.classList.contains('viral-button')) {
            this.trackEvent('cta_click', {
                text: target.textContent?.trim(),
                type: Array.from(target.classList).find(c => c.includes('btn-'))
            });
        }

        if (target.classList.contains('share-button') || target.onclick?.toString().includes('share')) {
            this.trackEvent('share_attempt', { platform: this.detectSharePlatform(target) });
        }
    }

    /**
     * Detect share platform from button
     */
    detectSharePlatform(button) {
        if (button.classList.contains('share-twitter')) return 'twitter';
        if (button.classList.contains('share-facebook')) return 'facebook';
        if (button.classList.contains('share-linkedin')) return 'linkedin';
        if (button.classList.contains('share-reddit')) return 'reddit';
        return 'unknown';
    }

    /**
     * Track form submission
     */
    trackFormSubmit(e) {
        const form = e.target;

        const formEvent = {
            type: 'form_submit',
            timestamp: Date.now(),
            sessionId: this.sessionId,
            userId: this.userId,
            formId: form.id,
            formClass: form.className,
            formAction: form.action
        };

        this.addEvent(formEvent);
        this.trackEvent('conversion', { type: 'form_submit', formId: form.id });
    }

    /**
     * Setup scroll tracking
     */
    setupScrollTracking() {
        let scrollTimeout;
        const scrollThresholds = [25, 50, 75, 90, 100];
        const reachedThresholds = new Set();

        const handleScroll = () => {
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                const scrollTop = window.pageYOffset;
                const docHeight = document.documentElement.scrollHeight - window.innerHeight;
                const scrollPercent = Math.round((scrollTop / docHeight) * 100);

                this.sessionData.maxScrollDepth = Math.max(this.sessionData.maxScrollDepth, scrollPercent);

                // Track milestone scroll depths
                scrollThresholds.forEach(threshold => {
                    if (scrollPercent >= threshold && !reachedThresholds.has(threshold)) {
                        reachedThresholds.add(threshold);
                        this.trackEvent('scroll_milestone', {
                            depth: threshold,
                            maxDepth: this.sessionData.maxScrollDepth
                        });
                    }
                });
            }, 100);
        };

        window.addEventListener('scroll', handleScroll, { passive: true });
    }

    /**
     * Setup time tracking
     */
    setupTimeTracking() {
        this.sessionData.lastPageView = Date.now();

        // Track time milestones
        const timeMilestones = [30, 60, 120, 300, 600]; // seconds
        const reachedMilestones = new Set();

        setInterval(() => {
            const timeOnPage = Math.floor((Date.now() - this.sessionData.lastPageView) / 1000);

            timeMilestones.forEach(milestone => {
                if (timeOnPage >= milestone && !reachedMilestones.has(milestone)) {
                    reachedMilestones.add(milestone);
                    this.trackEvent('time_milestone', {
                        seconds: milestone,
                        timeOnPage: timeOnPage
                    });
                }
            });
        }, 5000);
    }

    /**
     * Track performance metrics
     */
    trackPerformanceMetrics() {
        if (!('performance' in window)) return;

        const navigation = performance.getEntriesByType('navigation')[0];
        const paintEntries = performance.getEntriesByType('paint');

        const metrics = {
            loadTime: navigation.loadEventEnd - navigation.loadEventStart,
            domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
            firstPaint: paintEntries.find(p => p.name === 'first-paint')?.startTime,
            firstContentfulPaint: paintEntries.find(p => p.name === 'first-contentful-paint')?.startTime,
            fcp: paintEntries.find(p => p.name === 'first-contentful-paint')?.startTime,
            lcp: this.getLCP(),
            cls: this.getCLS(),
            fid: this.getFID()
        };

        const performanceEvent = {
            type: 'performance',
            timestamp: Date.now(),
            sessionId: this.sessionId,
            userId: this.userId,
            metrics: metrics,
            connection: navigator.connection ? {
                effectiveType: navigator.connection.effectiveType,
                downlink: navigator.connection.downlink,
                rtt: navigator.connection.rtt
            } : null
        };

        this.addEvent(performanceEvent);
        this.trackEvent('performance_metrics', metrics);
    }

    /**
     * Get Largest Contentful Paint (simplified)
     */
    getLCP() {
        // Simplified LCP calculation
        return performance.getEntriesByType('largest-contentful-paint')[0]?.startTime || 0;
    }

    /**
     * Get Cumulative Layout Shift (simplified)
     */
    getCLS() {
        // Simplified CLS calculation
        return 0; // Would need proper CLS monitoring setup
    }

    /**
     * Get First Input Delay (simplified)
     */
    getFID() {
        // Simplified FID calculation
        return performance.getEntriesByType('first-input')[0]?.processingStart - performance.getEntriesByType('first-input')[0]?.startTime || 0;
    }

    /**
     * Track custom event
     */
    trackEvent(eventName, data = {}) {
        const event = {
            type: 'custom',
            timestamp: Date.now(),
            sessionId: this.sessionId,
            userId: this.userId,
            eventName: eventName,
            data: data
        };

        this.addEvent(event);

        // Send to external analytics
        this.sendToExternalAnalytics(eventName, data);

        // Console log for debugging
        console.log('📊 Analytics Event:', eventName, data);
    }

    /**
     * Add event to local storage
     */
    addEvent(event) {
        this.events.push(event);

        // Limit events array size
        if (this.events.length > 1000) {
            this.events = this.events.slice(-500);
        }

        // Store in localStorage periodically
        if (this.events.length % 10 === 0) {
            this.saveEvents();
        }
    }

    /**
     * Save events to localStorage
     */
    saveEvents() {
        try {
            localStorage.setItem('pymultiwfn_analytics_events', JSON.stringify(this.events));
            localStorage.setItem('pymultiwfn_session_data', JSON.stringify(this.sessionData));
        } catch (error) {
            console.warn('Failed to save analytics data:', error);
        }
    }

    /**
     * Load stored data
     */
    loadStoredData() {
        try {
            const storedEvents = localStorage.getItem('pymultiwfn_analytics_events');
            if (storedEvents) {
                this.events = JSON.parse(storedEvents);
            }

            const storedSessionData = localStorage.getItem('pymultiwfn_session_data');
            if (storedSessionData) {
                Object.assign(this.sessionData, JSON.parse(storedSessionData));
            }
        } catch (error) {
            console.warn('Failed to load analytics data:', error);
        }
    }

    /**
     * Send events to external analytics services
     */
    sendToExternalAnalytics(eventName, data) {
        // Google Analytics (if available)
        if (typeof gtag !== 'undefined') {
            gtag('event', eventName, data);
        }

        // Other analytics services can be added here
        // Mixpanel, Amplitude, Hotjar, etc.
    }

    /**
     * Flush events (send to server)
     */
    flushEvents() {
        if (this.events.length === 0) return;

        // In a real implementation, this would send events to your analytics server
        console.log('📊 Flushing', this.events.length, 'analytics events');

        // Clear events after flushing
        this.events = [];
        this.saveEvents();
    }

    /**
     * Get analytics summary
     */
    getAnalyticsSummary() {
        return {
            session: this.sessionData,
            totalEvents: this.events.length,
            recentEvents: this.events.slice(-10),
            eventTypes: this.events.reduce((types, event) => {
                types[event.type] = (types[event.type] || 0) + 1;
                return types;
            }, {})
        };
    }

    /**
     * Export analytics data
     */
    exportData() {
        return {
            sessionData: this.sessionData,
            events: this.events,
            exportDate: new Date().toISOString()
        };
    }

    /**
     * Clear all analytics data
     */
    clearData() {
        this.events = [];
        this.sessionData = {
            startTime: Date.now(),
            pageViews: 0,
            interactions: 0,
            timeOnPage: 0,
            scrollDepth: 0,
            maxScrollDepth: 0
        };

        localStorage.removeItem('pymultiwfn_analytics_events');
        localStorage.removeItem('pymultiwfn_session_data');
    }

    /**
     * Load user data from previous sessions
     */
    loadUserData() {
        const userData = localStorage.getItem('pymultiwfn_user_data');
        if (userData) {
            try {
                const parsed = JSON.parse(userData);
                this.trackEvent('returning_user', {
                    sessionsCount: parsed.sessionsCount || 1,
                    lastVisit: parsed.lastVisit
                });

                // Update user data
                parsed.sessionsCount = (parsed.sessionsCount || 1) + 1;
                parsed.lastVisit = new Date().toISOString();
                localStorage.setItem('pymultiwfn_user_data', JSON.stringify(parsed));
            } catch (error) {
                console.warn('Failed to parse user data:', error);
            }
        } else {
            this.trackEvent('new_user', {});
            localStorage.setItem('pymultiwfn_user_data', JSON.stringify({
                sessionsCount: 1,
                firstVisit: new Date().toISOString(),
                lastVisit: new Date().toISOString()
            }));
        }
    }
}

// Export for ES6 modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AnalyticsManager;
}