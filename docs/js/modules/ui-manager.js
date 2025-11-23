/**
 * ===== UI MANAGER MODULE =====
 * Handles UI interactions, responsive design, and user experience
 * Provides smooth interface without loading screens
 */

class UIManager {
    constructor(config = {}) {
        this.config = {
            enableSmoothScroll: true,
            enableLazyLoading: true,
            enableResponsiveMenus: true,
            ...config
        };

        this.isInitialized = false;
        this.init();
    }

    init() {
        this.setupResponsiveNavigation();
        this.setupSmoothScrolling();
        this.setupImageLazyLoading();
        this.setupMobileOptimizations();
        this.setupAccessibilityFeatures();
        this.isInitialized = true;

        console.log('🎨 UI Manager initialized');
    }

    /**
     * Setup responsive navigation for mobile
     */
    setupResponsiveNavigation() {
        if (!this.config.enableResponsiveMenus) return;

        // Create mobile menu toggle if needed
        this.createMobileMenuToggle();

        // Handle window resize
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.handleViewportChange();
            }, 250);
        });

        // Initial viewport check
        this.handleViewportChange();
    }

    /**
     * Create mobile menu toggle
     */
    createMobileMenuToggle() {
        const header = document.querySelector('.header');
        const nav = document.querySelector('.nav-menu');

        if (!header || !nav) return;

        // Check if mobile menu is needed
        if (window.innerWidth <= 768) {
            // Create hamburger menu button
            const menuToggle = document.createElement('button');
            menuToggle.className = 'mobile-menu-toggle';
            menuToggle.innerHTML = '<i class="fas fa-bars"></i>';
            menuToggle.setAttribute('aria-label', 'Toggle navigation menu');

            menuToggle.addEventListener('click', () => {
                nav.classList.toggle('mobile-menu-open');
                menuToggle.innerHTML = nav.classList.contains('mobile-menu-open')
                    ? '<i class="fas fa-times"></i>'
                    : '<i class="fas fa-bars"></i>';
            });

            header.insertBefore(menuToggle, nav);

            // Add mobile menu styles if not already present
            this.addMobileMenuStyles();
        }
    }

    /**
     * Add mobile menu styles
     */
    addMobileMenuStyles() {
        if (document.querySelector('#mobile-menu-styles')) return;

        const styles = document.createElement('style');
        styles.id = 'mobile-menu-styles';
        styles.textContent = `
            .mobile-menu-toggle {
                display: none;
                background: none;
                border: none;
                color: white;
                font-size: 1.5rem;
                cursor: pointer;
                padding: 0.5rem;
                margin-left: 1rem;
            }

            @media (max-width: 768px) {
                .mobile-menu-toggle {
                    display: block;
                }

                .nav-menu {
                    position: fixed;
                    top: 64px;
                    left: -100%;
                    width: 100%;
                    height: calc(100vh - 64px);
                    background: rgba(24, 144, 255, 0.98);
                    backdrop-filter: blur(10px);
                    flex-direction: column;
                    align-items: stretch;
                    padding: 2rem;
                    transition: left 0.3s ease;
                    z-index: 999;
                }

                .nav-menu.mobile-menu-open {
                    left: 0;
                }

                .nav-item {
                    text-align: center;
                    padding: 1rem;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                    margin: 0;
                }

                .nav-item:last-child {
                    border-bottom: none;
                }
            }
        `;

        document.head.appendChild(styles);
    }

    /**
     * Handle viewport changes
     */
    handleViewportChange() {
        const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
        const nav = document.querySelector('.nav-menu');

        if (window.innerWidth <= 768) {
            // Mobile view
            if (!mobileMenuToggle) {
                this.createMobileMenuToggle();
            }
        } else {
            // Desktop view
            if (mobileMenuToggle) {
                mobileMenuToggle.remove();
            }
            if (nav) {
                nav.classList.remove('mobile-menu-open');
            }
        }
    }

    /**
     * Setup smooth scrolling for anchor links
     */
    setupSmoothScrolling() {
        if (!this.config.enableSmoothScroll) return;

        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = anchor.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);

                if (targetElement) {
                    const headerOffset = 80; // Account for fixed header
                    const targetPosition = targetElement.offsetTop - headerOffset;

                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });

                    // Update URL without scrolling
                    history.pushState(null, null, `#${targetId}`);

                    // Update active navigation
                    this.updateActiveNavigation(targetId);
                }
            });
        });

        // Handle scroll-based navigation updates
        this.setupScrollNavigation();
    }

    /**
     * Setup scroll-based navigation highlighting
     */
    setupScrollNavigation() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.nav-item[href^="#"]');

        if (sections.length === 0 || navLinks.length === 0) return;

        const updateActiveSection = () => {
            let currentSection = '';
            const scrollPosition = window.pageYOffset + 150; // Offset for header

            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                const sectionHeight = section.offsetHeight;

                if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                    currentSection = section.id;
                }
            });

            // Update active nav items
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === `#${currentSection}`) {
                    link.classList.add('active');
                }
            });
        };

        // Throttled scroll handler
        let scrollTimeout;
        window.addEventListener('scroll', () => {
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(updateActiveSection, 50);
        });

        // Initial call
        updateActiveSection();
    }

    /**
     * Update active navigation item
     */
    updateActiveNavigation(targetId) {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('href') === `#${targetId}`) {
                item.classList.add('active');
            }
        });
    }

    /**
     * Setup lazy loading for images
     */
    setupImageLazyLoading() {
        if (!this.config.enableLazyLoading || !('IntersectionObserver' in window)) {
            this.loadAllImages();
            return;
        }

        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    this.loadImage(img);
                    imageObserver.unobserve(img);
                }
            });
        }, {
            rootMargin: '50px 0px',
            threshold: 0.01
        });

        // Observe all images with data-src
        document.querySelectorAll('img[data-src]').forEach(img => {
            imageObserver.observe(img);
        });
    }

    /**
     * Load single image
     */
    loadImage(img) {
        const src = img.dataset.src;
        if (!src) return;

        img.src = src;
        img.classList.add('loaded');
        img.removeAttribute('data-src');
    }

    /**
     * Load all images immediately (fallback)
     */
    loadAllImages() {
        document.querySelectorAll('img[data-src]').forEach(img => {
            this.loadImage(img);
        });
    }

    /**
     * Setup mobile optimizations
     */
    setupMobileOptimizations() {
        // Optimize touch events for mobile
        if ('ontouchstart' in window) {
            document.body.classList.add('touch-device');
        }

        // Add tap effects for mobile
        this.addMobileTapEffects();

        // Optimize viewport handling
        this.optimizeViewport();

        // Handle orientation changes
        window.addEventListener('orientationchange', () => {
            setTimeout(() => {
                this.handleViewportChange();
            }, 100);
        });
    }

    /**
     * Add tap effects for mobile devices
     */
    addMobileTapEffects() {
        const style = document.createElement('style');
        style.textContent = `
            .touch-device .btn,
            .touch-device .card,
            .touch-device .feature-card {
                -webkit-tap-highlight-color: rgba(24, 144, 255, 0.1);
            }

            .touch-device .btn:active {
                transform: scale(0.98);
            }
        `;
        document.head.appendChild(style);
    }

    /**
     * Optimize viewport settings
     */
    optimizeViewport() {
        let viewport = document.querySelector('meta[name="viewport"]');
        if (!viewport) {
            viewport = document.createElement('meta');
            viewport.name = 'viewport';
            document.head.appendChild(viewport);
        }

        viewport.content = 'width=device-width, initial-scale=1.0, viewport-fit=cover';
    }

    /**
     * Setup accessibility features
     */
    setupAccessibilityFeatures() {
        // Add ARIA labels to interactive elements
        this.addAriaLabels();

        // Setup keyboard navigation
        this.setupKeyboardNavigation();

        // Setup focus management
        this.setupFocusManagement();

        // Add skip links
        this.addSkipLinks();
    }

    /**
     * Add ARIA labels where needed
     */
    addAriaLabels() {
        // Add labels to social share buttons
        document.querySelectorAll('.share-button').forEach((button, index) => {
            const platform = button.classList.contains('share-twitter') ? 'Twitter' :
                           button.classList.contains('share-facebook') ? 'Facebook' :
                           button.classList.contains('share-linkedin') ? 'LinkedIn' : 'Social media';

            if (!button.getAttribute('aria-label')) {
                button.setAttribute('aria-label', `Share on ${platform}`);
            }
        });

        // Add labels to navigation links
        document.querySelectorAll('.nav-item').forEach(link => {
            if (!link.getAttribute('aria-label') && link.textContent.trim()) {
                link.setAttribute('aria-label', `Navigate to ${link.textContent.trim()} section`);
            }
        });
    }

    /**
     * Setup keyboard navigation
     */
    setupKeyboardNavigation() {
        // Add keyboard shortcuts info
        document.addEventListener('keydown', (e) => {
            // Alt + M for mobile menu toggle
            if (e.altKey && e.key === 'm') {
                const mobileToggle = document.querySelector('.mobile-menu-toggle');
                if (mobileToggle) {
                    mobileToggle.click();
                }
            }
        });

        // Ensure focus stays within mobile menu when open
        const nav = document.querySelector('.nav-menu');
        if (nav) {
            nav.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    nav.classList.remove('mobile-menu-open');
                    document.querySelector('.mobile-menu-toggle')?.focus();
                }
            });
        }
    }

    /**
     * Setup focus management
     */
    setupFocusManagement() {
        // Improve focus visibility
        const style = document.createElement('style');
        style.textContent = `
            :focus {
                outline: 2px solid #1890ff;
                outline-offset: 2px;
            }

            .btn:focus,
            .nav-item:focus {
                outline-offset: 1px;
            }

            /* Better focus for keyboard users */
            .keyboard-user :focus {
                outline: 3px solid #1890ff;
                outline-offset: 2px;
            }
        `;
        document.head.appendChild(style);

        // Detect keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                document.body.classList.add('keyboard-user');
            }
        });

        document.addEventListener('mousedown', () => {
            document.body.classList.remove('keyboard-user');
        });
    }

    /**
     * Add skip links for accessibility
     */
    addSkipLinks() {
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.className = 'skip-link';
        skipLink.textContent = 'Skip to main content';
        skipLink.setAttribute('aria-label', 'Skip to main content');

        const style = document.createElement('style');
        style.textContent = `
            .skip-link {
                position: absolute;
                top: -40px;
                left: 6px;
                background: #1890ff;
                color: white;
                padding: 8px;
                text-decoration: none;
                z-index: 10000;
                border-radius: 4px;
            }

            .skip-link:focus {
                top: 6px;
            }
        `;

        document.head.appendChild(style);
        document.body.insertBefore(skipLink, document.body.firstChild);
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'info', duration = 3000) {
        const toast = document.createElement('div');
        toast.className = `toast toast--${type}`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'polite');

        const icon = type === 'success' ? 'check-circle' :
                   type === 'error' ? 'exclamation-circle' :
                   type === 'warning' ? 'exclamation-triangle' : 'info-circle';

        toast.innerHTML = `
            <div class="toast-content">
                <i class="fas fa-${icon}"></i>
                <span>${message}</span>
                <button class="toast-close" aria-label="Close notification">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        // Add styles if not present
        if (!document.querySelector('#toast-styles')) {
            const styles = document.createElement('style');
            styles.id = 'toast-styles';
            styles.textContent = `
                .toast {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                    padding: 16px;
                    min-width: 300px;
                    max-width: 500px;
                    z-index: 10001;
                    transform: translateX(100%);
                    transition: transform 0.3s ease;
                }

                .toast.show {
                    transform: translateX(0);
                }

                .toast--success { border-left: 4px solid #52c41a; }
                .toast--error { border-left: 4px solid #ff4d4f; }
                .toast--warning { border-left: 4px solid #faad14; }
                .toast--info { border-left: 4px solid #1890ff; }

                .toast-content {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }

                .toast-content i {
                    flex-shrink: 0;
                    font-size: 1.2em;
                }

                .toast--success i { color: #52c41a; }
                .toast--error i { color: #ff4d4f; }
                .toast--warning i { color: #faad14; }
                .toast--info i { color: #1890ff; }

                .toast-content span {
                    flex: 1;
                    color: #333;
                }

                .toast-close {
                    background: none;
                    border: none;
                    color: #999;
                    cursor: pointer;
                    padding: 4px;
                    border-radius: 4px;
                    transition: all 0.2s ease;
                }

                .toast-close:hover {
                    background: #f5f5f5;
                    color: #666;
                }

                @media (max-width: 768px) {
                    .toast {
                        left: 20px;
                        right: 20px;
                        bottom: 20px;
                        max-width: none;
                    }
                }
            `;
            document.head.appendChild(styles);
        }

        document.body.appendChild(toast);

        // Show toast
        setTimeout(() => toast.classList.add('show'), 10);

        // Setup close button
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => this.removeToast(toast));

        // Auto remove
        if (duration > 0) {
            setTimeout(() => this.removeToast(toast), duration);
        }

        return toast;
    }

    /**
     * Remove toast notification
     */
    removeToast(toast) {
        toast.classList.remove('show');
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 300);
    }

    /**
     * Handle resize events
     */
    handleResize() {
        this.handleViewportChange();

        // Reinitialize any size-dependent components
        if (window.innerWidth <= 768) {
            this.optimizeForMobile();
        } else {
            this.optimizeForDesktop();
        }
    }

    /**
     * Optimize for mobile
     */
    optimizeForMobile() {
        document.body.classList.add('mobile-optimized');
        document.body.classList.remove('desktop-optimized');
    }

    /**
     * Optimize for desktop
     */
    optimizeForDesktop() {
        document.body.classList.add('desktop-optimized');
        document.body.classList.remove('mobile-optimized');
    }

    /**
     * Cleanup UI manager
     */
    cleanup() {
        // Remove event listeners
        window.removeEventListener('resize', this.handleViewportChange);

        // Clean up any dynamic elements
        document.querySelectorAll('.mobile-menu-toggle').forEach(el => el.remove());
        document.querySelectorAll('.toast').forEach(el => el.remove());
        document.querySelectorAll('.skip-link').forEach(el => el.remove());

        this.isInitialized = false;
    }
}

// Export for ES6 modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIManager;
}