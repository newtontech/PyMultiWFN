/**
 * ===== ANIMATIONS MODULE =====
 * Handles all animations, transitions, and interactive effects
 * Provides smooth user experience with performance optimizations
 */

class AnimationManager {
    constructor() {
        this.isReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        this.intersectionObserver = null;
        this.counters = new Map();
        this.init();
    }

    init() {
        this.setupIntersectionObserver();
        this.bindScrollEvents();
        this.initCounters();
        this.setupParallaxEffects();
        this.bindHoverEffects();
    }

    /**
     * Setup intersection observer for scroll animations
     */
    setupIntersectionObserver() {
        if (this.isReducedMotion) return;

        const options = {
            root: null,
            rootMargin: '0px',
            threshold: [0.1, 0.5]
        };

        this.intersectionObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.animateElement(entry.target, entry.intersectionRatio);
                }
            });
        }, options);

        // Observe elements that should animate
        this.observeAnimatedElements();
    }

    /**
     * Observe elements that should have scroll animations
     */
    observeAnimatedElements() {
        const animatedElements = document.querySelectorAll(`
            .card,
            .testimonial-card,
            .feature-card,
            .conversion-feature,
            .comparison-row,
            .stat-item,
            .section-title,
            .hero-title,
            .hero-subtitle,
            .hero-description
        `);

        animatedElements.forEach(element => {
            // Add initial state
            element.classList.add('animate-on-scroll');
            element.style.opacity = '0';
            element.style.transform = 'translateY(30px)';

            this.intersectionObserver.observe(element);
        });
    }

    /**
     * Animate individual element
     */
    animateElement(element, intersectionRatio) {
        if (element.dataset.animated === 'true') return;

        const delay = element.dataset.animationDelay || 0;
        const duration = element.dataset.animationDuration || 800;

        setTimeout(() => {
            element.style.transition = `all ${duration}ms cubic-bezier(0.4, 0, 0.2, 1)`;
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';

            // Add animation-specific classes
            if (element.classList.contains('card')) {
                this.animateCard(element);
            } else if (element.classList.contains('stat-item')) {
                this.animateStatItem(element);
            } else if (element.classList.contains('feature-card')) {
                this.animateFeatureCard(element);
            }

            element.dataset.animated = 'true';
        }, delay);
    }

    /**
     * Animate cards with staggered effect
     */
    animateCard(card) {
        card.style.transform = 'translateY(0) scale(1)';

        // Add subtle glow effect
        setTimeout(() => {
            card.style.boxShadow = '0 20px 40px rgba(0, 0, 0, 0.1)';
        }, 200);
    }

    /**
     * Animate stat items with counter effect
     */
    animateStatItem(statItem) {
        const numberElement = statItem.querySelector('.stat-number');
        if (numberElement && !numberElement.dataset.counted) {
            this.animateCounter(numberElement);
            numberElement.dataset.counted = 'true';
        }
    }

    /**
     * Animate feature cards with icon animation
     */
    animateFeatureCard(featureCard) {
        const icon = featureCard.querySelector('.feature-icon');
        if (icon) {
            icon.style.transform = 'scale(1) rotate(360deg)';
            icon.style.transition = 'transform 0.8s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
        }
    }

    /**
     * Initialize animated counters
     */
    initCounters() {
        const counterElements = document.querySelectorAll('.stat-number, #starCount, #userCount');
        counterElements.forEach(element => {
            const finalValue = this.parseNumber(element.textContent);
            if (finalValue > 0) {
                this.counters.set(element, {
                    start: 0,
                    end: finalValue,
                    current: 0,
                    duration: 2000,
                    startTime: null,
                    formatted: element.textContent.includes(',') || element.textContent.includes('+')
                });
            }
        });
    }

    /**
     * Parse number from text element
     */
    parseNumber(text) {
        return parseInt(text.replace(/[^0-9]/g, '')) || 0;
    }

    /**
     * Animate counter
     */
    animateCounter(element) {
        const counter = this.counters.get(element);
        if (!counter) return;

        const animate = (timestamp) => {
            if (!counter.startTime) counter.startTime = timestamp;
            const progress = Math.min((timestamp - counter.startTime) / counter.duration, 1);

            // Easing function
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            counter.current = Math.floor(counter.start + (counter.end - counter.start) * easeOutQuart);

            // Format number
            if (counter.formatted) {
                element.textContent = counter.current.toLocaleString() + (element.textContent.includes('+') ? '+' : '');
            } else {
                element.textContent = counter.current;
            }

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                // Trigger any callbacks
                this.onCounterComplete(element, counter.end);
            }
        };

        requestAnimationFrame(animate);
    }

    /**
     * Handle counter completion
     */
    onCounterComplete(element, finalValue) {
        // Add pulse animation
        element.style.transform = 'scale(1.1)';
        setTimeout(() => {
            element.style.transform = 'scale(1)';
        }, 200);

        // Trigger custom event
        const event = new CustomEvent('counterComplete', {
            detail: { element, finalValue }
        });
        document.dispatchEvent(event);
    }

    /**
     * Setup parallax effects
     */
    setupParallaxEffects() {
        if (this.isReducedMotion) return;

        const parallaxElements = document.querySelectorAll('.hero, .hero::before');

        const handleParallax = () => {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.5;

            parallaxElements.forEach(element => {
                element.style.transform = `translateY(${rate}px)`;
            });
        };

        let ticking = false;
        const requestTick = () => {
            if (!ticking) {
                window.requestAnimationFrame(handleParallax);
                ticking = true;
                setTimeout(() => { ticking = false; }, 16);
            }
        };

        window.addEventListener('scroll', requestTick);
    }

    /**
     * Bind scroll events
     */
    bindScrollEvents() {
        // Scroll progress indicator
        this.createScrollProgress();

        // Smooth scroll for anchor links
        this.bindSmoothScroll();

        // Header scroll effects
        this.bindHeaderScroll();
    }

    /**
     * Create scroll progress indicator
     */
    createScrollProgress() {
        const progressBar = document.createElement('div');
        progressBar.id = 'scrollProgress';
        progressBar.innerHTML = '<div class="scroll-progress-bar"></div>';
        progressBar.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: rgba(0, 0, 0, 0.1);
            z-index: 9999;
        `;

        const bar = progressBar.querySelector('.scroll-progress-bar');
        bar.style.cssText = `
            height: 100%;
            background: linear-gradient(90deg, #1890ff, #40a9ff);
            width: 0%;
            transform-origin: left;
            transition: transform 0.3s ease;
        `;

        document.body.appendChild(progressBar);

        // Update progress on scroll
        const updateProgress = () => {
            const scrollTop = window.pageYOffset;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPercent = (scrollTop / docHeight) * 100;
            bar.style.transform = `scaleX(${scrollPercent / 100})`;
        };

        window.addEventListener('scroll', updateProgress, { passive: true });
    }

    /**
     * Bind smooth scroll for anchor links
     */
    bindSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(anchor.getAttribute('href'));
                if (target) {
                    const headerOffset = 80; // Account for fixed header
                    const targetPosition = target.offsetTop - headerOffset;

                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });

                    // Update active nav state
                    this.updateActiveNavLink(anchor.getAttribute('href'));
                }
            });
        });
    }

    /**
     * Bind header scroll effects
     */
    bindHeaderScroll() {
        const header = document.querySelector('.header');
        if (!header) return;

        let lastScrollTop = 0;
        const threshold = 100;

        const handleHeaderScroll = () => {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

            if (scrollTop > threshold) {
                header.style.background = 'rgba(24, 144, 255, 0.95)';
                header.style.backdropFilter = 'blur(10px)';
                header.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.1)';
            } else {
                header.style.background = 'linear-gradient(135deg, #1890ff 0%, #096dd9 100%)';
                header.style.backdropFilter = 'none';
                header.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';
            }

            lastScrollTop = scrollTop;
        };

        window.addEventListener('scroll', handleHeaderScroll, { passive: true });
    }

    /**
     * Update active navigation link
     */
    updateActiveNavLink(targetId) {
        document.querySelectorAll('.nav-item').forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === targetId) {
                link.classList.add('active');
            }
        });
    }

    /**
     * Bind hover effects
     */
    bindHoverEffects() {
        // Button hover effects
        document.querySelectorAll('.btn, .viral-button').forEach(button => {
            this.addButtonHoverEffect(button);
        });

        // Card hover effects
        document.querySelectorAll('.card, .testimonial-card').forEach(card => {
            this.addCardHoverEffect(card);
        });

        // Feature card hover effects
        document.querySelectorAll('.feature-card').forEach(card => {
            this.addFeatureCardHoverEffect(card);
        });
    }

    /**
     * Add hover effect to buttons
     */
    addButtonHoverEffect(button) {
        button.addEventListener('mouseenter', () => {
            if (!this.isReducedMotion) {
                button.style.transform = 'translateY(-2px) scale(1.05)';
            }
        });

        button.addEventListener('mouseleave', () => {
            button.style.transform = 'translateY(0) scale(1)';
        });
    }

    /**
     * Add hover effect to cards
     */
    addCardHoverEffect(card) {
        card.addEventListener('mouseenter', () => {
            if (!this.isReducedMotion) {
                card.style.transform = 'translateY(-5px) scale(1.02)';
                card.style.boxShadow = '0 20px 40px rgba(0, 0, 0, 0.15)';
            }
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0) scale(1)';
            card.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.1)';
        });
    }

    /**
     * Add hover effect to feature cards
     */
    addFeatureCardHoverEffect(card) {
        card.addEventListener('mouseenter', () => {
            if (!this.isReducedMotion) {
                const icon = card.querySelector('.feature-icon');
                if (icon) {
                    icon.style.transform = 'scale(1.1) rotate(15deg)';
                }
                card.style.transform = 'translateY(-8px)';
            }
        });

        card.addEventListener('mouseleave', () => {
            const icon = card.querySelector('.feature-icon');
            if (icon) {
                icon.style.transform = 'scale(1) rotate(0deg)';
            }
            card.style.transform = 'translateY(0)';
        });
    }

    /**
     * Create floating animation for elements
     */
    addFloatingAnimation(element, options = {}) {
        const {
            duration = 3000,
            amplitude = 10,
            delay = 0
        } = options;

        if (this.isReducedMotion) return;

        element.style.animation = `float ${duration}ms ease-in-out ${delay}ms infinite`;

        // Add keyframes if not already added
        if (!document.querySelector('#float-keyframes')) {
            const style = document.createElement('style');
            style.id = 'float-keyframes';
            style.textContent = `
                @keyframes float {
                    0%, 100% { transform: translateY(0px); }
                    50% { transform: translateY(-${amplitude}px); }
                }
            `;
            document.head.appendChild(style);
        }
    }

    /**
     * Create pulse animation for elements
     */
    addPulseAnimation(element, options = {}) {
        const {
            duration = 2000,
            scale = 1.05,
            delay = 0
        } = options;

        if (this.isReducedMotion) return;

        element.style.animation = `pulse ${duration}ms ease-in-out ${delay}ms infinite`;

        // Add keyframes if not already added
        if (!document.querySelector('#pulse-keyframes')) {
            const style = document.createElement('style');
            style.id = 'pulse-keyframes';
            style.textContent = `
                @keyframes pulse {
                    0%, 100% { transform: scale(1); }
                    50% { transform: scale(${scale}); }
                }
            `;
            document.head.appendChild(style);
        }
    }

    /**
     * Animate GitHub stars counter with random increments
     */
    animateGitHubStars() {
        const starElement = document.getElementById('starCount');
        if (!starElement) return;

        let stars = parseInt(starElement.textContent.replace(/[^0-9]/g, '')) || 1247;
        const targetStars = 10000;
        const increment = () => {
            if (stars < targetStars && Math.random() > 0.7) {
                stars += Math.floor(Math.random() * 3) + 1;
                starElement.textContent = stars.toLocaleString() + '+';
            }
        };

        setInterval(increment, 2000);
    }

    /**
     * Cleanup animations
     */
    cleanup() {
        if (this.intersectionObserver) {
            this.intersectionObserver.disconnect();
        }

        // Remove event listeners
        window.removeEventListener('scroll', this.handleScroll);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.animationManager = new AnimationManager();

    // Start GitHub stars animation after a delay
    setTimeout(() => {
        window.animationManager.animateGitHubStars();
    }, 2000);
});

// Export for ES6 modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AnimationManager;
}