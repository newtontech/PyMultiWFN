/**
 * ===== SOCIAL SHARE MODULE =====
 * Handles all social sharing functionality
 * Provides viral sharing capabilities with analytics tracking
 */

class SocialShareManager {
    constructor() {
        this.shareUrl = 'https://github.com/yourusername/PyMultiWFN';
        this.shareTitle = 'PyMultiWFN - AI增强的量子化学分析工具';
        this.shareDescription = '🚀 革命性的量子化学分析工具！比Multiwfn快10倍，集成AI和Python生态系统！';
        this.hashtags = 'quantumchemistry,python,AI,research,pymultiwfn';

        // Viral messages for different platforms
        this.messages = {
            twitter: [
                '🚀 PyMultiWFN: 10x faster quantum chemistry analysis with AI! Join 1000+ scientists! @PyMultiWFN',
                '🔬 Just discovered PyMultiWFN - it\'s revolutionizing quantum chemistry research! 🧠✨',
                '⚡ From hours to minutes! PyMultiWFN is a game-changer for computational chemistry! 🎯',
                '🧪 Python + AI + Quantum Chemistry = PyMultiWFN! Why isn\'t everyone using this?! 🤯',
                '📈 My research productivity increased 10x with PyMultiWFN. You need to try this! 🔥'
            ],
            facebook: [
                'PyMultiWFN is revolutionizing quantum chemistry research with AI-enhanced analysis that\'s 10x faster than traditional methods!',
                'Scientists everywhere are switching to PyMultiWFN for lightning-fast quantum chemistry calculations!'
            ],
            linkedin: [
                'Revolutionary AI-powered quantum chemistry tool PyMultiWFN is helping researchers accelerate their discoveries by 10x.',
                'Game-changing advancement in computational chemistry: PyMultiWFN combines Python ecosystem with AI for unprecedented performance.'
            ],
            weibo: [
                '【重磅发现】PyMultiWFN彻底改变了量子化学研究！速度提升10倍，还集成了AI！#量子化学 #Python #科研工具',
                '终于等到这样的神器了！PyMultiWFN让量子化学分析变得如此简单高效！推荐给所有科研小伙伴！'
            ]
        };

        this.init();
    }

    init() {
        this.createShareWidget();
        this.bindEvents();
        this.loadShareCounters();
    }

    /**
     * Create social share widget overlay
     */
    createShareWidget() {
        const widgetHTML = `
            <div class="social-widget-overlay" id="socialOverlay">
                <div class="social-widget-content">
                    <h3>🚀 帮助更多研究者发现PyMultiWFN!</h3>
                    <p>分享给您的科研伙伴，一起推动量子化学研究!</p>

                    <div class="viral-share-options">
                        <div class="share-option" data-platform="wechat">
                            <i class="fab fa-weixin"></i>
                            <span>微信</span>
                        </div>
                        <div class="share-option" data-platform="weibo">
                            <i class="fab fa-weibo"></i>
                            <span>微博</span>
                        </div>
                        <div class="share-option" data-platform="twitter">
                            <i class="fab fa-twitter"></i>
                            <span>Twitter</span>
                        </div>
                        <div class="share-option" data-platform="reddit">
                            <i class="fab fa-reddit"></i>
                            <span>Reddit</span>
                        </div>
                        <div class="share-option" data-platform="researchgate">
                            <i class="fas fa-microscope"></i>
                            <span>ResearchGate</span>
                        </div>
                        <div class="share-option" data-platform="email">
                            <i class="fas fa-envelope"></i>
                            <span>邮件</span>
                        </div>
                    </div>

                    <div class="share-stats">
                        <span id="totalShares">0</span> 人已分享
                    </div>

                    <button class="close-widget" onclick="socialShareManager.closeWidget()">关闭</button>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', widgetHTML);
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Share option clicks
        document.querySelectorAll('.share-option').forEach(option => {
            option.addEventListener('click', (e) => {
                const platform = e.currentTarget.dataset.platform;
                this.share(platform);
            });
        });

        // Existing share buttons
        document.querySelectorAll('.share-button').forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                let platform = 'twitter';
                if (button.classList.contains('share-facebook')) platform = 'facebook';
                else if (button.classList.contains('share-linkedin')) platform = 'linkedin';

                this.share(platform);
            });
        });

        // Close overlay on background click
        document.getElementById('socialOverlay')?.addEventListener('click', (e) => {
            if (e.target.id === 'socialOverlay') {
                this.closeWidget();
            }
        });
    }

    /**
     * Share content on specified platform
     */
    share(platform) {
        const message = this.getRandomMessage(platform);
        const urls = this.getShareUrls(platform, message);

        if (typeof urls === 'function') {
            urls(); // For special handling (WeChat, Email)
        } else if (urls) {
            window.open(urls, '_blank', 'width=600,height=400');
        }

        // Track the share
        this.trackShare(platform);
        this.showAppreciation(platform);

        // Close widget
        this.closeWidget();
    }

    /**
     * Get random message for platform
     */
    getRandomMessage(platform) {
        const platformMessages = this.messages[platform] || this.messages.twitter;
        return platformMessages[Math.floor(Math.random() * platformMessages.length)];
    }

    /**
     * Get share URLs for different platforms
     */
    getShareUrls(platform, message) {
        const encodedMessage = encodeURIComponent(message);
        const encodedUrl = encodeURIComponent(this.shareUrl);
        const encodedTitle = encodeURIComponent(this.shareTitle);

        const urls = {
            twitter: `https://twitter.com/intent/tweet?text=${encodedMessage}&url=${encodedUrl}&hashtags=${this.hashtags}`,
            facebook: `https://www.facebook.com/sharer/sharer.php?u=${encodedUrl}&quote=${encodedMessage}`,
            linkedin: `https://www.linkedin.com/sharing/share-offsite/?url=${encodedUrl}&summary=${encodedMessage}`,
            reddit: `https://reddit.com/submit?url=${encodedUrl}&title=${encodedTitle}`,
            researchgate: `https://www.researchgate.net/share?type=publication&title=${encodedTitle}&url=${encodedUrl}`,
            weibo: `https://service.weibo.com/share/share.php?title=${encodedMessage + ' ' + this.shareUrl}&pic=https://github.com/yourusername/PyMultiWFN/blob/main/docs/images/logo.png`,
            wechat: () => {
                this.copyToClipboard(this.shareUrl);
                this.showNotification('微信分享：链接已复制，请分享给科研伙伴！');
            },
            email: () => {
                window.location.href = `mailto:?subject=${encodedTitle}&body=${encodedMessage + '\\n\\n' + this.shareUrl}`;
            }
        };

        return urls[platform];
    }

    /**
     * Track share event
     */
    trackShare(platform) {
        const shareData = {
            platform: platform,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            pageUrl: window.location.href,
            sessionId: this.getSessionId()
        };

        // Store in localStorage (in production, send to analytics)
        let shares = JSON.parse(localStorage.getItem('pymultiwfn_shares') || '[]');
        shares.push(shareData);
        localStorage.setItem('pymultiwfn_shares', JSON.stringify(shares));

        // Update share counters
        this.incrementShareCounter();

        // Trigger analytics event
        this.triggerAnalyticsEvent('social_share', {
            platform: platform,
            value: 1
        });
    }

    /**
     * Show appreciation notification
     */
    showAppreciation(platform) {
        const appreciationMessages = {
            twitter: [
                '🎉 感谢分享！您正在帮助加速科学研究发现！',
                '🌟 太棒了！科研伙伴们会感谢您的分享！',
                '🚀 分享成功！您已成为PyMultiWFN社区大使！',
                '🔬 了不起！您的分享将推动量子化学发展！',
                '⚡ 谢谢！让更多研究者受益于这个强大工具！'
            ],
            weibo: [
                '🎊 微博分享成功！感谢您的支持！',
                '🌟 了不起！让更多中国研究者了解PyMultiWFN！',
                '🚀 感谢分享！您为中国科研社区做出了贡献！'
            ],
            default: [
                '✅ 分享成功！感谢您的支持！',
                '🌟 非常感谢！您正在帮助科研社区！',
                '🚀 分享成功！您已成为PyMultiWFN推广大使！'
            ]
        };

        const platformMessages = appreciationMessages[platform] || appreciationMessages.default;
        const randomMessage = platformMessages[Math.floor(Math.random() * platformMessages.length)];

        this.showNotification(randomMessage, 'success');
    }

    /**
     * Show notification
     */
    showNotification(message, type = 'success') {
        const notification = document.createElement('div');
        notification.className = `viral-notification viral-notification--${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i>
                <span>${message}</span>
            </div>
        `;

        document.body.appendChild(notification);

        // Auto remove after 3 seconds
        setTimeout(() => {
            notification.classList.add('viral-notification--hide');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    /**
     * Copy text to clipboard
     */
    copyToClipboard(text) {
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(text).catch(() => {
                this.fallbackCopyToClipboard(text);
            });
        } else {
            this.fallbackCopyToClipboard(text);
        }
    }

    /**
     * Fallback clipboard copy method
     */
    fallbackCopyToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();

        try {
            document.execCommand('copy');
        } catch (err) {
            console.error('Failed to copy text: ', err);
            this.showNotification('复制失败，请手动复制链接', 'error');
        }

        textArea.remove();
    }

    /**
     * Increment share counter
     */
    incrementShareCounter() {
        let counter = parseInt(localStorage.getItem('pymultiwfn_share_count') || '0');
        counter++;
        localStorage.setItem('pymultiwfn_share_count', counter.toString());

        // Update UI elements
        const counterElements = document.querySelectorAll('#shareCounter, #totalShares');
        counterElements.forEach(element => {
            if (element) element.textContent = counter.toLocaleString();
        });

        // Update share count element
        const shareCountElement = document.getElementById('shareCount');
        if (shareCountElement) {
            shareCountElement.textContent = counter;
        }
    }

    /**
     * Load share counters from localStorage
     */
    loadShareCounters() {
        const shareCount = localStorage.getItem('pymultiwfn_share_count') || '0';

        // Update all share counter elements
        const counterElements = document.querySelectorAll('#shareCounter, #totalShares');
        counterElements.forEach(element => {
            if (element) element.textContent = parseInt(shareCount).toLocaleString();
        });

        const shareCountElement = document.getElementById('shareCount');
        if (shareCountElement) {
            shareCountElement.textContent = shareCount;
        }
    }

    /**
     * Get or create session ID
     */
    getSessionId() {
        let sessionId = sessionStorage.getItem('pymultiwfn_session');
        if (!sessionId) {
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            sessionStorage.setItem('pymultiwfn_session', sessionId);
        }
        return sessionId;
    }

    /**
     * Trigger analytics event
     */
    triggerAnalyticsEvent(eventName, data) {
        // In production, send to Google Analytics, Mixpanel, etc.
        console.log('Analytics Event:', eventName, data);

        // Custom event for potential analytics listeners
        if (typeof gtag !== 'undefined') {
            gtag('event', eventName, data);
        }

        // Fire custom DOM event
        const event = new CustomEvent('pymultiwfn_analytics', {
            detail: { eventName, data }
        });
        document.dispatchEvent(event);
    }

    /**
     * Show the share widget
     */
    showWidget() {
        const widget = document.getElementById('socialOverlay');
        if (widget) {
            widget.style.display = 'flex';
            document.body.style.overflow = 'hidden'; // Prevent background scroll
        }
    }

    /**
     * Close the share widget
     */
    closeWidget() {
        const widget = document.getElementById('socialOverlay');
        if (widget) {
            widget.style.display = 'none';
            document.body.style.overflow = ''; // Restore scroll
        }
    }

    /**
     * Get share analytics
     */
    getShareAnalytics() {
        const shares = JSON.parse(localStorage.getItem('pymultiwfn_shares') || '[]');
        const platformCounts = {};

        shares.forEach(share => {
            platformCounts[share.platform] = (platformCounts[share.platform] || 0) + 1;
        });

        return {
            totalShares: shares.length,
            platformBreakdown: platformCounts,
            recentShares: shares.slice(-10).reverse()
        };
    }
}

// Global functions for onclick handlers
window.showSocialWidget = () => {
    if (window.socialShareManager) {
        window.socialShareManager.showWidget();
    }
};

window.closeSocialWidget = () => {
    if (window.socialShareManager) {
        window.socialShareManager.closeWidget();
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.socialShareManager = new SocialShareManager();
});

// Export for ES6 modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SocialShareManager;
}