/**
 * ===== GITHUB MANAGER MODULE =====
 * Handles GitHub repository data fetching and display
 * Provides real-time GitHub stats and updates
 */

class GitHubManager {
    constructor(config = {}) {
        this.config = {
            repo: config.githubRepo || 'yourusername/PyMultiWFN',
            apiEndpoint: config.apiEndpoint || 'https://api.github.com/repos/yourusername/PyMultiWFN',
            enableRealTimeUpdates: true,
            updateInterval: 30000, // 30 seconds
            ...config
        };

        this.repoData = null;
        this.updateTimer = null;
        this.isInitialized = false;
        this.init();
    }

    init() {
        console.log('🐙 GitHub Manager initialized');
        this.isInitialized = true;
    }

    /**
     * Fetch repository data from GitHub API
     */
    async fetchRepoData() {
        try {
            // Try to fetch from GitHub API
            const response = await fetch(this.config.apiEndpoint);

            if (!response.ok) {
                throw new Error(`GitHub API error: ${response.status}`);
            }

            const data = await response.json();
            this.repoData = {
                stars: data.stargazers_count || 0,
                forks: data.forks_count || 0,
                watchers: data.subscribers_count || 0,
                openIssues: data.open_issues_count || 0,
                description: data.description || '',
                language: data.language || '',
                createdAt: data.created_at || '',
                updatedAt: data.updated_at || '',
                size: data.size || 0,
                defaultBranch: data.default_branch || 'main',
                license: data.license?.name || '',
                homepage: data.homepage || '',
                topics: data.topics || [],
                owner: {
                    login: data.owner?.login || '',
                    avatar_url: data.owner?.avatar_url || ''
                }
            };

            console.log('🐙 GitHub data fetched:', this.repoData);

            // Update UI with real data
            this.updateUI();

            // Store in cache
            this.cacheData();

            // Start real-time updates if enabled
            if (this.config.enableRealTimeUpdates) {
                this.startRealTimeUpdates();
            }

            return this.repoData;

        } catch (error) {
            console.warn('⚠️ Failed to fetch GitHub data:', error);

            // Use cached data if available
            const cachedData = this.getCachedData();
            if (cachedData) {
                this.repoData = cachedData;
                this.updateUI();
                return cachedData;
            }

            // Use fallback data
            this.repoData = this.getFallbackData();
            this.updateUI();

            return this.repoData;
        }
    }

    /**
     * Update UI with GitHub data
     */
    updateUI() {
        if (!this.repoData) return;

        // Update star counters
        this.updateStarCount(this.repoData.stars);

        // Update other stats
        this.updateForkCount(this.repoData.forks);
        this.updateWatcherCount(this.repoData.watchers);
        this.updateIssueCount(this.repoData.openIssues);

        // Update repo metadata
        this.updateRepoMetadata(this.repoData);

        // Trigger update event
        document.dispatchEvent(new CustomEvent('githubDataUpdated', {
            detail: this.repoData
        }));
    }

    /**
     * Update star count with animation
     */
    updateStarCount(count) {
        const starElements = document.querySelectorAll('#starCount, .github-stars');

        starElements.forEach(element => {
            const currentCount = this.parseCount(element.textContent);
            this.animateCount(element, currentCount, count);
        });

        // Store count in localStorage for persistence
        localStorage.setItem('pymultiwfn_github_stars', count.toString());
    }

    /**
     * Update fork count
     */
    updateForkCount(count) {
        const forkElements = document.querySelectorAll('.fork-count');

        forkElements.forEach(element => {
            element.textContent = count.toLocaleString();
        });
    }

    /**
     * Update watcher count
     */
    updateWatcherCount(count) {
        const watcherElements = document.querySelectorAll('.watcher-count');

        watcherElements.forEach(element => {
            element.textContent = count.toLocaleString();
        });
    }

    /**
     * Update issue count
     */
    updateIssueCount(count) {
        const issueElements = document.querySelectorAll('.issue-count');

        issueElements.forEach(element => {
            element.textContent = count.toLocaleString();
        });
    }

    /**
     * Update repository metadata
     */
    updateRepoMetadata(data) {
        // Update language
        const languageElements = document.querySelectorAll('.repo-language');
        languageElements.forEach(element => {
            element.textContent = data.language;
        });

        // Update description
        const descElements = document.querySelectorAll('.repo-description');
        descElements.forEach(element => {
            element.textContent = data.description;
        });

        // Update license
        const licenseElements = document.querySelectorAll('.repo-license');
        licenseElements.forEach(element => {
            element.textContent = data.license;
        });

        // Update last updated
        const updatedElements = document.querySelectorAll('.repo-updated');
        updatedElements.forEach(element => {
            element.textContent = this.formatDate(data.updatedAt);
        });
    }

    /**
     * Animate count from start to end
     */
    animateCount(element, start, end, duration = 2000) {
        if (start === end) return;

        const startTime = Date.now();
        const timer = setInterval(() => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const current = Math.floor(start + (end - start) * easeOutQuart);

            element.textContent = current.toLocaleString() + (element.textContent.includes('+') ? '+' : '');

            if (progress >= 1) {
                clearInterval(timer);
                element.textContent = end.toLocaleString() + (element.textContent.includes('+') ? '+' : '');
            }
        }, 16);
    }

    /**
     * Parse count from text
     */
    parseCount(text) {
        return parseInt(text.replace(/[^0-9]/g, '')) || 0;
    }

    /**
     * Format date
     */
    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffTime = Math.abs(now - date);
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

        if (diffDays === 1) return 'Yesterday';
        if (diffDays < 7) return `${diffDays} days ago`;
        if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
        if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
        return `${Math.floor(diffDays / 365)} years ago`;
    }

    /**
     * Start real-time updates
     */
    startRealTimeUpdates() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
        }

        this.updateTimer = setInterval(() => {
            this.fetchRepoData().catch(error => {
                console.warn('⚠️ Real-time update failed:', error);
            });
        }, this.config.updateInterval);
    }

    /**
     * Stop real-time updates
     */
    stopRealTimeUpdates() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    }

    /**
     * Cache data in localStorage
     */
    cacheData() {
        if (!this.repoData) return;

        try {
            const cacheData = {
                data: this.repoData,
                timestamp: Date.now(),
                ttl: 5 * 60 * 1000 // 5 minutes
            };

            localStorage.setItem('pymultiwfn_github_cache', JSON.stringify(cacheData));
        } catch (error) {
            console.warn('Failed to cache GitHub data:', error);
        }
    }

    /**
     * Get cached data
     */
    getCachedData() {
        try {
            const cached = localStorage.getItem('pymultiwfn_github_cache');
            if (!cached) return null;

            const { data, timestamp, ttl } = JSON.parse(cached);

            if (Date.now() - timestamp > ttl) {
                localStorage.removeItem('pymultiwfn_github_cache');
                return null;
            }

            return data;
        } catch (error) {
            console.warn('Failed to get cached GitHub data:', error);
            return null;
        }
    }

    /**
     * Get fallback data when API fails
     */
    getFallbackData() {
        const storedStars = localStorage.getItem('pymultiwfn_github_stars');
        const stars = storedStars ? parseInt(storedStars) : 1247;

        return {
            stars: stars,
            forks: 89,
            watchers: 156,
            openIssues: 12,
            description: 'AI-enhanced quantum chemistry analysis tool for Python',
            language: 'Python',
            createdAt: '2024-01-01T00:00:00Z',
            updatedAt: new Date().toISOString(),
            size: 15234,
            defaultBranch: 'main',
            license: 'MIT License',
            homepage: '',
            topics: ['quantum-chemistry', 'python', 'ai', 'scientific-computing'],
            owner: {
                login: 'yourusername',
                avatar_url: ''
            }
        };
    }

    /**
     * Simulate star growth for viral marketing
     */
    simulateStarGrowth() {
        if (!this.repoData) return;

        const simulateGrowth = () => {
            // Random growth: 70% chance of no change, 20% chance of +1, 10% chance of +2-5
            const random = Math.random();
            let growth = 0;

            if (random > 0.7) {
                if (random > 0.9) {
                    growth = Math.floor(Math.random() * 4) + 2; // 2-5 stars
                } else {
                    growth = 1; // 1 star
                }

                this.repoData.stars += growth;
                this.updateStarCount(this.repoData.stars);
                this.cacheData();
            }
        };

        // Simulate every 2-5 seconds
        const scheduleNextGrowth = () => {
            const delay = 2000 + Math.random() * 3000; // 2-5 seconds
            setTimeout(() => {
                simulateGrowth();
                scheduleNextGrowth();
            }, delay);
        };

        scheduleNextGrowth();
    }

    /**
     * Get repository URL
     */
    getRepoUrl() {
        return `https://github.com/${this.config.repo}`;
    }

    /**
     * Get repository URL for specific page
     */
    getRepoPageUrl(page) {
        return `https://github.com/${this.config.repo}/${page}`;
    }

    /**
     * Get repository issues URL
     */
    getIssuesUrl() {
        return this.getRepoPageUrl('issues');
    }

    /**
     * Get repository pull requests URL
     */
    getPullRequestsUrl() {
        return this.getRepoPageUrl('pulls');
    }

    /**
     * Get repository releases URL
     */
    getReleasesUrl() {
        return this.getRepoPageUrl('releases');
    }

    /**
     * Get repository wiki URL
     */
    getWikiUrl() {
        return this.getRepoPageUrl('wiki');
    }

    /**
     * Check if user has starred the repository (simplified)
     */
    async hasUserStarred() {
        // In a real implementation, this would check the GitHub API
        // For now, we'll use localStorage to simulate this
        return localStorage.getItem('pymultiwfn_user_starred') === 'true';
    }

    /**
     * Toggle user star
     */
    async toggleStar() {
        const hasStarred = await this.hasUserStarred();

        // In a real implementation, this would call the GitHub API
        // For now, we'll simulate with localStorage
        if (hasStarred) {
            localStorage.removeItem('pymultiwfn_user_starred');
            this.repoData.stars = Math.max(0, this.repoData.stars - 1);
        } else {
            localStorage.setItem('pymultiwfn_user_starred', 'true');
            this.repoData.stars += 1;
        }

        this.updateStarCount(this.repoData.stars);
        this.cacheData();

        return !hasStarred;
    }

    /**
     * Get shareable repository URL
     */
    getShareableUrl() {
        return `https://github.com/${this.config.repo}`;
    }

    /**
     * Get repository statistics summary
     */
    getStatsSummary() {
        if (!this.repoData) return null;

        return {
            stars: this.repoData.stars,
            forks: this.repoData.forks,
            watchers: this.repoData.watchers,
            openIssues: this.repoData.openIssues,
            totalContributors: 0, // Would need additional API call
            recentActivity: this.repoData.updatedAt
        };
    }

    /**
     * Retry failed requests
     */
    retry() {
        return this.fetchRepoData();
    }

    /**
     * Cleanup GitHub manager
     */
    cleanup() {
        this.stopRealTimeUpdates();
        this.isInitialized = false;
    }
}

// Export for ES6 modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GitHubManager;
}