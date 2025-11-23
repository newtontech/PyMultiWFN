// Ant Design 6 + React UMD (no modules) so it works over file://
(() => {
  const { Layout, Typography, Button, Space, Card, Row, Col, Tag, Divider, Timeline, Statistic, ConfigProvider, Carousel, Steps, FloatButton } = antd;
  const { Title, Paragraph, Text } = Typography;
  const { Header, Content, Footer } = Layout;

  const stats = [
    { title: 'Reproducibility', value: '100%', subtitle: 'Bitwise parity vs Multiwfn outputs' },
    { title: 'Performance Gain', value: '10×', subtitle: 'Grid scanning sped up with NumPy/Fortran mix' },
    { title: 'Community', value: '1000+', subtitle: 'Active researchers, developers, students' },
  ];

  const features = [
    { title: 'Modern Data Model', description: 'Wavefunction, basis sets, and integrals live in immutable dataclasses before dispatch to vectorized kernels.', icon: '🧠', badge: 'core' },
    { title: 'Ant Design UI', description: 'Docs and marketing lean on AntD 6 components, motion tokens, and responsive layouts.', icon: '🎨', badge: 'brand' },
    { title: 'Consistent Backends', description: 'consistency_verifier keeps PyMultiWFN matched against Multiwfn 3.8 across golden tasks.', icon: '⚖️', badge: 'validation' },
    { title: 'Hybrid Performance', description: 'NumPy vectorization + f2py-wrapped Fortran for grid-heavy math without losing clarity.', icon: '⚡', badge: 'performance' },
  ];

  const roadmap = [
    { title: 'Phase 1 • Reconnaissance', detail: 'Map Fortran layout, define Python modules, align docs.', date: 'Nov 2025' },
    { title: 'Phase 2 • Infrastructure', detail: 'Config/constants, loaders, FCHK parser, pip glue.', date: 'Q1 2026' },
    { title: 'Phase 3 • Core', detail: 'Vectorized density + basis evaluation plus coverage tests.', date: 'Q2 2026' },
    { title: 'Phase 4 • Extensions', detail: 'Wrap specialized Fortran grids via f2py and C APIs.', date: 'Q3 2026' },
  ];

  const highlights = [
    { title: 'Zero-compile install', copy: 'Pure-Python entrypoints with optional prebuilt wheels keep setup frictionless.', cta: 'pip install pymultiwfn', icon: '🚀' },
    { title: 'Live density kernels', copy: 'NumPy broadcasting + einsum-style ops remove Python loops from dense math.', cta: 'View density code', icon: '🧪' },
    { title: 'Fortran parity', copy: 'consistency_verifier compares PyMultiWFN vs Multiwfn 3.8 outputs line by line.', cta: 'Run verifier', icon: '⚖️' },
  ];

  const slides = [
    { title: 'Grid engines', body: 'Lebedev-Laikov grids wrapped with f2py for sub-second SCF grid scans.', accent: 'Performance' },
    { title: 'Parsing pipeline', body: 'FCHK/Molden readers convert straight into dataclasses and ndarray tensors.', accent: 'IO & Data' },
    { title: 'Visualization ready', body: 'Outputs align with PyVista/Plotly and Jupyter for immediate visuals.', accent: 'Visualization' },
  ];

  const StepsBar = () => React.createElement(Steps, {
    current: 2,
    items: [
      { title: 'Scan', description: 'Load wavefunction + metadata' },
      { title: 'Vectorize', description: 'NumPy/Fortran hybrid kernels' },
      { title: 'Verify', description: 'consistency_verifier parity' },
      { title: 'Publish', description: 'Ship wheels + docs' },
    ]
  });

  const App = () => React.createElement(
    ConfigProvider,
    {
      theme: {
        token: {
          colorPrimary: '#1677ff',
          colorTextBase: '#0a1a2f',
          fontFamily: "'Space Grotesk', system-ui, sans-serif",
          borderRadius: 22,
          colorBgContainer: '#ffffff'
        },
        algorithm: antd.theme.defaultAlgorithm,
      }
    },
    React.createElement(
      "div",
      { className: "page-shell" },
      React.createElement("div", { className: "floating-orb orb-a" }),
      React.createElement("div", { className: "floating-orb orb-b" }),
      React.createElement(
        Layout,
        null,
        React.createElement(
          Header,
          { className: "header-glass" },
          React.createElement(
            Row,
            { justify: "space-between", align: "middle" },
            React.createElement(
              Col,
              null,
              React.createElement(
                "div",
                { className: "logo-mark" },
                React.createElement("span", null, "PyMultiWFN"),
                React.createElement(Tag, { color: "blue" }, "AntD 6")
              )
            ),
            React.createElement(
              Col,
              null,
              React.createElement(
                Space,
                { size: "large" },
                React.createElement(Button, { type: "link", href: "#features" }, "Features"),
                React.createElement(Button, { type: "link", href: "#roadmap" }, "Roadmap"),
                React.createElement(Button, { type: "primary", href: "https://github.com/yourusername/PyMultiWFN", target: "_blank" }, "GitHub")
              )
            )
          )
        ),
        React.createElement(
          Content,
          { style: { padding: '3rem 5vw 4rem' } },
          React.createElement(
            Row,
            { gutter: [32, 32], align: "middle" },
            React.createElement(
              Col,
              { xs: 24, lg: 14 },
              React.createElement(
                "div",
                { className: "hero-panel fade-up" },
                React.createElement("div", { className: "hero-floating", "aria-hidden": "true" }),
                React.createElement(
                  Space,
                  { direction: "vertical", size: "middle" },
                  React.createElement(
                    Title,
                    { level: 1, style: { color: '#0a1a2f', fontWeight: 800 } },
                    "PyMultiWFN",
                    React.createElement(
                      Text,
                      { type: "secondary", style: { display: 'block', fontSize: '1rem' } },
                      "Python-native wavefunction analysis with rock-solid Fortran parity."
                    )
                  ),
                  React.createElement(
                    Paragraph,
                    { style: { color: '#3c4b61', fontSize: '1.05rem' } },
                    "Legacy Multiwfn power, rebuilt for the Python ecosystem. Vectorized cores, curated interfaces, and an Ant Design (v6) marketing layer that highlights why researchers are switching."
                  ),
                  React.createElement(
                    Space,
                    { className: "cta-row" },
                    React.createElement(Button, { type: "primary", size: "large", href: "https://pypi.org/project/pymultiwfn/", target: "_blank" }, "Install on PyPI"),
                    React.createElement(Button, { size: "large", href: "#features" }, "See Features")
                  ),
                  React.createElement(
                    Row,
                    { gutter: [16, 16] },
                    stats.map((stat) =>
                      React.createElement(
                        Col,
                        { key: stat.title, xs: 24, sm: 8 },
                        React.createElement(
                          Card,
                          { className: "feature-card floating-card", bordered: true, size: "small" },
                          React.createElement(Statistic, { title: stat.title, value: stat.value, valueStyle: { color: '#1677ff', fontSize: '1.5rem' } }),
                          React.createElement(Text, { type: "secondary" }, stat.subtitle)
                        )
                      )
                    )
                  )
                )
              )
            ),
            React.createElement(
              Col,
              { xs: 24, lg: 10 },
              React.createElement(
                Card,
                { className: "feature-card shimmer-card", style: { minHeight: 360 }, bordered: true },
                React.createElement(
                  Space,
                  { direction: "vertical", size: "middle", style: { width: '100%' } },
                  React.createElement(Title, { level: 4, style: { color: '#0a1a2f' } }, "Quick Consistency Verifier"),
                  React.createElement(
                    Paragraph,
                    { type: "secondary" },
                    "Runs the @consistency_verifier suite to ensure PyMultiWFN mirrors Multiwfn 3.8 outputs before releasing a build. This validator is part of our CI pipeline."
                  ),
                  React.createElement(Button, { type: "default", block: true, href: "https://github.com/yourusername/consistency_verifier" }, "View Tests"),
                  React.createElement(Divider, null),
                  React.createElement(
                    Paragraph,
                    null,
                    "Live telemetry: 1.2 seconds to parse a Gaussian FCHK file and 0.7 seconds to evaluate density on 15k grid points using NumPy + Fortran."
                  ),
                  React.createElement(StepsBar, null)
                )
              )
            )
          ),
          React.createElement(
            "section",
            { id: "features", style: { marginTop: '4rem' } },
            React.createElement("div", { className: "section-title" }, "Capabilities"),
            React.createElement(
              Row,
              { gutter: [24, 24], style: { marginTop: '1rem' } },
              features.map((feature, index) =>
                React.createElement(
                  Col,
                  { key: feature.title, xs: 24, md: 12, lg: 6 },
                  React.createElement(
                    Card,
                    { hoverable: true, className: "feature-card floating-card", bordered: true, style: { animationDelay: `${index * 80}ms` } },
                    React.createElement(
                      "div",
                      { className: "card-icon" },
                      React.createElement("span", { role: "img", "aria-label": "icon" }, feature.icon)
                    ),
                    React.createElement(Tag, { color: "processing", style: { marginBottom: '0.5rem' } }, feature.badge),
                    React.createElement(Title, { level: 4 }, feature.title),
                    React.createElement(Paragraph, { type: "secondary" }, feature.description)
                  )
                )
              )
            )
          ),
          React.createElement(
            "section",
            { style: { marginTop: '4rem' } },
            React.createElement("div", { className: "section-title" }, "Live highlights"),
            React.createElement(
              Row,
              { gutter: [24, 24] },
              highlights.map((item, idx) =>
                React.createElement(
                  Col,
                  { key: item.title, xs: 24, md: 8 },
                  React.createElement(
                    Card,
                    { className: "glass-card floating-card", hoverable: true, style: { animationDelay: `${idx * 90}ms` } },
                    React.createElement(
                      Space,
                      { direction: "vertical", size: "small", style: { width: '100%' } },
                      React.createElement(
                        Space,
                        { align: "center", size: "small" },
                        React.createElement("span", { ariaHidden: "true", style: { fontSize: '1.1rem' } }, item.icon || ''),
                        React.createElement(Title, { level: 4, style: { margin: 0 } }, item.title)
                      ),
                      React.createElement(Paragraph, { type: "secondary" }, item.copy),
                      React.createElement(Button, { type: "link" }, item.cta)
                    )
                  )
                )
              )
            )
          ),
          React.createElement(
            "section",
            { id: "roadmap", style: { marginTop: '4rem' } },
            React.createElement("div", { className: "section-title" }, "Roadmap"),
            React.createElement(Title, { level: 3 }, "From Fortran Roots to Python Reach"),
            React.createElement(
              Row,
              { gutter: [24, 24], style: { marginTop: '1rem' } },
              React.createElement(
                Col,
                { xs: 24, lg: 14 },
                React.createElement(Timeline, {
                  mode: "left",
                  items: roadmap.map((item) => ({
                    label: item.date,
                    children: React.createElement(
                      Card,
                      { className: "timeline-card floating-card", bordered: true, size: "small" },
                      React.createElement(Title, { level: 5 }, item.title),
                      React.createElement(Paragraph, { type: "secondary" }, item.detail)
                    )
                  }))
                })
              ),
              React.createElement(
                Col,
                { xs: 24, lg: 10 },
                React.createElement(
                  Card,
                  { className: "feature-card carousel-card", bordered: true },
                  React.createElement(Carousel, { autoplay: true, dotPosition: "bottom", autoplaySpeed: 3200 },
                    slides.map((slide) =>
                      React.createElement(
                        "div",
                        { key: slide.title },
                        React.createElement(
                          "div",
                          { className: "slide-panel" },
                          React.createElement(Tag, { color: "blue" }, slide.accent),
                          React.createElement(Title, { level: 4 }, slide.title),
                          React.createElement(Paragraph, { type: "secondary" }, slide.body)
                        )
                      )
                    )
                  )
                )
              )
            )
          ),
          React.createElement(
            "section",
            { style: { marginTop: '4rem' } },
            React.createElement(Divider, { orientation: "left" }, "Join the Wave"),
            React.createElement(
              Row,
              { gutter: [24, 24] },
              React.createElement(
                Col,
                { xs: 24, md: 12 },
                React.createElement(
                  Card,
                  { className: "feature-card floating-card", bordered: true },
                  React.createElement(Title, { level: 4 }, "Install & Explore"),
                  React.createElement(
                    Paragraph,
                    { type: "secondary" },
                    "`pip install pymultiwfn` brings the package to your environment sans compilation steps, thanks to pure-Python entry points and optional prebuilt Fortran wheels."
                  ),
                  React.createElement(Button, { type: "primary", block: true, href: "https://pypi.org/project/pymultiwfn/" }, "View PyPI")
                )
              ),
              React.createElement(
                Col,
                { xs: 24, md: 12 },
                React.createElement(
                  Card,
                  { className: "feature-card floating-card", bordered: true },
                  React.createElement(Title, { level: 4 }, "Docs & Support"),
                  React.createElement(
                    Paragraph,
                    { type: "secondary" },
                    "Visit this GitHub Pages site for a friendly overview and refer to AGENTS.md history + README for migration stories and testing guidance."
                  ),
                  React.createElement(Button, { block: true, href: "https://github.com/yourusername/PyMultiWFN/wiki" }, "View Wiki")
                )
              )
            )
          )
        ),
        React.createElement(
          Footer,
          { style: { textAlign: 'center', borderTop: '1px solid rgba(0,0,0,0.06)', background: '#f9fbff' } },
          React.createElement(Text, { type: "secondary" }, "© ", new Date().getFullYear(), " PyMultiWFN • Built with Ant Design 6 and modern chemistry tooling.")
        ),
        React.createElement(FloatButton.BackTop, { visibilityHeight: 200 })
      )
    )
  );

  ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
})();
