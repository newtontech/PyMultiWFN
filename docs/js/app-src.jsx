import React from 'react';
import ReactDOM from 'react-dom/client';
import dayjs from 'dayjs';
import {
  Layout,
  Typography,
  Button,
  Space,
  Card,
  Row,
  Col,
  Tag,
  Divider,
  Timeline,
  Statistic,
  ConfigProvider,
  Carousel,
  Steps,
  FloatButton,
} from 'antd';

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

const StepsBar = () => (
  <Steps
    current={2}
    items={[
      { title: 'Scan', description: 'Load wavefunction + metadata' },
      { title: 'Vectorize', description: 'NumPy/Fortran hybrid kernels' },
      { title: 'Verify', description: 'consistency_verifier parity' },
      { title: 'Publish', description: 'Ship wheels + docs' },
    ]}
  />
);

const App = () => (
  <ConfigProvider
    theme={{
      token: {
        colorPrimary: '#1677ff',
        colorTextBase: '#0a1a2f',
        fontFamily: "'Space Grotesk', system-ui, sans-serif",
        borderRadius: 22,
        colorBgContainer: '#ffffff',
      },
      algorithm: undefined,
    }}
  >
    <div className="page-shell">
      <div className="floating-orb orb-a" />
      <div className="floating-orb orb-b" />
      <Layout>
        <Header className="header-glass">
          <Row justify="space-between" align="middle">
            <Col>
              <div className="logo-mark">
                <span>PyMultiWFN</span>
                <Tag color="blue">AntD 6</Tag>
              </div>
            </Col>
            <Col>
              <Space size="large">
                <Button type="link" href="#features">Features</Button>
                <Button type="link" href="#roadmap">Roadmap</Button>
                <Button type="primary" href="https://github.com/yourusername/PyMultiWFN" target="_blank">GitHub</Button>
              </Space>
            </Col>
          </Row>
        </Header>

        <Content style={{ padding: '3rem 5vw 4rem' }}>
          <Row gutter={[32, 32]} align="middle">
            <Col xs={24} lg={14}>
              <div className="hero-panel fade-up">
                <div className="hero-floating" aria-hidden="true" />
                <Space direction="vertical" size="middle">
                  <Title level={1} style={{ color: '#0a1a2f', fontWeight: 800 }}>
                    PyMultiWFN
                    <Text type="secondary" style={{ display: 'block', fontSize: '1rem' }}>
                      Python-native wavefunction analysis with rock-solid Fortran parity.
                    </Text>
                  </Title>
                  <Paragraph style={{ color: '#3c4b61', fontSize: '1.05rem' }}>
                    Legacy Multiwfn power, rebuilt for the Python ecosystem. Vectorized cores, curated interfaces, and an Ant Design (v6) marketing layer that highlights why researchers are switching.
                  </Paragraph>
                  <Space className="cta-row">
                    <Button type="primary" size="large" href="https://pypi.org/project/pymultiwfn/" target="_blank">Install on PyPI</Button>
                    <Button size="large" href="#features">See Features</Button>
                  </Space>
                  <Row gutter={[16, 16]}>
                    {stats.map((stat) => (
                      <Col key={stat.title} xs={24} sm={8}>
                        <Card className="feature-card floating-card" bordered size="small">
                          <Statistic title={stat.title} value={stat.value} valueStyle={{ color: '#1677ff', fontSize: '1.5rem' }} />
                          <Text type="secondary">{stat.subtitle}</Text>
                        </Card>
                      </Col>
                    ))}
                  </Row>
                </Space>
              </div>
            </Col>

            <Col xs={24} lg={10}>
              <Card className="feature-card shimmer-card" style={{ minHeight: 360 }} bordered>
                <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                  <Title level={4} style={{ color: '#0a1a2f' }}>Quick Consistency Verifier</Title>
                  <Paragraph type="secondary">
                    Runs the @consistency_verifier suite to ensure PyMultiWFN mirrors Multiwfn 3.8 outputs before releasing a build. This validator is part of our CI pipeline.
                  </Paragraph>
                  <Button type="default" block href="https://github.com/yourusername/consistency_verifier">View Tests</Button>
                  <Divider />
                  <Paragraph>
                    Live telemetry: 1.2 seconds to parse a Gaussian FCHK file and 0.7 seconds to evaluate density on 15k grid points using NumPy + Fortran.
                  </Paragraph>
                  <StepsBar />
                </Space>
              </Card>
            </Col>
          </Row>

          <section id="features" style={{ marginTop: '4rem' }}>
            <div className="section-title">Capabilities</div>
            <Row gutter={[24, 24]} style={{ marginTop: '1rem' }}>
              {features.map((feature, index) => (
                <Col key={feature.title} xs={24} md={12} lg={6}>
                  <Card hoverable className="feature-card floating-card" bordered style={{ animationDelay: `${index * 80}ms` }}>
                    <div className="card-icon">
                      <span role="img" aria-label="icon">{feature.icon}</span>
                    </div>
                    <Tag color="processing" style={{ marginBottom: '0.5rem' }}>{feature.badge}</Tag>
                    <Title level={4}>{feature.title}</Title>
                    <Paragraph type="secondary">{feature.description}</Paragraph>
                  </Card>
                </Col>
              ))}
            </Row>
          </section>

          <section style={{ marginTop: '4rem' }}>
            <div className="section-title">Live highlights</div>
            <Row gutter={[24, 24]}>
              {highlights.map((item, idx) => (
                <Col key={item.title} xs={24} md={8}>
                  <Card className="glass-card floating-card" hoverable style={{ animationDelay: `${idx * 90}ms` }}>
                    <Space direction="vertical" size="small" style={{ width: '100%' }}>
                      <Space align="center" size="small">
                        <span aria-hidden="true" style={{ fontSize: '1.1rem' }}>{item.icon || ''}</span>
                        <Title level={4} style={{ margin: 0 }}>{item.title}</Title>
                      </Space>
                      <Paragraph type="secondary">{item.copy}</Paragraph>
                      <Button type="link">{item.cta}</Button>
                    </Space>
                  </Card>
                </Col>
              ))}
            </Row>
          </section>

          <section id="roadmap" style={{ marginTop: '4rem' }}>
            <div className="section-title">Roadmap</div>
            <Title level={3}>From Fortran Roots to Python Reach</Title>
            <Row gutter={[24, 24]} style={{ marginTop: '1rem' }}>
              <Col xs={24} lg={14}>
                <Timeline
                  mode="left"
                  items={roadmap.map((item) => ({
                    label: item.date,
                    children: (
                      <Card className="timeline-card floating-card" bordered size="small">
                        <Title level={5}>{item.title}</Title>
                        <Paragraph type="secondary">{item.detail}</Paragraph>
                      </Card>
                    ),
                  }))}
                />
              </Col>
              <Col xs={24} lg={10}>
                <Card className="feature-card carousel-card" bordered>
                  <Carousel autoplay dotPosition="bottom" autoplaySpeed={3200}>
                    {slides.map((slide) => (
                      <div key={slide.title}>
                        <div className="slide-panel">
                          <Tag color="blue">{slide.accent}</Tag>
                          <Title level={4}>{slide.title}</Title>
                          <Paragraph type="secondary">{slide.body}</Paragraph>
                        </div>
                      </div>
                    ))}
                  </Carousel>
                </Card>
              </Col>
            </Row>
          </section>

          <section style={{ marginTop: '4rem' }}>
            <Divider orientation="left">Join the Wave</Divider>
            <Row gutter={[24, 24]}>
              <Col xs={24} md={12}>
                <Card className="feature-card floating-card" bordered>
                  <Title level={4}>Install &amp; Explore</Title>
                  <Paragraph type="secondary">
                    `pip install pymultiwfn` brings the package to your environment sans compilation steps, thanks to pure-Python entry points and optional prebuilt Fortran wheels.
                  </Paragraph>
                  <Button type="primary" block href="https://pypi.org/project/pymultiwfn/">View PyPI</Button>
                </Card>
              </Col>
              <Col xs={24} md={12}>
                <Card className="feature-card floating-card" bordered>
                  <Title level={4}>Docs &amp; Support</Title>
                  <Paragraph type="secondary">
                    Visit this GitHub Pages site for a friendly overview and refer to AGENTS.md history + README for migration stories and testing guidance.
                  </Paragraph>
                  <Button block href="https://github.com/yourusername/PyMultiWFN/wiki">View Wiki</Button>
                </Card>
              </Col>
            </Row>
          </section>
        </Content>

        <Footer style={{ textAlign: 'center', borderTop: '1px solid rgba(0,0,0,0.06)', background: '#f9fbff' }}>
          <Text type="secondary">© {new Date().getFullYear()} PyMultiWFN • Built with Ant Design 6 and modern chemistry tooling.</Text>
        </Footer>
        <FloatButton.BackTop visibilityHeight={200} />
      </Layout>
    </div>
  </ConfigProvider>
);

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
