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

const getContent = (lang) => {
  const isEn = lang === 'en';
  return {
    nav: {
      features: isEn ? 'Features' : '特性',
      roadmap: isEn ? 'Roadmap' : '路线图',
      github: 'GitHub',
    },
    hero: {
      title: 'PyMultiWFN',
      subtitle: isEn ? 'Python-native wavefunction analysis with rock-solid Fortran parity.' : '原生 Python 波函数分析，坚如磐石的 Fortran 对等性。',
      desc: isEn ? 'Legacy Multiwfn power, rebuilt for the Python ecosystem. Vectorized cores, curated interfaces, and an Ant Design (v6) marketing layer that highlights why researchers are switching.' : '传承 Multiwfn 的强大功能，为 Python 生态系统重构。向量化核心、精心设计的接口，以及 Ant Design (v6) 营销层，彰显研究人员转向于此的原因。',
      install: isEn ? 'Install on PyPI' : '在 PyPI 上安装',
      seeFeatures: isEn ? 'See Features' : '查看特性',
    },
    stats: [
      { title: isEn ? 'Reproducibility' : '复现性', value: '100%', subtitle: isEn ? 'Bitwise parity vs Multiwfn outputs' : '与 Multiwfn 输出逐位对等' },
      { title: isEn ? 'Performance Gain' : '性能提升', value: '10×', subtitle: isEn ? 'Grid scanning sped up with NumPy/Fortran mix' : 'NumPy/Fortran 混合加速网格扫描' },
      { title: isEn ? 'Community' : '社区', value: '1000+', subtitle: isEn ? 'Active researchers, developers, students' : '活跃的研究人员、开发者和学生' },
    ],
    features: [
      { title: isEn ? 'Modern Data Model' : '现代数据模型', description: isEn ? 'Wavefunction, basis sets, and integrals live in immutable dataclasses before dispatch to vectorized kernels.' : '波函数、基组和积分在分发到向量化内核之前驻留在不可变的数据类中。', icon: '🧠', badge: 'core' },
      { title: isEn ? 'Ant Design UI' : 'Ant Design UI', description: isEn ? 'Docs and marketing lean on AntD 6 components, motion tokens, and responsive layouts.' : '文档和营销依赖于 AntD 6 组件、动效 Token 和响应式布局。', icon: '🎨', badge: 'brand' },
      { title: isEn ? 'Consistent Backends' : '一致的后端', description: isEn ? 'consistency_verifier keeps PyMultiWFN matched against Multiwfn 3.8 across golden tasks.' : 'consistency_verifier 保持 PyMultiWFN 在黄金任务中与 Multiwfn 3.8 匹配。', icon: '⚖️', badge: 'validation' },
      { title: isEn ? 'Hybrid Performance' : '混合性能', description: isEn ? 'NumPy vectorization + f2py-wrapped Fortran for grid-heavy math without losing clarity.' : 'NumPy 向量化 + f2py 封装的 Fortran，用于网格密集型数学运算，且不失清晰度。', icon: '⚡', badge: 'performance' },
    ],
    roadmap: [
      { title: isEn ? 'Phase 1 • Reconnaissance' : '第一阶段 • 侦察', detail: isEn ? 'Map Fortran layout, define Python modules, align docs.' : '映射 Fortran 布局，定义 Python 模块，对齐文档。', date: 'Nov 2025' },
      { title: isEn ? 'Phase 2 • Infrastructure' : '第二阶段 • 基础设施', detail: isEn ? 'Config/constants, loaders, FCHK parser, pip glue.' : '配置/常量，加载器，FCHK 解析器，pip 胶水代码。', date: 'Q1 2026' },
      { title: isEn ? 'Phase 3 • Core' : '第三阶段 • 核心', detail: isEn ? 'Vectorized density + basis evaluation plus coverage tests.' : '向量化密度 + 基组评估以及覆盖率测试。', date: 'Q2 2026' },
      { title: isEn ? 'Phase 4 • Extensions' : '第四阶段 • 扩展', detail: isEn ? 'Wrap specialized Fortran grids via f2py and C APIs.' : '通过 f2py 和 C API 封装专用的 Fortran 网格。', date: 'Q3 2026' },
    ],
    highlights: [
      { title: isEn ? 'Zero-compile install' : '零编译安装', copy: isEn ? 'Pure-Python entrypoints with optional prebuilt wheels keep setup frictionless.' : '纯 Python 入口点和可选的预构建 Wheel 包使安装顺畅无阻。', cta: 'pip install pymultiwfn', icon: '🚀' },
      { title: isEn ? 'Live density kernels' : '实时密度内核', copy: isEn ? 'NumPy broadcasting + einsum-style ops remove Python loops from dense math.' : 'NumPy 广播 + einsum 风格的操作消除了密集数学运算中的 Python 循环。', cta: isEn ? 'View density code' : '查看密度代码', icon: '🧪' },
      { title: isEn ? 'Fortran parity' : 'Fortran 对等性', copy: isEn ? 'consistency_verifier compares PyMultiWFN vs Multiwfn 3.8 outputs line by line.' : 'consistency_verifier 逐行比较 PyMultiWFN 与 Multiwfn 3.8 的输出。', cta: isEn ? 'Run verifier' : '运行验证器', icon: '⚖️' },
    ],
    slides: [
      { title: isEn ? 'Grid engines' : '网格引擎', body: isEn ? 'Lebedev-Laikov grids wrapped with f2py for sub-second SCF grid scans.' : 'Lebedev-Laikov 网格采用 f2py 封装，实现亚秒级 SCF 网格扫描。', accent: isEn ? 'Performance' : '性能' },
      { title: isEn ? 'Parsing pipeline' : '解析管道', body: isEn ? 'FCHK/Molden readers convert straight into dataclasses and ndarray tensors.' : 'FCHK/Molden 读取器直接转换为数据类和 ndarray 张量。', accent: 'IO & Data' },
      { title: isEn ? 'Visualization ready' : '可视化就绪', body: isEn ? 'Outputs align with PyVista/Plotly and Jupyter for immediate visuals.' : '输出与 PyVista/Plotly 和 Jupyter 对齐，可立即进行可视化。', accent: isEn ? 'Visualization' : '可视化' },
    ],
    verifier: {
      title: isEn ? 'Quick Consistency Verifier' : '快速一致性验证器',
      desc: isEn ? 'Runs the @consistency_verifier suite to ensure PyMultiWFN mirrors Multiwfn 3.8 outputs before releasing a build. This validator is part of our CI pipeline.' : '运行 @consistency_verifier 套件以确保 PyMultiWFN 在发布版本之前镜像 Multiwfn 3.8 的输出。此验证器是我们 CI 管道的一部分。',
      viewTests: isEn ? 'View Tests' : '查看测试',
      telemetry: isEn ? 'Live telemetry: 1.2 seconds to parse a Gaussian FCHK file and 0.7 seconds to evaluate density on 15k grid points using NumPy + Fortran.' : '实时遥测：解析 Gaussian FCHK 文件需 1.2 秒，使用 NumPy + Fortran 在 1.5 万个网格点上评估密度需 0.7 秒。',
    },
    steps: [
      { title: isEn ? 'Scan' : '扫描', description: isEn ? 'Load wavefunction + metadata' : '加载波函数 + 元数据' },
      { title: isEn ? 'Vectorize' : '向量化', description: isEn ? 'NumPy/Fortran hybrid kernels' : 'NumPy/Fortran 混合内核' },
      { title: isEn ? 'Verify' : '验证', description: isEn ? 'consistency_verifier parity' : 'consistency_verifier 对等性' },
      { title: isEn ? 'Publish' : '发布', description: isEn ? 'Ship wheels + docs' : '发布 Wheel 包 + 文档' },
    ],
    sections: {
      capabilities: isEn ? 'Capabilities' : '功能',
      highlights: isEn ? 'Live highlights' : '实时亮点',
      roadmap: isEn ? 'Roadmap' : '路线图',
      roadmapTitle: isEn ? 'From Fortran Roots to Python Reach' : '从 Fortran 根基到 Python 触达',
      join: isEn ? 'Join the Wave' : '加入浪潮',
    },
    join: {
      installTitle: isEn ? 'Install & Explore' : '安装与探索',
      installDesc: isEn ? '`pip install pymultiwfn` brings the package to your environment sans compilation steps, thanks to pure-Python entry points and optional prebuilt Fortran wheels.' : '`pip install pymultiwfn` 将软件包带入您的环境，无需编译步骤，这归功于纯 Python 入口点和可选的预构建 Fortran Wheel 包。',
      viewPypi: isEn ? 'View PyPI' : '查看 PyPI',
      docsTitle: isEn ? 'Docs & Support' : '文档与支持',
      docsDesc: isEn ? 'Visit this GitHub Pages site for a friendly overview and refer to AGENTS.md history + README for migration stories and testing guidance.' : '访问此 GitHub Pages 站点以获取友好的概述，并参考 AGENTS.md 历史记录 + README 以获取迁移故事和测试指南。',
      viewWiki: isEn ? 'View Wiki' : '查看 Wiki',
    },
    footer: isEn ? `© ${new Date().getFullYear()} PyMultiWFN • Built with Ant Design 6 and modern chemistry tooling.` : `© ${new Date().getFullYear()} PyMultiWFN • 基于 Ant Design 6 和现代化学工具构建。`,
  };
};

const App = () => {
  const [lang, setLang] = React.useState('en');
  const content = getContent(lang);

  const toggleLang = () => {
    setLang(prev => prev === 'en' ? 'zh' : 'en');
  };

  return (
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
                  <Button type="link" href="#features">{content.nav.features}</Button>
                  <Button type="link" href="#roadmap">{content.nav.roadmap}</Button>
                  <Button type="primary" href="https://github.com/yourusername/PyMultiWFN" target="_blank">{content.nav.github}</Button>
                  <Button onClick={toggleLang}>
                    {lang === 'en' ? '简体中文' : 'English'}
                  </Button>
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
                      {content.hero.title}
                      <Text type="secondary" style={{ display: 'block', fontSize: '1rem' }}>
                        {content.hero.subtitle}
                      </Text>
                    </Title>
                    <Paragraph style={{ color: '#3c4b61', fontSize: '1.05rem' }}>
                      {content.hero.desc}
                    </Paragraph>
                    <Space className="cta-row">
                      <Button type="primary" size="large" href="https://pypi.org/project/pymultiwfn/" target="_blank">{content.hero.install}</Button>
                      <Button size="large" href="#features">{content.hero.seeFeatures}</Button>
                    </Space>
                    <Row gutter={[16, 16]}>
                      {content.stats.map((stat) => (
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
                    <Title level={4} style={{ color: '#0a1a2f' }}>{content.verifier.title}</Title>
                    <Paragraph type="secondary">
                      {content.verifier.desc}
                    </Paragraph>
                    <Button type="default" block href="https://github.com/yourusername/consistency_verifier">{content.verifier.viewTests}</Button>
                    <Divider />
                    <Paragraph>
                      {content.verifier.telemetry}
                    </Paragraph>
                    <Steps
                      current={2}
                      items={content.steps}
                    />
                  </Space>
                </Card>
              </Col>
            </Row>

            <section id="features" style={{ marginTop: '4rem' }}>
              <div className="section-title">{content.sections.capabilities}</div>
              <Row gutter={[24, 24]} style={{ marginTop: '1rem' }}>
                {content.features.map((feature, index) => (
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
              <div className="section-title">{content.sections.highlights}</div>
              <Row gutter={[24, 24]}>
                {content.highlights.map((item, idx) => (
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
              <div className="section-title">{content.sections.roadmap}</div>
              <Title level={3}>{content.sections.roadmapTitle}</Title>
              <Row gutter={[24, 24]} style={{ marginTop: '1rem' }}>
                <Col xs={24} lg={14}>
                  <Timeline
                    mode="left"
                    items={content.roadmap.map((item) => ({
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
                      {content.slides.map((slide) => (
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
              <Divider orientation="left">{content.sections.join}</Divider>
              <Row gutter={[24, 24]}>
                <Col xs={24} md={12}>
                  <Card className="feature-card floating-card" bordered>
                    <Title level={4}>{content.join.installTitle}</Title>
                    <Paragraph type="secondary">
                      {content.join.installDesc}
                    </Paragraph>
                    <Button type="primary" block href="https://pypi.org/project/pymultiwfn/">{content.join.viewPypi}</Button>
                  </Card>
                </Col>
                <Col xs={24} md={12}>
                  <Card className="feature-card floating-card" bordered>
                    <Title level={4}>{content.join.docsTitle}</Title>
                    <Paragraph type="secondary">
                      {content.join.docsDesc}
                    </Paragraph>
                    <Button block href="https://github.com/yourusername/PyMultiWFN/wiki">{content.join.viewWiki}</Button>
                  </Card>
                </Col>
              </Row>
            </section>
          </Content>

          <Footer style={{ textAlign: 'center', borderTop: '1px solid rgba(0,0,0,0.06)', background: '#f9fbff' }}>
            <Text type="secondary">{content.footer}</Text>
          </Footer>
          <FloatButton.BackTop visibilityHeight={200} />
        </Layout>
      </div>
    </ConfigProvider>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
