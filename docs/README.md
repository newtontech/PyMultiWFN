# PyMultiWFN 文档目录

## 目录结构

```
docs/
├── README.md              # 本文档
├── index.html             # GitHub Pages 入口
├── css/, js/, components/ # 文档站静态资源
├── user_guide.md          # 用户指南
├── TESTING.md             # 标准测试指南
├── TESTING_GUIDE.md       # 中文测试扩展说明
├── summaries/             # 开发、验证、状态和阶段报告归档
│   ├── development/       # 开发过程记录
│   ├── final/             # 最终报告
│   ├── reports/           # 阶段报告和规划记录
│   ├── status/            # session/status 快照
│   └── verification/      # 测试与验证报告
├── tasks/                 # 历史任务、coder/verifier prompt
└── meetings/              # 会议记录
```

## 核心项目文档

项目核心文档位于仓库根目录：
- `README.md` - 项目介绍
- `architecture_plan.md` - 架构设计、仓库边界和进度列表
- `pyproject.toml` / `setup.cfg` - 包元数据、测试、覆盖率和静态检查配置
- `.github/` - GitHub 配置

## 仓库边界

- `pymultiwfn/` 是唯一参与 Python 包安装的源码包。
- `Multiwfn_3.8_dev_src_Linux_2025-Nov-23/` 是上游参考源码，仅用于迁移、对照和一致性分析。
- `Multiwfn_3.8_bin_Linux_noGUI/` 与 `Multiwfn_3.8_bin_Linux_noGUI.zip` 保留为参考运行和一致性验证资产。

## 开发流程

1. 日常开发记录 → 归档到 `summaries/development/` 或 `summaries/status/`
2. 阶段计划与报告 → 归档到 `summaries/reports/`
3. 验证与测试结果 → 归档到 `summaries/verification/`
4. 项目里程碑 → 归档到 `summaries/final/`
5. 任务提示和执行说明 → 归档到 `tasks/`
