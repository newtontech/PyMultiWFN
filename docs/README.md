# PyMultiWFN 文档目录

## 目录结构

```
docs/
├── README.md              # 本文档
├── summmaries/            # 开发总结报告
│   ├── hourly/           # 小时报告（历史归档）
│   ├── daily/            # 日报（历史归档）
│   └── final/            # 最终报告
├── tasks/                # 任务文档
├── meetings/             # 会议记录
├── AGENTS.md             # Agent 配置
├── CLAUDE.md             # Claude 配置
└── FEISHU_QUICKSTART.md  # 飞书快速开始
```

## 核心项目文档

项目核心文档位于仓库根目录：
- `README.md` - 项目介绍
- `ACKNOWLEDGMENTS.md` - 致谢
- `.github/` - GitHub 配置

## 开发流程

1. 日常开发 → 生成小时报告 → 归档到 `summaries/hourly/`
2. 每日结束 → 生成日报 → 归档到 `summaries/daily/`
3. 项目里程碑 → 生成最终报告 → 归档到 `summaries/final/`
