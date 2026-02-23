# NeuroAgent Modern UI

现代化的 NeuroAgent 用户界面，使用 Tailwind CSS 和 D3.js 构建。

## 特性

- **现代化设计**: 使用 Tailwind CSS 构建的清爽界面
- **响应式布局**: 适配各种屏幕尺寸
- **交互式知识图谱**: 使用 D3.js 的可视化力导向图
- **流畅动画**: 平滑的过渡和交互动效
- **直观导航**: 侧边栏导航，清晰的视觉层次

## 页面

| 路由 | 页面 | 描述 |
|------|------|------|
| `/query` | 智能对话 | 现代化的聊天界面，支持文件上传 |
| `/kg-viz` | 知识图谱可视化 | 交互式 D3.js 图可视化 |
| `/upload` | 文档上传 | 支持拖拽上传的现代化界面 |
| `/config` | 配置设置 | 系统配置管理 |

## 启动

```bash
python launch_ui.py
```

然后访问: http://localhost:5000

## 技术栈

- **后端**: Flask
- **CSS**: Tailwind CSS (CDN)
- **图标**: Font Awesome 6
- **字体**: Inter (Google Fonts)
- **图表**: D3.js v7

## 项目结构

```
核心库 ui 目录（见仓库结构）
├── app.py                      # Flask 主应用
├── routers/
│   └── kg_visualization.py    # KG 可视化路由
└── templates/
    ├── base_modern.html        # 基础模板
    ├── query_modern.html       # 对话页面
    ├── kg_viz_modern.html      # KG 可视化页面
    ├── upload_modern.html      # 上传页面
    └── config_modern.html      # 配置页面
```

## 截图

- 对话界面: 现代化的聊天界面，支持多种记忆模式
- 知识图谱: 可交互的力导向图，支持缩放、拖拽、筛选
- 文件上传: 拖拽上传，进度显示
