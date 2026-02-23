# NeuroAgent UI 重构变更日志

## 2024-02-15 - 完整 UI 重构

### 新增文件

#### 模板文件（核心库 `ui/templates/`）
- `base_modern.html` - 现代化的基础模板
  - Tailwind CSS 设计系统
  - 侧边栏导航
  - Glass morphism 效果
  - Toast 通知系统
  
- `query_modern.html` - 现代化聊天界面
  - 三种记忆模式选择 (标准/深度/快速)
  - 文件上传预览
  - 会话历史侧边栏
  - 知识库统计
  
- `kg_viz_modern.html` - 交互式知识图谱
  - D3.js 力导向图
  - 层级筛选 (领域/概念/实例)
  - 节点详情面板
  - 搜索和缩放功能
  
- `upload_modern.html` - 拖拽上传界面
  - 文件拖放支持
  - 上传进度显示
  - 格式图标预览
  
- `config_modern.html` - 配置管理界面
  - API 设置
  - 记忆参数调整
  - 存储管理

#### Python 文件
- `launch_ui.py` - 一键启动脚本
  - 自动打开浏览器
  - 显示可用路由
  - 友好的启动信息

### 修改的文件

#### 核心库 `ui/app.py`
- 移除了按钮注入代码 (修复重复按钮问题)
- 更新了路由使用新模板:
  - `/query` → `query_modern.html`
  - `/knowledge-graph` → `kg_viz_modern.html`  
  - `/upload` → `upload_modern.html`
  - `/config` → `config_modern.html`

#### 核心库 `ui/routers/kg_visualization.py`
- 简化 `/kg-viz` 路由使用模板渲染
- 更新 `/api/v1/graph/hierarchical` 使用真实数据

### 删除/废弃的功能
- 移除了 HTML 按钮注入代码
- 旧模板仍保留但不再使用

### 设计系统

#### 颜色主题
- 主色: `#ff6b35` (橙色)
- 辅助色: `#667eea` (紫色)
- 背景: `#f8fafc` (浅灰)

#### 字体
- Inter (Google Fonts)

#### 图标
- Font Awesome 6

### 技术特性

1. **响应式设计** - 适配各种屏幕
2. **流畅动画** - 使用 CSS transitions
3. **交互反馈** - Toast 通知系统
4. **渐进加载** - 骨架屏和加载动画
5. **错误处理** - 友好的错误提示

### 如何启动

```bash
python launch_ui.py
```

访问 http://localhost:5000

### 端口
- Flask UI: 5000
- FastAPI (可选): 8000
