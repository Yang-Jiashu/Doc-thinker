# 知识图谱可视化使用指南

## 概述

NeuroAgent 现在支持交互式知识图谱可视化，可以直观展示 **Domain → Concept → Instance** 三层结构。

## 启动 UI

```bash
# 方式1: 使用 V2 启动脚本
python run_ui_v2.py

# 方式2: 指定端口
python run_ui_v2.py --port 8080

# 方式3: 使用原启动脚本
python run_ui.py
```

## 访问可视化

启动后打开浏览器访问：

- **新版本**: http://localhost:5000/kg-viz
- **原版本**: http://localhost:5000/knowledge-graph

## 可视化功能

### 1. 层级展示

| 层级 | 颜色 | 节点大小 | 示例 |
|------|------|---------|------|
| Domain (领域) | 红色 | 最大 | 人工智能、医学 |
| Concept (概念) | 蓝色 | 中等 | 深度学习、CNN |
| Instance (实例) | 绿色 | 最小 | ResNet论文 |

### 2. 交互操作

| 操作 | 说明 |
|------|------|
| **拖拽节点** | 调整节点位置 |
| **滚轮缩放** | 放大/缩小视图 |
| **点击节点** | 查看节点详情 |
| **悬停节点** | 显示工具提示 |

### 3. 筛选功能

- **仅领域**: 只显示 L3 Domain 节点
- **仅概念**: 只显示 L2 Concept 节点
- **仅实例**: 只显示 L1 Instance 节点
- **全部**: 显示所有层级

### 4. 统计面板

实时显示：
- 领域数量
- 概念数量
- 实例数量
- 总节点数
- 总关系数

## API 接口

### 获取层级化图谱数据

```bash
GET http://localhost:5000/api/v1/graph/hierarchical
```

返回：
```json
{
  "status": "success",
  "data": {
    "nodes": [
      {"id": "domain:ai", "name": "人工智能", "level": 3, "type": "domain"},
      {"id": "concept:dl", "name": "深度学习", "level": 2, "type": "concept"},
      {"id": "paper:resnet", "name": "ResNet论文", "level": 1, "type": "instance"}
    ],
    "links": [
      {"source": "domain:ai", "target": "concept:dl", "type": "contains"},
      {"source": "concept:dl", "target": "paper:resnet", "type": "has_instance"}
    ]
  }
}
```

### 获取图谱统计

```bash
GET http://localhost:5000/api/v1/graph/stats
```

返回：
```json
{
  "status": "success",
  "data": {
    "total_nodes": 100,
    "total_edges": 150,
    "domain_count": 5,
    "concept_count": 30,
    "instance_count": 65
  }
}
```

## 集成 neuro_core

如果系统中已有 `neuro_core` 数据，可视化会自动加载真实数据：

```python
# neuro_core 数据自动加载流程
1. 检查 neuro_core 是否可用
2. 如果可用，从 HierarchicalKG 加载节点和边
3. 如果不可用，使用模拟数据展示
```

数据加载优先级：
1. `neuro_core.HierarchicalKG` (真实数据)
2. Mock Data (演示数据)

## 自定义样式

编辑核心库下 `ui/templates/kg_visualization.html` 修改样式：

```css
/* 修改层级颜色 */
.level-3 { fill: #e74c3c; }  /* Domain */
.level-2 { fill: #3498db; }  /* Concept */
.level-1 { fill: #2ecc71; }  /* Instance */

/* 修改节点大小 */
.attr('r', d => d.level === 3 ? 30 : d.level === 2 ? 20 : 15)
```

## 故障排除

### 页面显示空白

1. 检查浏览器控制台是否有 JavaScript 错误
2. 确认 D3.js CDN 可访问
3. 尝试刷新页面

### API 返回错误

1. 检查 Flask 服务是否正常运行
2. 确认 `neuro_core` 模块可导入
3. 查看控制台错误日志

### 中文显示乱码

1. 确保 HTML 文件使用 UTF-8 编码
2. 检查浏览器编码设置

## 未来优化

- [ ] 支持节点搜索高亮
- [ ] 添加关系类型筛选
- [ ] 支持导出图片 (PNG/SVG)
- [ ] 添加时间轴视图
- [ ] 支持 3D 可视化

## 截图示例

```
                    [人工智能]           ← Domain (红色，大)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   [深度学习]      [计算机视觉]     [NLP]    ← Concept (蓝色，中)
        │               │               │
    [ResNet]       [CNN笔记]      [BERT]   ← Instance (绿色，小)
```
