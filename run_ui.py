#!/usr/bin/env python3
"""
whitecat UI 启动脚本

在项目根目录执行:
  python run_ui.py

将启动 Flask UI (端口 5000)，并强制从本项目核心库的 ui/templates 加载模板。
聊天页: http://127.0.0.1:5000/query
知识图谱: http://127.0.0.1:5000/knowledge-graph
KG可视化: http://127.0.0.1:5000/kg-viz  (新)
"""
import os
import sys

# 确保当前目录为项目根，并优先使用本项目代码
_ROOT = os.path.abspath(os.path.dirname(__file__))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 启动 UI（使用本项目内的 app，保证模板从项目内加载）
from raganything.ui.app import app, config

# 注册知识图谱可视化蓝图（如果尚未注册）
try:
    from raganything.ui.routers.kg_visualization import kg_viz_bp
    if 'kg_viz.kg_viz_page' not in [rule.endpoint for rule in app.url_map.iter_rules()]:
        app.register_blueprint(kg_viz_bp)
        print("[whitecat] 知识图谱可视化路由已注册: /kg-viz")
except Exception as e:
    print(f"[whitecat] 知识图谱可视化路由注册失败: {e}")

if __name__ == "__main__":
    host = getattr(config, "ui_host", "0.0.0.0")
    port = getattr(config, "ui_port", 5000)
    print()
    print("  ========================================")
    print("  whitecat UI")
    print("  ========================================")
    print("  聊天:     http://127.0.0.1:{}/query".format(port))
    print("  KG管理:   http://127.0.0.1:{}/knowledge-graph".format(port))
    print("  KG可视化: http://127.0.0.1:{}/kg-viz".format(port))
    print("  ========================================")
    print()
    app.run(host=host, port=port, debug=False, use_reloader=False)
