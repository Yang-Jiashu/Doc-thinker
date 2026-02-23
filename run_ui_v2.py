#!/usr/bin/env python3
"""
NeuroAgent UI V2 - 启动脚本

启动 Flask UI 服务，包含知识图谱可视化

用法:
    python run_ui_v2.py              # 启动 UI (默认端口 5000)
    python run_ui_v2.py --port 8080  # 指定端口
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroAgent UI V2")
    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=5000, help="端口号")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()
    
    # 确保工作目录正确
    os.chdir(Path(__file__).parent)
    
    # 导入 Flask 应用
    try:
        from raganything.ui.app import app, config
        
        # 覆盖配置
        config.ui_host = args.host
        config.ui_port = args.port
        
        print("\n" + "=" * 60)
        print("  NeuroAgent UI V2")
        print("=" * 60)
        print(f"\n  服务地址:")
        print(f"    - 主界面: http://{args.host}:{args.port}/query")
        print(f"    - 知识图谱 (新): http://{args.host}:{args.port}/kg-viz")
        print(f"    - 原知识图谱: http://{args.host}:{args.port}/knowledge-graph")
        print(f"\n  API 端点:")
        print(f"    - KG 数据: http://{args.host}:{args.port}/api/v1/graph/hierarchical")
        print(f"    - KG 统计: http://{args.host}:{args.port}/api/v1/graph/stats")
        print("\n" + "=" * 60 + "\n")
        
        # 启动应用
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=False
        )
        
    except ImportError as e:
        print(f"\n错误: 无法导入 Flask 应用: {e}")
        print("请确保已安装依赖: pip install flask flask-cors requests\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
