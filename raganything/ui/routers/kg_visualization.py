"""
知识图谱可视化 API 路由
"""

from flask import Blueprint, jsonify, render_template

kg_viz_bp = Blueprint('kg_viz', __name__)

@kg_viz_bp.route('/kg-viz')
def kg_viz_page():
    """知识图谱可视化页面 - 使用现代化模板"""
    return render_template('kg_viz_modern.html')

@kg_viz_bp.route('/api/v1/graph/hierarchical')
def get_hierarchical_graph():
    """获取层级化知识图谱数据 - 返回兼容前端的数据格式"""
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    
    # 尝试从实际KG加载数据
    try:
        from neuro_core.cognitive.memory import CognitiveMemory
        memory = CognitiveMemory(storage_dir='./neuro_memory_verify_data')
        kg = memory.hierarchical_kg
        
        # 转换为前端需要的格式
        graph = {}
        for node_id, node in kg.graph.items():
            level = node.get('level', 1)
            graph[node_id] = {
                'name': node.get('name', node.get('title', node_id)),
                'level': level,
                'description': node.get('description', ''),
                'activation': node.get('activation', 0),
                'connections': list(node.get('connections', set()))
            }
        return jsonify({'graph': graph})
    except Exception as e:
        # 如果失败，返回空数据让前端使用演示数据
        return jsonify({'graph': {}, 'error': str(e)})
