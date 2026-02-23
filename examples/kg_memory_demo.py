"""
KG 架构记忆 + 自动联想演示

展示核心创新:
1. KG 作为记忆骨架：Episode 是节点，Entity/Relation 是边
2. 自动联想：写入时自动建立关联，查询时扩散激活
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuro_core import MemoryEngine, Episode
from neuro_core.knowledge_graph_memory import KGMemoryArchitecture
from neuro_core.auto_association import AutoAssociator, AssociativeMemoryTrigger


async def demo_kg_architecture():
    """演示 KG 架构记忆"""
    print("=" * 70)
    print("[Brain] 演示1: KG 作为记忆架构")
    print("=" * 70)
    
    # 创建记忆引擎
    memory = MemoryEngine(working_dir="./demo_data")
    kg = KGMemoryArchitecture(memory.graph)
    
    print("\n[In] 写入3个相关 Episode...")
    
    # Episode 1: 关于深度学习的对话
    ep1 = Episode(
        episode_id="ep-dl-001",
        source_type="chat",
        summary="用户询问深度学习的基本概念",
        key_points=["深度学习是机器学习的分支", "使用多层神经网络"],
        concepts=["深度学习", "神经网络", "机器学习"],
        entity_ids=["深度学习", "神经网络", "反向传播"],
        relation_triples=[("深度学习", "使用", "神经网络"), ("神经网络", "包含", "反向传播")],
    )
    kg.add_episode_to_kg(ep1)
    print(f"  [OK] 写入: {ep1.summary}")
    print(f"    - Entities: {ep1.entity_ids}")
    print(f"    - Relations: {ep1.relation_triples}")
    
    # Episode 2: 关于 CNN 的对话（相关）
    ep2 = Episode(
        episode_id="ep-cnn-002",
        source_type="chat",
        summary="用户询问 CNN 在图像识别中的应用",
        key_points=["CNN适合图像处理", "有卷积层和池化层"],
        concepts=["CNN", "图像识别", "深度学习"],  # 共享 "深度学习"
        entity_ids=["CNN", "卷积层", "深度学习"],  # 共享 "深度学习"
        relation_triples=[("CNN", "是一种", "神经网络"), ("CNN", "用于", "图像识别")],
    )
    kg.add_episode_to_kg(ep2)
    print(f"\n  [OK] 写入: {ep2.summary}")
    print(f"    - Entities: {ep2.entity_ids}")
    print(f"    - 与 ep-dl-001 共享 Entity: '深度学习'")
    
    # Episode 3: 关于天气的对话（不相关）
    ep3 = Episode(
        episode_id="ep-weather-003",
        source_type="chat",
        summary="用户询问明天天气",
        key_points=["需要查天气预报"],
        concepts=["天气", "预报"],
        entity_ids=["天气", "明天"],
        relation_triples=[],
    )
    kg.add_episode_to_kg(ep3)
    print(f"\n  [OK] 写入: {ep3.summary}")
    print(f"    - Entities: {ep3.entity_ids}")
    print(f"    - 与前面 Episodes 无共享 Entity")
    
    # 显示 KG 结构
    print("\n[Stats] KG 结构统计:")
    print(f"  - Episode 节点: {len([n for n in kg.graph.nodes if n.startswith('ep-')])}")
    print(f"  - Entity 节点: {len([n for n in kg.graph.nodes if not n.startswith('ep-')])}")
    print(f"  - 总边数: {len(kg.graph.edges)}")
    
    # 显示自动建立的关联
    print("\n[Link] 自动建立的关联:")
    for edge in kg.graph.edges:
        if edge.edge_type.value == "episode_similarity":
            print(f"  - {edge.source_id} ←{edge.edge_type.value}→ {edge.target_id}")
            if edge.metadata:
                print(f"    reason: {edge.metadata.get('reason', '')}")
    
    return kg, [ep1, ep2, ep3]


async def demo_spreading_activation(kg: KGMemoryArchitecture, episodes: list):
    """演示扩散激活联想"""
    print("\n" + "=" * 70)
    print("[Wave] 演示2: 扩散激活联想")
    print("=" * 70)
    
    print("\n[Target] 种子节点: ep-dl-001 (深度学习对话)")
    print("[Search] 扩散联想检索 (max_hops=2)...")
    
    results = kg.spreading_recall(
        seed_episode_ids=["ep-dl-001"],
        max_hops=2,
        min_activation=0.1
    )
    
    print(f"\n[Star] 激活的节点 (共 {len(results)} 个):")
    for node_id, score, path in results[:10]:
        if node_id.startswith("ep-"):
            ep = kg._episode_index.get(node_id)
            if ep:
                print(f"\n  [{score:.3f}] {node_id}")
                print(f"    摘要: {ep.summary[:50]}...")
                print(f"    联想路径: {' -> '.join(path)}")
        else:
            print(f"\n  [{score:.3f}] {node_id} (Entity)")
    
    # 对比：天气对话应该不会被激活
    print("\n[Pin] 观察:")
    print("  - ep-cnn-002 (CNN对话) 被高度激活（共享'深度学习'）")
    print("  - ep-weather-003 (天气对话) 未被激活（无关联）")


async def demo_auto_association(kg: KGMemoryArchitecture):
    """演示自动联想"""
    print("\n" + "=" * 70)
    print("[Bolt] 演示3: 自动联想（写入时触发）")
    print("=" * 70)
    
    associator = AutoAssociator(kg)
    
    # 模拟写入新 Episode 时的自动联想
    print("\n[Write] 写入新 Episode: '用户询问 RNN 和 LSTM'")
    new_ep = Episode(
        episode_id="ep-rnn-004",
        source_type="chat",
        summary="用户询问 RNN 和 LSTM 的区别",
        key_points=["RNN有梯度消失问题", "LSTM是解决之道"],
        concepts=["RNN", "LSTM", "深度学习", "序列模型"],
        entity_ids=["RNN", "LSTM", "深度学习"],  # 共享 "深度学习"
        relation_triples=[("LSTM", "解决", "梯度消失"), ("LSTM", "是", "RNN的变体")],
    )
    
    # 先加入 KG
    kg.add_episode_to_kg(new_ep)
    
    # 触发自动联想
    print("\n[Bell] 自动联想结果:")
    associations = await associator.on_insert_association(new_ep)
    
    print(f"  - 语义相似的 Episodes: {len(associations['semantic_similar'])}")
    print(f"  - 共享 Entities: {len(associations['shared_entities'])}")
    for assoc in associations['shared_entities']:
        ep = kg._episode_index.get(assoc['episode_id'])
        print(f"    * 与 '{ep.summary[:40]}...' 共享: {assoc['shared']}")
    
    print(f"  - 共享 Concepts: {len(associations['shared_concepts'])}")


async def demo_spontaneous_recall(kg: KGMemoryArchitecture):
    """演示自发回忆"""
    print("\n" + "=" * 70)
    print("[Think] 演示4: 自发回忆（被动触发）")
    print("=" * 70)
    
    associator = AutoAssociator(kg)
    trigger = AssociativeMemoryTrigger(associator)
    
    print("\n[Movie] 场景：用户提到了 '深度学习'")
    print("（系统没有主动查询，而是自动联想相关记忆）")
    
    recalls = trigger.activate("深度学习", intensity=1.0)
    
    print(f"\n[Idea] 自发浮现的记忆 ({len(recalls)} 个):")
    for recall in recalls:
        ep = recall["episode"]
        print(f"\n  [激活度: {recall['activation']:.2f}] {ep.episode_id}")
        print(f"    摘要: {ep.summary}")
        print(f"    触发词: {recall['trigger']}")


async def demo_memory_subgraph(kg: KGMemoryArchitecture):
    """演示记忆子图可视化"""
    print("\n" + "=" * 70)
    print("[Web] 演示5: 记忆子图（局部记忆网络）")
    print("=" * 70)
    
    print("\n[Pin] 以 ep-dl-001 为中心的记忆子图 (depth=2):")
    subgraph = kg.get_episode_subgraph("ep-dl-001", depth=2)
    
    print(f"\n  节点数: {len(subgraph['nodes'])}")
    print(f"  边数: {len(subgraph['edges'])}")
    
    print("\n  相关 Episodes:")
    for node_id, summary in subgraph['episodes'].items():
        print(f"    - {node_id}: {summary[:50]}...")
    
    print("\n  连接关系:")
    for edge in subgraph['edges'][:5]:
        print(f"    {edge['source']} --[{edge['type']}]--> {edge['target']}")
    if len(subgraph['edges']) > 5:
        print(f"    ... 还有 {len(subgraph['edges']) - 5} 条边")


async def main():
    print("\n" + "=" * 60)
    print("   KG架构记忆 + 自动联想 演示")
    print("   核心创新：以 KG 为骨架，Episode 为节点，自动联想")
    print("=" * 60 + "\n")
    
    # 演示1: KG 架构
    kg, episodes = await demo_kg_architecture()
    
    # 演示2: 扩散激活
    await demo_spreading_activation(kg, episodes)
    
    # 演示3: 自动联想
    await demo_auto_association(kg)
    
    # 演示4: 自发回忆
    await demo_spontaneous_recall(kg)
    
    # 演示5: 记忆子图
    await demo_memory_subgraph(kg)
    
    print("\n" + "=" * 70)
    print("[Star] 演示完成！")
    print("\n核心创新总结:")
    print("  1. KG 是记忆架构：Episode/Entity/Relation 都是 KG 节点")
    print("  2. 自动联想：写入时自动建立关联，无需人工标注")
    print("  3. 扩散激活：查询时沿图传播，多路径联想")
    print("  4. 自发回忆：关键词触发，被动浮现相关记忆")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
