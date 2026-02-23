"""
上下文编排器演示

展示如何根据指令智能选择 KG 的不同部分作为上下文
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuro_core import MemoryEngine, Episode
from neuro_core.hierarchical_kg import HierarchicalKG, HierarchyLevel
from neuro_core.context_composer import ContextComposer, CompositionConfig, ContextStrategy


async def setup_demo_kg():
    """设置演示用的层级化 KG"""
    memory = MemoryEngine(working_dir="./context_demo_data")
    hkg = HierarchicalKG(memory.graph)
    
    # 添加多层级的知识结构
    
    # Level 1: 具体论文 (Instance)
    papers = [
        Episode(
            episode_id="paper_001",
            source_type="document",
            summary="ResNet: Deep Residual Learning for Image Recognition",
            key_points=["提出残差连接", "解决梯度消失", "152层网络"],
            concepts=["CNN", "ResNet", "深度学习", "计算机视觉"],
            entity_ids=["ResNet", "ImageNet", "卷积层"],
        ),
        Episode(
            episode_id="paper_002",
            source_type="document",
            summary="Attention Is All You Need (Transformer)",
            key_points=["提出自注意力机制", "无需RNN", "并行计算"],
            concepts=["Transformer", "Attention", "深度学习", "NLP"],
            entity_ids=["Transformer", "BERT", "GPT", "Self-Attention"],
        ),
        Episode(
            episode_id="paper_003",
            source_type="document",
            summary="Generative Adversarial Networks",
            key_points=["生成器与判别器对抗", "生成逼真图像"],
            concepts=["GAN", "生成模型", "深度学习", "计算机视觉"],
            entity_ids=["GAN", "生成器", "判别器"],
        ),
        Episode(
            episode_id="paper_004",
            source_type="document",
            summary="Deep Residual Networks with 1000+ Layers",
            key_points=["超深层网络", "残差学习", "性能分析"],
            concepts=["ResNet", "深度学习", "计算机视觉"],
            entity_ids=["ResNet", "梯度消失", "批量归一化"],
        ),
    ]
    
    for paper in papers:
        hkg.add_episode_with_hierarchy(paper)
    
    print(f"KG 构建完成:")
    print(f"  - Instance 节点: {len(hkg._level_nodes[HierarchyLevel.INSTANCE])}")
    print(f"  - Concept 节点: {len(hkg._level_nodes[HierarchyLevel.CONCEPT])}")
    print(f"  - Domain 节点: {len(hkg._level_nodes[HierarchyLevel.ABSTRACT])}")
    
    return hkg


async def demo_intent_parsing():
    """演示意图解析"""
    print("\n" + "=" * 70)
    print("[演示1] 意图解析：从指令提取策略")
    print("=" * 70)
    
    from neuro_core.context_composer import IntentParser
    
    parser = IntentParser()
    
    test_instructions = [
        "总结一下深度学习的要点",
        "详细说说 ResNet 的技术细节",
        "对比 CNN 和 Transformer 的区别",
        "相关的论文都给我",
        "深入讲讲注意力机制的原理",
        "给我关于 GAN 的所有信息",
    ]
    
    for instruction in test_instructions:
        config = parser.parse_instruction(instruction)
        print(f"\n指令: '{instruction}'")
        print(f"  策略: {config.strategy.value}")
        print(f"  层级权重: 领域={config.hierarchy_weights[3]:.1f}, "
              f"概念={config.hierarchy_weights[2]:.1f}, "
              f"实例={config.hierarchy_weights[1]:.1f}")
        print(f"  最大节点: {config.max_nodes}")


async def demo_context_composition(hkg: HierarchicalKG):
    """演示上下文编排"""
    print("\n" + "=" * 70)
    print("[演示2] 上下文编排：不同策略的对比")
    print("=" * 70)
    
    composer = ContextComposer(hkg)
    
    query = "ResNet"
    
    strategies = [
        ("ABSTRACT_ONLY", "总结模式（只提取高层抽象）"),
        ("CONCRETE_ONLY", "细节模式（只提取具体细节）"),
        ("BALANCED", "平衡模式（各层兼顾）"),
        ("BREADTH_FIRST", "广度优先（多路径浅层）"),
        ("DEPTH_FIRST", "深度优先（单路径深入）"),
    ]
    
    for strategy_name, desc in strategies:
        print(f"\n{'-' * 70}")
        print(f"策略: {strategy_name} - {desc}")
        print(f"{'-' * 70}")
        
        config = CompositionConfig(
            strategy=ContextStrategy[strategy_name],
            max_nodes=5,
            include_paths=True,
        )
        
        nodes = composer.compose(query=query, config=config)
        
        print(f"检索到 {len(nodes)} 个节点:")
        for i, node in enumerate(nodes, 1):
            level_name = {3: "[领域]", 2: "[概念]", 1: "[实例]"}.get(node.level, "[?]")
            print(f"  {i}. {level_name} {node.content[:50]}")
            print(f"     相关性: {node.relevance:.3f}, 来源: {node.source}")
            if node.path:
                print(f"     路径: {' -> '.join(node.path[-3:])}")


async def demo_natural_language_control(hkg: HierarchicalKG):
    """演示自然语言控制"""
    print("\n" + "=" * 70)
    print("[演示3] 自然语言控制：一句话调整上下文")
    print("=" * 70)
    
    composer = ContextComposer(hkg)
    
    scenarios = [
        {
            "query": "深度学习",
            "instruction": "总结一下深度学习的主要概念",
            "expected": "提取高层领域和核心概念",
        },
        {
            "query": "Transformer",
            "instruction": "详细说说 Transformer 的技术细节和实现",
            "expected": "提取具体论文和技术要点",
        },
        {
            "query": "ResNet",
            "instruction": "对比 ResNet 相关的所有论文",
            "expected": "提取多篇论文进行关联",
        },
        {
            "query": "GAN",
            "instruction": "给我关于 GAN 的前3个关键信息",
            "expected": "限制数量，提取最相关的",
        },
    ]
    
    for scenario in scenarios:
        print(f"\n场景: {scenario['expected']}")
        print(f"查询: '{scenario['query']}'")
        print(f"指令: '{scenario['instruction']}'")
        print("-" * 50)
        
        nodes = composer.compose(
            query=scenario["query"],
            instruction=scenario["instruction"],
        )
        
        print(f"编排结果 ({len(nodes)} 个节点):")
        context_text = composer.format_context(nodes, format_type="text")
        print(context_text)


async def demo_layered_retrieval(hkg: HierarchicalKG):
    """演示分层检索控制"""
    print("\n" + "=" * 70)
    print("[演示4] 分层检索：精细控制每层数量")
    print("=" * 70)
    
    composer = ContextComposer(hkg)
    query = "深度学习"
    
    # 配置 1: 偏重抽象层（适合快速了解）
    print("\n[配置1] 偏重抽象层 (适合：快速了解领域)")
    config1 = CompositionConfig(
        strategy=ContextStrategy.BALANCED,
        hierarchy_weights={
            HierarchyLevel.ABSTRACT: 0.5,   # 高权重
            HierarchyLevel.CONCEPT: 0.3,
            HierarchyLevel.INSTANCE: 0.2,
        },
        max_nodes=6,
    )
    nodes1 = composer.compose(query=query, config=config1)
    print(f"检索结果: {len(nodes1)} 个节点")
    level_dist = {3: 0, 2: 0, 1: 0}
    for n in nodes1:
        level_dist[n.level] = level_dist.get(n.level, 0) + 1
    print(f"层级分布: 领域={level_dist[3]}, 概念={level_dist[2]}, 实例={level_dist[1]}")
    
    # 配置 2: 偏重实例层（适合深入研究）
    print("\n[配置2] 偏重实例层 (适合：深入研究具体论文)")
    config2 = CompositionConfig(
        strategy=ContextStrategy.BALANCED,
        hierarchy_weights={
            HierarchyLevel.ABSTRACT: 0.1,
            HierarchyLevel.CONCEPT: 0.2,
            HierarchyLevel.INSTANCE: 0.7,   # 高权重
        },
        max_nodes=6,
    )
    nodes2 = composer.compose(query=query, config=config2)
    print(f"检索结果: {len(nodes2)} 个节点")
    level_dist = {3: 0, 2: 0, 1: 0}
    for n in nodes2:
        level_dist[n.level] = level_dist.get(n.level, 0) + 1
    print(f"层级分布: 领域={level_dist[3]}, 概念={level_dist[2]}, 实例={level_dist[1]}")
    
    # 配置 3: 仅概念层（适合技术选型）
    print("\n[配置3] 仅概念层 (适合：了解技术方案)")
    config3 = CompositionConfig(
        strategy=ContextStrategy.BALANCED,
        hierarchy_weights={
            HierarchyLevel.ABSTRACT: 0.0,
            HierarchyLevel.CONCEPT: 1.0,    # 只有概念层
            HierarchyLevel.INSTANCE: 0.0,
        },
        max_nodes=6,
    )
    nodes3 = composer.compose(query=query, config=config3)
    print(f"检索结果: {len(nodes3)} 个节点")
    for n in nodes3:
        print(f"  - {n.content}")


async def demo_context_evolution(hkg: HierarchicalKG):
    """演示上下文随对话演化"""
    print("\n" + "=" * 70)
    print("[演示5] 上下文演化：多轮对话中的动态调整")
    print("=" * 70)
    
    composer = ContextComposer(hkg)
    
    # 模拟多轮对话
    conversation = [
        {
            "turn": 1,
            "user": "介绍一下深度学习",
            "instruction": "总结一下",
            "expected": "高层概述",
        },
        {
            "turn": 2,
            "user": "ResNet 是什么？",
            "instruction": "详细说说",
            "expected": "具体细节",
        },
        {
            "turn": 3,
            "user": "和 Transformer 有什么区别？",
            "instruction": "对比分析",
            "expected": "关联对比",
        },
    ]
    
    existing_context = []  # 维护已使用的上下文
    
    for turn in conversation:
        print(f"\n[第 {turn['turn']} 轮]")
        print(f"用户: {turn['user']}")
        print(f"指令: {turn['instruction']}")
        print(f"预期: {turn['expected']}")
        print("-" * 50)
        
        nodes = composer.compose(
            query=turn["user"],
            instruction=turn["instruction"],
            existing_context=existing_context,  # 去重
        )
        
        print(f"新增上下文 ({len(nodes)} 个节点):")
        for node in nodes:
            level_name = {3: "领域", 2: "概念", 1: "实例"}.get(node.level, "?")
            print(f"  [{level_name}] {node.content[:40]}")
            existing_context.append(node.node_id)
        
        print(f"\n累计使用节点: {len(existing_context)} 个")


async def main():
    print("\n" + "=" * 70)
    print("上下文编排器演示")
    print("核心：通过指令智能控制 KG 上下文提取")
    print("=" * 70)
    
    # 准备数据
    print("\n[准备] 构建演示知识图谱...")
    hkg = await setup_demo_kg()
    
    # 运行演示
    await demo_intent_parsing()
    await demo_context_composition(hkg)
    await demo_natural_language_control(hkg)
    await demo_layered_retrieval(hkg)
    await demo_context_evolution(hkg)
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("\n核心价值:")
    print("  1. 自然语言指令控制上下文提取策略")
    print("  2. 精细的层级控制：父节点/子节点/特定层")
    print("  3. 动态平衡：相关性 vs 多样性 vs Token 限制")
    print("  4. 可解释性：保留联想路径")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
