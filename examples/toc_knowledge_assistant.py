"""
ToC 个人知识助手演示

场景: 用户上传学习资料，系统自动构建层级化知识图谱
核心: 文档是输入，KG 自动联想生成新边/节点
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuro_core import MemoryEngine, Episode
from neuro_core.hierarchical_kg import HierarchicalKG, HierarchyLevel
from neuro_core.knowledge_graph_memory import KGMemoryArchitecture
from perception.document import DocumentPerceiver


async def demo_document_to_hierarchy():
    """演示: 文档 -> 层级化 KG"""
    print("=" * 70)
    print("ToC 场景: 文档感知 -> 层级化知识图谱")
    print("=" * 70)
    
    # 初始化
    memory = MemoryEngine(working_dir="./toc_demo_data")
    kg = KGMemoryArchitecture(memory.graph)
    hkg = HierarchicalKG(kg.graph)
    
    # 创建文档感知器
    doc_perceiver = DocumentPerceiver(
        extract_hierarchy=True,
    )
    
    print("\n[场景1] 用户上传 AI 论文")
    print("-" * 70)
    
    # 模拟文档路径
    doc_path = "papers/deep_learning_survey.pdf"
    
    # 感知文档
    result = await doc_perceiver.perceive(
        doc_path,
        doc_id="doc_dl_001",
        user_context={"user_id": "user_001", "importance": "high"}
    )
    
    main_ep = result.episode
    print(f"\n主 Episode 生成:")
    print(f"  ID: {main_ep.episode_id}")
    print(f"  标题: {main_ep.summary}")
    print(f"  概念: {main_ep.concepts}")
    print(f"  领域: {result.metadata.get('domain', 'unknown')}")
    
    # 添加到层级化 KG
    hierarchy_result = hkg.add_episode_with_hierarchy(main_ep)
    
    print(f"\n层级化关联建立:")
    print(f"  关联概念数: {len(hierarchy_result['concepts_linked'])}")
    print(f"  关联领域: {hierarchy_result['domains_linked']}")
    
    # 显示向上抽象
    print(f"\n向上抽象 (从实例到高阶):")
    abstractions = hkg.upward_abstraction(main_ep.episode_id, max_levels=2)
    for abs_item in abstractions:
        level_name = {3: "领域", 2: "概念"}.get(abs_item["level"], "未知")
        print(f"  -> [{level_name}] {abs_item['name']} (权重: {abs_item['weight']:.2f})")
    
    return hkg, main_ep


async def demo_cross_document_association(hkg: HierarchicalKG):
    """演示: 跨文档自动联想"""
    print("\n" + "=" * 70)
    print("[场景2] 用户上传另一篇相关论文")
    print("=" * 70)
    
    # 第二篇文档
    ep2 = Episode(
        episode_id="doc_cnn_002",
        source_type="document",
        summary="CNN 图像识别最新进展",
        concepts=["CNN", "图像识别", "深度学习"],  # 共享 "深度学习"
        entity_ids=["ResNet", "ImageNet"],
        relation_triples=[("ResNet", "用于", "图像识别")],
    )
    
    print(f"\n新文档: {ep2.summary}")
    print(f"  概念: {ep2.concepts}")
    
    # 添加到层级化 KG (自动建立关联)
    result = hkg.add_episode_with_hierarchy(ep2)
    
    print(f"\n自动联想结果:")
    print(f"  关联到概念: {result['concepts_linked']}")
    
    # 检查是否自动关联到第一篇文档
    print(f"\n跨文档关联:")
    print(f"  - 两篇文档共享概念: '深度学习'")
    print(f"  - 自动建立边: doc_cnn_002 --[EPISODE_SIMILARITY]--> doc_dl_001")
    
    # 向下具体化: 从概念找到所有相关文档
    print(f"\n向下具体化 (从概念找实例):")
    print("  查询: '深度学习' 领域有哪些文档?")
    
    concept_id = "concept:深度学习"
    instances = hkg.downward_concretization(concept_id, max_depth=2)
    
    print(f"  找到 {len(instances)} 个相关实例:")
    for inst in instances[:5]:
        if inst.get("episode"):
            ep = inst["episode"]
            print(f"    - [{inst['activation']:.2f}] {ep.summary}")


async def demo_hierarchical_query(hkg: HierarchicalKG):
    """演示: 层级化查询"""
    print("\n" + "=" * 70)
    print("[场景3] 层级化联想查询")
    print("=" * 70)
    
    # 场景: 用户提到一个高阶概念
    print("\n用户查询: '人工智能'")
    print("(这是一个高阶领域概念)")
    
    domain_id = "domain:人工智能"
    
    # 向下具体化: 找到该领域下的所有内容
    results = hkg.downward_concretization(
        domain_id,
        max_depth=3,
        min_activation=0.2
    )
    
    # 按层级分组
    by_level = {3: [], 2: [], 1: []}
    for r in results:
        level = r.get("level", 0)
        if level in by_level:
            by_level[level].append(r)
    
    print("\n层级化联想结果:")
    
    # Level 3: 领域
    print(f"\n  [领域层] 人工智能")
    
    # Level 2: 概念
    print(f"\n  [概念层] 相关技术:")
    for concept in by_level[HierarchyLevel.CONCEPT][:5]:
        print(f"    - {concept.get('name', '')} (激活: {concept['activation']:.2f})")
    
    # Level 1: 实例
    print(f"\n  [实例层] 具体文档:")
    for inst in by_level[HierarchyLevel.INSTANCE][:5]:
        if inst.get("episode"):
            ep = inst["episode"]
            print(f"    - [{inst['activation']:.2f}] {ep.episode_id}: {ep.summary}")


async def demo_spontaneous_learning(hkg: HierarchicalKG):
    """演示: 自动联想促进学习"""
    print("\n" + "=" * 70)
    print("[场景4] 自动联想促进学习")
    print("=" * 70)
    
    print("\n场景: 用户正在阅读一篇新论文")
    print("系统检测到新概念 'Transformer'")
    
    # 新文档
    new_ep = Episode(
        episode_id="doc_transformer_003",
        source_type="document",
        summary="Attention Is All You Need",
        concepts=["Transformer", "Attention", "深度学习"],
        entity_ids=["BERT", "GPT", "Self-Attention"],
    )
    
    # 添加到 KG
    hkg.add_episode_with_hierarchy(new_ep)
    
    print("\n系统自动联想:")
    
    # 向上抽象: 发现属于 "人工智能" 领域
    abstractions = hkg.upward_abstraction(new_ep.episode_id)
    for abs_item in abstractions:
        if "深度学习" in abs_item.get("name", ""):
            print(f"  [OK] 向上: 这属于 '{abs_item['name']}' 领域")
    
    # 横向关联: 找到同领域其他论文
    print(f"\n  [OK] 横向: 该领域你还有以下笔记:")
    concept_id = "concept:深度学习"
    related = hkg.downward_concretization(concept_id)
    for r in related[:3]:
        if r.get("episode") and r["episode"].episode_id != new_ep.episode_id:
            print(f"    - {r['episode'].summary}")
    
    print(f"\n  [Tip] 系统提示:")
    print(f"     '这篇论文与之前的 CNN 论文都涉及深度学习，")
    print(f"      建议对比 CNN 和 Transformer 的异同。'")


async def demo_kg_evolution(hkg: HierarchicalKG):
    """演示: KG 随时间演化"""
    print("\n" + "=" * 70)
    print("[场景5] 知识图谱演化")
    print("=" * 70)
    
    print("\n知识图谱统计:")
    stats = {
        "instance_nodes": len(hkg._level_nodes[HierarchyLevel.INSTANCE]),
        "concept_nodes": len(hkg._level_nodes[HierarchyLevel.CONCEPT]),
        "domain_nodes": len(hkg._level_nodes[HierarchyLevel.ABSTRACT]),
        "total_edges": len(hkg.graph.edges),
    }
    
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n概念层级结构:")
    hierarchy = hkg.get_concept_hierarchy()
    for domain_info in hierarchy.get("domains", []):
        domain_name = domain_info.get("name", domain_info["id"])
        print(f"\n  [领域] {domain_name}")
        
        # 获取该领域的概念
        domain_id = domain_info["id"]
        concepts = hkg._domain_concepts.get(domain_id, set())
        for concept_id in list(concepts)[:5]:
            concept_name = concept_id.replace("concept:", "")
            print(f"    --- [概念] {concept_name}")
            
            # 获取该概念的实例
            instances = hkg._get_concept_instances(concept_id)
            for inst_id in instances[:3]:
                print(f"        --- [实例] {inst_id}")


async def main():
    print("\n" + "=" * 70)
    print("ToC 个人知识助手 - 层级化 KG 演示")
    print("核心: 文档 -> 层级化记忆 -> 自动联想")
    print("=" * 70)
    
    # 场景1: 文档感知
    hkg, main_ep = await demo_document_to_hierarchy()
    
    # 场景2: 跨文档联想
    await demo_cross_document_association(hkg)
    
    # 场景3: 层级化查询
    await demo_hierarchical_query(hkg)
    
    # 场景4: 学习辅助
    await demo_spontaneous_learning(hkg)
    
    # 场景5: KG 演化
    await demo_kg_evolution(hkg)
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("\n核心价值:")
    print("  1. 文档是用户知识的自然输入方式")
    print("  2. KG 自动从文档提取层级结构 (高阶->低阶)")
    print("  3. 自动联想生成新边/节点，构建知识网络")
    print("  4. 支持向上抽象和向下具体化的双向联想")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
