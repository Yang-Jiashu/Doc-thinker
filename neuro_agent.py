"""
NeuroAgent - 类人脑智能体 (增强版)

核心特性：
1. KG-Based Memory Architecture - 以知识图谱为记忆骨架
2. Auto-Association - 自动联想机制
"""

import asyncio
import argparse
from typing import Any, Dict, List, Optional

from neuro_core import (
    MemoryEngine,
    KGMemoryArchitecture,
    AutoAssociator,
    AssociativeMemoryTrigger,
)
from cognition import CognitiveProcessor
from agent import NeuroAgent, SessionManager
from perception.chat import ChatPerceiver
from api.server import create_app


class NeuroAgentKG(NeuroAgent):
    """
    增强版 NeuroAgent，集成 KG 架构和自动联想
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化 KG 架构
        self.kg_arch = KGMemoryArchitecture(self.memory.graph)
        
        # 初始化自动联想器
        self.auto_assoc = AutoAssociator(self.kg_arch)
        self.assoc_trigger = AssociativeMemoryTrigger(self.auto_assoc)
        
        # 注册联想钩子
        self.auto_assoc.register_hook(self._on_association)
    
    async def _on_association(self, episode, associations):
        """联想触发时的回调"""
        if associations.get('shared_entities'):
            print(f"[联想] 新记忆与 {len(associations['shared_entities'])} 个已有记忆相关")
    
    async def perceive(self, input_data: Any, source_type: str = "chat", **kwargs):
        """
        感知输入（增强版）
        
        1. 调用父类感知
        2. 使用 KG 架构添加 Episode
        3. 触发自动联想
        """
        # 使用感知器生成 Episode
        perceiver = self.perceivers.get(source_type)
        if not perceiver:
            raise ValueError(f"No perceiver for {source_type}")
        
        result = await perceiver.perceive(input_data, **kwargs)
        episode = result.episode
        
        # 使用 KG 架构添加（自动建立关联）
        self.kg_arch.add_episode_to_kg(episode)
        
        # 触发自动联想
        await self.auto_assoc.on_insert_association(episode)
        
        return episode
    
    async def recall(self, query: str, top_k: int = 10, **kwargs) -> List[Dict]:
        """
        回忆（增强版 - 使用 KG 扩散联想）
        """
        # 先进行向量检索找到种子
        query_emb = await self.memory._embed([query])
        if not query_emb or not query_emb[0]:
            return []
        
        similar = self.memory.episode_vectors.query(query_emb[0], top_k=3)
        seed_ids = [ep_id for ep_id, _ in similar]
        
        if not seed_ids:
            return []
        
        # 使用 KG 扩散联想
        results = self.kg_arch.spreading_recall(
            seed_episode_ids=seed_ids,
            max_hops=2,
            min_activation=0.1
        )
        
        # 格式化结果
        formatted = []
        for node_id, score, path in results[:top_k]:
            if node_id.startswith("ep-"):
                ep = self.kg_arch._episode_index.get(node_id)
                if ep:
                    formatted.append({
                        "episode": ep,
                        "score": score,
                        "path": path,
                        "summary": ep.summary,
                    })
        
        return formatted
    
    async def spontaneous_recall(self, trigger: str) -> List[Dict]:
        """
        自发回忆（被动联想）
        
        当系统检测到关键词时，自动浮现相关记忆
        """
        recalls = self.assoc_trigger.activate(trigger, intensity=1.0)
        return [
            {
                "episode": r["episode"],
                "activation": r["activation"],
                "trigger": r["trigger"],
            }
            for r in recalls
        ]
    
    def get_memory_subgraph(self, episode_id: str, depth: int = 2) -> Dict:
        """获取记忆子图（用于可视化）"""
        return self.kg_arch.get_episode_subgraph(episode_id, depth)
    
    def get_memory_stats(self) -> Dict:
        """获取记忆统计"""
        episodes = self.kg_arch._episode_index
        nodes = self.kg_arch.graph.nodes
        edges = self.kg_arch.graph.edges
        
        return {
            "total_episodes": len(episodes),
            "total_nodes": len(nodes),
            "entity_nodes": len([n for n in nodes if not n.startswith("ep-")]),
            "total_edges": len(edges),
            "edge_types": self._count_edge_types(edges),
        }
    
    def _count_edge_types(self, edges) -> Dict[str, int]:
        """统计边类型分布"""
        counts = {}
        for edge in edges:
            et = edge.edge_type.value
            counts[et] = counts.get(et, 0) + 1
        return counts


async def interactive_chat(agent: NeuroAgentKG):
    """交互式聊天（KG增强版）"""
    print("=" * 60)
    print("NeuroAgent KG - 类人脑智能体")
    print("输入 'quit' 退出, 'stats' 查看记忆统计")
    print("输入 'subgraph <ep_id>' 查看记忆子图")
    print("=" * 60)
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if not query:
                continue
            
            if query.lower() == "quit":
                agent.save()
                print("记忆已保存，再见!")
                break
            
            if query.lower() == "stats":
                stats = agent.get_memory_stats()
                print("\n记忆统计:")
                for key, value in stats.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
                continue
            
            if query.startswith("subgraph "):
                ep_id = query.split()[1]
                subgraph = agent.get_memory_subgraph(ep_id)
                print(f"\n记忆子图 ({ep_id}):")
                print(f"  节点: {subgraph.get('nodes', [])}")
                continue
            
            # 生成回答
            print("思考中...")
            result = await agent.respond(query)
            
            print(f"\nAgent: {result['answer']}")
            
            # 显示检索路径
            if result.get('memories'):
                print(f"\n相关记忆 ({len(result['memories'])}):")
                for i, mem in enumerate(result['memories'][:3], 1):
                    path_str = " -> ".join(mem.get('path', [])[-3:])  # 只显示最后3跳
                    print(f"  {i}. [{mem['score']:.2f}] {mem['summary'][:50]}...")
                    print(f"     路径: {path_str}")
        
        except KeyboardInterrupt:
            agent.save()
            print("\n记忆已保存，再见!")
            break
        except Exception as e:
            print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="NeuroAgent KG")
    parser.add_argument("--server", action="store_true", help="启动 API 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--working-dir", default="./neuro_agent_data")
    
    args = parser.parse_args()
    
    # 模拟 LLM 和 Embedding
    async def mock_llm(prompt: str) -> str:
        return f"[LLM回答] 基于: {prompt[:100]}..."
    
    def mock_embedding(texts):
        import random
        return [[random.random() for _ in range(128)] for _ in texts]
    
    # 创建增强版 Agent
    agent = NeuroAgentKG(
        llm_func=mock_llm,
        embedding_func=mock_embedding,
        working_dir=args.working_dir,
    )
    
    # 注册感知器
    chat_perceiver = ChatPerceiver(cognitive_processor=agent.cognition)
    agent.register_perceiver("chat", chat_perceiver)
    
    print(f"Agent 初始化完成")
    print(f"工作目录: {args.working_dir}")
    
    if args.server:
        import uvicorn
        app = create_app(agent)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        asyncio.run(interactive_chat(agent))


if __name__ == "__main__":
    main()
