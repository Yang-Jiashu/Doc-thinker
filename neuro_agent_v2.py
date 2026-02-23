"""
NeuroAgent V2 - 集成上下文编排的智能体

新增功能：
1. 指令控制上下文提取
2. 层级化 KG 支持
3. 对话上下文演化
"""

import asyncio
import argparse
from typing import Any, Dict, List, Optional

from neuro_core import (
    MemoryEngine,
    HierarchicalKG,
    ContextComposer,
    CompositionConfig,
    ContextStrategy,
)
from cognition import CognitiveProcessor
from perception.chat import ChatPerceiver


class NeuroAgentV2:
    """
    NeuroAgent V2 - 带上下文编排的个人知识助手
    """
    
    def __init__(
        self,
        llm_func=None,
        embedding_func=None,
        working_dir: str = "./neuro_agent_data",
    ):
        # 初始化记忆引擎
        self.memory = MemoryEngine(
            embedding_func=embedding_func,
            llm_func=llm_func,
            working_dir=working_dir,
        )
        
        # 初始化层级化 KG
        self.hkg = HierarchicalKG(self.memory.graph)
        
        # 初始化上下文编排器
        self.composer = ContextComposer(self.hkg)
        
        # 其他组件
        self.llm_func = llm_func
        self.embedding_func = embedding_func
        self.cognition = CognitiveProcessor(
            llm_func=llm_func,
            embedding_func=embedding_func,
        )
        
        # 会话状态
        self.session_contexts: Dict[str, List[str]] = {}  # session_id -> used_node_ids
        
        # 加载已有记忆
        self.memory.load()
    
    async def respond(
        self,
        query: str,
        instruction: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        生成回答（带上下文编排）
        
        Args:
            query: 用户查询
            instruction: 控制指令（如"总结一下"、"详细说说"）
            session_id: 会话ID
        """
        # 1. 获取会话已使用的上下文
        used_nodes = self.session_contexts.get(session_id, []) if session_id else []
        
        # 2. 编排上下文
        context_nodes = self.composer.compose(
            query=query,
            instruction=instruction,
            existing_context=used_nodes if used_nodes else None,
        )
        
        # 3. 格式化上下文
        context_text = self.composer.format_context(context_nodes, format_type="text")
        
        # 4. 构建系统提示
        system_prompt = self._build_system_prompt(context_nodes)
        
        # 5. 构建用户提示
        user_prompt = f"""基于以下上下文回答用户问题。

{context_text}

用户问题: {query}

要求:
- 如果上下文信息不足，请说明
- 可以引用上下文中的具体信息
- 保持回答简洁、准确
"""
        
        # 6. 调用 LLM
        if self.llm_func:
            answer = await self.llm_func(user_prompt)
        else:
            answer = f"[模拟回答] 基于 {len(context_nodes)} 个上下文节点"
        
        # 7. 更新会话上下文
        if session_id:
            new_nodes = [n.node_id for n in context_nodes]
            if session_id not in self.session_contexts:
                self.session_contexts[session_id] = []
            self.session_contexts[session_id].extend(new_nodes)
        
        return {
            "answer": answer,
            "query": query,
            "instruction": instruction,
            "context_nodes": [
                {
                    "id": n.node_id,
                    "content": n.content[:100],
                    "level": n.level,
                    "relevance": n.relevance,
                }
                for n in context_nodes
            ],
            "context_stats": {
                "total_nodes": len(context_nodes),
                "abstract_nodes": len([n for n in context_nodes if n.level == 3]),
                "concept_nodes": len([n for n in context_nodes if n.level == 2]),
                "instance_nodes": len([n for n in context_nodes if n.level == 1]),
            }
        }
    
    def _build_system_prompt(self, context_nodes: List) -> str:
        """构建系统提示"""
        # 根据上下文组成调整系统角色
        if not context_nodes:
            return "你是一个知识助手。"
        
        # 分析上下文层级分布
        levels = [n.level for n in context_nodes]
        
        if sum(levels) / len(levels) > 2.5:  # 高层抽象多
            return "你是一个领域专家，擅长总结和概述。"
        elif sum(levels) / len(levels) < 1.5:  # 低层细节多
            return "你是一个技术专家，擅长详细解释和实现。"
        else:
            return "你是一个知识丰富的助手，能够平衡概览和细节。"
    
    async def add_document(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """添加文档到记忆"""
        from perception.document import DocumentPerceiver
        
        # 创建文档感知器
        doc_perceiver = DocumentPerceiver(
            cognitive_processor=self.cognition,
            extract_hierarchy=True,
        )
        
        # 感知文档
        result = await doc_perceiver.perceive(
            file_path=file_path,
            doc_id=doc_id,
        )
        
        # 添加到层级化 KG
        main_ep = result.episode
        hierarchy_result = self.hkg.add_episode_with_hierarchy(main_ep)
        
        return {
            "episode_id": main_ep.episode_id,
            "summary": main_ep.summary,
            "concepts": main_ep.concepts,
            "hierarchy": hierarchy_result,
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        return {
            "episodes": len(self.hkg._episode_index),
            "abstract_nodes": len(self.hkg._level_nodes[3]),
            "concept_nodes": len(self.hkg._level_nodes[2]),
            "instance_nodes": len(self.hkg._level_nodes[1]),
            "total_edges": len(self.hkg.graph.edges),
        }
    
    def reset_session(self, session_id: str):
        """重置会话上下文"""
        if session_id in self.session_contexts:
            del self.session_contexts[session_id]
    
    def save(self):
        """保存记忆"""
        self.memory.save()


async def interactive_chat(agent: NeuroAgentV2):
    """交互式聊天"""
    print("=" * 70)
    print("NeuroAgent V2 - 上下文编排版")
    print("=" * 70)
    print("\n指令示例:")
    print("  '总结一下深度学习' - 提取高层抽象")
    print("  '详细说说ResNet' - 提取具体细节")
    print("  '对比CNN和Transformer' - 关系聚焦")
    print("  'stats' - 查看记忆统计")
    print("  'reset' - 重置会话")
    print("  'quit' - 退出")
    print("=" * 70)
    
    session_id = f"session_{int(asyncio.get_event_loop().time())}"
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("\n保存记忆...")
                agent.save()
                print("再见!")
                break
            
            if user_input.lower() == "stats":
                stats = agent.get_memory_stats()
                print("\n记忆统计:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            if user_input.lower() == "reset":
                agent.reset_session(session_id)
                print("\n会话已重置")
                continue
            
            # 解析查询和指令
            # 简单规则: 如果包含控制词，整句作为指令
            control_words = ["总结", "详细", "对比", "深入", "相关"]
            instruction = None
            for word in control_words:
                if word in user_input:
                    instruction = user_input
                    break
            
            # 生成回答
            print("\n思考中...")
            result = await agent.respond(
                query=user_input,
                instruction=instruction,
                session_id=session_id,
            )
            
            print(f"\nAgent: {result['answer']}")
            
            # 显示上下文信息
            stats = result['context_stats']
            if stats['total_nodes'] > 0:
                print(f"\n[上下文] 共 {stats['total_nodes']} 个节点:", end="")
                if stats['abstract_nodes'] > 0:
                    print(f" 领域={stats['abstract_nodes']}", end="")
                if stats['concept_nodes'] > 0:
                    print(f" 概念={stats['concept_nodes']}", end="")
                if stats['instance_nodes'] > 0:
                    print(f" 实例={stats['instance_nodes']}", end="")
                print()
        
        except KeyboardInterrupt:
            print("\n\n保存记忆...")
            agent.save()
            print("再见!")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="NeuroAgent V2")
    parser.add_argument("--working-dir", default="./neuro_agent_data")
    args = parser.parse_args()
    
    # 模拟函数
    async def mock_llm(prompt: str) -> str:
        return f"[基于上下文生成回答] 输入长度: {len(prompt)}"
    
    def mock_embedding(texts):
        import random
        return [[random.random() for _ in range(128)] for _ in texts]
    
    # 创建 Agent
    print("初始化 NeuroAgent V2...")
    agent = NeuroAgentV2(
        llm_func=mock_llm,
        embedding_func=mock_embedding,
        working_dir=args.working_dir,
    )
    
    # 添加一些示例数据
    if agent.get_memory_stats()["episodes"] == 0:
        print("添加示例数据...")
        from neuro_core import Episode
        
        # 添加示例论文
        papers = [
            Episode(
                episode_id="paper_001",
                source_type="document",
                summary="Deep Residual Learning for Image Recognition",
                concepts=["CNN", "ResNet", "深度学习"],
                entity_ids=["ResNet", "ImageNet"],
            ),
            Episode(
                episode_id="paper_002",
                source_type="document",
                summary="Attention Is All You Need",
                concepts=["Transformer", "Attention", "深度学习"],
                entity_ids=["Transformer", "BERT"],
            ),
        ]
        for paper in papers:
            agent.hkg.add_episode_with_hierarchy(paper)
    
    print(f"初始化完成!")
    print(f"记忆统计: {agent.get_memory_stats()}")
    
    # 启动交互
    asyncio.run(interactive_chat(agent))


if __name__ == "__main__":
    main()
