"""
NeuroAgent 基础使用示例

演示如何使用类人脑智能体进行对话和记忆管理
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuro_core import MemoryEngine
from cognition import CognitiveProcessor
from agent import NeuroAgent, SessionManager
from perception.chat import ChatPerceiver


# 模拟 LLM 函数
async def mock_llm(prompt: str) -> str:
    """模拟 LLM 调用"""
    import json
    if "理解" in prompt or "understanding" in prompt.lower():
        return json.dumps({
            "summary": "用户询问AI相关问题",
            "key_points": ["AI介绍", "机器学习算法"],
            "concepts": ["人工智能", "机器学习"],
            "reasoning": "用户想了解AI领域",
            "action_items": []
        })
    return f"这是模拟回答。提示长度: {len(prompt)}"


# 模拟 Embedding 函数
def mock_embedding(texts):
    """模拟 Embedding 调用"""
    import random
    return [[random.random() for _ in range(128)] for _ in texts]


async def main():
    print("=" * 60)
    print("NeuroAgent 基础使用示例")
    print("=" * 60)
    
    # 1. 创建 Agent
    print("\n1. 创建 NeuroAgent...")
    agent = NeuroAgent(
        llm_func=mock_llm,
        embedding_func=mock_embedding,
        working_dir="./example_data",
    )
    
    # 2. 注册感知器
    print("\n2. 注册 ChatPerceiver...")
    chat_perceiver = ChatPerceiver(cognitive_processor=agent.cognition)
    agent.register_perceiver("chat", chat_perceiver)
    
    # 3. 模拟感知对话并存储记忆
    print("\n3. 存储记忆...")
    
    # 第一次对话
    messages1 = [
        {"role": "user", "content": "你好，我想了解一下人工智能"},
        {"role": "assistant", "content": "你好！人工智能是一个广泛的领域，包括机器学习、深度学习等。"}
    ]
    ep1 = await agent.perceive(messages1, source_type="chat", session_id="session_001")
    print(f"  - 存储了 Episode: {ep1.episode_id[:20]}...")
    
    # 第二次对话
    messages2 = [
        {"role": "user", "content": "机器学习有哪些算法？"},
        {"role": "assistant", "content": "常见的机器学习算法包括：决策树、随机森林、SVM、神经网络等。"}
    ]
    ep2 = await agent.perceive(messages2, source_type="chat", session_id="session_001")
    print(f"  - 存储了 Episode: {ep2.episode_id[:20]}...")
    
    # 第三次对话（不同话题）
    messages3 = [
        {"role": "user", "content": "今天天气怎么样？"},
        {"role": "assistant", "content": "抱歉，我没有获取实时天气信息的能力。"}
    ]
    ep3 = await agent.perceive(messages3, source_type="chat", session_id="session_002")
    print(f"  - 存储了 Episode: {ep3.episode_id[:20]}...")
    
    # 4. 检索记忆
    print("\n4. 检索相关记忆...")
    
    # 检索 AI 相关记忆
    query1 = "人工智能"
    results1 = await agent.recall(query1, top_k=5)
    print(f"\n  查询: '{query1}'")
    print(f"  找到 {len(results1)} 条相关记忆:")
    for i, r in enumerate(results1, 1):
        print(f"    {i}. [{r['source']}] {r['summary'][:50]}... (score: {r['score']:.3f})")
    
    # 检索机器学习相关记忆
    query2 = "机器学习算法"
    results2 = await agent.recall(query2, top_k=5)
    print(f"\n  查询: '{query2}'")
    print(f"  找到 {len(results2)} 条相关记忆:")
    for i, r in enumerate(results2, 1):
        print(f"    {i}. [{r['source']}] {r['summary'][:50]}... (score: {r['score']:.3f})")
    
    # 5. 生成回答
    print("\n5. 生成回答...")
    response = await agent.respond("什么是深度学习？", session_id="session_001")
    print(f"  查询: {response['query']}")
    print(f"  回答: {response['answer'][:100]}...")
    print(f"  检索到 {len(response['memories'])} 条记忆")
    
    # 6. 记忆统计
    print("\n6. 记忆统计...")
    episodes = agent.memory.episode_store.all_episodes()
    print(f"  - 总 Episode 数: {len(episodes)}")
    print(f"  - 图节点数: {len(agent.memory.graph._nodes)}")
    print(f"  - 图边数: {len(agent.memory.graph._edge_index)}")
    
    # 7. 保存记忆
    print("\n7. 保存记忆...")
    agent.save()
    print("  - 记忆已保存到 ./example_data")
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
