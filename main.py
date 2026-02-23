"""
NeuroAgent - 主入口

启动类人脑智能体

用法：
    python main.py              # 启动交互模式
    python main.py --server     # 启动 API 服务
    python main.py --chat       # 启动聊天模式
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 确保能导入本地模块
sys.path.insert(0, str(Path(__file__).parent))

from neuro_core import MemoryEngine
from cognition import CognitiveProcessor
from retrieval import HybridRetriever
from agent import NeuroAgent, SessionManager
from perception.chat import ChatPerceiver
from api.server import create_app


async def interactive_chat(agent: NeuroAgent):
    """交互式聊天"""
    print("=" * 50)
    print("🧠 NeuroAgent - 类人脑智能体")
    print("输入 'quit' 退出, 'save' 保存记忆, 'consolidate' 巩固记忆")
    print("=" * 50)
    
    session_id = None
    
    while True:
        try:
            query = input("\n👤 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() == "quit":
                print("💾 保存记忆...")
                agent.save()
                print("👋 再见!")
                break
            
            if query.lower() == "save":
                agent.save()
                print("✅ 记忆已保存")
                continue
            
            if query.lower() == "consolidate":
                print("🔄 执行记忆巩固...")
                result = await agent.consolidate()
                print(f"✅ 巩固完成: {result}")
                continue
            
            # 生成回答
            print("🤔 思考中...")
            result = await agent.respond(query, session_id=session_id)
            
            if not session_id:
                session_id = result.get("session_id")
            
            print(f"\n🤖 Agent: {result['answer']}")
            
            # 显示检索到的记忆
            if result['memories']:
                print(f"\n💭 相关记忆 ({len(result['memories'])}):")
                for i, mem in enumerate(result['memories'][:3], 1):
                    print(f"  {i}. [{mem['source']}] {mem['summary'][:80]}...")
        
        except KeyboardInterrupt:
            print("\n\n💾 保存记忆...")
            agent.save()
            print("👋 再见!")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


def create_simple_llm():
    """创建简单的 LLM 函数（示例）"""
    async def llm_func(prompt: str) -> str:
        # 这里应该调用实际的 LLM API
        # 示例：直接返回提示的前100字符
        return f"[LLM 模拟回答] 收到提示: {prompt[:100]}..."
    return llm_func


def create_simple_embedding():
    """创建简单的 embedding 函数（示例）"""
    def embedding_func(texts):
        # 这里应该调用实际的 embedding API
        # 示例：返回随机向量
        import random
        return [[random.random() for _ in range(128)] for _ in texts]
    return embedding_func


def main():
    parser = argparse.ArgumentParser(description="NeuroAgent - 类人脑智能体")
    parser.add_argument("--server", action="store_true", help="启动 API 服务")
    parser.add_argument("--chat", action="store_true", help="启动聊天模式")
    parser.add_argument("--host", default="0.0.0.0", help="API 服务主机")
    parser.add_argument("--port", type=int, default=8000, help="API 服务端口")
    parser.add_argument("--working-dir", default="./neuro_agent_data", help="工作目录")
    
    args = parser.parse_args()
    
    # 创建 Agent
    print("🚀 初始化 NeuroAgent...")
    
    llm_func = create_simple_llm()
    embedding_func = create_simple_embedding()
    
    agent = NeuroAgent(
        llm_func=llm_func,
        embedding_func=embedding_func,
        working_dir=args.working_dir,
    )
    
    # 注册感知器
    chat_perceiver = ChatPerceiver(cognitive_processor=agent.cognition)
    agent.register_perceiver("chat", chat_perceiver)
    
    print(f"✅ Agent 初始化完成")
    print(f"   工作目录: {args.working_dir}")
    print(f"   Episode 数量: {len(agent.memory.episode_store.all_episodes())}")
    
    if args.server:
        # 启动 API 服务
        try:
            import uvicorn
            app = create_app(agent)
            print(f"\n🌐 启动 API 服务: http://{args.host}:{args.port}")
            uvicorn.run(app, host=args.host, port=args.port)
        except ImportError:
            print("❌ 需要安装 uvicorn: pip install uvicorn")
    
    elif args.chat or not args.server:
        # 启动交互式聊天
        asyncio.run(interactive_chat(agent))


if __name__ == "__main__":
    main()
