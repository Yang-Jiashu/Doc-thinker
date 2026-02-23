#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 neuro_memory（类人脑联想与记忆重连）是否好用。

用法:
  # 1. 仅用内存 mock embedding，不调 API（快速自检）
  python scripts/verify_neuro_memory.py

  # 2. 用真实 embedding（需配置 OPENAI_API_KEY 或项目 .env）
  python scripts/verify_neuro_memory.py --embed

  # 3. 用真实 embedding + LLM 做巩固（会调 API，较慢）
  python scripts/verify_neuro_memory.py --embed --llm
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# 保证项目根在 path 中
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mock_embedding(texts):
    """简单 mock：按字符 hash 成固定维度向量，相同文本相似度高。"""
    import hashlib
    dim = 64
    out = []
    for t in texts:
        h = hashlib.sha256((t or "").encode()).hexdigest()
        vec = [((int(h[i : i + 2], 16) / 255.0) - 0.5) * 2 for i in range(0, min(dim * 2, len(h) - 1), 2)]
        if len(vec) < dim:
            vec.extend([0.0] * (dim - len(vec)))
        out.append(vec[:dim])
    return out


async def run_verify(use_real_embed: bool, use_llm: bool):
    from neuro_memory import MemoryEngine, Episode

    # 1. 初始化引擎
    if use_real_embed:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            import numpy as np
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY"))

            async def real_embed(texts):
                if isinstance(texts, str):
                    texts = [texts]
                resp = await client.embeddings.create(model="text-embedding-3-small", input=texts)
                arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
                return arr.tolist()

            embedding_func = real_embed
            print("[OK] 使用真实 OpenAI embedding")
        except Exception as e:
            print(f"[WARN] 真实 embedding 不可用 ({e})，退回 mock")
            embedding_func = lambda texts: _mock_embedding(texts if isinstance(texts, list) else [texts])
    else:
        embedding_func = lambda texts: _mock_embedding(texts if isinstance(texts, list) else [texts])
        print("[OK] 使用 mock embedding（同内容会相似）")

    llm_func = None
    if use_llm:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY"))

            async def llm(prompt):
                r = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                return r.choices[0].message.content if r.choices else ""

            llm_func = llm
            print("[OK] 使用真实 LLM 做巩固推断")
        except Exception as e:
            print(f"[WARN] LLM 不可用 ({e})，巩固将不跑跨事件推断")

    work_dir = ROOT / "neuro_memory_verify_data"
    engine = MemoryEngine(
        embedding_func=embedding_func,
        llm_func=llm_func,
        working_dir=str(work_dir),
    )

    # 2. 写入几段「经历」——内容有重叠、有类比
    print("\n--- 写入经历 ---")
    episodes_text = [
        ("公司 A 收购了公司 B，整合了其技术团队。", ["收购", "整合", "技术团队"]),
        ("公司 C 并购了公司 D，保留了核心研发人员。", ["并购", "保留", "研发"]),
        ("今天开会讨论了 Q2 的销售目标，要求增长 20%。", ["会议", "销售目标", "增长"]),
        ("昨日会议确定了下一季度营收目标，同比提升 20%。", ["会议", "营收", "同比"]),
    ]
    for summary, concepts in episodes_text:
        ep = await engine.add_observation(
            summary=summary,
            concepts=concepts,
            source_type="doc",
        )
        print(f"  + {summary[:50]}... -> episode_id={ep.episode_id[:20]}...")

    # 3. 查图与向量
    all_ep = engine.episode_store.all_episodes()
    edges = engine.graph.get_all_edges()
    print(f"\n--- 当前状态: {len(all_ep)} episodes, {len(edges)} 条边 ---")

    # 4. 类比检索：问一个与「收购/会议」都沾边的问题
    print("\n--- 类比检索（query: 公司并购后如何保留人才？）---")
    analogies = await engine.retrieve_analogies(
        "公司并购后如何保留人才？",
        top_k=5,
        then_spread=True,
    )
    for i, (ep, score, hint) in enumerate(analogies, 1):
        print(f"  {i}. [score={score:.3f}] {ep.summary[:60]}...")
        if hint:
            print(f"      hint: {hint}")

    # 5. 巩固（会加强 episode 之间的边）
    print("\n--- 执行一次巩固 ---")
    try:
        result = await engine.consolidate(recent_n=10, run_llm=bool(llm_func))
        print(f"  巩固结果: edges_added={result.get('edges_added', 0)}, pairs_processed={result.get('pairs_processed', 0)}")
    except Exception as e:
        print(f"  巩固异常（可忽略）: {e}")
    edges_after = len(engine.graph.get_all_edges())
    print(f"  巩固后边数: {len(edges)} -> {edges_after}")

    # 6. 再检索一次，看是否仍能命中
    print("\n--- 再次类比检索（query: 季度目标会议）---")
    analogies2 = await engine.retrieve_analogies("季度目标会议", top_k=3, then_spread=False)
    for i, (ep, score, _) in enumerate(analogies2, 1):
        print(f"  {i}. [score={score:.3f}] {ep.summary[:60]}...")

    engine.save()
    print("\n--- 验证完成：数据已保存到", work_dir, "---")
    return True


def main():
    parser = argparse.ArgumentParser(description="验证 neuro_memory 是否好用")
    parser.add_argument("--embed", action="store_true", help="使用真实 embedding API")
    parser.add_argument("--llm", action="store_true", help="巩固时使用 LLM 做跨事件推断")
    args = parser.parse_args()
    asyncio.run(run_verify(use_real_embed=args.embed, use_llm=args.llm))


if __name__ == "__main__":
    main()
