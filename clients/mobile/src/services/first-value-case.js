// Curated onboarding material, verified against the version-pinned arXiv abstracts.
// This is not generated live, an evaluation result, or the user's own project memory.
// Keep original-language excerpts distinct from our Chinese explanations.
function freezeTree(value) {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(freezeTree);
    Object.freeze(value);
  }
  return value;
}

export const firstValueCase = freezeTree({
  id: 'more-context-is-not-enough',
  version: 1,
  title: '把更多资料交给 AI，就一定答得更好吗？',
  question: '把更多资料交给 AI，就一定答得更好吗？',
  answer: '不一定。相关证据可能帮助回答，但模型是否用到关键证据也很重要。不能把资料数量直接当作答案质量。',
  boundary: '人工整理的真实论文样例，非实时模型生成，也不是本产品效果评测。两篇论文的模型、任务和年代不同，不能据此断言所有模型都一样，或 token 越多效果越差；也没有验证本产品的长期记忆能力。',
  nextQuestion: '在我的资料中，回答这个问题真正需要哪几段？关键证据有没有被漏掉或淹没？',
  sources: [
    {
      id: 'rag-2020',
      title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
      authors: 'Patrick Lewis 等',
      year: 2020,
      url: 'https://arxiv.org/abs/2005.11401v4',
      locator: '摘要（Abstract），开放域问答结果句',
      quote: 'set the state-of-the-art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures.',
      explanation: '作者在三项开放域问答任务上报告了 RAG 的优势。这支持“检索得到的外部证据可以有帮助”，不是“给任何模型塞更多资料都会更好”。',
    },
    {
      id: 'lost-middle-2023',
      title: 'Lost in the Middle: How Language Models Use Long Contexts',
      authors: 'Nelson F. Liu 等',
      year: 2023,
      url: 'https://arxiv.org/abs/2307.03172v3',
      locator: '摘要（Abstract），相关信息位置影响结果句',
      quote: 'performance can degrade significantly when changing the position of relevant information',
      explanation: '作者研究多文档问答与键值检索，发现相关信息的位置能显著影响被测模型表现；长上下文中间的信息可能更难被利用。这不是对所有新模型的定论。',
    },
  ],
  steps: [
    {
      id: 'retrieved-evidence',
      title: '找到证据，确实可能有帮助',
      detail: 'RAG 论文在其评估的开放域问答任务中取得更好的结果。注意：它评估的是具体检索与生成系统，不是资料数量本身。',
      status: 'supported',
      sourceIds: ['rag-2020'],
    },
    {
      id: 'context-position',
      title: '放进去了，不等于用到了',
      detail: '另一篇论文发现，关键内容放在哪里会影响被测模型的表现。内容已在输入里，也可能没有被有效利用。',
      status: 'supported',
      sourceIds: ['lost-middle-2023'],
    },
    {
      id: 'bounded-conclusion',
      title: '要检查关键证据，而不只数资料',
      detail: '整理者的综合判断：先核对相关证据是否被找全、是否被模型利用，再评估实际回答。两篇论文不能直接证明本产品更好，也没有证明减少 token 就一定更准确。',
      status: 'synthesis',
      sourceIds: ['rag-2020', 'lost-middle-2023'],
    },
  ],
});
