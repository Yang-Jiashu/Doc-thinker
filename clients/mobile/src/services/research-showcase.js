// Hand-labelled teaching material. Nothing here retrieves documents or runs a model.
const freezeSource = source => Object.freeze(source);

export const sources = Object.freeze([
  {
    id: 'investment', title: '本地构造示例 · 投资的前提', kind: '示例前提',
    quote: '可再生能源投资只有在项目建成并稳定运行后，才可能增加国内能源供给。',
  },
  {
    id: 'supply', title: '本地构造示例 · 新增供给', kind: '示例观察',
    quote: '项目新增了国内能源供给，但这段材料没有说明新增供给是替代原有进口，还是满足新增需求。',
  },
  {
    id: 'bridge', title: '本地构造示例 · 关键连接', kind: '示例条件',
    quote: '只有新增国内供给实际替代进口能源时，进口依赖才会下降。',
  },
  {
    id: 'risk', title: '本地构造示例 · 风险的范围', kind: '示例条件',
    quote: '在进口来源与其他条件不变时，减少对该来源的依赖，可以降低其供应中断带来的暴露；这不代表电网、储能或设备供应风险同时下降。',
  },
  {
    id: 'case-a', title: '本地构造示例 · 虚构案例 A', kind: '虚构支持案例',
    quote: '地区 A 的新增风光与储能替代了部分进口燃料；在需求平稳时，燃料进口依赖下降。',
  },
  {
    id: 'case-b', title: '本地构造示例 · 虚构案例 B', kind: '虚构支持案例',
    quote: '地区 B 扩建水电并替代部分进口电力；在需求平稳时，电力进口依赖下降。',
  },
  {
    id: 'case-counterexample', title: '本地构造示例 · 虚构反例 C', kind: '虚构反例',
    quote: '地区 C 的新增光伏用于满足新增负荷，原有进口燃料没有减少；新增供给没有带来进口依赖下降。',
  },
  {
    id: 'analogy-match', title: '本地构造示例 · 跨领域对应', kind: '示例类比',
    quote: '能源侧设想多个独立供给来源，以分散单一来源中断的影响；科研侧可对应多个独立证据来源，以检查同一结论是否依赖单一证据。',
  },
  {
    id: 'analogy-mismatch', title: '本地构造示例 · 类比的失配', kind: '示例失配',
    quote: '能源来源可以互相补充供给；科研证据不能按数量相加来保证结论成立。多篇论文也可能共享同一数据或同一种偏差。',
  },
].map(freezeSource));

export const modes = Object.freeze([
  Object.freeze({ id: 'deduction', label: '检查条件', technicalLabel: '还缺哪一环', question: '增加可再生能源投资，就能让能源更安全吗？' }),
  Object.freeze({ id: 'induction', label: '比较案例', technicalLabel: '也看看反例', question: '几个地区的经历里，有什么值得继续验证的共同点？' }),
  Object.freeze({ id: 'analogy', label: '借鉴思路', technicalLabel: '保留差异与边界', question: '能源多来源的思路，能怎么启发我的研究？' }),
]);

export function getSource(id) {
  return sources.find(source => source.id === id) || null;
}

function deduction(includeBridge) {
  const steps = [
    { id: 'investment', title: '先看投资能否变成供给', detail: '项目需要建成并稳定运行，投资才可能带来新增国内供给。', status: 'supported', sourceIds: ['investment'] },
    { id: 'supply', title: '确认已经有新增供给', detail: '示例材料写了供给增加，但没有交代这些能源最终用在哪里。', status: 'supported', sourceIds: ['supply'] },
  ];
  if (!includeBridge) {
    steps.push({ id: 'missing-bridge', title: '这里缺了一跳', detail: '新增供给与进口变化之间缺少依据。目前只能停在“供给增加”，继续判断供应风险还不够。', status: 'gap', sourceIds: ['supply'] });
    return {
      heading: '供给增加了，进口变没变？',
      answer: '目前只能判断：项目满足前提时，可能增加国内供给。缺少它与进口变化之间的依据，暂不能推出进口依赖或供应风险下降。',
      badge: '本地构造示例 · 证据缺口',
      limit: '连接线只展示待检查的关系。当前可见片段不足以支持最后的结论，也不能证明因果。',
      nextQuestion: '新增供给去了哪里？先找能说明进口变化的材料。',
      steps, sourceIds: ['investment', 'supply'],
    };
  }
  steps.push(
    { id: 'bridge', title: '补上供给与进口之间的条件', detail: getSource('bridge').quote, status: 'conditional', sourceIds: ['bridge'] },
    { id: 'risk', title: '把结论限定在一种风险', detail: '若进口依赖确实下降，在其他条件不变时，可减少该进口来源中断带来的暴露。电网、储能和设备风险还要另查。', status: 'conditional', sourceIds: ['risk'] },
  );
  return {
    heading: '投资增加，安全未必增加。',
    answer: '如果项目稳定运行，新增供给确实替代进口，并且其他条件不变，就可能降低对该进口来源中断的暴露。整体能源安全是否改善，还要检查电网、储能和设备供应。',
    badge: '本地构造示例 · 有条件的结论',
    limit: '这条链说明需要哪些前提。图上的连接和示例引用都不能证明现实因果，也没有证明这些条件已经满足。',
    nextQuestion: '怎样验证新增供给真的替代了进口，而不是只满足了新增需求？',
    steps, sourceIds: ['investment', 'supply', 'bridge', 'risk'],
  };
}

function induction() {
  return {
    heading: '增加供给之后，还要看它去了哪。',
    answer: '两个支持案例都有“新增供给实际替代进口、需求平稳”这两个条件。反例中，新增供给只满足了新增需求。可以先把“是否替代进口”当作值得验证的检查项。',
    badge: '本地构造示例 · 候选规律',
    limit: '这里只比较两个支持案例和一个反例，全部是虚构案例。不能估算概率或推成普遍规律，需要真实样本继续验证。',
    nextQuestion: '有没有需求平稳、确实替代进口，但供应风险仍未下降的真实案例？',
    steps: [
      { id: 'case-a', title: '支持案例 A：风光与储能', detail: '替代了部分进口燃料，需求平稳，燃料进口依赖下降。', status: 'supported', sourceIds: ['case-a'] },
      { id: 'case-b', title: '支持案例 B：水电', detail: '替代了部分进口电力，需求平稳，电力进口依赖下降。', status: 'supported', sourceIds: ['case-b'] },
      { id: 'case-counterexample', title: '反例 C：新增光伏', detail: '供给满足新增负荷，原有进口没有减少；“新增供给就降低进口依赖”在这个虚构案例里不成立。', status: 'counterexample', sourceIds: ['case-counterexample'] },
      { id: 'pattern', title: '留下一个待验证的共同点', detail: '后续查资料时，同时记录实际替代、需求变化及不符合预期的案例。', status: 'pattern', sourceIds: ['case-a', 'case-b', 'case-counterexample'] },
    ],
    sourceIds: ['case-a', 'case-b', 'case-counterexample'],
  };
}

function analogy() {
  return {
    heading: '多篇论文，也可能只有一份证据。',
    answer: '一个待验证的假设：像检查能源是否过度依赖单一来源一样，检查研究结论是否过度依赖同一份数据。试着寻找来源独立、方法不同的证据，再比较它们是否支持同一结论。',
    badge: '本地构造示例 · 待验证假设',
    limit: '类比只提供检查方向，不能证明科研结论。证据有共同偏差时，增加数量也不等于增加独立支持。',
    nextQuestion: '这些论文真的独立吗？先比较它们的数据来源、方法和共同假设。',
    steps: [
      { id: 'energy-side', title: '原来的场景：能源多来源', detail: '关注单一供给来源中断时，其他来源是否还能提供供给。', status: 'hypothesis', sourceIds: ['analogy-match'] },
      { id: 'research-side', title: '借到研究里：多独立证据', detail: '把“供给来源”对应为“证据来源”，把“单点依赖”对应为“只依赖同一数据或方法”。这是待验证的映射。', status: 'hypothesis', sourceIds: ['analogy-match'] },
      { id: 'mismatch', title: '对不上的地方也要看', detail: '能源可以补足供给，论文数量却不能补成真理；多篇论文可能共享数据与偏差。', status: 'mismatch', sourceIds: ['analogy-mismatch'] },
      { id: 'check', title: '变成一个能执行的检查', detail: '给现有论文标注数据与方法，找一份真正独立的证据，同时保留相反结果。类比假设是否有用，还要看这一步的实际结果。', status: 'hypothesis', sourceIds: ['analogy-match', 'analogy-mismatch'] },
    ],
    sourceIds: ['analogy-match', 'analogy-mismatch'],
  };
}

// Unknown modes fail closed. includeBridge affects deduction only.
export function buildShowcase(modeId, { includeBridge = true } = {}) {
  if (modeId === 'deduction') return deduction(includeBridge === true);
  if (modeId === 'induction') return induction();
  if (modeId === 'analogy') return analogy();
  throw new RangeError('Unknown research showcase mode');
}
