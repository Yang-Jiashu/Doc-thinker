// A visible, editable request recipe, not a model-specific reasoning mode.
// Original user text remains intact; the normal composer enforces send budgets.
const prefix = '请基于当前会话资料核对以下问题或判断：\n\n';
const suffix = '\n\n请先给出简短结论，再分别说明：\n1. 支持该判断的原文依据与出处。\n2. 必要条件、限制或相反证据。\n3. 现有资料还不能支持的部分。\n没有检索到不等于不存在。区分原文与推断，不补造来源，不强行补全因果链；涉及多个判断时分别核对。';
export function buildEvidenceCheckQuestion(text) {
  if (typeof text !== 'string' || !text.trim()) return '';
  if (text.startsWith(prefix) && text.endsWith(suffix)) return text;
  return prefix + text + suffix;
}
