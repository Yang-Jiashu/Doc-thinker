import { MAX_RESEARCH_SOURCES, isResearchSourceLabel } from './research-source-contract.js';

export const MAX_EVIDENCE_ITEMS = MAX_RESEARCH_SOURCES;
export const MAX_EVIDENCE_CONTENT = 800;
export const MAX_EVIDENCE_TITLE = 100;
const MAX_SCANNED_ITEMS = 20;

// Treat the server's content as a display excerpt, not as a verified citation.
// Bound work before processing text; never stringify arbitrary metadata objects.
function boundedText(value, limit) {
  if (typeof value !== 'string') return { text: '', truncated: false };
  const bounded = value.slice(0, limit * 2 + 2)
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u202a-\u202e\u2066-\u2069]/g, '').trim();
  let text = '', count = 0;
  for (const character of bounded) {
    if (count === limit) return { text: text + '…', truncated: true };
    text += character;
    count += 1;
  }
  const truncated = value.length > limit * 2 + 2;
  return { text: truncated && text ? text + '…' : text, truncated };
}

export function normalizeAnswerEvidence(sources) {
  if (!Array.isArray(sources)) return { items: [], limited: false };
  const items = [];
  let scanned = 0;
  while (scanned < Math.min(sources.length, MAX_SCANNED_ITEMS) && items.length < MAX_EVIDENCE_ITEMS) {
    const source = sources[scanned++];
    if (!source || typeof source !== 'object' || Array.isArray(source)) continue;
    if (source.label !== undefined && !isResearchSourceLabel(source.label)) continue;
    const content = boundedText(source.content, MAX_EVIDENCE_CONTENT);
    if (!content.text) continue;
    const title = boundedText(source.title, MAX_EVIDENCE_TITLE).text
      || boundedText(source.filename, MAX_EVIDENCE_TITLE).text
      || `来源片段 ${items.length + 1}`;
    const research = isResearchSourceLabel(source.label);
    items.push({ title, content: content.text, truncated: content.truncated || source.truncated === true,
      ...(research ? {label:source.label,page:Number.isInteger(source.page) && source.page > 0 ? source.page : null} : {}) });
  }
  return { items, limited: scanned < sources.length };
}
