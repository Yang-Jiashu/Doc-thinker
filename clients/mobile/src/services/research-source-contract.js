// Final retained packet: up to eight baseline excerpts plus two Insight additions.
// This is not a retrieval expansion limit or a claim that citations are verified.
export const MAX_RESEARCH_SOURCES = 10;
export const isResearchSourceLabel = value => typeof value === 'string' && /^S(?:[1-9]|10)$/.test(value);

export function boundedAnswerSources(value, research = false) {
  if (!Array.isArray(value)) return [];
  const sources = value.slice(0,research ? MAX_RESEARCH_SOURCES : 5);
  // An out-of-protocol research label must not become an unlabeled legacy quote.
  return research ? sources.filter(source=>isResearchSourceLabel(source?.label)) : sources;
}
