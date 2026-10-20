// A deliberately small, presentation-only Markdown subset. No HTML, URL handling,
// source verification, network calls, or recursive parsing belongs in this layer.
export const MAX_ANSWER_FORMAT_CHARS = 80000;
export const MAX_ANSWER_BLOCKS = 240;
export const MAX_ANSWER_INLINE_TOKENS = 1400;
export const MAX_ANSWER_TABLE_ROWS = 40;
export const MAX_ANSWER_TABLE_COLUMNS = 12;

// Bound incomplete link look-ahead too: a string of unmatched '[' must not
// trigger a scan of the entire remaining answer at every character.
const inlinePattern = /!?\[[^\]\r\n]{0,256}\]\([^\r\n)]{0,2048}\)|https?:\/\/[^\s]+|`[^`\r\n]+`|\*\*[^*\r\n]+\*\*|__[^_\r\n]+__|\*(?!\*)[^*\r\n]+\*(?!\*)|_(?!_)[^_\r\n]+_(?!_)|\[S[1-9]\d{0,3}(?:\s*[,，]\s*S[1-9]\d{0,3})*\]/g;

function inline(text, budget) {
  const tokens = [];
  const append = (kind, value) => {
    if (!value) return;
    if (kind === 'text' && tokens[tokens.length - 1]?.kind === 'text') tokens[tokens.length - 1].text += value;
    else { tokens.push({ kind, text: value }); budget.count += 1; }
    if (budget.count > MAX_ANSWER_INLINE_TOKENS) throw new RangeError('format-complexity');
  };
  // A fresh regex avoids state leaking between cells/paragraphs or concurrent renders.
  const pattern = new RegExp(inlinePattern.source, 'g');
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    append('text', text.slice(cursor, match.index));
    const value = match[0];
    let kind = 'text', content = value;
    if (value.startsWith('`')) { kind = 'code'; content = value.slice(1, -1); }
    else if (value.startsWith('**') || value.startsWith('__')) {
      // Underscores in identifiers are data, not formatting.
      if (!value.startsWith('__') || !/[\p{L}\p{N}_]/u.test(text[match.index - 1] || '')) {
        kind = 'strong'; content = value.slice(2, -2);
      }
    } else if (value.startsWith('*') || value.startsWith('_')) {
      const before = text[match.index - 1] || '', after = text[match.index + value.length] || '';
      if (value.startsWith('*') || (!/[\p{L}\p{N}_]/u.test(before) && !/[\p{L}\p{N}_]/u.test(after))) {
        kind = 'em'; content = value.slice(1, -1);
      }
    } else if (/^\[S[1-9]/.test(value) && !value.includes('](')) kind = 'citation';
    append(kind, content);
    cursor = match.index + value.length;
  }
  append('text', text.slice(cursor));
  return tokens;
}

function linesOf(text) {
  const lines = [];
  let start = 0;
  while (start < text.length) {
    const end = text.indexOf('\n', start);
    const raw = text.slice(start, end < 0 ? text.length : end + 1);
    lines.push({ raw, text: raw.replace(/\r?\n$/, '') });
    start = end < 0 ? text.length : end + 1;
  }
  return lines;
}

function tableCells(line) {
  let text = line.trim();
  if (!text.includes('|')) return null;
  if (text.startsWith('|')) text = text.slice(1);
  if (text.endsWith('|') && !text.endsWith('\\|')) text = text.slice(0, -1);
  const cells = [];
  let current = '', code = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '\\' && text[i + 1] === '|') { current += '|'; i += 1; }
    else if (char === '`') { code = !code; current += char; }
    else if (char === '|' && !code) { cells.push(current.trim()); current = ''; }
    else current += char;
  }
  cells.push(current.trim());
  return cells;
}

function tableAt(lines, index) {
  if (!lines[index + 1]) return null;
  const header = tableCells(lines[index].text), separator = tableCells(lines[index + 1].text);
  if (!header || header.length < 2 || header.length > MAX_ANSWER_TABLE_COLUMNS || !separator
    || separator.length !== header.length || !separator.every(cell => /^:?-{3,}:?$/.test(cell))) return null;
  const rows = [];
  let end = index + 2;
  while (end < lines.length && lines[end].text.trim() && lines[end].text.includes('|')) {
    const row = tableCells(lines[end].text);
    // Ragged/large tables stay literal rather than dropping cells or rows.
    if (!row || row.length !== header.length || rows.length >= MAX_ANSWER_TABLE_ROWS) return null;
    rows.push(row); end += 1;
  }
  return { header, rows, end };
}

const fence = line => line.match(/^ {0,3}(`{3,}|~{3,})([^`~]*)$/);
const heading = line => line.match(/^ {0,3}(#{1,6})[ \t]+(.+)$/);
const quote = line => line.match(/^ {0,3}>[ \t]?(.*)$/);
const list = line => line.match(/^( {0,8})([-+*]|\d{1,9}[.)])[ \t]+(.+)$/);
const isStart = (lines, i) => fence(lines[i].text) || heading(lines[i].text)
  || quote(lines[i].text) || list(lines[i].text) || tableAt(lines, i);

/** Render data only. `raw` always contains the entire original answer for copying.
 * Oversize/over-complex input uses one literal text block, never truncation.
 */
export function parseAnswerBody(value) {
  const raw = typeof value === 'string' ? value : '';
  const literal = reason => ({ raw, plain: true, reason, blocks: raw ? [{ type: 'plain', text: raw }] : [] });
  if (!raw) return { raw, plain: false, reason: '', blocks: [] };
  if (raw.length > MAX_ANSWER_FORMAT_CHARS) return literal('length');
  const lines = linesOf(raw), blocks = [], budget = { count: 0 };
  const add = block => {
    blocks.push(block);
    if (blocks.length > MAX_ANSWER_BLOCKS) throw new RangeError('format-complexity');
  };
  try {
    let i = 0;
    while (i < lines.length) {
      const line = lines[i].text;
      if (!line.trim()) { i += 1; continue; }
      const opening = fence(line);
      if (opening) {
        const marker = opening[1][0], size = opening[1].length;
        let end = i + 1;
        while (end < lines.length) {
          const closing = lines[end].text.match(/^ {0,3}(`{3,}|~{3,})[ \t]*$/);
          if (closing && closing[1][0] === marker && closing[1].length >= size) break;
          end += 1;
        }
        add({ type: 'code', language: opening[2].trim(), text: lines.slice(i + 1, end).map(item => item.raw).join('') });
        i = end < lines.length ? end + 1 : end; continue;
      }
      const table = tableAt(lines, i);
      if (table) {
        add({ type: 'table', header: table.header.map(cell => inline(cell, budget)), rows: table.rows.map(row => row.map(cell => inline(cell, budget))) });
        i = table.end; continue;
      }
      const title = heading(line);
      if (title) { add({ type: 'heading', level: title[1].length, tokens: inline(title[2], budget) }); i += 1; continue; }
      if (quote(line)) {
        const content = [];
        while (i < lines.length && quote(lines[i].text)) { content.push(quote(lines[i].text)[1]); i += 1; }
        add({ type: 'quote', tokens: inline(content.join('\n'), budget) }); continue;
      }
      const item = list(line);
      if (item) {
        const ordered = /^\d/.test(item[2]), items = [];
        while (i < lines.length) {
          const next = list(lines[i].text);
          if (!next || /^\d/.test(next[2]) !== ordered) break;
          items.push({ marker: ordered ? next[2] : '•', indent: next[1].length, tokens: inline(next[3], budget) });
          i += 1;
        }
        add({ type: 'list', ordered, items }); continue;
      }
      const content = [line]; i += 1;
      while (i < lines.length && lines[i].text.trim() && !isStart(lines, i)) { content.push(lines[i].text); i += 1; }
      add({ type: 'paragraph', tokens: inline(content.join('\n'), budget) });
    }
    // Whitespace-only answers are still available exactly as returned.
    return blocks.length ? { raw, plain: false, reason: '', blocks } : literal('');
  } catch (error) {
    if (!(error instanceof RangeError) || error.message !== 'format-complexity') throw error;
    return literal('complexity');
  }
}
