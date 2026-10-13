// Local-only delivery of the server's literal Markdown, never a new summary.
export const MAX_RESEARCH_EXPORT_BYTES = 800000;
export const EXPORT_COPY_PART_UNITS = 40000;

export function validResearchExport(value) {
  if (typeof value !== 'string' || !value.trim() || value.length > MAX_RESEARCH_EXPORT_BYTES) return false;
  let bytes = 0;
  for (const character of value) {
    const point = character.codePointAt(0);
    if (point >= 0xd800 && point <= 0xdfff) return false;
    bytes += point <= 0x7f ? 1 : point <= 0x7ff ? 2 : point <= 0xffff ? 3 : 4;
    if (bytes > MAX_RESEARCH_EXPORT_BYTES) return false;
  }
  return true;
}

export function researchExportParts(value) {
  if (value === '') return [];
  if (!validResearchExport(value)) throw new Error('导出内容格式或大小不符合要求，没有截断内容。');
  const parts = [];
  for (let start = 0; start < value.length;) {
    let end = Math.min(value.length, start + EXPORT_COPY_PART_UNITS);
    const previous = value.charCodeAt(end - 1);
    if (end < value.length && previous >= 0xd800 && previous <= 0xdbff) end -= 1;
    parts.push(value.slice(start, end));
    start = end;
  }
  return parts;
}

export function downloadResearchMarkdown(value, sessionId, {
  document = globalThis.document, URL = globalThis.URL, Blob = globalThis.Blob,
  defer = globalThis.setTimeout, isCurrent = () => true,
} = {}) {
  if (!validResearchExport(value) || !/^#[0-9]{5}$/.test(sessionId)) {
    throw new Error('导出内容或会话不符合要求，没有下载。');
  }
  if (!document?.createElement || !document?.body?.appendChild || !URL?.createObjectURL
      || !URL?.revokeObjectURL || typeof Blob !== 'function' || typeof defer !== 'function') {
    throw new Error('当前环境不支持本地下载，请使用下方分段查看与复制，内容已保留。');
  }
  let link, url;
  const revoke = () => { try { if (url) URL.revokeObjectURL(url); } catch { /* No network fallback. */ } };
  try {
    if (!isCurrent()) throw new Error();
    link = document.createElement('a');
    if (typeof link.click !== 'function' || !('download' in link)) throw new Error();
    const filename = `DocThinker-research-${sessionId.slice(1)}.md`;
    url = URL.createObjectURL(new Blob([value], {type:'text/markdown;charset=utf-8'}));
    link.href = url; link.download = filename; link.hidden = true;
    document.body.appendChild(link);
    if (!isCurrent()) throw new Error();
    link.click();
    // Revoking immediately can race the browser's local download hand-off.
    defer(revoke, 15000);
    return {requested:true, filename};
  } catch {
    revoke();
    throw new Error('没有完成下载请求，请使用下方分段查看与复制，内容已保留。');
  } finally {
    try { link?.remove(); } catch { /* Temporary anchor only; no remote upload. */ }
  }
}
