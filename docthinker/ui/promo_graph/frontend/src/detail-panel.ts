import { splitSourceIds } from "./graph-data";
import type { ChunkEvidence, GraphNode } from "./types";

function setText(element: HTMLElement | null, value: string): void {
  if (element) element.textContent = value;
}

export class DetailPanel {
  private root: HTMLElement;
  private title: HTMLElement;
  private meta: HTMLElement;
  private description: HTMLElement;
  private sourceIds: HTMLElement;
  private chunks: HTMLElement;
  private chunkStatus: HTMLElement;

  constructor(root: HTMLElement) {
    this.root = root;
    this.title = root.querySelector("[data-detail-title]") as HTMLElement;
    this.meta = root.querySelector("[data-detail-meta]") as HTMLElement;
    this.description = root.querySelector("[data-detail-description]") as HTMLElement;
    this.sourceIds = root.querySelector("[data-source-ids]") as HTMLElement;
    this.chunks = root.querySelector("[data-chunks]") as HTMLElement;
    this.chunkStatus = root.querySelector("[data-chunk-status]") as HTMLElement;
  }

  show(node: GraphNode): void {
    this.root.hidden = false;
    setText(this.title, node.label);
    setText(this.meta, `${node.type || "unknown"} · ${node.degree} 条连接`);
    setText(this.description, node.description);
    this.sourceIds.replaceChildren();
    const ids = splitSourceIds(node.sourceId);
    if (!ids.length) {
      const empty = document.createElement("span");
      empty.className = "detail-empty";
      empty.textContent = "暂无来源 chunk";
      this.sourceIds.append(empty);
    } else {
      ids.forEach(id => {
        const code = document.createElement("code");
        code.textContent = id;
        this.sourceIds.append(code);
      });
    }
    this.chunks.replaceChildren();
    this.chunkStatus.textContent = "正在读取原文证据…";
  }

  showChunks(chunks: ChunkEvidence[], error = ""): void {
    this.chunks.replaceChildren();
    if (error) {
      this.chunkStatus.textContent = error;
      return;
    }
    this.chunkStatus.textContent = chunks.length ? `${chunks.length} 个原文 chunk` : "未找到原文 chunk";
    chunks.forEach(chunk => {
      const article = document.createElement("article");
      article.className = "chunk-card";
      article.style.contentVisibility = "auto";
      article.style.containIntrinsicSize = "220px";
      const header = document.createElement("header");
      const id = document.createElement("code");
      id.textContent = chunk.chunk_id;
      const order = document.createElement("span");
      order.textContent = chunk.missing ? "missing" : chunk.chunk_order_index === undefined ? "source" : `order ${chunk.chunk_order_index}`;
      header.append(id, order);
      const content = document.createElement("p");
      content.textContent = chunk.content || "该 chunk 没有可读取的文本内容。";
      article.append(header, content);
      this.chunks.append(article);
    });
  }

  hide(): void {
    this.root.hidden = true;
    this.chunks.replaceChildren();
  }
}
