import { splitSourceIds } from "./graph-data";
import type { ChunkEvidence, GraphEdge, GraphNode } from "./types";

function setText(element: HTMLElement | null, value: string): void {
  if (element) element.textContent = value;
}

export class DetailPanel {
  private root: HTMLElement;
  private title: HTMLElement;
  private kind: HTMLElement;
  private meta: HTMLElement;
  private descriptionHeading: HTMLElement;
  private description: HTMLElement;
  private evidenceSection: HTMLElement;
  private evidenceStatus: HTMLElement;
  private evidenceList: HTMLElement;
  private sourceIds: HTMLElement;
  private chunks: HTMLElement;
  private chunkStatus: HTMLElement;

  constructor(root: HTMLElement) {
    this.root = root;
    this.title = root.querySelector("[data-detail-title]") as HTMLElement;
    this.kind = root.querySelector("[data-detail-kind]") as HTMLElement;
    this.meta = root.querySelector("[data-detail-meta]") as HTMLElement;
    this.descriptionHeading = root.querySelector("[data-description-heading]") as HTMLElement;
    this.description = root.querySelector("[data-detail-description]") as HTMLElement;
    this.evidenceSection = root.querySelector("[data-relation-evidence]") as HTMLElement;
    this.evidenceStatus = root.querySelector("[data-evidence-status]") as HTMLElement;
    this.evidenceList = root.querySelector("[data-evidence-list]") as HTMLElement;
    this.sourceIds = root.querySelector("[data-source-ids]") as HTMLElement;
    this.chunks = root.querySelector("[data-chunks]") as HTMLElement;
    this.chunkStatus = root.querySelector("[data-chunk-status]") as HTMLElement;
  }

  show(node: GraphNode): void {
    this.root.hidden = false;
    this.root.dataset.detailType = "node";
    setText(this.kind, "FOCUSED NODE");
    setText(this.title, node.label);
    setText(this.meta, `${node.type || "unknown"} · ${node.degree} 条连接`);
    setText(this.descriptionHeading, "节点描述");
    setText(this.description, node.description);
    this.evidenceSection.hidden = true;
    this.evidenceList.replaceChildren();
    this.showSourceIds(splitSourceIds(node.sourceId));
    this.resetChunks();
  }

  showEdge(edge: GraphEdge, source: GraphNode, target: GraphNode): void {
    this.root.hidden = false;
    this.root.dataset.detailType = "edge";
    setText(this.kind, edge.kind === "eclrr_v4" ? "ECLRR-V4 EDGE" : "FACT EDGE");
    setText(this.title, edge.label || "related");
    const edgeType = edge.kind === "eclrr_v4" ? "ECLRR-v4 推断关系 · 虚线" : "入库事实关系 · 实线";
    const family = edge.relationFamily ? ` · ${edge.relationFamily}` : "";
    const direction = edge.direction ? ` · ${edge.direction}` : "";
    setText(this.meta, `${source.label} → ${target.label} · ${edgeType}${family}${direction}`);
    setText(this.descriptionHeading, "关系描述");
    setText(this.description, edge.description);
    this.showSourceIds(edge.evidenceChunkIds.length ? edge.evidenceChunkIds : splitSourceIds(edge.sourceId));
    this.showEvidence(edge);
    this.resetChunks();
  }

  private showSourceIds(ids: string[]): void {
    this.sourceIds.replaceChildren();
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
  }

  private showEvidence(edge: GraphEdge): void {
    this.evidenceList.replaceChildren();
    const hasEvidence = edge.evidenceChain.length > 0 || edge.pathUsed.length > 0;
    this.evidenceSection.hidden = !hasEvidence;
    if (!hasEvidence) return;

    const score = edge.judgeScores.total ?? edge.decisionScore;
    this.evidenceStatus.textContent = score === undefined || score === null ? "" : `Judge ${score}/10`;
    if (edge.pathUsed.length) {
      const path = document.createElement("p");
      path.className = "evidence-path";
      path.textContent = edge.pathUsed.join(" → ");
      this.evidenceList.append(path);
    }
    edge.evidenceChain.forEach((item, index) => {
      const article = document.createElement("article");
      article.className = "evidence-item";
      const heading = document.createElement("strong");
      const endpoints = [item.source, item.target].filter(Boolean).join(" → ");
      heading.textContent = `${index + 1}. ${endpoints || item.relation || "证据"}`;
      const chunk = document.createElement("code");
      chunk.textContent = item.chunk_id || "未标注 chunk";
      const quote = document.createElement("p");
      quote.textContent = item.quote || "未提供引用文本";
      article.append(heading, chunk, quote);
      this.evidenceList.append(article);
    });
  }

  private resetChunks(): void {
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
    delete this.root.dataset.detailType;
    this.evidenceList.replaceChildren();
    this.chunks.replaceChildren();
  }
}
