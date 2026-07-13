import {
  ArrowLeft,
  Hand,
  Maximize2,
  PanelRight,
  RotateCw,
  Search,
  Sparkles,
  TriangleAlert,
  X,
  ZoomIn,
  ZoomOut,
  createIcons,
} from "lucide";
import "./styles.css";
import { AutoRotationController } from "./auto-rotation-controller";
import { DetailPanel } from "./detail-panel";
import { normalizeGraph } from "./graph-data";
import { GestureController } from "./gesture/gesture-controller";
import { InputController } from "./input-controller";
import { LayoutController } from "./layout-controller";
import { StarMapRenderer } from "./star-map-renderer";
import type { EntityChunkResponse, GraphModel, RawGraphResponse, SessionRecord } from "./types";

declare global {
  interface Window {
    __PROMO_GRAPH_DEBUG__?: {
      stats: () => ReturnType<StarMapRenderer["getStats"]> & { nodes: number; edges: number };
      zoom: (ratio: number) => void;
      select: (nodeIndex: number) => void;
      selectEdge: (edgeIndex: number) => void;
      fit: () => void;
      autoRotate: (enabled: boolean) => void;
      rotation: () => {
        decision: ReturnType<AutoRotationController["evaluate"]>;
        state: ReturnType<PromoGraphApp["rotationDebugState"]>;
      };
      gesture: () => ReturnType<GestureController["getState"]>;
      attract: (nodeIndex: number) => void;
      releaseAttraction: () => void;
      position: (nodeIndex: number) => ReturnType<StarMapRenderer["getNodeWorldPosition"]>;
      edgePosition: (edgeIndex: number) => ReturnType<StarMapRenderer["getEdgeScreenMidpoint"]>;
    };
  }
}

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required element: ${selector}`);
  return element;
}

function sessionId(session: SessionRecord): string {
  return String(session.id ?? session.session_id ?? "");
}

function sessionLabel(session: SessionRecord): string {
  const id = sessionId(session);
  const title = String(session.title ?? session.name ?? "未命名会话");
  return id ? `${id} | ${title}` : title;
}

const AUTO_ROTATION_STORAGE_KEY = "docthinker.promo.autoRotate";
const SHARED_SESSION_STORAGE_KEY = "current_session_id";
const LEGACY_PROMO_SESSION_STORAGE_KEY = "docthinker.promo.session";

interface EclrrRunStatus {
  session_id: string;
  status: "idle" | "running" | "completed" | "failed";
  max_new_edges: number;
  started_at?: string;
  finished_at?: string;
  reviewed: number;
  proposed: number;
  accepted: number;
  rejected: number;
  error?: string;
}

function storedSessionPreference(): string {
  try {
    return localStorage.getItem(SHARED_SESSION_STORAGE_KEY)
      || localStorage.getItem(LEGACY_PROMO_SESSION_STORAGE_KEY)
      || "";
  } catch {
    return "";
  }
}

function storeSessionPreference(sessionIdValue: string): void {
  try {
    localStorage.setItem(SHARED_SESSION_STORAGE_KEY, sessionIdValue);
    localStorage.setItem(LEGACY_PROMO_SESSION_STORAGE_KEY, sessionIdValue);
  } catch {
    // Storage can be unavailable in privacy-restricted browser contexts.
  }
}

function storedAutoRotationPreference(): boolean | undefined {
  try {
    const stored = localStorage.getItem(AUTO_ROTATION_STORAGE_KEY);
    if (stored === "true") return true;
    if (stored === "false") return false;
  } catch {
    return undefined;
  }
  return undefined;
}

class PromoGraphApp {
  private root = required<HTMLElement>("#promo-app");
  private stage = required<HTMLElement>("#star-map-stage");
  private sessionSelect = required<HTMLSelectElement>("#session-select");
  private searchInput = required<HTMLInputElement>("#node-search");
  private searchResults = required<HTMLElement>("#search-results");
  private panelRoot = required<HTMLElement>("#node-panel");
  private panelToggle = required<HTMLButtonElement>("#toggle-panel");
  private rotationButton = required<HTMLButtonElement>("#toggle-auto-rotation");
  private eclrrButton = required<HTMLButtonElement>("#run-eclrr");
  private gestureButton = required<HTMLButtonElement>("#toggle-gesture");
  private gestureLayer = required<HTMLElement>("#gesture-layer");
  private gestureVideo = required<HTMLVideoElement>("#gesture-video");
  private gestureCursor = required<HTMLElement>("#gesture-cursor");
  private loading = required<HTMLElement>("#loading-state");
  private loadingDetail = required<HTMLElement>("#loading-detail");
  private errorState = required<HTMLElement>("#error-state");
  private errorMessage = required<HTMLElement>("#error-message");
  private status = required<HTMLElement>("#graph-status");
  private counts = required<HTMLElement>("#graph-counts");
  private zoomStatus = required<HTMLElement>("#zoom-status");
  private apiPrefix = this.root.dataset.apiPrefix || "/api/v1";
  private renderer: StarMapRenderer;
  private layout = new LayoutController();
  private autoRotation = new AutoRotationController({}, storedAutoRotationPreference());
  private input: InputController;
  private gesture: GestureController;
  private detail = new DetailPanel(this.panelRoot);
  private model: GraphModel | null = null;
  private currentSessionId = "";
  private graphAbort: AbortController | null = null;
  private chunksAbort: AbortController | null = null;
  private selectedIndex = -1;
  private selectedEdgeIndex = -1;
  private graphReady = false;
  private selectedSearchResult = -1;
  private statsTimer = 0;
  private lastRotationUiKey = "";
  private mouseOrbitInteracting = false;
  private gestureInteracting = false;
  private mouseNodeDragging = false;
  private gestureNodeDragging = false;
  private attractionNode = -1;
  private attractionRequest = 0;
  private eclrrPollTimer = 0;
  private eclrrCompletionSeen = "";

  constructor() {
    this.renderer = new StarMapRenderer(this.stage);
    this.gesture = new GestureController(
      this.stage,
      this.gestureLayer,
      this.gestureVideo,
      this.gestureCursor,
      this.renderer,
      {
        onSelect: nodeIndex => this.selectNode(nodeIndex),
        onClearSelection: () => this.clearSelection(),
        onNodeAttractionStart: nodeIndex => this.beginNodeAttraction(nodeIndex),
        onNodeAttractionEnd: () => this.endNodeAttraction(),
        onHover: nodeIndex => this.onHover(nodeIndex),
        onNodeMoved: nodeIndex => this.setStatus("节点位置已固定", this.model?.nodes[nodeIndex]?.label),
        onGestureActiveChange: active => {
          this.gestureInteracting = active;
          this.updateCombinedInteractionState();
        },
        onNodeDraggingChange: active => {
          this.gestureNodeDragging = active;
          this.updateCombinedInteractionState();
        },
      },
    );
    this.input = new InputController(this.renderer, {
      onSelect: nodeIndex => this.selectNode(nodeIndex),
      onSelectEdge: edgeIndex => this.selectEdge(edgeIndex),
      onNodeAttractionStart: nodeIndex => this.beginNodeAttraction(nodeIndex),
      onNodeAttractionEnd: () => this.endNodeAttraction(),
      onHover: nodeIndex => this.onHover(nodeIndex),
      onEscape: () => this.clearSelection(),
      onNodeMoved: nodeIndex => this.setStatus("节点位置已固定", this.model?.nodes[nodeIndex]?.label),
      onPointerInsideChange: inside => this.autoRotation.setPointerInside(inside),
      onPointerDownChange: down => this.autoRotation.setPointerDown(down),
      onOrbitInteractionChange: active => {
        this.mouseOrbitInteracting = active;
        this.updateCombinedInteractionState();
      },
      onNodeDraggingChange: active => {
        this.mouseNodeDragging = active;
        this.updateCombinedInteractionState();
      },
      onWheel: () => this.autoRotation.noteWheel(),
      onKeyboardInteraction: () => this.autoRotation.markInteraction(),
    });
    this.bindControls();
    this.renderer.onViewChanged(() => this.updateStats());
    this.renderer.onFrame((now, deltaSeconds) => {
      this.tickAutoRotation(now, deltaSeconds);
      this.gesture.tick(now);
    });
    this.autoRotation.setDocumentVisible(!document.hidden);
    document.addEventListener("visibilitychange", this.onVisibilityChange);
    this.root.addEventListener("pointerdown", this.onRootPointerDown, true);
    if (!document.hidden) this.renderer.start();
    this.statsTimer = window.setInterval(() => this.updateStats(), 250);
    window.__PROMO_GRAPH_DEBUG__ = {
      stats: () => ({ ...this.renderer.getStats(), nodes: this.model?.nodes.length ?? 0, edges: this.model?.edges.length ?? 0 }),
      zoom: ratio => {
        const current = this.renderer.zoomRatio;
        this.renderer.zoomAt(this.stage.clientWidth / 2, this.stage.clientHeight / 2, ratio / Math.max(0.001, current));
      },
      select: nodeIndex => this.selectNode(nodeIndex),
      selectEdge: edgeIndex => this.selectEdge(edgeIndex),
      fit: () => this.renderer.fitToGraph(),
      autoRotate: enabled => this.setAutoRotationEnabled(enabled),
      rotation: () => ({ decision: this.autoRotation.evaluate(), state: this.rotationDebugState() }),
      gesture: () => this.gesture.getState(),
      attract: nodeIndex => this.beginNodeAttraction(nodeIndex),
      releaseAttraction: () => this.endNodeAttraction(),
      position: nodeIndex => this.renderer.getNodeWorldPosition(nodeIndex),
      edgePosition: edgeIndex => this.renderer.getEdgeScreenMidpoint(edgeIndex),
    };
  }

  private bindControls(): void {
    required<HTMLButtonElement>("#zoom-in").addEventListener("click", () => this.renderer.zoomAt(this.stage.clientWidth / 2, this.stage.clientHeight / 2, 1.28));
    required<HTMLButtonElement>("#zoom-out").addEventListener("click", () => this.renderer.zoomAt(this.stage.clientWidth / 2, this.stage.clientHeight / 2, 0.78));
    required<HTMLButtonElement>("#fit-graph").addEventListener("click", () => this.renderer.fitToGraph());
    this.rotationButton.addEventListener("click", () => this.setAutoRotationEnabled(!this.autoRotation.snapshot.enabled));
    this.eclrrButton.addEventListener("click", () => void this.startEclrr());
    this.gestureButton.addEventListener("click", () => void this.toggleGestureControl());
    required<HTMLButtonElement>("#close-panel").addEventListener("click", () => this.closePanel());
    required<HTMLButtonElement>("#retry-load").addEventListener("click", () => void this.loadGraph());
    this.panelToggle.addEventListener("click", () => {
      if (this.selectedIndex < 0 && this.selectedEdgeIndex < 0) return;
      this.root.classList.contains("is-panel-open") ? this.closePanel() : this.openPanel();
    });
    this.sessionSelect.addEventListener("change", () => {
      window.clearTimeout(this.eclrrPollTimer);
      this.eclrrCompletionSeen = "";
      this.setEclrrButtonRunning(false);
      this.currentSessionId = this.sessionSelect.value;
      storeSessionPreference(this.currentSessionId);
      const url = new URL(window.location.href);
      url.searchParams.set("session_id", this.currentSessionId);
      window.history.replaceState({}, "", url);
      void this.loadGraph().then(() => this.refreshEclrrStatus());
    });
    this.searchInput.addEventListener("input", () => this.updateSearchResults());
    this.searchInput.addEventListener("keydown", event => this.onSearchKeyDown(event));
    this.searchInput.addEventListener("blur", () => window.setTimeout(() => this.hideSearchResults(), 120));
  }

  async init(): Promise<void> {
    createIcons({ icons: { ArrowLeft, Hand, Maximize2, PanelRight, RotateCw, Search, Sparkles, TriangleAlert, X, ZoomIn, ZoomOut } });
    this.updateRotationButton();
    this.updateGestureButton();
    await this.loadSessions();
    if (this.currentSessionId) {
      await this.loadGraph();
      await this.refreshEclrrStatus();
    }
  }

  private async loadSessions(): Promise<void> {
    this.showLoading("读取会话列表");
    const response = await fetch(`${this.apiPrefix}/sessions`, { cache: "no-store" });
    if (!response.ok) throw new Error(`会话列表读取失败（HTTP ${response.status}）`);
    const payload = await response.json();
    const sessions: SessionRecord[] = Array.isArray(payload) ? payload : Array.isArray(payload.sessions) ? payload.sessions : [];
    this.sessionSelect.replaceChildren();
    sessions.forEach(session => {
      const id = sessionId(session);
      if (!id) return;
      const option = document.createElement("option");
      option.value = id;
      option.textContent = sessionLabel(session);
      this.sessionSelect.append(option);
    });
    const querySession = new URLSearchParams(window.location.search).get("session_id") || "";
    const stored = storedSessionPreference();
    const available = new Set(sessions.map(sessionId));
    this.currentSessionId = [querySession, stored, sessionId(sessions[0] ?? {})].find(id => id && available.has(id)) || "";
    this.sessionSelect.value = this.currentSessionId;
    if (this.currentSessionId) storeSessionPreference(this.currentSessionId);
    if (!this.currentSessionId) throw new Error("当前没有可显示的知识图谱会话");
  }

  private async loadGraph(): Promise<void> {
    if (!this.currentSessionId) return;
    this.graphAbort?.abort();
    this.graphAbort = new AbortController();
    this.graphReady = false;
    this.autoRotation.setLayoutStable(false);
    this.clearSelection();
    this.showLoading("读取完整节点与关系");
    try {
      const url = `${this.apiPrefix}/knowledge-graph/data?session_id=${encodeURIComponent(this.currentSessionId)}&scope=full`;
      const response = await fetch(url, { cache: "no-store", signal: this.graphAbort.signal });
      if (!response.ok) throw new Error(`图谱接口返回 HTTP ${response.status}`);
      const payload = await response.json() as RawGraphResponse;
      const error = String(payload.metadata?.error ?? "");
      if (error) throw new Error(error);
      const model = normalizeGraph(payload);
      if (!model.nodes.length) throw new Error("该会话还没有可显示的知识图谱节点");
      this.model = model;
      this.showLoading(model.nodes.length >= 2_000 ? "正在计算多层 ForceAtlas2 布局" : "正在计算二维力导向布局");
      const layoutResult = await this.layout.layout(model);
      this.renderer.setGraph(model, layoutResult.positions);
      this.graphReady = true;
      this.autoRotation.setLayoutStable(true);
      this.loading.hidden = true;
      this.errorState.hidden = true;
      const totalNodes = Number(model.metadata.total_nodes ?? model.nodes.length);
      const truncated = Boolean(model.metadata.truncated) || totalNodes > model.nodes.length;
      this.setStatus(truncated ? "图谱数据被截断" : layoutResult.cached ? "知识星图已从缓存恢复" : "知识星图已就绪");
      this.updateStats();
    } catch (error) {
      this.graphReady = false;
      this.autoRotation.setLayoutStable(false);
      if ((error as Error).name === "AbortError" || (error as Error).message === "Layout request cancelled") return;
      this.showError((error as Error).message || "未知错误");
    }
  }

  private selectNode(nodeIndex: number): void {
    if (!this.model || nodeIndex < 0) {
      this.clearSelection();
      return;
    }
    if (nodeIndex === this.selectedIndex) {
      this.clearSelection();
      return;
    }
    this.selectedIndex = nodeIndex;
    this.selectedEdgeIndex = -1;
    this.autoRotation.setHasSelection(true);
    const node = this.model.nodes[nodeIndex];
    this.renderer.setSelection(nodeIndex);
    this.detail.show(node);
    this.openPanel();
    this.setStatus("已聚焦节点", node.label);
    void this.loadChunks(node.id);
  }

  private selectEdge(edgeIndex: number): void {
    if (!this.model || edgeIndex < 0 || edgeIndex >= this.model.edges.length) {
      this.clearSelection();
      return;
    }
    if (edgeIndex === this.selectedEdgeIndex) {
      this.clearSelection();
      return;
    }
    this.selectedIndex = -1;
    this.selectedEdgeIndex = edgeIndex;
    this.autoRotation.setHasSelection(true);
    const edge = this.model.edges[edgeIndex];
    const source = this.model.nodes[edge.source];
    const target = this.model.nodes[edge.target];
    this.renderer.setEdgeSelection(edgeIndex);
    this.detail.showEdge(edge, source, target);
    this.openPanel();
    this.setStatus(edge.kind === "eclrr_v4" ? "已聚焦 ECLRR-v4 推断关系" : "已聚焦事实关系", `${source.label} → ${target.label}`);
    void this.loadEdgeChunks(edgeIndex);
  }

  private beginNodeAttraction(nodeIndex: number): void {
    if (!this.model || nodeIndex < 0 || nodeIndex >= this.model.nodes.length) return;
    this.attractionNode = nodeIndex;
    const request = ++this.attractionRequest;
    this.renderer.setSelection(nodeIndex);
    this.renderer.beginFocus(nodeIndex);
    void this.layout.focus(nodeIndex).then(result => {
      if (request !== this.attractionRequest || this.attractionNode !== nodeIndex || !result.indices.length) return;
      this.renderer.animateFocus(result.indices, result.positions, 350);
    }).catch(() => undefined);
  }

  private endNodeAttraction(): void {
    if (this.attractionNode < 0) return;
    this.attractionNode = -1;
    this.attractionRequest += 1;
    this.renderer.clearFocus(320);
    if (this.selectedEdgeIndex >= 0) this.renderer.setEdgeSelection(this.selectedEdgeIndex);
    else this.renderer.setSelection(this.selectedIndex);
  }

  private clearSelection(): void {
    this.attractionNode = -1;
    this.attractionRequest += 1;
    this.selectedIndex = -1;
    this.selectedEdgeIndex = -1;
    this.autoRotation.setHasSelection(false);
    this.chunksAbort?.abort();
    this.renderer.clearSelection(true);
    this.detail.hide();
    this.closePanel();
    this.setStatus(this.model ? "知识星图已就绪" : "准备加载");
  }

  private async loadChunks(entityId: string): Promise<void> {
    this.chunksAbort?.abort();
    const controller = new AbortController();
    this.chunksAbort = controller;
    try {
      const url = `${this.apiPrefix}/knowledge-graph/entity-chunks?session_id=${encodeURIComponent(this.currentSessionId)}&entity_id=${encodeURIComponent(entityId)}&max_chunks=0`;
      const response = await fetch(url, { cache: "no-store", signal: controller.signal });
      const payload = await response.json() as EntityChunkResponse;
      if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);
      if (this.model?.nodes[this.selectedIndex]?.id === entityId) this.detail.showChunks(payload.chunks ?? []);
    } catch (error) {
      if ((error as Error).name !== "AbortError") this.detail.showChunks([], `原文读取失败：${(error as Error).message}`);
    }
  }

  private async loadEdgeChunks(edgeIndex: number): Promise<void> {
    const edge = this.model?.edges[edgeIndex];
    if (!edge) return;
    this.chunksAbort?.abort();
    const controller = new AbortController();
    this.chunksAbort = controller;
    try {
      const params = new URLSearchParams({
        session_id: this.currentSessionId,
        source_id: edge.sourceId,
        edge_id: edge.id,
        max_chunks: "0",
      });
      const response = await fetch(`${this.apiPrefix}/knowledge-graph/edge-chunks?${params}`, { cache: "no-store", signal: controller.signal });
      const payload = await response.json() as EntityChunkResponse;
      if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);
      if (this.selectedEdgeIndex === edgeIndex && this.model?.edges[edgeIndex]?.id === edge.id) this.detail.showChunks(payload.chunks ?? []);
    } catch (error) {
      if ((error as Error).name !== "AbortError" && this.selectedEdgeIndex === edgeIndex) {
        this.detail.showChunks([], `原文读取失败：${(error as Error).message}`);
      }
    }
  }

  private updateSearchResults(): void {
    if (!this.model) return;
    const query = this.searchInput.value.trim().toLocaleLowerCase();
    this.searchResults.replaceChildren();
    this.selectedSearchResult = -1;
    if (!query) {
      this.renderer.setSearchMatch(-1);
      this.hideSearchResults();
      return;
    }
    const matches = this.model.nodes
      .map((node, nodeIndex) => ({ node, nodeIndex, label: node.label.toLocaleLowerCase() }))
      .filter(item => item.label.includes(query) || item.node.type.toLocaleLowerCase().includes(query))
      .sort((a, b) => Number(b.label.startsWith(query)) - Number(a.label.startsWith(query)) || b.node.degree - a.node.degree)
      .slice(0, 12);
    matches.forEach((item, resultIndex) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "search-result";
      button.role = "option";
      button.dataset.nodeIndex = String(item.nodeIndex);
      button.dataset.resultIndex = String(resultIndex);
      const label = document.createElement("strong");
      label.textContent = item.node.label;
      const meta = document.createElement("small");
      meta.textContent = `${item.node.type} · ${item.node.degree} 条连接`;
      button.append(label, meta);
      button.addEventListener("pointerdown", event => event.preventDefault());
      button.addEventListener("click", () => this.chooseSearchResult(item.nodeIndex));
      this.searchResults.append(button);
    });
    this.searchResults.hidden = matches.length === 0;
    this.searchInput.setAttribute("aria-expanded", String(matches.length > 0));
    this.renderer.setSearchMatch(matches[0]?.nodeIndex ?? -1);
  }

  private onSearchKeyDown(event: KeyboardEvent): void {
    const results = [...this.searchResults.querySelectorAll<HTMLButtonElement>(".search-result")];
    if (!results.length) return;
    if (event.key === "ArrowDown") this.selectedSearchResult = Math.min(results.length - 1, this.selectedSearchResult + 1);
    else if (event.key === "ArrowUp") this.selectedSearchResult = Math.max(0, this.selectedSearchResult - 1);
    else if (event.key === "Enter") {
      const target = results[Math.max(0, this.selectedSearchResult)];
      if (target) this.chooseSearchResult(Number(target.dataset.nodeIndex));
      event.preventDefault();
      return;
    } else if (event.key === "Escape") {
      this.hideSearchResults();
      return;
    } else return;
    results.forEach((result, index) => result.setAttribute("aria-selected", String(index === this.selectedSearchResult)));
    results[this.selectedSearchResult]?.scrollIntoView({ block: "nearest" });
    event.preventDefault();
  }

  private chooseSearchResult(nodeIndex: number): void {
    this.selectNode(nodeIndex);
    this.renderer.centerOnNode(nodeIndex, 2.1);
    this.searchInput.value = this.model?.nodes[nodeIndex]?.label ?? "";
    this.renderer.setSearchMatch(nodeIndex);
    this.hideSearchResults();
  }

  private hideSearchResults(): void {
    this.searchResults.hidden = true;
    this.searchInput.setAttribute("aria-expanded", "false");
  }

  private onHover(nodeIndex: number): void {
    if (nodeIndex >= 0 && this.model) this.status.textContent = this.model.nodes[nodeIndex].label;
    else if (this.selectedIndex < 0 && this.selectedEdgeIndex < 0 && this.model) this.status.textContent = "知识星图已就绪";
  }

  private openPanel(): void {
    if (this.selectedIndex < 0 && this.selectedEdgeIndex < 0) return;
    this.panelRoot.hidden = false;
    requestAnimationFrame(() => this.root.classList.add("is-panel-open"));
    this.panelToggle.setAttribute("aria-expanded", "true");
  }

  private closePanel(): void {
    this.root.classList.remove("is-panel-open");
    this.panelToggle.setAttribute("aria-expanded", "false");
    window.setTimeout(() => {
      if (!this.root.classList.contains("is-panel-open")) this.panelRoot.hidden = true;
    }, 230);
  }

  private showLoading(detail: string): void {
    this.loading.hidden = false;
    this.loadingDetail.textContent = detail;
    this.errorState.hidden = true;
  }

  private showError(message: string): void {
    this.loading.hidden = true;
    this.errorState.hidden = false;
    this.errorMessage.textContent = message;
    this.setStatus("图谱加载失败");
  }

  private setStatus(message: string, target = ""): void {
    this.status.textContent = target ? `${message} · ${target}` : message;
  }

  private updateStats(): void {
    if (document.hidden) return;
    const stats = this.renderer.getStats();
    this.counts.textContent = `${this.model?.nodes.length ?? 0} 节点 · ${this.model?.edges.length ?? 0} 关系`;
    this.zoomStatus.textContent = `${stats.zoomRatio.toFixed(2)}×`;
    this.stage.dataset.webglReady = String(this.graphReady);
    this.stage.dataset.nodeCount = String(this.model?.nodes.length ?? 0);
    this.stage.dataset.edgeCount = String(this.model?.edges.length ?? 0);
    this.stage.dataset.selectedNode = String(this.selectedIndex);
    this.stage.dataset.selectedEdge = String(this.selectedEdgeIndex);
    this.stage.dataset.labelLevel = String(stats.labelLevel);
    this.stage.dataset.zoomRatio = stats.zoomRatio.toFixed(3);
    this.stage.dataset.fps = stats.fps.toFixed(1);
    this.stage.dataset.drawCalls = String(stats.calls);
    this.stage.dataset.labelCount = String(stats.labels);
    this.stage.dataset.glyphCount = String(stats.glyphs);
    this.stage.dataset.triangles = String(stats.triangles);
  }

  private setEclrrButtonRunning(running: boolean): void {
    this.eclrrButton.disabled = running;
    this.eclrrButton.classList.toggle("is-running", running);
    const title = running
      ? "ECLRR-v4 自进化正在运行"
      : "运行 ECLRR-v4 自进化，最多生成 40 条新边";
    this.eclrrButton.title = title;
    this.eclrrButton.setAttribute("aria-label", title);
  }

  private async startEclrr(): Promise<void> {
    if (!this.currentSessionId || this.eclrrButton.disabled) return;
    this.setEclrrButtonRunning(true);
    this.setStatus("ECLRR-v4 自进化启动中", "最多写回 40 条新边");
    try {
      const response = await fetch(`${this.apiPrefix}/knowledge-graph/eclrr-v4/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: this.currentSessionId, max_new_edges: 40 }),
      });
      const payload = await response.json() as EclrrRunStatus & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      this.applyEclrrStatus(payload);
    } catch (error) {
      this.setEclrrButtonRunning(false);
      this.setStatus("ECLRR-v4 启动失败", (error as Error).message);
    }
  }

  private async refreshEclrrStatus(): Promise<void> {
    if (!this.currentSessionId) return;
    try {
      const url = `${this.apiPrefix}/knowledge-graph/eclrr-v4/status?session_id=${encodeURIComponent(this.currentSessionId)}`;
      const response = await fetch(url, { cache: "no-store" });
      const payload = await response.json() as EclrrRunStatus;
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      this.applyEclrrStatus(payload);
    } catch (error) {
      this.setEclrrButtonRunning(false);
      this.setStatus("ECLRR-v4 状态读取失败", (error as Error).message);
    }
  }

  private applyEclrrStatus(payload: EclrrRunStatus): void {
    window.clearTimeout(this.eclrrPollTimer);
    if (payload.status === "running") {
      this.setEclrrButtonRunning(true);
      this.setStatus("ECLRR-v4 自进化中", `最多 ${payload.max_new_edges} 条新边`);
      this.eclrrPollTimer = window.setTimeout(() => void this.refreshEclrrStatus(), 1_500);
      return;
    }

    this.setEclrrButtonRunning(false);
    if (payload.status === "failed") {
      this.setStatus("ECLRR-v4 自进化失败", payload.error || "未知错误");
      return;
    }
    if (payload.status !== "completed") return;

    const completionKey = payload.finished_at || payload.started_at || "completed";
    if (completionKey === this.eclrrCompletionSeen) return;
    this.eclrrCompletionSeen = completionKey;
    void this.loadGraph().then(() => {
      this.setStatus(
        "ECLRR-v4 自进化完成",
        `审核 ${payload.reviewed} · 提议 ${payload.proposed} · 写回 ${payload.accepted} · 拒绝 ${payload.rejected}`,
      );
    });
  }

  private setAutoRotationEnabled(enabled: boolean): void {
    this.autoRotation.setEnabled(enabled);
    try {
      localStorage.setItem(AUTO_ROTATION_STORAGE_KEY, String(enabled));
    } catch {
      // Storage can be unavailable in privacy-restricted browser contexts.
    }
    this.updateRotationButton();
  }

  private async toggleGestureControl(): Promise<void> {
    if (this.gesture.getState().enabled) {
      this.gesture.stop();
      this.gestureInteracting = false;
      this.gestureNodeDragging = false;
      this.updateCombinedInteractionState();
      this.updateGestureButton();
      return;
    }
    this.gestureButton.disabled = true;
    this.gestureButton.title = "正在启动手势控制";
    this.gestureButton.setAttribute("aria-label", "正在启动手势控制");
    const started = await this.gesture.start();
    this.gestureButton.disabled = false;
    this.updateGestureButton(started);
  }

  private updateGestureButton(enabled = this.gesture.getState().enabled): void {
    const title = enabled ? "关闭手势控制" : "开启手势控制";
    this.gestureButton.title = title;
    this.gestureButton.setAttribute("aria-label", title);
    this.gestureButton.setAttribute("aria-pressed", String(enabled));
  }

  private tickAutoRotation(now: number, deltaSeconds: number): void {
    const decision = this.autoRotation.evaluate(now);
    if (decision.shouldRotate) {
      this.renderer.advanceAutomaticOrbit(deltaSeconds, this.autoRotation.config.angularSpeedRadPerSecond);
    }
    const uiKey = `${this.autoRotation.snapshot.enabled}:${decision.pausedReason ?? "rotating"}`;
    if (uiKey !== this.lastRotationUiKey) {
      this.lastRotationUiKey = uiKey;
      this.updateRotationButton(decision.pausedReason);
    }
    this.stage.dataset.autoRotating = String(decision.shouldRotate);
    this.stage.dataset.rotationPauseReason = decision.pausedReason ?? "";
  }

  private updateRotationButton(pausedReason = this.autoRotation.evaluate().pausedReason): void {
    const enabled = this.autoRotation.snapshot.enabled;
    const title = !enabled
      ? "自动旋转已关闭"
      : pausedReason === "selection"
        ? "自动旋转已开启，当前因选中图谱元素而暂停"
        : "自动旋转已开启";
    this.rotationButton.title = title;
    this.rotationButton.setAttribute("aria-label", title);
    this.rotationButton.setAttribute("aria-pressed", String(enabled));
  }

  private rotationDebugState() {
    return {
      ...this.autoRotation.snapshot,
      camera: this.renderer.getCameraOrbitState(),
    };
  }

  private updateCombinedInteractionState(): void {
    this.autoRotation.setOrbitInteracting(this.mouseOrbitInteracting || this.gestureInteracting);
    this.autoRotation.setNodeDragging(this.mouseNodeDragging || this.gestureNodeDragging);
  }

  private onRootPointerDown = () => this.autoRotation.markInteraction();

  private onVisibilityChange = () => {
    const visible = !document.hidden;
    this.autoRotation.setDocumentVisible(visible);
    this.gesture.setSuspended(!visible);
    if (visible) this.renderer.start();
    else this.renderer.stop();
    this.updateRotationButton();
  };

  dispose(): void {
    window.clearInterval(this.statsTimer);
    window.clearTimeout(this.eclrrPollTimer);
    this.graphAbort?.abort();
    this.chunksAbort?.abort();
    document.removeEventListener("visibilitychange", this.onVisibilityChange);
    this.root.removeEventListener("pointerdown", this.onRootPointerDown, true);
    this.input.dispose();
    this.gesture.dispose();
    this.layout.dispose();
    this.renderer.dispose();
    delete window.__PROMO_GRAPH_DEBUG__;
  }
}

const app = new PromoGraphApp();
void app.init().catch(error => {
  const loading = document.querySelector<HTMLElement>("#loading-state");
  const errorState = document.querySelector<HTMLElement>("#error-state");
  const errorMessage = document.querySelector<HTMLElement>("#error-message");
  if (loading) loading.hidden = true;
  if (errorState) errorState.hidden = false;
  if (errorMessage) errorMessage.textContent = (error as Error).message;
});
window.addEventListener("beforeunload", () => app.dispose(), { once: true });
