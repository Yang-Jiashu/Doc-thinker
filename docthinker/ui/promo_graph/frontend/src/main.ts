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
import { GestureController, pauseGestureForManualInput, pauseGestureForPointer } from "./gesture/gesture-controller";
import { normalizeGesturePreferences } from "./gesture/gesture-preferences";
import { graphViewUrl, legacyGestureHelpRequested, readGraphViewState } from "./view-state";
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
const GESTURE_PREFERENCES_KEY = "docthinker.gesture.preferences.v1";

function storedGesturePreferences() {
  try {
    return normalizeGesturePreferences(JSON.parse(localStorage.getItem(GESTURE_PREFERENCES_KEY) || "{}"));
  } catch {
    return normalizeGesturePreferences({});
  }
}

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
  private gestureDialog = required<HTMLDialogElement>("#gesture-settings");
  private cameraStart = required<HTMLButtonElement>("#camera-start");
  private cameraPause = required<HTMLButtonElement>("#camera-pause");
  private cameraStop = required<HTMLButtonElement>("#camera-stop");
  private cameraStatus = required<HTMLElement>("#camera-status");
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
  private initialViewState = readGraphViewState(window.location.search);
  private pendingNodeId = this.initialViewState.node;
  private helpRequested = legacyGestureHelpRequested(window.location.search);
  private headerObserver: ResizeObserver | null = null;
  private disposed = false;

  constructor() {
    this.renderer = new StarMapRenderer(this.stage);
    this.gesture = new GestureController(
      this.stage,
      this.gestureLayer,
      this.gestureVideo,
      this.gestureCursor,
      this.renderer,
      {
        onSelect: nodeIndex => this.selectNode(nodeIndex, false),
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
        onStateChange: () => { this.updateGestureButton(); this.updateCombinedInteractionState(); },
      },
      storedGesturePreferences(),
    );
    this.searchInput.value = this.initialViewState.search;
    this.input = new InputController(this.renderer, {
      onSelect: nodeIndex => this.selectNode(nodeIndex),
      onSelectEdge: edgeIndex => this.selectEdge(edgeIndex),
      onNodeAttractionStart: nodeIndex => this.beginNodeAttraction(nodeIndex),
      onNodeAttractionEnd: () => this.endNodeAttraction(),
      onHover: nodeIndex => this.onHover(nodeIndex),
      onEscape: () => this.clearSelection(),
      onNodeMoved: nodeIndex => this.setStatus("节点位置已固定", this.model?.nodes[nodeIndex]?.label),
      onPointerInsideChange: inside => this.autoRotation.setPointerInside(inside),
      onPointerDownChange: down => {
        this.autoRotation.setPointerDown(down);
        if (pauseGestureForPointer(this.gesture, down)) this.setStatus("鼠标 / 触控已接管", "手势已暂停，可在手势设置中恢复");
      },
      onOrbitInteractionChange: active => {
        this.mouseOrbitInteracting = active;
        this.updateCombinedInteractionState();
      },
      onNodeDraggingChange: active => {
        this.mouseNodeDragging = active;
        this.updateCombinedInteractionState();
      },
      onWheel: () => { this.takeManualControl(); this.autoRotation.noteWheel(); },
      onKeyboardInteraction: () => { this.takeManualControl(); this.autoRotation.markInteraction(); },
    });
    this.bindControls();
    this.renderer.onViewChanged(() => this.updateStats());
    this.renderer.onFrame((now, deltaSeconds) => {
      this.tickAutoRotation(now, deltaSeconds);
      this.gesture.tick(now);
    });
    this.autoRotation.setDocumentVisible(!document.hidden);
    document.addEventListener("visibilitychange", this.onVisibilityChange);
    window.addEventListener("pagehide", this.onPageHide);
    window.addEventListener("pageshow", this.onPageShow);
    window.addEventListener("blur", this.onWindowBlur);
    this.root.addEventListener("pointerdown", this.onRootPointerDown, true);
    if (!document.hidden) this.renderer.start();
    this.statsTimer = window.setInterval(() => this.updateStats(), 250);
    const header = required<HTMLElement>(".app-bar");
    this.headerObserver = new ResizeObserver(() => this.root.style.setProperty("--toolbar-height", `${header.offsetHeight}px`));
    this.headerObserver.observe(header);
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
    for (const id of ["#standard-view", "#immersive-view"]) {
      required<HTMLAnchorElement>(id).addEventListener("click", event => {
        if (!this.currentSessionId) {
          event.preventDefault();
          this.showError("请先选择有效会话，再切换视图。");
        }
      });
    }
    required<HTMLButtonElement>("#zoom-in").addEventListener("click", () => {
      this.takeManualControl(); this.renderer.zoomAt(this.stage.clientWidth / 2, this.stage.clientHeight / 2, 1.28);
    });
    required<HTMLButtonElement>("#zoom-out").addEventListener("click", () => {
      this.takeManualControl(); this.renderer.zoomAt(this.stage.clientWidth / 2, this.stage.clientHeight / 2, 0.78);
    });
    required<HTMLButtonElement>("#fit-graph").addEventListener("click", () => { this.takeManualControl(); this.renderer.fitToGraph(); });
    this.rotationButton.addEventListener("click", () => this.setAutoRotationEnabled(!this.autoRotation.snapshot.enabled));
    this.eclrrButton.addEventListener("click", () => void this.startEclrr());
    this.gestureButton.addEventListener("click", () => this.openGestureSettings());
    required<HTMLButtonElement>("#close-gesture-settings").addEventListener("click", () => this.gestureDialog.close());
    this.gestureDialog.addEventListener("close", () => {
      this.gestureButton.setAttribute("aria-expanded", "false");
      this.gestureButton.focus();
      this.updateCombinedInteractionState();
    });
    this.cameraStart.addEventListener("click", () => void this.gesture.start().then(started => {
      if (started && !document.hidden) this.gestureDialog.close();
    }));
    this.cameraPause.addEventListener("click", () => {
      const resume = this.gesture.getState().suspended;
      this.gesture.setSuspended(!resume);
      if (resume) this.gestureDialog.close();
    });
    this.cameraStop.addEventListener("click", () => this.gesture.stop());
    this.gestureDialog.querySelectorAll<HTMLSelectElement | HTMLInputElement>("[data-gesture-pref]").forEach(control => {
      control.addEventListener("change", () => this.saveGestureSettings());
    });
    required<HTMLButtonElement>("#reset-gesture-settings").addEventListener("click", () => {
      this.gesture.setPreferences(normalizeGesturePreferences({}));
      this.persistGestureSettings();
      this.populateGestureSettings();
    });
    required<HTMLAnchorElement>("#standard-view").addEventListener("click", () => this.gesture.stop());
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
      this.pendingNodeId = "";
      this.searchInput.value = "";
      this.clearSelection(false);
      this.currentSessionId = this.sessionSelect.value;
      storeSessionPreference(this.currentSessionId);
      this.syncViewState();
      const session = this.currentSessionId;
      void this.loadGraph().then(() => { if (session === this.currentSessionId) return this.refreshEclrrStatus(); });
    });
    this.searchInput.addEventListener("input", () => { this.updateSearchResults(); this.syncViewState(); });
    this.searchInput.addEventListener("keydown", event => this.onSearchKeyDown(event));
    this.searchInput.addEventListener("blur", () => window.setTimeout(() => this.hideSearchResults(), 120));
  }

  async init(): Promise<void> {
    createIcons({ icons: { ArrowLeft, Hand, Maximize2, PanelRight, RotateCw, Search, Sparkles, TriangleAlert, X, ZoomIn, ZoomOut } });
    this.updateRotationButton();
    this.updateGestureButton();
    this.populateGestureSettings();
    if (this.helpRequested) this.openGestureSettings();
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
    const querySession = this.initialViewState.sessionId;
    const stored = storedSessionPreference();
    const available = new Set(sessions.map(sessionId));
    if (querySession && !available.has(querySession)) {
      this.currentSessionId = "";
      this.pendingNodeId = "";
      this.sessionSelect.value = "";
      throw new Error("链接中的会话不存在或不可访问，请从会话列表重新选择。");
    }
    this.currentSessionId = [querySession, stored, sessionId(sessions[0] ?? {})].find(id => id && available.has(id)) || "";
    this.sessionSelect.value = this.currentSessionId;
    if (querySession && querySession !== this.currentSessionId) this.pendingNodeId = "";
    if (this.currentSessionId) storeSessionPreference(this.currentSessionId);
    if (!this.currentSessionId) throw new Error("当前没有可显示的知识图谱会话");
    this.syncViewState();
  }

  private async loadGraph(): Promise<void> {
    if (!this.currentSessionId) return;
    this.graphAbort?.abort();
    const controller = new AbortController();
    const session = this.currentSessionId;
    this.graphAbort = controller;
    const current = () => !this.disposed && this.graphAbort === controller && session === this.currentSessionId && !controller.signal.aborted;
    const selectedId = this.model?.nodes[this.selectedIndex]?.id || this.pendingNodeId;
    this.pendingNodeId = selectedId;
    this.graphReady = false;
    this.updateGestureButton();
    this.autoRotation.setLayoutStable(false);
    this.gesture.setSuspended(true);
    this.input.resetInteraction();
    this.clearSelection(false);
    this.model = null;
    this.hideSearchResults();
    this.searchResults.replaceChildren();
    this.renderer.setGraph(normalizeGraph({ nodes: [], edges: [] }), new Float32Array());
    this.showLoading("读取完整节点与关系");
    try {
      const params = new URLSearchParams({ session_id: session, scope: "full" });
      const response = await fetch(`${this.apiPrefix}/knowledge-graph/data?${params}`, { cache: "no-store", signal: controller.signal });
      if (!response.ok) throw new Error(`图谱接口返回 HTTP ${response.status}`);
      const payload = await response.json() as RawGraphResponse;
      if (!current()) return;
      const error = String(payload.metadata?.error ?? "");
      if (error) throw new Error(error);
      const model = normalizeGraph(payload);
      if (!model.nodes.length) throw new Error("该会话还没有可显示的知识图谱节点");
      this.showLoading(model.nodes.length >= 2_000 ? "正在计算多层 ForceAtlas2 布局" : "正在计算二维力导向布局");
      const layoutResult = await this.layout.layout(model);
      if (!current()) return;
      this.model = model;
      this.renderer.setGraph(model, layoutResult.positions);
      this.graphReady = true;
      this.autoRotation.setLayoutStable(true);
      this.loading.hidden = true;
      this.errorState.hidden = true;
      const totalNodes = Number(model.metadata.total_nodes ?? model.nodes.length);
      const truncated = Boolean(model.metadata.truncated) || totalNodes > model.nodes.length;
      this.setStatus(truncated ? "图谱数据被截断" : layoutResult.cached ? "知识星图已从缓存恢复" : "知识星图已就绪");
      const restoredIndex = model.nodes.findIndex(node => node.id === selectedId);
      this.pendingNodeId = "";
      if (restoredIndex >= 0) {
        this.selectNode(restoredIndex, false);
        this.renderer.centerOnNode(restoredIndex, 2.1);
      } else if (selectedId) this.setStatus("该节点不在当前图谱中", "会话和搜索条件已保留");
      this.updateSearchResults();
      this.hideSearchResults();
      this.syncViewState();
      this.updateGestureButton();
      this.updateStats();
    } catch (error) {
      if (!current()) return;
      this.graphReady = false;
      this.autoRotation.setLayoutStable(false);
      if ((error as Error).name === "AbortError" || (error as Error).message === "Layout request cancelled") return;
      this.showError((error as Error).message || "未知错误");
      this.updateGestureButton();
    }
  }

  private selectNode(nodeIndex: number, toggle = true): void {
    if (!this.model || nodeIndex < 0 || nodeIndex >= this.model.nodes.length) {
      this.clearSelection();
      return;
    }
    if (nodeIndex === this.selectedIndex) {
      if (toggle) this.clearSelection();
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
    this.syncViewState();
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
    this.syncViewState();
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

  private clearSelection(sync = true): void {
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
    if (sync) { this.pendingNodeId = ""; this.syncViewState(); }
  }

  private async loadChunks(entityId: string): Promise<void> {
    this.chunksAbort?.abort();
    const controller = new AbortController();
    this.chunksAbort = controller;
    const session = this.currentSessionId;
    const current = () => !this.disposed && !controller.signal.aborted && this.chunksAbort === controller && this.currentSessionId === session;
    try {
      const params = new URLSearchParams({ session_id: session, entity_id: entityId, max_chunks: "0" });
      const url = `${this.apiPrefix}/knowledge-graph/entity-chunks?${params}`;
      const response = await fetch(url, { cache: "no-store", signal: controller.signal });
      const payload = await response.json() as EntityChunkResponse;
      if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);
      if (current() && this.model?.nodes[this.selectedIndex]?.id === entityId) this.detail.showChunks(payload.chunks ?? []);
    } catch (error) {
      if (current() && (error as Error).name !== "AbortError") this.detail.showChunks([], `原文读取失败：${(error as Error).message}`);
    }
  }

  private async loadEdgeChunks(edgeIndex: number): Promise<void> {
    const edge = this.model?.edges[edgeIndex];
    if (!edge) return;
    this.chunksAbort?.abort();
    const controller = new AbortController();
    this.chunksAbort = controller;
    const session = this.currentSessionId;
    const current = () => !this.disposed && !controller.signal.aborted && this.chunksAbort === controller && this.currentSessionId === session;
    try {
      const params = new URLSearchParams({
        session_id: session,
        source_id: edge.sourceId,
        edge_id: edge.id,
        max_chunks: "0",
      });
      const response = await fetch(`${this.apiPrefix}/knowledge-graph/edge-chunks?${params}`, { cache: "no-store", signal: controller.signal });
      const payload = await response.json() as EntityChunkResponse;
      if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);
      if (current() && this.selectedEdgeIndex === edgeIndex && this.model?.edges[edgeIndex]?.id === edge.id) this.detail.showChunks(payload.chunks ?? []);
    } catch (error) {
      if (current() && (error as Error).name !== "AbortError" && this.selectedEdgeIndex === edgeIndex) {
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
    this.selectNode(nodeIndex, false);
    this.renderer.centerOnNode(nodeIndex, 2.1);
    this.searchInput.value = this.model?.nodes[nodeIndex]?.label ?? "";
    this.syncViewState();
    this.renderer.setSearchMatch(nodeIndex);
    this.hideSearchResults();
  }

  private hideSearchResults(): void {
    this.searchResults.hidden = true;
    this.searchInput.setAttribute("aria-expanded", "false");
  }

  private onHover(nodeIndex: number): void {
    if (nodeIndex >= 0 && this.model?.nodes[nodeIndex]) this.status.textContent = this.model.nodes[nodeIndex].label;
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
    const session = this.currentSessionId;
    this.setEclrrButtonRunning(true);
    this.setStatus("ECLRR-v4 自进化启动中", "最多写回 40 条新边");
    try {
      const response = await fetch(`${this.apiPrefix}/knowledge-graph/eclrr-v4/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: session, max_new_edges: 40 }),
      });
      const payload = await response.json() as EclrrRunStatus & { detail?: string };
      if (this.disposed || session !== this.currentSessionId) return;
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      this.applyEclrrStatus(payload);
    } catch (error) {
      if (this.disposed || session !== this.currentSessionId) return;
      this.setEclrrButtonRunning(false);
      this.setStatus("ECLRR-v4 启动失败", (error as Error).message);
    }
  }

  private async refreshEclrrStatus(): Promise<void> {
    if (!this.currentSessionId) return;
    const session = this.currentSessionId;
    try {
      const url = `${this.apiPrefix}/knowledge-graph/eclrr-v4/status?session_id=${encodeURIComponent(session)}`;
      const response = await fetch(url, { cache: "no-store" });
      const payload = await response.json() as EclrrRunStatus;
      if (this.disposed || session !== this.currentSessionId) return;
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      this.applyEclrrStatus(payload);
    } catch (error) {
      if (this.disposed || session !== this.currentSessionId) return;
      this.setEclrrButtonRunning(false);
      this.setStatus("ECLRR-v4 状态读取失败", (error as Error).message);
    }
  }

  private applyEclrrStatus(payload: EclrrRunStatus): void {
    if (this.disposed || payload.session_id !== this.currentSessionId) return;
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
      if (this.disposed || payload.session_id !== this.currentSessionId) return;
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

  private syncViewState(): void {
    const state = {
      sessionId: this.currentSessionId,
      node: this.model?.nodes[this.selectedIndex]?.id || this.pendingNodeId,
      search: this.searchInput.value,
    };
    required<HTMLAnchorElement>("#standard-view").href = graphViewUrl("/knowledge-graph", state);
    required<HTMLAnchorElement>("#immersive-view").href = graphViewUrl(window.location.pathname, state);
    window.history.replaceState({}, "", graphViewUrl(window.location.pathname, state));
  }

  private openGestureSettings(): void {
    this.gesture.setSuspended(true);
    this.input.resetInteraction();
    this.populateGestureSettings();
    if (!this.gestureDialog.open) this.gestureDialog.showModal();
    this.gestureButton.setAttribute("aria-expanded", "true");
    this.updateCombinedInteractionState();
  }

  private populateGestureSettings(): void {
    const preferences = this.gesture.getPreferences();
    this.gestureDialog.querySelectorAll<HTMLSelectElement | HTMLInputElement>("[data-gesture-pref]").forEach(control => {
      const key = control.dataset.gesturePref as keyof typeof preferences;
      control.value = String(preferences[key]);
      const output = this.gestureDialog.querySelector<HTMLOutputElement>(`[data-gesture-output="${key}"]`);
      if (output) output.value = key.endsWith("Ms") ? `${preferences[key]} 毫秒` : `${preferences[key]}×`;
    });
    const inspect = preferences.mode === "inspect";
    required<HTMLElement>("#gesture-mode-help").textContent = inspect
      ? "单独伸出食指并停留选中节点；捏合节点可拖动，捏合空白可平移；双手捏合并改变距离可缩放。张开手掌不会改变视图。"
      : "张掌水平移动可旋转，张掌静止可持续放大；单手捏合可持续缩小。松手再切换动作，适合演示，不适合精细选点。";
    required<HTMLElement>("#gesture-guide").textContent = inspect
      ? "精准模式 · 食指停留选点 · 捏合拖动/平移 · 双手捏合缩放"
      : "演示模式 · 张掌移动旋转/静止放大 · 单手捏合缩小";
    this.updateGestureButton();
    this.updateCombinedInteractionState();
  }

  private saveGestureSettings(): void {
    const preferences = this.gesture.getPreferences();
    const mode = required<HTMLSelectElement>("#gesture-mode").value;
    const dominantHand = required<HTMLSelectElement>("#gesture-hand").value;
    const gain = Number(required<HTMLInputElement>("#gesture-sensitivity").value);
    this.gesture.setPreferences(normalizeGesturePreferences({
      ...preferences, mode, dominantHand, panGain: gain, zoomGain: gain, orbitGain: gain,
      dwellMs: Number(required<HTMLInputElement>("#gesture-dwell").value),
      smoothingMs: Number(required<HTMLInputElement>("#gesture-smoothing").value),
    }));
    this.persistGestureSettings();
    this.populateGestureSettings();
  }

  private persistGestureSettings(): void {
    try {
      localStorage.setItem(GESTURE_PREFERENCES_KEY, JSON.stringify(this.gesture.getPreferences()));
      required<HTMLElement>("#gesture-preference-status").textContent = "设置仅保存在此浏览器；不会保存摄像头开启状态。";
    } catch {
      required<HTMLElement>("#gesture-preference-status").textContent = "浏览器不允许保存，设置仅本次有效。";
    }
  }

  private updateGestureButton(): void {
    const state = this.gesture.getState();
    const label = state.starting ? "正在开启" : !state.enabled ? "摄像头关闭" : state.suspended ? "摄像头暂停" : "手势已开启";
    this.cameraStatus.textContent = state.error || (state.starting
      ? "等待摄像头授权或加载本地识别模型，可随时点击关闭取消。"
      : state.enabled && state.suspended ? "已暂停识别和画面采集。点击恢复继续；点击关闭释放摄像头。"
        : state.enabled ? "摄像头画面仅在本地处理。鼠标、触控、滚轮或图谱快捷键会接管并暂停手势；恢复需手动确认。"
          : "摄像头未开启。仅点击「开启摄像头」才请求权限；鼠标始终可用。" );
    this.cameraStatus.dataset.state = state.error ? "error" : state.layerState;
    required<HTMLElement>("#camera-indicator").textContent = label;
    this.gestureButton.title = `手势设置 · ${label}`;
    this.cameraStart.disabled = state.enabled || state.starting || !this.graphReady;
    this.cameraStart.title = this.graphReady ? "仅此操作请求摄像头权限" : "图谱加载完成后可开启";
    this.cameraPause.disabled = !state.enabled;
    this.cameraPause.textContent = state.suspended ? "恢复" : "暂停";
    this.cameraStop.disabled = !state.enabled && !state.starting;
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
    const gestureState = this.gesture?.getState();
    const preciseControl = gestureState?.enabled && !gestureState.suspended && this.gesture.getPreferences().mode === "inspect";
    this.autoRotation.setOrbitInteracting(this.gestureDialog.open || this.mouseOrbitInteracting || this.gestureInteracting || Boolean(preciseControl));
    this.autoRotation.setNodeDragging(this.mouseNodeDragging || this.gestureNodeDragging);
  }

  private onRootPointerDown = () => this.autoRotation.markInteraction();
  private onWindowBlur = () => {
    // A browser permission prompt may itself blur the window while starting.
    if (this.gesture.getState().enabled) this.gesture.setSuspended(true);
  };

  private takeManualControl(): void {
    if (pauseGestureForManualInput(this.gesture)) this.setStatus("手动操作已接管", "手势已暂停，可在手势设置中恢复");
  }

  private onVisibilityChange = () => {
    const visible = !document.hidden;
    this.autoRotation.setDocumentVisible(visible);
    // Returning to this tab never silently resumes camera capture.
    if (!visible) { this.gesture.setSuspended(true); this.input.resetInteraction(); }
    if (visible) this.renderer.start();
    else this.renderer.stop();
    this.updateRotationButton();
  };

  private onPageHide = (event: PageTransitionEvent) => {
    this.input.resetInteraction();
    this.gesture.stop();
    this.renderer.stop();
    if (!event.persisted) this.dispose();
  };

  private onPageShow = () => {
    if (!this.disposed && !document.hidden) this.renderer.start();
  };

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    window.clearInterval(this.statsTimer);
    window.clearTimeout(this.eclrrPollTimer);
    this.graphAbort?.abort();
    this.chunksAbort?.abort();
    document.removeEventListener("visibilitychange", this.onVisibilityChange);
    window.removeEventListener("pagehide", this.onPageHide);
    window.removeEventListener("pageshow", this.onPageShow);
    window.removeEventListener("blur", this.onWindowBlur);
    this.headerObserver?.disconnect();
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
