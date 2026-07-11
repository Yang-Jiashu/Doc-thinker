import * as THREE from "three";
import { neighborsOf } from "./graph-data";
import { GpuLabelLayer } from "./gpu-label-layer";
import {
  resolveLabelCollisions,
  selectLabelCandidates,
  SemanticZoomPolicy,
  type ScreenNode,
} from "./semantic-labels";
import type { GraphModel } from "./types";
import { computeVisualDepths } from "./visual-depth";

const NODE_VERTEX_SHADER = `
  attribute vec3 aColor;
  attribute float aSize;
  attribute float aState;
  uniform float uPixelRatio;
  uniform float uSemanticScale;
  varying vec3 vColor;
  varying float vState;
  void main() {
    vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * viewPosition;
    float stateScale = aState > 2.5 ? 1.7 : aState > 1.5 ? 1.28 : aState > 0.5 ? 0.88 : 1.0;
    gl_PointSize = clamp(aSize * uPixelRatio * uSemanticScale * stateScale, 2.0, 34.0);
    vColor = aColor;
    vState = aState;
  }
`;

const NODE_FRAGMENT_SHADER = `
  varying vec3 vColor;
  varying float vState;
  void main() {
    float distanceFromCenter = length(gl_PointCoord - vec2(0.5));
    float aa = max(fwidth(distanceFromCenter), 0.006);
    float disc = 1.0 - smoothstep(0.49 - aa, 0.49 + aa, distanceFromCenter);
    if (disc <= 0.0) discard;
    float alpha = vState > 0.5 && vState < 1.5 ? 0.18 : 0.96;
    vec3 color = vColor;
    if (vState > 2.5) {
      float inner = 1.0 - smoothstep(0.34 - aa, 0.34 + aa, distanceFromCenter);
      float ring = smoothstep(0.35 - aa, 0.35 + aa, distanceFromCenter) * disc;
      color = mix(vec3(0.94, 1.0, 1.0), vec3(0.39, 0.94, 1.0), ring);
      alpha = max(inner, ring);
    } else if (vState > 1.5) {
      color = mix(vColor, vec3(0.83, 1.0, 0.98), 0.42);
    }
    gl_FragColor = vec4(color, alpha * disc);
  }
`;

const PICK_VERTEX_SHADER = `
  attribute vec3 aPickColor;
  attribute float aSize;
  uniform float uPixelRatio;
  varying vec3 vPickColor;
  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = max(14.0, aSize * uPixelRatio * 1.5);
    vPickColor = aPickColor;
  }
`;

const PICK_FRAGMENT_SHADER = `
  varying vec3 vPickColor;
  void main() {
    if (length(gl_PointCoord - vec2(0.5)) > 0.5) discard;
    gl_FragColor = vec4(vPickColor, 1.0);
  }
`;

const EDGE_VERTEX_SHADER = `
  attribute vec3 aColor;
  attribute float aState;
  attribute float aLineT;
  attribute float aDashed;
  varying vec3 vColor;
  varying float vState;
  varying float vLineT;
  varying float vDashed;
  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    vColor = aColor;
    vState = aState;
    vLineT = aLineT;
    vDashed = aDashed;
  }
`;

const EDGE_FRAGMENT_SHADER = `
  varying vec3 vColor;
  varying float vState;
  varying float vLineT;
  varying float vDashed;
  void main() {
    if (vDashed > 0.5 && mod(vLineT * 22.0, 1.0) > 0.58) discard;
    float alpha = vState > 2.5 ? 0.98 : vState > 1.5 ? 0.78 : vState > 0.5 ? 0.025 : 0.105;
    vec3 color = vState > 2.5 ? mix(vColor, vec3(0.94, 1.0, 1.0), 0.72) : vColor;
    gl_FragColor = vec4(color, alpha);
  }
`;

interface FocusAnimation {
  indices: Uint32Array;
  starts: Float32Array;
  targets: Float32Array;
  startedAt: number;
  duration: number;
  clearing: boolean;
}

export interface RenderStats {
  fps: number;
  calls: number;
  points: number;
  lines: number;
  triangles: number;
  labels: number;
  glyphs: number;
  zoomRatio: number;
  labelLevel: number;
  renderer: string;
}

export interface CameraOrbitState {
  position: [number, number, number];
  target: [number, number, number];
  zoom: number;
  zoomRatio: number;
  azimuth: number;
  polar: number;
}

export class StarMapRenderer {
  readonly canvas: HTMLCanvasElement;
  private host: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private pickingScene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 2000);
  private orbitTarget = new THREE.Vector3();
  private orbitRadius = 1_000;
  private orbitAzimuth = 0;
  private orbitPolar = 0;
  private readonly automaticOrbitPolar = THREE.MathUtils.degToRad(13);
  private raycaster = new THREE.Raycaster();
  private viewPlane = new THREE.Plane();
  private resolution = new THREE.Vector2(1, 1);
  private labels: GpuLabelLayer;
  private model: GraphModel | null = null;
  private basePositions = new Float32Array();
  private renderPositions = new Float32Array();
  private nodePoints: THREE.Points | null = null;
  private pickingPoints: THREE.Points | null = null;
  private edgeLines: THREE.LineSegments | null = null;
  private pickingTarget = new THREE.WebGLRenderTarget(1, 1, {
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: true,
    stencilBuffer: false,
  });
  private width = 1;
  private height = 1;
  private fitZoom = 1;
  private selectedIndex = -1;
  private selectedEdgeIndex = -1;
  private hoveredIndex = -1;
  private searchIndex = -1;
  private focusIndices = new Uint32Array();
  private focusAnimation: FocusAnimation | null = null;
  private focusAnchorIndex = -1;
  private labelPolicy = new SemanticZoomPolicy();
  private labelsDirty = true;
  private lastLabelUpdate = 0;
  private running = false;
  private animationFrame = 0;
  private lastFrameAt = performance.now();
  private fps = 60;
  private frameSamples: number[] = [];
  private viewChangedCallback: (() => void) | null = null;
  private frameCallback: ((now: number, deltaSeconds: number) => void) | null = null;
  private rendererName = "WebGL2";

  constructor(host: HTMLElement) {
    this.host = host;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.setClearColor(0x05070d, 1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    this.canvas = this.renderer.domElement;
    const context = this.renderer.getContext();
    const debugInfo = context.getExtension("WEBGL_debug_renderer_info");
    this.rendererName = debugInfo
      ? String(context.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL))
      : String(context.getParameter(context.RENDERER));
    this.canvas.className = "star-map-canvas";
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute("role", "application");
    this.canvas.setAttribute("aria-label", "二维语义缩放知识星图。使用滚轮缩放，拖动画布平移，点击节点或关系边查看证据。");
    host.replaceChildren(this.canvas);
    this.applyCameraOrbit();
    this.labels = new GpuLabelLayer(this.scene, this.resolution);
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
  }

  private disposeGraphObjects(): void {
    const disposedGeometries = new Set<THREE.BufferGeometry>();
    [this.nodePoints, this.pickingPoints, this.edgeLines].forEach(object => {
      if (!object) return;
      object.parent?.remove(object);
      if (!disposedGeometries.has(object.geometry)) {
        object.geometry.dispose();
        disposedGeometries.add(object.geometry);
      }
      if (Array.isArray(object.material)) object.material.forEach(material => material.dispose());
      else object.material.dispose();
    });
    this.nodePoints = null;
    this.pickingPoints = null;
    this.edgeLines = null;
  }

  setGraph(model: GraphModel, positions: Float32Array): void {
    this.disposeGraphObjects();
    this.model = model;
    this.basePositions = positions.slice();
    this.renderPositions = positions.slice();
    this.focusIndices = new Uint32Array();
    this.focusAnimation = null;
    this.focusAnchorIndex = -1;
    const visualDepth = computeVisualDepths(this.basePositions, model.hashes);
    for (let index = 0; index < model.nodes.length; index += 1) {
      this.renderPositions[index * 3 + 2] = visualDepth.depths[index];
    }

    const nodeGeometry = new THREE.BufferGeometry();
    nodeGeometry.setAttribute("position", new THREE.BufferAttribute(this.renderPositions, 3).setUsage(THREE.DynamicDrawUsage));
    nodeGeometry.setAttribute("aColor", new THREE.BufferAttribute(model.colors, 3));
    nodeGeometry.setAttribute("aSize", new THREE.BufferAttribute(model.sizes, 1));
    nodeGeometry.setAttribute("aState", new THREE.BufferAttribute(new Float32Array(model.nodes.length), 1).setUsage(THREE.DynamicDrawUsage));
    const pickColors = new Float32Array(model.nodes.length * 3);
    for (let index = 0; index < model.nodes.length; index += 1) {
      const encoded = index + 1;
      pickColors[index * 3] = (encoded & 0xff) / 255;
      pickColors[index * 3 + 1] = ((encoded >>> 8) & 0xff) / 255;
      pickColors[index * 3 + 2] = ((encoded >>> 16) & 0xff) / 255;
    }
    nodeGeometry.setAttribute("aPickColor", new THREE.BufferAttribute(pickColors, 3));

    const nodeMaterial = new THREE.ShaderMaterial({
      vertexShader: NODE_VERTEX_SHADER,
      fragmentShader: NODE_FRAGMENT_SHADER,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        uPixelRatio: { value: this.renderer.getPixelRatio() },
        uSemanticScale: { value: 1 },
      },
    });
    this.nodePoints = new THREE.Points(nodeGeometry, nodeMaterial);
    this.nodePoints.frustumCulled = false;
    this.nodePoints.renderOrder = 5;
    this.scene.add(this.nodePoints);

    const pickingMaterial = new THREE.ShaderMaterial({
      vertexShader: PICK_VERTEX_SHADER,
      fragmentShader: PICK_FRAGMENT_SHADER,
      depthTest: false,
      depthWrite: false,
      uniforms: { uPixelRatio: { value: this.renderer.getPixelRatio() } },
    });
    this.pickingPoints = new THREE.Points(nodeGeometry, pickingMaterial);
    this.pickingPoints.frustumCulled = false;
    this.pickingScene.add(this.pickingPoints);

    const edgePositions = new Float32Array(model.edges.length * 6);
    const edgeColors = new Float32Array(model.edges.length * 6);
    const edgeStates = new Float32Array(model.edges.length * 2);
    const edgeLineT = new Float32Array(model.edges.length * 2);
    const edgeDashed = new Float32Array(model.edges.length * 2);
    model.edges.forEach((edge, index) => {
      const color = edge.isPromoted ? [0.73, 0.58, 1] : [0.54, 0.69, 0.8];
      edgeColors.set(color, index * 6);
      edgeColors.set(color, index * 6 + 3);
      edgeLineT[index * 2] = 0;
      edgeLineT[index * 2 + 1] = 1;
      edgeDashed[index * 2] = edge.isPromoted ? 1 : 0;
      edgeDashed[index * 2 + 1] = edge.isPromoted ? 1 : 0;
    });
    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute("position", new THREE.BufferAttribute(edgePositions, 3).setUsage(THREE.DynamicDrawUsage));
    edgeGeometry.setAttribute("aColor", new THREE.BufferAttribute(edgeColors, 3));
    edgeGeometry.setAttribute("aState", new THREE.BufferAttribute(edgeStates, 1).setUsage(THREE.DynamicDrawUsage));
    edgeGeometry.setAttribute("aLineT", new THREE.BufferAttribute(edgeLineT, 1));
    edgeGeometry.setAttribute("aDashed", new THREE.BufferAttribute(edgeDashed, 1));
    const edgeMaterial = new THREE.ShaderMaterial({
      vertexShader: EDGE_VERTEX_SHADER,
      fragmentShader: EDGE_FRAGMENT_SHADER,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    this.edgeLines = new THREE.LineSegments(edgeGeometry, edgeMaterial);
    this.edgeLines.frustumCulled = false;
    this.edgeLines.renderOrder = 1;
    this.scene.add(this.edgeLines);
    this.syncEdgePositions();
    this.clearSelection(false);
    this.fitToGraph(false);
    this.labelsDirty = true;
    this.frameSamples = [];
    this.fps = 60;
    this.lastFrameAt = performance.now();
  }

  private syncEdgePositions(): void {
    if (!this.model || !this.edgeLines) return;
    const attribute = this.edgeLines.geometry.getAttribute("position") as THREE.BufferAttribute;
    const values = attribute.array as Float32Array;
    this.model.edges.forEach((edge, index) => {
      values[index * 6] = this.renderPositions[edge.source * 3];
      values[index * 6 + 1] = this.renderPositions[edge.source * 3 + 1];
      values[index * 6 + 2] = this.renderPositions[edge.source * 3 + 2];
      values[index * 6 + 3] = this.renderPositions[edge.target * 3];
      values[index * 6 + 4] = this.renderPositions[edge.target * 3 + 1];
      values[index * 6 + 5] = this.renderPositions[edge.target * 3 + 2];
    });
    attribute.needsUpdate = true;
  }

  private syncNodePositions(): void {
    if (!this.nodePoints) return;
    (this.nodePoints.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
    this.syncEdgePositions();
    this.labelsDirty = true;
  }

  setSelection(nodeIndex: number): void {
    if (!this.model || !this.nodePoints || !this.edgeLines) return;
    this.selectedIndex = nodeIndex;
    this.selectedEdgeIndex = -1;
    this.refreshSelectionStates();
  }

  setEdgeSelection(edgeIndex: number): void {
    if (!this.model || !this.nodePoints || !this.edgeLines) return;
    this.selectedIndex = -1;
    this.selectedEdgeIndex = edgeIndex >= 0 && edgeIndex < this.model.edges.length ? edgeIndex : -1;
    this.refreshSelectionStates();
  }

  private refreshSelectionStates(): void {
    if (!this.model || !this.nodePoints || !this.edgeLines) return;
    const nodeIndex = this.selectedIndex;
    const edgeIndex = this.selectedEdgeIndex;
    const neighbors = nodeIndex >= 0 ? new Set(neighborsOf(this.model, nodeIndex)) : new Set<number>();
    const selectedEdge = edgeIndex >= 0 ? this.model.edges[edgeIndex] : undefined;
    const nodeStates = (this.nodePoints.geometry.getAttribute("aState") as THREE.BufferAttribute).array as Float32Array;
    for (let index = 0; index < nodeStates.length; index += 1) {
      if (nodeIndex >= 0) nodeStates[index] = index === nodeIndex ? 3 : neighbors.has(index) ? 2 : 1;
      else if (selectedEdge) nodeStates[index] = index === selectedEdge.source || index === selectedEdge.target ? 3 : 1;
      else nodeStates[index] = 0;
    }
    if (this.hoveredIndex >= 0 && this.hoveredIndex !== nodeIndex) nodeStates[this.hoveredIndex] = 4;
    (this.nodePoints.geometry.getAttribute("aState") as THREE.BufferAttribute).needsUpdate = true;
    const edgeStates = (this.edgeLines.geometry.getAttribute("aState") as THREE.BufferAttribute).array as Float32Array;
    this.model.edges.forEach((edge, index) => {
      const state = nodeIndex >= 0
        ? edge.source === nodeIndex || edge.target === nodeIndex ? 2 : 1
        : edgeIndex >= 0 ? index === edgeIndex ? 3 : 1 : 0;
      edgeStates[index * 2] = state;
      edgeStates[index * 2 + 1] = state;
    });
    (this.edgeLines.geometry.getAttribute("aState") as THREE.BufferAttribute).needsUpdate = true;
    this.labelsDirty = true;
  }

  clearSelection(animate = true): void {
    this.selectedIndex = -1;
    this.selectedEdgeIndex = -1;
    this.refreshSelectionStates();
    if (animate) this.clearFocus(280);
  }

  setHovered(nodeIndex: number): void {
    if (!this.model || !this.nodePoints || nodeIndex === this.hoveredIndex) return;
    this.hoveredIndex = nodeIndex;
    this.refreshSelectionStates();
  }

  setSearchMatch(nodeIndex: number): void {
    this.searchIndex = nodeIndex;
    this.labelsDirty = true;
  }

  beginFocus(nodeIndex: number): void {
    if (nodeIndex < 0 || nodeIndex >= this.basePositions.length / 3) return;
    this.focusAnchorIndex = nodeIndex;
    this.focusIndices = Uint32Array.of(nodeIndex);
    this.focusAnimation = null;
  }

  animateFocus(indices: Uint32Array, targetPositions: Float32Array, duration = 350): void {
    const union = new Set<number>(this.focusIndices);
    indices.forEach(index => union.add(index));
    const allIndices = Uint32Array.from(union);
    const starts = new Float32Array(allIndices.length * 2);
    const targets = new Float32Array(allIndices.length * 2);
    const targetMap = new Map<number, [number, number]>();
    const anchor = this.focusAnchorIndex;
    const anchorDx = anchor >= 0 ? this.renderPositions[anchor * 3] - this.basePositions[anchor * 3] : 0;
    const anchorDy = anchor >= 0 ? this.renderPositions[anchor * 3 + 1] - this.basePositions[anchor * 3 + 1] : 0;
    indices.forEach((index, offset) => targetMap.set(index, [
      targetPositions[offset * 2] + anchorDx,
      targetPositions[offset * 2 + 1] + anchorDy,
    ]));
    allIndices.forEach((index, offset) => {
      starts[offset * 2] = this.renderPositions[index * 3];
      starts[offset * 2 + 1] = this.renderPositions[index * 3 + 1];
      const target = targetMap.get(index);
      targets[offset * 2] = target?.[0] ?? this.basePositions[index * 3];
      targets[offset * 2 + 1] = target?.[1] ?? this.basePositions[index * 3 + 1];
    });
    this.focusIndices = indices.slice();
    this.focusAnimation = { indices: allIndices, starts, targets, startedAt: performance.now(), duration, clearing: false };
  }

  clearFocus(duration = 280): void {
    if (!this.focusIndices.length) return;
    const starts = new Float32Array(this.focusIndices.length * 2);
    const targets = new Float32Array(this.focusIndices.length * 2);
    this.focusIndices.forEach((index, offset) => {
      starts[offset * 2] = this.renderPositions[index * 3];
      starts[offset * 2 + 1] = this.renderPositions[index * 3 + 1];
      targets[offset * 2] = this.basePositions[index * 3];
      targets[offset * 2 + 1] = this.basePositions[index * 3 + 1];
    });
    this.focusAnimation = {
      indices: this.focusIndices.slice(), starts, targets,
      startedAt: performance.now(), duration, clearing: true,
    };
  }

  setNodeTransientPosition(nodeIndex: number, x: number, y: number): void {
    if (nodeIndex < 0 || nodeIndex >= this.renderPositions.length / 3) return;
    const dx = x - this.renderPositions[nodeIndex * 3];
    const dy = y - this.renderPositions[nodeIndex * 3 + 1];
    if (!Number.isFinite(dx) || !Number.isFinite(dy)) return;

    const affected = this.focusIndices.length ? this.focusIndices : Uint32Array.of(nodeIndex);
    affected.forEach(index => {
      this.renderPositions[index * 3] += dx;
      this.renderPositions[index * 3 + 1] += dy;
    });
    this.renderPositions[nodeIndex * 3] = x;
    this.renderPositions[nodeIndex * 3 + 1] = y;

    if (this.focusAnimation && !this.focusAnimation.clearing) {
      for (let offset = 0; offset < this.focusAnimation.indices.length; offset += 1) {
        this.focusAnimation.starts[offset * 2] += dx;
        this.focusAnimation.starts[offset * 2 + 1] += dy;
        this.focusAnimation.targets[offset * 2] += dx;
        this.focusAnimation.targets[offset * 2 + 1] += dy;
      }
    }
    this.syncNodePositions();
  }

  setNodeWorldPosition(nodeIndex: number, x: number, y: number): void {
    this.basePositions[nodeIndex * 3] = x;
    this.basePositions[nodeIndex * 3 + 1] = y;
    this.renderPositions[nodeIndex * 3] = x;
    this.renderPositions[nodeIndex * 3 + 1] = y;
    this.syncNodePositions();
  }

  getNodeWorldPosition(nodeIndex: number): { x: number; y: number; z: number } {
    return {
      x: this.renderPositions[nodeIndex * 3],
      y: this.renderPositions[nodeIndex * 3 + 1],
      z: this.renderPositions[nodeIndex * 3 + 2],
    };
  }

  fitToGraph(notify = true): void {
    if (!this.model?.nodes.length) return;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let index = 0; index < this.basePositions.length; index += 3) {
      minX = Math.min(minX, this.basePositions[index]);
      maxX = Math.max(maxX, this.basePositions[index]);
      minY = Math.min(minY, this.basePositions[index + 1]);
      maxY = Math.max(maxY, this.basePositions[index + 1]);
    }
    this.orbitTarget.set((minX + maxX) / 2, (minY + maxY) / 2, 0);
    this.orbitAzimuth = 0;
    this.orbitPolar = this.automaticOrbitPolar;
    const availableWidth = Math.max(160, this.width - 96);
    const availableHeight = Math.max(160, this.height - 112);
    this.camera.zoom = Math.max(0.02, Math.min(availableWidth / Math.max(160, maxX - minX), availableHeight / Math.max(160, maxY - minY)));
    this.fitZoom = this.camera.zoom;
    this.camera.updateProjectionMatrix();
    this.applyCameraOrbit();
    this.labelPolicy.reset(1);
    this.labelsDirty = true;
    if (notify) this.viewChangedCallback?.();
  }

  zoomAt(screenX: number, screenY: number, factor: number): void {
    const before = this.screenToWorld(screenX, screenY, this.orbitTarget);
    this.camera.zoom = THREE.MathUtils.clamp(this.camera.zoom * factor, this.fitZoom * 0.28, this.fitZoom * 20);
    this.camera.updateProjectionMatrix();
    const after = this.screenToWorld(screenX, screenY, this.orbitTarget);
    const shift = new THREE.Vector3(before.x - after.x, before.y - after.y, before.z - after.z);
    this.orbitTarget.add(shift);
    this.camera.position.add(shift);
    this.labelsDirty = true;
    this.viewChangedCallback?.();
  }

  panBy(screenDx: number, screenDy: number): void {
    this.camera.updateMatrixWorld();
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 1);
    const shift = right.multiplyScalar(-screenDx / this.camera.zoom).add(up.multiplyScalar(screenDy / this.camera.zoom));
    this.orbitTarget.add(shift);
    this.camera.position.add(shift);
    this.labelsDirty = true;
    this.viewChangedCallback?.();
  }

  centerOnNode(nodeIndex: number, minimumRatio = 2.1): void {
    const position = this.getNodeWorldPosition(nodeIndex);
    this.orbitTarget.set(position.x, position.y, position.z);
    this.applyCameraOrbit();
    if (this.zoomRatio < minimumRatio) this.camera.zoom = this.fitZoom * minimumRatio;
    this.camera.updateProjectionMatrix();
    this.labelsDirty = true;
    this.viewChangedCallback?.();
  }

  screenToWorld(
    screenX: number,
    screenY: number,
    planeAnchor: { x: number; y: number; z: number } = this.orbitTarget,
    planeNormal?: { x: number; y: number; z: number },
  ): { x: number; y: number; z: number } {
    const pointer = new THREE.Vector2(screenX / this.width * 2 - 1, -(screenY / this.height) * 2 + 1);
    this.camera.updateMatrixWorld();
    this.raycaster.setFromCamera(pointer, this.camera);
    const normal = planeNormal
      ? new THREE.Vector3(planeNormal.x, planeNormal.y, planeNormal.z).normalize()
      : this.camera.getWorldDirection(new THREE.Vector3());
    this.viewPlane.setFromNormalAndCoplanarPoint(normal, new THREE.Vector3(planeAnchor.x, planeAnchor.y, planeAnchor.z));
    const point = this.raycaster.ray.intersectPlane(this.viewPlane, new THREE.Vector3());
    return point
      ? { x: point.x, y: point.y, z: point.z }
      : { x: planeAnchor.x, y: planeAnchor.y, z: planeAnchor.z };
  }

  projectNode(nodeIndex: number): { x: number; y: number; visible: boolean } {
    const point = new THREE.Vector3(
      this.renderPositions[nodeIndex * 3],
      this.renderPositions[nodeIndex * 3 + 1],
      this.renderPositions[nodeIndex * 3 + 2],
    ).project(this.camera);
    return {
      x: (point.x + 1) * this.width / 2,
      y: (1 - point.y) * this.height / 2,
      visible: Math.abs(point.x) <= 1.08 && Math.abs(point.y) <= 1.08,
    };
  }

  pick(screenX: number, screenY: number): number {
    if (!this.pickingPoints || !this.model || screenX < 0 || screenY < 0 || screenX > this.width || screenY > this.height) return -1;
    const pixel = new Uint8Array(4);
    this.camera.setViewOffset(this.width, this.height, Math.floor(screenX), Math.floor(screenY), 1, 1);
    this.renderer.setRenderTarget(this.pickingTarget);
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.clear();
    this.renderer.render(this.pickingScene, this.camera);
    this.renderer.readRenderTargetPixels(this.pickingTarget, 0, 0, 1, 1, pixel);
    this.renderer.setRenderTarget(null);
    this.renderer.setClearColor(0x05070d, 1);
    this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
    const decoded = pixel[0] + (pixel[1] << 8) + (pixel[2] << 16) - 1;
    return decoded >= 0 && decoded < this.model.nodes.length ? decoded : -1;
  }

  pickEdge(screenX: number, screenY: number, tolerance = 7): number {
    if (!this.model || screenX < 0 || screenY < 0 || screenX > this.width || screenY > this.height) return -1;
    let nearestIndex = -1;
    let nearestDistanceSquared = tolerance * tolerance;
    this.model.edges.forEach((edge, edgeIndex) => {
      const source = this.projectNode(edge.source);
      const target = this.projectNode(edge.target);
      const minX = Math.min(source.x, target.x) - tolerance;
      const maxX = Math.max(source.x, target.x) + tolerance;
      const minY = Math.min(source.y, target.y) - tolerance;
      const maxY = Math.max(source.y, target.y) + tolerance;
      if (screenX < minX || screenX > maxX || screenY < minY || screenY > maxY) return;
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const lengthSquared = dx * dx + dy * dy;
      if (lengthSquared < 1) return;
      const ratio = THREE.MathUtils.clamp(((screenX - source.x) * dx + (screenY - source.y) * dy) / lengthSquared, 0, 1);
      const projectedX = source.x + dx * ratio;
      const projectedY = source.y + dy * ratio;
      const distanceSquared = (screenX - projectedX) ** 2 + (screenY - projectedY) ** 2;
      if (distanceSquared <= nearestDistanceSquared) {
        nearestDistanceSquared = distanceSquared;
        nearestIndex = edgeIndex;
      }
    });
    return nearestIndex;
  }

  getEdgeScreenMidpoint(edgeIndex: number): { x: number; y: number } | null {
    const edge = this.model?.edges[edgeIndex];
    if (!edge) return null;
    const source = this.projectNode(edge.source);
    const target = this.projectNode(edge.target);
    return { x: (source.x + target.x) / 2, y: (source.y + target.y) / 2 };
  }

  private refreshLabels(now: number): void {
    if (!this.model || !this.labelsDirty || now - this.lastLabelUpdate < 70) return;
    const level = this.labelPolicy.update(this.zoomRatio);
    const screenNodes: ScreenNode[] = this.model.nodes.map((_, nodeIndex) => ({ nodeIndex, ...this.projectNode(nodeIndex) }));
    const selectedEdge = this.selectedEdgeIndex >= 0 ? this.model.edges[this.selectedEdgeIndex] : undefined;
    const special = new Set([
      this.selectedIndex,
      selectedEdge?.source ?? -1,
      selectedEdge?.target ?? -1,
      this.hoveredIndex,
      this.searchIndex,
    ].filter(index => index >= 0));
    const candidates = selectLabelCandidates(this.model, screenNodes, level, special);
    const placements = resolveLabelCollisions(candidates, this.width, this.height);
    this.labels.update(placements, this.renderPositions);
    this.labelsDirty = false;
    this.lastLabelUpdate = now;
  }

  private tickFocus(now: number): void {
    const animation = this.focusAnimation;
    if (!animation) return;
    const raw = Math.min(1, (now - animation.startedAt) / animation.duration);
    const eased = 1 - Math.pow(1 - raw, 3);
    animation.indices.forEach((index, offset) => {
      this.renderPositions[index * 3] = THREE.MathUtils.lerp(animation.starts[offset * 2], animation.targets[offset * 2], eased);
      this.renderPositions[index * 3 + 1] = THREE.MathUtils.lerp(animation.starts[offset * 2 + 1], animation.targets[offset * 2 + 1], eased);
    });
    this.syncNodePositions();
    if (raw >= 1) {
      if (animation.clearing) {
        this.focusIndices = new Uint32Array();
        this.focusAnchorIndex = -1;
      }
      this.focusAnimation = null;
    }
  }

  private frame = (now: number) => {
    if (!this.running) return;
    const frameDelta = Math.max(1, now - this.lastFrameAt);
    this.frameCallback?.(now, Math.min(0.1, frameDelta / 1_000));
    this.tickFocus(now);
    const ratio = this.zoomRatio;
    if (this.nodePoints) {
      const material = this.nodePoints.material as THREE.ShaderMaterial;
      material.uniforms.uSemanticScale.value = THREE.MathUtils.clamp(0.8 + Math.log2(Math.max(0.35, ratio)) * 0.14, 0.65, 1.65);
    }
    this.refreshLabels(now);
    this.renderer.render(this.scene, this.camera);
    if (frameDelta < 250) {
      this.frameSamples.push(frameDelta);
      if (this.frameSamples.length > 60) this.frameSamples.shift();
      const average = this.frameSamples.reduce((sum, value) => sum + value, 0) / this.frameSamples.length;
      this.fps = 1000 / Math.max(1, average);
    }
    this.lastFrameAt = now;
    this.animationFrame = requestAnimationFrame(this.frame);
  };

  start(): void {
    if (this.running) return;
    this.running = true;
    this.lastFrameAt = performance.now();
    this.animationFrame = requestAnimationFrame(this.frame);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.animationFrame);
  }

  resize(): void {
    const bounds = this.host.getBoundingClientRect();
    this.width = Math.max(1, Math.round(bounds.width));
    this.height = Math.max(1, Math.round(bounds.height));
    this.camera.left = -this.width / 2;
    this.camera.right = this.width / 2;
    this.camera.top = this.height / 2;
    this.camera.bottom = -this.height / 2;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.width, this.height, false);
    this.resolution.set(this.width, this.height);
    this.labels.resize(this.width, this.height);
    this.labelsDirty = true;
  }

  onViewChanged(callback: () => void): void {
    this.viewChangedCallback = callback;
  }

  onFrame(callback: (now: number, deltaSeconds: number) => void): void {
    this.frameCallback = callback;
  }

  advanceAutomaticOrbit(deltaSeconds: number, angularSpeedRadPerSecond: number): void {
    const tiltBlend = 1 - Math.exp(-Math.max(0, deltaSeconds) * 2.6);
    this.orbitPolar = THREE.MathUtils.lerp(this.orbitPolar, this.automaticOrbitPolar, tiltBlend);
    this.orbitAzimuth = (this.orbitAzimuth + angularSpeedRadPerSecond * deltaSeconds) % (Math.PI * 2);
    this.applyCameraOrbit();
    this.labelsDirty = true;
  }

  getCameraOrbitState(): CameraOrbitState {
    return {
      position: this.camera.position.toArray() as [number, number, number],
      target: this.orbitTarget.toArray() as [number, number, number],
      zoom: this.camera.zoom,
      zoomRatio: this.zoomRatio,
      azimuth: this.orbitAzimuth,
      polar: this.orbitPolar,
    };
  }

  private applyCameraOrbit(): void {
    const sinPolar = Math.sin(this.orbitPolar);
    const offset = new THREE.Vector3(
      this.orbitRadius * sinPolar * Math.cos(this.orbitAzimuth),
      this.orbitRadius * sinPolar * Math.sin(this.orbitAzimuth),
      this.orbitRadius * Math.cos(this.orbitPolar),
    );
    this.camera.position.copy(this.orbitTarget).add(offset);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(this.orbitTarget);
    this.camera.updateMatrixWorld();
  }

  get zoomRatio(): number {
    return this.fitZoom > 0 ? this.camera.zoom / this.fitZoom : 1;
  }

  get selectedNodeIndex(): number {
    return this.selectedIndex;
  }

  get selectedRelationIndex(): number {
    return this.selectedEdgeIndex;
  }

  getStats(): RenderStats {
    return {
      fps: this.fps,
      calls: this.renderer.info.render.calls,
      points: this.renderer.info.render.points,
      lines: this.renderer.info.render.lines,
      triangles: this.renderer.info.render.triangles,
      labels: this.labels.visibleLabelCount,
      glyphs: this.labels.visibleGlyphCount,
      zoomRatio: this.zoomRatio,
      labelLevel: this.labelPolicy.currentLevel,
      renderer: this.rendererName,
    };
  }

  dispose(): void {
    this.stop();
    this.disposeGraphObjects();
    this.labels.dispose();
    this.pickingTarget.dispose();
    this.renderer.dispose();
  }
}
