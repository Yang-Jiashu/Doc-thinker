import * as THREE from "three";
import type { LabelPlacement } from "./types";

interface GlyphInfo {
  page: number;
  uv: [number, number, number, number];
  advance: number;
}

interface AtlasPage {
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  texture: THREE.CanvasTexture;
  material: THREE.ShaderMaterial;
  mesh: THREE.Points | null;
  count: number;
}

const ATLAS_SIZE = 2048;
const CELL_SIZE = 64;
const CELLS_PER_ROW = ATLAS_SIZE / CELL_SIZE;
const CELLS_PER_PAGE = CELLS_PER_ROW * CELLS_PER_ROW;
const ATLAS_FONT_SIZE = 40;
const DISPLAY_FONT_SIZE = 13;

const LABEL_VERTEX_SHADER = `
  attribute vec2 aOffset;
  attribute float aSize;
  attribute vec4 aUvRect;
  attribute vec3 aColor;
  attribute float aOpacity;
  uniform vec2 uResolution;
  uniform float uPixelRatio;
  varying vec4 vUvRect;
  varying vec3 vColor;
  varying float vOpacity;
  void main() {
    vec4 clip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    clip.xy += vec2(aOffset.x, -aOffset.y) * 2.0 / uResolution * clip.w;
    gl_Position = clip;
    gl_PointSize = aSize * uPixelRatio;
    vUvRect = aUvRect;
    vColor = aColor;
    vOpacity = aOpacity;
  }
`;

const LABEL_FRAGMENT_SHADER = `
  uniform sampler2D uAtlas;
  varying vec4 vUvRect;
  varying vec3 vColor;
  varying float vOpacity;
  void main() {
    vec2 glyphUv = mix(vUvRect.xy, vUvRect.zw, vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y));
    float alpha = texture2D(uAtlas, glyphUv).a;
    alpha = smoothstep(0.08, 0.62, alpha) * vOpacity;
    if (alpha < 0.02) discard;
    gl_FragColor = vec4(vColor, alpha);
  }
`;

export class GpuLabelLayer {
  private scene: THREE.Scene;
  private resolution: THREE.Vector2;
  private glyphs = new Map<string, GlyphInfo>();
  private pages: AtlasPage[] = [];
  visibleLabelCount = 0;
  visibleGlyphCount = 0;

  constructor(scene: THREE.Scene, resolution: THREE.Vector2) {
    this.scene = scene;
    this.resolution = resolution;
  }

  private addPage(): AtlasPage {
    const canvas = document.createElement("canvas");
    canvas.width = ATLAS_SIZE;
    canvas.height = ATLAS_SIZE;
    const context = canvas.getContext("2d", { alpha: true });
    if (!context) throw new Error("Canvas 2D is required for graph labels");
    context.clearRect(0, 0, ATLAS_SIZE, ATLAS_SIZE);
    context.fillStyle = "#ffffff";
    context.font = `600 ${ATLAS_FONT_SIZE}px Inter, "Microsoft YaHei", "Noto Sans SC", sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.generateMipmaps = false;
    const material = new THREE.ShaderMaterial({
      vertexShader: LABEL_VERTEX_SHADER,
      fragmentShader: LABEL_FRAGMENT_SHADER,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        uAtlas: { value: texture },
        uResolution: { value: this.resolution },
        uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 1.75) },
      },
    });
    const page = { canvas, context, texture, material, mesh: null, count: 0 };
    this.pages.push(page);
    return page;
  }

  private ensureGlyph(character: string): GlyphInfo {
    const existing = this.glyphs.get(character);
    if (existing) return existing;
    let page = this.pages[this.pages.length - 1];
    if (!page || page.count >= CELLS_PER_PAGE) page = this.addPage();
    const index = page.count++;
    const column = index % CELLS_PER_ROW;
    const row = Math.floor(index / CELLS_PER_ROW);
    const x = column * CELL_SIZE;
    const y = row * CELL_SIZE;
    page.context.fillText(character, x + CELL_SIZE / 2, y + CELL_SIZE / 2 + 1);
    page.texture.needsUpdate = true;
    const measured = page.context.measureText(character).width;
    const u0 = x / ATLAS_SIZE;
    const u1 = (x + CELL_SIZE) / ATLAS_SIZE;
    const v0 = 1 - (y + CELL_SIZE) / ATLAS_SIZE;
    const v1 = 1 - y / ATLAS_SIZE;
    const glyph = { page: this.pages.length - 1, uv: [u0, v0, u1, v1] as [number, number, number, number], advance: measured };
    this.glyphs.set(character, glyph);
    return glyph;
  }

  update(placements: LabelPlacement[], positions: Float32Array): void {
    this.visibleLabelCount = placements.length;
    const instances = new Map<number, Array<{
      anchor: [number, number, number];
      offset: [number, number];
      size: [number, number];
      uv: [number, number, number, number];
      color: [number, number, number];
      opacity: number;
    }>>();

    placements.forEach(placement => {
      const characters = [...placement.text];
      const glyphs = characters.map(character => this.ensureGlyph(character));
      const scale = DISPLAY_FONT_SIZE / ATLAS_FONT_SIZE;
      const advances = glyphs.map(glyph => Math.max(4, glyph.advance * scale));
      const totalWidth = advances.reduce((sum, value) => sum + value, 0);
      let cursor = placement.offsetX + Math.max(0, (placement.width - totalWidth) / 2);
      glyphs.forEach((glyph, index) => {
        const width = Math.max(12, CELL_SIZE * scale);
        const height = CELL_SIZE * scale;
        const advance = advances[index];
        const list = instances.get(glyph.page) ?? [];
        list.push({
          anchor: [
            positions[placement.nodeIndex * 3],
            positions[placement.nodeIndex * 3 + 1],
            positions[placement.nodeIndex * 3 + 2],
          ],
          offset: [cursor + advance / 2, placement.offsetY + placement.height / 2],
          size: [width, height],
          uv: glyph.uv,
          color: placement.color,
          opacity: placement.opacity,
        });
        instances.set(glyph.page, list);
        cursor += advance;
      });
    });
    this.visibleGlyphCount = [...instances.values()].reduce((sum, page) => sum + page.length, 0);

    this.pages.forEach((page, pageIndex) => {
      if (page.mesh) {
        this.scene.remove(page.mesh);
        page.mesh.geometry.dispose();
        page.mesh = null;
      }
      const pageInstances = instances.get(pageIndex) ?? [];
      if (!pageInstances.length) return;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pageInstances.flatMap(item => item.anchor)), 3));
      geometry.setAttribute("aOffset", new THREE.BufferAttribute(new Float32Array(pageInstances.flatMap(item => item.offset)), 2));
      geometry.setAttribute("aSize", new THREE.BufferAttribute(new Float32Array(pageInstances.map(item => Math.max(item.size[0], item.size[1]))), 1));
      geometry.setAttribute("aUvRect", new THREE.BufferAttribute(new Float32Array(pageInstances.flatMap(item => item.uv)), 4));
      geometry.setAttribute("aColor", new THREE.BufferAttribute(new Float32Array(pageInstances.flatMap(item => item.color)), 3));
      geometry.setAttribute("aOpacity", new THREE.BufferAttribute(new Float32Array(pageInstances.map(item => item.opacity)), 1));
      page.mesh = new THREE.Points(geometry, page.material);
      page.mesh.frustumCulled = false;
      page.mesh.renderOrder = 10;
      this.scene.add(page.mesh);
    });
  }

  resize(width: number, height: number): void {
    this.resolution.set(width, height);
  }

  dispose(): void {
    this.pages.forEach(page => {
      if (page.mesh) {
        this.scene.remove(page.mesh);
        page.mesh.geometry.dispose();
      }
      page.material.dispose();
      page.texture.dispose();
    });
    this.pages = [];
    this.glyphs.clear();
  }
}
