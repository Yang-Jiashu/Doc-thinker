//#region node_modules/lucide/dist/esm/defaultAttributes.mjs
var e = {
	xmlns: "http://www.w3.org/2000/svg",
	width: 24,
	height: 24,
	viewBox: "0 0 24 24",
	fill: "none",
	stroke: "currentColor",
	"stroke-width": 2,
	"stroke-linecap": "round",
	"stroke-linejoin": "round"
}, t = ([e, n, r]) => {
	let i = document.createElementNS("http://www.w3.org/2000/svg", e);
	return Object.keys(n).forEach((e) => {
		i.setAttribute(e, String(n[e]));
	}), r?.length && r.forEach((e) => {
		let n = t(e);
		i.appendChild(n);
	}), i;
}, n = (n, r = {}) => t([
	"svg",
	{
		...e,
		...r
	},
	n
]), r = (e) => {
	for (let t in e) if (t.startsWith("aria-") || t === "role" || t === "title") return !0;
	return !1;
}, i = (...e) => e.filter((e, t, n) => !!e && e.trim() !== "" && n.indexOf(e) === t).join(" ").trim(), a = (e) => e.replace(/^([A-Z])|[\s-_]+(\w)/g, (e, t, n) => n ? n.toUpperCase() : t.toLowerCase()), o = (e) => {
	let t = a(e);
	return t.charAt(0).toUpperCase() + t.slice(1);
}, s = (e) => Array.from(e.attributes).reduce((e, t) => (e[t.name] = t.value, e), {}), c = (e) => typeof e == "string" ? e : !e || !e.class ? "" : e.class && typeof e.class == "string" ? e.class.split(" ") : e.class && Array.isArray(e.class) ? e.class : "", l = (t, { nameAttr: a, icons: l, attrs: u }) => {
	let d = t.getAttribute(a);
	if (d == null) return;
	let f = l[o(d)];
	if (!f) return console.warn(`${t.outerHTML} icon name was not found in the provided icons object.`);
	let p = s(t), m = r(p) ? {} : { "aria-hidden": "true" }, h = {
		...e,
		"data-lucide": d,
		...m,
		...u,
		...p
	}, g = c(p), _ = c(u), v = i("lucide", `lucide-${d}`, ...g, ..._);
	v && Object.assign(h, { class: v });
	let y = n(f, h);
	return t.parentNode?.replaceChild(y, t);
}, u = [["path", { d: "m12 19-7-7 7-7" }], ["path", { d: "M19 12H5" }]], d = [
	["path", { d: "M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2" }],
	["path", { d: "M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2" }],
	["path", { d: "M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8" }],
	["path", { d: "M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" }]
], f = [
	["path", { d: "M15 3h6v6" }],
	["path", { d: "m21 3-7 7" }],
	["path", { d: "m3 21 7-7" }],
	["path", { d: "M9 21H3v-6" }]
], p = [["rect", {
	width: "18",
	height: "18",
	x: "3",
	y: "3",
	rx: "2"
}], ["path", { d: "M15 3v18" }]], m = [["path", { d: "M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8" }], ["path", { d: "M21 3v5h-5" }]], h = [["path", { d: "m21 21-4.34-4.34" }], ["circle", {
	cx: "11",
	cy: "11",
	r: "8"
}]], g = [
	["path", { d: "m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3" }],
	["path", { d: "M12 9v4" }],
	["path", { d: "M12 17h.01" }]
], _ = [["path", { d: "M18 6 6 18" }], ["path", { d: "m6 6 12 12" }]], v = [
	["circle", {
		cx: "11",
		cy: "11",
		r: "8"
	}],
	["line", {
		x1: "21",
		x2: "16.65",
		y1: "21",
		y2: "16.65"
	}],
	["line", {
		x1: "11",
		x2: "11",
		y1: "8",
		y2: "14"
	}],
	["line", {
		x1: "8",
		x2: "14",
		y1: "11",
		y2: "11"
	}]
], y = [
	["circle", {
		cx: "11",
		cy: "11",
		r: "8"
	}],
	["line", {
		x1: "21",
		x2: "16.65",
		y1: "21",
		y2: "16.65"
	}],
	["line", {
		x1: "8",
		x2: "14",
		y1: "11",
		y2: "11"
	}]
], b = ({ icons: e = {}, nameAttr: t = "data-lucide", attrs: n = {}, root: r = document, inTemplates: i } = {}) => {
	if (!Object.values(e).length) throw Error("Please provide an icons object.\nIf you want to use all the icons you can import it like:\n `import { createIcons, icons } from 'lucide';\nlucide.createIcons({icons});`");
	if (r === void 0) throw Error("`createIcons()` only works in a browser environment.");
	if (Array.from(r.querySelectorAll(`[${t}]`)).forEach((r) => l(r, {
		nameAttr: t,
		icons: e,
		attrs: n
	})), i && Array.from(r.querySelectorAll("template")).forEach((r) => b({
		icons: e,
		nameAttr: t,
		attrs: n,
		root: r.content,
		inTemplates: i
	})), t === "data-lucide") {
		let t = r.querySelectorAll("[icon-name]");
		t.length > 0 && (console.warn("[Lucide] Some icons were found with the now deprecated icon-name attribute. These will still be replaced for backwards compatibility, but will no longer be supported in v1.0 and you should switch to data-lucide"), Array.from(t).forEach((t) => l(t, {
			nameAttr: "icon-name",
			icons: e,
			attrs: n
		})));
	}
}, x = {
	enabledByDefault: !0,
	angularSpeedRadPerSecond: Math.PI * 2 / 45,
	resumeDelayMs: 1200,
	wheelIdleMs: 180,
	pauseWhenPointerOutside: !1,
	pauseWhenDocumentHidden: !0,
	pauseWhileLayoutRunning: !0
}, S = class {
	config;
	state;
	wheelActiveUntil = 0;
	constructor(e = {}, t, n = performance.now()) {
		this.config = {
			...x,
			...e
		}, this.state = {
			enabled: t ?? this.config.enabledByDefault,
			pointerInside: !1,
			pointerDown: !1,
			orbitInteracting: !1,
			nodeDragging: !1,
			wheelActive: !1,
			hasSelection: !1,
			layoutStable: !1,
			documentVisible: !0,
			lastInteractionAt: n
		};
	}
	get snapshot() {
		return { ...this.state };
	}
	setEnabled(e, t = performance.now()) {
		this.state.enabled !== e && (this.state.enabled = e, this.markInteraction(t));
	}
	setPointerInside(e, t = performance.now()) {
		this.state.pointerInside !== e && (this.state.pointerInside = e, this.markInteraction(t));
	}
	setPointerDown(e, t = performance.now()) {
		this.state.pointerDown !== e && (this.state.pointerDown = e, this.markInteraction(t));
	}
	setOrbitInteracting(e, t = performance.now()) {
		this.state.orbitInteracting !== e && (this.state.orbitInteracting = e, this.markInteraction(t));
	}
	setNodeDragging(e, t = performance.now()) {
		this.state.nodeDragging !== e && (this.state.nodeDragging = e, this.markInteraction(t));
	}
	noteWheel(e = performance.now()) {
		this.state.wheelActive = !0, this.wheelActiveUntil = e + this.config.wheelIdleMs, this.markInteraction(e);
	}
	setHasSelection(e, t = performance.now()) {
		this.state.hasSelection !== e && (this.state.hasSelection = e, this.markInteraction(t));
	}
	setLayoutStable(e, t = performance.now()) {
		this.state.layoutStable !== e && (this.state.layoutStable = e, this.markInteraction(t));
	}
	setDocumentVisible(e, t = performance.now()) {
		this.state.documentVisible !== e && (this.state.documentVisible = e, this.markInteraction(t));
	}
	markInteraction(e = performance.now()) {
		this.state.lastInteractionAt = e;
	}
	evaluate(e = performance.now()) {
		this.state.wheelActive && e >= this.wheelActiveUntil && (this.state.wheelActive = !1);
		let t = this.pauseReason(e);
		return {
			shouldRotate: t === null,
			pausedReason: t
		};
	}
	pauseReason(e) {
		return this.state.enabled ? this.config.pauseWhenDocumentHidden && !this.state.documentVisible ? "document-hidden" : this.config.pauseWhileLayoutRunning && !this.state.layoutStable ? "layout-running" : this.state.hasSelection ? "selection" : this.config.pauseWhenPointerOutside && !this.state.pointerInside ? "pointer-outside" : this.state.pointerDown ? "pointer-down" : this.state.nodeDragging ? "node-drag" : this.state.orbitInteracting ? "orbit-interaction" : this.state.wheelActive ? "wheel" : e - this.state.lastInteractionAt < this.config.resumeDelayMs ? "resume-delay" : null : "disabled";
	}
}, C = [
	[
		101 / 255,
		231 / 255,
		242 / 255
	],
	[
		123 / 255,
		167 / 255,
		1
	],
	[
		244 / 255,
		200 / 255,
		106 / 255
	],
	[
		1,
		122 / 255,
		122 / 255
	],
	[
		185 / 255,
		148 / 255,
		1
	],
	[
		218 / 255,
		229 / 255,
		244 / 255
	]
], w = /* @__PURE__ */ new Map([
	["person", 0],
	["people", 0],
	["human", 0],
	["人物", 0],
	["location", 1],
	["place", 1],
	["space", 1],
	["地点", 1],
	["空间", 1],
	["artifact", 2],
	["object", 2],
	["item", 2],
	["物品", 2],
	["器物", 2],
	["event", 3],
	["事件", 3],
	["concept", 4],
	["content", 4],
	["概念", 4],
	["内容", 4]
]);
function T(e) {
	return String(e ?? "").trim();
}
function E(e) {
	return e && typeof e == "object" && "id" in e ? T(e.id) : T(e);
}
function D(e) {
	return e === !0 || e === 1 || [
		"1",
		"true",
		"yes"
	].includes(T(e).toLowerCase());
}
function O(e) {
	if (typeof e != "string") return e;
	try {
		return JSON.parse(e);
	} catch {
		return;
	}
}
function k(e) {
	let t = O(e);
	return Array.isArray(t) ? t : [];
}
function A(e) {
	let t = O(e);
	return !t || typeof t != "object" || Array.isArray(t) ? {} : Object.fromEntries(Object.entries(t).flatMap(([e, t]) => Number.isFinite(Number(t)) ? [[e, Number(t)]] : []));
}
function ee(e) {
	let t = 2166136261;
	for (let n = 0; n < e.length; n += 1) t ^= e.charCodeAt(n), t = Math.imul(t, 16777619);
	return t >>> 0;
}
function te(e) {
	let t = Array.isArray(e) ? e : T(e).split("<SEP>"), n = /* @__PURE__ */ new Set();
	return t.map(T).filter((e) => e.length > 0 && !n.has(e) && !!n.add(e));
}
function j(e) {
	return T(e).split("<SEP>").map((e) => e.trim()).filter(Boolean).join("\n") || "暂无可展示的描述。";
}
function ne(e) {
	return w.get(T(e).toLowerCase()) ?? 5;
}
function re(e, t) {
	let n = 2166136261, r = (e) => {
		n ^= e, n = Math.imul(n, 16777619);
	};
	return e.forEach((e) => r(ee(e.id))), t.forEach((e) => {
		r(e.source), r(e.target);
	}), `${e.length}-${t.length}-${(n >>> 0).toString(16)}`;
}
function ie(e, t) {
	let n = new Uint32Array(e);
	for (let e = 0; e < t.length; e += 2) {
		let r = t[e], i = t[e + 1];
		r !== i && (n[r] += 1, n[i] += 1);
	}
	let r = new Uint32Array(e + 1);
	for (let t = 0; t < e; t += 1) r[t + 1] = r[t] + n[t];
	let i = new Uint32Array(r[e]), a = r.slice(0, e);
	for (let e = 0; e < t.length; e += 2) {
		let n = t[e], r = t[e + 1];
		n !== r && (i[a[n]++] = r, i[a[r]++] = n);
	}
	return {
		offsets: r,
		neighbors: i
	};
}
function ae(e, t) {
	return e.csrNeighbors.subarray(e.csrOffsets[t], e.csrOffsets[t + 1]);
}
function oe(e) {
	let t = Array.isArray(e.nodes) ? e.nodes : [], n = Array.isArray(e.edges) ? e.edges : Array.isArray(e.links) ? e.links : [], r = [], i = /* @__PURE__ */ new Map();
	t.forEach((e) => {
		let t = T(e.id ?? e.entity_id ?? e.label);
		if (!t || i.has(t)) return;
		let n = T(e.type ?? e.entity_type) || "unknown", a = ne(n);
		i.set(t, r.length), r.push({
			id: t,
			label: T(e.label) || t,
			type: n,
			description: j(e.description),
			sourceId: T(e.source_id),
			filePath: T(e.file_path),
			degree: Math.max(0, Number(e.degree) || 0),
			group: a,
			color: C[a],
			size: 4,
			isExpanded: String(e.is_expanded ?? "").toLowerCase() === "true" || String(e.is_expanded) === "1",
			isImageNode: String(e.is_image_node ?? "").toLowerCase() === "true" || String(e.is_image_node) === "1"
		});
	});
	let a = new Uint32Array(r.length), o = [];
	n.forEach((e, t) => {
		let n = E(e.source ?? e.src_id), r = E(e.target ?? e.tgt_id), s = i.get(n), c = i.get(r);
		if (s === void 0 || c === void 0 || s === c) return;
		a[s] += 1, a[c] += 1;
		let l = T(e.review_status).toLowerCase(), u = T(e.provenance).toLowerCase(), d = T(e.algorithm_version).toLowerCase(), f = D(e.is_promoted) || l === "promoted" && u === "eclrr_v4" && d === "eclrr_v4", p = te(e.source_id), m = k(e.evidence_chunk_ids).map(T).filter(Boolean), h = Number(e.decision_score);
		o.push({
			id: T(e.id) || `edge-${t}`,
			source: s,
			target: c,
			label: T(e.relation ?? e.label) || "related",
			description: j(e.description),
			sourceId: T(e.source_id),
			weight: Number(e.weight) || 1,
			isDiscovered: D(e.is_discovered),
			isPromoted: f,
			kind: f || T(e.edge_kind).toLowerCase() === "eclrr_v4" ? "eclrr_v4" : "original",
			relationFamily: T(e.relation_family),
			direction: T(e.direction),
			relationId: T(e.relation_id),
			canonicalKey: T(e.canonical_key),
			pathUsed: k(e.path_used).map(T).filter(Boolean),
			supportingPaths: k(e.supporting_paths),
			evidenceChain: k(e.evidence_chain),
			evidenceChunkIds: m.length ? m : p,
			judgeScores: A(e.judge_scores),
			decisionScore: Number.isFinite(h) ? h : null
		});
	});
	let s = new Float32Array(r.length * 3), c = new Float32Array(r.length * 3), l = new Float32Array(r.length), u = new Float32Array(r.length), d = new Uint8Array(r.length), f = new Uint32Array(r.length);
	r.forEach((e, t) => {
		e.degree = Math.max(e.degree, a[t]), e.size = Math.min(12, 3.8 + Math.sqrt(e.degree + 1) * .72), s[t * 3] = ((ee(e.id) & 65535) / 65535 - .5) * 300, s[t * 3 + 1] = ((ee(e.id) >>> 16 & 65535) / 65535 - .5) * 300, s[t * 3 + 2] = 0, c.set(e.color, t * 3), l[t] = e.size, u[t] = e.degree, d[t] = e.group, f[t] = ee(e.id);
	});
	let p = new Uint32Array(o.length * 2);
	o.forEach((e, t) => {
		p[t * 2] = e.source, p[t * 2 + 1] = e.target;
	});
	let m = ie(r.length, p);
	return {
		nodes: r,
		edges: o,
		nodeIndex: i,
		positions: s,
		colors: c,
		sizes: l,
		degrees: u,
		groups: d,
		hashes: f,
		edgePairs: p,
		csrOffsets: m.offsets,
		csrNeighbors: m.neighbors,
		fingerprint: re(r, o),
		metadata: e.metadata ?? {}
	};
}
//#endregion
//#region frontend/src/detail-panel.ts
function se(e, t) {
	e && (e.textContent = t);
}
var ce = class {
	root;
	title;
	kind;
	meta;
	descriptionHeading;
	description;
	evidenceSection;
	evidenceStatus;
	evidenceList;
	sourceIds;
	chunks;
	chunkStatus;
	constructor(e) {
		this.root = e, this.title = e.querySelector("[data-detail-title]"), this.kind = e.querySelector("[data-detail-kind]"), this.meta = e.querySelector("[data-detail-meta]"), this.descriptionHeading = e.querySelector("[data-description-heading]"), this.description = e.querySelector("[data-detail-description]"), this.evidenceSection = e.querySelector("[data-relation-evidence]"), this.evidenceStatus = e.querySelector("[data-evidence-status]"), this.evidenceList = e.querySelector("[data-evidence-list]"), this.sourceIds = e.querySelector("[data-source-ids]"), this.chunks = e.querySelector("[data-chunks]"), this.chunkStatus = e.querySelector("[data-chunk-status]");
	}
	show(e) {
		this.root.hidden = !1, this.root.dataset.detailType = "node", se(this.kind, "FOCUSED NODE"), se(this.title, e.label), se(this.meta, `${e.type || "unknown"} · ${e.degree} 条连接`), se(this.descriptionHeading, "节点描述"), se(this.description, e.description), this.evidenceSection.hidden = !0, this.evidenceList.replaceChildren(), this.showSourceIds(te(e.sourceId)), this.resetChunks();
	}
	showEdge(e, t, n) {
		this.root.hidden = !1, this.root.dataset.detailType = "edge", se(this.kind, e.kind === "eclrr_v4" ? "ECLRR-V4 EDGE" : "FACT EDGE"), se(this.title, e.label || "related");
		let r = e.kind === "eclrr_v4" ? "ECLRR-v4 推断关系 · 虚线" : "入库事实关系 · 实线", i = e.relationFamily ? ` · ${e.relationFamily}` : "", a = e.direction ? ` · ${e.direction}` : "";
		se(this.meta, `${t.label} → ${n.label} · ${r}${i}${a}`), se(this.descriptionHeading, "关系描述"), se(this.description, e.description), this.showSourceIds(e.evidenceChunkIds.length ? e.evidenceChunkIds : te(e.sourceId)), this.showEvidence(e), this.resetChunks();
	}
	showSourceIds(e) {
		if (this.sourceIds.replaceChildren(), e.length) e.forEach((e) => {
			let t = document.createElement("code");
			t.textContent = e, this.sourceIds.append(t);
		});
		else {
			let e = document.createElement("span");
			e.className = "detail-empty", e.textContent = "暂无来源 chunk", this.sourceIds.append(e);
		}
	}
	showEvidence(e) {
		this.evidenceList.replaceChildren();
		let t = e.evidenceChain.length > 0 || e.pathUsed.length > 0;
		if (this.evidenceSection.hidden = !t, !t) return;
		let n = e.judgeScores.total ?? e.decisionScore;
		if (this.evidenceStatus.textContent = n == null ? "" : `Judge ${n}/10`, e.pathUsed.length) {
			let t = document.createElement("p");
			t.className = "evidence-path", t.textContent = e.pathUsed.join(" → "), this.evidenceList.append(t);
		}
		e.evidenceChain.forEach((e, t) => {
			let n = document.createElement("article");
			n.className = "evidence-item";
			let r = document.createElement("strong"), i = [e.source, e.target].filter(Boolean).join(" → ");
			r.textContent = `${t + 1}. ${i || e.relation || "证据"}`;
			let a = document.createElement("code");
			a.textContent = e.chunk_id || "未标注 chunk";
			let o = document.createElement("p");
			o.textContent = e.quote || "未提供引用文本", n.append(r, a, o), this.evidenceList.append(n);
		});
	}
	resetChunks() {
		this.chunks.replaceChildren(), this.chunkStatus.textContent = "正在读取原文证据…";
	}
	showChunks(e, t = "") {
		if (this.chunks.replaceChildren(), t) {
			this.chunkStatus.textContent = t;
			return;
		}
		this.chunkStatus.textContent = e.length ? `${e.length} 个原文 chunk` : "未找到原文 chunk", e.forEach((e) => {
			let t = document.createElement("article");
			t.className = "chunk-card", t.style.contentVisibility = "auto", t.style.containIntrinsicSize = "220px";
			let n = document.createElement("header"), r = document.createElement("code");
			r.textContent = e.chunk_id;
			let i = document.createElement("span");
			i.textContent = e.missing ? "missing" : e.chunk_order_index === void 0 ? "source" : `order ${e.chunk_order_index}`, n.append(r, i);
			let a = document.createElement("p");
			a.textContent = e.content || "该 chunk 没有可读取的文本内容。", t.append(n, a), this.chunks.append(t);
		});
	}
	hide() {
		this.root.hidden = !0, delete this.root.dataset.detailType, this.evidenceList.replaceChildren(), this.chunks.replaceChildren();
	}
};
//#endregion
//#region frontend/src/gesture/gesture-metrics.ts
function le(e, t) {
	let n = String(e ?? "").trim().toLowerCase();
	return n === "left" ? "left" : n === "right" ? "right" : t >= .5 ? "left" : "right";
}
function ue(e, t) {
	return Math.hypot(e.x - t.x, e.y - t.y, (e.z ?? 0) - (t.z ?? 0));
}
function de(e, t = !1) {
	if (e.length < 21) return null;
	let n = e[0], r = e[4], i = e[8], a = Math.max(ue(n, e[9]), .035), o = ue(r, i) / a, s = t ? o < .34 : o < .24, c = [
		8,
		12,
		16,
		20
	], l = [
		6,
		10,
		14,
		18
	], u = c.reduce((t, r, i) => {
		let a = ue(e[r], n) > ue(e[l[i]], n) * 1.08;
		return t + Number(a);
	}, 0), d = [
		0,
		5,
		9,
		13,
		17
	].reduce((t, n) => ({
		x: t.x + e[n].x / 5,
		y: t.y + e[n].y / 5
	}), {
		x: 0,
		y: 0
	});
	return {
		point: {
			x: 1 - i.x,
			y: i.y
		},
		thumbPoint: {
			x: 1 - r.x,
			y: r.y
		},
		palmPoint: {
			x: 1 - d.x,
			y: d.y
		},
		pinch: s,
		pinchDistance: o,
		openPalm: !s && u >= 4,
		extendedFingers: u
	};
}
function fe(e, t, n, r = "right") {
	let i = (e) => ({
		x: e.x * t,
		y: e.y * n
	});
	return {
		...e,
		handedness: r,
		point: i(e.point),
		thumbPoint: i(e.thumbPoint),
		palmPoint: i(e.palmPoint)
	};
}
function pe(e, t, n = .34) {
	return e ? {
		x: e.x * (1 - n) + t.x * n,
		y: e.y * (1 - n) + t.y * n
	} : { ...t };
}
//#endregion
//#region node_modules/@mediapipe/tasks-vision/vision_bundle.mjs
var me = typeof self < "u" ? self : {};
function he(e, t) {
	t: {
		for (var n = ["CLOSURE_FLAGS"], r = me, i = 0; i < n.length; i++) if ((r = r[n[i]]) == null) {
			n = null;
			break t;
		}
		n = r;
	}
	return (e = n && n[e]) ?? t;
}
function ge() {
	throw Error("Invalid UTF8");
}
function _e(e, t) {
	return t = String.fromCharCode.apply(null, t), e == null ? t : e + t;
}
var ve, ye, be = typeof TextDecoder < "u", xe, Se = typeof TextEncoder < "u";
function Ce(e) {
	if (Se) e = (xe ||= new TextEncoder()).encode(e);
	else {
		let n = 0, r = new Uint8Array(3 * e.length);
		for (let i = 0; i < e.length; i++) {
			var t = e.charCodeAt(i);
			if (t < 128) r[n++] = t;
			else {
				if (t < 2048) r[n++] = t >> 6 | 192;
				else {
					if (t >= 55296 && t <= 57343) {
						if (t <= 56319 && i < e.length) {
							let a = e.charCodeAt(++i);
							if (a >= 56320 && a <= 57343) {
								t = 1024 * (t - 55296) + a - 56320 + 65536, r[n++] = t >> 18 | 240, r[n++] = t >> 12 & 63 | 128, r[n++] = t >> 6 & 63 | 128, r[n++] = 63 & t | 128;
								continue;
							}
							i--;
						}
						t = 65533;
					}
					r[n++] = t >> 12 | 224, r[n++] = t >> 6 & 63 | 128;
				}
				r[n++] = 63 & t | 128;
			}
		}
		e = n === r.length ? r : r.subarray(0, n);
	}
	return e;
}
function we(e) {
	me.setTimeout((() => {
		throw e;
	}), 0);
}
var Te, M = he(610401301, !1), Ee = he(748402147, !0);
function De() {
	var e = me.navigator;
	return (e &&= e.userAgent) ? e : "";
}
var Oe = me.navigator;
function N(e) {
	return N[" "](e), e;
}
Te = Oe && Oe.userAgentData || null, N[" "] = function() {};
var ke = {}, P = null;
function F(e) {
	let t = e.length, n = 3 * t / 4;
	n % 3 ? n = Math.floor(n) : "=.".indexOf(e[t - 1]) != -1 && (n = "=.".indexOf(e[t - 2]) == -1 ? n - 1 : n - 2);
	let r = new Uint8Array(n), i = 0;
	return function(e, t) {
		function n(t) {
			for (; r < e.length;) {
				let t = e.charAt(r++), n = P[t];
				if (n != null) return n;
				if (!/^[\s\xa0]*$/.test(t)) throw Error("Unknown base64 encoding at char: " + t);
			}
			return t;
		}
		Ae();
		let r = 0;
		for (;;) {
			let e = n(-1), r = n(0), i = n(64), a = n(64);
			if (a === 64 && e === -1) break;
			t(e << 2 | r >> 4), i != 64 && (t(r << 4 & 240 | i >> 2), a != 64 && t(i << 6 & 192 | a));
		}
	}(e, (function(e) {
		r[i++] = e;
	})), i === n ? r : r.subarray(0, i);
}
function Ae() {
	if (!P) {
		P = {};
		var e = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".split(""), t = [
			"+/=",
			"+/",
			"-_=",
			"-_.",
			"-_"
		];
		for (let n = 0; n < 5; n++) {
			let r = e.concat(t[n].split(""));
			ke[n] = r;
			for (let e = 0; e < r.length; e++) {
				let t = r[e];
				P[t] === void 0 && (P[t] = e);
			}
		}
	}
}
var je = typeof Uint8Array < "u", Me = !(!(M && Te && Te.brands.length > 0) && (De().indexOf("Trident") != -1 || De().indexOf("MSIE") != -1)) && typeof btoa == "function", Ne = /[-_.]/g, Pe = {
	"-": "+",
	_: "/",
	".": "="
};
function Fe(e) {
	return Pe[e] || "";
}
function Ie(e) {
	if (!Me) return F(e);
	e = Ne.test(e) ? e.replace(Ne, Fe) : e, e = atob(e);
	let t = new Uint8Array(e.length);
	for (let n = 0; n < e.length; n++) t[n] = e.charCodeAt(n);
	return t;
}
function Le(e) {
	return je && e != null && e instanceof Uint8Array;
}
var Re = {};
function ze() {
	return He ||= new Ve(null, Re);
}
function Be(e) {
	We(Re);
	var t = e.g;
	return (t = t == null || Le(t) ? t : typeof t == "string" ? Ie(t) : null) == null ? t : e.g = t;
}
var Ve = class {
	h() {
		return new Uint8Array(Be(this) || 0);
	}
	constructor(e, t) {
		if (We(t), this.g = e, e != null && e.length === 0) throw Error("ByteString should be constructed with non-empty values");
	}
}, He, Ue;
function We(e) {
	if (e !== Re) throw Error("illegal external caller");
}
function Ge(e, t) {
	e.__closure__error__context__984382 ||= {}, e.__closure__error__context__984382.severity = t;
}
function Ke(e) {
	return Ge(e = Error(e), "warning"), e;
}
function qe(e, t) {
	if (e != null) {
		var n = Ue ??= {}, r = n[e] || 0;
		r >= t || (n[e] = r + 1, Ge(e = Error(), "incident"), we(e));
	}
}
function Je() {
	return typeof BigInt == "function";
}
var Ye = typeof Symbol == "function" && typeof Symbol() == "symbol";
function Xe(e, t, n = !1) {
	return typeof Symbol == "function" && typeof Symbol() == "symbol" ? n && Symbol.for && e ? Symbol.for(e) : e == null ? Symbol() : Symbol(e) : t;
}
var Ze = Xe("jas", void 0, !0), Qe = Xe(void 0, "0di"), $e = Xe(void 0, "1oa"), et = Xe(void 0, Symbol()), tt = Xe(void 0, "0ub"), nt = Xe(void 0, "0ubs"), rt = Xe(void 0, "0ubsb"), it = Xe(void 0, "0actk"), at = Xe("m_m", "Pa", !0), ot = Xe(), st = { Ga: {
	value: 0,
	configurable: !0,
	writable: !0,
	enumerable: !1
} }, ct = Object.defineProperties, I = Ye ? Ze : "Ga", lt, ut = [];
function dt(e, t) {
	Ye || I in e || ct(e, st), e[I] |= t;
}
function ft(e, t) {
	Ye || I in e || ct(e, st), e[I] = t;
}
function pt(e) {
	return dt(e, 34), e;
}
function mt(e) {
	return dt(e, 8192), e;
}
ft(ut, 7), lt = Object.freeze(ut);
var ht = {};
function gt(e, t) {
	return t === void 0 ? e.h !== _t && !!(2 & (0 | e.v[I])) : !!(2 & t) && e.h !== _t;
}
var _t = {};
function vt(e, t) {
	if (e != null) {
		if (typeof e == "string") e = e ? new Ve(e, Re) : ze();
		else if (e.constructor !== Ve) if (Le(e)) e = e.length ? new Ve(new Uint8Array(e), Re) : ze();
		else {
			if (!t) throw Error();
			e = void 0;
		}
	}
	return e;
}
var yt = class {
	constructor(e, t, n) {
		this.g = e, this.h = t, this.l = n;
	}
	next() {
		let e = this.g.next();
		return e.done || (e.value = this.h.call(this.l, e.value)), e;
	}
	[Symbol.iterator]() {
		return this;
	}
}, bt = Object.freeze({});
function xt(e, t, n) {
	let r = 128 & t ? 0 : -1, i = e.length;
	var a;
	(a = !!i) && (a = (a = e[i - 1]) != null && typeof a == "object" && a.constructor === Object);
	let o = i + (a ? -1 : 0);
	for (t = 128 & t ? 1 : 0; t < o; t++) n(t - r, e[t]);
	if (a) {
		e = e[i - 1];
		for (let t in e) !isNaN(t) && n(+t, e[t]);
	}
}
var St = {};
function Ct(e) {
	return 128 & e ? St : void 0;
}
function wt(e) {
	return e.Na = !0, e;
}
var Tt = wt(((e) => typeof e == "number")), Et = wt(((e) => typeof e == "string")), Dt = wt(((e) => typeof e == "boolean")), Ot = typeof me.BigInt == "function" && typeof me.BigInt(0) == "bigint";
function kt(e) {
	var t = e;
	if (Et(t)) {
		if (!/^\s*(?:-?[1-9]\d*|0)?\s*$/.test(t)) throw Error(String(t));
	} else if (Tt(t) && !Number.isSafeInteger(t)) throw Error(String(t));
	return Ot ? BigInt(e) : e = Dt(e) ? e ? "1" : "0" : Et(e) ? e.trim() || "0" : String(e);
}
var At = wt(((e) => Ot ? e >= Mt && e <= Pt : e[0] === "-" ? Ft(e, jt) : Ft(e, Nt))), jt = (-(2 ** 53 - 1)).toString(), Mt = Ot ? BigInt(-(2 ** 53 - 1)) : void 0, Nt = (2 ** 53 - 1).toString(), Pt = Ot ? BigInt(2 ** 53 - 1) : void 0;
function Ft(e, t) {
	if (e.length > t.length) return !1;
	if (e.length < t.length || e === t) return !0;
	for (let n = 0; n < e.length; n++) {
		let r = e[n], i = t[n];
		if (r > i) return !1;
		if (r < i) return !0;
	}
}
var It = typeof Uint8Array.prototype.slice == "function", Lt, Rt = 0, zt = 0;
function Bt(e) {
	let t = e >>> 0;
	Rt = t, zt = (e - t) / 4294967296 >>> 0;
}
function Vt(e) {
	if (e < 0) {
		Bt(-e);
		let [t, n] = Xt(Rt, zt);
		Rt = t >>> 0, zt = n >>> 0;
	} else Bt(e);
}
function Ht(e) {
	let t = Lt ||= /* @__PURE__ */ new DataView(/* @__PURE__ */ new ArrayBuffer(8));
	t.setFloat32(0, +e, !0), zt = 0, Rt = t.getUint32(0, !0);
}
function Ut(e, t) {
	let n = 4294967296 * t + (e >>> 0);
	return Number.isSafeInteger(n) ? n : Kt(e, t);
}
function Wt(e, t) {
	return kt(Je() ? BigInt.asUintN(64, (BigInt(t >>> 0) << BigInt(32)) + BigInt(e >>> 0)) : Kt(e, t));
}
function Gt(e, t) {
	return Je() ? kt(BigInt.asIntN(64, (BigInt.asUintN(32, BigInt(t)) << BigInt(32)) + BigInt.asUintN(32, BigInt(e)))) : kt(Jt(e, t));
}
function Kt(e, t) {
	if (e >>>= 0, (t >>>= 0) <= 2097151) var n = "" + (4294967296 * t + e);
	else Je() ? n = "" + (BigInt(t) << BigInt(32) | BigInt(e)) : (e = (16777215 & e) + 6777216 * (n = 16777215 & (e >>> 24 | t << 8)) + 6710656 * (t = t >> 16 & 65535), n += 8147497 * t, t *= 2, e >= 1e7 && (n += e / 1e7 >>> 0, e %= 1e7), n >= 1e7 && (t += n / 1e7 >>> 0, n %= 1e7), n = t + qt(n) + qt(e));
	return n;
}
function qt(e) {
	return e = String(e), "0000000".slice(e.length) + e;
}
function Jt(e, t) {
	if (2147483648 & t) if (Je()) e = "" + (BigInt(0 | t) << BigInt(32) | BigInt(e >>> 0));
	else {
		let [n, r] = Xt(e, t);
		e = "-" + Kt(n, r);
	}
	else e = Kt(e, t);
	return e;
}
function Yt(e) {
	if (e.length < 16) Vt(Number(e));
	else if (Je()) e = BigInt(e), Rt = Number(e & BigInt(4294967295)) >>> 0, zt = Number(e >> BigInt(32) & BigInt(4294967295));
	else {
		let t = +(e[0] === "-");
		zt = Rt = 0;
		let n = e.length;
		for (let r = t, i = (n - t) % 6 + t; i <= n; r = i, i += 6) {
			let t = Number(e.slice(r, i));
			zt *= 1e6, Rt = 1e6 * Rt + t, Rt >= 4294967296 && (zt += Math.trunc(Rt / 4294967296), zt >>>= 0, Rt >>>= 0);
		}
		if (t) {
			let [e, t] = Xt(Rt, zt);
			Rt = e, zt = t;
		}
	}
}
function Xt(e, t) {
	return t = ~t, e ? e = 1 + ~e : t += 1, [e, t];
}
function Zt(e) {
	return Array.prototype.slice.call(e);
}
var Qt = typeof BigInt == "function" ? BigInt.asIntN : void 0, $t = typeof BigInt == "function" ? BigInt.asUintN : void 0, en = Number.isSafeInteger, tn = Number.isFinite, nn = Math.trunc, rn = kt(0);
function an(e) {
	if (e != null && typeof e != "number") throw Error(`Value of float/double field must be a number, found ${typeof e}: ${e}`);
	return e;
}
function on(e) {
	return e == null || typeof e == "number" ? e : e === "NaN" || e === "Infinity" || e === "-Infinity" ? Number(e) : void 0;
}
function sn(e) {
	if (e != null && typeof e != "boolean") {
		var t = typeof e;
		throw Error(`Expected boolean but got ${t == "object" ? e ? Array.isArray(e) ? "array" : t : "null" : t}: ${e}`);
	}
	return e;
}
function cn(e) {
	return e == null || typeof e == "boolean" ? e : typeof e == "number" ? !!e : void 0;
}
var ln = /^-?([1-9][0-9]*|0)(\.[0-9]+)?$/;
function un(e) {
	switch (typeof e) {
		case "bigint": return !0;
		case "number": return tn(e);
		case "string": return ln.test(e);
		default: return !1;
	}
}
function dn(e) {
	if (e == null) return e;
	if (typeof e == "string" && e) e = +e;
	else if (typeof e != "number") return;
	return tn(e) ? 0 | e : void 0;
}
function fn(e) {
	if (e == null) return e;
	if (typeof e == "string" && e) e = +e;
	else if (typeof e != "number") return;
	return tn(e) ? e >>> 0 : void 0;
}
function pn(e) {
	let t = e.length;
	return (e[0] === "-" ? t < 20 || t === 20 && e <= "-9223372036854775808" : t < 19 || t === 19 && e <= "9223372036854775807") ? e : (Yt(e), Jt(Rt, zt));
}
function mn(e) {
	if (e = nn(e), !en(e)) {
		Vt(e);
		var t = Rt, n = zt;
		(e = 2147483648 & n) && (n = ~n >>> 0, (t = 1 + ~t >>> 0) == 0 && (n = n + 1 >>> 0)), e = typeof (t = Ut(t, n)) == "number" ? e ? -t : t : e ? "-" + t : t;
	}
	return e;
}
function hn(e) {
	var t = nn(Number(e));
	return en(t) ? String(t) : ((t = e.indexOf(".")) !== -1 && (e = e.substring(0, t)), pn(e));
}
function gn(e) {
	var t = nn(Number(e));
	return en(t) ? kt(t) : ((t = e.indexOf(".")) !== -1 && (e = e.substring(0, t)), Je() ? kt(Qt(64, BigInt(e))) : kt(pn(e)));
}
function _n(e) {
	return en(e) ? e = kt(mn(e)) : (e = nn(e), en(e) ? e = String(e) : (Vt(e), e = Jt(Rt, zt)), e = kt(e)), e;
}
function vn(e) {
	let t = typeof e;
	return e == null ? e : t === "bigint" ? kt(Qt(64, e)) : un(e) ? t === "string" ? gn(e) : _n(e) : void 0;
}
function yn(e) {
	if (typeof e != "string") throw Error();
	return e;
}
function bn(e) {
	if (e != null && typeof e != "string") throw Error();
	return e;
}
function xn(e) {
	return e == null || typeof e == "string" ? e : void 0;
}
function Sn(e, t, n, r) {
	return e != null && e[at] === ht ? e : Array.isArray(e) ? ((r = (n = 0 | e[I]) | 32 & r | 2 & r) !== n && ft(e, r), new t(e)) : (n ? 2 & r ? ((e = t[Qe]) || (pt((e = new t()).v), e = t[Qe] = e), t = e) : t = new t() : t = void 0, t);
}
function Cn(e, t, n) {
	if (t) t: {
		if (!un(t = e)) throw Ke("int64");
		switch (typeof t) {
			case "string":
				t = gn(t);
				break t;
			case "bigint":
				t = kt(Qt(64, t));
				break t;
			default: t = _n(t);
		}
	}
	else t = vn(e);
	return (e = t) ?? (n ? rn : void 0);
}
var wn = {}, Tn = function() {
	try {
		return N(new class extends Map {
			constructor() {
				super();
			}
		}()), !1;
	} catch {
		return !0;
	}
}(), En = class {
	constructor() {
		this.g = /* @__PURE__ */ new Map();
	}
	get(e) {
		return this.g.get(e);
	}
	set(e, t) {
		return this.g.set(e, t), this.size = this.g.size, this;
	}
	delete(e) {
		return e = this.g.delete(e), this.size = this.g.size, e;
	}
	clear() {
		this.g.clear(), this.size = this.g.size;
	}
	has(e) {
		return this.g.has(e);
	}
	entries() {
		return this.g.entries();
	}
	keys() {
		return this.g.keys();
	}
	values() {
		return this.g.values();
	}
	forEach(e, t) {
		return this.g.forEach(e, t);
	}
	[Symbol.iterator]() {
		return this.entries();
	}
}, Dn = Tn ? (Object.setPrototypeOf(En.prototype, Map.prototype), Object.defineProperties(En.prototype, { size: {
	value: 0,
	configurable: !0,
	enumerable: !0,
	writable: !0
} }), En) : class extends Map {
	constructor() {
		super();
	}
};
function On(e) {
	return e;
}
function kn(e) {
	if (2 & e.J) throw Error("Cannot mutate an immutable Map");
}
var An = class extends Dn {
	constructor(e, t, n = On, r = On) {
		super(), this.J = 0 | e[I], this.K = t, this.S = n, this.fa = this.K ? jn : r;
		for (let i = 0; i < e.length; i++) {
			let a = e[i], o = n(a[0], !1, !0), s = a[1];
			t ? s === void 0 && (s = null) : s = r(a[1], !1, !0, void 0, void 0, this.J), super.set(o, s);
		}
	}
	V(e) {
		return mt(Array.from(super.entries(), e));
	}
	clear() {
		kn(this), super.clear();
	}
	delete(e) {
		return kn(this), super.delete(this.S(e, !0, !1));
	}
	entries() {
		if (this.K) {
			var e = super.keys();
			e = new yt(e, Mn, this);
		} else e = super.entries();
		return e;
	}
	values() {
		if (this.K) {
			var e = super.keys();
			e = new yt(e, An.prototype.get, this);
		} else e = super.values();
		return e;
	}
	forEach(e, t) {
		this.K ? super.forEach(((n, r, i) => {
			e.call(t, i.get(r), r, i);
		})) : super.forEach(e, t);
	}
	set(e, t) {
		return kn(this), (e = this.S(e, !0, !1)) == null ? this : t == null ? (super.delete(e), this) : super.set(e, this.fa(t, !0, !0, this.K, !1, this.J));
	}
	Ma(e) {
		let t = this.S(e[0], !1, !0);
		e = e[1], e = this.K ? e === void 0 ? null : e : this.fa(e, !1, !0, void 0, !1, this.J), super.set(t, e);
	}
	has(e) {
		return super.has(this.S(e, !1, !1));
	}
	get(e) {
		e = this.S(e, !1, !1);
		let t = super.get(e);
		if (t !== void 0) {
			var n = this.K;
			return n ? ((n = this.fa(t, !1, !0, n, this.ra, this.J)) !== t && super.set(e, n), n) : t;
		}
	}
	[Symbol.iterator]() {
		return this.entries();
	}
};
function jn(e, t, n, r, i, a) {
	return e = Sn(e, r, n, a), i && (e = $n(e)), e;
}
function Mn(e) {
	return [e, this.get(e)];
}
var Nn;
function Pn() {
	return Nn ||= new An(pt([]), void 0, void 0, void 0, wn);
}
function Fn(e) {
	return et ? e[et] : void 0;
}
function In(e, t) {
	for (let n in e) !isNaN(n) && t(e, +n, e[n]);
}
An.prototype.toJSON = void 0;
var Ln = class {}, Rn = { Ka: !0 };
function zn(e, t) {
	t < 100 || qe(nt, 1);
}
function Bn(e, t, n, r) {
	let i = r !== void 0;
	r = !!r;
	var a, o = et;
	!i && Ye && o && (a = e[o]) && In(a, zn), o = [];
	var s = e.length;
	let c;
	a = 4294967295;
	let l = !1, u = !!(64 & t), d = u ? 128 & t ? 0 : -1 : void 0;
	1 & t || (c = s && e[s - 1], typeof c == "object" && c && c.constructor === Object ? a = --s : c = void 0, !u || 128 & t || i || (l = !0, a = a - d + d)), t = void 0;
	for (var f = 0; f < s; f++) {
		let i = e[f];
		if (i != null && (i = n(i, r)) != null) if (u && f >= a) {
			let e = f - d;
			(t ??= {})[e] = i;
		} else o[f] = i;
	}
	if (c) for (let e in c) {
		if ((s = c[e]) == null || (s = n(s, r)) == null) continue;
		let i;
		f = +e, u && !Number.isNaN(f) && (i = f + d) < a ? o[i] = s : (t ??= {})[e] = s;
	}
	return t && (l ? o.push(t) : o[a] = t), i && et && (e = Fn(e)) && e instanceof Ln && (o[et] = function(e) {
		let t = new Ln();
		return In(e, ((e, n, r) => {
			t[n] = Zt(r);
		})), t.da = e.da, t;
	}(e)), o;
}
function Vn(e) {
	return e[0] = Hn(e[0]), e[1] = Hn(e[1]), e;
}
function Hn(e) {
	switch (typeof e) {
		case "number": return Number.isFinite(e) ? e : "" + e;
		case "bigint": return At(e) ? Number(e) : "" + e;
		case "boolean": return +!!e;
		case "object":
			if (Array.isArray(e)) {
				var t = 0 | e[I];
				return e.length === 0 && 1 & t ? void 0 : Bn(e, t, Hn);
			}
			if (e != null && e[at] === ht) return Gn(e);
			if (e instanceof Ve) {
				if ((t = e.g) == null) e = "";
				else if (typeof t == "string") e = t;
				else {
					if (Me) {
						for (var n = "", r = 0, i = t.length - 10240; r < i;) n += String.fromCharCode.apply(null, t.subarray(r, r += 10240));
						n += String.fromCharCode.apply(null, r ? t.subarray(r) : t), t = btoa(n);
					} else {
						n === void 0 && (n = 0), Ae(), n = ke[n], r = Array(Math.floor(t.length / 3)), i = n[64] || "";
						let e = 0, l = 0;
						for (; e < t.length - 2; e += 3) {
							var a = t[e], o = t[e + 1], s = t[e + 2], c = n[a >> 2];
							a = n[(3 & a) << 4 | o >> 4], o = n[(15 & o) << 2 | s >> 6], s = n[63 & s], r[l++] = c + a + o + s;
						}
						switch (c = 0, s = i, t.length - e) {
							case 2: s = n[(15 & (c = t[e + 1])) << 2] || i;
							case 1: t = t[e], r[l] = n[t >> 2] + n[(3 & t) << 4 | c >> 4] + s + i;
						}
						t = r.join("");
					}
					e = e.g = t;
				}
				return e;
			}
			return e instanceof An ? e = e.size === 0 ? void 0 : e.V(Vn) : void 0;
	}
	return e;
}
var Un, Wn;
function Gn(e) {
	return Bn(e = e.v, 0 | e[I], Hn);
}
function Kn(e, t) {
	return qn(e, t[0], t[1]);
}
function qn(e, t, n, r = 0) {
	if (e == null) {
		var i = 32;
		n ? (e = [n], i |= 128) : e = [], t && (i = -16760833 & i | (1023 & t) << 14);
	} else {
		if (!Array.isArray(e)) throw Error("narr");
		if (i = 0 | e[I], Ee && 1 & i) throw Error("rfarr");
		if (2048 & i && !(2 & i) && function() {
			if (Ee) throw Error("carr");
			qe(it, 5);
		}(), 256 & i) throw Error("farr");
		if (64 & i) return (i | r) !== i && ft(e, i | r), e;
		if (n && (i |= 128, n !== e[0])) throw Error("mid");
		t: {
			i |= 64;
			var a = (n = e).length;
			if (a) {
				var o = a - 1;
				let e = n[o];
				if (typeof e == "object" && e && e.constructor === Object) {
					if ((o -= t = 128 & i ? 0 : -1) >= 1024) throw Error("pvtlmt");
					for (var s in e) (a = +s) < o && (n[a + t] = e[s], delete e[s]);
					i = -16760833 & i | (1023 & o) << 14;
					break t;
				}
			}
			if (t) {
				if ((s = Math.max(t, a - (128 & i ? 0 : -1))) > 1024) throw Error("spvt");
				i = -16760833 & i | (1023 & s) << 14;
			}
		}
	}
	return ft(e, 64 | i | r), e;
}
function Jn(e, t) {
	if (typeof e != "object") return e;
	if (Array.isArray(e)) {
		var n = 0 | e[I];
		return e.length === 0 && 1 & n ? void 0 : Yn(e, n, t);
	}
	if (e != null && e[at] === ht) return Zn(e);
	if (e instanceof An) {
		if (2 & (t = e.J)) return e;
		if (!e.size) return;
		if (n = pt(e.V()), e.K) for (e = 0; e < n.length; e++) {
			let r = n[e], i = r[1];
			i = typeof i != "object" || !i ? void 0 : i != null && i[at] === ht ? Zn(i) : Array.isArray(i) ? Yn(i, 0 | i[I], !!(32 & t)) : void 0, r[1] = i;
		}
		return n;
	}
	return e instanceof Ve ? e : void 0;
}
function Yn(e, t, n) {
	return 2 & t || (!n || 4096 & t || 16 & t ? e = Qn(e, t, !1, n && !(16 & t)) : (dt(e, 34), 4 & t && Object.freeze(e))), e;
}
function Xn(e, t, n) {
	return e = new e.constructor(t), n && (e.h = _t), e.m = _t, e;
}
function Zn(e) {
	let t = e.v, n = 0 | t[I];
	return gt(e, n) ? e : rr(e, t, n) ? Xn(e, t) : Qn(t, n);
}
function Qn(e, t, n, r) {
	return r ??= !!(34 & t), e = Bn(e, t, Jn, r), r = 32, n && (r |= 2), ft(e, t = 16769217 & t | r), e;
}
function $n(e) {
	let t = e.v, n = 0 | t[I];
	return gt(e, n) ? rr(e, t, n) ? Xn(e, t, !0) : new e.constructor(Qn(t, n, !1)) : e;
}
function er(e) {
	if (e.h !== _t) return !1;
	var t = e.v;
	return dt(t = Qn(t, 0 | t[I]), 2048), e.v = t, e.h = void 0, e.m = void 0, !0;
}
function tr(e) {
	if (!er(e) && gt(e, 0 | e.v[I])) throw Error();
}
function nr(e, t) {
	t === void 0 && (t = 0 | e[I]), 32 & t && !(4096 & t) && ft(e, 4096 | t);
}
function rr(e, t, n) {
	return !!(2 & n) || !(!(32 & n) || 4096 & n) && (ft(t, 2 | n), e.h = _t, !0);
}
var ir = kt(0), ar = {};
function or(e, t, n, r, i) {
	if ((t = sr(e.v, t, n, i)) !== null || r && e.m !== _t) return t;
}
function sr(e, t, n, r) {
	if (t === -1) return null;
	let i = t + (n ? 0 : -1), a = e.length - 1, o, s;
	if (!(a < 1 + (n ? 0 : -1))) {
		if (i >= a) if (o = e[a], typeof o == "object" && o && o.constructor === Object) n = o[t], s = !0;
		else {
			if (i !== a) return;
			n = o;
		}
		else n = e[i];
		if (r && n != null) {
			if ((r = r(n)) == null) return r;
			if (!Object.is(r, n)) return s ? o[t] = r : e[i] = r, r;
		}
		return n;
	}
}
function cr(e, t, n, r) {
	tr(e), lr(e = e.v, 0 | e[I], t, n, r);
}
function lr(e, t, n, r, i) {
	let a = n + (i ? 0 : -1);
	var o = e.length - 1;
	if (o >= 1 + (i ? 0 : -1) && a >= o) {
		let i = e[o];
		if (typeof i == "object" && i && i.constructor === Object) return i[n] = r, t;
	}
	return a <= o ? (e[a] = r, t) : (r !== void 0 && (n >= (o = (t ??= 0 | e[I]) >> 14 & 1023 || 536870912) ? r != null && (e[o + (i ? 0 : -1)] = { [n]: r }) : e[a] = r), t);
}
function ur() {
	return bt === void 0 ? 2 : 4;
}
function dr(e, t, n, r, i) {
	let a = e.v, o = 0 | a[I];
	r = gt(e, o) ? 1 : r, i = !!i || r === 3, r === 2 && er(e) && (a = e.v, o = 0 | a[I]);
	let s = (e = pr(a, t)) === lt ? 7 : 0 | e[I], c = mr(s, o);
	var l = !(4 & c);
	if (l) {
		4 & c && (e = Zt(e), s = 0, c = Ar(c, o), o = lr(a, o, t, e));
		let r = 0, i = 0;
		for (; r < e.length; r++) {
			let t = n(e[r]);
			t != null && (e[i++] = t);
		}
		i < r && (e.length = i), n = -513 & (4 | c), c = n &= -1025, c &= -4097;
	}
	return c !== s && (ft(e, c), 2 & c && Object.freeze(e)), fr(e, c, a, o, t, r, l, i);
}
function fr(e, t, n, r, i, a, o, s) {
	let c = t;
	return a === 1 || a === 4 && (2 & t || !(16 & t) && 32 & r) ? hr(t) || ((t |= !e.length || o && !(4096 & t) || 32 & r && !(4096 & t || 16 & t) ? 2 : 256) !== c && ft(e, t), Object.freeze(e)) : (a === 2 && hr(t) && (e = Zt(e), c = 0, t = Ar(t, r), r = lr(n, r, i, e)), hr(t) || (s || (t |= 16), t !== c && ft(e, t))), 2 & t || !(4096 & t || 16 & t) || nr(n, r), e;
}
function pr(e, t, n) {
	return e = sr(e, t, n), Array.isArray(e) ? e : lt;
}
function mr(e, t) {
	return 2 & t && (e |= 2), 1 | e;
}
function hr(e) {
	return !!(2 & e) && !!(4 & e) || !!(256 & e);
}
function gr(e) {
	return vt(e, !0);
}
function _r(e) {
	e = Zt(e);
	for (let t = 0; t < e.length; t++) {
		let n = e[t] = Zt(e[t]);
		Array.isArray(n[1]) && (n[1] = pt(n[1]));
	}
	return mt(e);
}
function vr(e, t, n, r) {
	tr(e), lr(e = e.v, 0 | e[I], t, (r === "0" ? Number(n) === 0 : n === r) ? void 0 : n);
}
function yr(e, t, n) {
	if (2 & t) throw Error();
	let r = Ct(t), i = pr(e, n, r), a = i === lt ? 7 : 0 | i[I], o = mr(a, t);
	return (2 & o || hr(o) || 16 & o) && (o === a || hr(o) || ft(i, o), i = Zt(i), a = 0, o = Ar(o, t), lr(e, t, n, i, r)), o &= -13, o !== a && ft(i, o), i;
}
function br(e, t) {
	var n = zo;
	return Cr(xr(e = e.v), e, void 0, n) === t ? t : -1;
}
function xr(e) {
	if (Ye) return e[$e] ?? (e[$e] = /* @__PURE__ */ new Map());
	if ($e in e) return e[$e];
	let t = /* @__PURE__ */ new Map();
	return Object.defineProperty(e, $e, { value: t }), t;
}
function Sr(e, t, n, r, i) {
	let a = xr(e), o = Cr(a, e, t, n, i);
	return o !== r && (o && (t = lr(e, t, o, void 0, i)), a.set(n, r)), t;
}
function Cr(e, t, n, r, i) {
	let a = e.get(r);
	if (a != null) return a;
	a = 0;
	for (let e = 0; e < r.length; e++) {
		let o = r[e];
		sr(t, o, i) != null && (a !== 0 && (n = lr(t, n, a, void 0, i)), a = o);
	}
	return e.set(r, a), a;
}
function wr(e, t, n) {
	let r = 0 | e[I], i = Ct(r), a = sr(e, n, i), o;
	if (a != null && a[at] === ht) {
		if (!gt(a)) return er(a), a.v;
		o = a.v;
	} else Array.isArray(a) && (o = a);
	if (o) {
		let e = 0 | o[I];
		2 & e && (o = Qn(o, e));
	}
	return o = Kn(o, t), o !== a && lr(e, r, n, o, i), o;
}
function Tr(e, t, n, r, i) {
	let a = !1;
	if ((r = sr(e, r, i, ((e) => {
		let r = Sn(e, n, !1, t);
		return a = r !== e && r != null, r;
	}))) != null) return a && !gt(r) && nr(e, t), r;
}
function L(e, t, n, r) {
	let i = e.v, a = 0 | i[I];
	if ((t = Tr(i, a, t, n, r)) == null) return t;
	if (a = 0 | i[I], !gt(e, a)) {
		let o = $n(t);
		o !== t && (er(e) && (i = e.v, a = 0 | i[I]), a = lr(i, a, n, t = o, r), nr(i, a));
	}
	return t;
}
function Er(e, t, n, r, i, a, o, s) {
	var c = gt(e, n);
	a = c ? 1 : a, o = !!o || a === 3, c = s && !c, (a === 2 || c) && er(e) && (n = 0 | (t = e.v)[I]);
	var l = (e = pr(t, i)) === lt ? 7 : 0 | e[I], u = mr(l, n);
	if (s = !(4 & u)) {
		var d = e, f = n;
		let t = !!(2 & u);
		t && (f |= 2);
		let i = !t, a = !0, o = 0, s = 0;
		for (; o < d.length; o++) {
			let e = Sn(d[o], r, !1, f);
			if (e instanceof r) {
				if (!t) {
					let t = gt(e);
					i &&= !t, a &&= t;
				}
				d[s++] = e;
			}
		}
		s < o && (d.length = s), u |= 4, u = a ? -4097 & u : 4096 | u, u = i ? 8 | u : -9 & u;
	}
	if (u !== l && (ft(e, u), 2 & u && Object.freeze(e)), c && !(8 & u || !e.length && (a === 1 || a === 4 && (2 & u || !(16 & u) && 32 & n)))) {
		for (hr(u) && (e = Zt(e), u = Ar(u, n), n = lr(t, n, i, e)), r = e, c = u, l = 0; l < r.length; l++) (d = r[l]) !== (u = $n(d)) && (r[l] = u);
		c |= 8, ft(e, u = c = r.length ? 4096 | c : -4097 & c);
	}
	return fr(e, u, t, n, i, a, s, o);
}
function Dr(e, t, n) {
	let r = e.v;
	return Er(e, r, 0 | r[I], t, n, ur(), !1, !0);
}
function Or(e) {
	return e ??= void 0, e;
}
function R(e, t, n, r, i) {
	return cr(e, n, r = Or(r), i), r && !gt(r) && nr(e.v), e;
}
function kr(e, t, n, r) {
	t: {
		var i = r = Or(r);
		tr(e);
		let a = e.v, o = 0 | a[I];
		if (i == null) {
			let e = xr(a);
			if (Cr(e, a, o, n) !== t) break t;
			e.set(n, 0);
		} else o = Sr(a, o, n, t);
		lr(a, o, t, i);
	}
	r && !gt(r) && nr(e.v);
}
function Ar(e, t) {
	return -273 & (2 & t ? 2 | e : -3 & e);
}
function jr(e, t, n, r) {
	var i = r;
	tr(e), e = Er(e, r = e.v, 0 | r[I], n, t, 2, !0), i ??= new n(), e.push(i), t = n = e === lt ? 7 : 0 | e[I], (i = gt(i)) ? (n &= -9, e.length === 1 && (n &= -4097)) : n |= 4096, n !== t && ft(e, n), i || nr(r);
}
function Mr(e, t, n) {
	return dn(or(e, t, void 0, n));
}
function Nr(e, t) {
	return or(e, t, void 0, void 0, on) ?? 0;
}
function Pr(e, t, n) {
	if (n != null) {
		if (typeof n != "number" || !tn(n)) throw Ke("int32");
		n |= 0;
	}
	cr(e, t, n);
}
function z(e, t, n) {
	cr(e, t, an(n));
}
function Fr(e, t, n) {
	vr(e, t, bn(n), "");
}
function Ir(e, t, n) {
	{
		tr(e);
		let o = e.v, s = 0 | o[I];
		if (n == null) lr(o, s, t);
		else {
			var r = e = n === lt ? 7 : 0 | n[I], i = hr(e), a = i || Object.isFrozen(n);
			for (i || (e = 0), a ||= (n = Zt(n), r = 0, e = Ar(e, s), !1), e |= 5, e |= (4 & e ? 512 & e ? 512 : 1024 & e ? 1024 : 0 : void 0) ?? 1024, i = 0; i < n.length; i++) {
				let t = n[i], o = yn(t);
				Object.is(t, o) || (a &&= (n = Zt(n), r = 0, e = Ar(e, s), !1), n[i] = o);
			}
			e !== r && (a && (n = Zt(n), e = Ar(e, s)), ft(n, e)), lr(o, s, t, n);
		}
	}
}
function Lr(e, t, n) {
	tr(e), dr(e, t, xn, 2, !0).push(yn(n));
}
var Rr = class {
	constructor(e, t, n) {
		if (this.buffer = e, n && !t) throw Error();
		this.g = t;
	}
};
function zr(e, t) {
	if (typeof e == "string") return new Rr(Ie(e), t);
	if (Array.isArray(e)) return new Rr(new Uint8Array(e), t);
	if (e.constructor === Uint8Array) return new Rr(e, !1);
	if (e.constructor === ArrayBuffer) return e = new Uint8Array(e), new Rr(e, !1);
	if (e.constructor === Ve) return t = Be(e) || /* @__PURE__ */ new Uint8Array(), new Rr(t, !0, e);
	if (e instanceof Uint8Array) return e = e.constructor === Uint8Array ? e : new Uint8Array(e.buffer, e.byteOffset, e.byteLength), new Rr(e, !1);
	throw Error();
}
function Br(e, t) {
	let n, r = 0, i = 0, a = 0, o = e.h, s = e.g;
	do
		n = o[s++], r |= (127 & n) << a, a += 7;
	while (a < 32 && 128 & n);
	if (a > 32) for (i |= (127 & n) >> 4, a = 3; a < 32 && 128 & n; a += 7) n = o[s++], i |= (127 & n) << a;
	if (Kr(e, s), !(128 & n)) return t(r >>> 0, i >>> 0);
	throw Error();
}
function Vr(e) {
	let t = 0, n = e.g, r = n + 10, i = e.h;
	for (; n < r;) {
		let r = i[n++];
		if (t |= r, !(128 & r)) return Kr(e, n), !!(127 & t);
	}
	throw Error();
}
function Hr(e) {
	let t = e.h, n = e.g, r = t[n++], i = 127 & r;
	if (128 & r && (r = t[n++], i |= (127 & r) << 7, 128 & r && (r = t[n++], i |= (127 & r) << 14, 128 & r && (r = t[n++], i |= (127 & r) << 21, 128 & r && (r = t[n++], i |= r << 28, 128 & r && 128 & t[n++] && 128 & t[n++] && 128 & t[n++] && 128 & t[n++] && 128 & t[n++]))))) throw Error();
	return Kr(e, n), i;
}
function Ur(e) {
	return Hr(e) >>> 0;
}
function Wr(e) {
	var t = e.h;
	let n = e.g;
	var r = t[n], i = t[n + 1];
	let a = t[n + 2];
	return t = t[n + 3], Kr(e, e.g + 4), e = 2 * ((i = (r << 0 | i << 8 | a << 16 | t << 24) >>> 0) >> 31) + 1, r = i >>> 23 & 255, i &= 8388607, r == 255 ? i ? NaN : e * Infinity : r == 0 ? 1401298464324817e-60 * e * i : e * 2 ** (r - 150) * (i + 8388608);
}
function Gr(e) {
	return Hr(e);
}
function Kr(e, t) {
	if (e.g = t, t > e.l) throw Error();
}
function qr(e, t) {
	if (t < 0) throw Error();
	let n = e.g;
	if ((t = n + t) > e.l) throw Error();
	return e.g = t, n;
}
function Jr(e, t) {
	if (t == 0) return ze();
	var n = qr(e, t);
	return e.Y && e.j ? n = e.h.subarray(n, n + t) : (e = e.h, n = n === (t = n + t) ? /* @__PURE__ */ new Uint8Array() : It ? e.slice(n, t) : new Uint8Array(e.subarray(n, t))), n.length == 0 ? ze() : new Ve(n, Re);
}
var Yr = [];
function Xr(e, t, n, r) {
	if (ai.length) {
		let i = ai.pop();
		return i.o(r), i.g.init(e, t, n, r), i;
	}
	return new ii(e, t, n, r);
}
function Zr(e) {
	e.g.clear(), e.l = -1, e.h = -1, ai.length < 100 && ai.push(e);
}
function Qr(e) {
	var t = e.g;
	if (t.g == t.l) return !1;
	e.m = e.g.g;
	var n = Ur(e.g);
	if (t = n >>> 3, !((n &= 7) >= 0 && n <= 5) || t < 1) throw Error();
	return e.l = t, e.h = n, !0;
}
function $r(e) {
	switch (e.h) {
		case 0:
			e.h == 0 ? Vr(e.g) : $r(e);
			break;
		case 1:
			Kr(e = e.g, e.g + 8);
			break;
		case 2:
			if (e.h != 2) $r(e);
			else {
				var t = Ur(e.g);
				Kr(e = e.g, e.g + t);
			}
			break;
		case 5:
			Kr(e = e.g, e.g + 4);
			break;
		case 3:
			for (t = e.l;;) {
				if (!Qr(e)) throw Error();
				if (e.h == 4) {
					if (e.l != t) throw Error();
					break;
				}
				$r(e);
			}
			break;
		default: throw Error();
	}
}
function ei(e, t, n) {
	let r = e.g.l;
	var i = Ur(e.g);
	let a = (i = e.g.g + i) - r;
	if (a <= 0 && (e.g.l = i, n(t, e, void 0, void 0, void 0), a = i - e.g.g), a) throw Error();
	return e.g.g = i, e.g.l = r, t;
}
function ti(e) {
	var t = Ur(e.g), n = qr(e = e.g, t);
	if (e = e.h, be) {
		var r, i = e;
		(r = ye) || (r = ye = new TextDecoder("utf-8", { fatal: !0 })), t = n + t, i = n === 0 && t === i.length ? i : i.subarray(n, t);
		try {
			var a = r.decode(i);
		} catch (e) {
			if (ve === void 0) {
				try {
					r.decode(new Uint8Array([128]));
				} catch {}
				try {
					r.decode(new Uint8Array([97])), ve = !0;
				} catch {
					ve = !1;
				}
			}
			throw !ve && (ye = void 0), e;
		}
	} else {
		t = (a = n) + t, n = [];
		let s, c = null;
		for (; a < t;) {
			var o = e[a++];
			o < 128 ? n.push(o) : o < 224 ? a >= t ? ge() : (s = e[a++], o < 194 || (192 & s) != 128 ? (a--, ge()) : n.push((31 & o) << 6 | 63 & s)) : o < 240 ? a >= t - 1 ? ge() : (s = e[a++], (192 & s) != 128 || o === 224 && s < 160 || o === 237 && s >= 160 || (192 & (r = e[a++])) != 128 ? (a--, ge()) : n.push((15 & o) << 12 | (63 & s) << 6 | 63 & r)) : o <= 244 ? a >= t - 2 ? ge() : (s = e[a++], (192 & s) != 128 || s - 144 + (o << 28) >> 30 || (192 & (r = e[a++])) != 128 || (192 & (i = e[a++])) != 128 ? (a--, ge()) : (o = (7 & o) << 18 | (63 & s) << 12 | (63 & r) << 6 | 63 & i, o -= 65536, n.push(55296 + (o >> 10 & 1023), 56320 + (1023 & o)))) : ge(), n.length >= 8192 && (c = _e(c, n), n.length = 0);
		}
		a = _e(c, n);
	}
	return a;
}
function ni(e) {
	let t = Ur(e.g);
	return Jr(e.g, t);
}
function ri(e, t, n) {
	var r = Ur(e.g);
	for (r = e.g.g + r; e.g.g < r;) n.push(t(e.g));
}
var ii = class {
	constructor(e, t, n, r) {
		if (Yr.length) {
			let i = Yr.pop();
			i.init(e, t, n, r), e = i;
		} else e = new class {
			constructor(e, t, n, r) {
				this.h = null, this.j = !1, this.g = this.l = this.m = 0, this.init(e, t, n, r);
			}
			init(e, t, n, { Y: r = !1, ea: i = !1 } = {}) {
				this.Y = r, this.ea = i, e && (e = zr(e, this.ea), this.h = e.buffer, this.j = e.g, this.m = t || 0, this.l = n === void 0 ? this.h.length : this.m + n, this.g = this.m);
			}
			clear() {
				this.h = null, this.j = !1, this.g = this.l = this.m = 0, this.Y = !1;
			}
		}(e, t, n, r);
		this.g = e, this.m = this.g.g, this.h = this.l = -1, this.o(r);
	}
	o({ ha: e = !1 } = {}) {
		this.ha = e;
	}
}, ai = [];
function oi(e) {
	return e ? /^\d+$/.test(e) ? (Yt(e), new si(Rt, zt)) : null : ci ||= new si(0, 0);
}
var si = class {
	constructor(e, t) {
		this.h = e >>> 0, this.g = t >>> 0;
	}
}, ci;
function li(e) {
	return e ? /^-?\d+$/.test(e) ? (Yt(e), new ui(Rt, zt)) : null : di ||= new ui(0, 0);
}
var ui = class {
	constructor(e, t) {
		this.h = e >>> 0, this.g = t >>> 0;
	}
}, di;
function fi(e, t, n) {
	for (; n > 0 || t > 127;) e.g.push(127 & t | 128), t = (t >>> 7 | n << 25) >>> 0, n >>>= 7;
	e.g.push(t);
}
function pi(e, t) {
	for (; t > 127;) e.g.push(127 & t | 128), t >>>= 7;
	e.g.push(t);
}
function mi(e, t) {
	if (t >= 0) pi(e, t);
	else {
		for (let n = 0; n < 9; n++) e.g.push(127 & t | 128), t >>= 7;
		e.g.push(1);
	}
}
function hi(e) {
	var t = Rt;
	e.g.push(t >>> 0 & 255), e.g.push(t >>> 8 & 255), e.g.push(t >>> 16 & 255), e.g.push(t >>> 24 & 255);
}
function gi(e, t) {
	t.length !== 0 && (e.l.push(t), e.h += t.length);
}
function _i(e, t, n) {
	pi(e.g, 8 * t + n);
}
function vi(e, t) {
	return _i(e, t, 2), t = e.g.end(), gi(e, t), t.push(e.h), t;
}
function yi(e, t) {
	var n = t.pop();
	for (n = e.h + e.g.length() - n; n > 127;) t.push(127 & n | 128), n >>>= 7, e.h++;
	t.push(n), e.h++;
}
function bi(e, t, n) {
	_i(e, t, 2), pi(e.g, n.length), gi(e, e.g.end()), gi(e, n);
}
function xi(e, t, n, r) {
	n != null && (t = vi(e, t), r(n, e), yi(e, t));
}
function Si() {
	let e = class {
		constructor() {
			throw Error();
		}
	};
	return Object.setPrototypeOf(e, e.prototype), e;
}
var Ci = Si(), wi = Si(), Ti = Si(), Ei = Si(), Di = Si(), Oi = Si(), ki = Si(), Ai = Si(), ji = Si(), Mi = Si();
function Ni(e, t, n) {
	var r = e.v;
	et && et in r && (r = r[et]) && delete r[t.g], t.h ? t.j(e, t.h, t.g, n, t.l) : t.j(e, t.g, n, t.l);
}
var B = class {
	constructor(e, t) {
		this.v = qn(e, t, void 0, 2048);
	}
	toJSON() {
		return Gn(this);
	}
	j() {
		var e = js, t = this.v, n = e.g, r = et;
		if (Ye && r && t[r]?.[n] != null && qe(tt, 3), t = e.g, ot && et && ot === void 0 && (r = (n = this.v)[et]) && (r = r.da)) try {
			r(n, t, Rn);
		} catch (e) {
			we(e);
		}
		return e.h ? e.m(this, e.h, e.g, e.l) : e.m(this, e.g, e.defaultValue, e.l);
	}
	clone() {
		let e = this.v, t = 0 | e[I];
		return rr(this, e, t) ? Xn(this, e, !0) : new this.constructor(Qn(e, t, !1));
	}
};
B.prototype[at] = ht, B.prototype.toString = function() {
	return this.v.toString();
};
var Pi = class {
	constructor(e, t, n) {
		this.g = e, this.h = t, e = Ci, this.l = !!e && n === e || !1;
	}
};
function Fi(e, t) {
	return new Pi(e, t, Ci);
}
function Ii(e, t, n, r, i) {
	xi(e, n, Ji(t, r), i);
}
var Li = Fi((function(e, t, n, r, i) {
	return e.h === 2 && (ei(e, wr(t, r, n), i), !0);
}), Ii), Ri = Fi((function(e, t, n, r, i) {
	return e.h === 2 && (ei(e, wr(t, r, n), i), !0);
}), Ii), zi = Symbol(), Bi = Symbol(), Vi = Symbol(), Hi = Symbol(), Ui = Symbol(), Wi, Gi;
function Ki(e, t, n, r) {
	var i = r[e];
	if (i) return i;
	(i = {}).qa = r, i.T = function(e) {
		switch (typeof e) {
			case "boolean": return Un ||= [
				0,
				void 0,
				!0
			];
			case "number": return e > 0 ? void 0 : e === 0 ? Wn ||= [0, void 0] : [-e, void 0];
			case "string": return [0, e];
			case "object": return e;
		}
	}(r[0]);
	var a = r[1];
	let o = 1;
	a && a.constructor === Object && (i.ba = a, typeof (a = r[++o]) == "function" && (i.ma = !0, Wi ??= a, Gi ??= r[o + 1], a = r[o += 2]));
	let s = {};
	for (; a && Array.isArray(a) && a.length && typeof a[0] == "number" && a[0] > 0;) {
		for (var c = 0; c < a.length; c++) s[a[c]] = a;
		a = r[++o];
	}
	for (c = 1; a !== void 0;) {
		let e;
		typeof a == "number" && (c += a, a = r[++o]);
		var l = void 0;
		if (a instanceof Pi ? e = a : (e = Li, o--), e?.l) {
			a = r[++o], l = r;
			var u = o;
			typeof a == "function" && (a = a(), l[u] = a), l = a;
		}
		for (u = c + 1, typeof (a = r[++o]) == "number" && a < 0 && (u -= a, a = r[++o]); c < u; c++) {
			let r = s[c];
			l ? n(i, c, e, l, r) : t(i, c, e, r);
		}
	}
	return r[e] = i;
}
function qi(e) {
	return Array.isArray(e) ? e[0] instanceof Pi ? e : [Ri, e] : [e, void 0];
}
function Ji(e, t) {
	return e instanceof B ? e.v : Array.isArray(e) ? Kn(e, t) : void 0;
}
function Yi(e, t, n, r) {
	let i = n.g;
	e[t] = r ? (e, t, n) => i(e, t, n, r) : i;
}
function Xi(e, t, n, r, i) {
	let a = n.g, o, s;
	e[t] = (e, t, n) => a(e, t, n, s ||= Ki(Bi, Yi, Xi, r).T, o ||= Zi(r), i);
}
function Zi(e) {
	let t = e[Vi];
	if (t != null) return t;
	let n = Ki(Bi, Yi, Xi, e);
	return t = n.ma ? (e, t) => Wi(e, t, n) : (e, t) => {
		for (; Qr(t) && t.h != 4;) {
			var r = t.l, i = n[r];
			if (i == null) {
				var a = n.ba;
				(a &&= a[r]) && (a = $i(a)) != null && (i = n[r] = a);
			}
			if (i == null || !i(t, e, r)) {
				if (i = (a = t).m, $r(a), a.ha) var o = void 0;
				else o = a.g.g - i, a.g.g = i, o = Jr(a.g, o);
				i = void 0, a = e, o && ((i = a[et] ?? (a[et] = new Ln()))[r] ?? (i[r] = [])).push(o);
			}
		}
		return (e = Fn(e)) && (e.da = n.qa[Ui]), !0;
	}, e[Vi] = t, e[Ui] = Qi.bind(e), t;
}
function Qi(e, t, n, r) {
	var i = this[Bi];
	let a = this[Vi], o = Kn(void 0, i.T), s = Fn(e);
	if (s) {
		var c = !1, l = i.ba;
		if (l) {
			if (i = (t, n, i) => {
				if (i.length !== 0) if (l[n]) for (let e of i) {
					t = Xr(e);
					try {
						c = !0, a(o, t);
					} finally {
						Zr(t);
					}
				}
				else r?.(e, n, i);
			}, t == null) In(s, i);
			else if (s != null) {
				let e = s[t];
				e && i(s, t, e);
			}
			if (c) {
				let r = 0 | e[I];
				if (2 & r && 2048 & r && !n?.Ka) throw Error();
				let i = Ct(r), a = (t, a) => {
					if (sr(e, t, i) != null) {
						if (n?.Qa === 1) return;
						throw Error();
					}
					a != null && (r = lr(e, r, t, a, i)), delete s[t];
				};
				t == null ? xt(o, 0 | o[I], ((e, t) => {
					a(e, t);
				})) : a(t, sr(o, t, i));
			}
		}
	}
}
function $i(e) {
	let t = (e = qi(e))[0].g;
	if (e = e[1]) {
		let n = Zi(e), r = Ki(Bi, Yi, Xi, e).T;
		return (e, i, a) => t(e, i, a, r, n);
	}
	return t;
}
function ea(e, t, n) {
	e[t] = n.h;
}
function ta(e, t, n, r) {
	let i, a, o = n.h;
	e[t] = (e, t, n) => o(e, t, n, a ||= Ki(zi, ea, ta, r).T, i ||= na(r));
}
function na(e) {
	let t = e[Hi];
	if (!t) {
		let n = Ki(zi, ea, ta, e);
		t = (e, t) => ra(e, t, n), e[Hi] = t;
	}
	return t;
}
function ra(e, t, n) {
	xt(e, 0 | e[I], ((e, r) => {
		if (r != null) {
			var i = function(e, t) {
				var n = e[t];
				if (n) return n;
				if ((n = e.ba) && (n = n[t])) {
					var r = (n = qi(n))[0].h;
					if (n = n[1]) {
						let t = na(n), i = Ki(zi, ea, ta, n).T;
						n = e.ma ? Gi(i, t) : (e, n, a) => r(e, n, a, i, t);
					} else n = r;
					return e[t] = n;
				}
			}(n, e);
			i ? i(t, r, e) : e < 500 || qe(rt, 3);
		}
	})), (e = Fn(e)) && In(e, ((e, n, r) => {
		for (gi(t, t.g.end()), e = 0; e < r.length; e++) gi(t, Be(r[e]) || /* @__PURE__ */ new Uint8Array());
	}));
}
var ia = kt(0);
function aa(e, t) {
	if (Array.isArray(t)) {
		var n = 0 | t[I];
		if (4 & n) return t;
		for (var r = 0, i = 0; r < t.length; r++) {
			let n = e(t[r]);
			n != null && (t[i++] = n);
		}
		return i < r && (t.length = i), (e = -1537 & (5 | n)) !== n && ft(t, e), 2 & e && Object.freeze(t), t;
	}
}
function oa(e, t, n) {
	return new Pi(e, t, n);
}
function sa(e, t, n) {
	return new Pi(e, t, n);
}
function ca(e, t, n) {
	lr(e, 0 | e[I], t, n, Ct(0 | e[I]));
}
var la = Fi((function(e, t, n, r, i) {
	if (e.h !== 2) return !1;
	if (e = Zt(e = ei(e, Kn([void 0, void 0], r), i)), i = Ct(r = 0 | t[I]), 2 & r) throw Error();
	let a = sr(t, n, i);
	if (a instanceof An) 2 & a.J ? (a = a.V(), a.push(e), lr(t, r, n, a, i)) : a.Ma(e);
	else if (Array.isArray(a)) {
		var o = 0 | a[I];
		8192 & o || ft(a, o |= 8192), 2 & o && (a = _r(a), lr(t, r, n, a, i)), a.push(e);
	} else lr(t, r, n, mt([e]), i);
	return !0;
}), (function(e, t, n, r, i) {
	if (t instanceof An) t.forEach(((t, a) => {
		xi(e, n, Kn([a, t], r), i);
	}));
	else if (Array.isArray(t)) {
		for (let a = 0; a < t.length; a++) {
			let o = t[a];
			Array.isArray(o) && xi(e, n, Kn(o, r), i);
		}
		mt(t);
	}
}));
function ua(e, t, n) {
	(t = on(t)) != null && (_i(e, n, 5), e = e.g, Ht(t), hi(e));
}
function da(e, t, n) {
	if (t = function(e) {
		if (e == null) return e;
		let t = typeof e;
		if (t === "bigint") return String(Qt(64, e));
		if (un(e)) {
			if (t === "string") return hn(e);
			if (t === "number") return mn(e);
		}
	}(t), t != null && (typeof t == "string" && li(t), t != null)) switch (_i(e, n, 0), typeof t) {
		case "number":
			e = e.g, Vt(t), fi(e, Rt, zt);
			break;
		case "bigint":
			n = BigInt.asUintN(64, t), n = new ui(Number(n & BigInt(4294967295)), Number(n >> BigInt(32))), fi(e.g, n.h, n.g);
			break;
		default: n = li(t), fi(e.g, n.h, n.g);
	}
}
function fa(e, t, n) {
	(t = dn(t)) != null && t != null && (_i(e, n, 0), mi(e.g, t));
}
function pa(e, t, n) {
	(t = cn(t)) != null && (_i(e, n, 0), e.g.g.push(+!!t));
}
function ma(e, t, n) {
	(t = xn(t)) != null && bi(e, n, Ce(t));
}
function ha(e, t, n, r, i) {
	xi(e, n, Ji(t, r), i);
}
function ga(e, t, n) {
	(t = t == null || typeof t == "string" || t instanceof Ve ? t : void 0) != null && bi(e, n, zr(t, !0).buffer);
}
function _a(e, t, n) {
	(t = fn(t)) != null && t != null && (_i(e, n, 0), pi(e.g, t));
}
function va(e, t, n) {
	return (e.h === 5 || e.h === 2) && (t = yr(t, 0 | t[I], n), e.h == 2 ? ri(e, Wr, t) : t.push(Wr(e.g)), !0);
}
var ya = oa((function(e, t, n) {
	return e.h === 5 && (ca(t, n, Wr(e.g)), !0);
}), ua, Ai), ba = sa(va, (function(e, t, n) {
	if ((t = aa(on, t)) != null) for (let o = 0; o < t.length; o++) {
		var r = e, i = n, a = t[o];
		a != null && (_i(r, i, 5), r = r.g, Ht(a), hi(r));
	}
}), Ai), xa = sa(va, (function(e, t, n) {
	if ((t = aa(on, t)) != null && t.length) {
		_i(e, n, 2), pi(e.g, 4 * t.length);
		for (let r = 0; r < t.length; r++) n = e.g, Ht(t[r]), hi(n);
	}
}), Ai), Sa = oa((function(e, t, n) {
	return e.h === 5 && (ca(t, n, (e = Wr(e.g)) === 0 ? void 0 : e), !0);
}), ua, Ai), Ca = oa((function(e, t, n) {
	return e.h === 0 ? (ca(t, n, Br(e.g, Gt)), e = !0) : e = !1, e;
}), da, Oi), wa = oa((function(e, t, n) {
	return e.h === 0 ? (ca(t, n, (e = Br(e.g, Gt)) === ia ? void 0 : e), t = !0) : t = !1, t;
}), da, Oi), Ta = oa((function(e, t, n) {
	return e.h === 0 ? (ca(t, n, Br(e.g, Wt)), e = !0) : e = !1, e;
}), (function(e, t, n) {
	if (t = function(e) {
		if (e == null) return e;
		var t = typeof e;
		if (t === "bigint") return String($t(64, e));
		if (un(e)) {
			if (t === "string") return t = nn(Number(e)), en(t) && t >= 0 ? e = String(t) : ((t = e.indexOf(".")) !== -1 && (e = e.substring(0, t)), (t = e[0] !== "-" && ((t = e.length) < 20 || t === 20 && e <= "18446744073709551615")) || (Yt(e), e = Kt(Rt, zt))), e;
			if (t === "number") return (e = nn(e)) >= 0 && en(e) || (Vt(e), e = Ut(Rt, zt)), e;
		}
	}(t), t != null && (typeof t == "string" && oi(t), t != null)) switch (_i(e, n, 0), typeof t) {
		case "number":
			e = e.g, Vt(t), fi(e, Rt, zt);
			break;
		case "bigint":
			n = BigInt.asUintN(64, t), n = new si(Number(n & BigInt(4294967295)), Number(n >> BigInt(32))), fi(e.g, n.h, n.g);
			break;
		default: n = oi(t), fi(e.g, n.h, n.g);
	}
}), ki), Ea = oa((function(e, t, n) {
	return e.h === 0 && (ca(t, n, Hr(e.g)), !0);
}), fa, Ei), Da = sa((function(e, t, n) {
	return (e.h === 0 || e.h === 2) && (t = yr(t, 0 | t[I], n), e.h == 2 ? ri(e, Hr, t) : t.push(Hr(e.g)), !0);
}), (function(e, t, n) {
	if ((t = aa(dn, t)) != null && t.length) {
		n = vi(e, n);
		for (let n = 0; n < t.length; n++) mi(e.g, t[n]);
		yi(e, n);
	}
}), Ei), Oa = oa((function(e, t, n) {
	return e.h === 0 && (ca(t, n, (e = Hr(e.g)) === 0 ? void 0 : e), !0);
}), fa, Ei), ka = oa((function(e, t, n) {
	return e.h === 0 && (ca(t, n, Vr(e.g)), !0);
}), pa, wi), Aa = oa((function(e, t, n) {
	return e.h === 0 && (ca(t, n, !1 === (e = Vr(e.g)) ? void 0 : e), !0);
}), pa, wi), ja = sa((function(e, t, n) {
	return e.h === 2 && (e = ti(e), yr(t, 0 | t[I], n).push(e), !0);
}), (function(e, t, n) {
	if ((t = aa(xn, t)) != null) for (let o = 0; o < t.length; o++) {
		var r = e, i = n, a = t[o];
		a != null && bi(r, i, Ce(a));
	}
}), Ti), Ma = oa((function(e, t, n) {
	return e.h === 2 && (ca(t, n, (e = ti(e)) === "" ? void 0 : e), !0);
}), ma, Ti), Na = oa((function(e, t, n) {
	return e.h === 2 && (ca(t, n, ti(e)), !0);
}), ma, Ti), Pa = function(e, t, n = Ci) {
	return new Pi(e, t, n);
}((function(e, t, n, r, i) {
	return e.h === 2 && (r = Kn(void 0, r), yr(t, 0 | t[I], n).push(r), ei(e, r, i), !0);
}), (function(e, t, n, r, i) {
	if (Array.isArray(t)) {
		for (let a = 0; a < t.length; a++) ha(e, t[a], n, r, i);
		1 & (e = 0 | t[I]) || ft(t, 1 | e);
	}
})), Fa = Fi((function(e, t, n, r, i, a) {
	if (e.h !== 2) return !1;
	let o = 0 | t[I];
	return Sr(t, o, a, n, Ct(o)), ei(e, t = wr(t, r, n), i), !0;
}), ha), Ia = oa((function(e, t, n) {
	return e.h === 2 && (ca(t, n, ni(e)), !0);
}), ga, ji), La = sa((function(e, t, n) {
	return (e.h === 0 || e.h === 2) && (t = yr(t, 0 | t[I], n), e.h == 2 ? ri(e, Ur, t) : t.push(Ur(e.g)), !0);
}), (function(e, t, n) {
	if ((t = aa(fn, t)) != null) for (let o = 0; o < t.length; o++) {
		var r = e, i = n, a = t[o];
		a != null && (_i(r, i, 0), pi(r.g, a));
	}
}), Di), Ra = oa((function(e, t, n) {
	return e.h === 0 && (ca(t, n, (e = Ur(e.g)) === 0 ? void 0 : e), !0);
}), _a, Di), za = oa((function(e, t, n) {
	return e.h === 0 && (ca(t, n, Hr(e.g)), !0);
}), (function(e, t, n) {
	(t = dn(t)) != null && (t = parseInt(t, 10), _i(e, n, 0), mi(e.g, t));
}), Mi), Ba = class {
	constructor(e, t) {
		var n = io;
		this.g = e, this.h = t, this.m = L, this.j = R, this.defaultValue = void 0, this.l = n.Oa == null ? void 0 : St;
	}
	register() {
		N(this);
	}
};
function Va(e, t) {
	return new Ba(e, t);
}
function Ha(e, t) {
	return (n, r) => {
		{
			let a = { ea: !0 };
			r && Object.assign(a, r), n = Xr(n, void 0, void 0, a);
			try {
				let r = new e(), a = r.v;
				Zi(t)(a, n);
				var i = r;
			} finally {
				Zr(n);
			}
		}
		return i;
	};
}
function Ua(e) {
	return function() {
		let t = new class {
			constructor() {
				this.l = [], this.h = 0, this.g = new class {
					constructor() {
						this.g = [];
					}
					length() {
						return this.g.length;
					}
					end() {
						let e = this.g;
						return this.g = [], e;
					}
				}();
			}
		}();
		ra(this.v, t, Ki(zi, ea, ta, e)), gi(t, t.g.end());
		let n = new Uint8Array(t.h), r = t.l, i = r.length, a = 0;
		for (let e = 0; e < i; e++) {
			let t = r[e];
			n.set(t, a), a += t.length;
		}
		return t.l = [n], n;
	};
}
var Wa = class extends B {
	constructor(e) {
		super(e);
	}
}, Ga = [
	0,
	Ma,
	oa((function(e, t, n) {
		return e.h === 2 && (ca(t, n, (e = ni(e)) === ze() ? void 0 : e), !0);
	}), (function(e, t, n) {
		if (t != null) {
			if (t instanceof B) {
				let r = t.Ra;
				r ? (t = r(t), t != null && bi(e, n, zr(t, !0).buffer)) : qe(rt, 3);
				return;
			}
			if (Array.isArray(t)) return void qe(rt, 3);
		}
		ga(e, t, n);
	}), ji)
], Ka, qa = globalThis.trustedTypes;
function Ja(e) {
	var t;
	return Ka === void 0 && (Ka = function() {
		let e = null;
		if (!qa) return e;
		try {
			let t = (e) => e;
			e = qa.createPolicy("goog#html", {
				createHTML: t,
				createScript: t,
				createScriptURL: t
			});
		} catch {}
		return e;
	}()), e = (t = Ka) ? t.createScriptURL(e) : e, new class {
		constructor(e) {
			this.g = e;
		}
		toString() {
			return this.g + "";
		}
	}(e);
}
function Ya(e, ...t) {
	if (t.length === 0) return Ja(e[0]);
	let n = e[0];
	for (let r = 0; r < t.length; r++) n += encodeURIComponent(t[r]) + e[r + 1];
	return Ja(n);
}
var Xa = [
	0,
	Ea,
	za,
	ka,
	-1,
	Da,
	za,
	-1,
	ka
], Za = class extends B {
	constructor(e) {
		super(e);
	}
}, Qa = [
	0,
	ka,
	Na,
	ka,
	za,
	-1,
	sa((function(e, t, n) {
		return (e.h === 0 || e.h === 2) && (t = yr(t, 0 | t[I], n), e.h == 2 ? ri(e, Gr, t) : t.push(Hr(e.g)), !0);
	}), (function(e, t, n) {
		if ((t = aa(dn, t)) != null && t.length) {
			n = vi(e, n);
			for (let n = 0; n < t.length; n++) mi(e.g, t[n]);
			yi(e, n);
		}
	}), Mi),
	Na,
	-1,
	[
		0,
		ka,
		-1
	],
	za,
	ka,
	-1
], $a = [
	0,
	3,
	ka,
	-1,
	2,
	[
		0,
		[2],
		Ea,
		Fa,
		[0, oa((function(e, t, n) {
			return e.h === 0 && (ca(t, n, Ur(e.g)), !0);
		}), _a, Di)]
	],
	[
		0,
		za,
		ka,
		za,
		ka,
		za,
		ka,
		Na,
		-1
	],
	[
		0,
		[3, 4],
		Na,
		-1,
		Fa,
		[0, Ea],
		Fa,
		[0, za]
	],
	[0]
], eo = [
	0,
	Na,
	-2
], to = class extends B {
	constructor(e) {
		super(e);
	}
}, no = [0], ro = [
	0,
	Ea,
	ka,
	1,
	ka,
	-4
], io = class extends B {
	constructor(e) {
		super(e, 2);
	}
}, ao = {};
ao[336783863] = [
	0,
	Na,
	ka,
	-1,
	Ea,
	[
		0,
		[
			1,
			2,
			3,
			4,
			5,
			6,
			7,
			8,
			9
		],
		Fa,
		no,
		Fa,
		Qa,
		Fa,
		eo,
		Fa,
		ro,
		Fa,
		Xa,
		Fa,
		[
			0,
			Na,
			-2
		],
		Fa,
		[
			0,
			Na,
			za
		],
		Fa,
		$a,
		Fa,
		[
			0,
			za,
			-1,
			ka
		]
	],
	[0, Na],
	ka,
	[
		0,
		[1, 3],
		[2, 4],
		Fa,
		[0, Da],
		-1,
		Fa,
		[0, ja],
		-1,
		Pa,
		[
			0,
			Na,
			-1
		]
	],
	Na
];
var oo = [
	0,
	wa,
	-1,
	Aa,
	-3,
	wa,
	Da,
	Ma,
	Oa,
	wa,
	-1,
	Aa,
	Oa,
	Aa,
	-2,
	Ma
];
function so(e, t) {
	Lr(e, 3, t);
}
function V(e, t) {
	Lr(e, 4, t);
}
var co = class extends B {
	constructor(e) {
		super(e, 500);
	}
	o(e) {
		return R(this, 0, 7, e);
	}
}, lo = [-1, {}], uo = [
	0,
	Na,
	1,
	lo
], fo = [
	0,
	Na,
	ja,
	lo
];
function po(e, t) {
	jr(e, 1, co, t);
}
function mo(e, t) {
	Lr(e, 10, t);
}
function ho(e, t) {
	Lr(e, 15, t);
}
var go = class extends B {
	constructor(e) {
		super(e, 500);
	}
	o(e) {
		return R(this, 0, 1001, e);
	}
}, _o = [
	-500,
	Pa,
	[
		-500,
		Ma,
		-1,
		ja,
		-3,
		[
			-2,
			ao,
			ka
		],
		Pa,
		Ga,
		Oa,
		-1,
		uo,
		fo,
		Pa,
		[
			0,
			Ma,
			Aa
		],
		Ma,
		oo,
		Oa,
		ja,
		987,
		ja
	],
	4,
	Pa,
	[
		-500,
		Na,
		-1,
		[-1, {}],
		998,
		Na
	],
	Pa,
	[
		-500,
		Na,
		ja,
		-1,
		[
			-2,
			{},
			ka
		],
		997,
		ja,
		-1
	],
	Oa,
	Pa,
	[
		-500,
		Na,
		ja,
		lo,
		998,
		ja
	],
	ja,
	Oa,
	uo,
	fo,
	Pa,
	[
		0,
		Ma,
		-1,
		lo
	],
	ja,
	-2,
	oo,
	Ma,
	-1,
	Aa,
	[
		0,
		Aa,
		Ra
	],
	978,
	lo,
	Pa,
	Ga
];
go.prototype.g = Ua(_o);
var vo = Ha(go, _o), yo = class extends B {
	constructor(e) {
		super(e);
	}
}, bo = class extends B {
	constructor(e) {
		super(e);
	}
	g() {
		return Dr(this, yo, 1);
	}
}, xo = [
	0,
	Pa,
	[
		0,
		Ea,
		ya,
		Na,
		-1
	]
], So = Ha(bo, xo), Co = class extends B {
	constructor(e) {
		super(e);
	}
}, wo = class extends B {
	constructor(e) {
		super(e);
	}
}, To = class extends B {
	constructor(e) {
		super(e);
	}
	l() {
		return L(this, Co, 2);
	}
	g() {
		return Dr(this, wo, 5);
	}
}, Eo = Ha(class extends B {
	constructor(e) {
		super(e);
	}
}, [
	0,
	ja,
	Da,
	xa,
	[
		0,
		za,
		[
			0,
			Ea,
			-3
		],
		[
			0,
			ya,
			-3
		],
		[
			0,
			Ea,
			-1,
			[
				0,
				Pa,
				[
					0,
					Ea,
					-2
				]
			]
		],
		Pa,
		[
			0,
			ya,
			-1,
			Na,
			ya
		]
	],
	Na,
	-1,
	Ca,
	Pa,
	[
		0,
		Ea,
		ya
	],
	ja,
	Ca
]), Do = class extends B {
	constructor(e) {
		super(e);
	}
}, Oo = Ha(class extends B {
	constructor(e) {
		super(e);
	}
}, [
	0,
	Pa,
	[
		0,
		ya,
		-4
	]
]), ko = class extends B {
	constructor(e) {
		super(e);
	}
}, Ao = Ha(class extends B {
	constructor(e) {
		super(e);
	}
}, [
	0,
	Pa,
	[
		0,
		ya,
		-4
	]
]), jo = class extends B {
	constructor(e) {
		super(e);
	}
}, Mo = [
	0,
	Ea,
	-1,
	xa,
	za
], No = class extends B {
	constructor(e) {
		super(e);
	}
};
No.prototype.g = Ua([
	0,
	ya,
	-4,
	Ca
]);
var Po = class extends B {
	constructor(e) {
		super(e);
	}
}, Fo = Ha(class extends B {
	constructor(e) {
		super(e);
	}
}, [
	0,
	Pa,
	[
		0,
		1,
		Ea,
		Na,
		xo
	],
	Ca
]), Io = class extends B {
	constructor(e) {
		super(e);
	}
}, Lo = class extends B {
	constructor(e) {
		super(e);
	}
	na() {
		return or(this, 1, void 0, void 0, gr) ?? ze();
	}
}, Ro = class extends B {
	constructor(e) {
		super(e);
	}
}, zo = [1, 2], Bo = Ha(class extends B {
	constructor(e) {
		super(e);
	}
}, [
	0,
	Pa,
	[
		0,
		zo,
		Fa,
		[0, xa],
		Fa,
		[0, Ia],
		Ea,
		Na
	],
	Ca
]), Vo = class extends B {
	constructor(e) {
		super(e);
	}
}, Ho = [
	0,
	Na,
	Ea,
	ya,
	ja,
	-1
], Uo = class extends B {
	constructor(e) {
		super(e);
	}
}, Wo = [
	0,
	ka,
	-1
], Go = class extends B {
	constructor(e) {
		super(e);
	}
}, Ko = [
	1,
	2,
	3,
	4,
	5,
	6
], qo = class extends B {
	constructor(e) {
		super(e);
	}
	g() {
		return or(this, 1, void 0, void 0, gr) != null;
	}
	l() {
		return xn(or(this, 2)) != null;
	}
}, Jo = class extends B {
	constructor(e) {
		super(e);
	}
	g() {
		return cn(or(this, 2)) ?? !1;
	}
}, Yo = [
	0,
	Ia,
	Na,
	[
		0,
		Ea,
		Ca,
		-1
	],
	[
		0,
		Ta,
		Ca
	]
], Xo = [
	0,
	Yo,
	ka,
	[
		0,
		Ko,
		Fa,
		ro,
		Fa,
		Qa,
		Fa,
		Xa,
		Fa,
		no,
		Fa,
		eo,
		Fa,
		$a
	],
	za
], Zo = class extends B {
	constructor(e) {
		super(e);
	}
}, Qo = [
	0,
	Xo,
	ya,
	-1,
	Ea
], $o = Va(502141897, Zo);
ao[502141897] = Qo;
var es = Ha(class extends B {
	constructor(e) {
		super(e);
	}
}, [
	0,
	[
		0,
		za,
		-1,
		ba,
		La
	],
	Mo
]), ts = class extends B {
	constructor(e) {
		super(e);
	}
}, ns = class extends B {
	constructor(e) {
		super(e);
	}
}, rs = [
	0,
	Xo,
	ya,
	[0, Xo],
	ka
], is = Va(508968150, ns);
ao[508968150] = [
	0,
	Xo,
	Qo,
	rs,
	ya,
	[0, [0, Yo]]
], ao[508968149] = rs;
var as = class extends B {
	constructor(e) {
		super(e);
	}
	l() {
		return L(this, Vo, 2);
	}
	g() {
		cr(this, 2);
	}
}, os = [
	0,
	Xo,
	Ho
];
ao[478825465] = os;
var ss = class extends B {
	constructor(e) {
		super(e);
	}
}, cs = class extends B {
	constructor(e) {
		super(e);
	}
}, ls = class extends B {
	constructor(e) {
		super(e);
	}
}, us = class extends B {
	constructor(e) {
		super(e);
	}
}, ds = class extends B {
	constructor(e) {
		super(e);
	}
}, fs = [
	0,
	Xo,
	[0, Xo],
	os,
	-1
], ps = [
	0,
	Xo,
	ya,
	Ea
], ms = [
	0,
	Xo,
	ya
], hs = [
	0,
	Xo,
	ps,
	ms,
	ya
], gs = Va(479097054, ds);
ao[479097054] = [
	0,
	Xo,
	hs,
	fs
], ao[463370452] = fs, ao[464864288] = ps;
var _s = Va(462713202, us);
ao[462713202] = hs, ao[474472470] = ms;
var vs = class extends B {
	constructor(e) {
		super(e);
	}
}, ys = class extends B {
	constructor(e) {
		super(e);
	}
}, bs = class extends B {
	constructor(e) {
		super(e);
	}
}, xs = class extends B {
	constructor(e) {
		super(e);
	}
}, Ss = [
	0,
	Xo,
	ya,
	-1,
	Ea
], Cs = [
	0,
	Xo,
	ya,
	ka
];
xs.prototype.g = Ua([
	0,
	Xo,
	ms,
	[0, Xo],
	Qo,
	rs,
	Ss,
	Cs
]);
var ws = class extends B {
	constructor(e) {
		super(e);
	}
}, Ts = Va(456383383, ws);
ao[456383383] = [
	0,
	Xo,
	Ho
];
var Es = class extends B {
	constructor(e) {
		super(e);
	}
}, Ds = Va(476348187, Es);
ao[476348187] = [
	0,
	Xo,
	Wo
];
var Os = class extends B {
	constructor(e) {
		super(e);
	}
}, ks = class extends B {
	constructor(e) {
		super(e);
	}
}, As = [
	0,
	za,
	-1
], js = Va(458105876, class extends B {
	constructor(e) {
		super(e);
	}
	g() {
		let e;
		var t = this.v;
		let n = 0 | t[I];
		return e = gt(this, n), t = function(e, t, n, r) {
			var i = ks;
			!r && er(e) && (n = 0 | (t = e.v)[I]);
			var a = sr(t, 2);
			if (e = !1, a == null) {
				if (r) return Pn();
				a = [];
			} else if (a.constructor === An) {
				if (!(2 & a.J) || r) return a;
				a = a.V();
			} else Array.isArray(a) ? e = !!(2 & (0 | a[I])) : a = [];
			if (r) {
				if (!a.length) return Pn();
				e || (e = !0, pt(a));
			} else e && (e = !1, mt(a), a = _r(a));
			return !e && 32 & n && dt(a, 32), n = lr(t, n, 2, r = new An(a, i, Cn, void 0)), e || nr(t, n), r;
		}(this, t, n, e), !e && ks && (t.ra = !0), t;
	}
});
ao[458105876] = [
	0,
	As,
	la,
	[
		!0,
		Ca,
		[
			0,
			Na,
			-1,
			ja
		]
	],
	[
		0,
		Da,
		ka,
		za
	]
];
var Ms = class extends B {
	constructor(e) {
		super(e);
	}
}, Ns = Va(458105758, Ms);
ao[458105758] = [
	0,
	Xo,
	Na,
	As
];
var Ps = class extends B {
	constructor(e) {
		super(e);
	}
}, Fs = [
	0,
	Sa,
	-1,
	Aa
], Is = class extends B {
	constructor(e) {
		super(e);
	}
}, Ls = class extends B {
	constructor(e) {
		super(e);
	}
}, Rs = [1, 2];
Ls.prototype.g = Ua([
	0,
	Rs,
	Fa,
	Fs,
	Fa,
	[
		0,
		Pa,
		Fs
	]
]);
var zs = class extends B {
	constructor(e) {
		super(e);
	}
}, Bs = Va(443442058, zs);
ao[443442058] = [
	0,
	Xo,
	Na,
	Ea,
	ya,
	ja,
	-1,
	ka,
	ya
], ao[514774813] = Ss;
var Vs = class extends B {
	constructor(e) {
		super(e);
	}
}, Hs = Va(516587230, Vs);
function Us(e, t) {
	return t = t ? t.clone() : new Vo(), e.displayNamesLocale === void 0 ? e.displayNamesLocale === void 0 && cr(t, 1) : cr(t, 1, bn(e.displayNamesLocale)), e.maxResults === void 0 ? "maxResults" in e && cr(t, 2) : Pr(t, 2, e.maxResults), e.scoreThreshold === void 0 ? "scoreThreshold" in e && cr(t, 3) : z(t, 3, e.scoreThreshold), e.categoryAllowlist === void 0 ? "categoryAllowlist" in e && cr(t, 4) : Ir(t, 4, e.categoryAllowlist), e.categoryDenylist === void 0 ? "categoryDenylist" in e && cr(t, 5) : Ir(t, 5, e.categoryDenylist), t;
}
function Ws(e) {
	let t = Number(e);
	return Number.isSafeInteger(t) ? t : String(e);
}
function Gs(e, t = -1, n = "") {
	return {
		categories: e.map(((e) => ({
			index: Mr(e, 1) ?? 0 ?? -1,
			score: Nr(e, 2) ?? 0,
			categoryName: xn(or(e, 3)) ?? "" ?? "",
			displayName: xn(or(e, 4)) ?? "" ?? ""
		}))),
		headIndex: t,
		headName: n
	};
}
function Ks(e) {
	let t = { classifications: Dr(e, Po, 1).map(((e) => Gs(L(e, bo, 4)?.g() ?? [], Mr(e, 2) ?? 0, xn(or(e, 3)) ?? ""))) };
	return function(e) {
		return e == null ? e : typeof e == "bigint" ? (At(e) ? e = Number(e) : (e = Qt(64, e), e = At(e) ? Number(e) : String(e)), e) : un(e) ? typeof e == "number" ? mn(e) : hn(e) : void 0;
	}(or(e, 2, void 0, void 0, vn)) != null && (t.timestampMs = Ws(or(e, 2, void 0, void 0, vn) ?? ir)), t;
}
function qs(e) {
	var t = dr(e, 3, on, ur()), n = dr(e, 2, dn, ur()), r = dr(e, 1, xn, ur()), i = dr(e, 9, xn, ur());
	let a = {
		categories: [],
		keypoints: []
	};
	for (let e = 0; e < t.length; e++) a.categories.push({
		score: t[e],
		index: n[e] ?? -1,
		categoryName: r[e] ?? "",
		displayName: i[e] ?? ""
	});
	if ((t = L(e, To, 4)?.l()) && (a.boundingBox = {
		originX: Mr(t, 1, ar) ?? 0,
		originY: Mr(t, 2, ar) ?? 0,
		width: Mr(t, 3, ar) ?? 0,
		height: Mr(t, 4, ar) ?? 0,
		angle: 0
	}), L(e, To, 4)?.g().length) for (let t of L(e, To, 4).g()) a.keypoints.push({
		x: or(t, 1, void 0, ar, on) ?? 0,
		y: or(t, 2, void 0, ar, on) ?? 0,
		score: or(t, 4, void 0, ar, on) ?? 0,
		label: xn(or(t, 3, void 0, ar)) ?? ""
	});
	return a;
}
function Js(e) {
	let t = [];
	for (let n of Dr(e, ko, 1)) t.push({
		x: Nr(n, 1) ?? 0,
		y: Nr(n, 2) ?? 0,
		z: Nr(n, 3) ?? 0,
		visibility: Nr(n, 4) ?? 0
	});
	return t;
}
function Ys(e) {
	let t = [];
	for (let n of Dr(e, Do, 1)) t.push({
		x: Nr(n, 1) ?? 0,
		y: Nr(n, 2) ?? 0,
		z: Nr(n, 3) ?? 0,
		visibility: Nr(n, 4) ?? 0
	});
	return t;
}
function Xs(e) {
	return Array.from(e, ((e) => e > 127 ? e - 256 : e));
}
function Zs(e, t) {
	if (e.length !== t.length) throw Error(`Cannot compute cosine similarity between embeddings of different sizes (${e.length} vs. ${t.length}).`);
	let n = 0, r = 0, i = 0;
	for (let a = 0; a < e.length; a++) n += e[a] * t[a], r += e[a] * e[a], i += t[a] * t[a];
	if (r <= 0 || i <= 0) throw Error("Cannot compute cosine similarity on embedding with 0 norm.");
	return n / Math.sqrt(r * i);
}
var Qs;
ao[516587230] = [
	0,
	Xo,
	Ss,
	Cs,
	ya
], ao[518928384] = Cs;
var $s = new Uint8Array([
	0,
	97,
	115,
	109,
	1,
	0,
	0,
	0,
	1,
	5,
	1,
	96,
	0,
	1,
	123,
	3,
	2,
	1,
	0,
	10,
	10,
	1,
	8,
	0,
	65,
	0,
	253,
	15,
	253,
	98,
	11
]);
async function ec(e) {
	if (e) return !0;
	if (Qs === void 0) try {
		await WebAssembly.instantiate($s), Qs = !0;
	} catch {
		Qs = !1;
	}
	return Qs;
}
async function tc(e, t, n) {
	return {
		wasmLoaderPath: `${t}/${e}_${n = `wasm${n ? "_module" : ""}${await ec(n) ? "" : "_nosimd"}_internal`}.js`,
		wasmBinaryPath: `${t}/${e}_${n}.wasm`
	};
}
var nc = class {};
function rc() {
	var e = navigator;
	return typeof OffscreenCanvas < "u" && (!function(e = navigator) {
		return (e = e.userAgent).includes("Safari") && !e.includes("Chrome");
	}(e) || !!((e = e.userAgent.match(/Version\/([\d]+).*Safari/)) && e.length >= 1 && Number(e[1]) >= 17));
}
async function ic(e) {
	if (typeof importScripts != "function") {
		let t = document.createElement("script");
		return t.src = e.toString(), t.crossOrigin = "anonymous", new Promise(((e, n) => {
			t.addEventListener("load", (() => {
				e();
			}), !1), t.addEventListener("error", ((e) => {
				n(e);
			}), !1), document.body.appendChild(t);
		}));
	}
	try {
		importScripts(e.toString());
	} catch (t) {
		if (!(t instanceof TypeError)) throw t;
		{
			let t = self.import;
			t ? await t(e.toString()) : await import(e.toString());
		}
	}
}
function ac(e) {
	return e.videoWidth === void 0 ? e.naturalWidth === void 0 ? e.displayWidth === void 0 ? [e.width, e.height] : [e.displayWidth, e.displayHeight] : [e.naturalWidth, e.naturalHeight] : [e.videoWidth, e.videoHeight];
}
function H(e, t, n) {
	e.m || console.error("No wasm multistream support detected: ensure dependency inclusion of :gl_graph_runner_internal_multi_input target"), n(t = e.i.stringToNewUTF8(t)), e.i._free(t);
}
function oc(e, t, n) {
	if (!e.i.canvas) throw Error("No OpenGL canvas configured.");
	if (n ? e.i._bindTextureToStream(n) : e.i._bindTextureToCanvas(), !(n = e.i.canvas.getContext("webgl2") || e.i.canvas.getContext("webgl"))) throw Error("Failed to obtain WebGL context from the provided canvas. `getContext()` should only be invoked with `webgl` or `webgl2`.");
	e.i.gpuOriginForWebTexturesIsBottomLeft && n.pixelStorei(n.UNPACK_FLIP_Y_WEBGL, !0), n.texImage2D(n.TEXTURE_2D, 0, n.RGBA, n.RGBA, n.UNSIGNED_BYTE, t), e.i.gpuOriginForWebTexturesIsBottomLeft && n.pixelStorei(n.UNPACK_FLIP_Y_WEBGL, !1);
	let [r, i] = ac(t);
	return !e.l || r === e.i.canvas.width && i === e.i.canvas.height || (e.i.canvas.width = r, e.i.canvas.height = i), [r, i];
}
function sc(e, t, n) {
	e.m || console.error("No wasm multistream support detected: ensure dependency inclusion of :gl_graph_runner_internal_multi_input target");
	let r = new Uint32Array(t.length);
	for (let n = 0; n < t.length; n++) r[n] = e.i.stringToNewUTF8(t[n]);
	t = e.i._malloc(4 * r.length), e.i.HEAPU32.set(r, t >> 2), n(t);
	for (let t of r) e.i._free(t);
	e.i._free(t);
}
function cc(e, t, n) {
	e.i.simpleListeners = e.i.simpleListeners || {}, e.i.simpleListeners[t] = n;
}
function lc(e, t, n) {
	let r = [];
	e.i.simpleListeners = e.i.simpleListeners || {}, e.i.simpleListeners[t] = (e, t, i) => {
		t ? (n(r, i), r = []) : r.push(e);
	};
}
nc.forVisionTasks = function(e, t = !1) {
	return tc("vision", e ?? Ya``, t);
}, nc.forTextTasks = function(e, t = !1) {
	return tc("text", e ?? Ya``, t);
}, nc.forGenAiTasks = function(e, t = !1) {
	return tc("genai", e ?? Ya``, t);
}, nc.forAudioTasks = function(e, t = !1) {
	return tc("audio", e ?? Ya``, t);
}, nc.isSimdSupported = function(e = !1) {
	return ec(e);
};
async function uc(e, t, n, r) {
	return e = await (async (e, t, n, r, i) => {
		if (t && await ic(t), !self.ModuleFactory || n && (await ic(n), !self.ModuleFactory)) throw Error("ModuleFactory not set.");
		return self.Module && i && ((t = self.Module).locateFile = i.locateFile, i.mainScriptUrlOrBlob && (t.mainScriptUrlOrBlob = i.mainScriptUrlOrBlob)), i = await self.ModuleFactory(self.Module || i), self.ModuleFactory = self.Module = void 0, new e(i, r);
	})(e, n.wasmLoaderPath, n.assetLoaderPath, t, { locateFile: (e) => e.endsWith(".wasm") ? n.wasmBinaryPath.toString() : n.assetBinaryPath && e.endsWith(".data") ? n.assetBinaryPath.toString() : e }), await e.o(r), e;
}
function dc(e, t) {
	let n = L(e.baseOptions, qo, 1) || new qo();
	typeof t == "string" ? (cr(n, 2, bn(t)), cr(n, 1)) : t instanceof Uint8Array && (cr(n, 1, vt(t, !1)), cr(n, 2)), R(e.baseOptions, 0, 1, n);
}
function fc(e) {
	try {
		let t = e.H.length;
		if (t === 1) throw Error(e.H[0].message);
		if (t > 1) throw Error("Encountered multiple errors: " + e.H.map(((e) => e.message)).join(", "));
	} finally {
		e.H = [];
	}
}
function U(e, t) {
	e.C = Math.max(e.C, t);
}
function pc(e, t) {
	e.B = new co(), Fr(e.B, 2, "PassThroughCalculator"), so(e.B, "free_memory"), V(e.B, "free_memory_unused_out"), mo(t, "free_memory"), po(t, e.B);
}
function mc(e, t) {
	so(e.B, t), V(e.B, t + "_unused_out");
}
function hc(e) {
	e.g.addBoolToStream(!0, "free_memory", e.C);
}
var gc = class {
	constructor(e) {
		this.g = e, this.H = [], this.C = 0, this.g.setAutoRenderToScreen(!1);
	}
	l(e, t = !0) {
		if (t) {
			let t = e.baseOptions || {};
			if (e.baseOptions?.modelAssetBuffer && e.baseOptions?.modelAssetPath) throw Error("Cannot set both baseOptions.modelAssetPath and baseOptions.modelAssetBuffer");
			if (!(L(this.baseOptions, qo, 1)?.g() || L(this.baseOptions, qo, 1)?.l() || e.baseOptions?.modelAssetBuffer || e.baseOptions?.modelAssetPath)) throw Error("Either baseOptions.modelAssetPath or baseOptions.modelAssetBuffer must be set");
			if (function(e, t) {
				let n = L(e.baseOptions, Go, 3);
				if (!n) {
					var r = n = new Go(), i = new to();
					kr(r, 4, Ko, i);
				}
				"delegate" in t && (t.delegate === "GPU" ? (t = n, r = new Za(), kr(t, 2, Ko, r)) : (t = n, r = new to(), kr(t, 4, Ko, r))), R(e.baseOptions, 0, 3, n);
			}(this, t), t.modelAssetPath) return fetch(t.modelAssetPath.toString()).then(((e) => {
				if (e.ok) return e.arrayBuffer();
				throw Error(`Failed to fetch model: ${t.modelAssetPath} (${e.status})`);
			})).then(((e) => {
				try {
					this.g.i.FS_unlink("/model.dat");
				} catch {}
				this.g.i.FS_createDataFile("/", "model.dat", new Uint8Array(e), !0, !1, !1), dc(this, "/model.dat"), this.m(), this.L();
			}));
			if (t.modelAssetBuffer instanceof Uint8Array) dc(this, t.modelAssetBuffer);
			else if (t.modelAssetBuffer) return async function(e) {
				let t = [];
				for (var n = 0;;) {
					let { done: r, value: i } = await e.read();
					if (r) break;
					t.push(i), n += i.length;
				}
				if (t.length === 0) return /* @__PURE__ */ new Uint8Array();
				if (t.length === 1) return t[0];
				e = new Uint8Array(n), n = 0;
				for (let r of t) e.set(r, n), n += r.length;
				return e;
			}(t.modelAssetBuffer).then(((e) => {
				dc(this, e), this.m(), this.L();
			}));
		}
		return this.m(), this.L(), Promise.resolve();
	}
	L() {}
	ca() {
		let e;
		if (this.g.ca(((t) => {
			e = vo(t);
		})), !e) throw Error("Failed to retrieve CalculatorGraphConfig");
		return e;
	}
	setGraph(e, t) {
		this.g.attachErrorListener(((e, t) => {
			this.H.push(Error(t));
		})), this.g.Ja(), this.g.setGraph(e, t), this.B = void 0, fc(this);
	}
	finishProcessing() {
		this.g.finishProcessing(), fc(this);
	}
	close() {
		this.B = void 0, this.g.closeGraph();
	}
};
function _c(e, t) {
	if (!e) throw Error(`Unable to obtain required WebGL resource: ${t}`);
	return e;
}
gc.prototype.close = gc.prototype.close;
var vc = class {
	constructor(e, t, n, r) {
		this.g = e, this.h = t, this.m = n, this.l = r;
	}
	bind() {
		this.g.bindVertexArray(this.h);
	}
	close() {
		this.g.deleteVertexArray(this.h), this.g.deleteBuffer(this.m), this.g.deleteBuffer(this.l);
	}
};
function yc(e, t, n) {
	let r = e.g;
	if (n = _c(r.createShader(n), "Failed to create WebGL shader"), r.shaderSource(n, t), r.compileShader(n), !r.getShaderParameter(n, r.COMPILE_STATUS)) throw Error(`Could not compile WebGL shader: ${r.getShaderInfoLog(n)}`);
	return r.attachShader(e.h, n), n;
}
function bc(e, t) {
	let n = e.g, r = _c(n.createVertexArray(), "Failed to create vertex array");
	n.bindVertexArray(r);
	let i = _c(n.createBuffer(), "Failed to create buffer");
	n.bindBuffer(n.ARRAY_BUFFER, i), n.enableVertexAttribArray(e.O), n.vertexAttribPointer(e.O, 2, n.FLOAT, !1, 0, 0), n.bufferData(n.ARRAY_BUFFER, new Float32Array([
		-1,
		-1,
		-1,
		1,
		1,
		1,
		1,
		-1
	]), n.STATIC_DRAW);
	let a = _c(n.createBuffer(), "Failed to create buffer");
	return n.bindBuffer(n.ARRAY_BUFFER, a), n.enableVertexAttribArray(e.L), n.vertexAttribPointer(e.L, 2, n.FLOAT, !1, 0, 0), n.bufferData(n.ARRAY_BUFFER, new Float32Array(t ? [
		0,
		1,
		0,
		0,
		1,
		0,
		1,
		1
	] : [
		0,
		0,
		0,
		1,
		1,
		1,
		1,
		0
	]), n.STATIC_DRAW), n.bindBuffer(n.ARRAY_BUFFER, null), n.bindVertexArray(null), new vc(n, r, i, a);
}
function xc(e, t) {
	if (e.g) {
		if (t !== e.g) throw Error("Cannot change GL context once initialized");
	} else e.g = t;
}
function Sc(e, t, n, r) {
	return xc(e, t), e.h || (e.m(), e.D()), n ? (e.u ||= bc(e, !0), n = e.u) : (e.A ||= bc(e, !1), n = e.A), t.useProgram(e.h), n.bind(), e.l(), e = r(), n.g.bindVertexArray(null), e;
}
function Cc(e, t, n) {
	return xc(e, t), e = _c(t.createTexture(), "Failed to create texture"), t.bindTexture(t.TEXTURE_2D, e), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_S, t.CLAMP_TO_EDGE), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_T, t.CLAMP_TO_EDGE), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_MIN_FILTER, n ?? t.LINEAR), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_MAG_FILTER, n ?? t.LINEAR), t.bindTexture(t.TEXTURE_2D, null), e;
}
function wc(e, t, n) {
	xc(e, t), e.B ||= _c(t.createFramebuffer(), "Failed to create framebuffe."), t.bindFramebuffer(t.FRAMEBUFFER, e.B), t.framebufferTexture2D(t.FRAMEBUFFER, t.COLOR_ATTACHMENT0, t.TEXTURE_2D, n, 0);
}
function Tc(e) {
	e.g?.bindFramebuffer(e.g.FRAMEBUFFER, null);
}
var Ec = class {
	H() {
		return "\n  precision mediump float;\n  varying vec2 vTex;\n  uniform sampler2D inputTexture;\n  void main() {\n    gl_FragColor = texture2D(inputTexture, vTex);\n  }\n ";
	}
	m() {
		let e = this.g;
		if (this.h = _c(e.createProgram(), "Failed to create WebGL program"), this.X = yc(this, "\n  attribute vec2 aVertex;\n  attribute vec2 aTex;\n  varying vec2 vTex;\n  void main(void) {\n    gl_Position = vec4(aVertex, 0.0, 1.0);\n    vTex = aTex;\n  }", e.VERTEX_SHADER), this.W = yc(this, this.H(), e.FRAGMENT_SHADER), e.linkProgram(this.h), !e.getProgramParameter(this.h, e.LINK_STATUS)) throw Error(`Error during program linking: ${e.getProgramInfoLog(this.h)}`);
		this.O = e.getAttribLocation(this.h, "aVertex"), this.L = e.getAttribLocation(this.h, "aTex");
	}
	D() {}
	l() {}
	close() {
		if (this.h) {
			let e = this.g;
			e.deleteProgram(this.h), e.deleteShader(this.X), e.deleteShader(this.W);
		}
		this.B && this.g.deleteFramebuffer(this.B), this.A && this.A.close(), this.u && this.u.close();
	}
}, Dc = class extends Ec {
	H() {
		return "\n  precision mediump float;\n  uniform sampler2D backgroundTexture;\n  uniform sampler2D maskTexture;\n  uniform sampler2D colorMappingTexture;\n  varying vec2 vTex;\n  void main() {\n    vec4 backgroundColor = texture2D(backgroundTexture, vTex);\n    float category = texture2D(maskTexture, vTex).r;\n    vec4 categoryColor = texture2D(colorMappingTexture, vec2(category, 0.0));\n    gl_FragColor = mix(backgroundColor, categoryColor, categoryColor.a);\n  }\n ";
	}
	D() {
		let e = this.g;
		e.activeTexture(e.TEXTURE1), this.C = Cc(this, e, e.LINEAR), e.activeTexture(e.TEXTURE2), this.j = Cc(this, e, e.NEAREST);
	}
	m() {
		super.m();
		let e = this.g;
		this.P = _c(e.getUniformLocation(this.h, "backgroundTexture"), "Uniform location"), this.U = _c(e.getUniformLocation(this.h, "colorMappingTexture"), "Uniform location"), this.M = _c(e.getUniformLocation(this.h, "maskTexture"), "Uniform location");
	}
	l() {
		super.l();
		let e = this.g;
		e.uniform1i(this.M, 0), e.uniform1i(this.P, 1), e.uniform1i(this.U, 2);
	}
	close() {
		this.C && this.g.deleteTexture(this.C), this.j && this.g.deleteTexture(this.j), super.close();
	}
}, Oc = class extends Ec {
	H() {
		return "\n  precision mediump float;\n  uniform sampler2D maskTexture;\n  uniform sampler2D defaultTexture;\n  uniform sampler2D overlayTexture;\n  varying vec2 vTex;\n  void main() {\n    float confidence = texture2D(maskTexture, vTex).r;\n    vec4 defaultColor = texture2D(defaultTexture, vTex);\n    vec4 overlayColor = texture2D(overlayTexture, vTex);\n    // Apply the alpha from the overlay and merge in the default color\n    overlayColor = mix(defaultColor, overlayColor, overlayColor.a);\n    gl_FragColor = mix(defaultColor, overlayColor, confidence);\n  }\n ";
	}
	D() {
		let e = this.g;
		e.activeTexture(e.TEXTURE1), this.j = Cc(this, e), e.activeTexture(e.TEXTURE2), this.C = Cc(this, e);
	}
	m() {
		super.m();
		let e = this.g;
		this.M = _c(e.getUniformLocation(this.h, "defaultTexture"), "Uniform location"), this.P = _c(e.getUniformLocation(this.h, "overlayTexture"), "Uniform location"), this.I = _c(e.getUniformLocation(this.h, "maskTexture"), "Uniform location");
	}
	l() {
		super.l();
		let e = this.g;
		e.uniform1i(this.I, 0), e.uniform1i(this.M, 1), e.uniform1i(this.P, 2);
	}
	close() {
		this.j && this.g.deleteTexture(this.j), this.C && this.g.deleteTexture(this.C), super.close();
	}
};
function kc(e, t) {
	switch (t) {
		case 0: return e.g.find(((e) => e instanceof Uint8Array));
		case 1: return e.g.find(((e) => e instanceof Float32Array));
		case 2: return e.g.find(((e) => typeof WebGLTexture < "u" && e instanceof WebGLTexture));
		default: throw Error(`Type is not supported: ${t}`);
	}
}
function Ac(e) {
	var t = kc(e, 1);
	if (!t) {
		if (t = kc(e, 0)) t = new Float32Array(t).map(((e) => e / 255));
		else {
			t = new Float32Array(e.width * e.height);
			let r = Mc(e);
			var n = Pc(e);
			if (wc(n, r, jc(e)), "iPad Simulator;iPhone Simulator;iPod Simulator;iPad;iPhone;iPod".split(";").includes(navigator.platform) || navigator.userAgent.includes("Mac") && "document" in self && "ontouchend" in self.document) {
				n = new Float32Array(e.width * e.height * 4), r.readPixels(0, 0, e.width, e.height, r.RGBA, r.FLOAT, n);
				for (let e = 0, r = 0; e < t.length; ++e, r += 4) t[e] = n[r];
			} else r.readPixels(0, 0, e.width, e.height, r.RED, r.FLOAT, t);
		}
		e.g.push(t);
	}
	return t;
}
function jc(e) {
	let t = kc(e, 2);
	if (!t) {
		let n = Mc(e);
		t = Fc(e);
		let r = Ac(e), i = Nc(e);
		n.texImage2D(n.TEXTURE_2D, 0, i, e.width, e.height, 0, n.RED, n.FLOAT, r), Ic(e);
	}
	return t;
}
function Mc(e) {
	if (!e.canvas) throw Error("Conversion to different image formats require that a canvas is passed when initializing the image.");
	return e.h ||= _c(e.canvas.getContext("webgl2"), "You cannot use a canvas that is already bound to a different type of rendering context."), e.h;
}
function Nc(e) {
	if (e = Mc(e), !Lc) if (e.getExtension("EXT_color_buffer_float") && e.getExtension("OES_texture_float_linear") && e.getExtension("EXT_float_blend")) Lc = e.R32F;
	else {
		if (!e.getExtension("EXT_color_buffer_half_float")) throw Error("GPU does not fully support 4-channel float32 or float16 formats");
		Lc = e.R16F;
	}
	return Lc;
}
function Pc(e) {
	return e.l ||= new Ec(), e.l;
}
function Fc(e) {
	let t = Mc(e);
	t.viewport(0, 0, e.width, e.height), t.activeTexture(t.TEXTURE0);
	let n = kc(e, 2);
	return n || (n = Cc(Pc(e), t, e.m ? t.LINEAR : t.NEAREST), e.g.push(n), e.j = !0), t.bindTexture(t.TEXTURE_2D, n), n;
}
function Ic(e) {
	e.h.bindTexture(e.h.TEXTURE_2D, null);
}
var Lc, Rc = class {
	constructor(e, t, n, r, i, a, o) {
		this.g = e, this.m = t, this.j = n, this.canvas = r, this.l = i, this.width = a, this.height = o, this.j && --zc === 0 && console.error("You seem to be creating MPMask instances without invoking .close(). This leaks resources.");
	}
	Fa() {
		return !!kc(this, 0);
	}
	ka() {
		return !!kc(this, 1);
	}
	R() {
		return !!kc(this, 2);
	}
	ja() {
		return (t = kc(e = this, 0)) || (t = Ac(e), t = new Uint8Array(t.map(((e) => Math.round(255 * e)))), e.g.push(t)), t;
		var e, t;
	}
	ia() {
		return Ac(this);
	}
	N() {
		return jc(this);
	}
	clone() {
		let e = [];
		for (let t of this.g) {
			let n;
			if (t instanceof Uint8Array) n = new Uint8Array(t);
			else if (t instanceof Float32Array) n = new Float32Array(t);
			else {
				if (!(t instanceof WebGLTexture)) throw Error(`Type is not supported: ${t}`);
				{
					let e = Mc(this), t = Pc(this);
					e.activeTexture(e.TEXTURE1), n = Cc(t, e, this.m ? e.LINEAR : e.NEAREST), e.bindTexture(e.TEXTURE_2D, n);
					let r = Nc(this);
					e.texImage2D(e.TEXTURE_2D, 0, r, this.width, this.height, 0, e.RED, e.FLOAT, null), e.bindTexture(e.TEXTURE_2D, null), wc(t, e, n), Sc(t, e, !1, (() => {
						Fc(this), e.clearColor(0, 0, 0, 0), e.clear(e.COLOR_BUFFER_BIT), e.drawArrays(e.TRIANGLE_FAN, 0, 4), Ic(this);
					})), Tc(t), Ic(this);
				}
			}
			e.push(n);
		}
		return new Rc(e, this.m, this.R(), this.canvas, this.l, this.width, this.height);
	}
	close() {
		this.j && Mc(this).deleteTexture(kc(this, 2)), zc = -1;
	}
};
Rc.prototype.close = Rc.prototype.close, Rc.prototype.clone = Rc.prototype.clone, Rc.prototype.getAsWebGLTexture = Rc.prototype.N, Rc.prototype.getAsFloat32Array = Rc.prototype.ia, Rc.prototype.getAsUint8Array = Rc.prototype.ja, Rc.prototype.hasWebGLTexture = Rc.prototype.R, Rc.prototype.hasFloat32Array = Rc.prototype.ka, Rc.prototype.hasUint8Array = Rc.prototype.Fa;
var zc = 250, Bc = {
	color: "white",
	lineWidth: 4,
	radius: 6
};
function Vc(e) {
	return {
		...Bc,
		fillColor: (e ||= {}).color,
		...e
	};
}
function Hc(e, t) {
	return e instanceof Function ? e(t) : e;
}
function Uc(e, t, n) {
	return Math.max(Math.min(t, n), Math.min(Math.max(t, n), e));
}
function Wc(e) {
	if (!e.l) throw Error("CPU rendering requested but CanvasRenderingContext2D not provided.");
	return e.l;
}
function Gc(e) {
	if (!e.j) throw Error("GPU rendering requested but WebGL2RenderingContext not provided.");
	return e.j;
}
function Kc(e, t, n) {
	if (t.R()) n(t.N());
	else {
		let r = t.ka() ? t.ia() : t.ja();
		e.m = e.m ?? new Ec();
		let i = Gc(e);
		n((e = new Rc([r], t.m, !1, i.canvas, e.m, t.width, t.height)).N()), e.close();
	}
}
function qc(e, t, n, r) {
	let i = function(e) {
		return e.g ||= new Dc(), e.g;
	}(e), a = Gc(e), o = Array.isArray(n) ? new ImageData(new Uint8ClampedArray(n), 1, 1) : n;
	Sc(i, a, !0, (() => {
		(function(e, t, n, r) {
			let i = e.g;
			if (i.activeTexture(i.TEXTURE0), i.bindTexture(i.TEXTURE_2D, t), i.activeTexture(i.TEXTURE1), i.bindTexture(i.TEXTURE_2D, e.C), i.texImage2D(i.TEXTURE_2D, 0, i.RGBA, i.RGBA, i.UNSIGNED_BYTE, n), e.I && function(e, t) {
				if (e !== t) return !1;
				e = e.entries(), t = t.entries();
				for (let [n, r] of e) {
					e = n;
					let i = r, a = t.next();
					if (a.done) return !1;
					let [o, s] = a.value;
					if (e !== o || i[0] !== s[0] || i[1] !== s[1] || i[2] !== s[2] || i[3] !== s[3]) return !1;
				}
				return !!t.next().done;
			}(e.I, r)) i.activeTexture(i.TEXTURE2), i.bindTexture(i.TEXTURE_2D, e.j);
			else {
				e.I = r;
				let t = Array(1024).fill(0);
				r.forEach(((e, n) => {
					if (e.length !== 4) throw Error(`Color at index ${n} is not a four-channel value.`);
					t[4 * n] = e[0], t[4 * n + 1] = e[1], t[4 * n + 2] = e[2], t[4 * n + 3] = e[3];
				})), i.activeTexture(i.TEXTURE2), i.bindTexture(i.TEXTURE_2D, e.j), i.texImage2D(i.TEXTURE_2D, 0, i.RGBA, 256, 1, 0, i.RGBA, i.UNSIGNED_BYTE, new Uint8Array(t));
			}
		})(i, t, o, r), a.clearColor(0, 0, 0, 0), a.clear(a.COLOR_BUFFER_BIT), a.drawArrays(a.TRIANGLE_FAN, 0, 4);
		let e = i.g;
		e.activeTexture(e.TEXTURE0), e.bindTexture(e.TEXTURE_2D, null), e.activeTexture(e.TEXTURE1), e.bindTexture(e.TEXTURE_2D, null), e.activeTexture(e.TEXTURE2), e.bindTexture(e.TEXTURE_2D, null);
	}));
}
function Jc(e, t, n, r) {
	let i = Gc(e), a = function(e) {
		return e.h ||= new Oc(), e.h;
	}(e), o = Array.isArray(n) ? new ImageData(new Uint8ClampedArray(n), 1, 1) : n, s = Array.isArray(r) ? new ImageData(new Uint8ClampedArray(r), 1, 1) : r;
	Sc(a, i, !0, (() => {
		var e = a.g;
		e.activeTexture(e.TEXTURE0), e.bindTexture(e.TEXTURE_2D, t), e.activeTexture(e.TEXTURE1), e.bindTexture(e.TEXTURE_2D, a.j), e.texImage2D(e.TEXTURE_2D, 0, e.RGBA, e.RGBA, e.UNSIGNED_BYTE, o), e.activeTexture(e.TEXTURE2), e.bindTexture(e.TEXTURE_2D, a.C), e.texImage2D(e.TEXTURE_2D, 0, e.RGBA, e.RGBA, e.UNSIGNED_BYTE, s), i.clearColor(0, 0, 0, 0), i.clear(i.COLOR_BUFFER_BIT), i.drawArrays(i.TRIANGLE_FAN, 0, 4), i.bindTexture(i.TEXTURE_2D, null), (e = a.g).activeTexture(e.TEXTURE0), e.bindTexture(e.TEXTURE_2D, null), e.activeTexture(e.TEXTURE1), e.bindTexture(e.TEXTURE_2D, null), e.activeTexture(e.TEXTURE2), e.bindTexture(e.TEXTURE_2D, null);
	}));
}
var Yc = class {
	constructor(e, t) {
		typeof CanvasRenderingContext2D < "u" && e instanceof CanvasRenderingContext2D || e instanceof OffscreenCanvasRenderingContext2D ? (this.l = e, this.j = t) : this.j = e;
	}
	ya(e, t) {
		if (e) {
			var n = Wc(this);
			t = Vc(t), n.save();
			var r = n.canvas, i = 0;
			for (let a of e) n.fillStyle = Hc(t.fillColor, {
				index: i,
				from: a
			}), n.strokeStyle = Hc(t.color, {
				index: i,
				from: a
			}), n.lineWidth = Hc(t.lineWidth, {
				index: i,
				from: a
			}), (e = new Path2D()).arc(a.x * r.width, a.y * r.height, Hc(t.radius, {
				index: i,
				from: a
			}), 0, 2 * Math.PI), n.fill(e), n.stroke(e), ++i;
			n.restore();
		}
	}
	xa(e, t, n) {
		if (e && t) {
			var r = Wc(this);
			n = Vc(n), r.save();
			var i = r.canvas, a = 0;
			for (let o of t) {
				r.beginPath(), t = e[o.start];
				let s = e[o.end];
				t && s && (r.strokeStyle = Hc(n.color, {
					index: a,
					from: t,
					to: s
				}), r.lineWidth = Hc(n.lineWidth, {
					index: a,
					from: t,
					to: s
				}), r.moveTo(t.x * i.width, t.y * i.height), r.lineTo(s.x * i.width, s.y * i.height)), ++a, r.stroke();
			}
			r.restore();
		}
	}
	ua(e, t) {
		let n = Wc(this);
		t = Vc(t), n.save(), n.beginPath(), n.lineWidth = Hc(t.lineWidth, {}), n.strokeStyle = Hc(t.color, {}), n.fillStyle = Hc(t.fillColor, {}), n.moveTo(e.originX, e.originY), n.lineTo(e.originX + e.width, e.originY), n.lineTo(e.originX + e.width, e.originY + e.height), n.lineTo(e.originX, e.originY + e.height), n.lineTo(e.originX, e.originY), n.stroke(), n.fill(), n.restore();
	}
	va(e, t, n = [
		0,
		0,
		0,
		255
	]) {
		this.l ? function(e, t, n, r) {
			let i = Gc(e);
			Kc(e, t, ((t) => {
				qc(e, t, n, r), (t = Wc(e)).drawImage(i.canvas, 0, 0, t.canvas.width, t.canvas.height);
			}));
		}(this, e, n, t) : qc(this, e.N(), n, t);
	}
	wa(e, t, n) {
		this.l ? function(e, t, n, r) {
			let i = Gc(e);
			Kc(e, t, ((t) => {
				Jc(e, t, n, r), (t = Wc(e)).drawImage(i.canvas, 0, 0, t.canvas.width, t.canvas.height);
			}));
		}(this, e, t, n) : Jc(this, e.N(), t, n);
	}
	close() {
		this.g?.close(), this.g = void 0, this.h?.close(), this.h = void 0, this.m?.close(), this.m = void 0;
	}
};
function Xc(e, t) {
	switch (t) {
		case 0: return e.g.find(((e) => e instanceof ImageData));
		case 1: return e.g.find(((e) => typeof ImageBitmap < "u" && e instanceof ImageBitmap));
		case 2: return e.g.find(((e) => typeof WebGLTexture < "u" && e instanceof WebGLTexture));
		default: throw Error(`Type is not supported: ${t}`);
	}
}
function Zc(e) {
	var t = Xc(e, 0);
	if (!t) {
		t = $c(e);
		let n = el(e), r = new Uint8Array(e.width * e.height * 4);
		wc(n, t, Qc(e)), t.readPixels(0, 0, e.width, e.height, t.RGBA, t.UNSIGNED_BYTE, r), Tc(n), t = new ImageData(new Uint8ClampedArray(r.buffer), e.width, e.height), e.g.push(t);
	}
	return t;
}
function Qc(e) {
	let t = Xc(e, 2);
	if (!t) {
		let n = $c(e);
		t = tl(e);
		let r = Xc(e, 1) || Zc(e);
		n.texImage2D(n.TEXTURE_2D, 0, n.RGBA, n.RGBA, n.UNSIGNED_BYTE, r), nl(e);
	}
	return t;
}
function $c(e) {
	if (!e.canvas) throw Error("Conversion to different image formats require that a canvas is passed when initializing the image.");
	return e.h ||= _c(e.canvas.getContext("webgl2"), "You cannot use a canvas that is already bound to a different type of rendering context."), e.h;
}
function el(e) {
	return e.l ||= new Ec(), e.l;
}
function tl(e) {
	let t = $c(e);
	t.viewport(0, 0, e.width, e.height), t.activeTexture(t.TEXTURE0);
	let n = Xc(e, 2);
	return n || (n = Cc(el(e), t), e.g.push(n), e.m = !0), t.bindTexture(t.TEXTURE_2D, n), n;
}
function nl(e) {
	e.h.bindTexture(e.h.TEXTURE_2D, null);
}
function rl(e) {
	let t = $c(e);
	return Sc(el(e), t, !0, (() => function(e, t) {
		let n = e.canvas;
		if (n.width === e.width && n.height === e.height) return t();
		let r = n.width, i = n.height;
		return n.width = e.width, n.height = e.height, e = t(), n.width = r, n.height = i, e;
	}(e, (() => {
		if (t.bindFramebuffer(t.FRAMEBUFFER, null), t.clearColor(0, 0, 0, 0), t.clear(t.COLOR_BUFFER_BIT), t.drawArrays(t.TRIANGLE_FAN, 0, 4), !(e.canvas instanceof OffscreenCanvas)) throw Error("Conversion to ImageBitmap requires that the MediaPipe Tasks is initialized with an OffscreenCanvas");
		return e.canvas.transferToImageBitmap();
	}))));
}
Yc.prototype.close = Yc.prototype.close, Yc.prototype.drawConfidenceMask = Yc.prototype.wa, Yc.prototype.drawCategoryMask = Yc.prototype.va, Yc.prototype.drawBoundingBox = Yc.prototype.ua, Yc.prototype.drawConnectors = Yc.prototype.xa, Yc.prototype.drawLandmarks = Yc.prototype.ya, Yc.lerp = function(e, t, n, r, i) {
	return Uc(r * (1 - (e - t) / (n - t)) + i * (1 - (n - e) / (n - t)), r, i);
}, Yc.clamp = Uc;
var il = class {
	constructor(e, t, n, r, i, a, o) {
		this.g = e, this.j = t, this.m = n, this.canvas = r, this.l = i, this.width = a, this.height = o, (this.j || this.m) && --al === 0 && console.error("You seem to be creating MPImage instances without invoking .close(). This leaks resources.");
	}
	Ea() {
		return !!Xc(this, 0);
	}
	la() {
		return !!Xc(this, 1);
	}
	R() {
		return !!Xc(this, 2);
	}
	Ca() {
		return Zc(this);
	}
	Ba() {
		var e = Xc(this, 1);
		return e || (Qc(this), tl(this), e = rl(this), nl(this), this.g.push(e), this.j = !0), e;
	}
	N() {
		return Qc(this);
	}
	clone() {
		let e = [];
		for (let t of this.g) {
			let n;
			if (t instanceof ImageData) n = new ImageData(t.data, this.width, this.height);
			else if (t instanceof WebGLTexture) {
				let e = $c(this), t = el(this);
				e.activeTexture(e.TEXTURE1), n = Cc(t, e), e.bindTexture(e.TEXTURE_2D, n), e.texImage2D(e.TEXTURE_2D, 0, e.RGBA, this.width, this.height, 0, e.RGBA, e.UNSIGNED_BYTE, null), e.bindTexture(e.TEXTURE_2D, null), wc(t, e, n), Sc(t, e, !1, (() => {
					tl(this), e.clearColor(0, 0, 0, 0), e.clear(e.COLOR_BUFFER_BIT), e.drawArrays(e.TRIANGLE_FAN, 0, 4), nl(this);
				})), Tc(t), nl(this);
			} else {
				if (!(t instanceof ImageBitmap)) throw Error(`Type is not supported: ${t}`);
				Qc(this), tl(this), n = rl(this), nl(this);
			}
			e.push(n);
		}
		return new il(e, this.la(), this.R(), this.canvas, this.l, this.width, this.height);
	}
	close() {
		this.j && Xc(this, 1).close(), this.m && $c(this).deleteTexture(Xc(this, 2)), al = -1;
	}
};
il.prototype.close = il.prototype.close, il.prototype.clone = il.prototype.clone, il.prototype.getAsWebGLTexture = il.prototype.N, il.prototype.getAsImageBitmap = il.prototype.Ba, il.prototype.getAsImageData = il.prototype.Ca, il.prototype.hasWebGLTexture = il.prototype.R, il.prototype.hasImageBitmap = il.prototype.la, il.prototype.hasImageData = il.prototype.Ea;
var al = 250;
function ol(...e) {
	return e.map((([e, t]) => ({
		start: e,
		end: t
	})));
}
var sl = function(e) {
	return class extends e {
		Ja() {
			this.i._registerModelResourcesGraphService();
		}
	};
}((cl = class {
	constructor(e, t) {
		this.l = !0, this.i = e, this.g = null, this.h = 0, this.m = typeof this.i._addIntToInputStream == "function", t === void 0 ? rc() ? this.i.canvas = new OffscreenCanvas(1, 1) : (console.warn("OffscreenCanvas not supported and GraphRunner constructor glCanvas parameter is undefined. Creating backup canvas."), this.i.canvas = document.createElement("canvas")) : this.i.canvas = t;
	}
	async initializeGraph(e) {
		let t = await (await fetch(e)).arrayBuffer();
		e = !(e.endsWith(".pbtxt") || e.endsWith(".textproto")), this.setGraph(new Uint8Array(t), e);
	}
	setGraphFromString(e) {
		this.setGraph(new TextEncoder().encode(e), !1);
	}
	setGraph(e, t) {
		let n = e.length, r = this.i._malloc(n);
		this.i.HEAPU8.set(e, r), t ? this.i._changeBinaryGraph(n, r) : this.i._changeTextGraph(n, r), this.i._free(r);
	}
	configureAudio(e, t, n, r, i) {
		this.i._configureAudio || console.warn("Attempting to use configureAudio without support for input audio. Is build dep \":gl_graph_runner_audio\" missing?"), H(this, r || "input_audio", ((r) => {
			H(this, i ||= "audio_header", ((i) => {
				this.i._configureAudio(r, i, e, t ?? 0, n);
			}));
		}));
	}
	setAutoResizeCanvas(e) {
		this.l = e;
	}
	setAutoRenderToScreen(e) {
		this.i._setAutoRenderToScreen(e);
	}
	setGpuBufferVerticalFlip(e) {
		this.i.gpuOriginForWebTexturesIsBottomLeft = e;
	}
	ca(e) {
		cc(this, "__graph_config__", ((t) => {
			e(t);
		})), H(this, "__graph_config__", ((e) => {
			this.i._getGraphConfig(e, void 0);
		})), delete this.i.simpleListeners.__graph_config__;
	}
	attachErrorListener(e) {
		this.i.errorListener = e;
	}
	attachEmptyPacketListener(e, t) {
		this.i.emptyPacketListeners = this.i.emptyPacketListeners || {}, this.i.emptyPacketListeners[e] = t;
	}
	addAudioToStream(e, t, n) {
		this.addAudioToStreamWithShape(e, 0, 0, t, n);
	}
	addAudioToStreamWithShape(e, t, n, r, i) {
		let a = 4 * e.length;
		this.h !== a && (this.g && this.i._free(this.g), this.g = this.i._malloc(a), this.h = a), this.i.HEAPF32.set(e, this.g / 4), H(this, r, ((e) => {
			this.i._addAudioToInputStream(this.g, t, n, e, i);
		}));
	}
	addGpuBufferToStream(e, t, n) {
		H(this, t, ((t) => {
			let [r, i] = oc(this, e, t);
			this.i._addBoundTextureToStream(t, r, i, n);
		}));
	}
	addBoolToStream(e, t, n) {
		H(this, t, ((t) => {
			this.i._addBoolToInputStream(e, t, n);
		}));
	}
	addDoubleToStream(e, t, n) {
		H(this, t, ((t) => {
			this.i._addDoubleToInputStream(e, t, n);
		}));
	}
	addFloatToStream(e, t, n) {
		H(this, t, ((t) => {
			this.i._addFloatToInputStream(e, t, n);
		}));
	}
	addIntToStream(e, t, n) {
		H(this, t, ((t) => {
			this.i._addIntToInputStream(e, t, n);
		}));
	}
	addUintToStream(e, t, n) {
		H(this, t, ((t) => {
			this.i._addUintToInputStream(e, t, n);
		}));
	}
	addStringToStream(e, t, n) {
		H(this, t, ((t) => {
			H(this, e, ((e) => {
				this.i._addStringToInputStream(e, t, n);
			}));
		}));
	}
	addStringRecordToStream(e, t, n) {
		H(this, t, ((t) => {
			sc(this, Object.keys(e), ((r) => {
				sc(this, Object.values(e), ((i) => {
					this.i._addFlatHashMapToInputStream(r, i, Object.keys(e).length, t, n);
				}));
			}));
		}));
	}
	addProtoToStream(e, t, n, r) {
		H(this, n, ((n) => {
			H(this, t, ((t) => {
				let i = this.i._malloc(e.length);
				this.i.HEAPU8.set(e, i), this.i._addProtoToInputStream(i, e.length, t, n, r), this.i._free(i);
			}));
		}));
	}
	addEmptyPacketToStream(e, t) {
		H(this, e, ((e) => {
			this.i._addEmptyPacketToInputStream(e, t);
		}));
	}
	addBoolVectorToStream(e, t, n) {
		H(this, t, ((t) => {
			let r = this.i._allocateBoolVector(e.length);
			if (!r) throw Error("Unable to allocate new bool vector on heap.");
			for (let t of e) this.i._addBoolVectorEntry(r, t);
			this.i._addBoolVectorToInputStream(r, t, n);
		}));
	}
	addDoubleVectorToStream(e, t, n) {
		H(this, t, ((t) => {
			let r = this.i._allocateDoubleVector(e.length);
			if (!r) throw Error("Unable to allocate new double vector on heap.");
			for (let t of e) this.i._addDoubleVectorEntry(r, t);
			this.i._addDoubleVectorToInputStream(r, t, n);
		}));
	}
	addFloatVectorToStream(e, t, n) {
		H(this, t, ((t) => {
			let r = this.i._allocateFloatVector(e.length);
			if (!r) throw Error("Unable to allocate new float vector on heap.");
			for (let t of e) this.i._addFloatVectorEntry(r, t);
			this.i._addFloatVectorToInputStream(r, t, n);
		}));
	}
	addIntVectorToStream(e, t, n) {
		H(this, t, ((t) => {
			let r = this.i._allocateIntVector(e.length);
			if (!r) throw Error("Unable to allocate new int vector on heap.");
			for (let t of e) this.i._addIntVectorEntry(r, t);
			this.i._addIntVectorToInputStream(r, t, n);
		}));
	}
	addUintVectorToStream(e, t, n) {
		H(this, t, ((t) => {
			let r = this.i._allocateUintVector(e.length);
			if (!r) throw Error("Unable to allocate new unsigned int vector on heap.");
			for (let t of e) this.i._addUintVectorEntry(r, t);
			this.i._addUintVectorToInputStream(r, t, n);
		}));
	}
	addStringVectorToStream(e, t, n) {
		H(this, t, ((t) => {
			let r = this.i._allocateStringVector(e.length);
			if (!r) throw Error("Unable to allocate new string vector on heap.");
			for (let t of e) H(this, t, ((e) => {
				this.i._addStringVectorEntry(r, e);
			}));
			this.i._addStringVectorToInputStream(r, t, n);
		}));
	}
	addBoolToInputSidePacket(e, t) {
		H(this, t, ((t) => {
			this.i._addBoolToInputSidePacket(e, t);
		}));
	}
	addDoubleToInputSidePacket(e, t) {
		H(this, t, ((t) => {
			this.i._addDoubleToInputSidePacket(e, t);
		}));
	}
	addFloatToInputSidePacket(e, t) {
		H(this, t, ((t) => {
			this.i._addFloatToInputSidePacket(e, t);
		}));
	}
	addIntToInputSidePacket(e, t) {
		H(this, t, ((t) => {
			this.i._addIntToInputSidePacket(e, t);
		}));
	}
	addUintToInputSidePacket(e, t) {
		H(this, t, ((t) => {
			this.i._addUintToInputSidePacket(e, t);
		}));
	}
	addStringToInputSidePacket(e, t) {
		H(this, t, ((t) => {
			H(this, e, ((e) => {
				this.i._addStringToInputSidePacket(e, t);
			}));
		}));
	}
	addProtoToInputSidePacket(e, t, n) {
		H(this, n, ((n) => {
			H(this, t, ((t) => {
				let r = this.i._malloc(e.length);
				this.i.HEAPU8.set(e, r), this.i._addProtoToInputSidePacket(r, e.length, t, n), this.i._free(r);
			}));
		}));
	}
	addBoolVectorToInputSidePacket(e, t) {
		H(this, t, ((t) => {
			let n = this.i._allocateBoolVector(e.length);
			if (!n) throw Error("Unable to allocate new bool vector on heap.");
			for (let t of e) this.i._addBoolVectorEntry(n, t);
			this.i._addBoolVectorToInputSidePacket(n, t);
		}));
	}
	addDoubleVectorToInputSidePacket(e, t) {
		H(this, t, ((t) => {
			let n = this.i._allocateDoubleVector(e.length);
			if (!n) throw Error("Unable to allocate new double vector on heap.");
			for (let t of e) this.i._addDoubleVectorEntry(n, t);
			this.i._addDoubleVectorToInputSidePacket(n, t);
		}));
	}
	addFloatVectorToInputSidePacket(e, t) {
		H(this, t, ((t) => {
			let n = this.i._allocateFloatVector(e.length);
			if (!n) throw Error("Unable to allocate new float vector on heap.");
			for (let t of e) this.i._addFloatVectorEntry(n, t);
			this.i._addFloatVectorToInputSidePacket(n, t);
		}));
	}
	addIntVectorToInputSidePacket(e, t) {
		H(this, t, ((t) => {
			let n = this.i._allocateIntVector(e.length);
			if (!n) throw Error("Unable to allocate new int vector on heap.");
			for (let t of e) this.i._addIntVectorEntry(n, t);
			this.i._addIntVectorToInputSidePacket(n, t);
		}));
	}
	addUintVectorToInputSidePacket(e, t) {
		H(this, t, ((t) => {
			let n = this.i._allocateUintVector(e.length);
			if (!n) throw Error("Unable to allocate new unsigned int vector on heap.");
			for (let t of e) this.i._addUintVectorEntry(n, t);
			this.i._addUintVectorToInputSidePacket(n, t);
		}));
	}
	addStringVectorToInputSidePacket(e, t) {
		H(this, t, ((t) => {
			let n = this.i._allocateStringVector(e.length);
			if (!n) throw Error("Unable to allocate new string vector on heap.");
			for (let t of e) H(this, t, ((e) => {
				this.i._addStringVectorEntry(n, e);
			}));
			this.i._addStringVectorToInputSidePacket(n, t);
		}));
	}
	attachBoolListener(e, t) {
		cc(this, e, t), H(this, e, ((e) => {
			this.i._attachBoolListener(e);
		}));
	}
	attachBoolVectorListener(e, t) {
		lc(this, e, t), H(this, e, ((e) => {
			this.i._attachBoolVectorListener(e);
		}));
	}
	attachIntListener(e, t) {
		cc(this, e, t), H(this, e, ((e) => {
			this.i._attachIntListener(e);
		}));
	}
	attachIntVectorListener(e, t) {
		lc(this, e, t), H(this, e, ((e) => {
			this.i._attachIntVectorListener(e);
		}));
	}
	attachUintListener(e, t) {
		cc(this, e, t), H(this, e, ((e) => {
			this.i._attachUintListener(e);
		}));
	}
	attachUintVectorListener(e, t) {
		lc(this, e, t), H(this, e, ((e) => {
			this.i._attachUintVectorListener(e);
		}));
	}
	attachDoubleListener(e, t) {
		cc(this, e, t), H(this, e, ((e) => {
			this.i._attachDoubleListener(e);
		}));
	}
	attachDoubleVectorListener(e, t) {
		lc(this, e, t), H(this, e, ((e) => {
			this.i._attachDoubleVectorListener(e);
		}));
	}
	attachFloatListener(e, t) {
		cc(this, e, t), H(this, e, ((e) => {
			this.i._attachFloatListener(e);
		}));
	}
	attachFloatVectorListener(e, t) {
		lc(this, e, t), H(this, e, ((e) => {
			this.i._attachFloatVectorListener(e);
		}));
	}
	attachStringListener(e, t) {
		cc(this, e, t), H(this, e, ((e) => {
			this.i._attachStringListener(e);
		}));
	}
	attachStringVectorListener(e, t) {
		lc(this, e, t), H(this, e, ((e) => {
			this.i._attachStringVectorListener(e);
		}));
	}
	attachProtoListener(e, t, n) {
		cc(this, e, t), H(this, e, ((e) => {
			this.i._attachProtoListener(e, n || !1);
		}));
	}
	attachProtoVectorListener(e, t, n) {
		lc(this, e, t), H(this, e, ((e) => {
			this.i._attachProtoVectorListener(e, n || !1);
		}));
	}
	attachAudioListener(e, t, n) {
		this.i._attachAudioListener || console.warn("Attempting to use attachAudioListener without support for output audio. Is build dep \":gl_graph_runner_audio_out\" missing?"), cc(this, e, ((e, n) => {
			e = new Float32Array(e.buffer, e.byteOffset, e.length / 4), t(e, n);
		})), H(this, e, ((e) => {
			this.i._attachAudioListener(e, n || !1);
		}));
	}
	finishProcessing() {
		this.i._waitUntilIdle();
	}
	closeGraph() {
		this.i._closeGraph(), this.i.simpleListeners = void 0, this.i.emptyPacketListeners = void 0;
	}
}, class extends cl {
	get ga() {
		return this.i;
	}
	pa(e, t, n) {
		H(this, t, ((t) => {
			let [r, i] = oc(this, e, t);
			this.ga._addBoundTextureAsImageToStream(t, r, i, n);
		}));
	}
	Z(e, t) {
		cc(this, e, t), H(this, e, ((e) => {
			this.ga._attachImageListener(e);
		}));
	}
	aa(e, t) {
		lc(this, e, t), H(this, e, ((e) => {
			this.ga._attachImageVectorListener(e);
		}));
	}
})), cl, ll = class extends sl {};
async function W(e, t, n) {
	return async function(e, t, n, r) {
		return uc(e, t, n, r);
	}(e, n.canvas ?? (rc() ? void 0 : document.createElement("canvas")), t, n);
}
function ul(e, t, n, r) {
	if (e.U) {
		let a = new No();
		if (n?.regionOfInterest) {
			if (!e.oa) throw Error("This task doesn't support region-of-interest.");
			var i = n.regionOfInterest;
			if (i.left >= i.right || i.top >= i.bottom) throw Error("Expected RectF with left < right and top < bottom.");
			if (i.left < 0 || i.top < 0 || i.right > 1 || i.bottom > 1) throw Error("Expected RectF values to be in [0,1].");
			z(a, 1, (i.left + i.right) / 2), z(a, 2, (i.top + i.bottom) / 2), z(a, 4, i.right - i.left), z(a, 3, i.bottom - i.top);
		} else z(a, 1, .5), z(a, 2, .5), z(a, 4, 1), z(a, 3, 1);
		if (n?.rotationDegrees) {
			if (n?.rotationDegrees % 90 != 0) throw Error("Expected rotation to be a multiple of 90°.");
			if (z(a, 5, -Math.PI * n.rotationDegrees / 180), n?.rotationDegrees % 180 != 0) {
				let [e, r] = ac(t);
				n = Nr(a, 3) * r / e, i = Nr(a, 4) * e / r, z(a, 4, n), z(a, 3, i);
			}
		}
		e.g.addProtoToStream(a.g(), "mediapipe.NormalizedRect", e.U, r);
	}
	e.g.pa(t, e.X, r ?? performance.now()), e.finishProcessing();
}
function dl(e, t, n) {
	if (e.baseOptions?.g()) throw Error("Task is not initialized with image mode. 'runningMode' must be set to 'IMAGE'.");
	ul(e, t, n, e.C + 1);
}
function fl(e, t, n, r) {
	if (!e.baseOptions?.g()) throw Error("Task is not initialized with video mode. 'runningMode' must be set to 'VIDEO'.");
	ul(e, t, n, r);
}
function pl(e, t, n, r) {
	var i = t.data;
	let a = t.width, o = a * (t = t.height);
	if ((i instanceof Uint8Array || i instanceof Float32Array) && i.length !== o) throw Error("Unsupported channel count: " + i.length / o);
	return e = new Rc([i], n, !1, e.g.i.canvas, e.P, a, t), r ? e.clone() : e;
}
var ml = class extends gc {
	constructor(e, t, n, r) {
		super(e), this.g = e, this.X = t, this.U = n, this.oa = r, this.P = new Ec();
	}
	l(e, t = !0) {
		if ("runningMode" in e && cr(this.baseOptions, 2, sn(!!e.runningMode && e.runningMode !== "IMAGE")), e.canvas !== void 0 && this.g.i.canvas !== e.canvas) throw Error("You must create a new task to reset the canvas.");
		return super.l(e, t);
	}
	close() {
		this.P.close(), super.close();
	}
};
ml.prototype.close = ml.prototype.close;
var hl = class extends ml {
	constructor(e, t) {
		super(new ll(e, t), "image_in", "norm_rect_in", !1), this.j = { detections: [] }, R(e = this.h = new Zo(), 0, 1, t = new Jo()), z(this.h, 2, .5), z(this.h, 3, .3);
	}
	get baseOptions() {
		return L(this.h, Jo, 1);
	}
	set baseOptions(e) {
		R(this.h, 0, 1, e);
	}
	o(e) {
		return "minDetectionConfidence" in e && z(this.h, 2, e.minDetectionConfidence ?? .5), "minSuppressionThreshold" in e && z(this.h, 3, e.minSuppressionThreshold ?? .3), this.l(e);
	}
	F(e, t) {
		return this.j = { detections: [] }, dl(this, e, t), this.j;
	}
	G(e, t, n) {
		return this.j = { detections: [] }, fl(this, e, n, t), this.j;
	}
	m() {
		var e = new go();
		mo(e, "image_in"), mo(e, "norm_rect_in"), ho(e, "detections");
		let t = new io();
		Ni(t, $o, this.h);
		let n = new co();
		Fr(n, 2, "mediapipe.tasks.vision.face_detector.FaceDetectorGraph"), so(n, "IMAGE:image_in"), so(n, "NORM_RECT:norm_rect_in"), V(n, "DETECTIONS:detections"), n.o(t), po(e, n), this.g.attachProtoVectorListener("detections", ((e, t) => {
			for (let t of e) e = Eo(t), this.j.detections.push(qs(e));
			U(this, t);
		})), this.g.attachEmptyPacketListener("detections", ((e) => {
			U(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
hl.prototype.detectForVideo = hl.prototype.G, hl.prototype.detect = hl.prototype.F, hl.prototype.setOptions = hl.prototype.o, hl.createFromModelPath = async function(e, t) {
	return W(hl, e, { baseOptions: { modelAssetPath: t } });
}, hl.createFromModelBuffer = function(e, t) {
	return W(hl, e, { baseOptions: { modelAssetBuffer: t } });
}, hl.createFromOptions = function(e, t) {
	return W(hl, e, t);
};
var gl = ol([61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314], [314, 405], [405, 321], [321, 375], [375, 291], [61, 185], [185, 40], [40, 39], [39, 37], [37, 0], [0, 267], [267, 269], [269, 270], [270, 409], [409, 291], [78, 95], [95, 88], [88, 178], [178, 87], [87, 14], [14, 317], [317, 402], [402, 318], [318, 324], [324, 308], [78, 191], [191, 80], [80, 81], [81, 82], [82, 13], [13, 312], [312, 311], [311, 310], [310, 415], [415, 308]), _l = ol([263, 249], [249, 390], [390, 373], [373, 374], [374, 380], [380, 381], [381, 382], [382, 362], [263, 466], [466, 388], [388, 387], [387, 386], [386, 385], [385, 384], [384, 398], [398, 362]), vl = ol([276, 283], [283, 282], [282, 295], [295, 285], [300, 293], [293, 334], [334, 296], [296, 336]), yl = ol([474, 475], [475, 476], [476, 477], [477, 474]), bl = ol([33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155], [155, 133], [33, 246], [246, 161], [161, 160], [160, 159], [159, 158], [158, 157], [157, 173], [173, 133]), xl = ol([46, 53], [53, 52], [52, 65], [65, 55], [70, 63], [63, 105], [105, 66], [66, 107]), Sl = ol([469, 470], [470, 471], [471, 472], [472, 469]), Cl = ol([10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397], [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152], [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172], [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162], [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10]), wl = [
	...gl,
	..._l,
	...vl,
	...bl,
	...xl,
	...Cl
], Tl = ol([127, 34], [34, 139], [139, 127], [11, 0], [0, 37], [37, 11], [232, 231], [231, 120], [120, 232], [72, 37], [37, 39], [39, 72], [128, 121], [121, 47], [47, 128], [232, 121], [121, 128], [128, 232], [104, 69], [69, 67], [67, 104], [175, 171], [171, 148], [148, 175], [118, 50], [50, 101], [101, 118], [73, 39], [39, 40], [40, 73], [9, 151], [151, 108], [108, 9], [48, 115], [115, 131], [131, 48], [194, 204], [204, 211], [211, 194], [74, 40], [40, 185], [185, 74], [80, 42], [42, 183], [183, 80], [40, 92], [92, 186], [186, 40], [230, 229], [229, 118], [118, 230], [202, 212], [212, 214], [214, 202], [83, 18], [18, 17], [17, 83], [76, 61], [61, 146], [146, 76], [160, 29], [29, 30], [30, 160], [56, 157], [157, 173], [173, 56], [106, 204], [204, 194], [194, 106], [135, 214], [214, 192], [192, 135], [203, 165], [165, 98], [98, 203], [21, 71], [71, 68], [68, 21], [51, 45], [45, 4], [4, 51], [144, 24], [24, 23], [23, 144], [77, 146], [146, 91], [91, 77], [205, 50], [50, 187], [187, 205], [201, 200], [200, 18], [18, 201], [91, 106], [106, 182], [182, 91], [90, 91], [91, 181], [181, 90], [85, 84], [84, 17], [17, 85], [206, 203], [203, 36], [36, 206], [148, 171], [171, 140], [140, 148], [92, 40], [40, 39], [39, 92], [193, 189], [189, 244], [244, 193], [159, 158], [158, 28], [28, 159], [247, 246], [246, 161], [161, 247], [236, 3], [3, 196], [196, 236], [54, 68], [68, 104], [104, 54], [193, 168], [168, 8], [8, 193], [117, 228], [228, 31], [31, 117], [189, 193], [193, 55], [55, 189], [98, 97], [97, 99], [99, 98], [126, 47], [47, 100], [100, 126], [166, 79], [79, 218], [218, 166], [155, 154], [154, 26], [26, 155], [209, 49], [49, 131], [131, 209], [135, 136], [136, 150], [150, 135], [47, 126], [126, 217], [217, 47], [223, 52], [52, 53], [53, 223], [45, 51], [51, 134], [134, 45], [211, 170], [170, 140], [140, 211], [67, 69], [69, 108], [108, 67], [43, 106], [106, 91], [91, 43], [230, 119], [119, 120], [120, 230], [226, 130], [130, 247], [247, 226], [63, 53], [53, 52], [52, 63], [238, 20], [20, 242], [242, 238], [46, 70], [70, 156], [156, 46], [78, 62], [62, 96], [96, 78], [46, 53], [53, 63], [63, 46], [143, 34], [34, 227], [227, 143], [123, 117], [117, 111], [111, 123], [44, 125], [125, 19], [19, 44], [236, 134], [134, 51], [51, 236], [216, 206], [206, 205], [205, 216], [154, 153], [153, 22], [22, 154], [39, 37], [37, 167], [167, 39], [200, 201], [201, 208], [208, 200], [36, 142], [142, 100], [100, 36], [57, 212], [212, 202], [202, 57], [20, 60], [60, 99], [99, 20], [28, 158], [158, 157], [157, 28], [35, 226], [226, 113], [113, 35], [160, 159], [159, 27], [27, 160], [204, 202], [202, 210], [210, 204], [113, 225], [225, 46], [46, 113], [43, 202], [202, 204], [204, 43], [62, 76], [76, 77], [77, 62], [137, 123], [123, 116], [116, 137], [41, 38], [38, 72], [72, 41], [203, 129], [129, 142], [142, 203], [64, 98], [98, 240], [240, 64], [49, 102], [102, 64], [64, 49], [41, 73], [73, 74], [74, 41], [212, 216], [216, 207], [207, 212], [42, 74], [74, 184], [184, 42], [169, 170], [170, 211], [211, 169], [170, 149], [149, 176], [176, 170], [105, 66], [66, 69], [69, 105], [122, 6], [6, 168], [168, 122], [123, 147], [147, 187], [187, 123], [96, 77], [77, 90], [90, 96], [65, 55], [55, 107], [107, 65], [89, 90], [90, 180], [180, 89], [101, 100], [100, 120], [120, 101], [63, 105], [105, 104], [104, 63], [93, 137], [137, 227], [227, 93], [15, 86], [86, 85], [85, 15], [129, 102], [102, 49], [49, 129], [14, 87], [87, 86], [86, 14], [55, 8], [8, 9], [9, 55], [100, 47], [47, 121], [121, 100], [145, 23], [23, 22], [22, 145], [88, 89], [89, 179], [179, 88], [6, 122], [122, 196], [196, 6], [88, 95], [95, 96], [96, 88], [138, 172], [172, 136], [136, 138], [215, 58], [58, 172], [172, 215], [115, 48], [48, 219], [219, 115], [42, 80], [80, 81], [81, 42], [195, 3], [3, 51], [51, 195], [43, 146], [146, 61], [61, 43], [171, 175], [175, 199], [199, 171], [81, 82], [82, 38], [38, 81], [53, 46], [46, 225], [225, 53], [144, 163], [163, 110], [110, 144], [52, 65], [65, 66], [66, 52], [229, 228], [228, 117], [117, 229], [34, 127], [127, 234], [234, 34], [107, 108], [108, 69], [69, 107], [109, 108], [108, 151], [151, 109], [48, 64], [64, 235], [235, 48], [62, 78], [78, 191], [191, 62], [129, 209], [209, 126], [126, 129], [111, 35], [35, 143], [143, 111], [117, 123], [123, 50], [50, 117], [222, 65], [65, 52], [52, 222], [19, 125], [125, 141], [141, 19], [221, 55], [55, 65], [65, 221], [3, 195], [195, 197], [197, 3], [25, 7], [7, 33], [33, 25], [220, 237], [237, 44], [44, 220], [70, 71], [71, 139], [139, 70], [122, 193], [193, 245], [245, 122], [247, 130], [130, 33], [33, 247], [71, 21], [21, 162], [162, 71], [170, 169], [169, 150], [150, 170], [188, 174], [174, 196], [196, 188], [216, 186], [186, 92], [92, 216], [2, 97], [97, 167], [167, 2], [141, 125], [125, 241], [241, 141], [164, 167], [167, 37], [37, 164], [72, 38], [38, 12], [12, 72], [38, 82], [82, 13], [13, 38], [63, 68], [68, 71], [71, 63], [226, 35], [35, 111], [111, 226], [101, 50], [50, 205], [205, 101], [206, 92], [92, 165], [165, 206], [209, 198], [198, 217], [217, 209], [165, 167], [167, 97], [97, 165], [220, 115], [115, 218], [218, 220], [133, 112], [112, 243], [243, 133], [239, 238], [238, 241], [241, 239], [214, 135], [135, 169], [169, 214], [190, 173], [173, 133], [133, 190], [171, 208], [208, 32], [32, 171], [125, 44], [44, 237], [237, 125], [86, 87], [87, 178], [178, 86], [85, 86], [86, 179], [179, 85], [84, 85], [85, 180], [180, 84], [83, 84], [84, 181], [181, 83], [201, 83], [83, 182], [182, 201], [137, 93], [93, 132], [132, 137], [76, 62], [62, 183], [183, 76], [61, 76], [76, 184], [184, 61], [57, 61], [61, 185], [185, 57], [212, 57], [57, 186], [186, 212], [214, 207], [207, 187], [187, 214], [34, 143], [143, 156], [156, 34], [79, 239], [239, 237], [237, 79], [123, 137], [137, 177], [177, 123], [44, 1], [1, 4], [4, 44], [201, 194], [194, 32], [32, 201], [64, 102], [102, 129], [129, 64], [213, 215], [215, 138], [138, 213], [59, 166], [166, 219], [219, 59], [242, 99], [99, 97], [97, 242], [2, 94], [94, 141], [141, 2], [75, 59], [59, 235], [235, 75], [24, 110], [110, 228], [228, 24], [25, 130], [130, 226], [226, 25], [23, 24], [24, 229], [229, 23], [22, 23], [23, 230], [230, 22], [26, 22], [22, 231], [231, 26], [112, 26], [26, 232], [232, 112], [189, 190], [190, 243], [243, 189], [221, 56], [56, 190], [190, 221], [28, 56], [56, 221], [221, 28], [27, 28], [28, 222], [222, 27], [29, 27], [27, 223], [223, 29], [30, 29], [29, 224], [224, 30], [247, 30], [30, 225], [225, 247], [238, 79], [79, 20], [20, 238], [166, 59], [59, 75], [75, 166], [60, 75], [75, 240], [240, 60], [147, 177], [177, 215], [215, 147], [20, 79], [79, 166], [166, 20], [187, 147], [147, 213], [213, 187], [112, 233], [233, 244], [244, 112], [233, 128], [128, 245], [245, 233], [128, 114], [114, 188], [188, 128], [114, 217], [217, 174], [174, 114], [131, 115], [115, 220], [220, 131], [217, 198], [198, 236], [236, 217], [198, 131], [131, 134], [134, 198], [177, 132], [132, 58], [58, 177], [143, 35], [35, 124], [124, 143], [110, 163], [163, 7], [7, 110], [228, 110], [110, 25], [25, 228], [356, 389], [389, 368], [368, 356], [11, 302], [302, 267], [267, 11], [452, 350], [350, 349], [349, 452], [302, 303], [303, 269], [269, 302], [357, 343], [343, 277], [277, 357], [452, 453], [453, 357], [357, 452], [333, 332], [332, 297], [297, 333], [175, 152], [152, 377], [377, 175], [347, 348], [348, 330], [330, 347], [303, 304], [304, 270], [270, 303], [9, 336], [336, 337], [337, 9], [278, 279], [279, 360], [360, 278], [418, 262], [262, 431], [431, 418], [304, 408], [408, 409], [409, 304], [310, 415], [415, 407], [407, 310], [270, 409], [409, 410], [410, 270], [450, 348], [348, 347], [347, 450], [422, 430], [430, 434], [434, 422], [313, 314], [314, 17], [17, 313], [306, 307], [307, 375], [375, 306], [387, 388], [388, 260], [260, 387], [286, 414], [414, 398], [398, 286], [335, 406], [406, 418], [418, 335], [364, 367], [367, 416], [416, 364], [423, 358], [358, 327], [327, 423], [251, 284], [284, 298], [298, 251], [281, 5], [5, 4], [4, 281], [373, 374], [374, 253], [253, 373], [307, 320], [320, 321], [321, 307], [425, 427], [427, 411], [411, 425], [421, 313], [313, 18], [18, 421], [321, 405], [405, 406], [406, 321], [320, 404], [404, 405], [405, 320], [315, 16], [16, 17], [17, 315], [426, 425], [425, 266], [266, 426], [377, 400], [400, 369], [369, 377], [322, 391], [391, 269], [269, 322], [417, 465], [465, 464], [464, 417], [386, 257], [257, 258], [258, 386], [466, 260], [260, 388], [388, 466], [456, 399], [399, 419], [419, 456], [284, 332], [332, 333], [333, 284], [417, 285], [285, 8], [8, 417], [346, 340], [340, 261], [261, 346], [413, 441], [441, 285], [285, 413], [327, 460], [460, 328], [328, 327], [355, 371], [371, 329], [329, 355], [392, 439], [439, 438], [438, 392], [382, 341], [341, 256], [256, 382], [429, 420], [420, 360], [360, 429], [364, 394], [394, 379], [379, 364], [277, 343], [343, 437], [437, 277], [443, 444], [444, 283], [283, 443], [275, 440], [440, 363], [363, 275], [431, 262], [262, 369], [369, 431], [297, 338], [338, 337], [337, 297], [273, 375], [375, 321], [321, 273], [450, 451], [451, 349], [349, 450], [446, 342], [342, 467], [467, 446], [293, 334], [334, 282], [282, 293], [458, 461], [461, 462], [462, 458], [276, 353], [353, 383], [383, 276], [308, 324], [324, 325], [325, 308], [276, 300], [300, 293], [293, 276], [372, 345], [345, 447], [447, 372], [352, 345], [345, 340], [340, 352], [274, 1], [1, 19], [19, 274], [456, 248], [248, 281], [281, 456], [436, 427], [427, 425], [425, 436], [381, 256], [256, 252], [252, 381], [269, 391], [391, 393], [393, 269], [200, 199], [199, 428], [428, 200], [266, 330], [330, 329], [329, 266], [287, 273], [273, 422], [422, 287], [250, 462], [462, 328], [328, 250], [258, 286], [286, 384], [384, 258], [265, 353], [353, 342], [342, 265], [387, 259], [259, 257], [257, 387], [424, 431], [431, 430], [430, 424], [342, 353], [353, 276], [276, 342], [273, 335], [335, 424], [424, 273], [292, 325], [325, 307], [307, 292], [366, 447], [447, 345], [345, 366], [271, 303], [303, 302], [302, 271], [423, 266], [266, 371], [371, 423], [294, 455], [455, 460], [460, 294], [279, 278], [278, 294], [294, 279], [271, 272], [272, 304], [304, 271], [432, 434], [434, 427], [427, 432], [272, 407], [407, 408], [408, 272], [394, 430], [430, 431], [431, 394], [395, 369], [369, 400], [400, 395], [334, 333], [333, 299], [299, 334], [351, 417], [417, 168], [168, 351], [352, 280], [280, 411], [411, 352], [325, 319], [319, 320], [320, 325], [295, 296], [296, 336], [336, 295], [319, 403], [403, 404], [404, 319], [330, 348], [348, 349], [349, 330], [293, 298], [298, 333], [333, 293], [323, 454], [454, 447], [447, 323], [15, 16], [16, 315], [315, 15], [358, 429], [429, 279], [279, 358], [14, 15], [15, 316], [316, 14], [285, 336], [336, 9], [9, 285], [329, 349], [349, 350], [350, 329], [374, 380], [380, 252], [252, 374], [318, 402], [402, 403], [403, 318], [6, 197], [197, 419], [419, 6], [318, 319], [319, 325], [325, 318], [367, 364], [364, 365], [365, 367], [435, 367], [367, 397], [397, 435], [344, 438], [438, 439], [439, 344], [272, 271], [271, 311], [311, 272], [195, 5], [5, 281], [281, 195], [273, 287], [287, 291], [291, 273], [396, 428], [428, 199], [199, 396], [311, 271], [271, 268], [268, 311], [283, 444], [444, 445], [445, 283], [373, 254], [254, 339], [339, 373], [282, 334], [334, 296], [296, 282], [449, 347], [347, 346], [346, 449], [264, 447], [447, 454], [454, 264], [336, 296], [296, 299], [299, 336], [338, 10], [10, 151], [151, 338], [278, 439], [439, 455], [455, 278], [292, 407], [407, 415], [415, 292], [358, 371], [371, 355], [355, 358], [340, 345], [345, 372], [372, 340], [346, 347], [347, 280], [280, 346], [442, 443], [443, 282], [282, 442], [19, 94], [94, 370], [370, 19], [441, 442], [442, 295], [295, 441], [248, 419], [419, 197], [197, 248], [263, 255], [255, 359], [359, 263], [440, 275], [275, 274], [274, 440], [300, 383], [383, 368], [368, 300], [351, 412], [412, 465], [465, 351], [263, 467], [467, 466], [466, 263], [301, 368], [368, 389], [389, 301], [395, 378], [378, 379], [379, 395], [412, 351], [351, 419], [419, 412], [436, 426], [426, 322], [322, 436], [2, 164], [164, 393], [393, 2], [370, 462], [462, 461], [461, 370], [164, 0], [0, 267], [267, 164], [302, 11], [11, 12], [12, 302], [268, 12], [12, 13], [13, 268], [293, 300], [300, 301], [301, 293], [446, 261], [261, 340], [340, 446], [330, 266], [266, 425], [425, 330], [426, 423], [423, 391], [391, 426], [429, 355], [355, 437], [437, 429], [391, 327], [327, 326], [326, 391], [440, 457], [457, 438], [438, 440], [341, 382], [382, 362], [362, 341], [459, 457], [457, 461], [461, 459], [434, 430], [430, 394], [394, 434], [414, 463], [463, 362], [362, 414], [396, 369], [369, 262], [262, 396], [354, 461], [461, 457], [457, 354], [316, 403], [403, 402], [402, 316], [315, 404], [404, 403], [403, 315], [314, 405], [405, 404], [404, 314], [313, 406], [406, 405], [405, 313], [421, 418], [418, 406], [406, 421], [366, 401], [401, 361], [361, 366], [306, 408], [408, 407], [407, 306], [291, 409], [409, 408], [408, 291], [287, 410], [410, 409], [409, 287], [432, 436], [436, 410], [410, 432], [434, 416], [416, 411], [411, 434], [264, 368], [368, 383], [383, 264], [309, 438], [438, 457], [457, 309], [352, 376], [376, 401], [401, 352], [274, 275], [275, 4], [4, 274], [421, 428], [428, 262], [262, 421], [294, 327], [327, 358], [358, 294], [433, 416], [416, 367], [367, 433], [289, 455], [455, 439], [439, 289], [462, 370], [370, 326], [326, 462], [2, 326], [326, 370], [370, 2], [305, 460], [460, 455], [455, 305], [254, 449], [449, 448], [448, 254], [255, 261], [261, 446], [446, 255], [253, 450], [450, 449], [449, 253], [252, 451], [451, 450], [450, 252], [256, 452], [452, 451], [451, 256], [341, 453], [453, 452], [452, 341], [413, 464], [464, 463], [463, 413], [441, 413], [413, 414], [414, 441], [258, 442], [442, 441], [441, 258], [257, 443], [443, 442], [442, 257], [259, 444], [444, 443], [443, 259], [260, 445], [445, 444], [444, 260], [467, 342], [342, 445], [445, 467], [459, 458], [458, 250], [250, 459], [289, 392], [392, 290], [290, 289], [290, 328], [328, 460], [460, 290], [376, 433], [433, 435], [435, 376], [250, 290], [290, 392], [392, 250], [411, 416], [416, 433], [433, 411], [341, 463], [463, 464], [464, 341], [453, 464], [464, 465], [465, 453], [357, 465], [465, 412], [412, 357], [343, 412], [412, 399], [399, 343], [360, 363], [363, 440], [440, 360], [437, 399], [399, 456], [456, 437], [420, 456], [456, 363], [363, 420], [401, 435], [435, 288], [288, 401], [372, 383], [383, 353], [353, 372], [339, 255], [255, 249], [249, 339], [448, 261], [261, 255], [255, 448], [133, 243], [243, 190], [190, 133], [133, 155], [155, 112], [112, 133], [33, 246], [246, 247], [247, 33], [33, 130], [130, 25], [25, 33], [398, 384], [384, 286], [286, 398], [362, 398], [398, 414], [414, 362], [362, 463], [463, 341], [341, 362], [263, 359], [359, 467], [467, 263], [263, 249], [249, 255], [255, 263], [466, 467], [467, 260], [260, 466], [75, 60], [60, 166], [166, 75], [238, 239], [239, 79], [79, 238], [162, 127], [127, 139], [139, 162], [72, 11], [11, 37], [37, 72], [121, 232], [232, 120], [120, 121], [73, 72], [72, 39], [39, 73], [114, 128], [128, 47], [47, 114], [233, 232], [232, 128], [128, 233], [103, 104], [104, 67], [67, 103], [152, 175], [175, 148], [148, 152], [119, 118], [118, 101], [101, 119], [74, 73], [73, 40], [40, 74], [107, 9], [9, 108], [108, 107], [49, 48], [48, 131], [131, 49], [32, 194], [194, 211], [211, 32], [184, 74], [74, 185], [185, 184], [191, 80], [80, 183], [183, 191], [185, 40], [40, 186], [186, 185], [119, 230], [230, 118], [118, 119], [210, 202], [202, 214], [214, 210], [84, 83], [83, 17], [17, 84], [77, 76], [76, 146], [146, 77], [161, 160], [160, 30], [30, 161], [190, 56], [56, 173], [173, 190], [182, 106], [106, 194], [194, 182], [138, 135], [135, 192], [192, 138], [129, 203], [203, 98], [98, 129], [54, 21], [21, 68], [68, 54], [5, 51], [51, 4], [4, 5], [145, 144], [144, 23], [23, 145], [90, 77], [77, 91], [91, 90], [207, 205], [205, 187], [187, 207], [83, 201], [201, 18], [18, 83], [181, 91], [91, 182], [182, 181], [180, 90], [90, 181], [181, 180], [16, 85], [85, 17], [17, 16], [205, 206], [206, 36], [36, 205], [176, 148], [148, 140], [140, 176], [165, 92], [92, 39], [39, 165], [245, 193], [193, 244], [244, 245], [27, 159], [159, 28], [28, 27], [30, 247], [247, 161], [161, 30], [174, 236], [236, 196], [196, 174], [103, 54], [54, 104], [104, 103], [55, 193], [193, 8], [8, 55], [111, 117], [117, 31], [31, 111], [221, 189], [189, 55], [55, 221], [240, 98], [98, 99], [99, 240], [142, 126], [126, 100], [100, 142], [219, 166], [166, 218], [218, 219], [112, 155], [155, 26], [26, 112], [198, 209], [209, 131], [131, 198], [169, 135], [135, 150], [150, 169], [114, 47], [47, 217], [217, 114], [224, 223], [223, 53], [53, 224], [220, 45], [45, 134], [134, 220], [32, 211], [211, 140], [140, 32], [109, 67], [67, 108], [108, 109], [146, 43], [43, 91], [91, 146], [231, 230], [230, 120], [120, 231], [113, 226], [226, 247], [247, 113], [105, 63], [63, 52], [52, 105], [241, 238], [238, 242], [242, 241], [124, 46], [46, 156], [156, 124], [95, 78], [78, 96], [96, 95], [70, 46], [46, 63], [63, 70], [116, 143], [143, 227], [227, 116], [116, 123], [123, 111], [111, 116], [1, 44], [44, 19], [19, 1], [3, 236], [236, 51], [51, 3], [207, 216], [216, 205], [205, 207], [26, 154], [154, 22], [22, 26], [165, 39], [39, 167], [167, 165], [199, 200], [200, 208], [208, 199], [101, 36], [36, 100], [100, 101], [43, 57], [57, 202], [202, 43], [242, 20], [20, 99], [99, 242], [56, 28], [28, 157], [157, 56], [124, 35], [35, 113], [113, 124], [29, 160], [160, 27], [27, 29], [211, 204], [204, 210], [210, 211], [124, 113], [113, 46], [46, 124], [106, 43], [43, 204], [204, 106], [96, 62], [62, 77], [77, 96], [227, 137], [137, 116], [116, 227], [73, 41], [41, 72], [72, 73], [36, 203], [203, 142], [142, 36], [235, 64], [64, 240], [240, 235], [48, 49], [49, 64], [64, 48], [42, 41], [41, 74], [74, 42], [214, 212], [212, 207], [207, 214], [183, 42], [42, 184], [184, 183], [210, 169], [169, 211], [211, 210], [140, 170], [170, 176], [176, 140], [104, 105], [105, 69], [69, 104], [193, 122], [122, 168], [168, 193], [50, 123], [123, 187], [187, 50], [89, 96], [96, 90], [90, 89], [66, 65], [65, 107], [107, 66], [179, 89], [89, 180], [180, 179], [119, 101], [101, 120], [120, 119], [68, 63], [63, 104], [104, 68], [234, 93], [93, 227], [227, 234], [16, 15], [15, 85], [85, 16], [209, 129], [129, 49], [49, 209], [15, 14], [14, 86], [86, 15], [107, 55], [55, 9], [9, 107], [120, 100], [100, 121], [121, 120], [153, 145], [145, 22], [22, 153], [178, 88], [88, 179], [179, 178], [197, 6], [6, 196], [196, 197], [89, 88], [88, 96], [96, 89], [135, 138], [138, 136], [136, 135], [138, 215], [215, 172], [172, 138], [218, 115], [115, 219], [219, 218], [41, 42], [42, 81], [81, 41], [5, 195], [195, 51], [51, 5], [57, 43], [43, 61], [61, 57], [208, 171], [171, 199], [199, 208], [41, 81], [81, 38], [38, 41], [224, 53], [53, 225], [225, 224], [24, 144], [144, 110], [110, 24], [105, 52], [52, 66], [66, 105], [118, 229], [229, 117], [117, 118], [227, 34], [34, 234], [234, 227], [66, 107], [107, 69], [69, 66], [10, 109], [109, 151], [151, 10], [219, 48], [48, 235], [235, 219], [183, 62], [62, 191], [191, 183], [142, 129], [129, 126], [126, 142], [116, 111], [111, 143], [143, 116], [118, 117], [117, 50], [50, 118], [223, 222], [222, 52], [52, 223], [94, 19], [19, 141], [141, 94], [222, 221], [221, 65], [65, 222], [196, 3], [3, 197], [197, 196], [45, 220], [220, 44], [44, 45], [156, 70], [70, 139], [139, 156], [188, 122], [122, 245], [245, 188], [139, 71], [71, 162], [162, 139], [149, 170], [170, 150], [150, 149], [122, 188], [188, 196], [196, 122], [206, 216], [216, 92], [92, 206], [164, 2], [2, 167], [167, 164], [242, 141], [141, 241], [241, 242], [0, 164], [164, 37], [37, 0], [11, 72], [72, 12], [12, 11], [12, 38], [38, 13], [13, 12], [70, 63], [63, 71], [71, 70], [31, 226], [226, 111], [111, 31], [36, 101], [101, 205], [205, 36], [203, 206], [206, 165], [165, 203], [126, 209], [209, 217], [217, 126], [98, 165], [165, 97], [97, 98], [237, 220], [220, 218], [218, 237], [237, 239], [239, 241], [241, 237], [210, 214], [214, 169], [169, 210], [140, 171], [171, 32], [32, 140], [241, 125], [125, 237], [237, 241], [179, 86], [86, 178], [178, 179], [180, 85], [85, 179], [179, 180], [181, 84], [84, 180], [180, 181], [182, 83], [83, 181], [181, 182], [194, 201], [201, 182], [182, 194], [177, 137], [137, 132], [132, 177], [184, 76], [76, 183], [183, 184], [185, 61], [61, 184], [184, 185], [186, 57], [57, 185], [185, 186], [216, 212], [212, 186], [186, 216], [192, 214], [214, 187], [187, 192], [139, 34], [34, 156], [156, 139], [218, 79], [79, 237], [237, 218], [147, 123], [123, 177], [177, 147], [45, 44], [44, 4], [4, 45], [208, 201], [201, 32], [32, 208], [98, 64], [64, 129], [129, 98], [192, 213], [213, 138], [138, 192], [235, 59], [59, 219], [219, 235], [141, 242], [242, 97], [97, 141], [97, 2], [2, 141], [141, 97], [240, 75], [75, 235], [235, 240], [229, 24], [24, 228], [228, 229], [31, 25], [25, 226], [226, 31], [230, 23], [23, 229], [229, 230], [231, 22], [22, 230], [230, 231], [232, 26], [26, 231], [231, 232], [233, 112], [112, 232], [232, 233], [244, 189], [189, 243], [243, 244], [189, 221], [221, 190], [190, 189], [222, 28], [28, 221], [221, 222], [223, 27], [27, 222], [222, 223], [224, 29], [29, 223], [223, 224], [225, 30], [30, 224], [224, 225], [113, 247], [247, 225], [225, 113], [99, 60], [60, 240], [240, 99], [213, 147], [147, 215], [215, 213], [60, 20], [20, 166], [166, 60], [192, 187], [187, 213], [213, 192], [243, 112], [112, 244], [244, 243], [244, 233], [233, 245], [245, 244], [245, 128], [128, 188], [188, 245], [188, 114], [114, 174], [174, 188], [134, 131], [131, 220], [220, 134], [174, 217], [217, 236], [236, 174], [236, 198], [198, 134], [134, 236], [215, 177], [177, 58], [58, 215], [156, 143], [143, 124], [124, 156], [25, 110], [110, 7], [7, 25], [31, 228], [228, 25], [25, 31], [264, 356], [356, 368], [368, 264], [0, 11], [11, 267], [267, 0], [451, 452], [452, 349], [349, 451], [267, 302], [302, 269], [269, 267], [350, 357], [357, 277], [277, 350], [350, 452], [452, 357], [357, 350], [299, 333], [333, 297], [297, 299], [396, 175], [175, 377], [377, 396], [280, 347], [347, 330], [330, 280], [269, 303], [303, 270], [270, 269], [151, 9], [9, 337], [337, 151], [344, 278], [278, 360], [360, 344], [424, 418], [418, 431], [431, 424], [270, 304], [304, 409], [409, 270], [272, 310], [310, 407], [407, 272], [322, 270], [270, 410], [410, 322], [449, 450], [450, 347], [347, 449], [432, 422], [422, 434], [434, 432], [18, 313], [313, 17], [17, 18], [291, 306], [306, 375], [375, 291], [259, 387], [387, 260], [260, 259], [424, 335], [335, 418], [418, 424], [434, 364], [364, 416], [416, 434], [391, 423], [423, 327], [327, 391], [301, 251], [251, 298], [298, 301], [275, 281], [281, 4], [4, 275], [254, 373], [373, 253], [253, 254], [375, 307], [307, 321], [321, 375], [280, 425], [425, 411], [411, 280], [200, 421], [421, 18], [18, 200], [335, 321], [321, 406], [406, 335], [321, 320], [320, 405], [405, 321], [314, 315], [315, 17], [17, 314], [423, 426], [426, 266], [266, 423], [396, 377], [377, 369], [369, 396], [270, 322], [322, 269], [269, 270], [413, 417], [417, 464], [464, 413], [385, 386], [386, 258], [258, 385], [248, 456], [456, 419], [419, 248], [298, 284], [284, 333], [333, 298], [168, 417], [417, 8], [8, 168], [448, 346], [346, 261], [261, 448], [417, 413], [413, 285], [285, 417], [326, 327], [327, 328], [328, 326], [277, 355], [355, 329], [329, 277], [309, 392], [392, 438], [438, 309], [381, 382], [382, 256], [256, 381], [279, 429], [429, 360], [360, 279], [365, 364], [364, 379], [379, 365], [355, 277], [277, 437], [437, 355], [282, 443], [443, 283], [283, 282], [281, 275], [275, 363], [363, 281], [395, 431], [431, 369], [369, 395], [299, 297], [297, 337], [337, 299], [335, 273], [273, 321], [321, 335], [348, 450], [450, 349], [349, 348], [359, 446], [446, 467], [467, 359], [283, 293], [293, 282], [282, 283], [250, 458], [458, 462], [462, 250], [300, 276], [276, 383], [383, 300], [292, 308], [308, 325], [325, 292], [283, 276], [276, 293], [293, 283], [264, 372], [372, 447], [447, 264], [346, 352], [352, 340], [340, 346], [354, 274], [274, 19], [19, 354], [363, 456], [456, 281], [281, 363], [426, 436], [436, 425], [425, 426], [380, 381], [381, 252], [252, 380], [267, 269], [269, 393], [393, 267], [421, 200], [200, 428], [428, 421], [371, 266], [266, 329], [329, 371], [432, 287], [287, 422], [422, 432], [290, 250], [250, 328], [328, 290], [385, 258], [258, 384], [384, 385], [446, 265], [265, 342], [342, 446], [386, 387], [387, 257], [257, 386], [422, 424], [424, 430], [430, 422], [445, 342], [342, 276], [276, 445], [422, 273], [273, 424], [424, 422], [306, 292], [292, 307], [307, 306], [352, 366], [366, 345], [345, 352], [268, 271], [271, 302], [302, 268], [358, 423], [423, 371], [371, 358], [327, 294], [294, 460], [460, 327], [331, 279], [279, 294], [294, 331], [303, 271], [271, 304], [304, 303], [436, 432], [432, 427], [427, 436], [304, 272], [272, 408], [408, 304], [395, 394], [394, 431], [431, 395], [378, 395], [395, 400], [400, 378], [296, 334], [334, 299], [299, 296], [6, 351], [351, 168], [168, 6], [376, 352], [352, 411], [411, 376], [307, 325], [325, 320], [320, 307], [285, 295], [295, 336], [336, 285], [320, 319], [319, 404], [404, 320], [329, 330], [330, 349], [349, 329], [334, 293], [293, 333], [333, 334], [366, 323], [323, 447], [447, 366], [316, 15], [15, 315], [315, 316], [331, 358], [358, 279], [279, 331], [317, 14], [14, 316], [316, 317], [8, 285], [285, 9], [9, 8], [277, 329], [329, 350], [350, 277], [253, 374], [374, 252], [252, 253], [319, 318], [318, 403], [403, 319], [351, 6], [6, 419], [419, 351], [324, 318], [318, 325], [325, 324], [397, 367], [367, 365], [365, 397], [288, 435], [435, 397], [397, 288], [278, 344], [344, 439], [439, 278], [310, 272], [272, 311], [311, 310], [248, 195], [195, 281], [281, 248], [375, 273], [273, 291], [291, 375], [175, 396], [396, 199], [199, 175], [312, 311], [311, 268], [268, 312], [276, 283], [283, 445], [445, 276], [390, 373], [373, 339], [339, 390], [295, 282], [282, 296], [296, 295], [448, 449], [449, 346], [346, 448], [356, 264], [264, 454], [454, 356], [337, 336], [336, 299], [299, 337], [337, 338], [338, 151], [151, 337], [294, 278], [278, 455], [455, 294], [308, 292], [292, 415], [415, 308], [429, 358], [358, 355], [355, 429], [265, 340], [340, 372], [372, 265], [352, 346], [346, 280], [280, 352], [295, 442], [442, 282], [282, 295], [354, 19], [19, 370], [370, 354], [285, 441], [441, 295], [295, 285], [195, 248], [248, 197], [197, 195], [457, 440], [440, 274], [274, 457], [301, 300], [300, 368], [368, 301], [417, 351], [351, 465], [465, 417], [251, 301], [301, 389], [389, 251], [394, 395], [395, 379], [379, 394], [399, 412], [412, 419], [419, 399], [410, 436], [436, 322], [322, 410], [326, 2], [2, 393], [393, 326], [354, 370], [370, 461], [461, 354], [393, 164], [164, 267], [267, 393], [268, 302], [302, 12], [12, 268], [312, 268], [268, 13], [13, 312], [298, 293], [293, 301], [301, 298], [265, 446], [446, 340], [340, 265], [280, 330], [330, 425], [425, 280], [322, 426], [426, 391], [391, 322], [420, 429], [429, 437], [437, 420], [393, 391], [391, 326], [326, 393], [344, 440], [440, 438], [438, 344], [458, 459], [459, 461], [461, 458], [364, 434], [434, 394], [394, 364], [428, 396], [396, 262], [262, 428], [274, 354], [354, 457], [457, 274], [317, 316], [316, 402], [402, 317], [316, 315], [315, 403], [403, 316], [315, 314], [314, 404], [404, 315], [314, 313], [313, 405], [405, 314], [313, 421], [421, 406], [406, 313], [323, 366], [366, 361], [361, 323], [292, 306], [306, 407], [407, 292], [306, 291], [291, 408], [408, 306], [291, 287], [287, 409], [409, 291], [287, 432], [432, 410], [410, 287], [427, 434], [434, 411], [411, 427], [372, 264], [264, 383], [383, 372], [459, 309], [309, 457], [457, 459], [366, 352], [352, 401], [401, 366], [1, 274], [274, 4], [4, 1], [418, 421], [421, 262], [262, 418], [331, 294], [294, 358], [358, 331], [435, 433], [433, 367], [367, 435], [392, 289], [289, 439], [439, 392], [328, 462], [462, 326], [326, 328], [94, 2], [2, 370], [370, 94], [289, 305], [305, 455], [455, 289], [339, 254], [254, 448], [448, 339], [359, 255], [255, 446], [446, 359], [254, 253], [253, 449], [449, 254], [253, 252], [252, 450], [450, 253], [252, 256], [256, 451], [451, 252], [256, 341], [341, 452], [452, 256], [414, 413], [413, 463], [463, 414], [286, 441], [441, 414], [414, 286], [286, 258], [258, 441], [441, 286], [258, 257], [257, 442], [442, 258], [257, 259], [259, 443], [443, 257], [259, 260], [260, 444], [444, 259], [260, 467], [467, 445], [445, 260], [309, 459], [459, 250], [250, 309], [305, 289], [289, 290], [290, 305], [305, 290], [290, 460], [460, 305], [401, 376], [376, 435], [435, 401], [309, 250], [250, 392], [392, 309], [376, 411], [411, 433], [433, 376], [453, 341], [341, 464], [464, 453], [357, 453], [453, 465], [465, 357], [343, 357], [357, 412], [412, 343], [437, 343], [343, 399], [399, 437], [344, 360], [360, 440], [440, 344], [420, 437], [437, 456], [456, 420], [360, 420], [420, 363], [363, 360], [361, 401], [401, 288], [288, 361], [265, 372], [372, 353], [353, 265], [390, 339], [339, 249], [249, 390], [339, 448], [448, 255], [255, 339]);
function El(e) {
	e.j = {
		faceLandmarks: [],
		faceBlendshapes: [],
		facialTransformationMatrixes: []
	};
}
var Dl = class extends ml {
	constructor(e, t) {
		super(new ll(e, t), "image_in", "norm_rect", !1), this.j = {
			faceLandmarks: [],
			faceBlendshapes: [],
			facialTransformationMatrixes: []
		}, this.outputFacialTransformationMatrixes = this.outputFaceBlendshapes = !1, R(e = this.h = new ns(), 0, 1, t = new Jo()), this.A = new ts(), R(this.h, 0, 3, this.A), this.u = new Zo(), R(this.h, 0, 2, this.u), Pr(this.u, 4, 1), z(this.u, 2, .5), z(this.A, 2, .5), z(this.h, 4, .5);
	}
	get baseOptions() {
		return L(this.h, Jo, 1);
	}
	set baseOptions(e) {
		R(this.h, 0, 1, e);
	}
	o(e) {
		return "numFaces" in e && Pr(this.u, 4, e.numFaces ?? 1), "minFaceDetectionConfidence" in e && z(this.u, 2, e.minFaceDetectionConfidence ?? .5), "minTrackingConfidence" in e && z(this.h, 4, e.minTrackingConfidence ?? .5), "minFacePresenceConfidence" in e && z(this.A, 2, e.minFacePresenceConfidence ?? .5), "outputFaceBlendshapes" in e && (this.outputFaceBlendshapes = !!e.outputFaceBlendshapes), "outputFacialTransformationMatrixes" in e && (this.outputFacialTransformationMatrixes = !!e.outputFacialTransformationMatrixes), this.l(e);
	}
	F(e, t) {
		return El(this), dl(this, e, t), this.j;
	}
	G(e, t, n) {
		return El(this), fl(this, e, n, t), this.j;
	}
	m() {
		var e = new go();
		mo(e, "image_in"), mo(e, "norm_rect"), ho(e, "face_landmarks");
		let t = new io();
		Ni(t, is, this.h);
		let n = new co();
		Fr(n, 2, "mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph"), so(n, "IMAGE:image_in"), so(n, "NORM_RECT:norm_rect"), V(n, "NORM_LANDMARKS:face_landmarks"), n.o(t), po(e, n), this.g.attachProtoVectorListener("face_landmarks", ((e, t) => {
			for (let t of e) e = Ao(t), this.j.faceLandmarks.push(Js(e));
			U(this, t);
		})), this.g.attachEmptyPacketListener("face_landmarks", ((e) => {
			U(this, e);
		})), this.outputFaceBlendshapes && (ho(e, "blendshapes"), V(n, "BLENDSHAPES:blendshapes"), this.g.attachProtoVectorListener("blendshapes", ((e, t) => {
			if (this.outputFaceBlendshapes) for (let t of e) e = So(t), this.j.faceBlendshapes.push(Gs(e.g() ?? []));
			U(this, t);
		})), this.g.attachEmptyPacketListener("blendshapes", ((e) => {
			U(this, e);
		}))), this.outputFacialTransformationMatrixes && (ho(e, "face_geometry"), V(n, "FACE_GEOMETRY:face_geometry"), this.g.attachProtoVectorListener("face_geometry", ((e, t) => {
			if (this.outputFacialTransformationMatrixes) for (let t of e) (e = L(e = es(t), jo, 2)) && this.j.facialTransformationMatrixes.push({
				rows: Mr(e, 1) ?? 0 ?? 0,
				columns: Mr(e, 2) ?? 0 ?? 0,
				data: dr(e, 3, on, ur()).slice() ?? []
			});
			U(this, t);
		})), this.g.attachEmptyPacketListener("face_geometry", ((e) => {
			U(this, e);
		}))), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
Dl.prototype.detectForVideo = Dl.prototype.G, Dl.prototype.detect = Dl.prototype.F, Dl.prototype.setOptions = Dl.prototype.o, Dl.createFromModelPath = function(e, t) {
	return W(Dl, e, { baseOptions: { modelAssetPath: t } });
}, Dl.createFromModelBuffer = function(e, t) {
	return W(Dl, e, { baseOptions: { modelAssetBuffer: t } });
}, Dl.createFromOptions = function(e, t) {
	return W(Dl, e, t);
}, Dl.FACE_LANDMARKS_LIPS = gl, Dl.FACE_LANDMARKS_LEFT_EYE = _l, Dl.FACE_LANDMARKS_LEFT_EYEBROW = vl, Dl.FACE_LANDMARKS_LEFT_IRIS = yl, Dl.FACE_LANDMARKS_RIGHT_EYE = bl, Dl.FACE_LANDMARKS_RIGHT_EYEBROW = xl, Dl.FACE_LANDMARKS_RIGHT_IRIS = Sl, Dl.FACE_LANDMARKS_FACE_OVAL = Cl, Dl.FACE_LANDMARKS_CONTOURS = wl, Dl.FACE_LANDMARKS_TESSELATION = Tl;
var Ol = ol([0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]);
function kl(e) {
	e.gestures = [], e.landmarks = [], e.worldLandmarks = [], e.handedness = [];
}
function Al(e) {
	return e.gestures.length === 0 ? {
		gestures: [],
		landmarks: [],
		worldLandmarks: [],
		handedness: [],
		handednesses: []
	} : {
		gestures: e.gestures,
		landmarks: e.landmarks,
		worldLandmarks: e.worldLandmarks,
		handedness: e.handedness,
		handednesses: e.handedness
	};
}
function jl(e, t = !0) {
	let n = [];
	for (let i of e) {
		var r = So(i);
		e = [];
		for (let n of r.g()) r = t && Mr(n, 1) != null ? Mr(n, 1) ?? 0 : -1, e.push({
			score: Nr(n, 2) ?? 0,
			index: r,
			categoryName: xn(or(n, 3)) ?? "" ?? "",
			displayName: xn(or(n, 4)) ?? "" ?? ""
		});
		n.push(e);
	}
	return n;
}
var Ml = class extends ml {
	constructor(e, t) {
		super(new ll(e, t), "image_in", "norm_rect", !1), this.gestures = [], this.landmarks = [], this.worldLandmarks = [], this.handedness = [], R(e = this.j = new ds(), 0, 1, t = new Jo()), this.u = new us(), R(this.j, 0, 2, this.u), this.D = new ls(), R(this.u, 0, 3, this.D), this.A = new cs(), R(this.u, 0, 2, this.A), this.h = new ss(), R(this.j, 0, 3, this.h), z(this.A, 2, .5), z(this.u, 4, .5), z(this.D, 2, .5);
	}
	get baseOptions() {
		return L(this.j, Jo, 1);
	}
	set baseOptions(e) {
		R(this.j, 0, 1, e);
	}
	o(e) {
		if (Pr(this.A, 3, e.numHands ?? 1), "minHandDetectionConfidence" in e && z(this.A, 2, e.minHandDetectionConfidence ?? .5), "minTrackingConfidence" in e && z(this.u, 4, e.minTrackingConfidence ?? .5), "minHandPresenceConfidence" in e && z(this.D, 2, e.minHandPresenceConfidence ?? .5), e.cannedGesturesClassifierOptions) {
			var t = new as(), n = t, r = Us(e.cannedGesturesClassifierOptions, L(this.h, as, 3)?.l());
			R(n, 0, 2, r), R(this.h, 0, 3, t);
		} else e.cannedGesturesClassifierOptions === void 0 && L(this.h, as, 3)?.g();
		return e.customGesturesClassifierOptions ? (R(n = t = new as(), 0, 2, r = Us(e.customGesturesClassifierOptions, L(this.h, as, 4)?.l())), R(this.h, 0, 4, t)) : e.customGesturesClassifierOptions === void 0 && L(this.h, as, 4)?.g(), this.l(e);
	}
	Ha(e, t) {
		return kl(this), dl(this, e, t), Al(this);
	}
	Ia(e, t, n) {
		return kl(this), fl(this, e, n, t), Al(this);
	}
	m() {
		var e = new go();
		mo(e, "image_in"), mo(e, "norm_rect"), ho(e, "hand_gestures"), ho(e, "hand_landmarks"), ho(e, "world_hand_landmarks"), ho(e, "handedness");
		let t = new io();
		Ni(t, gs, this.j);
		let n = new co();
		Fr(n, 2, "mediapipe.tasks.vision.gesture_recognizer.GestureRecognizerGraph"), so(n, "IMAGE:image_in"), so(n, "NORM_RECT:norm_rect"), V(n, "HAND_GESTURES:hand_gestures"), V(n, "LANDMARKS:hand_landmarks"), V(n, "WORLD_LANDMARKS:world_hand_landmarks"), V(n, "HANDEDNESS:handedness"), n.o(t), po(e, n), this.g.attachProtoVectorListener("hand_landmarks", ((e, t) => {
			for (let t of e) {
				e = Ao(t);
				let n = [];
				for (let t of Dr(e, ko, 1)) n.push({
					x: Nr(t, 1) ?? 0,
					y: Nr(t, 2) ?? 0,
					z: Nr(t, 3) ?? 0,
					visibility: Nr(t, 4) ?? 0
				});
				this.landmarks.push(n);
			}
			U(this, t);
		})), this.g.attachEmptyPacketListener("hand_landmarks", ((e) => {
			U(this, e);
		})), this.g.attachProtoVectorListener("world_hand_landmarks", ((e, t) => {
			for (let t of e) {
				e = Oo(t);
				let n = [];
				for (let t of Dr(e, Do, 1)) n.push({
					x: Nr(t, 1) ?? 0,
					y: Nr(t, 2) ?? 0,
					z: Nr(t, 3) ?? 0,
					visibility: Nr(t, 4) ?? 0
				});
				this.worldLandmarks.push(n);
			}
			U(this, t);
		})), this.g.attachEmptyPacketListener("world_hand_landmarks", ((e) => {
			U(this, e);
		})), this.g.attachProtoVectorListener("hand_gestures", ((e, t) => {
			this.gestures.push(...jl(e, !1)), U(this, t);
		})), this.g.attachEmptyPacketListener("hand_gestures", ((e) => {
			U(this, e);
		})), this.g.attachProtoVectorListener("handedness", ((e, t) => {
			this.handedness.push(...jl(e)), U(this, t);
		})), this.g.attachEmptyPacketListener("handedness", ((e) => {
			U(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
function Nl(e) {
	return {
		landmarks: e.landmarks,
		worldLandmarks: e.worldLandmarks,
		handednesses: e.handedness,
		handedness: e.handedness
	};
}
Ml.prototype.recognizeForVideo = Ml.prototype.Ia, Ml.prototype.recognize = Ml.prototype.Ha, Ml.prototype.setOptions = Ml.prototype.o, Ml.createFromModelPath = function(e, t) {
	return W(Ml, e, { baseOptions: { modelAssetPath: t } });
}, Ml.createFromModelBuffer = function(e, t) {
	return W(Ml, e, { baseOptions: { modelAssetBuffer: t } });
}, Ml.createFromOptions = function(e, t) {
	return W(Ml, e, t);
}, Ml.HAND_CONNECTIONS = Ol;
var Pl = class extends ml {
	constructor(e, t) {
		super(new ll(e, t), "image_in", "norm_rect", !1), this.landmarks = [], this.worldLandmarks = [], this.handedness = [], R(e = this.h = new us(), 0, 1, t = new Jo()), this.u = new ls(), R(this.h, 0, 3, this.u), this.j = new cs(), R(this.h, 0, 2, this.j), Pr(this.j, 3, 1), z(this.j, 2, .5), z(this.u, 2, .5), z(this.h, 4, .5);
	}
	get baseOptions() {
		return L(this.h, Jo, 1);
	}
	set baseOptions(e) {
		R(this.h, 0, 1, e);
	}
	o(e) {
		return "numHands" in e && Pr(this.j, 3, e.numHands ?? 1), "minHandDetectionConfidence" in e && z(this.j, 2, e.minHandDetectionConfidence ?? .5), "minTrackingConfidence" in e && z(this.h, 4, e.minTrackingConfidence ?? .5), "minHandPresenceConfidence" in e && z(this.u, 2, e.minHandPresenceConfidence ?? .5), this.l(e);
	}
	F(e, t) {
		return this.landmarks = [], this.worldLandmarks = [], this.handedness = [], dl(this, e, t), Nl(this);
	}
	G(e, t, n) {
		return this.landmarks = [], this.worldLandmarks = [], this.handedness = [], fl(this, e, n, t), Nl(this);
	}
	m() {
		var e = new go();
		mo(e, "image_in"), mo(e, "norm_rect"), ho(e, "hand_landmarks"), ho(e, "world_hand_landmarks"), ho(e, "handedness");
		let t = new io();
		Ni(t, _s, this.h);
		let n = new co();
		Fr(n, 2, "mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph"), so(n, "IMAGE:image_in"), so(n, "NORM_RECT:norm_rect"), V(n, "LANDMARKS:hand_landmarks"), V(n, "WORLD_LANDMARKS:world_hand_landmarks"), V(n, "HANDEDNESS:handedness"), n.o(t), po(e, n), this.g.attachProtoVectorListener("hand_landmarks", ((e, t) => {
			for (let t of e) e = Ao(t), this.landmarks.push(Js(e));
			U(this, t);
		})), this.g.attachEmptyPacketListener("hand_landmarks", ((e) => {
			U(this, e);
		})), this.g.attachProtoVectorListener("world_hand_landmarks", ((e, t) => {
			for (let t of e) e = Oo(t), this.worldLandmarks.push(Ys(e));
			U(this, t);
		})), this.g.attachEmptyPacketListener("world_hand_landmarks", ((e) => {
			U(this, e);
		})), this.g.attachProtoVectorListener("handedness", ((e, t) => {
			var n = this.handedness, r = n.push;
			let i = [];
			for (let t of e) {
				e = So(t);
				let n = [];
				for (let t of e.g()) n.push({
					score: Nr(t, 2) ?? 0,
					index: Mr(t, 1) ?? 0 ?? -1,
					categoryName: xn(or(t, 3)) ?? "" ?? "",
					displayName: xn(or(t, 4)) ?? "" ?? ""
				});
				i.push(n);
			}
			r.call(n, ...i), U(this, t);
		})), this.g.attachEmptyPacketListener("handedness", ((e) => {
			U(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
Pl.prototype.detectForVideo = Pl.prototype.G, Pl.prototype.detect = Pl.prototype.F, Pl.prototype.setOptions = Pl.prototype.o, Pl.createFromModelPath = function(e, t) {
	return W(Pl, e, { baseOptions: { modelAssetPath: t } });
}, Pl.createFromModelBuffer = function(e, t) {
	return W(Pl, e, { baseOptions: { modelAssetBuffer: t } });
}, Pl.createFromOptions = function(e, t) {
	return W(Pl, e, t);
}, Pl.HAND_CONNECTIONS = Ol;
var Fl = ol([0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20], [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32]);
function Il(e) {
	e.h = {
		faceLandmarks: [],
		faceBlendshapes: [],
		poseLandmarks: [],
		poseWorldLandmarks: [],
		poseSegmentationMasks: [],
		leftHandLandmarks: [],
		leftHandWorldLandmarks: [],
		rightHandLandmarks: [],
		rightHandWorldLandmarks: []
	};
}
function Ll(e) {
	try {
		if (!e.D) return e.h;
		e.D(e.h);
	} finally {
		hc(e);
	}
}
function Rl(e, t) {
	e = Ao(e), t.push(Js(e));
}
var zl = class extends ml {
	constructor(e, t) {
		super(new ll(e, t), "input_frames_image", null, !1), this.h = {
			faceLandmarks: [],
			faceBlendshapes: [],
			poseLandmarks: [],
			poseWorldLandmarks: [],
			poseSegmentationMasks: [],
			leftHandLandmarks: [],
			leftHandWorldLandmarks: [],
			rightHandLandmarks: [],
			rightHandWorldLandmarks: []
		}, this.outputPoseSegmentationMasks = this.outputFaceBlendshapes = !1, R(e = this.j = new xs(), 0, 1, t = new Jo()), this.I = new ls(), R(this.j, 0, 2, this.I), this.W = new vs(), R(this.j, 0, 3, this.W), this.u = new Zo(), R(this.j, 0, 4, this.u), this.O = new ts(), R(this.j, 0, 5, this.O), this.A = new ys(), R(this.j, 0, 6, this.A), this.M = new bs(), R(this.j, 0, 7, this.M), z(this.u, 2, .5), z(this.u, 3, .3), z(this.O, 2, .5), z(this.A, 2, .5), z(this.A, 3, .3), z(this.M, 2, .5), z(this.I, 2, .5);
	}
	get baseOptions() {
		return L(this.j, Jo, 1);
	}
	set baseOptions(e) {
		R(this.j, 0, 1, e);
	}
	o(e) {
		return "minFaceDetectionConfidence" in e && z(this.u, 2, e.minFaceDetectionConfidence ?? .5), "minFaceSuppressionThreshold" in e && z(this.u, 3, e.minFaceSuppressionThreshold ?? .3), "minFacePresenceConfidence" in e && z(this.O, 2, e.minFacePresenceConfidence ?? .5), "outputFaceBlendshapes" in e && (this.outputFaceBlendshapes = !!e.outputFaceBlendshapes), "minPoseDetectionConfidence" in e && z(this.A, 2, e.minPoseDetectionConfidence ?? .5), "minPoseSuppressionThreshold" in e && z(this.A, 3, e.minPoseSuppressionThreshold ?? .3), "minPosePresenceConfidence" in e && z(this.M, 2, e.minPosePresenceConfidence ?? .5), "outputPoseSegmentationMasks" in e && (this.outputPoseSegmentationMasks = !!e.outputPoseSegmentationMasks), "minHandLandmarksConfidence" in e && z(this.I, 2, e.minHandLandmarksConfidence ?? .5), this.l(e);
	}
	F(e, t, n) {
		let r = typeof t == "function" ? {} : t;
		return this.D = typeof t == "function" ? t : n, Il(this), dl(this, e, r), Ll(this);
	}
	G(e, t, n, r) {
		let i = typeof n == "function" ? {} : n;
		return this.D = typeof n == "function" ? n : r, Il(this), fl(this, e, i, t), Ll(this);
	}
	m() {
		var e = new go();
		mo(e, "input_frames_image"), ho(e, "pose_landmarks"), ho(e, "pose_world_landmarks"), ho(e, "face_landmarks"), ho(e, "left_hand_landmarks"), ho(e, "left_hand_world_landmarks"), ho(e, "right_hand_landmarks"), ho(e, "right_hand_world_landmarks");
		let t = new io(), n = new Wa();
		Fr(n, 1, "type.googleapis.com/mediapipe.tasks.vision.holistic_landmarker.proto.HolisticLandmarkerGraphOptions"), function(e, t) {
			if (t != null) if (Array.isArray(t)) cr(e, 2, Bn(t, 0, Hn));
			else {
				if (!(typeof t == "string" || t instanceof Ve || Le(t))) throw Error("invalid value in Any.value field: " + t + " expected a ByteString, a base64 encoded string, a Uint8Array or a jspb array");
				vr(e, 2, vt(t, !1), ze());
			}
		}(n, this.j.g());
		let r = new co();
		Fr(r, 2, "mediapipe.tasks.vision.holistic_landmarker.HolisticLandmarkerGraph"), jr(r, 8, Wa, n), so(r, "IMAGE:input_frames_image"), V(r, "POSE_LANDMARKS:pose_landmarks"), V(r, "POSE_WORLD_LANDMARKS:pose_world_landmarks"), V(r, "FACE_LANDMARKS:face_landmarks"), V(r, "LEFT_HAND_LANDMARKS:left_hand_landmarks"), V(r, "LEFT_HAND_WORLD_LANDMARKS:left_hand_world_landmarks"), V(r, "RIGHT_HAND_LANDMARKS:right_hand_landmarks"), V(r, "RIGHT_HAND_WORLD_LANDMARKS:right_hand_world_landmarks"), r.o(t), po(e, r), pc(this, e), this.g.attachProtoListener("pose_landmarks", ((e, t) => {
			Rl(e, this.h.poseLandmarks), U(this, t);
		})), this.g.attachEmptyPacketListener("pose_landmarks", ((e) => {
			U(this, e);
		})), this.g.attachProtoListener("pose_world_landmarks", ((e, t) => {
			var n = this.h.poseWorldLandmarks;
			e = Oo(e), n.push(Ys(e)), U(this, t);
		})), this.g.attachEmptyPacketListener("pose_world_landmarks", ((e) => {
			U(this, e);
		})), this.outputPoseSegmentationMasks && (V(r, "POSE_SEGMENTATION_MASK:pose_segmentation_mask"), mc(this, "pose_segmentation_mask"), this.g.Z("pose_segmentation_mask", ((e, t) => {
			this.h.poseSegmentationMasks = [pl(this, e, !0, !this.D)], U(this, t);
		})), this.g.attachEmptyPacketListener("pose_segmentation_mask", ((e) => {
			this.h.poseSegmentationMasks = [], U(this, e);
		}))), this.g.attachProtoListener("face_landmarks", ((e, t) => {
			Rl(e, this.h.faceLandmarks), U(this, t);
		})), this.g.attachEmptyPacketListener("face_landmarks", ((e) => {
			U(this, e);
		})), this.outputFaceBlendshapes && (ho(e, "extra_blendshapes"), V(r, "FACE_BLENDSHAPES:extra_blendshapes"), this.g.attachProtoListener("extra_blendshapes", ((e, t) => {
			var n = this.h.faceBlendshapes;
			this.outputFaceBlendshapes && (e = So(e), n.push(Gs(e.g() ?? []))), U(this, t);
		})), this.g.attachEmptyPacketListener("extra_blendshapes", ((e) => {
			U(this, e);
		}))), this.g.attachProtoListener("left_hand_landmarks", ((e, t) => {
			Rl(e, this.h.leftHandLandmarks), U(this, t);
		})), this.g.attachEmptyPacketListener("left_hand_landmarks", ((e) => {
			U(this, e);
		})), this.g.attachProtoListener("left_hand_world_landmarks", ((e, t) => {
			var n = this.h.leftHandWorldLandmarks;
			e = Oo(e), n.push(Ys(e)), U(this, t);
		})), this.g.attachEmptyPacketListener("left_hand_world_landmarks", ((e) => {
			U(this, e);
		})), this.g.attachProtoListener("right_hand_landmarks", ((e, t) => {
			Rl(e, this.h.rightHandLandmarks), U(this, t);
		})), this.g.attachEmptyPacketListener("right_hand_landmarks", ((e) => {
			U(this, e);
		})), this.g.attachProtoListener("right_hand_world_landmarks", ((e, t) => {
			var n = this.h.rightHandWorldLandmarks;
			e = Oo(e), n.push(Ys(e)), U(this, t);
		})), this.g.attachEmptyPacketListener("right_hand_world_landmarks", ((e) => {
			U(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
zl.prototype.detectForVideo = zl.prototype.G, zl.prototype.detect = zl.prototype.F, zl.prototype.setOptions = zl.prototype.o, zl.createFromModelPath = function(e, t) {
	return W(zl, e, { baseOptions: { modelAssetPath: t } });
}, zl.createFromModelBuffer = function(e, t) {
	return W(zl, e, { baseOptions: { modelAssetBuffer: t } });
}, zl.createFromOptions = function(e, t) {
	return W(zl, e, t);
}, zl.HAND_CONNECTIONS = Ol, zl.POSE_CONNECTIONS = Fl, zl.FACE_LANDMARKS_LIPS = gl, zl.FACE_LANDMARKS_LEFT_EYE = _l, zl.FACE_LANDMARKS_LEFT_EYEBROW = vl, zl.FACE_LANDMARKS_LEFT_IRIS = yl, zl.FACE_LANDMARKS_RIGHT_EYE = bl, zl.FACE_LANDMARKS_RIGHT_EYEBROW = xl, zl.FACE_LANDMARKS_RIGHT_IRIS = Sl, zl.FACE_LANDMARKS_FACE_OVAL = Cl, zl.FACE_LANDMARKS_CONTOURS = wl, zl.FACE_LANDMARKS_TESSELATION = Tl;
var Bl = class extends ml {
	constructor(e, t) {
		super(new ll(e, t), "input_image", "norm_rect", !0), this.j = { classifications: [] }, R(e = this.h = new ws(), 0, 1, t = new Jo());
	}
	get baseOptions() {
		return L(this.h, Jo, 1);
	}
	set baseOptions(e) {
		R(this.h, 0, 1, e);
	}
	o(e) {
		return R(this.h, 0, 2, Us(e, L(this.h, Vo, 2))), this.l(e);
	}
	sa(e, t) {
		return this.j = { classifications: [] }, dl(this, e, t), this.j;
	}
	ta(e, t, n) {
		return this.j = { classifications: [] }, fl(this, e, n, t), this.j;
	}
	m() {
		var e = new go();
		mo(e, "input_image"), mo(e, "norm_rect"), ho(e, "classifications");
		let t = new io();
		Ni(t, Ts, this.h);
		let n = new co();
		Fr(n, 2, "mediapipe.tasks.vision.image_classifier.ImageClassifierGraph"), so(n, "IMAGE:input_image"), so(n, "NORM_RECT:norm_rect"), V(n, "CLASSIFICATIONS:classifications"), n.o(t), po(e, n), this.g.attachProtoListener("classifications", ((e, t) => {
			this.j = Ks(Fo(e)), U(this, t);
		})), this.g.attachEmptyPacketListener("classifications", ((e) => {
			U(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
Bl.prototype.classifyForVideo = Bl.prototype.ta, Bl.prototype.classify = Bl.prototype.sa, Bl.prototype.setOptions = Bl.prototype.o, Bl.createFromModelPath = function(e, t) {
	return W(Bl, e, { baseOptions: { modelAssetPath: t } });
}, Bl.createFromModelBuffer = function(e, t) {
	return W(Bl, e, { baseOptions: { modelAssetBuffer: t } });
}, Bl.createFromOptions = function(e, t) {
	return W(Bl, e, t);
};
var Vl = class extends ml {
	constructor(e, t) {
		super(new ll(e, t), "image_in", "norm_rect", !0), this.h = new Es(), this.embeddings = { embeddings: [] }, R(e = this.h, 0, 1, t = new Jo());
	}
	get baseOptions() {
		return L(this.h, Jo, 1);
	}
	set baseOptions(e) {
		R(this.h, 0, 1, e);
	}
	o(e) {
		var t = this.h, n = L(this.h, Uo, 2);
		return n = n ? n.clone() : new Uo(), e.l2Normalize === void 0 ? "l2Normalize" in e && cr(n, 1) : cr(n, 1, sn(e.l2Normalize)), e.quantize === void 0 ? "quantize" in e && cr(n, 2) : cr(n, 2, sn(e.quantize)), R(t, 0, 2, n), this.l(e);
	}
	za(e, t) {
		return dl(this, e, t), this.embeddings;
	}
	Aa(e, t, n) {
		return fl(this, e, n, t), this.embeddings;
	}
	m() {
		var e = new go();
		mo(e, "image_in"), mo(e, "norm_rect"), ho(e, "embeddings_out");
		let t = new io();
		Ni(t, Ds, this.h);
		let n = new co();
		Fr(n, 2, "mediapipe.tasks.vision.image_embedder.ImageEmbedderGraph"), so(n, "IMAGE:image_in"), so(n, "NORM_RECT:norm_rect"), V(n, "EMBEDDINGS:embeddings_out"), n.o(t), po(e, n), this.g.attachProtoListener("embeddings_out", ((e, t) => {
			e = Bo(e), this.embeddings = function(e) {
				return {
					embeddings: Dr(e, Ro, 1).map(((e) => {
						let t = {
							headIndex: Mr(e, 3) ?? 0 ?? -1,
							headName: xn(or(e, 4)) ?? "" ?? ""
						};
						var n = e.v;
						return Tr(n, 0 | n[I], Io, br(e, 1)) === void 0 ? (n = /* @__PURE__ */ new Uint8Array(), t.quantizedEmbedding = L(e, Lo, br(e, 2), void 0)?.na()?.h() ?? n) : (e = dr(e = L(e, Io, br(e, 1), void 0), 1, on, ur()), t.floatEmbedding = e.slice()), t;
					})),
					timestampMs: Ws(or(e, 2, void 0, void 0, vn) ?? ir)
				};
			}(e), U(this, t);
		})), this.g.attachEmptyPacketListener("embeddings_out", ((e) => {
			U(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
Vl.cosineSimilarity = function(e, t) {
	if (e.floatEmbedding && t.floatEmbedding) e = Zs(e.floatEmbedding, t.floatEmbedding);
	else {
		if (!e.quantizedEmbedding || !t.quantizedEmbedding) throw Error("Cannot compute cosine similarity between quantized and float embeddings.");
		e = Zs(Xs(e.quantizedEmbedding), Xs(t.quantizedEmbedding));
	}
	return e;
}, Vl.prototype.embedForVideo = Vl.prototype.Aa, Vl.prototype.embed = Vl.prototype.za, Vl.prototype.setOptions = Vl.prototype.o, Vl.createFromModelPath = function(e, t) {
	return W(Vl, e, { baseOptions: { modelAssetPath: t } });
}, Vl.createFromModelBuffer = function(e, t) {
	return W(Vl, e, { baseOptions: { modelAssetBuffer: t } });
}, Vl.createFromOptions = function(e, t) {
	return W(Vl, e, t);
};
var Hl = class {
	constructor(e, t, n) {
		this.confidenceMasks = e, this.categoryMask = t, this.qualityScores = n;
	}
	close() {
		this.confidenceMasks?.forEach(((e) => {
			e.close();
		})), this.categoryMask?.close();
	}
};
function Ul(e) {
	let t = function(e) {
		return Dr(e, co, 1);
	}(e.ca()).filter(((e) => (xn(or(e, 1)) ?? "").includes("mediapipe.tasks.TensorsToSegmentationCalculator")));
	if (e.u = [], t.length > 1) throw Error("The graph has more than one mediapipe.tasks.TensorsToSegmentationCalculator.");
	t.length === 1 && (L(t[0], io, 7)?.j()?.g() ?? /* @__PURE__ */ new Map()).forEach(((t, n) => {
		e.u[Number(n)] = xn(or(t, 1)) ?? "";
	}));
}
function Wl(e) {
	e.categoryMask = void 0, e.confidenceMasks = void 0, e.qualityScores = void 0;
}
function Gl(e) {
	try {
		let t = new Hl(e.confidenceMasks, e.categoryMask, e.qualityScores);
		if (!e.j) return t;
		e.j(t);
	} finally {
		hc(e);
	}
}
Hl.prototype.close = Hl.prototype.close;
var Kl = class extends ml {
	constructor(e, t) {
		super(new ll(e, t), "image_in", "norm_rect", !1), this.u = [], this.outputCategoryMask = !1, this.outputConfidenceMasks = !0, this.h = new Ms(), this.A = new Os(), R(this.h, 0, 3, this.A), R(e = this.h, 0, 1, t = new Jo());
	}
	get baseOptions() {
		return L(this.h, Jo, 1);
	}
	set baseOptions(e) {
		R(this.h, 0, 1, e);
	}
	o(e) {
		return e.displayNamesLocale === void 0 ? "displayNamesLocale" in e && cr(this.h, 2) : cr(this.h, 2, bn(e.displayNamesLocale)), "outputCategoryMask" in e && (this.outputCategoryMask = e.outputCategoryMask ?? !1), "outputConfidenceMasks" in e && (this.outputConfidenceMasks = e.outputConfidenceMasks ?? !0), super.l(e);
	}
	L() {
		Ul(this);
	}
	segment(e, t, n) {
		let r = typeof t == "function" ? {} : t;
		return this.j = typeof t == "function" ? t : n, Wl(this), dl(this, e, r), Gl(this);
	}
	La(e, t, n, r) {
		let i = typeof n == "function" ? {} : n;
		return this.j = typeof n == "function" ? n : r, Wl(this), fl(this, e, i, t), Gl(this);
	}
	Da() {
		return this.u;
	}
	m() {
		var e = new go();
		mo(e, "image_in"), mo(e, "norm_rect");
		let t = new io();
		Ni(t, Ns, this.h);
		let n = new co();
		Fr(n, 2, "mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph"), so(n, "IMAGE:image_in"), so(n, "NORM_RECT:norm_rect"), n.o(t), po(e, n), pc(this, e), this.outputConfidenceMasks && (ho(e, "confidence_masks"), V(n, "CONFIDENCE_MASKS:confidence_masks"), mc(this, "confidence_masks"), this.g.aa("confidence_masks", ((e, t) => {
			this.confidenceMasks = e.map(((e) => pl(this, e, !0, !this.j))), U(this, t);
		})), this.g.attachEmptyPacketListener("confidence_masks", ((e) => {
			this.confidenceMasks = [], U(this, e);
		}))), this.outputCategoryMask && (ho(e, "category_mask"), V(n, "CATEGORY_MASK:category_mask"), mc(this, "category_mask"), this.g.Z("category_mask", ((e, t) => {
			this.categoryMask = pl(this, e, !1, !this.j), U(this, t);
		})), this.g.attachEmptyPacketListener("category_mask", ((e) => {
			this.categoryMask = void 0, U(this, e);
		}))), ho(e, "quality_scores"), V(n, "QUALITY_SCORES:quality_scores"), this.g.attachFloatVectorListener("quality_scores", ((e, t) => {
			this.qualityScores = e, U(this, t);
		})), this.g.attachEmptyPacketListener("quality_scores", ((e) => {
			this.categoryMask = void 0, U(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
Kl.prototype.getLabels = Kl.prototype.Da, Kl.prototype.segmentForVideo = Kl.prototype.La, Kl.prototype.segment = Kl.prototype.segment, Kl.prototype.setOptions = Kl.prototype.o, Kl.createFromModelPath = function(e, t) {
	return W(Kl, e, { baseOptions: { modelAssetPath: t } });
}, Kl.createFromModelBuffer = function(e, t) {
	return W(Kl, e, { baseOptions: { modelAssetBuffer: t } });
}, Kl.createFromOptions = function(e, t) {
	return W(Kl, e, t);
};
var ql = class {
	constructor(e, t, n) {
		this.confidenceMasks = e, this.categoryMask = t, this.qualityScores = n;
	}
	close() {
		this.confidenceMasks?.forEach(((e) => {
			e.close();
		})), this.categoryMask?.close();
	}
};
ql.prototype.close = ql.prototype.close;
var Jl = class extends ml {
	constructor(e, t) {
		super(new ll(e, t), "image_in", "norm_rect_in", !1), this.outputCategoryMask = !1, this.outputConfidenceMasks = !0, this.h = new Ms(), this.u = new Os(), R(this.h, 0, 3, this.u), R(e = this.h, 0, 1, t = new Jo());
	}
	get baseOptions() {
		return L(this.h, Jo, 1);
	}
	set baseOptions(e) {
		R(this.h, 0, 1, e);
	}
	o(e) {
		return "outputCategoryMask" in e && (this.outputCategoryMask = e.outputCategoryMask ?? !1), "outputConfidenceMasks" in e && (this.outputConfidenceMasks = e.outputConfidenceMasks ?? !0), super.l(e);
	}
	segment(e, t, n, r) {
		let i = typeof n == "function" ? {} : n;
		if (this.j = typeof n == "function" ? n : r, this.qualityScores = this.categoryMask = this.confidenceMasks = void 0, n = this.C + 1, r = new Ls(), t.keypoint && t.scribble) throw Error("Cannot provide both keypoint and scribble.");
		if (t.keypoint) {
			var a = new Ps();
			vr(a, 3, sn(!0), !1), vr(a, 1, an(t.keypoint.x), 0), vr(a, 2, an(t.keypoint.y), 0), kr(r, 1, Rs, a);
		} else {
			if (!t.scribble) throw Error("Must provide either a keypoint or a scribble.");
			{
				let e = new Is();
				for (a of t.scribble) vr(t = new Ps(), 3, sn(!0), !1), vr(t, 1, an(a.x), 0), vr(t, 2, an(a.y), 0), jr(e, 1, Ps, t);
				kr(r, 2, Rs, e);
			}
		}
		this.g.addProtoToStream(r.g(), "mediapipe.tasks.vision.interactive_segmenter.proto.RegionOfInterest", "roi_in", n), dl(this, e, i);
		t: {
			try {
				let e = new ql(this.confidenceMasks, this.categoryMask, this.qualityScores);
				if (!this.j) {
					var o = e;
					break t;
				}
				this.j(e);
			} finally {
				hc(this);
			}
			o = void 0;
		}
		return o;
	}
	m() {
		var e = new go();
		mo(e, "image_in"), mo(e, "roi_in"), mo(e, "norm_rect_in");
		let t = new io();
		Ni(t, Ns, this.h);
		let n = new co();
		Fr(n, 2, "mediapipe.tasks.vision.interactive_segmenter.InteractiveSegmenterGraphV2"), so(n, "IMAGE:image_in"), so(n, "ROI:roi_in"), so(n, "NORM_RECT:norm_rect_in"), n.o(t), po(e, n), pc(this, e), this.outputConfidenceMasks && (ho(e, "confidence_masks"), V(n, "CONFIDENCE_MASKS:confidence_masks"), mc(this, "confidence_masks"), this.g.aa("confidence_masks", ((e, t) => {
			this.confidenceMasks = e.map(((e) => pl(this, e, !0, !this.j))), U(this, t);
		})), this.g.attachEmptyPacketListener("confidence_masks", ((e) => {
			this.confidenceMasks = [], U(this, e);
		}))), this.outputCategoryMask && (ho(e, "category_mask"), V(n, "CATEGORY_MASK:category_mask"), mc(this, "category_mask"), this.g.Z("category_mask", ((e, t) => {
			this.categoryMask = pl(this, e, !1, !this.j), U(this, t);
		})), this.g.attachEmptyPacketListener("category_mask", ((e) => {
			this.categoryMask = void 0, U(this, e);
		}))), ho(e, "quality_scores"), V(n, "QUALITY_SCORES:quality_scores"), this.g.attachFloatVectorListener("quality_scores", ((e, t) => {
			this.qualityScores = e, U(this, t);
		})), this.g.attachEmptyPacketListener("quality_scores", ((e) => {
			this.categoryMask = void 0, U(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
Jl.prototype.segment = Jl.prototype.segment, Jl.prototype.setOptions = Jl.prototype.o, Jl.createFromModelPath = function(e, t) {
	return W(Jl, e, { baseOptions: { modelAssetPath: t } });
}, Jl.createFromModelBuffer = function(e, t) {
	return W(Jl, e, { baseOptions: { modelAssetBuffer: t } });
}, Jl.createFromOptions = function(e, t) {
	return W(Jl, e, t);
};
var Yl = class extends ml {
	constructor(e, t) {
		super(new ll(e, t), "input_frame_gpu", "norm_rect", !1), this.j = { detections: [] }, R(e = this.h = new zs(), 0, 1, t = new Jo());
	}
	get baseOptions() {
		return L(this.h, Jo, 1);
	}
	set baseOptions(e) {
		R(this.h, 0, 1, e);
	}
	o(e) {
		return e.displayNamesLocale === void 0 ? "displayNamesLocale" in e && cr(this.h, 2) : cr(this.h, 2, bn(e.displayNamesLocale)), e.maxResults === void 0 ? "maxResults" in e && cr(this.h, 3) : Pr(this.h, 3, e.maxResults), e.scoreThreshold === void 0 ? "scoreThreshold" in e && cr(this.h, 4) : z(this.h, 4, e.scoreThreshold), e.categoryAllowlist === void 0 ? "categoryAllowlist" in e && cr(this.h, 5) : Ir(this.h, 5, e.categoryAllowlist), e.categoryDenylist === void 0 ? "categoryDenylist" in e && cr(this.h, 6) : Ir(this.h, 6, e.categoryDenylist), this.l(e);
	}
	F(e, t) {
		return this.j = { detections: [] }, dl(this, e, t), this.j;
	}
	G(e, t, n) {
		return this.j = { detections: [] }, fl(this, e, n, t), this.j;
	}
	m() {
		var e = new go();
		mo(e, "input_frame_gpu"), mo(e, "norm_rect"), ho(e, "detections");
		let t = new io();
		Ni(t, Bs, this.h);
		let n = new co();
		Fr(n, 2, "mediapipe.tasks.vision.ObjectDetectorGraph"), so(n, "IMAGE:input_frame_gpu"), so(n, "NORM_RECT:norm_rect"), V(n, "DETECTIONS:detections"), n.o(t), po(e, n), this.g.attachProtoVectorListener("detections", ((e, t) => {
			for (let t of e) e = Eo(t), this.j.detections.push(qs(e));
			U(this, t);
		})), this.g.attachEmptyPacketListener("detections", ((e) => {
			U(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
Yl.prototype.detectForVideo = Yl.prototype.G, Yl.prototype.detect = Yl.prototype.F, Yl.prototype.setOptions = Yl.prototype.o, Yl.createFromModelPath = async function(e, t) {
	return W(Yl, e, { baseOptions: { modelAssetPath: t } });
}, Yl.createFromModelBuffer = function(e, t) {
	return W(Yl, e, { baseOptions: { modelAssetBuffer: t } });
}, Yl.createFromOptions = function(e, t) {
	return W(Yl, e, t);
};
var Xl = class {
	constructor(e, t, n) {
		this.landmarks = e, this.worldLandmarks = t, this.segmentationMasks = n;
	}
	close() {
		this.segmentationMasks?.forEach(((e) => {
			e.close();
		}));
	}
};
function Zl(e) {
	e.landmarks = [], e.worldLandmarks = [], e.segmentationMasks = void 0;
}
function Ql(e) {
	try {
		let t = new Xl(e.landmarks, e.worldLandmarks, e.segmentationMasks);
		if (!e.u) return t;
		e.u(t);
	} finally {
		hc(e);
	}
}
Xl.prototype.close = Xl.prototype.close;
var $l = class extends ml {
	constructor(e, t) {
		super(new ll(e, t), "image_in", "norm_rect", !1), this.landmarks = [], this.worldLandmarks = [], this.outputSegmentationMasks = !1, R(e = this.h = new Vs(), 0, 1, t = new Jo()), this.A = new bs(), R(this.h, 0, 3, this.A), this.j = new ys(), R(this.h, 0, 2, this.j), Pr(this.j, 4, 1), z(this.j, 2, .5), z(this.A, 2, .5), z(this.h, 4, .5);
	}
	get baseOptions() {
		return L(this.h, Jo, 1);
	}
	set baseOptions(e) {
		R(this.h, 0, 1, e);
	}
	o(e) {
		return "numPoses" in e && Pr(this.j, 4, e.numPoses ?? 1), "minPoseDetectionConfidence" in e && z(this.j, 2, e.minPoseDetectionConfidence ?? .5), "minTrackingConfidence" in e && z(this.h, 4, e.minTrackingConfidence ?? .5), "minPosePresenceConfidence" in e && z(this.A, 2, e.minPosePresenceConfidence ?? .5), "outputSegmentationMasks" in e && (this.outputSegmentationMasks = e.outputSegmentationMasks ?? !1), this.l(e);
	}
	F(e, t, n) {
		let r = typeof t == "function" ? {} : t;
		return this.u = typeof t == "function" ? t : n, Zl(this), dl(this, e, r), Ql(this);
	}
	G(e, t, n, r) {
		let i = typeof n == "function" ? {} : n;
		return this.u = typeof n == "function" ? n : r, Zl(this), fl(this, e, i, t), Ql(this);
	}
	m() {
		var e = new go();
		mo(e, "image_in"), mo(e, "norm_rect"), ho(e, "normalized_landmarks"), ho(e, "world_landmarks"), ho(e, "segmentation_masks");
		let t = new io();
		Ni(t, Hs, this.h);
		let n = new co();
		Fr(n, 2, "mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph"), so(n, "IMAGE:image_in"), so(n, "NORM_RECT:norm_rect"), V(n, "NORM_LANDMARKS:normalized_landmarks"), V(n, "WORLD_LANDMARKS:world_landmarks"), n.o(t), po(e, n), pc(this, e), this.g.attachProtoVectorListener("normalized_landmarks", ((e, t) => {
			this.landmarks = [];
			for (let t of e) e = Ao(t), this.landmarks.push(Js(e));
			U(this, t);
		})), this.g.attachEmptyPacketListener("normalized_landmarks", ((e) => {
			this.landmarks = [], U(this, e);
		})), this.g.attachProtoVectorListener("world_landmarks", ((e, t) => {
			this.worldLandmarks = [];
			for (let t of e) e = Oo(t), this.worldLandmarks.push(Ys(e));
			U(this, t);
		})), this.g.attachEmptyPacketListener("world_landmarks", ((e) => {
			this.worldLandmarks = [], U(this, e);
		})), this.outputSegmentationMasks && (V(n, "SEGMENTATION_MASK:segmentation_masks"), mc(this, "segmentation_masks"), this.g.aa("segmentation_masks", ((e, t) => {
			this.segmentationMasks = e.map(((e) => pl(this, e, !0, !this.u))), U(this, t);
		})), this.g.attachEmptyPacketListener("segmentation_masks", ((e) => {
			this.segmentationMasks = [], U(this, e);
		}))), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
$l.prototype.detectForVideo = $l.prototype.G, $l.prototype.detect = $l.prototype.F, $l.prototype.setOptions = $l.prototype.o, $l.createFromModelPath = function(e, t) {
	return W($l, e, { baseOptions: { modelAssetPath: t } });
}, $l.createFromModelBuffer = function(e, t) {
	return W($l, e, { baseOptions: { modelAssetBuffer: t } });
}, $l.createFromOptions = function(e, t) {
	return W($l, e, t);
}, $l.POSE_CONNECTIONS = Fl;
//#endregion
//#region frontend/src/gesture/mediapipe-adapter.ts
function eu(e) {
	let t = import.meta.url.slice(0, import.meta.url.lastIndexOf("/") + 1);
	return new URL(e, t).href;
}
async function tu() {
	let e = eu("mediapipe/wasm").replace(/\/$/, ""), t = eu("mediapipe/hand_landmarker.task"), n = await nc.forVisionTasks(e), r = {
		baseOptions: {
			modelAssetPath: t,
			delegate: "GPU"
		},
		runningMode: "VIDEO",
		numHands: 2,
		minHandDetectionConfidence: .68,
		minHandPresenceConfidence: .68,
		minTrackingConfidence: .68
	};
	try {
		return await Pl.createFromOptions(n, r);
	} catch {
		return await Pl.createFromOptions(n, {
			...r,
			baseOptions: {
				modelAssetPath: t,
				delegate: "CPU"
			}
		});
	}
}
//#endregion
//#region frontend/src/gesture/gesture-state-machine.ts
var nu = class {
	port;
	rightAction = "idle";
	hoveredNode = -1;
	hoverSince = 0;
	candidateNode = -1;
	pinchFrames = 0;
	releaseFrames = 0;
	cooldownUntil = 0;
	capturedNode = -1;
	actionPoint = null;
	actionMoved = !1;
	leftPalmPoint = null;
	twoPalmSince = 0;
	twoPalmResetDone = !1;
	constructor(e) {
		this.port = e;
	}
	process(e, t) {
		let n = e.find((e) => e.handedness === "left") ?? null, r = e.find((e) => e.handedness === "right") ?? null;
		if (!n && !r) {
			this.port.setGestureActive(!1), this.releaseRightAction(!1, t), this.clearRightPointer(), this.leftPalmPoint = null, this.twoPalmSince = 0, this.twoPalmResetDone = !1;
			return;
		}
		this.port.setGestureActive(!0), !this.processTwoPalmReset(n, r, t) && (this.processLeftPalmZoom(n), this.processRightPointer(r, t));
	}
	reset(e = performance.now()) {
		this.releaseRightAction(!1, e), this.clearRightPointer(), this.leftPalmPoint = null, this.port.setGestureActive(!1);
	}
	processTwoPalmReset(e, t, n) {
		return !e?.openPalm || !t?.openPalm ? (this.twoPalmSince = 0, this.twoPalmResetDone = !1, !1) : (this.releaseRightAction(!1, n), this.leftPalmPoint = null, this.port.setCursor(t.point, "idle"), this.twoPalmSince ||= n, !this.twoPalmResetDone && n - this.twoPalmSince >= 420 && (this.port.clearSelection(), this.port.fitToGraph(), this.twoPalmResetDone = !0), !0);
	}
	processLeftPalmZoom(e) {
		if (!e?.openPalm || e.pinch) {
			this.leftPalmPoint = null;
			return;
		}
		if (!this.leftPalmPoint) {
			this.leftPalmPoint = { ...e.palmPoint };
			return;
		}
		let t = e.palmPoint.y - this.leftPalmPoint.y;
		Math.abs(t) >= 2 && (this.port.zoomAt(e.palmPoint, Math.exp(-t / 230)), this.leftPalmPoint = { ...e.palmPoint });
	}
	processRightPointer(e, t) {
		if (!e) {
			this.releaseRightAction(!1, t), this.clearRightPointer();
			return;
		}
		this.port.setCursor(e.point, this.rightAction), (!e.pinch || this.rightAction !== "idle") && this.updateHover(e.point, t), this.processRightPinch(e, t);
	}
	updateHover(e, t) {
		if (this.rightAction !== "idle") return;
		let n = this.port.pick(e);
		n !== this.hoveredNode && (this.hoveredNode = n, this.hoverSince = t, this.candidateNode = -1), n >= 0 && t - this.hoverSince >= 140 ? this.candidateNode = n : n < 0 && (this.candidateNode = -1), this.port.hover(this.candidateNode >= 0 ? this.candidateNode : n);
	}
	processRightPinch(e, t) {
		if (t < this.cooldownUntil) return;
		if (e.pinch ? (this.pinchFrames += 1, this.releaseFrames = 0) : (this.releaseFrames += 1, this.pinchFrames = 0), !e.pinch) {
			this.releaseFrames >= 2 && this.rightAction !== "idle" && this.releaseRightAction(this.rightAction === "node-captured", t);
			return;
		}
		if (this.pinchFrames < 3 || (this.rightAction === "idle" && (this.actionPoint = { ...e.point }, this.actionMoved = !1, this.candidateNode >= 0 ? (this.rightAction = "node-captured", this.capturedNode = this.candidateNode, this.port.select(this.capturedNode), this.port.beginNodeDrag(this.capturedNode)) : this.rightAction = "canvas-pan"), !this.actionPoint)) return;
		let n = e.point.x - this.actionPoint.x, r = e.point.y - this.actionPoint.y;
		this.rightAction === "node-captured" && this.capturedNode >= 0 ? (!this.actionMoved && Math.hypot(n, r) >= 5 && (this.actionMoved = !0, this.port.setNodeDragging(!0)), this.actionMoved && this.port.moveNode(this.capturedNode, e.point)) : this.rightAction === "canvas-pan" && Math.hypot(n, r) >= 1 && (this.actionMoved = !0, this.port.panBy(n, r), this.actionPoint = { ...e.point }), this.port.setCursor(e.point, this.rightAction);
	}
	releaseRightAction(e, t) {
		this.rightAction === "node-captured" && this.capturedNode >= 0 && (this.port.endNodeDrag(this.capturedNode, this.actionMoved), this.port.setNodeDragging(!1), e && this.port.clearSelection()), this.rightAction = "idle", this.capturedNode = -1, this.actionPoint = null, this.actionMoved = !1, this.pinchFrames = 0, this.releaseFrames = 0, this.cooldownUntil = t + 160;
	}
	clearRightPointer() {
		this.hoveredNode = -1, this.hoverSince = 0, this.candidateNode = -1, this.port.hover(-1), this.port.setCursor(null, "idle");
	}
}, ru = class {
	stage;
	layer;
	video;
	cursor;
	renderer;
	callbacks;
	stateMachine;
	landmarker = null;
	stream = null;
	enabled = !1;
	suspended = !1;
	lastInferenceAt = 0;
	lastVideoTime = -1;
	previousRightPinch = !1;
	smoothedRightPoint = null;
	dragAnchors = /* @__PURE__ */ new Map();
	constructor(e, t, n, r, i, a) {
		this.stage = e, this.layer = t, this.video = n, this.cursor = r, this.renderer = i, this.callbacks = a, this.stateMachine = new nu(this.createInputPort());
	}
	async start() {
		if (this.enabled) return !0;
		if (!navigator.mediaDevices?.getUserMedia) return this.layer.dataset.state = "unavailable", !1;
		try {
			return this.stream = await navigator.mediaDevices.getUserMedia({
				audio: !1,
				video: {
					facingMode: "user",
					width: { ideal: 640 },
					height: { ideal: 480 },
					frameRate: {
						ideal: 30,
						max: 30
					}
				}
			}), this.landmarker = await tu(), this.video.srcObject = this.stream, await this.video.play(), this.enabled = !0, this.layer.hidden = !1, this.layer.dataset.state = "active", !0;
		} catch (e) {
			return console.warn("Gesture camera unavailable:", e), this.stop(), this.layer.dataset.state = "unavailable", !1;
		}
	}
	tick(e) {
		if (!(!this.enabled || this.suspended || !this.landmarker || this.video.readyState < 2) && !(e - this.lastInferenceAt < 1e3 / 24 || this.video.currentTime === this.lastVideoTime)) {
			this.lastInferenceAt = e, this.lastVideoTime = this.video.currentTime;
			try {
				let t = this.landmarker.detectForVideo(this.video, e);
				this.processLandmarks(t.landmarks, t.handedness);
			} catch (e) {
				console.warn("HandLandmarker frame failed:", e);
			}
		}
	}
	setSuspended(e) {
		this.suspended = e, this.stream?.getVideoTracks().forEach((t) => {
			t.enabled = !e;
		}), e && this.stateMachine.reset();
	}
	stop() {
		this.enabled = !1, this.stateMachine.reset(), this.stream?.getTracks().forEach((e) => e.stop()), this.stream = null, this.video.pause(), this.video.srcObject = null, this.landmarker?.close(), this.landmarker = null, this.layer.hidden = !0, this.layer.dataset.state = "idle", this.cursor.style.opacity = "0", this.dragAnchors.clear();
	}
	dispose() {
		this.stop();
	}
	getState() {
		return {
			enabled: this.enabled,
			suspended: this.suspended,
			layerState: this.layer.dataset.state || "idle"
		};
	}
	processLandmarks(e, t) {
		let n = this.stage.clientWidth, r = this.stage.clientHeight, i = e.map((e) => 1 - (e[8]?.x ?? .5)), a = e.map((e, n) => le(t[n]?.[0]?.categoryName, i[n]));
		e.length >= 2 && new Set(a).size < 2 && (a = i.map((e) => le(void 0, e)));
		let o = e.flatMap((e, t) => {
			let i = a[t], o = de(e, i === "right" && this.previousRightPinch);
			return o ? [fe(o, n, r, i)] : [];
		}), s = o.find((e) => e.handedness === "right");
		s ? (this.smoothedRightPoint = pe(this.smoothedRightPoint, s.point), s.point = this.smoothedRightPoint, this.previousRightPinch = s.pinch) : (this.previousRightPinch = !1, this.smoothedRightPoint = null), this.stateMachine.process(o, performance.now());
	}
	createInputPort() {
		return {
			pick: (e) => this.renderer.pick(e.x, e.y),
			hover: (e) => {
				this.renderer.setHovered(e), this.callbacks.onHover(e);
			},
			select: (e) => this.callbacks.onSelect(e),
			clearSelection: () => this.callbacks.onClearSelection(),
			fitToGraph: () => this.renderer.fitToGraph(),
			panBy: (e, t) => this.renderer.panBy(e, t),
			zoomAt: (e, t) => this.renderer.zoomAt(e.x, e.y, t),
			beginNodeDrag: (e) => {
				this.dragAnchors.set(e, this.renderer.getNodeWorldPosition(e)), this.callbacks.onNodeAttractionStart(e);
			},
			moveNode: (e, t) => {
				let n = this.dragAnchors.get(e) ?? this.renderer.getNodeWorldPosition(e), r = this.renderer.screenToWorld(t.x, t.y, n, {
					x: 0,
					y: 0,
					z: 1
				});
				this.renderer.setNodeTransientPosition(e, r.x, r.y);
			},
			endNodeDrag: (e, t) => {
				this.dragAnchors.delete(e), this.callbacks.onNodeAttractionEnd();
			},
			setCursor: (e, t) => {
				if (!e) {
					this.cursor.style.opacity = "0";
					return;
				}
				this.cursor.style.opacity = "1", this.cursor.style.left = `${e.x}px`, this.cursor.style.top = `${e.y}px`, this.cursor.dataset.mode = t;
			},
			setGestureActive: (e) => this.callbacks.onGestureActiveChange(e),
			setNodeDragging: (e) => this.callbacks.onNodeDraggingChange(e)
		};
	}
}, iu = class {
	renderer;
	canvas;
	callbacks;
	pointers = /* @__PURE__ */ new Map();
	activePointer = -1;
	downNode = -1;
	downEdge = -1;
	draggingNode = -1;
	attractionNode = -1;
	moved = !1;
	lastX = 0;
	lastY = 0;
	pinchDistance = 0;
	pinchCenter = null;
	hoverFrame = 0;
	hoverPoint = null;
	dragPlaneAnchor = null;
	constructor(e, t) {
		this.renderer = e, this.canvas = e.canvas, this.callbacks = t, this.canvas.addEventListener("wheel", this.onWheel, { passive: !1 }), this.canvas.addEventListener("pointerdown", this.onPointerDown), this.canvas.addEventListener("pointermove", this.onPointerMove), this.canvas.addEventListener("pointerup", this.onPointerUp), this.canvas.addEventListener("pointercancel", this.onPointerUp), this.canvas.addEventListener("pointerenter", this.onPointerEnter), this.canvas.addEventListener("pointerleave", this.onPointerLeave), this.canvas.addEventListener("contextmenu", (e) => e.preventDefault()), window.addEventListener("keydown", this.onKeyDown);
	}
	localPoint(e) {
		let t = this.canvas.getBoundingClientRect();
		return {
			x: e.clientX - t.left,
			y: e.clientY - t.top
		};
	}
	onWheel = (e) => {
		e.preventDefault(), this.callbacks.onWheel();
		let t = this.localPoint(e), n = Math.exp(-e.deltaY * .0014);
		this.renderer.zoomAt(t.x, t.y, n);
	};
	onPointerDown = (e) => {
		let t = this.localPoint(e);
		if (this.pointers.set(e.pointerId, t), this.callbacks.onPointerInsideChange(!0), this.callbacks.onPointerDownChange(!0), this.canvas.setPointerCapture(e.pointerId), this.pointers.size === 2) {
			this.attractionNode >= 0 && this.callbacks.onNodeAttractionEnd(), this.attractionNode = -1, this.activePointer = -1, this.draggingNode = -1, this.dragPlaneAnchor = null, this.downNode = -1, this.downEdge = -1, this.callbacks.onNodeDraggingChange(!1), this.callbacks.onOrbitInteractionChange(!0), this.updatePinchBaseline();
			return;
		}
		this.activePointer = e.pointerId, this.lastX = t.x, this.lastY = t.y, this.moved = !1, this.downNode = this.renderer.pick(t.x, t.y), this.downEdge = this.downNode >= 0 ? -1 : this.renderer.pickEdge(t.x, t.y), this.downNode >= 0 && (this.attractionNode = this.downNode, this.callbacks.onNodeAttractionStart(this.downNode));
	};
	updatePinchBaseline() {
		let [e, t] = [...this.pointers.values()];
		!e || !t || (this.pinchDistance = Math.hypot(e.x - t.x, e.y - t.y), this.pinchCenter = {
			x: (e.x + t.x) / 2,
			y: (e.y + t.y) / 2
		});
	}
	onPointerMove = (e) => {
		let t = this.localPoint(e);
		if (this.pointers.has(e.pointerId) && this.pointers.set(e.pointerId, t), this.pointers.size >= 2) {
			let [e, t] = [...this.pointers.values()], n = Math.max(1, Math.hypot(e.x - t.x, e.y - t.y)), r = {
				x: (e.x + t.x) / 2,
				y: (e.y + t.y) / 2
			};
			this.pinchDistance > 0 && this.renderer.zoomAt(r.x, r.y, n / this.pinchDistance), this.pinchCenter && this.renderer.panBy(r.x - this.pinchCenter.x, r.y - this.pinchCenter.y), this.pinchDistance = n, this.pinchCenter = r;
			return;
		}
		if (this.activePointer === e.pointerId) {
			let e = t.x - this.lastX, n = t.y - this.lastY;
			if (!this.moved && Math.hypot(e, n) > 4 && (this.moved = !0), this.moved && this.downNode >= 0 && this.draggingNode < 0 && (this.draggingNode = this.downNode, this.dragPlaneAnchor = this.renderer.getNodeWorldPosition(this.draggingNode), this.callbacks.onNodeDraggingChange(!0)), this.draggingNode >= 0) {
				let e = this.renderer.screenToWorld(t.x, t.y, this.dragPlaneAnchor ?? void 0, {
					x: 0,
					y: 0,
					z: 1
				});
				this.renderer.setNodeTransientPosition(this.draggingNode, e.x, e.y);
			} else this.moved && (this.callbacks.onOrbitInteractionChange(!0), this.renderer.panBy(e, n));
			this.lastX = t.x, this.lastY = t.y;
			return;
		}
		e.pointerType !== "touch" && this.scheduleHover(t);
	};
	scheduleHover(e) {
		this.hoverPoint = e, !this.hoverFrame && (this.hoverFrame = requestAnimationFrame(() => {
			if (this.hoverFrame = 0, !this.hoverPoint) return;
			let e = this.renderer.pick(this.hoverPoint.x, this.hoverPoint.y);
			this.renderer.setHovered(e), this.callbacks.onHover(e), this.canvas.style.cursor = e >= 0 ? "pointer" : "grab";
		}));
	}
	onPointerUp = (e) => {
		let t = this.localPoint(e), n = this.activePointer === e.pointerId;
		if (this.pointers.delete(e.pointerId), this.callbacks.onPointerDownChange(this.pointers.size > 0), this.pointers.size < 2 && (this.pinchDistance = 0, this.pinchCenter = null), !n) {
			if (this.pointers.size === 1) {
				let [e, t] = [...this.pointers.entries()][0];
				this.activePointer = e, this.lastX = t.x, this.lastY = t.y;
			}
			return;
		}
		if (this.draggingNode < 0 && !this.moved) {
			let e = this.downNode >= 0 ? this.downNode : this.renderer.pick(t.x, t.y);
			if (e >= 0) this.callbacks.onSelect(e);
			else {
				let e = this.downEdge >= 0 ? this.downEdge : this.renderer.pickEdge(t.x, t.y);
				e >= 0 ? this.callbacks.onSelectEdge(e) : this.callbacks.onSelect(-1);
			}
		}
		this.attractionNode >= 0 && this.callbacks.onNodeAttractionEnd(), this.activePointer = -1, this.downNode = -1, this.downEdge = -1, this.draggingNode = -1, this.attractionNode = -1, this.dragPlaneAnchor = null, this.moved = !1, this.callbacks.onNodeDraggingChange(!1), this.callbacks.onOrbitInteractionChange(!1);
	};
	onPointerEnter = () => this.callbacks.onPointerInsideChange(!0);
	onPointerLeave = () => this.callbacks.onPointerInsideChange(!1);
	onKeyDown = (e) => {
		if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.target instanceof HTMLTextAreaElement) {
			e.key === "Escape" && this.callbacks.onEscape();
			return;
		}
		let t = this.canvas.clientWidth / 2, n = this.canvas.clientHeight / 2;
		if (this.callbacks.onKeyboardInteraction(), e.key === "+" || e.key === "=") this.renderer.zoomAt(t, n, 1.25);
		else if (e.key === "-" || e.key === "_") this.renderer.zoomAt(t, n, .8);
		else if (e.key === "0") this.renderer.fitToGraph();
		else if (e.key === "Escape") this.callbacks.onEscape();
		else if (e.key === "ArrowLeft") this.renderer.panBy(48, 0);
		else if (e.key === "ArrowRight") this.renderer.panBy(-48, 0);
		else if (e.key === "ArrowUp") this.renderer.panBy(0, 48);
		else if (e.key === "ArrowDown") this.renderer.panBy(0, -48);
		else return;
		e.preventDefault();
	};
	dispose() {
		this.attractionNode >= 0 && this.callbacks.onNodeAttractionEnd(), this.canvas.removeEventListener("wheel", this.onWheel), this.canvas.removeEventListener("pointerdown", this.onPointerDown), this.canvas.removeEventListener("pointermove", this.onPointerMove), this.canvas.removeEventListener("pointerup", this.onPointerUp), this.canvas.removeEventListener("pointercancel", this.onPointerUp), this.canvas.removeEventListener("pointerenter", this.onPointerEnter), this.canvas.removeEventListener("pointerleave", this.onPointerLeave), window.removeEventListener("keydown", this.onKeyDown), cancelAnimationFrame(this.hoverFrame);
	}
}, au = /* @__PURE__ */ new Map(), ou = "docthinker-promo-layouts", su = "layouts";
function cu() {
	return new Promise((e, t) => {
		let n = indexedDB.open(ou, 1);
		n.onupgradeneeded = () => {
			n.result.objectStoreNames.contains(su) || n.result.createObjectStore(su, { keyPath: "key" });
		}, n.onsuccess = () => e(n.result), n.onerror = () => t(n.error);
	});
}
var lu = class {
	async get(e, t) {
		let n = au.get(e);
		if (n?.length === t * 3) return n.slice();
		if (typeof indexedDB > "u") return null;
		try {
			let n = await cu(), r = await new Promise((t, r) => {
				let i = n.transaction(su, "readonly").objectStore(su).get(e);
				i.onsuccess = () => t(i.result), i.onerror = () => r(i.error);
			});
			if (n.close(), !r || r.nodeCount !== t || r.positions.byteLength !== t * 3 * 4) return null;
			let i = new Float32Array(r.positions.slice(0));
			return au.set(e, i.slice()), i;
		} catch {
			return null;
		}
	}
	async set(e, t) {
		if (au.set(e, t.slice()), !(typeof indexedDB > "u")) try {
			let n = await cu();
			await new Promise((r, i) => {
				let a = n.transaction(su, "readwrite");
				a.objectStore(su).put({
					key: e,
					nodeCount: t.length / 3,
					positions: t.slice().buffer,
					savedAt: Date.now()
				}), a.oncomplete = () => r(), a.onerror = () => i(a.error);
			}), n.close();
		} catch {}
	}
}, uu = class {
	worker = null;
	requestId = 0;
	pending = /* @__PURE__ */ new Map();
	cache = new lu();
	createWorker() {
		this.disposeWorker();
		let e = new Worker(new URL(
			/* @vite-ignore */
			"" + new URL("assets/layout.worker-BDkPaKkA.js", import.meta.url).href,
			"" + import.meta.url
		), { type: "module" });
		return e.onmessage = (e) => {
			let t = this.pending.get(e.data.requestId);
			t && (this.pending.delete(e.data.requestId), e.data.type === "focus" ? t.resolve({
				indices: e.data.indices,
				positions: e.data.positions
			}) : t.resolve(e.data.positions));
		}, e.onerror = (e) => {
			let t = Error(e.message || "Graph layout worker failed");
			this.pending.forEach((e) => e.reject(t)), this.pending.clear();
		}, this.worker = e, e;
	}
	graphMessage(e) {
		return {
			nodeCount: e.nodes.length,
			hashes: e.hashes.slice(),
			degrees: e.degrees.slice(),
			groups: e.groups.slice(),
			edgePairs: e.edgePairs.slice()
		};
	}
	async layout(e) {
		let t = this.createWorker(), n = `v2:${e.fingerprint}`, r = await this.cache.get(n, e.nodes.length);
		if (r) {
			let n = this.graphMessage(e), i = r.slice();
			return t.postMessage({
				type: "hydrate",
				requestId: 0,
				...n,
				positions: i
			}, [
				n.hashes.buffer,
				n.degrees.buffer,
				n.groups.buffer,
				n.edgePairs.buffer,
				i.buffer
			]), {
				positions: r,
				cached: !0
			};
		}
		let i = ++this.requestId, a = this.graphMessage(e), o = new Promise((e, t) => this.pending.set(i, {
			resolve: (t) => e(t),
			reject: t
		}));
		t.postMessage({
			type: "layout",
			requestId: i,
			...a
		}, [
			a.hashes.buffer,
			a.degrees.buffer,
			a.groups.buffer,
			a.edgePairs.buffer
		]);
		let s = await o;
		return this.cache.set(n, s), {
			positions: s,
			cached: !1
		};
	}
	async focus(e) {
		if (!this.worker) return {
			indices: /* @__PURE__ */ new Uint32Array(),
			positions: /* @__PURE__ */ new Float32Array()
		};
		let t = ++this.requestId, n = new Promise((e, n) => this.pending.set(t, {
			resolve: (t) => e(t),
			reject: n
		}));
		return this.worker.postMessage({
			type: "focus",
			requestId: t,
			nodeIndex: e
		}), await n;
	}
	disposeWorker() {
		this.worker?.terminate(), this.worker = null;
		let e = /* @__PURE__ */ Error("Layout request cancelled");
		this.pending.forEach((t) => t.reject(e)), this.pending.clear();
	}
	dispose() {
		this.disposeWorker();
	}
}, du = 1e3, fu = 1001, pu = 1002, mu = 1003, hu = 1004, gu = 1005, _u = 1006, vu = 1007, yu = 1008, bu = 1009, xu = 1010, Su = 1011, Cu = 1012, wu = 1013, Tu = 1014, Eu = 1015, Du = 1016, Ou = 1017, ku = 1018, Au = 1020, ju = 35902, Mu = 35899, Nu = 1021, Pu = 1022, Fu = 1023, Iu = 1026, Lu = 1027, Ru = 1028, zu = 1029, Bu = 1030, Vu = 1031, Hu = 1033, Uu = 33776, Wu = 33777, Gu = 33778, Ku = 33779, qu = 35840, Ju = 35841, Yu = 35842, Xu = 35843, Zu = 36196, Qu = 37492, $u = 37496, ed = 37488, td = 37489, nd = 37490, rd = 37491, id = 37808, ad = 37809, od = 37810, sd = 37811, cd = 37812, ld = 37813, ud = 37814, dd = 37815, fd = 37816, pd = 37817, md = 37818, hd = 37819, gd = 37820, _d = 37821, vd = 36492, yd = 36494, bd = 36495, xd = 36283, Sd = 36284, Cd = 36285, wd = 36286, Td = 2300, Ed = 2301, Dd = 2302, Od = 2303, kd = 2400, Ad = 2401, jd = 2402, Md = 3200, Nd = "srgb", Pd = "srgb-linear", Fd = "linear", Id = "srgb", Ld = 7680, Rd = 35044, zd = 35048, Bd = 2e3;
function Vd(e) {
	for (let t = e.length - 1; t >= 0; --t) if (e[t] >= 65535) return !0;
	return !1;
}
function Hd(e) {
	return ArrayBuffer.isView(e) && !(e instanceof DataView);
}
function Ud(e) {
	return document.createElementNS("http://www.w3.org/1999/xhtml", e);
}
function Wd() {
	let e = Ud("canvas");
	return e.style.display = "block", e;
}
var Gd = {};
function Kd(...e) {
	let t = "THREE." + e.shift();
	console.log(t, ...e);
}
function qd(e) {
	let t = e[0];
	if (typeof t == "string" && t.startsWith("TSL:")) {
		let t = e[1];
		t && t.isStackTrace ? e[0] += " " + t.getLocation() : e[1] = "Stack trace not available. Enable \"THREE.Node.captureStackTrace\" to capture stack traces.";
	}
	return e;
}
function G(...e) {
	e = qd(e);
	let t = "THREE." + e.shift();
	{
		let n = e[0];
		n && n.isStackTrace ? console.warn(n.getError(t)) : console.warn(t, ...e);
	}
}
function K(...e) {
	e = qd(e);
	let t = "THREE." + e.shift();
	{
		let n = e[0];
		n && n.isStackTrace ? console.error(n.getError(t)) : console.error(t, ...e);
	}
}
function Jd(...e) {
	let t = e.join(" ");
	t in Gd || (Gd[t] = !0, G(...e));
}
function Yd(e, t, n) {
	return new Promise(function(r, i) {
		function a() {
			switch (e.clientWaitSync(t, e.SYNC_FLUSH_COMMANDS_BIT, 0)) {
				case e.WAIT_FAILED:
					i();
					break;
				case e.TIMEOUT_EXPIRED:
					setTimeout(a, n);
					break;
				default: r();
			}
		}
		setTimeout(a, n);
	});
}
var Xd = {
	0: 1,
	2: 6,
	4: 7,
	3: 5,
	1: 0,
	6: 2,
	7: 4,
	5: 3
}, Zd = class {
	addEventListener(e, t) {
		this._listeners === void 0 && (this._listeners = {});
		let n = this._listeners;
		n[e] === void 0 && (n[e] = []), n[e].indexOf(t) === -1 && n[e].push(t);
	}
	hasEventListener(e, t) {
		let n = this._listeners;
		return n !== void 0 && n[e] !== void 0 && n[e].indexOf(t) !== -1;
	}
	removeEventListener(e, t) {
		let n = this._listeners;
		if (n === void 0) return;
		let r = n[e];
		if (r !== void 0) {
			let e = r.indexOf(t);
			e !== -1 && r.splice(e, 1);
		}
	}
	dispatchEvent(e) {
		let t = this._listeners;
		if (t === void 0) return;
		let n = t[e.type];
		if (n !== void 0) {
			e.target = this;
			let t = n.slice(0);
			for (let n = 0, r = t.length; n < r; n++) t[n].call(this, e);
			e.target = null;
		}
	}
}, Qd = /* @__PURE__ */ "00.01.02.03.04.05.06.07.08.09.0a.0b.0c.0d.0e.0f.10.11.12.13.14.15.16.17.18.19.1a.1b.1c.1d.1e.1f.20.21.22.23.24.25.26.27.28.29.2a.2b.2c.2d.2e.2f.30.31.32.33.34.35.36.37.38.39.3a.3b.3c.3d.3e.3f.40.41.42.43.44.45.46.47.48.49.4a.4b.4c.4d.4e.4f.50.51.52.53.54.55.56.57.58.59.5a.5b.5c.5d.5e.5f.60.61.62.63.64.65.66.67.68.69.6a.6b.6c.6d.6e.6f.70.71.72.73.74.75.76.77.78.79.7a.7b.7c.7d.7e.7f.80.81.82.83.84.85.86.87.88.89.8a.8b.8c.8d.8e.8f.90.91.92.93.94.95.96.97.98.99.9a.9b.9c.9d.9e.9f.a0.a1.a2.a3.a4.a5.a6.a7.a8.a9.aa.ab.ac.ad.ae.af.b0.b1.b2.b3.b4.b5.b6.b7.b8.b9.ba.bb.bc.bd.be.bf.c0.c1.c2.c3.c4.c5.c6.c7.c8.c9.ca.cb.cc.cd.ce.cf.d0.d1.d2.d3.d4.d5.d6.d7.d8.d9.da.db.dc.dd.de.df.e0.e1.e2.e3.e4.e5.e6.e7.e8.e9.ea.eb.ec.ed.ee.ef.f0.f1.f2.f3.f4.f5.f6.f7.f8.f9.fa.fb.fc.fd.fe.ff".split("."), $d = 1234567, ef = Math.PI / 180, tf = 180 / Math.PI;
function nf() {
	let e = Math.random() * 4294967295 | 0, t = Math.random() * 4294967295 | 0, n = Math.random() * 4294967295 | 0, r = Math.random() * 4294967295 | 0;
	return (Qd[e & 255] + Qd[e >> 8 & 255] + Qd[e >> 16 & 255] + Qd[e >> 24 & 255] + "-" + Qd[t & 255] + Qd[t >> 8 & 255] + "-" + Qd[t >> 16 & 15 | 64] + Qd[t >> 24 & 255] + "-" + Qd[n & 63 | 128] + Qd[n >> 8 & 255] + "-" + Qd[n >> 16 & 255] + Qd[n >> 24 & 255] + Qd[r & 255] + Qd[r >> 8 & 255] + Qd[r >> 16 & 255] + Qd[r >> 24 & 255]).toLowerCase();
}
function q(e, t, n) {
	return Math.max(t, Math.min(n, e));
}
function rf(e, t) {
	return (e % t + t) % t;
}
function af(e, t, n, r, i) {
	return r + (e - t) * (i - r) / (n - t);
}
function of(e, t, n) {
	return e === t ? 0 : (n - e) / (t - e);
}
function sf(e, t, n) {
	return (1 - n) * e + n * t;
}
function cf(e, t, n, r) {
	return sf(e, t, 1 - Math.exp(-n * r));
}
function lf(e, t = 1) {
	return t - Math.abs(rf(e, t * 2) - t);
}
function uf(e, t, n) {
	return e <= t ? 0 : e >= n ? 1 : (e = (e - t) / (n - t), e * e * (3 - 2 * e));
}
function df(e, t, n) {
	return e <= t ? 0 : e >= n ? 1 : (e = (e - t) / (n - t), e * e * e * (e * (e * 6 - 15) + 10));
}
function ff(e, t) {
	return e + Math.floor(Math.random() * (t - e + 1));
}
function pf(e, t) {
	return e + Math.random() * (t - e);
}
function mf(e) {
	return e * (.5 - Math.random());
}
function hf(e) {
	e !== void 0 && ($d = e);
	let t = $d += 1831565813;
	return t = Math.imul(t ^ t >>> 15, t | 1), t ^= t + Math.imul(t ^ t >>> 7, t | 61), ((t ^ t >>> 14) >>> 0) / 4294967296;
}
function gf(e) {
	return e * ef;
}
function _f(e) {
	return e * tf;
}
function vf(e) {
	return (e & e - 1) == 0 && e !== 0;
}
function yf(e) {
	return 2 ** Math.ceil(Math.log(e) / Math.LN2);
}
function bf(e) {
	return 2 ** Math.floor(Math.log(e) / Math.LN2);
}
function xf(e, t, n, r, i) {
	let a = Math.cos, o = Math.sin, s = a(n / 2), c = o(n / 2), l = a((t + r) / 2), u = o((t + r) / 2), d = a((t - r) / 2), f = o((t - r) / 2), p = a((r - t) / 2), m = o((r - t) / 2);
	switch (i) {
		case "XYX":
			e.set(s * u, c * d, c * f, s * l);
			break;
		case "YZY":
			e.set(c * f, s * u, c * d, s * l);
			break;
		case "ZXZ":
			e.set(c * d, c * f, s * u, s * l);
			break;
		case "XZX":
			e.set(s * u, c * m, c * p, s * l);
			break;
		case "YXY":
			e.set(c * p, s * u, c * m, s * l);
			break;
		case "ZYZ":
			e.set(c * m, c * p, s * u, s * l);
			break;
		default: G("MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: " + i);
	}
}
function Sf(e, t) {
	switch (t.constructor) {
		case Float32Array: return e;
		case Uint32Array: return e / 4294967295;
		case Uint16Array: return e / 65535;
		case Uint8Array: return e / 255;
		case Int32Array: return Math.max(e / 2147483647, -1);
		case Int16Array: return Math.max(e / 32767, -1);
		case Int8Array: return Math.max(e / 127, -1);
		default: throw Error("THREE.MathUtils: Invalid component type.");
	}
}
function Cf(e, t) {
	switch (t.constructor) {
		case Float32Array: return e;
		case Uint32Array: return Math.round(e * 4294967295);
		case Uint16Array: return Math.round(e * 65535);
		case Uint8Array: return Math.round(e * 255);
		case Int32Array: return Math.round(e * 2147483647);
		case Int16Array: return Math.round(e * 32767);
		case Int8Array: return Math.round(e * 127);
		default: throw Error("THREE.MathUtils: Invalid component type.");
	}
}
var wf = {
	DEG2RAD: ef,
	RAD2DEG: tf,
	generateUUID: nf,
	clamp: q,
	euclideanModulo: rf,
	mapLinear: af,
	inverseLerp: of,
	lerp: sf,
	damp: cf,
	pingpong: lf,
	smoothstep: uf,
	smootherstep: df,
	randInt: ff,
	randFloat: pf,
	randFloatSpread: mf,
	seededRandom: hf,
	degToRad: gf,
	radToDeg: _f,
	isPowerOfTwo: vf,
	ceilPowerOfTwo: yf,
	floorPowerOfTwo: bf,
	setQuaternionFromProperEuler: xf,
	normalize: Cf,
	denormalize: Sf
}, Tf = class e {
	static {
		e.prototype.isVector2 = !0;
	}
	constructor(e = 0, t = 0) {
		this.x = e, this.y = t;
	}
	get width() {
		return this.x;
	}
	set width(e) {
		this.x = e;
	}
	get height() {
		return this.y;
	}
	set height(e) {
		this.y = e;
	}
	set(e, t) {
		return this.x = e, this.y = t, this;
	}
	setScalar(e) {
		return this.x = e, this.y = e, this;
	}
	setX(e) {
		return this.x = e, this;
	}
	setY(e) {
		return this.y = e, this;
	}
	setComponent(e, t) {
		switch (e) {
			case 0:
				this.x = t;
				break;
			case 1:
				this.y = t;
				break;
			default: throw Error("THREE.Vector2: index is out of range: " + e);
		}
		return this;
	}
	getComponent(e) {
		switch (e) {
			case 0: return this.x;
			case 1: return this.y;
			default: throw Error("THREE.Vector2: index is out of range: " + e);
		}
	}
	clone() {
		return new this.constructor(this.x, this.y);
	}
	copy(e) {
		return this.x = e.x, this.y = e.y, this;
	}
	add(e) {
		return this.x += e.x, this.y += e.y, this;
	}
	addScalar(e) {
		return this.x += e, this.y += e, this;
	}
	addVectors(e, t) {
		return this.x = e.x + t.x, this.y = e.y + t.y, this;
	}
	addScaledVector(e, t) {
		return this.x += e.x * t, this.y += e.y * t, this;
	}
	sub(e) {
		return this.x -= e.x, this.y -= e.y, this;
	}
	subScalar(e) {
		return this.x -= e, this.y -= e, this;
	}
	subVectors(e, t) {
		return this.x = e.x - t.x, this.y = e.y - t.y, this;
	}
	multiply(e) {
		return this.x *= e.x, this.y *= e.y, this;
	}
	multiplyScalar(e) {
		return this.x *= e, this.y *= e, this;
	}
	divide(e) {
		return this.x /= e.x, this.y /= e.y, this;
	}
	divideScalar(e) {
		return this.multiplyScalar(1 / e);
	}
	applyMatrix3(e) {
		let t = this.x, n = this.y, r = e.elements;
		return this.x = r[0] * t + r[3] * n + r[6], this.y = r[1] * t + r[4] * n + r[7], this;
	}
	min(e) {
		return this.x = Math.min(this.x, e.x), this.y = Math.min(this.y, e.y), this;
	}
	max(e) {
		return this.x = Math.max(this.x, e.x), this.y = Math.max(this.y, e.y), this;
	}
	clamp(e, t) {
		return this.x = q(this.x, e.x, t.x), this.y = q(this.y, e.y, t.y), this;
	}
	clampScalar(e, t) {
		return this.x = q(this.x, e, t), this.y = q(this.y, e, t), this;
	}
	clampLength(e, t) {
		let n = this.length();
		return this.divideScalar(n || 1).multiplyScalar(q(n, e, t));
	}
	floor() {
		return this.x = Math.floor(this.x), this.y = Math.floor(this.y), this;
	}
	ceil() {
		return this.x = Math.ceil(this.x), this.y = Math.ceil(this.y), this;
	}
	round() {
		return this.x = Math.round(this.x), this.y = Math.round(this.y), this;
	}
	roundToZero() {
		return this.x = Math.trunc(this.x), this.y = Math.trunc(this.y), this;
	}
	negate() {
		return this.x = -this.x, this.y = -this.y, this;
	}
	dot(e) {
		return this.x * e.x + this.y * e.y;
	}
	cross(e) {
		return this.x * e.y - this.y * e.x;
	}
	lengthSq() {
		return this.x * this.x + this.y * this.y;
	}
	length() {
		return Math.sqrt(this.x * this.x + this.y * this.y);
	}
	manhattanLength() {
		return Math.abs(this.x) + Math.abs(this.y);
	}
	normalize() {
		return this.divideScalar(this.length() || 1);
	}
	angle() {
		return Math.atan2(-this.y, -this.x) + Math.PI;
	}
	angleTo(e) {
		let t = Math.sqrt(this.lengthSq() * e.lengthSq());
		if (t === 0) return Math.PI / 2;
		let n = this.dot(e) / t;
		return Math.acos(q(n, -1, 1));
	}
	distanceTo(e) {
		return Math.sqrt(this.distanceToSquared(e));
	}
	distanceToSquared(e) {
		let t = this.x - e.x, n = this.y - e.y;
		return t * t + n * n;
	}
	manhattanDistanceTo(e) {
		return Math.abs(this.x - e.x) + Math.abs(this.y - e.y);
	}
	setLength(e) {
		return this.normalize().multiplyScalar(e);
	}
	lerp(e, t) {
		return this.x += (e.x - this.x) * t, this.y += (e.y - this.y) * t, this;
	}
	lerpVectors(e, t, n) {
		return this.x = e.x + (t.x - e.x) * n, this.y = e.y + (t.y - e.y) * n, this;
	}
	equals(e) {
		return e.x === this.x && e.y === this.y;
	}
	fromArray(e, t = 0) {
		return this.x = e[t], this.y = e[t + 1], this;
	}
	toArray(e = [], t = 0) {
		return e[t] = this.x, e[t + 1] = this.y, e;
	}
	fromBufferAttribute(e, t) {
		return this.x = e.getX(t), this.y = e.getY(t), this;
	}
	rotateAround(e, t) {
		let n = Math.cos(t), r = Math.sin(t), i = this.x - e.x, a = this.y - e.y;
		return this.x = i * n - a * r + e.x, this.y = i * r + a * n + e.y, this;
	}
	random() {
		return this.x = Math.random(), this.y = Math.random(), this;
	}
	*[Symbol.iterator]() {
		yield this.x, yield this.y;
	}
}, Ef = class {
	constructor(e = 0, t = 0, n = 0, r = 1) {
		this.isQuaternion = !0, this._x = e, this._y = t, this._z = n, this._w = r;
	}
	static slerpFlat(e, t, n, r, i, a, o) {
		let s = n[r + 0], c = n[r + 1], l = n[r + 2], u = n[r + 3], d = i[a + 0], f = i[a + 1], p = i[a + 2], m = i[a + 3];
		if (u !== m || s !== d || c !== f || l !== p) {
			let e = s * d + c * f + l * p + u * m;
			e < 0 && (d = -d, f = -f, p = -p, m = -m, e = -e);
			let t = 1 - o;
			if (e < .9995) {
				let n = Math.acos(e), r = Math.sin(n);
				t = Math.sin(t * n) / r, o = Math.sin(o * n) / r, s = s * t + d * o, c = c * t + f * o, l = l * t + p * o, u = u * t + m * o;
			} else {
				s = s * t + d * o, c = c * t + f * o, l = l * t + p * o, u = u * t + m * o;
				let e = 1 / Math.sqrt(s * s + c * c + l * l + u * u);
				s *= e, c *= e, l *= e, u *= e;
			}
		}
		e[t] = s, e[t + 1] = c, e[t + 2] = l, e[t + 3] = u;
	}
	static multiplyQuaternionsFlat(e, t, n, r, i, a) {
		let o = n[r], s = n[r + 1], c = n[r + 2], l = n[r + 3], u = i[a], d = i[a + 1], f = i[a + 2], p = i[a + 3];
		return e[t] = o * p + l * u + s * f - c * d, e[t + 1] = s * p + l * d + c * u - o * f, e[t + 2] = c * p + l * f + o * d - s * u, e[t + 3] = l * p - o * u - s * d - c * f, e;
	}
	get x() {
		return this._x;
	}
	set x(e) {
		this._x = e, this._onChangeCallback();
	}
	get y() {
		return this._y;
	}
	set y(e) {
		this._y = e, this._onChangeCallback();
	}
	get z() {
		return this._z;
	}
	set z(e) {
		this._z = e, this._onChangeCallback();
	}
	get w() {
		return this._w;
	}
	set w(e) {
		this._w = e, this._onChangeCallback();
	}
	set(e, t, n, r) {
		return this._x = e, this._y = t, this._z = n, this._w = r, this._onChangeCallback(), this;
	}
	clone() {
		return new this.constructor(this._x, this._y, this._z, this._w);
	}
	copy(e) {
		return this._x = e.x, this._y = e.y, this._z = e.z, this._w = e.w, this._onChangeCallback(), this;
	}
	setFromEuler(e, t = !0) {
		let n = e._x, r = e._y, i = e._z, a = e._order, o = Math.cos, s = Math.sin, c = o(n / 2), l = o(r / 2), u = o(i / 2), d = s(n / 2), f = s(r / 2), p = s(i / 2);
		switch (a) {
			case "XYZ":
				this._x = d * l * u + c * f * p, this._y = c * f * u - d * l * p, this._z = c * l * p + d * f * u, this._w = c * l * u - d * f * p;
				break;
			case "YXZ":
				this._x = d * l * u + c * f * p, this._y = c * f * u - d * l * p, this._z = c * l * p - d * f * u, this._w = c * l * u + d * f * p;
				break;
			case "ZXY":
				this._x = d * l * u - c * f * p, this._y = c * f * u + d * l * p, this._z = c * l * p + d * f * u, this._w = c * l * u - d * f * p;
				break;
			case "ZYX":
				this._x = d * l * u - c * f * p, this._y = c * f * u + d * l * p, this._z = c * l * p - d * f * u, this._w = c * l * u + d * f * p;
				break;
			case "YZX":
				this._x = d * l * u + c * f * p, this._y = c * f * u + d * l * p, this._z = c * l * p - d * f * u, this._w = c * l * u - d * f * p;
				break;
			case "XZY":
				this._x = d * l * u - c * f * p, this._y = c * f * u - d * l * p, this._z = c * l * p + d * f * u, this._w = c * l * u + d * f * p;
				break;
			default: G("Quaternion: .setFromEuler() encountered an unknown order: " + a);
		}
		return t === !0 && this._onChangeCallback(), this;
	}
	setFromAxisAngle(e, t) {
		let n = t / 2, r = Math.sin(n);
		return this._x = e.x * r, this._y = e.y * r, this._z = e.z * r, this._w = Math.cos(n), this._onChangeCallback(), this;
	}
	setFromRotationMatrix(e) {
		let t = e.elements, n = t[0], r = t[4], i = t[8], a = t[1], o = t[5], s = t[9], c = t[2], l = t[6], u = t[10], d = n + o + u;
		if (d > 0) {
			let e = .5 / Math.sqrt(d + 1);
			this._w = .25 / e, this._x = (l - s) * e, this._y = (i - c) * e, this._z = (a - r) * e;
		} else if (n > o && n > u) {
			let e = 2 * Math.sqrt(1 + n - o - u);
			this._w = (l - s) / e, this._x = .25 * e, this._y = (r + a) / e, this._z = (i + c) / e;
		} else if (o > u) {
			let e = 2 * Math.sqrt(1 + o - n - u);
			this._w = (i - c) / e, this._x = (r + a) / e, this._y = .25 * e, this._z = (s + l) / e;
		} else {
			let e = 2 * Math.sqrt(1 + u - n - o);
			this._w = (a - r) / e, this._x = (i + c) / e, this._y = (s + l) / e, this._z = .25 * e;
		}
		return this._onChangeCallback(), this;
	}
	setFromUnitVectors(e, t) {
		let n = e.dot(t) + 1;
		return n < 1e-8 ? (n = 0, Math.abs(e.x) > Math.abs(e.z) ? (this._x = -e.y, this._y = e.x, this._z = 0, this._w = n) : (this._x = 0, this._y = -e.z, this._z = e.y, this._w = n)) : (this._x = e.y * t.z - e.z * t.y, this._y = e.z * t.x - e.x * t.z, this._z = e.x * t.y - e.y * t.x, this._w = n), this.normalize();
	}
	angleTo(e) {
		return 2 * Math.acos(Math.abs(q(this.dot(e), -1, 1)));
	}
	rotateTowards(e, t) {
		let n = this.angleTo(e);
		if (n === 0) return this;
		let r = Math.min(1, t / n);
		return this.slerp(e, r), this;
	}
	identity() {
		return this.set(0, 0, 0, 1);
	}
	invert() {
		return this.conjugate();
	}
	conjugate() {
		return this._x *= -1, this._y *= -1, this._z *= -1, this._onChangeCallback(), this;
	}
	dot(e) {
		return this._x * e._x + this._y * e._y + this._z * e._z + this._w * e._w;
	}
	lengthSq() {
		return this._x * this._x + this._y * this._y + this._z * this._z + this._w * this._w;
	}
	length() {
		return Math.sqrt(this._x * this._x + this._y * this._y + this._z * this._z + this._w * this._w);
	}
	normalize() {
		let e = this.length();
		return e === 0 ? (this._x = 0, this._y = 0, this._z = 0, this._w = 1) : (e = 1 / e, this._x *= e, this._y *= e, this._z *= e, this._w *= e), this._onChangeCallback(), this;
	}
	multiply(e) {
		return this.multiplyQuaternions(this, e);
	}
	premultiply(e) {
		return this.multiplyQuaternions(e, this);
	}
	multiplyQuaternions(e, t) {
		let n = e._x, r = e._y, i = e._z, a = e._w, o = t._x, s = t._y, c = t._z, l = t._w;
		return this._x = n * l + a * o + r * c - i * s, this._y = r * l + a * s + i * o - n * c, this._z = i * l + a * c + n * s - r * o, this._w = a * l - n * o - r * s - i * c, this._onChangeCallback(), this;
	}
	slerp(e, t) {
		let n = e._x, r = e._y, i = e._z, a = e._w, o = this.dot(e);
		o < 0 && (n = -n, r = -r, i = -i, a = -a, o = -o);
		let s = 1 - t;
		if (o < .9995) {
			let e = Math.acos(o), c = Math.sin(e);
			s = Math.sin(s * e) / c, t = Math.sin(t * e) / c, this._x = this._x * s + n * t, this._y = this._y * s + r * t, this._z = this._z * s + i * t, this._w = this._w * s + a * t, this._onChangeCallback();
		} else this._x = this._x * s + n * t, this._y = this._y * s + r * t, this._z = this._z * s + i * t, this._w = this._w * s + a * t, this.normalize();
		return this;
	}
	slerpQuaternions(e, t, n) {
		return this.copy(e).slerp(t, n);
	}
	random() {
		let e = 2 * Math.PI * Math.random(), t = 2 * Math.PI * Math.random(), n = Math.random(), r = Math.sqrt(1 - n), i = Math.sqrt(n);
		return this.set(r * Math.sin(e), r * Math.cos(e), i * Math.sin(t), i * Math.cos(t));
	}
	equals(e) {
		return e._x === this._x && e._y === this._y && e._z === this._z && e._w === this._w;
	}
	fromArray(e, t = 0) {
		return this._x = e[t], this._y = e[t + 1], this._z = e[t + 2], this._w = e[t + 3], this._onChangeCallback(), this;
	}
	toArray(e = [], t = 0) {
		return e[t] = this._x, e[t + 1] = this._y, e[t + 2] = this._z, e[t + 3] = this._w, e;
	}
	fromBufferAttribute(e, t) {
		return this._x = e.getX(t), this._y = e.getY(t), this._z = e.getZ(t), this._w = e.getW(t), this._onChangeCallback(), this;
	}
	toJSON() {
		return this.toArray();
	}
	_onChange(e) {
		return this._onChangeCallback = e, this;
	}
	_onChangeCallback() {}
	*[Symbol.iterator]() {
		yield this._x, yield this._y, yield this._z, yield this._w;
	}
}, J = class e {
	static {
		e.prototype.isVector3 = !0;
	}
	constructor(e = 0, t = 0, n = 0) {
		this.x = e, this.y = t, this.z = n;
	}
	set(e, t, n) {
		return n === void 0 && (n = this.z), this.x = e, this.y = t, this.z = n, this;
	}
	setScalar(e) {
		return this.x = e, this.y = e, this.z = e, this;
	}
	setX(e) {
		return this.x = e, this;
	}
	setY(e) {
		return this.y = e, this;
	}
	setZ(e) {
		return this.z = e, this;
	}
	setComponent(e, t) {
		switch (e) {
			case 0:
				this.x = t;
				break;
			case 1:
				this.y = t;
				break;
			case 2:
				this.z = t;
				break;
			default: throw Error("THREE.Vector3: index is out of range: " + e);
		}
		return this;
	}
	getComponent(e) {
		switch (e) {
			case 0: return this.x;
			case 1: return this.y;
			case 2: return this.z;
			default: throw Error("THREE.Vector3: index is out of range: " + e);
		}
	}
	clone() {
		return new this.constructor(this.x, this.y, this.z);
	}
	copy(e) {
		return this.x = e.x, this.y = e.y, this.z = e.z, this;
	}
	add(e) {
		return this.x += e.x, this.y += e.y, this.z += e.z, this;
	}
	addScalar(e) {
		return this.x += e, this.y += e, this.z += e, this;
	}
	addVectors(e, t) {
		return this.x = e.x + t.x, this.y = e.y + t.y, this.z = e.z + t.z, this;
	}
	addScaledVector(e, t) {
		return this.x += e.x * t, this.y += e.y * t, this.z += e.z * t, this;
	}
	sub(e) {
		return this.x -= e.x, this.y -= e.y, this.z -= e.z, this;
	}
	subScalar(e) {
		return this.x -= e, this.y -= e, this.z -= e, this;
	}
	subVectors(e, t) {
		return this.x = e.x - t.x, this.y = e.y - t.y, this.z = e.z - t.z, this;
	}
	multiply(e) {
		return this.x *= e.x, this.y *= e.y, this.z *= e.z, this;
	}
	multiplyScalar(e) {
		return this.x *= e, this.y *= e, this.z *= e, this;
	}
	multiplyVectors(e, t) {
		return this.x = e.x * t.x, this.y = e.y * t.y, this.z = e.z * t.z, this;
	}
	applyEuler(e) {
		return this.applyQuaternion(Of.setFromEuler(e));
	}
	applyAxisAngle(e, t) {
		return this.applyQuaternion(Of.setFromAxisAngle(e, t));
	}
	applyMatrix3(e) {
		let t = this.x, n = this.y, r = this.z, i = e.elements;
		return this.x = i[0] * t + i[3] * n + i[6] * r, this.y = i[1] * t + i[4] * n + i[7] * r, this.z = i[2] * t + i[5] * n + i[8] * r, this;
	}
	applyNormalMatrix(e) {
		return this.applyMatrix3(e).normalize();
	}
	applyMatrix4(e) {
		let t = this.x, n = this.y, r = this.z, i = e.elements, a = 1 / (i[3] * t + i[7] * n + i[11] * r + i[15]);
		return this.x = (i[0] * t + i[4] * n + i[8] * r + i[12]) * a, this.y = (i[1] * t + i[5] * n + i[9] * r + i[13]) * a, this.z = (i[2] * t + i[6] * n + i[10] * r + i[14]) * a, this;
	}
	applyQuaternion(e) {
		let t = this.x, n = this.y, r = this.z, i = e.x, a = e.y, o = e.z, s = e.w, c = 2 * (a * r - o * n), l = 2 * (o * t - i * r), u = 2 * (i * n - a * t);
		return this.x = t + s * c + a * u - o * l, this.y = n + s * l + o * c - i * u, this.z = r + s * u + i * l - a * c, this;
	}
	project(e) {
		return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix);
	}
	unproject(e) {
		return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld);
	}
	transformDirection(e) {
		let t = this.x, n = this.y, r = this.z, i = e.elements;
		return this.x = i[0] * t + i[4] * n + i[8] * r, this.y = i[1] * t + i[5] * n + i[9] * r, this.z = i[2] * t + i[6] * n + i[10] * r, this.normalize();
	}
	divide(e) {
		return this.x /= e.x, this.y /= e.y, this.z /= e.z, this;
	}
	divideScalar(e) {
		return this.multiplyScalar(1 / e);
	}
	min(e) {
		return this.x = Math.min(this.x, e.x), this.y = Math.min(this.y, e.y), this.z = Math.min(this.z, e.z), this;
	}
	max(e) {
		return this.x = Math.max(this.x, e.x), this.y = Math.max(this.y, e.y), this.z = Math.max(this.z, e.z), this;
	}
	clamp(e, t) {
		return this.x = q(this.x, e.x, t.x), this.y = q(this.y, e.y, t.y), this.z = q(this.z, e.z, t.z), this;
	}
	clampScalar(e, t) {
		return this.x = q(this.x, e, t), this.y = q(this.y, e, t), this.z = q(this.z, e, t), this;
	}
	clampLength(e, t) {
		let n = this.length();
		return this.divideScalar(n || 1).multiplyScalar(q(n, e, t));
	}
	floor() {
		return this.x = Math.floor(this.x), this.y = Math.floor(this.y), this.z = Math.floor(this.z), this;
	}
	ceil() {
		return this.x = Math.ceil(this.x), this.y = Math.ceil(this.y), this.z = Math.ceil(this.z), this;
	}
	round() {
		return this.x = Math.round(this.x), this.y = Math.round(this.y), this.z = Math.round(this.z), this;
	}
	roundToZero() {
		return this.x = Math.trunc(this.x), this.y = Math.trunc(this.y), this.z = Math.trunc(this.z), this;
	}
	negate() {
		return this.x = -this.x, this.y = -this.y, this.z = -this.z, this;
	}
	dot(e) {
		return this.x * e.x + this.y * e.y + this.z * e.z;
	}
	lengthSq() {
		return this.x * this.x + this.y * this.y + this.z * this.z;
	}
	length() {
		return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
	}
	manhattanLength() {
		return Math.abs(this.x) + Math.abs(this.y) + Math.abs(this.z);
	}
	normalize() {
		return this.divideScalar(this.length() || 1);
	}
	setLength(e) {
		return this.normalize().multiplyScalar(e);
	}
	lerp(e, t) {
		return this.x += (e.x - this.x) * t, this.y += (e.y - this.y) * t, this.z += (e.z - this.z) * t, this;
	}
	lerpVectors(e, t, n) {
		return this.x = e.x + (t.x - e.x) * n, this.y = e.y + (t.y - e.y) * n, this.z = e.z + (t.z - e.z) * n, this;
	}
	cross(e) {
		return this.crossVectors(this, e);
	}
	crossVectors(e, t) {
		let n = e.x, r = e.y, i = e.z, a = t.x, o = t.y, s = t.z;
		return this.x = r * s - i * o, this.y = i * a - n * s, this.z = n * o - r * a, this;
	}
	projectOnVector(e) {
		let t = e.lengthSq();
		if (t === 0) return this.set(0, 0, 0);
		let n = e.dot(this) / t;
		return this.copy(e).multiplyScalar(n);
	}
	projectOnPlane(e) {
		return Df.copy(this).projectOnVector(e), this.sub(Df);
	}
	reflect(e) {
		return this.sub(Df.copy(e).multiplyScalar(2 * this.dot(e)));
	}
	angleTo(e) {
		let t = Math.sqrt(this.lengthSq() * e.lengthSq());
		if (t === 0) return Math.PI / 2;
		let n = this.dot(e) / t;
		return Math.acos(q(n, -1, 1));
	}
	distanceTo(e) {
		return Math.sqrt(this.distanceToSquared(e));
	}
	distanceToSquared(e) {
		let t = this.x - e.x, n = this.y - e.y, r = this.z - e.z;
		return t * t + n * n + r * r;
	}
	manhattanDistanceTo(e) {
		return Math.abs(this.x - e.x) + Math.abs(this.y - e.y) + Math.abs(this.z - e.z);
	}
	setFromSpherical(e) {
		return this.setFromSphericalCoords(e.radius, e.phi, e.theta);
	}
	setFromSphericalCoords(e, t, n) {
		let r = Math.sin(t) * e;
		return this.x = r * Math.sin(n), this.y = Math.cos(t) * e, this.z = r * Math.cos(n), this;
	}
	setFromCylindrical(e) {
		return this.setFromCylindricalCoords(e.radius, e.theta, e.y);
	}
	setFromCylindricalCoords(e, t, n) {
		return this.x = e * Math.sin(t), this.y = n, this.z = e * Math.cos(t), this;
	}
	setFromMatrixPosition(e) {
		let t = e.elements;
		return this.x = t[12], this.y = t[13], this.z = t[14], this;
	}
	setFromMatrixScale(e) {
		let t = this.setFromMatrixColumn(e, 0).length(), n = this.setFromMatrixColumn(e, 1).length(), r = this.setFromMatrixColumn(e, 2).length();
		return this.x = t, this.y = n, this.z = r, this;
	}
	setFromMatrixColumn(e, t) {
		return this.fromArray(e.elements, t * 4);
	}
	setFromMatrix3Column(e, t) {
		return this.fromArray(e.elements, t * 3);
	}
	setFromEuler(e) {
		return this.x = e._x, this.y = e._y, this.z = e._z, this;
	}
	setFromColor(e) {
		return this.x = e.r, this.y = e.g, this.z = e.b, this;
	}
	equals(e) {
		return e.x === this.x && e.y === this.y && e.z === this.z;
	}
	fromArray(e, t = 0) {
		return this.x = e[t], this.y = e[t + 1], this.z = e[t + 2], this;
	}
	toArray(e = [], t = 0) {
		return e[t] = this.x, e[t + 1] = this.y, e[t + 2] = this.z, e;
	}
	fromBufferAttribute(e, t) {
		return this.x = e.getX(t), this.y = e.getY(t), this.z = e.getZ(t), this;
	}
	random() {
		return this.x = Math.random(), this.y = Math.random(), this.z = Math.random(), this;
	}
	randomDirection() {
		let e = Math.random() * Math.PI * 2, t = Math.random() * 2 - 1, n = Math.sqrt(1 - t * t);
		return this.x = n * Math.cos(e), this.y = t, this.z = n * Math.sin(e), this;
	}
	*[Symbol.iterator]() {
		yield this.x, yield this.y, yield this.z;
	}
}, Df = /*@__PURE__*/ new J(), Of = /*@__PURE__*/ new Ef(), Y = class e {
	static {
		e.prototype.isMatrix3 = !0;
	}
	constructor(e, t, n, r, i, a, o, s, c) {
		this.elements = [
			1,
			0,
			0,
			0,
			1,
			0,
			0,
			0,
			1
		], e !== void 0 && this.set(e, t, n, r, i, a, o, s, c);
	}
	set(e, t, n, r, i, a, o, s, c) {
		let l = this.elements;
		return l[0] = e, l[1] = r, l[2] = o, l[3] = t, l[4] = i, l[5] = s, l[6] = n, l[7] = a, l[8] = c, this;
	}
	identity() {
		return this.set(1, 0, 0, 0, 1, 0, 0, 0, 1), this;
	}
	copy(e) {
		let t = this.elements, n = e.elements;
		return t[0] = n[0], t[1] = n[1], t[2] = n[2], t[3] = n[3], t[4] = n[4], t[5] = n[5], t[6] = n[6], t[7] = n[7], t[8] = n[8], this;
	}
	extractBasis(e, t, n) {
		return e.setFromMatrix3Column(this, 0), t.setFromMatrix3Column(this, 1), n.setFromMatrix3Column(this, 2), this;
	}
	setFromMatrix4(e) {
		let t = e.elements;
		return this.set(t[0], t[4], t[8], t[1], t[5], t[9], t[2], t[6], t[10]), this;
	}
	multiply(e) {
		return this.multiplyMatrices(this, e);
	}
	premultiply(e) {
		return this.multiplyMatrices(e, this);
	}
	multiplyMatrices(e, t) {
		let n = e.elements, r = t.elements, i = this.elements, a = n[0], o = n[3], s = n[6], c = n[1], l = n[4], u = n[7], d = n[2], f = n[5], p = n[8], m = r[0], h = r[3], g = r[6], _ = r[1], v = r[4], y = r[7], b = r[2], x = r[5], S = r[8];
		return i[0] = a * m + o * _ + s * b, i[3] = a * h + o * v + s * x, i[6] = a * g + o * y + s * S, i[1] = c * m + l * _ + u * b, i[4] = c * h + l * v + u * x, i[7] = c * g + l * y + u * S, i[2] = d * m + f * _ + p * b, i[5] = d * h + f * v + p * x, i[8] = d * g + f * y + p * S, this;
	}
	multiplyScalar(e) {
		let t = this.elements;
		return t[0] *= e, t[3] *= e, t[6] *= e, t[1] *= e, t[4] *= e, t[7] *= e, t[2] *= e, t[5] *= e, t[8] *= e, this;
	}
	determinant() {
		let e = this.elements, t = e[0], n = e[1], r = e[2], i = e[3], a = e[4], o = e[5], s = e[6], c = e[7], l = e[8];
		return t * a * l - t * o * c - n * i * l + n * o * s + r * i * c - r * a * s;
	}
	invert() {
		let e = this.elements, t = e[0], n = e[1], r = e[2], i = e[3], a = e[4], o = e[5], s = e[6], c = e[7], l = e[8], u = l * a - o * c, d = o * s - l * i, f = c * i - a * s, p = t * u + n * d + r * f;
		if (p === 0) return this.set(0, 0, 0, 0, 0, 0, 0, 0, 0);
		let m = 1 / p;
		return e[0] = u * m, e[1] = (r * c - l * n) * m, e[2] = (o * n - r * a) * m, e[3] = d * m, e[4] = (l * t - r * s) * m, e[5] = (r * i - o * t) * m, e[6] = f * m, e[7] = (n * s - c * t) * m, e[8] = (a * t - n * i) * m, this;
	}
	transpose() {
		let e, t = this.elements;
		return e = t[1], t[1] = t[3], t[3] = e, e = t[2], t[2] = t[6], t[6] = e, e = t[5], t[5] = t[7], t[7] = e, this;
	}
	getNormalMatrix(e) {
		return this.setFromMatrix4(e).invert().transpose();
	}
	transposeIntoArray(e) {
		let t = this.elements;
		return e[0] = t[0], e[1] = t[3], e[2] = t[6], e[3] = t[1], e[4] = t[4], e[5] = t[7], e[6] = t[2], e[7] = t[5], e[8] = t[8], this;
	}
	setUvTransform(e, t, n, r, i, a, o) {
		let s = Math.cos(i), c = Math.sin(i);
		return this.set(n * s, n * c, -n * (s * a + c * o) + a + e, -r * c, r * s, -r * (-c * a + s * o) + o + t, 0, 0, 1), this;
	}
	scale(e, t) {
		return Jd("Matrix3: .scale() is deprecated. Use .makeScale() instead."), this.premultiply(kf.makeScale(e, t)), this;
	}
	rotate(e) {
		return Jd("Matrix3: .rotate() is deprecated. Use .makeRotation() instead."), this.premultiply(kf.makeRotation(-e)), this;
	}
	translate(e, t) {
		return Jd("Matrix3: .translate() is deprecated. Use .makeTranslation() instead."), this.premultiply(kf.makeTranslation(e, t)), this;
	}
	makeTranslation(e, t) {
		return e.isVector2 ? this.set(1, 0, e.x, 0, 1, e.y, 0, 0, 1) : this.set(1, 0, e, 0, 1, t, 0, 0, 1), this;
	}
	makeRotation(e) {
		let t = Math.cos(e), n = Math.sin(e);
		return this.set(t, -n, 0, n, t, 0, 0, 0, 1), this;
	}
	makeScale(e, t) {
		return this.set(e, 0, 0, 0, t, 0, 0, 0, 1), this;
	}
	equals(e) {
		let t = this.elements, n = e.elements;
		for (let e = 0; e < 9; e++) if (t[e] !== n[e]) return !1;
		return !0;
	}
	fromArray(e, t = 0) {
		for (let n = 0; n < 9; n++) this.elements[n] = e[n + t];
		return this;
	}
	toArray(e = [], t = 0) {
		let n = this.elements;
		return e[t] = n[0], e[t + 1] = n[1], e[t + 2] = n[2], e[t + 3] = n[3], e[t + 4] = n[4], e[t + 5] = n[5], e[t + 6] = n[6], e[t + 7] = n[7], e[t + 8] = n[8], e;
	}
	clone() {
		return new this.constructor().fromArray(this.elements);
	}
}, kf = /*@__PURE__*/ new Y(), Af = /*@__PURE__*/ new Y().set(.4123908, .3575843, .1804808, .212639, .7151687, .0721923, .0193308, .1191948, .9505322), jf = /*@__PURE__*/ new Y().set(3.2409699, -1.5373832, -.4986108, -.9692436, 1.8759675, .0415551, .0556301, -.203977, 1.0569715);
function Mf() {
	let e = {
		enabled: !0,
		workingColorSpace: Pd,
		spaces: {},
		convert: function(e, t, n) {
			return this.enabled === !1 || t === n || !t || !n ? e : (this.spaces[t].transfer === "srgb" && (e.r = Nf(e.r), e.g = Nf(e.g), e.b = Nf(e.b)), this.spaces[t].primaries !== this.spaces[n].primaries && (e.applyMatrix3(this.spaces[t].toXYZ), e.applyMatrix3(this.spaces[n].fromXYZ)), this.spaces[n].transfer === "srgb" && (e.r = Pf(e.r), e.g = Pf(e.g), e.b = Pf(e.b)), e);
		},
		workingToColorSpace: function(e, t) {
			return this.convert(e, this.workingColorSpace, t);
		},
		colorSpaceToWorking: function(e, t) {
			return this.convert(e, t, this.workingColorSpace);
		},
		getPrimaries: function(e) {
			return this.spaces[e].primaries;
		},
		getTransfer: function(e) {
			return e === "" ? Fd : this.spaces[e].transfer;
		},
		getToneMappingMode: function(e) {
			return this.spaces[e].outputColorSpaceConfig.toneMappingMode || "standard";
		},
		getLuminanceCoefficients: function(e, t = this.workingColorSpace) {
			return e.fromArray(this.spaces[t].luminanceCoefficients);
		},
		define: function(e) {
			Object.assign(this.spaces, e);
		},
		_getMatrix: function(e, t, n) {
			return e.copy(this.spaces[t].toXYZ).multiply(this.spaces[n].fromXYZ);
		},
		_getDrawingBufferColorSpace: function(e) {
			return this.spaces[e].outputColorSpaceConfig.drawingBufferColorSpace;
		},
		_getUnpackColorSpace: function(e = this.workingColorSpace) {
			return this.spaces[e].workingColorSpaceConfig.unpackColorSpace;
		},
		fromWorkingColorSpace: function(t, n) {
			return Jd("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."), e.workingToColorSpace(t, n);
		},
		toWorkingColorSpace: function(t, n) {
			return Jd("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."), e.colorSpaceToWorking(t, n);
		}
	}, t = [
		.64,
		.33,
		.3,
		.6,
		.15,
		.06
	], n = [
		.2126,
		.7152,
		.0722
	], r = [.3127, .329];
	return e.define({
		[Pd]: {
			primaries: t,
			whitePoint: r,
			transfer: Fd,
			toXYZ: Af,
			fromXYZ: jf,
			luminanceCoefficients: n,
			workingColorSpaceConfig: { unpackColorSpace: Nd },
			outputColorSpaceConfig: { drawingBufferColorSpace: Nd }
		},
		[Nd]: {
			primaries: t,
			whitePoint: r,
			transfer: Id,
			toXYZ: Af,
			fromXYZ: jf,
			luminanceCoefficients: n,
			outputColorSpaceConfig: { drawingBufferColorSpace: Nd }
		}
	}), e;
}
var X = /*@__PURE__*/ Mf();
function Nf(e) {
	return e < .04045 ? e * .0773993808 : (e * .9478672986 + .0521327014) ** 2.4;
}
function Pf(e) {
	return e < .0031308 ? e * 12.92 : 1.055 * e ** .41666 - .055;
}
var Ff, If = class {
	static getDataURL(e, t = "image/png") {
		if (/^data:/i.test(e.src) || typeof HTMLCanvasElement > "u") return e.src;
		let n;
		if (e instanceof HTMLCanvasElement) n = e;
		else {
			Ff === void 0 && (Ff = Ud("canvas")), Ff.width = e.width, Ff.height = e.height;
			let t = Ff.getContext("2d");
			e instanceof ImageData ? t.putImageData(e, 0, 0) : t.drawImage(e, 0, 0, e.width, e.height), n = Ff;
		}
		return n.toDataURL(t);
	}
	static sRGBToLinear(e) {
		if (typeof HTMLImageElement < "u" && e instanceof HTMLImageElement || typeof HTMLCanvasElement < "u" && e instanceof HTMLCanvasElement || typeof ImageBitmap < "u" && e instanceof ImageBitmap) {
			let t = Ud("canvas");
			t.width = e.width, t.height = e.height;
			let n = t.getContext("2d");
			n.drawImage(e, 0, 0, e.width, e.height);
			let r = n.getImageData(0, 0, e.width, e.height), i = r.data;
			for (let e = 0; e < i.length; e++) i[e] = Nf(i[e] / 255) * 255;
			return n.putImageData(r, 0, 0), t;
		} else if (e.data) {
			let t = e.data.slice(0);
			for (let e = 0; e < t.length; e++) t instanceof Uint8Array || t instanceof Uint8ClampedArray ? t[e] = Math.floor(Nf(t[e] / 255) * 255) : t[e] = Nf(t[e]);
			return {
				data: t,
				width: e.width,
				height: e.height
			};
		} else return G("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."), e;
	}
}, Lf = 0, Rf = class {
	constructor(e = null) {
		this.isSource = !0, Object.defineProperty(this, "id", { value: Lf++ }), this.uuid = nf(), this.data = e, this.dataReady = !0, this.version = 0;
	}
	getSize(e) {
		let t = this.data;
		return typeof HTMLVideoElement < "u" && t instanceof HTMLVideoElement ? e.set(t.videoWidth, t.videoHeight, 0) : typeof VideoFrame < "u" && t instanceof VideoFrame ? e.set(t.displayWidth, t.displayHeight, 0) : t === null ? e.set(0, 0, 0) : e.set(t.width, t.height, t.depth || 0), e;
	}
	set needsUpdate(e) {
		e === !0 && this.version++;
	}
	toJSON(e) {
		let t = e === void 0 || typeof e == "string";
		if (!t && e.images[this.uuid] !== void 0) return e.images[this.uuid];
		let n = {
			uuid: this.uuid,
			url: ""
		}, r = this.data;
		if (r !== null) {
			let e;
			if (Array.isArray(r)) {
				e = [];
				for (let t = 0, n = r.length; t < n; t++) r[t].isDataTexture ? e.push(zf(r[t].image)) : e.push(zf(r[t]));
			} else e = zf(r);
			n.url = e;
		}
		return t || (e.images[this.uuid] = n), n;
	}
};
function zf(e) {
	return typeof HTMLImageElement < "u" && e instanceof HTMLImageElement || typeof HTMLCanvasElement < "u" && e instanceof HTMLCanvasElement || typeof ImageBitmap < "u" && e instanceof ImageBitmap ? If.getDataURL(e) : e.data ? {
		data: Array.from(e.data),
		width: e.width,
		height: e.height,
		type: e.data.constructor.name
	} : (G("Texture: Unable to serialize Texture."), {});
}
var Bf = 0, Vf = /*@__PURE__*/ new J(), Hf = class e extends Zd {
	constructor(t = e.DEFAULT_IMAGE, n = e.DEFAULT_MAPPING, r = fu, i = fu, a = _u, o = yu, s = Fu, c = bu, l = e.DEFAULT_ANISOTROPY, u = "") {
		super(), this.isTexture = !0, Object.defineProperty(this, "id", { value: Bf++ }), this.uuid = nf(), this.name = "", this.source = new Rf(t), this.mipmaps = [], this.mapping = n, this.channel = 0, this.wrapS = r, this.wrapT = i, this.magFilter = a, this.minFilter = o, this.anisotropy = l, this.format = s, this.internalFormat = null, this.type = c, this.offset = new Tf(0, 0), this.repeat = new Tf(1, 1), this.center = new Tf(0, 0), this.rotation = 0, this.matrixAutoUpdate = !0, this.matrix = new Y(), this.generateMipmaps = !0, this.premultiplyAlpha = !1, this.flipY = !0, this.unpackAlignment = 4, this.colorSpace = u, this.userData = {}, this.updateRanges = [], this.version = 0, this.onUpdate = null, this.renderTarget = null, this.isRenderTargetTexture = !1, this.isArrayTexture = !!(t && t.depth && t.depth > 1), this.pmremVersion = 0, this.normalized = !1;
	}
	get width() {
		return this.source.getSize(Vf).x;
	}
	get height() {
		return this.source.getSize(Vf).y;
	}
	get depth() {
		return this.source.getSize(Vf).z;
	}
	get image() {
		return this.source.data;
	}
	set image(e) {
		this.source.data = e;
	}
	updateMatrix() {
		this.matrix.setUvTransform(this.offset.x, this.offset.y, this.repeat.x, this.repeat.y, this.rotation, this.center.x, this.center.y);
	}
	addUpdateRange(e, t) {
		this.updateRanges.push({
			start: e,
			count: t
		});
	}
	clearUpdateRanges() {
		this.updateRanges.length = 0;
	}
	clone() {
		return new this.constructor().copy(this);
	}
	copy(e) {
		return this.name = e.name, this.source = e.source, this.mipmaps = e.mipmaps.slice(0), this.mapping = e.mapping, this.channel = e.channel, this.wrapS = e.wrapS, this.wrapT = e.wrapT, this.magFilter = e.magFilter, this.minFilter = e.minFilter, this.anisotropy = e.anisotropy, this.format = e.format, this.internalFormat = e.internalFormat, this.type = e.type, this.normalized = e.normalized, this.offset.copy(e.offset), this.repeat.copy(e.repeat), this.center.copy(e.center), this.rotation = e.rotation, this.matrixAutoUpdate = e.matrixAutoUpdate, this.matrix.copy(e.matrix), this.generateMipmaps = e.generateMipmaps, this.premultiplyAlpha = e.premultiplyAlpha, this.flipY = e.flipY, this.unpackAlignment = e.unpackAlignment, this.colorSpace = e.colorSpace, this.renderTarget = e.renderTarget, this.isRenderTargetTexture = e.isRenderTargetTexture, this.isArrayTexture = e.isArrayTexture, this.userData = JSON.parse(JSON.stringify(e.userData)), this.needsUpdate = !0, this;
	}
	setValues(e) {
		for (let t in e) {
			let n = e[t];
			if (n === void 0) {
				G(`Texture.setValues(): parameter '${t}' has value of undefined.`);
				continue;
			}
			let r = this[t];
			if (r === void 0) {
				G(`Texture.setValues(): property '${t}' does not exist.`);
				continue;
			}
			r && n && r.isVector2 && n.isVector2 || r && n && r.isVector3 && n.isVector3 || r && n && r.isMatrix3 && n.isMatrix3 ? r.copy(n) : this[t] = n;
		}
	}
	toJSON(e) {
		let t = e === void 0 || typeof e == "string";
		if (!t && e.textures[this.uuid] !== void 0) return e.textures[this.uuid];
		let n = {
			metadata: {
				version: 4.7,
				type: "Texture",
				generator: "Texture.toJSON"
			},
			uuid: this.uuid,
			name: this.name,
			image: this.source.toJSON(e).uuid,
			mapping: this.mapping,
			channel: this.channel,
			repeat: [this.repeat.x, this.repeat.y],
			offset: [this.offset.x, this.offset.y],
			center: [this.center.x, this.center.y],
			rotation: this.rotation,
			wrap: [this.wrapS, this.wrapT],
			format: this.format,
			internalFormat: this.internalFormat,
			type: this.type,
			normalized: this.normalized,
			colorSpace: this.colorSpace,
			minFilter: this.minFilter,
			magFilter: this.magFilter,
			anisotropy: this.anisotropy,
			flipY: this.flipY,
			generateMipmaps: this.generateMipmaps,
			premultiplyAlpha: this.premultiplyAlpha,
			unpackAlignment: this.unpackAlignment
		};
		return Object.keys(this.userData).length > 0 && (n.userData = this.userData), t || (e.textures[this.uuid] = n), n;
	}
	dispose() {
		this.dispatchEvent({ type: "dispose" });
	}
	transformUv(e) {
		if (this.mapping !== 300) return e;
		if (e.applyMatrix3(this.matrix), e.x < 0 || e.x > 1) switch (this.wrapS) {
			case du:
				e.x -= Math.floor(e.x);
				break;
			case fu:
				e.x = e.x < 0 ? 0 : 1;
				break;
			case pu:
				Math.abs(Math.floor(e.x) % 2) === 1 ? e.x = Math.ceil(e.x) - e.x : e.x -= Math.floor(e.x);
				break;
		}
		if (e.y < 0 || e.y > 1) switch (this.wrapT) {
			case du:
				e.y -= Math.floor(e.y);
				break;
			case fu:
				e.y = e.y < 0 ? 0 : 1;
				break;
			case pu:
				Math.abs(Math.floor(e.y) % 2) === 1 ? e.y = Math.ceil(e.y) - e.y : e.y -= Math.floor(e.y);
				break;
		}
		return this.flipY && (e.y = 1 - e.y), e;
	}
	set needsUpdate(e) {
		e === !0 && (this.version++, this.source.needsUpdate = !0);
	}
	set needsPMREMUpdate(e) {
		e === !0 && this.pmremVersion++;
	}
};
Hf.DEFAULT_IMAGE = null, Hf.DEFAULT_MAPPING = 300, Hf.DEFAULT_ANISOTROPY = 1;
var Uf = class e {
	static {
		e.prototype.isVector4 = !0;
	}
	constructor(e = 0, t = 0, n = 0, r = 1) {
		this.x = e, this.y = t, this.z = n, this.w = r;
	}
	get width() {
		return this.z;
	}
	set width(e) {
		this.z = e;
	}
	get height() {
		return this.w;
	}
	set height(e) {
		this.w = e;
	}
	set(e, t, n, r) {
		return this.x = e, this.y = t, this.z = n, this.w = r, this;
	}
	setScalar(e) {
		return this.x = e, this.y = e, this.z = e, this.w = e, this;
	}
	setX(e) {
		return this.x = e, this;
	}
	setY(e) {
		return this.y = e, this;
	}
	setZ(e) {
		return this.z = e, this;
	}
	setW(e) {
		return this.w = e, this;
	}
	setComponent(e, t) {
		switch (e) {
			case 0:
				this.x = t;
				break;
			case 1:
				this.y = t;
				break;
			case 2:
				this.z = t;
				break;
			case 3:
				this.w = t;
				break;
			default: throw Error("THREE.Vector4: index is out of range: " + e);
		}
		return this;
	}
	getComponent(e) {
		switch (e) {
			case 0: return this.x;
			case 1: return this.y;
			case 2: return this.z;
			case 3: return this.w;
			default: throw Error("THREE.Vector4: index is out of range: " + e);
		}
	}
	clone() {
		return new this.constructor(this.x, this.y, this.z, this.w);
	}
	copy(e) {
		return this.x = e.x, this.y = e.y, this.z = e.z, this.w = e.w === void 0 ? 1 : e.w, this;
	}
	add(e) {
		return this.x += e.x, this.y += e.y, this.z += e.z, this.w += e.w, this;
	}
	addScalar(e) {
		return this.x += e, this.y += e, this.z += e, this.w += e, this;
	}
	addVectors(e, t) {
		return this.x = e.x + t.x, this.y = e.y + t.y, this.z = e.z + t.z, this.w = e.w + t.w, this;
	}
	addScaledVector(e, t) {
		return this.x += e.x * t, this.y += e.y * t, this.z += e.z * t, this.w += e.w * t, this;
	}
	sub(e) {
		return this.x -= e.x, this.y -= e.y, this.z -= e.z, this.w -= e.w, this;
	}
	subScalar(e) {
		return this.x -= e, this.y -= e, this.z -= e, this.w -= e, this;
	}
	subVectors(e, t) {
		return this.x = e.x - t.x, this.y = e.y - t.y, this.z = e.z - t.z, this.w = e.w - t.w, this;
	}
	multiply(e) {
		return this.x *= e.x, this.y *= e.y, this.z *= e.z, this.w *= e.w, this;
	}
	multiplyScalar(e) {
		return this.x *= e, this.y *= e, this.z *= e, this.w *= e, this;
	}
	applyMatrix4(e) {
		let t = this.x, n = this.y, r = this.z, i = this.w, a = e.elements;
		return this.x = a[0] * t + a[4] * n + a[8] * r + a[12] * i, this.y = a[1] * t + a[5] * n + a[9] * r + a[13] * i, this.z = a[2] * t + a[6] * n + a[10] * r + a[14] * i, this.w = a[3] * t + a[7] * n + a[11] * r + a[15] * i, this;
	}
	divide(e) {
		return this.x /= e.x, this.y /= e.y, this.z /= e.z, this.w /= e.w, this;
	}
	divideScalar(e) {
		return this.multiplyScalar(1 / e);
	}
	setAxisAngleFromQuaternion(e) {
		this.w = 2 * Math.acos(e.w);
		let t = Math.sqrt(1 - e.w * e.w);
		return t < 1e-4 ? (this.x = 1, this.y = 0, this.z = 0) : (this.x = e.x / t, this.y = e.y / t, this.z = e.z / t), this;
	}
	setAxisAngleFromRotationMatrix(e) {
		let t, n, r, i, a = .01, o = .1, s = e.elements, c = s[0], l = s[4], u = s[8], d = s[1], f = s[5], p = s[9], m = s[2], h = s[6], g = s[10];
		if (Math.abs(l - d) < a && Math.abs(u - m) < a && Math.abs(p - h) < a) {
			if (Math.abs(l + d) < o && Math.abs(u + m) < o && Math.abs(p + h) < o && Math.abs(c + f + g - 3) < o) return this.set(1, 0, 0, 0), this;
			t = Math.PI;
			let e = (c + 1) / 2, s = (f + 1) / 2, _ = (g + 1) / 2, v = (l + d) / 4, y = (u + m) / 4, b = (p + h) / 4;
			return e > s && e > _ ? e < a ? (n = 0, r = .707106781, i = .707106781) : (n = Math.sqrt(e), r = v / n, i = y / n) : s > _ ? s < a ? (n = .707106781, r = 0, i = .707106781) : (r = Math.sqrt(s), n = v / r, i = b / r) : _ < a ? (n = .707106781, r = .707106781, i = 0) : (i = Math.sqrt(_), n = y / i, r = b / i), this.set(n, r, i, t), this;
		}
		let _ = Math.sqrt((h - p) * (h - p) + (u - m) * (u - m) + (d - l) * (d - l));
		return Math.abs(_) < .001 && (_ = 1), this.x = (h - p) / _, this.y = (u - m) / _, this.z = (d - l) / _, this.w = Math.acos((c + f + g - 1) / 2), this;
	}
	setFromMatrixPosition(e) {
		let t = e.elements;
		return this.x = t[12], this.y = t[13], this.z = t[14], this.w = t[15], this;
	}
	min(e) {
		return this.x = Math.min(this.x, e.x), this.y = Math.min(this.y, e.y), this.z = Math.min(this.z, e.z), this.w = Math.min(this.w, e.w), this;
	}
	max(e) {
		return this.x = Math.max(this.x, e.x), this.y = Math.max(this.y, e.y), this.z = Math.max(this.z, e.z), this.w = Math.max(this.w, e.w), this;
	}
	clamp(e, t) {
		return this.x = q(this.x, e.x, t.x), this.y = q(this.y, e.y, t.y), this.z = q(this.z, e.z, t.z), this.w = q(this.w, e.w, t.w), this;
	}
	clampScalar(e, t) {
		return this.x = q(this.x, e, t), this.y = q(this.y, e, t), this.z = q(this.z, e, t), this.w = q(this.w, e, t), this;
	}
	clampLength(e, t) {
		let n = this.length();
		return this.divideScalar(n || 1).multiplyScalar(q(n, e, t));
	}
	floor() {
		return this.x = Math.floor(this.x), this.y = Math.floor(this.y), this.z = Math.floor(this.z), this.w = Math.floor(this.w), this;
	}
	ceil() {
		return this.x = Math.ceil(this.x), this.y = Math.ceil(this.y), this.z = Math.ceil(this.z), this.w = Math.ceil(this.w), this;
	}
	round() {
		return this.x = Math.round(this.x), this.y = Math.round(this.y), this.z = Math.round(this.z), this.w = Math.round(this.w), this;
	}
	roundToZero() {
		return this.x = Math.trunc(this.x), this.y = Math.trunc(this.y), this.z = Math.trunc(this.z), this.w = Math.trunc(this.w), this;
	}
	negate() {
		return this.x = -this.x, this.y = -this.y, this.z = -this.z, this.w = -this.w, this;
	}
	dot(e) {
		return this.x * e.x + this.y * e.y + this.z * e.z + this.w * e.w;
	}
	lengthSq() {
		return this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w;
	}
	length() {
		return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w);
	}
	manhattanLength() {
		return Math.abs(this.x) + Math.abs(this.y) + Math.abs(this.z) + Math.abs(this.w);
	}
	normalize() {
		return this.divideScalar(this.length() || 1);
	}
	setLength(e) {
		return this.normalize().multiplyScalar(e);
	}
	lerp(e, t) {
		return this.x += (e.x - this.x) * t, this.y += (e.y - this.y) * t, this.z += (e.z - this.z) * t, this.w += (e.w - this.w) * t, this;
	}
	lerpVectors(e, t, n) {
		return this.x = e.x + (t.x - e.x) * n, this.y = e.y + (t.y - e.y) * n, this.z = e.z + (t.z - e.z) * n, this.w = e.w + (t.w - e.w) * n, this;
	}
	equals(e) {
		return e.x === this.x && e.y === this.y && e.z === this.z && e.w === this.w;
	}
	fromArray(e, t = 0) {
		return this.x = e[t], this.y = e[t + 1], this.z = e[t + 2], this.w = e[t + 3], this;
	}
	toArray(e = [], t = 0) {
		return e[t] = this.x, e[t + 1] = this.y, e[t + 2] = this.z, e[t + 3] = this.w, e;
	}
	fromBufferAttribute(e, t) {
		return this.x = e.getX(t), this.y = e.getY(t), this.z = e.getZ(t), this.w = e.getW(t), this;
	}
	random() {
		return this.x = Math.random(), this.y = Math.random(), this.z = Math.random(), this.w = Math.random(), this;
	}
	*[Symbol.iterator]() {
		yield this.x, yield this.y, yield this.z, yield this.w;
	}
}, Wf = class extends Zd {
	constructor(e = 1, t = 1, n = {}) {
		super(), n = Object.assign({
			generateMipmaps: !1,
			internalFormat: null,
			minFilter: _u,
			depthBuffer: !0,
			stencilBuffer: !1,
			resolveDepthBuffer: !0,
			resolveStencilBuffer: !0,
			depthTexture: null,
			samples: 0,
			count: 1,
			depth: 1,
			multiview: !1,
			useArrayDepthTexture: !1
		}, n), this.isRenderTarget = !0, this.width = e, this.height = t, this.depth = n.depth, this.scissor = new Uf(0, 0, e, t), this.scissorTest = !1, this.viewport = new Uf(0, 0, e, t), this.textures = [];
		let r = new Hf({
			width: e,
			height: t,
			depth: n.depth
		}), i = n.count;
		for (let e = 0; e < i; e++) this.textures[e] = r.clone(), this.textures[e].isRenderTargetTexture = !0, this.textures[e].renderTarget = this;
		this._setTextureOptions(n), this.depthBuffer = n.depthBuffer, this.stencilBuffer = n.stencilBuffer, this.resolveDepthBuffer = n.resolveDepthBuffer, this.resolveStencilBuffer = n.resolveStencilBuffer, this._depthTexture = null, this.depthTexture = n.depthTexture, this.samples = n.samples, this.multiview = n.multiview, this.useArrayDepthTexture = n.useArrayDepthTexture;
	}
	_setTextureOptions(e = {}) {
		let t = {
			minFilter: _u,
			generateMipmaps: !1,
			flipY: !1,
			internalFormat: null
		};
		e.mapping !== void 0 && (t.mapping = e.mapping), e.wrapS !== void 0 && (t.wrapS = e.wrapS), e.wrapT !== void 0 && (t.wrapT = e.wrapT), e.wrapR !== void 0 && (t.wrapR = e.wrapR), e.magFilter !== void 0 && (t.magFilter = e.magFilter), e.minFilter !== void 0 && (t.minFilter = e.minFilter), e.format !== void 0 && (t.format = e.format), e.type !== void 0 && (t.type = e.type), e.anisotropy !== void 0 && (t.anisotropy = e.anisotropy), e.colorSpace !== void 0 && (t.colorSpace = e.colorSpace), e.flipY !== void 0 && (t.flipY = e.flipY), e.generateMipmaps !== void 0 && (t.generateMipmaps = e.generateMipmaps), e.internalFormat !== void 0 && (t.internalFormat = e.internalFormat);
		for (let e = 0; e < this.textures.length; e++) this.textures[e].setValues(t);
	}
	get texture() {
		return this.textures[0];
	}
	set texture(e) {
		this.textures[0] = e;
	}
	set depthTexture(e) {
		this._depthTexture !== null && (this._depthTexture.renderTarget = null), e !== null && (e.renderTarget = this), this._depthTexture = e;
	}
	get depthTexture() {
		return this._depthTexture;
	}
	setSize(e, t, n = 1) {
		if (this.width !== e || this.height !== t || this.depth !== n) {
			this.width = e, this.height = t, this.depth = n;
			for (let r = 0, i = this.textures.length; r < i; r++) this.textures[r].image.width = e, this.textures[r].image.height = t, this.textures[r].image.depth = n, this.textures[r].isData3DTexture !== !0 && (this.textures[r].isArrayTexture = this.textures[r].image.depth > 1);
			this.dispose();
		}
		this.viewport.set(0, 0, e, t), this.scissor.set(0, 0, e, t);
	}
	clone() {
		return new this.constructor().copy(this);
	}
	copy(e) {
		this.width = e.width, this.height = e.height, this.depth = e.depth, this.scissor.copy(e.scissor), this.scissorTest = e.scissorTest, this.viewport.copy(e.viewport), this.textures.length = 0;
		for (let t = 0, n = e.textures.length; t < n; t++) {
			this.textures[t] = e.textures[t].clone(), this.textures[t].isRenderTargetTexture = !0, this.textures[t].renderTarget = this;
			let n = Object.assign({}, e.textures[t].image);
			this.textures[t].source = new Rf(n);
		}
		return this.depthBuffer = e.depthBuffer, this.stencilBuffer = e.stencilBuffer, this.resolveDepthBuffer = e.resolveDepthBuffer, this.resolveStencilBuffer = e.resolveStencilBuffer, e.depthTexture !== null && (this.depthTexture = e.depthTexture.clone()), this.samples = e.samples, this.multiview = e.multiview, this.useArrayDepthTexture = e.useArrayDepthTexture, this;
	}
	dispose() {
		this.dispatchEvent({ type: "dispose" });
	}
}, Gf = class extends Wf {
	constructor(e = 1, t = 1, n = {}) {
		super(e, t, n), this.isWebGLRenderTarget = !0;
	}
}, Kf = class extends Hf {
	constructor(e = null, t = 1, n = 1, r = 1) {
		super(null), this.isDataArrayTexture = !0, this.image = {
			data: e,
			width: t,
			height: n,
			depth: r
		}, this.magFilter = mu, this.minFilter = mu, this.wrapR = fu, this.generateMipmaps = !1, this.flipY = !1, this.unpackAlignment = 1, this.layerUpdates = /* @__PURE__ */ new Set();
	}
	addLayerUpdate(e) {
		this.layerUpdates.add(e);
	}
	clearLayerUpdates() {
		this.layerUpdates.clear();
	}
}, qf = class extends Hf {
	constructor(e = null, t = 1, n = 1, r = 1) {
		super(null), this.isData3DTexture = !0, this.image = {
			data: e,
			width: t,
			height: n,
			depth: r
		}, this.magFilter = mu, this.minFilter = mu, this.wrapR = fu, this.generateMipmaps = !1, this.flipY = !1, this.unpackAlignment = 1;
	}
}, Jf = class e {
	static {
		e.prototype.isMatrix4 = !0;
	}
	constructor(e, t, n, r, i, a, o, s, c, l, u, d, f, p, m, h) {
		this.elements = [
			1,
			0,
			0,
			0,
			0,
			1,
			0,
			0,
			0,
			0,
			1,
			0,
			0,
			0,
			0,
			1
		], e !== void 0 && this.set(e, t, n, r, i, a, o, s, c, l, u, d, f, p, m, h);
	}
	set(e, t, n, r, i, a, o, s, c, l, u, d, f, p, m, h) {
		let g = this.elements;
		return g[0] = e, g[4] = t, g[8] = n, g[12] = r, g[1] = i, g[5] = a, g[9] = o, g[13] = s, g[2] = c, g[6] = l, g[10] = u, g[14] = d, g[3] = f, g[7] = p, g[11] = m, g[15] = h, this;
	}
	identity() {
		return this.set(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), this;
	}
	clone() {
		return new e().fromArray(this.elements);
	}
	copy(e) {
		let t = this.elements, n = e.elements;
		return t[0] = n[0], t[1] = n[1], t[2] = n[2], t[3] = n[3], t[4] = n[4], t[5] = n[5], t[6] = n[6], t[7] = n[7], t[8] = n[8], t[9] = n[9], t[10] = n[10], t[11] = n[11], t[12] = n[12], t[13] = n[13], t[14] = n[14], t[15] = n[15], this;
	}
	copyPosition(e) {
		let t = this.elements, n = e.elements;
		return t[12] = n[12], t[13] = n[13], t[14] = n[14], this;
	}
	setFromMatrix3(e) {
		let t = e.elements;
		return this.set(t[0], t[3], t[6], 0, t[1], t[4], t[7], 0, t[2], t[5], t[8], 0, 0, 0, 0, 1), this;
	}
	extractBasis(e, t, n) {
		return this.determinantAffine() === 0 ? (e.set(1, 0, 0), t.set(0, 1, 0), n.set(0, 0, 1), this) : (e.setFromMatrixColumn(this, 0), t.setFromMatrixColumn(this, 1), n.setFromMatrixColumn(this, 2), this);
	}
	makeBasis(e, t, n) {
		return this.set(e.x, t.x, n.x, 0, e.y, t.y, n.y, 0, e.z, t.z, n.z, 0, 0, 0, 0, 1), this;
	}
	extractRotation(e) {
		if (e.determinantAffine() === 0) return this.identity();
		let t = this.elements, n = e.elements, r = 1 / Yf.setFromMatrixColumn(e, 0).length(), i = 1 / Yf.setFromMatrixColumn(e, 1).length(), a = 1 / Yf.setFromMatrixColumn(e, 2).length();
		return t[0] = n[0] * r, t[1] = n[1] * r, t[2] = n[2] * r, t[3] = 0, t[4] = n[4] * i, t[5] = n[5] * i, t[6] = n[6] * i, t[7] = 0, t[8] = n[8] * a, t[9] = n[9] * a, t[10] = n[10] * a, t[11] = 0, t[12] = 0, t[13] = 0, t[14] = 0, t[15] = 1, this;
	}
	makeRotationFromEuler(e) {
		let t = this.elements, n = e.x, r = e.y, i = e.z, a = Math.cos(n), o = Math.sin(n), s = Math.cos(r), c = Math.sin(r), l = Math.cos(i), u = Math.sin(i);
		if (e.order === "XYZ") {
			let e = a * l, n = a * u, r = o * l, i = o * u;
			t[0] = s * l, t[4] = -s * u, t[8] = c, t[1] = n + r * c, t[5] = e - i * c, t[9] = -o * s, t[2] = i - e * c, t[6] = r + n * c, t[10] = a * s;
		} else if (e.order === "YXZ") {
			let e = s * l, n = s * u, r = c * l, i = c * u;
			t[0] = e + i * o, t[4] = r * o - n, t[8] = a * c, t[1] = a * u, t[5] = a * l, t[9] = -o, t[2] = n * o - r, t[6] = i + e * o, t[10] = a * s;
		} else if (e.order === "ZXY") {
			let e = s * l, n = s * u, r = c * l, i = c * u;
			t[0] = e - i * o, t[4] = -a * u, t[8] = r + n * o, t[1] = n + r * o, t[5] = a * l, t[9] = i - e * o, t[2] = -a * c, t[6] = o, t[10] = a * s;
		} else if (e.order === "ZYX") {
			let e = a * l, n = a * u, r = o * l, i = o * u;
			t[0] = s * l, t[4] = r * c - n, t[8] = e * c + i, t[1] = s * u, t[5] = i * c + e, t[9] = n * c - r, t[2] = -c, t[6] = o * s, t[10] = a * s;
		} else if (e.order === "YZX") {
			let e = a * s, n = a * c, r = o * s, i = o * c;
			t[0] = s * l, t[4] = i - e * u, t[8] = r * u + n, t[1] = u, t[5] = a * l, t[9] = -o * l, t[2] = -c * l, t[6] = n * u + r, t[10] = e - i * u;
		} else if (e.order === "XZY") {
			let e = a * s, n = a * c, r = o * s, i = o * c;
			t[0] = s * l, t[4] = -u, t[8] = c * l, t[1] = e * u + i, t[5] = a * l, t[9] = n * u - r, t[2] = r * u - n, t[6] = o * l, t[10] = i * u + e;
		}
		return t[3] = 0, t[7] = 0, t[11] = 0, t[12] = 0, t[13] = 0, t[14] = 0, t[15] = 1, this;
	}
	makeRotationFromQuaternion(e) {
		return this.compose(Zf, e, Qf);
	}
	lookAt(e, t, n) {
		let r = this.elements;
		return tp.subVectors(e, t), tp.lengthSq() === 0 && (tp.z = 1), tp.normalize(), $f.crossVectors(n, tp), $f.lengthSq() === 0 && (Math.abs(n.z) === 1 ? tp.x += 1e-4 : tp.z += 1e-4, tp.normalize(), $f.crossVectors(n, tp)), $f.normalize(), ep.crossVectors(tp, $f), r[0] = $f.x, r[4] = ep.x, r[8] = tp.x, r[1] = $f.y, r[5] = ep.y, r[9] = tp.y, r[2] = $f.z, r[6] = ep.z, r[10] = tp.z, this;
	}
	multiply(e) {
		return this.multiplyMatrices(this, e);
	}
	premultiply(e) {
		return this.multiplyMatrices(e, this);
	}
	multiplyMatrices(e, t) {
		let n = e.elements, r = t.elements, i = this.elements, a = n[0], o = n[4], s = n[8], c = n[12], l = n[1], u = n[5], d = n[9], f = n[13], p = n[2], m = n[6], h = n[10], g = n[14], _ = n[3], v = n[7], y = n[11], b = n[15], x = r[0], S = r[4], C = r[8], w = r[12], T = r[1], E = r[5], D = r[9], O = r[13], k = r[2], A = r[6], ee = r[10], te = r[14], j = r[3], ne = r[7], re = r[11], ie = r[15];
		return i[0] = a * x + o * T + s * k + c * j, i[4] = a * S + o * E + s * A + c * ne, i[8] = a * C + o * D + s * ee + c * re, i[12] = a * w + o * O + s * te + c * ie, i[1] = l * x + u * T + d * k + f * j, i[5] = l * S + u * E + d * A + f * ne, i[9] = l * C + u * D + d * ee + f * re, i[13] = l * w + u * O + d * te + f * ie, i[2] = p * x + m * T + h * k + g * j, i[6] = p * S + m * E + h * A + g * ne, i[10] = p * C + m * D + h * ee + g * re, i[14] = p * w + m * O + h * te + g * ie, i[3] = _ * x + v * T + y * k + b * j, i[7] = _ * S + v * E + y * A + b * ne, i[11] = _ * C + v * D + y * ee + b * re, i[15] = _ * w + v * O + y * te + b * ie, this;
	}
	multiplyScalar(e) {
		let t = this.elements;
		return t[0] *= e, t[4] *= e, t[8] *= e, t[12] *= e, t[1] *= e, t[5] *= e, t[9] *= e, t[13] *= e, t[2] *= e, t[6] *= e, t[10] *= e, t[14] *= e, t[3] *= e, t[7] *= e, t[11] *= e, t[15] *= e, this;
	}
	determinant() {
		let e = this.elements, t = e[0], n = e[4], r = e[8], i = e[12], a = e[1], o = e[5], s = e[9], c = e[13], l = e[2], u = e[6], d = e[10], f = e[14], p = e[3], m = e[7], h = e[11], g = e[15], _ = s * f - c * d, v = o * f - c * u, y = o * d - s * u, b = a * f - c * l, x = a * d - s * l, S = a * u - o * l;
		return t * (m * _ - h * v + g * y) - n * (p * _ - h * b + g * x) + r * (p * v - m * b + g * S) - i * (p * y - m * x + h * S);
	}
	determinantAffine() {
		let e = this.elements, t = e[0], n = e[4], r = e[8], i = e[1], a = e[5], o = e[9], s = e[2], c = e[6], l = e[10];
		return t * (a * l - o * c) - n * (i * l - o * s) + r * (i * c - a * s);
	}
	transpose() {
		let e = this.elements, t;
		return t = e[1], e[1] = e[4], e[4] = t, t = e[2], e[2] = e[8], e[8] = t, t = e[6], e[6] = e[9], e[9] = t, t = e[3], e[3] = e[12], e[12] = t, t = e[7], e[7] = e[13], e[13] = t, t = e[11], e[11] = e[14], e[14] = t, this;
	}
	setPosition(e, t, n) {
		let r = this.elements;
		return e.isVector3 ? (r[12] = e.x, r[13] = e.y, r[14] = e.z) : (r[12] = e, r[13] = t, r[14] = n), this;
	}
	invert() {
		let e = this.elements, t = e[0], n = e[1], r = e[2], i = e[3], a = e[4], o = e[5], s = e[6], c = e[7], l = e[8], u = e[9], d = e[10], f = e[11], p = e[12], m = e[13], h = e[14], g = e[15], _ = t * o - n * a, v = t * s - r * a, y = t * c - i * a, b = n * s - r * o, x = n * c - i * o, S = r * c - i * s, C = l * m - u * p, w = l * h - d * p, T = l * g - f * p, E = u * h - d * m, D = u * g - f * m, O = d * g - f * h, k = _ * O - v * D + y * E + b * T - x * w + S * C;
		if (k === 0) return this.set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		let A = 1 / k;
		return e[0] = (o * O - s * D + c * E) * A, e[1] = (r * D - n * O - i * E) * A, e[2] = (m * S - h * x + g * b) * A, e[3] = (d * x - u * S - f * b) * A, e[4] = (s * T - a * O - c * w) * A, e[5] = (t * O - r * T + i * w) * A, e[6] = (h * y - p * S - g * v) * A, e[7] = (l * S - d * y + f * v) * A, e[8] = (a * D - o * T + c * C) * A, e[9] = (n * T - t * D - i * C) * A, e[10] = (p * x - m * y + g * _) * A, e[11] = (u * y - l * x - f * _) * A, e[12] = (o * w - a * E - s * C) * A, e[13] = (t * E - n * w + r * C) * A, e[14] = (m * v - p * b - h * _) * A, e[15] = (l * b - u * v + d * _) * A, this;
	}
	scale(e) {
		let t = this.elements, n = e.x, r = e.y, i = e.z;
		return t[0] *= n, t[4] *= r, t[8] *= i, t[1] *= n, t[5] *= r, t[9] *= i, t[2] *= n, t[6] *= r, t[10] *= i, t[3] *= n, t[7] *= r, t[11] *= i, this;
	}
	getMaxScaleOnAxis() {
		let e = this.elements, t = e[0] * e[0] + e[1] * e[1] + e[2] * e[2], n = e[4] * e[4] + e[5] * e[5] + e[6] * e[6], r = e[8] * e[8] + e[9] * e[9] + e[10] * e[10];
		return Math.sqrt(Math.max(t, n, r));
	}
	makeTranslation(e, t, n) {
		return e.isVector3 ? this.set(1, 0, 0, e.x, 0, 1, 0, e.y, 0, 0, 1, e.z, 0, 0, 0, 1) : this.set(1, 0, 0, e, 0, 1, 0, t, 0, 0, 1, n, 0, 0, 0, 1), this;
	}
	makeRotationX(e) {
		let t = Math.cos(e), n = Math.sin(e);
		return this.set(1, 0, 0, 0, 0, t, -n, 0, 0, n, t, 0, 0, 0, 0, 1), this;
	}
	makeRotationY(e) {
		let t = Math.cos(e), n = Math.sin(e);
		return this.set(t, 0, n, 0, 0, 1, 0, 0, -n, 0, t, 0, 0, 0, 0, 1), this;
	}
	makeRotationZ(e) {
		let t = Math.cos(e), n = Math.sin(e);
		return this.set(t, -n, 0, 0, n, t, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), this;
	}
	makeRotationAxis(e, t) {
		let n = Math.cos(t), r = Math.sin(t), i = 1 - n, a = e.x, o = e.y, s = e.z, c = i * a, l = i * o;
		return this.set(c * a + n, c * o - r * s, c * s + r * o, 0, c * o + r * s, l * o + n, l * s - r * a, 0, c * s - r * o, l * s + r * a, i * s * s + n, 0, 0, 0, 0, 1), this;
	}
	makeScale(e, t, n) {
		return this.set(e, 0, 0, 0, 0, t, 0, 0, 0, 0, n, 0, 0, 0, 0, 1), this;
	}
	makeShear(e, t, n, r, i, a) {
		return this.set(1, n, i, 0, e, 1, a, 0, t, r, 1, 0, 0, 0, 0, 1), this;
	}
	compose(e, t, n) {
		let r = this.elements, i = t._x, a = t._y, o = t._z, s = t._w, c = i + i, l = a + a, u = o + o, d = i * c, f = i * l, p = i * u, m = a * l, h = a * u, g = o * u, _ = s * c, v = s * l, y = s * u, b = n.x, x = n.y, S = n.z;
		return r[0] = (1 - (m + g)) * b, r[1] = (f + y) * b, r[2] = (p - v) * b, r[3] = 0, r[4] = (f - y) * x, r[5] = (1 - (d + g)) * x, r[6] = (h + _) * x, r[7] = 0, r[8] = (p + v) * S, r[9] = (h - _) * S, r[10] = (1 - (d + m)) * S, r[11] = 0, r[12] = e.x, r[13] = e.y, r[14] = e.z, r[15] = 1, this;
	}
	decompose(e, t, n) {
		let r = this.elements;
		e.x = r[12], e.y = r[13], e.z = r[14];
		let i = this.determinantAffine();
		if (i === 0) return n.set(1, 1, 1), t.identity(), this;
		let a = Yf.set(r[0], r[1], r[2]).length(), o = Yf.set(r[4], r[5], r[6]).length(), s = Yf.set(r[8], r[9], r[10]).length();
		i < 0 && (a = -a), Xf.copy(this);
		let c = 1 / a, l = 1 / o, u = 1 / s;
		return Xf.elements[0] *= c, Xf.elements[1] *= c, Xf.elements[2] *= c, Xf.elements[4] *= l, Xf.elements[5] *= l, Xf.elements[6] *= l, Xf.elements[8] *= u, Xf.elements[9] *= u, Xf.elements[10] *= u, t.setFromRotationMatrix(Xf), n.x = a, n.y = o, n.z = s, this;
	}
	makePerspective(e, t, n, r, i, a, o = Bd, s = !1) {
		let c = this.elements, l = 2 * i / (t - e), u = 2 * i / (n - r), d = (t + e) / (t - e), f = (n + r) / (n - r), p, m;
		if (s) p = i / (a - i), m = a * i / (a - i);
		else if (o === 2e3) p = -(a + i) / (a - i), m = -2 * a * i / (a - i);
		else if (o === 2001) p = -a / (a - i), m = -a * i / (a - i);
		else throw Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: " + o);
		return c[0] = l, c[4] = 0, c[8] = d, c[12] = 0, c[1] = 0, c[5] = u, c[9] = f, c[13] = 0, c[2] = 0, c[6] = 0, c[10] = p, c[14] = m, c[3] = 0, c[7] = 0, c[11] = -1, c[15] = 0, this;
	}
	makeOrthographic(e, t, n, r, i, a, o = Bd, s = !1) {
		let c = this.elements, l = 2 / (t - e), u = 2 / (n - r), d = -(t + e) / (t - e), f = -(n + r) / (n - r), p, m;
		if (s) p = 1 / (a - i), m = a / (a - i);
		else if (o === 2e3) p = -2 / (a - i), m = -(a + i) / (a - i);
		else if (o === 2001) p = -1 / (a - i), m = -i / (a - i);
		else throw Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: " + o);
		return c[0] = l, c[4] = 0, c[8] = 0, c[12] = d, c[1] = 0, c[5] = u, c[9] = 0, c[13] = f, c[2] = 0, c[6] = 0, c[10] = p, c[14] = m, c[3] = 0, c[7] = 0, c[11] = 0, c[15] = 1, this;
	}
	equals(e) {
		let t = this.elements, n = e.elements;
		for (let e = 0; e < 16; e++) if (t[e] !== n[e]) return !1;
		return !0;
	}
	fromArray(e, t = 0) {
		for (let n = 0; n < 16; n++) this.elements[n] = e[n + t];
		return this;
	}
	toArray(e = [], t = 0) {
		let n = this.elements;
		return e[t] = n[0], e[t + 1] = n[1], e[t + 2] = n[2], e[t + 3] = n[3], e[t + 4] = n[4], e[t + 5] = n[5], e[t + 6] = n[6], e[t + 7] = n[7], e[t + 8] = n[8], e[t + 9] = n[9], e[t + 10] = n[10], e[t + 11] = n[11], e[t + 12] = n[12], e[t + 13] = n[13], e[t + 14] = n[14], e[t + 15] = n[15], e;
	}
}, Yf = /*@__PURE__*/ new J(), Xf = /*@__PURE__*/ new Jf(), Zf = /*@__PURE__*/ new J(0, 0, 0), Qf = /*@__PURE__*/ new J(1, 1, 1), $f = /*@__PURE__*/ new J(), ep = /*@__PURE__*/ new J(), tp = /*@__PURE__*/ new J(), np = /*@__PURE__*/ new Jf(), rp = /*@__PURE__*/ new Ef(), ip = class e {
	constructor(t = 0, n = 0, r = 0, i = e.DEFAULT_ORDER) {
		this.isEuler = !0, this._x = t, this._y = n, this._z = r, this._order = i;
	}
	get x() {
		return this._x;
	}
	set x(e) {
		this._x = e, this._onChangeCallback();
	}
	get y() {
		return this._y;
	}
	set y(e) {
		this._y = e, this._onChangeCallback();
	}
	get z() {
		return this._z;
	}
	set z(e) {
		this._z = e, this._onChangeCallback();
	}
	get order() {
		return this._order;
	}
	set order(e) {
		this._order = e, this._onChangeCallback();
	}
	set(e, t, n, r = this._order) {
		return this._x = e, this._y = t, this._z = n, this._order = r, this._onChangeCallback(), this;
	}
	clone() {
		return new this.constructor(this._x, this._y, this._z, this._order);
	}
	copy(e) {
		return this._x = e._x, this._y = e._y, this._z = e._z, this._order = e._order, this._onChangeCallback(), this;
	}
	setFromRotationMatrix(e, t = this._order, n = !0) {
		let r = e.elements, i = r[0], a = r[4], o = r[8], s = r[1], c = r[5], l = r[9], u = r[2], d = r[6], f = r[10];
		switch (t) {
			case "XYZ":
				this._y = Math.asin(q(o, -1, 1)), Math.abs(o) < .9999999 ? (this._x = Math.atan2(-l, f), this._z = Math.atan2(-a, i)) : (this._x = Math.atan2(d, c), this._z = 0);
				break;
			case "YXZ":
				this._x = Math.asin(-q(l, -1, 1)), Math.abs(l) < .9999999 ? (this._y = Math.atan2(o, f), this._z = Math.atan2(s, c)) : (this._y = Math.atan2(-u, i), this._z = 0);
				break;
			case "ZXY":
				this._x = Math.asin(q(d, -1, 1)), Math.abs(d) < .9999999 ? (this._y = Math.atan2(-u, f), this._z = Math.atan2(-a, c)) : (this._y = 0, this._z = Math.atan2(s, i));
				break;
			case "ZYX":
				this._y = Math.asin(-q(u, -1, 1)), Math.abs(u) < .9999999 ? (this._x = Math.atan2(d, f), this._z = Math.atan2(s, i)) : (this._x = 0, this._z = Math.atan2(-a, c));
				break;
			case "YZX":
				this._z = Math.asin(q(s, -1, 1)), Math.abs(s) < .9999999 ? (this._x = Math.atan2(-l, c), this._y = Math.atan2(-u, i)) : (this._x = 0, this._y = Math.atan2(o, f));
				break;
			case "XZY":
				this._z = Math.asin(-q(a, -1, 1)), Math.abs(a) < .9999999 ? (this._x = Math.atan2(d, c), this._y = Math.atan2(o, i)) : (this._x = Math.atan2(-l, f), this._y = 0);
				break;
			default: G("Euler: .setFromRotationMatrix() encountered an unknown order: " + t);
		}
		return this._order = t, n === !0 && this._onChangeCallback(), this;
	}
	setFromQuaternion(e, t, n) {
		return np.makeRotationFromQuaternion(e), this.setFromRotationMatrix(np, t, n);
	}
	setFromVector3(e, t = this._order) {
		return this.set(e.x, e.y, e.z, t);
	}
	reorder(e) {
		return rp.setFromEuler(this), this.setFromQuaternion(rp, e);
	}
	equals(e) {
		return e._x === this._x && e._y === this._y && e._z === this._z && e._order === this._order;
	}
	fromArray(e) {
		return this._x = e[0], this._y = e[1], this._z = e[2], e[3] !== void 0 && (this._order = e[3]), this._onChangeCallback(), this;
	}
	toArray(e = [], t = 0) {
		return e[t] = this._x, e[t + 1] = this._y, e[t + 2] = this._z, e[t + 3] = this._order, e;
	}
	_onChange(e) {
		return this._onChangeCallback = e, this;
	}
	_onChangeCallback() {}
	*[Symbol.iterator]() {
		yield this._x, yield this._y, yield this._z, yield this._order;
	}
};
ip.DEFAULT_ORDER = "XYZ";
var ap = class {
	constructor() {
		this.mask = 1;
	}
	set(e) {
		this.mask = (1 << e | 0) >>> 0;
	}
	enable(e) {
		this.mask |= 1 << e | 0;
	}
	enableAll() {
		this.mask = -1;
	}
	toggle(e) {
		this.mask ^= 1 << e | 0;
	}
	disable(e) {
		this.mask &= ~(1 << e | 0);
	}
	disableAll() {
		this.mask = 0;
	}
	test(e) {
		return (this.mask & e.mask) !== 0;
	}
	isEnabled(e) {
		return (this.mask & (1 << e | 0)) != 0;
	}
}, op = 0, sp = /*@__PURE__*/ new J(), cp = /*@__PURE__*/ new Ef(), lp = /*@__PURE__*/ new Jf(), up = /*@__PURE__*/ new J(), dp = /*@__PURE__*/ new J(), fp = /*@__PURE__*/ new J(), pp = /*@__PURE__*/ new Ef(), mp = /*@__PURE__*/ new J(1, 0, 0), hp = /*@__PURE__*/ new J(0, 1, 0), gp = /*@__PURE__*/ new J(0, 0, 1), _p = { type: "added" }, vp = { type: "removed" }, yp = {
	type: "childadded",
	child: null
}, bp = {
	type: "childremoved",
	child: null
}, xp = class e extends Zd {
	constructor() {
		super(), this.isObject3D = !0, Object.defineProperty(this, "id", { value: op++ }), this.uuid = nf(), this.name = "", this.type = "Object3D", this.parent = null, this.children = [], this.up = e.DEFAULT_UP.clone();
		let t = new J(), n = new ip(), r = new Ef(), i = new J(1, 1, 1);
		function a() {
			r.setFromEuler(n, !1);
		}
		function o() {
			n.setFromQuaternion(r, void 0, !1);
		}
		n._onChange(a), r._onChange(o), Object.defineProperties(this, {
			position: {
				configurable: !0,
				enumerable: !0,
				value: t
			},
			rotation: {
				configurable: !0,
				enumerable: !0,
				value: n
			},
			quaternion: {
				configurable: !0,
				enumerable: !0,
				value: r
			},
			scale: {
				configurable: !0,
				enumerable: !0,
				value: i
			},
			modelViewMatrix: { value: new Jf() },
			normalMatrix: { value: new Y() }
		}), this.matrix = new Jf(), this.matrixWorld = new Jf(), this.matrixAutoUpdate = e.DEFAULT_MATRIX_AUTO_UPDATE, this.matrixWorldAutoUpdate = e.DEFAULT_MATRIX_WORLD_AUTO_UPDATE, this.matrixWorldNeedsUpdate = !1, this.layers = new ap(), this.visible = !0, this.castShadow = !1, this.receiveShadow = !1, this.frustumCulled = !0, this.renderOrder = 0, this.animations = [], this.customDepthMaterial = void 0, this.customDistanceMaterial = void 0, this.static = !1, this.userData = {}, this.pivot = null;
	}
	onBeforeShadow() {}
	onAfterShadow() {}
	onBeforeRender() {}
	onAfterRender() {}
	applyMatrix4(e) {
		this.matrixAutoUpdate && this.updateMatrix(), this.matrix.premultiply(e), this.matrix.decompose(this.position, this.quaternion, this.scale);
	}
	applyQuaternion(e) {
		return this.quaternion.premultiply(e), this;
	}
	setRotationFromAxisAngle(e, t) {
		this.quaternion.setFromAxisAngle(e, t);
	}
	setRotationFromEuler(e) {
		this.quaternion.setFromEuler(e, !0);
	}
	setRotationFromMatrix(e) {
		this.quaternion.setFromRotationMatrix(e);
	}
	setRotationFromQuaternion(e) {
		this.quaternion.copy(e);
	}
	rotateOnAxis(e, t) {
		return cp.setFromAxisAngle(e, t), this.quaternion.multiply(cp), this;
	}
	rotateOnWorldAxis(e, t) {
		return cp.setFromAxisAngle(e, t), this.quaternion.premultiply(cp), this;
	}
	rotateX(e) {
		return this.rotateOnAxis(mp, e);
	}
	rotateY(e) {
		return this.rotateOnAxis(hp, e);
	}
	rotateZ(e) {
		return this.rotateOnAxis(gp, e);
	}
	translateOnAxis(e, t) {
		return sp.copy(e).applyQuaternion(this.quaternion), this.position.add(sp.multiplyScalar(t)), this;
	}
	translateX(e) {
		return this.translateOnAxis(mp, e);
	}
	translateY(e) {
		return this.translateOnAxis(hp, e);
	}
	translateZ(e) {
		return this.translateOnAxis(gp, e);
	}
	localToWorld(e) {
		return this.updateWorldMatrix(!0, !1), e.applyMatrix4(this.matrixWorld);
	}
	worldToLocal(e) {
		return this.updateWorldMatrix(!0, !1), e.applyMatrix4(lp.copy(this.matrixWorld).invert());
	}
	lookAt(e, t, n) {
		e.isVector3 ? up.copy(e) : up.set(e, t, n);
		let r = this.parent;
		this.updateWorldMatrix(!0, !1), dp.setFromMatrixPosition(this.matrixWorld), this.isCamera || this.isLight ? lp.lookAt(dp, up, this.up) : lp.lookAt(up, dp, this.up), this.quaternion.setFromRotationMatrix(lp), r && (lp.extractRotation(r.matrixWorld), cp.setFromRotationMatrix(lp), this.quaternion.premultiply(cp.invert()));
	}
	add(e) {
		if (arguments.length > 1) {
			for (let e = 0; e < arguments.length; e++) this.add(arguments[e]);
			return this;
		}
		return e === this ? (K("Object3D.add: object can't be added as a child of itself.", e), this) : (e && e.isObject3D ? (e.removeFromParent(), e.parent = this, this.children.push(e), e.dispatchEvent(_p), yp.child = e, this.dispatchEvent(yp), yp.child = null) : K("Object3D.add: object not an instance of THREE.Object3D.", e), this);
	}
	remove(e) {
		if (arguments.length > 1) {
			for (let e = 0; e < arguments.length; e++) this.remove(arguments[e]);
			return this;
		}
		let t = this.children.indexOf(e);
		return t !== -1 && (e.parent = null, this.children.splice(t, 1), e.dispatchEvent(vp), bp.child = e, this.dispatchEvent(bp), bp.child = null), this;
	}
	removeFromParent() {
		let e = this.parent;
		return e !== null && e.remove(this), this;
	}
	clear() {
		return this.remove(...this.children);
	}
	attach(e) {
		return this.updateWorldMatrix(!0, !1), lp.copy(this.matrixWorld).invert(), e.parent !== null && (e.parent.updateWorldMatrix(!0, !1), lp.multiply(e.parent.matrixWorld)), e.applyMatrix4(lp), e.removeFromParent(), e.parent = this, this.children.push(e), e.updateWorldMatrix(!1, !0), e.dispatchEvent(_p), yp.child = e, this.dispatchEvent(yp), yp.child = null, this;
	}
	getObjectById(e) {
		return this.getObjectByProperty("id", e);
	}
	getObjectByName(e) {
		return this.getObjectByProperty("name", e);
	}
	getObjectByProperty(e, t) {
		if (this[e] === t) return this;
		for (let n = 0, r = this.children.length; n < r; n++) {
			let r = this.children[n].getObjectByProperty(e, t);
			if (r !== void 0) return r;
		}
	}
	getObjectsByProperty(e, t, n = []) {
		this[e] === t && n.push(this);
		let r = this.children;
		for (let i = 0, a = r.length; i < a; i++) r[i].getObjectsByProperty(e, t, n);
		return n;
	}
	getWorldPosition(e) {
		return this.updateWorldMatrix(!0, !1), e.setFromMatrixPosition(this.matrixWorld);
	}
	getWorldQuaternion(e) {
		return this.updateWorldMatrix(!0, !1), this.matrixWorld.decompose(dp, e, fp), e;
	}
	getWorldScale(e) {
		return this.updateWorldMatrix(!0, !1), this.matrixWorld.decompose(dp, pp, e), e;
	}
	getWorldDirection(e) {
		this.updateWorldMatrix(!0, !1);
		let t = this.matrixWorld.elements;
		return e.set(t[8], t[9], t[10]).normalize();
	}
	raycast() {}
	traverse(e) {
		e(this);
		let t = this.children;
		for (let n = 0, r = t.length; n < r; n++) t[n].traverse(e);
	}
	traverseVisible(e) {
		if (this.visible === !1) return;
		e(this);
		let t = this.children;
		for (let n = 0, r = t.length; n < r; n++) t[n].traverseVisible(e);
	}
	traverseAncestors(e) {
		let t = this.parent;
		t !== null && (e(t), t.traverseAncestors(e));
	}
	updateMatrix() {
		this.matrix.compose(this.position, this.quaternion, this.scale);
		let e = this.pivot;
		if (e !== null) {
			let t = e.x, n = e.y, r = e.z, i = this.matrix.elements;
			i[12] += t - i[0] * t - i[4] * n - i[8] * r, i[13] += n - i[1] * t - i[5] * n - i[9] * r, i[14] += r - i[2] * t - i[6] * n - i[10] * r;
		}
		this.matrixWorldNeedsUpdate = !0;
	}
	updateMatrixWorld(e) {
		this.matrixAutoUpdate && this.updateMatrix(), (this.matrixWorldNeedsUpdate || e) && (this.matrixWorldAutoUpdate === !0 && (this.parent === null ? this.matrixWorld.copy(this.matrix) : this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix)), this.matrixWorldNeedsUpdate = !1, e = !0);
		let t = this.children;
		for (let n = 0, r = t.length; n < r; n++) t[n].updateMatrixWorld(e);
	}
	updateWorldMatrix(e, t, n = !1) {
		let r = this.parent;
		if (e === !0 && r !== null && r.updateWorldMatrix(!0, !1), this.matrixAutoUpdate && this.updateMatrix(), (this.matrixWorldNeedsUpdate || n) && (this.matrixWorldAutoUpdate === !0 && (this.parent === null ? this.matrixWorld.copy(this.matrix) : this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix)), this.matrixWorldNeedsUpdate = !1, n = !0), t === !0) {
			let e = this.children;
			for (let t = 0, r = e.length; t < r; t++) e[t].updateWorldMatrix(!1, !0, n);
		}
	}
	toJSON(e) {
		let t = e === void 0 || typeof e == "string", n = {};
		t && (e = {
			geometries: {},
			materials: {},
			textures: {},
			images: {},
			shapes: {},
			skeletons: {},
			animations: {},
			nodes: {}
		}, n.metadata = {
			version: 4.7,
			type: "Object",
			generator: "Object3D.toJSON"
		});
		let r = {};
		r.uuid = this.uuid, r.type = this.type, this.name !== "" && (r.name = this.name), this.castShadow === !0 && (r.castShadow = !0), this.receiveShadow === !0 && (r.receiveShadow = !0), this.visible === !1 && (r.visible = !1), this.frustumCulled === !1 && (r.frustumCulled = !1), this.renderOrder !== 0 && (r.renderOrder = this.renderOrder), this.static !== !1 && (r.static = this.static), Object.keys(this.userData).length > 0 && (r.userData = this.userData), r.layers = this.layers.mask, r.matrix = this.matrix.toArray(), r.up = this.up.toArray(), this.pivot !== null && (r.pivot = this.pivot.toArray()), this.matrixAutoUpdate === !1 && (r.matrixAutoUpdate = !1), this.morphTargetDictionary !== void 0 && (r.morphTargetDictionary = Object.assign({}, this.morphTargetDictionary)), this.morphTargetInfluences !== void 0 && (r.morphTargetInfluences = this.morphTargetInfluences.slice()), this.isInstancedMesh && (r.type = "InstancedMesh", r.count = this.count, r.instanceMatrix = this.instanceMatrix.toJSON(), this.instanceColor !== null && (r.instanceColor = this.instanceColor.toJSON())), this.isBatchedMesh && (r.type = "BatchedMesh", r.perObjectFrustumCulled = this.perObjectFrustumCulled, r.sortObjects = this.sortObjects, r.drawRanges = this._drawRanges, r.reservedRanges = this._reservedRanges, r.geometryInfo = this._geometryInfo.map((e) => ({
			...e,
			boundingBox: e.boundingBox ? e.boundingBox.toJSON() : void 0,
			boundingSphere: e.boundingSphere ? e.boundingSphere.toJSON() : void 0
		})), r.instanceInfo = this._instanceInfo.map((e) => ({ ...e })), r.availableInstanceIds = this._availableInstanceIds.slice(), r.availableGeometryIds = this._availableGeometryIds.slice(), r.nextIndexStart = this._nextIndexStart, r.nextVertexStart = this._nextVertexStart, r.geometryCount = this._geometryCount, r.maxInstanceCount = this._maxInstanceCount, r.maxVertexCount = this._maxVertexCount, r.maxIndexCount = this._maxIndexCount, r.geometryInitialized = this._geometryInitialized, r.matricesTexture = this._matricesTexture.toJSON(e), r.indirectTexture = this._indirectTexture.toJSON(e), this._colorsTexture !== null && (r.colorsTexture = this._colorsTexture.toJSON(e)), this.boundingSphere !== null && (r.boundingSphere = this.boundingSphere.toJSON()), this.boundingBox !== null && (r.boundingBox = this.boundingBox.toJSON()));
		function i(t, n) {
			return t[n.uuid] === void 0 && (t[n.uuid] = n.toJSON(e)), n.uuid;
		}
		if (this.isScene) this.background && (this.background.isColor ? r.background = this.background.toJSON() : this.background.isTexture && (r.background = this.background.toJSON(e).uuid)), this.environment && this.environment.isTexture && this.environment.isRenderTargetTexture !== !0 && (r.environment = this.environment.toJSON(e).uuid);
		else if (this.isMesh || this.isLine || this.isPoints) {
			r.geometry = i(e.geometries, this.geometry);
			let t = this.geometry.parameters;
			if (t !== void 0 && t.shapes !== void 0) {
				let n = t.shapes;
				if (Array.isArray(n)) for (let t = 0, r = n.length; t < r; t++) {
					let r = n[t];
					i(e.shapes, r);
				}
				else i(e.shapes, n);
			}
		}
		if (this.isSkinnedMesh && (r.bindMode = this.bindMode, r.bindMatrix = this.bindMatrix.toArray(), this.skeleton !== void 0 && (i(e.skeletons, this.skeleton), r.skeleton = this.skeleton.uuid)), this.material !== void 0) if (Array.isArray(this.material)) {
			let t = [];
			for (let n = 0, r = this.material.length; n < r; n++) t.push(i(e.materials, this.material[n]));
			r.material = t;
		} else r.material = i(e.materials, this.material);
		if (this.children.length > 0) {
			r.children = [];
			for (let t = 0; t < this.children.length; t++) r.children.push(this.children[t].toJSON(e).object);
		}
		if (this.animations.length > 0) {
			r.animations = [];
			for (let t = 0; t < this.animations.length; t++) {
				let n = this.animations[t];
				r.animations.push(i(e.animations, n));
			}
		}
		if (t) {
			let t = a(e.geometries), r = a(e.materials), i = a(e.textures), o = a(e.images), s = a(e.shapes), c = a(e.skeletons), l = a(e.animations), u = a(e.nodes);
			t.length > 0 && (n.geometries = t), r.length > 0 && (n.materials = r), i.length > 0 && (n.textures = i), o.length > 0 && (n.images = o), s.length > 0 && (n.shapes = s), c.length > 0 && (n.skeletons = c), l.length > 0 && (n.animations = l), u.length > 0 && (n.nodes = u);
		}
		return n.object = r, n;
		function a(e) {
			let t = [];
			for (let n in e) {
				let r = e[n];
				delete r.metadata, t.push(r);
			}
			return t;
		}
	}
	clone(e) {
		return new this.constructor().copy(this, e);
	}
	copy(e, t = !0) {
		if (this.name = e.name, this.up.copy(e.up), this.position.copy(e.position), this.rotation.order = e.rotation.order, this.quaternion.copy(e.quaternion), this.scale.copy(e.scale), this.pivot = e.pivot === null ? null : e.pivot.clone(), this.matrix.copy(e.matrix), this.matrixWorld.copy(e.matrixWorld), this.matrixAutoUpdate = e.matrixAutoUpdate, this.matrixWorldAutoUpdate = e.matrixWorldAutoUpdate, this.matrixWorldNeedsUpdate = e.matrixWorldNeedsUpdate, this.layers.mask = e.layers.mask, this.visible = e.visible, this.castShadow = e.castShadow, this.receiveShadow = e.receiveShadow, this.frustumCulled = e.frustumCulled, this.renderOrder = e.renderOrder, this.static = e.static, this.animations = e.animations.slice(), this.userData = JSON.parse(JSON.stringify(e.userData)), t === !0) for (let t = 0; t < e.children.length; t++) {
			let n = e.children[t];
			this.add(n.clone());
		}
		return this;
	}
};
xp.DEFAULT_UP = /*@__PURE__*/ new J(0, 1, 0), xp.DEFAULT_MATRIX_AUTO_UPDATE = !0, xp.DEFAULT_MATRIX_WORLD_AUTO_UPDATE = !0;
var Sp = class extends xp {
	constructor() {
		super(), this.isGroup = !0, this.type = "Group";
	}
}, Cp = { type: "move" }, wp = class {
	constructor() {
		this._targetRay = null, this._grip = null, this._hand = null;
	}
	getHandSpace() {
		return this._hand === null && (this._hand = new Sp(), this._hand.matrixAutoUpdate = !1, this._hand.visible = !1, this._hand.joints = {}, this._hand.inputState = { pinching: !1 }), this._hand;
	}
	getTargetRaySpace() {
		return this._targetRay === null && (this._targetRay = new Sp(), this._targetRay.matrixAutoUpdate = !1, this._targetRay.visible = !1, this._targetRay.hasLinearVelocity = !1, this._targetRay.linearVelocity = new J(), this._targetRay.hasAngularVelocity = !1, this._targetRay.angularVelocity = new J()), this._targetRay;
	}
	getGripSpace() {
		return this._grip === null && (this._grip = new Sp(), this._grip.matrixAutoUpdate = !1, this._grip.visible = !1, this._grip.hasLinearVelocity = !1, this._grip.linearVelocity = new J(), this._grip.hasAngularVelocity = !1, this._grip.angularVelocity = new J(), this._grip.eventsEnabled = !1), this._grip;
	}
	dispatchEvent(e) {
		return this._targetRay !== null && this._targetRay.dispatchEvent(e), this._grip !== null && this._grip.dispatchEvent(e), this._hand !== null && this._hand.dispatchEvent(e), this;
	}
	connect(e) {
		if (e && e.hand) {
			let t = this._hand;
			if (t) for (let n of e.hand.values()) this._getHandJoint(t, n);
		}
		return this.dispatchEvent({
			type: "connected",
			data: e
		}), this;
	}
	disconnect(e) {
		return this.dispatchEvent({
			type: "disconnected",
			data: e
		}), this._targetRay !== null && (this._targetRay.visible = !1), this._grip !== null && (this._grip.visible = !1), this._hand !== null && (this._hand.visible = !1), this;
	}
	update(e, t, n) {
		let r = null, i = null, a = null, o = this._targetRay, s = this._grip, c = this._hand;
		if (e && t.session.visibilityState !== "visible-blurred") {
			if (c && e.hand) {
				a = !0;
				for (let r of e.hand.values()) {
					let e = t.getJointPose(r, n), i = this._getHandJoint(c, r);
					e !== null && (i.matrix.fromArray(e.transform.matrix), i.matrix.decompose(i.position, i.rotation, i.scale), i.matrixWorldNeedsUpdate = !0, i.jointRadius = e.radius), i.visible = e !== null;
				}
				let r = c.joints["index-finger-tip"], i = c.joints["thumb-tip"], o = r.position.distanceTo(i.position);
				c.inputState.pinching && o > .025 ? (c.inputState.pinching = !1, this.dispatchEvent({
					type: "pinchend",
					handedness: e.handedness,
					target: this
				})) : !c.inputState.pinching && o <= .015 && (c.inputState.pinching = !0, this.dispatchEvent({
					type: "pinchstart",
					handedness: e.handedness,
					target: this
				}));
			} else s !== null && e.gripSpace && (i = t.getPose(e.gripSpace, n), i !== null && (s.matrix.fromArray(i.transform.matrix), s.matrix.decompose(s.position, s.rotation, s.scale), s.matrixWorldNeedsUpdate = !0, i.linearVelocity ? (s.hasLinearVelocity = !0, s.linearVelocity.copy(i.linearVelocity)) : s.hasLinearVelocity = !1, i.angularVelocity ? (s.hasAngularVelocity = !0, s.angularVelocity.copy(i.angularVelocity)) : s.hasAngularVelocity = !1, s.eventsEnabled && s.dispatchEvent({
				type: "gripUpdated",
				data: e,
				target: this
			})));
			o !== null && (r = t.getPose(e.targetRaySpace, n), r === null && i !== null && (r = i), r !== null && (o.matrix.fromArray(r.transform.matrix), o.matrix.decompose(o.position, o.rotation, o.scale), o.matrixWorldNeedsUpdate = !0, r.linearVelocity ? (o.hasLinearVelocity = !0, o.linearVelocity.copy(r.linearVelocity)) : o.hasLinearVelocity = !1, r.angularVelocity ? (o.hasAngularVelocity = !0, o.angularVelocity.copy(r.angularVelocity)) : o.hasAngularVelocity = !1, this.dispatchEvent(Cp)));
		}
		return o !== null && (o.visible = r !== null), s !== null && (s.visible = i !== null), c !== null && (c.visible = a !== null), this;
	}
	_getHandJoint(e, t) {
		if (e.joints[t.jointName] === void 0) {
			let n = new Sp();
			n.matrixAutoUpdate = !1, n.visible = !1, e.joints[t.jointName] = n, e.add(n);
		}
		return e.joints[t.jointName];
	}
}, Tp = {
	aliceblue: 15792383,
	antiquewhite: 16444375,
	aqua: 65535,
	aquamarine: 8388564,
	azure: 15794175,
	beige: 16119260,
	bisque: 16770244,
	black: 0,
	blanchedalmond: 16772045,
	blue: 255,
	blueviolet: 9055202,
	brown: 10824234,
	burlywood: 14596231,
	cadetblue: 6266528,
	chartreuse: 8388352,
	chocolate: 13789470,
	coral: 16744272,
	cornflowerblue: 6591981,
	cornsilk: 16775388,
	crimson: 14423100,
	cyan: 65535,
	darkblue: 139,
	darkcyan: 35723,
	darkgoldenrod: 12092939,
	darkgray: 11119017,
	darkgreen: 25600,
	darkgrey: 11119017,
	darkkhaki: 12433259,
	darkmagenta: 9109643,
	darkolivegreen: 5597999,
	darkorange: 16747520,
	darkorchid: 10040012,
	darkred: 9109504,
	darksalmon: 15308410,
	darkseagreen: 9419919,
	darkslateblue: 4734347,
	darkslategray: 3100495,
	darkslategrey: 3100495,
	darkturquoise: 52945,
	darkviolet: 9699539,
	deeppink: 16716947,
	deepskyblue: 49151,
	dimgray: 6908265,
	dimgrey: 6908265,
	dodgerblue: 2003199,
	firebrick: 11674146,
	floralwhite: 16775920,
	forestgreen: 2263842,
	fuchsia: 16711935,
	gainsboro: 14474460,
	ghostwhite: 16316671,
	gold: 16766720,
	goldenrod: 14329120,
	gray: 8421504,
	green: 32768,
	greenyellow: 11403055,
	grey: 8421504,
	honeydew: 15794160,
	hotpink: 16738740,
	indianred: 13458524,
	indigo: 4915330,
	ivory: 16777200,
	khaki: 15787660,
	lavender: 15132410,
	lavenderblush: 16773365,
	lawngreen: 8190976,
	lemonchiffon: 16775885,
	lightblue: 11393254,
	lightcoral: 15761536,
	lightcyan: 14745599,
	lightgoldenrodyellow: 16448210,
	lightgray: 13882323,
	lightgreen: 9498256,
	lightgrey: 13882323,
	lightpink: 16758465,
	lightsalmon: 16752762,
	lightseagreen: 2142890,
	lightskyblue: 8900346,
	lightslategray: 7833753,
	lightslategrey: 7833753,
	lightsteelblue: 11584734,
	lightyellow: 16777184,
	lime: 65280,
	limegreen: 3329330,
	linen: 16445670,
	magenta: 16711935,
	maroon: 8388608,
	mediumaquamarine: 6737322,
	mediumblue: 205,
	mediumorchid: 12211667,
	mediumpurple: 9662683,
	mediumseagreen: 3978097,
	mediumslateblue: 8087790,
	mediumspringgreen: 64154,
	mediumturquoise: 4772300,
	mediumvioletred: 13047173,
	midnightblue: 1644912,
	mintcream: 16121850,
	mistyrose: 16770273,
	moccasin: 16770229,
	navajowhite: 16768685,
	navy: 128,
	oldlace: 16643558,
	olive: 8421376,
	olivedrab: 7048739,
	orange: 16753920,
	orangered: 16729344,
	orchid: 14315734,
	palegoldenrod: 15657130,
	palegreen: 10025880,
	paleturquoise: 11529966,
	palevioletred: 14381203,
	papayawhip: 16773077,
	peachpuff: 16767673,
	peru: 13468991,
	pink: 16761035,
	plum: 14524637,
	powderblue: 11591910,
	purple: 8388736,
	rebeccapurple: 6697881,
	red: 16711680,
	rosybrown: 12357519,
	royalblue: 4286945,
	saddlebrown: 9127187,
	salmon: 16416882,
	sandybrown: 16032864,
	seagreen: 3050327,
	seashell: 16774638,
	sienna: 10506797,
	silver: 12632256,
	skyblue: 8900331,
	slateblue: 6970061,
	slategray: 7372944,
	slategrey: 7372944,
	snow: 16775930,
	springgreen: 65407,
	steelblue: 4620980,
	tan: 13808780,
	teal: 32896,
	thistle: 14204888,
	tomato: 16737095,
	turquoise: 4251856,
	violet: 15631086,
	wheat: 16113331,
	white: 16777215,
	whitesmoke: 16119285,
	yellow: 16776960,
	yellowgreen: 10145074
}, Ep = {
	h: 0,
	s: 0,
	l: 0
}, Dp = {
	h: 0,
	s: 0,
	l: 0
};
function Op(e, t, n) {
	return n < 0 && (n += 1), n > 1 && --n, n < 1 / 6 ? e + (t - e) * 6 * n : n < 1 / 2 ? t : n < 2 / 3 ? e + (t - e) * 6 * (2 / 3 - n) : e;
}
var Z = class {
	constructor(e, t, n) {
		return this.isColor = !0, this.r = 1, this.g = 1, this.b = 1, this.set(e, t, n);
	}
	set(e, t, n) {
		if (t === void 0 && n === void 0) {
			let t = e;
			t && t.isColor ? this.copy(t) : typeof t == "number" ? this.setHex(t) : typeof t == "string" && this.setStyle(t);
		} else this.setRGB(e, t, n);
		return this;
	}
	setScalar(e) {
		return this.r = e, this.g = e, this.b = e, this;
	}
	setHex(e, t = Nd) {
		return e = Math.floor(e), this.r = (e >> 16 & 255) / 255, this.g = (e >> 8 & 255) / 255, this.b = (e & 255) / 255, X.colorSpaceToWorking(this, t), this;
	}
	setRGB(e, t, n, r = X.workingColorSpace) {
		return this.r = e, this.g = t, this.b = n, X.colorSpaceToWorking(this, r), this;
	}
	setHSL(e, t, n, r = X.workingColorSpace) {
		if (e = rf(e, 1), t = q(t, 0, 1), n = q(n, 0, 1), t === 0) this.r = this.g = this.b = n;
		else {
			let r = n <= .5 ? n * (1 + t) : n + t - n * t, i = 2 * n - r;
			this.r = Op(i, r, e + 1 / 3), this.g = Op(i, r, e), this.b = Op(i, r, e - 1 / 3);
		}
		return X.colorSpaceToWorking(this, r), this;
	}
	setStyle(e, t = Nd) {
		function n(t) {
			t !== void 0 && parseFloat(t) < 1 && G("Color: Alpha component of " + e + " will be ignored.");
		}
		let r;
		if (r = /^(\w+)\(([^\)]*)\)/.exec(e)) {
			let i, a = r[1], o = r[2];
			switch (a) {
				case "rgb":
				case "rgba":
					if (i = /^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o)) return n(i[4]), this.setRGB(Math.min(255, parseInt(i[1], 10)) / 255, Math.min(255, parseInt(i[2], 10)) / 255, Math.min(255, parseInt(i[3], 10)) / 255, t);
					if (i = /^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o)) return n(i[4]), this.setRGB(Math.min(100, parseInt(i[1], 10)) / 100, Math.min(100, parseInt(i[2], 10)) / 100, Math.min(100, parseInt(i[3], 10)) / 100, t);
					break;
				case "hsl":
				case "hsla":
					if (i = /^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o)) return n(i[4]), this.setHSL(parseFloat(i[1]) / 360, parseFloat(i[2]) / 100, parseFloat(i[3]) / 100, t);
					break;
				default: G("Color: Unknown color model " + e);
			}
		} else if (r = /^\#([A-Fa-f\d]+)$/.exec(e)) {
			let n = r[1], i = n.length;
			if (i === 3) return this.setRGB(parseInt(n.charAt(0), 16) / 15, parseInt(n.charAt(1), 16) / 15, parseInt(n.charAt(2), 16) / 15, t);
			if (i === 6) return this.setHex(parseInt(n, 16), t);
			G("Color: Invalid hex color " + e);
		} else if (e && e.length > 0) return this.setColorName(e, t);
		return this;
	}
	setColorName(e, t = Nd) {
		let n = Tp[e.toLowerCase()];
		return n === void 0 ? G("Color: Unknown color " + e) : this.setHex(n, t), this;
	}
	clone() {
		return new this.constructor(this.r, this.g, this.b);
	}
	copy(e) {
		return this.r = e.r, this.g = e.g, this.b = e.b, this;
	}
	copySRGBToLinear(e) {
		return this.r = Nf(e.r), this.g = Nf(e.g), this.b = Nf(e.b), this;
	}
	copyLinearToSRGB(e) {
		return this.r = Pf(e.r), this.g = Pf(e.g), this.b = Pf(e.b), this;
	}
	convertSRGBToLinear() {
		return this.copySRGBToLinear(this), this;
	}
	convertLinearToSRGB() {
		return this.copyLinearToSRGB(this), this;
	}
	getHex(e = Nd) {
		return X.workingToColorSpace(kp.copy(this), e), Math.round(q(kp.r * 255, 0, 255)) * 65536 + Math.round(q(kp.g * 255, 0, 255)) * 256 + Math.round(q(kp.b * 255, 0, 255));
	}
	getHexString(e = Nd) {
		return ("000000" + this.getHex(e).toString(16)).slice(-6);
	}
	getHSL(e, t = X.workingColorSpace) {
		X.workingToColorSpace(kp.copy(this), t);
		let n = kp.r, r = kp.g, i = kp.b, a = Math.max(n, r, i), o = Math.min(n, r, i), s, c, l = (o + a) / 2;
		if (o === a) s = 0, c = 0;
		else {
			let e = a - o;
			switch (c = l <= .5 ? e / (a + o) : e / (2 - a - o), a) {
				case n:
					s = (r - i) / e + (r < i ? 6 : 0);
					break;
				case r:
					s = (i - n) / e + 2;
					break;
				case i:
					s = (n - r) / e + 4;
					break;
			}
			s /= 6;
		}
		return e.h = s, e.s = c, e.l = l, e;
	}
	getRGB(e, t = X.workingColorSpace) {
		return X.workingToColorSpace(kp.copy(this), t), e.r = kp.r, e.g = kp.g, e.b = kp.b, e;
	}
	getStyle(e = Nd) {
		X.workingToColorSpace(kp.copy(this), e);
		let t = kp.r, n = kp.g, r = kp.b;
		return e === "srgb" ? `rgb(${Math.round(t * 255)},${Math.round(n * 255)},${Math.round(r * 255)})` : `color(${e} ${t.toFixed(3)} ${n.toFixed(3)} ${r.toFixed(3)})`;
	}
	offsetHSL(e, t, n) {
		return this.getHSL(Ep), this.setHSL(Ep.h + e, Ep.s + t, Ep.l + n);
	}
	add(e) {
		return this.r += e.r, this.g += e.g, this.b += e.b, this;
	}
	addColors(e, t) {
		return this.r = e.r + t.r, this.g = e.g + t.g, this.b = e.b + t.b, this;
	}
	addScalar(e) {
		return this.r += e, this.g += e, this.b += e, this;
	}
	sub(e) {
		return this.r = Math.max(0, this.r - e.r), this.g = Math.max(0, this.g - e.g), this.b = Math.max(0, this.b - e.b), this;
	}
	multiply(e) {
		return this.r *= e.r, this.g *= e.g, this.b *= e.b, this;
	}
	multiplyScalar(e) {
		return this.r *= e, this.g *= e, this.b *= e, this;
	}
	lerp(e, t) {
		return this.r += (e.r - this.r) * t, this.g += (e.g - this.g) * t, this.b += (e.b - this.b) * t, this;
	}
	lerpColors(e, t, n) {
		return this.r = e.r + (t.r - e.r) * n, this.g = e.g + (t.g - e.g) * n, this.b = e.b + (t.b - e.b) * n, this;
	}
	lerpHSL(e, t) {
		this.getHSL(Ep), e.getHSL(Dp);
		let n = sf(Ep.h, Dp.h, t), r = sf(Ep.s, Dp.s, t), i = sf(Ep.l, Dp.l, t);
		return this.setHSL(n, r, i), this;
	}
	setFromVector3(e) {
		return this.r = e.x, this.g = e.y, this.b = e.z, this;
	}
	applyMatrix3(e) {
		let t = this.r, n = this.g, r = this.b, i = e.elements;
		return this.r = i[0] * t + i[3] * n + i[6] * r, this.g = i[1] * t + i[4] * n + i[7] * r, this.b = i[2] * t + i[5] * n + i[8] * r, this;
	}
	equals(e) {
		return e.r === this.r && e.g === this.g && e.b === this.b;
	}
	fromArray(e, t = 0) {
		return this.r = e[t], this.g = e[t + 1], this.b = e[t + 2], this;
	}
	toArray(e = [], t = 0) {
		return e[t] = this.r, e[t + 1] = this.g, e[t + 2] = this.b, e;
	}
	fromBufferAttribute(e, t) {
		return this.r = e.getX(t), this.g = e.getY(t), this.b = e.getZ(t), this;
	}
	toJSON() {
		return this.getHex();
	}
	*[Symbol.iterator]() {
		yield this.r, yield this.g, yield this.b;
	}
}, kp = /*@__PURE__*/ new Z();
Z.NAMES = Tp;
var Ap = class extends xp {
	constructor() {
		super(), this.isScene = !0, this.type = "Scene", this.background = null, this.environment = null, this.fog = null, this.backgroundBlurriness = 0, this.backgroundIntensity = 1, this.backgroundRotation = new ip(), this.environmentIntensity = 1, this.environmentRotation = new ip(), this.overrideMaterial = null, typeof __THREE_DEVTOOLS__ < "u" && __THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe", { detail: this }));
	}
	copy(e, t) {
		return super.copy(e, t), e.background !== null && (this.background = e.background.clone()), e.environment !== null && (this.environment = e.environment.clone()), e.fog !== null && (this.fog = e.fog.clone()), this.backgroundBlurriness = e.backgroundBlurriness, this.backgroundIntensity = e.backgroundIntensity, this.backgroundRotation.copy(e.backgroundRotation), this.environmentIntensity = e.environmentIntensity, this.environmentRotation.copy(e.environmentRotation), e.overrideMaterial !== null && (this.overrideMaterial = e.overrideMaterial.clone()), this.matrixAutoUpdate = e.matrixAutoUpdate, this;
	}
	toJSON(e) {
		let t = super.toJSON(e);
		return this.fog !== null && (t.object.fog = this.fog.toJSON()), this.backgroundBlurriness > 0 && (t.object.backgroundBlurriness = this.backgroundBlurriness), this.backgroundIntensity !== 1 && (t.object.backgroundIntensity = this.backgroundIntensity), t.object.backgroundRotation = this.backgroundRotation.toArray(), this.environmentIntensity !== 1 && (t.object.environmentIntensity = this.environmentIntensity), t.object.environmentRotation = this.environmentRotation.toArray(), t;
	}
}, jp = /*@__PURE__*/ new J(), Mp = /*@__PURE__*/ new J(), Np = /*@__PURE__*/ new J(), Pp = /*@__PURE__*/ new J(), Fp = /*@__PURE__*/ new J(), Ip = /*@__PURE__*/ new J(), Lp = /*@__PURE__*/ new J(), Rp = /*@__PURE__*/ new J(), zp = /*@__PURE__*/ new J(), Bp = /*@__PURE__*/ new J(), Vp = /*@__PURE__*/ new Uf(), Hp = /*@__PURE__*/ new Uf(), Up = /*@__PURE__*/ new Uf(), Wp = class e {
	constructor(e = new J(), t = new J(), n = new J()) {
		this.a = e, this.b = t, this.c = n;
	}
	static getNormal(e, t, n, r) {
		r.subVectors(n, t), jp.subVectors(e, t), r.cross(jp);
		let i = r.lengthSq();
		return i > 0 ? r.multiplyScalar(1 / Math.sqrt(i)) : r.set(0, 0, 0);
	}
	static getBarycoord(e, t, n, r, i) {
		jp.subVectors(r, t), Mp.subVectors(n, t), Np.subVectors(e, t);
		let a = jp.dot(jp), o = jp.dot(Mp), s = jp.dot(Np), c = Mp.dot(Mp), l = Mp.dot(Np), u = a * c - o * o;
		if (u === 0) return i.set(0, 0, 0), null;
		let d = 1 / u, f = (c * s - o * l) * d, p = (a * l - o * s) * d;
		return i.set(1 - f - p, p, f);
	}
	static containsPoint(e, t, n, r) {
		return this.getBarycoord(e, t, n, r, Pp) !== null && Pp.x >= 0 && Pp.y >= 0 && Pp.x + Pp.y <= 1;
	}
	static getInterpolation(e, t, n, r, i, a, o, s) {
		return this.getBarycoord(e, t, n, r, Pp) === null ? (s.x = 0, s.y = 0, "z" in s && (s.z = 0), "w" in s && (s.w = 0), null) : (s.setScalar(0), s.addScaledVector(i, Pp.x), s.addScaledVector(a, Pp.y), s.addScaledVector(o, Pp.z), s);
	}
	static getInterpolatedAttribute(e, t, n, r, i, a) {
		return Vp.setScalar(0), Hp.setScalar(0), Up.setScalar(0), Vp.fromBufferAttribute(e, t), Hp.fromBufferAttribute(e, n), Up.fromBufferAttribute(e, r), a.setScalar(0), a.addScaledVector(Vp, i.x), a.addScaledVector(Hp, i.y), a.addScaledVector(Up, i.z), a;
	}
	static isFrontFacing(e, t, n, r) {
		return jp.subVectors(n, t), Mp.subVectors(e, t), jp.cross(Mp).dot(r) < 0;
	}
	set(e, t, n) {
		return this.a.copy(e), this.b.copy(t), this.c.copy(n), this;
	}
	setFromPointsAndIndices(e, t, n, r) {
		return this.a.copy(e[t]), this.b.copy(e[n]), this.c.copy(e[r]), this;
	}
	setFromAttributeAndIndices(e, t, n, r) {
		return this.a.fromBufferAttribute(e, t), this.b.fromBufferAttribute(e, n), this.c.fromBufferAttribute(e, r), this;
	}
	clone() {
		return new this.constructor().copy(this);
	}
	copy(e) {
		return this.a.copy(e.a), this.b.copy(e.b), this.c.copy(e.c), this;
	}
	getArea() {
		return jp.subVectors(this.c, this.b), Mp.subVectors(this.a, this.b), jp.cross(Mp).length() * .5;
	}
	getMidpoint(e) {
		return e.addVectors(this.a, this.b).add(this.c).multiplyScalar(1 / 3);
	}
	getNormal(t) {
		return e.getNormal(this.a, this.b, this.c, t);
	}
	getPlane(e) {
		return e.setFromCoplanarPoints(this.a, this.b, this.c);
	}
	getBarycoord(t, n) {
		return e.getBarycoord(t, this.a, this.b, this.c, n);
	}
	getInterpolation(t, n, r, i, a) {
		return e.getInterpolation(t, this.a, this.b, this.c, n, r, i, a);
	}
	containsPoint(t) {
		return e.containsPoint(t, this.a, this.b, this.c);
	}
	isFrontFacing(t) {
		return e.isFrontFacing(this.a, this.b, this.c, t);
	}
	intersectsBox(e) {
		return e.intersectsTriangle(this);
	}
	closestPointToPoint(e, t) {
		let n = this.a, r = this.b, i = this.c, a, o;
		Fp.subVectors(r, n), Ip.subVectors(i, n), Rp.subVectors(e, n);
		let s = Fp.dot(Rp), c = Ip.dot(Rp);
		if (s <= 0 && c <= 0) return t.copy(n);
		zp.subVectors(e, r);
		let l = Fp.dot(zp), u = Ip.dot(zp);
		if (l >= 0 && u <= l) return t.copy(r);
		let d = s * u - l * c;
		if (d <= 0 && s >= 0 && l <= 0) return a = s / (s - l), t.copy(n).addScaledVector(Fp, a);
		Bp.subVectors(e, i);
		let f = Fp.dot(Bp), p = Ip.dot(Bp);
		if (p >= 0 && f <= p) return t.copy(i);
		let m = f * c - s * p;
		if (m <= 0 && c >= 0 && p <= 0) return o = c / (c - p), t.copy(n).addScaledVector(Ip, o);
		let h = l * p - f * u;
		if (h <= 0 && u - l >= 0 && f - p >= 0) return Lp.subVectors(i, r), o = (u - l) / (u - l + (f - p)), t.copy(r).addScaledVector(Lp, o);
		let g = 1 / (h + m + d);
		return a = m * g, o = d * g, t.copy(n).addScaledVector(Fp, a).addScaledVector(Ip, o);
	}
	equals(e) {
		return e.a.equals(this.a) && e.b.equals(this.b) && e.c.equals(this.c);
	}
}, Gp = class {
	constructor(e = new J(Infinity, Infinity, Infinity), t = new J(-Infinity, -Infinity, -Infinity)) {
		this.isBox3 = !0, this.min = e, this.max = t;
	}
	set(e, t) {
		return this.min.copy(e), this.max.copy(t), this;
	}
	setFromArray(e) {
		this.makeEmpty();
		for (let t = 0, n = e.length; t < n; t += 3) this.expandByPoint(qp.fromArray(e, t));
		return this;
	}
	setFromBufferAttribute(e) {
		this.makeEmpty();
		for (let t = 0, n = e.count; t < n; t++) this.expandByPoint(qp.fromBufferAttribute(e, t));
		return this;
	}
	setFromPoints(e) {
		this.makeEmpty();
		for (let t = 0, n = e.length; t < n; t++) this.expandByPoint(e[t]);
		return this;
	}
	setFromCenterAndSize(e, t) {
		let n = qp.copy(t).multiplyScalar(.5);
		return this.min.copy(e).sub(n), this.max.copy(e).add(n), this;
	}
	setFromObject(e, t = !1) {
		return this.makeEmpty(), this.expandByObject(e, t);
	}
	clone() {
		return new this.constructor().copy(this);
	}
	copy(e) {
		return this.min.copy(e.min), this.max.copy(e.max), this;
	}
	makeEmpty() {
		return this.min.x = this.min.y = this.min.z = Infinity, this.max.x = this.max.y = this.max.z = -Infinity, this;
	}
	isEmpty() {
		return this.max.x < this.min.x || this.max.y < this.min.y || this.max.z < this.min.z;
	}
	getCenter(e) {
		return this.isEmpty() ? e.set(0, 0, 0) : e.addVectors(this.min, this.max).multiplyScalar(.5);
	}
	getSize(e) {
		return this.isEmpty() ? e.set(0, 0, 0) : e.subVectors(this.max, this.min);
	}
	expandByPoint(e) {
		return this.min.min(e), this.max.max(e), this;
	}
	expandByVector(e) {
		return this.min.sub(e), this.max.add(e), this;
	}
	expandByScalar(e) {
		return this.min.addScalar(-e), this.max.addScalar(e), this;
	}
	expandByObject(e, t = !1) {
		e.updateWorldMatrix(!1, !1);
		let n = e.geometry;
		if (n !== void 0) {
			let r = n.getAttribute("position");
			if (t === !0 && r !== void 0 && e.isInstancedMesh !== !0) for (let t = 0, n = r.count; t < n; t++) e.isMesh === !0 ? e.getVertexPosition(t, qp) : qp.fromBufferAttribute(r, t), qp.applyMatrix4(e.matrixWorld), this.expandByPoint(qp);
			else e.boundingBox === void 0 ? (n.boundingBox === null && n.computeBoundingBox(), Jp.copy(n.boundingBox)) : (e.boundingBox === null && e.computeBoundingBox(), Jp.copy(e.boundingBox)), Jp.applyMatrix4(e.matrixWorld), this.union(Jp);
		}
		let r = e.children;
		for (let e = 0, n = r.length; e < n; e++) this.expandByObject(r[e], t);
		return this;
	}
	containsPoint(e) {
		return e.x >= this.min.x && e.x <= this.max.x && e.y >= this.min.y && e.y <= this.max.y && e.z >= this.min.z && e.z <= this.max.z;
	}
	containsBox(e) {
		return this.min.x <= e.min.x && e.max.x <= this.max.x && this.min.y <= e.min.y && e.max.y <= this.max.y && this.min.z <= e.min.z && e.max.z <= this.max.z;
	}
	getParameter(e, t) {
		return t.set((e.x - this.min.x) / (this.max.x - this.min.x), (e.y - this.min.y) / (this.max.y - this.min.y), (e.z - this.min.z) / (this.max.z - this.min.z));
	}
	intersectsBox(e) {
		return e.max.x >= this.min.x && e.min.x <= this.max.x && e.max.y >= this.min.y && e.min.y <= this.max.y && e.max.z >= this.min.z && e.min.z <= this.max.z;
	}
	intersectsSphere(e) {
		return this.clampPoint(e.center, qp), qp.distanceToSquared(e.center) <= e.radius * e.radius;
	}
	intersectsPlane(e) {
		let t, n;
		return e.normal.x > 0 ? (t = e.normal.x * this.min.x, n = e.normal.x * this.max.x) : (t = e.normal.x * this.max.x, n = e.normal.x * this.min.x), e.normal.y > 0 ? (t += e.normal.y * this.min.y, n += e.normal.y * this.max.y) : (t += e.normal.y * this.max.y, n += e.normal.y * this.min.y), e.normal.z > 0 ? (t += e.normal.z * this.min.z, n += e.normal.z * this.max.z) : (t += e.normal.z * this.max.z, n += e.normal.z * this.min.z), t <= -e.constant && n >= -e.constant;
	}
	intersectsTriangle(e) {
		if (this.isEmpty()) return !1;
		this.getCenter(tm), nm.subVectors(this.max, tm), Yp.subVectors(e.a, tm), Xp.subVectors(e.b, tm), Zp.subVectors(e.c, tm), Qp.subVectors(Xp, Yp), $p.subVectors(Zp, Xp), em.subVectors(Yp, Zp);
		let t = [
			0,
			-Qp.z,
			Qp.y,
			0,
			-$p.z,
			$p.y,
			0,
			-em.z,
			em.y,
			Qp.z,
			0,
			-Qp.x,
			$p.z,
			0,
			-$p.x,
			em.z,
			0,
			-em.x,
			-Qp.y,
			Qp.x,
			0,
			-$p.y,
			$p.x,
			0,
			-em.y,
			em.x,
			0
		];
		return !am(t, Yp, Xp, Zp, nm) || (t = [
			1,
			0,
			0,
			0,
			1,
			0,
			0,
			0,
			1
		], !am(t, Yp, Xp, Zp, nm)) ? !1 : (rm.crossVectors(Qp, $p), t = [
			rm.x,
			rm.y,
			rm.z
		], am(t, Yp, Xp, Zp, nm));
	}
	clampPoint(e, t) {
		return t.copy(e).clamp(this.min, this.max);
	}
	distanceToPoint(e) {
		return this.clampPoint(e, qp).distanceTo(e);
	}
	getBoundingSphere(e) {
		return this.isEmpty() ? e.makeEmpty() : (this.getCenter(e.center), e.radius = this.getSize(qp).length() * .5), e;
	}
	intersect(e) {
		return this.min.max(e.min), this.max.min(e.max), this.isEmpty() && this.makeEmpty(), this;
	}
	union(e) {
		return this.min.min(e.min), this.max.max(e.max), this;
	}
	applyMatrix4(e) {
		return this.isEmpty() ? this : (Kp[0].set(this.min.x, this.min.y, this.min.z).applyMatrix4(e), Kp[1].set(this.min.x, this.min.y, this.max.z).applyMatrix4(e), Kp[2].set(this.min.x, this.max.y, this.min.z).applyMatrix4(e), Kp[3].set(this.min.x, this.max.y, this.max.z).applyMatrix4(e), Kp[4].set(this.max.x, this.min.y, this.min.z).applyMatrix4(e), Kp[5].set(this.max.x, this.min.y, this.max.z).applyMatrix4(e), Kp[6].set(this.max.x, this.max.y, this.min.z).applyMatrix4(e), Kp[7].set(this.max.x, this.max.y, this.max.z).applyMatrix4(e), this.setFromPoints(Kp), this);
	}
	translate(e) {
		return this.min.add(e), this.max.add(e), this;
	}
	equals(e) {
		return e.min.equals(this.min) && e.max.equals(this.max);
	}
	toJSON() {
		return {
			min: this.min.toArray(),
			max: this.max.toArray()
		};
	}
	fromJSON(e) {
		return this.min.fromArray(e.min), this.max.fromArray(e.max), this;
	}
}, Kp = [
	/*@__PURE__*/ new J(),
	/*@__PURE__*/ new J(),
	/*@__PURE__*/ new J(),
	/*@__PURE__*/ new J(),
	/*@__PURE__*/ new J(),
	/*@__PURE__*/ new J(),
	/*@__PURE__*/ new J(),
	/*@__PURE__*/ new J()
], qp = /*@__PURE__*/ new J(), Jp = /*@__PURE__*/ new Gp(), Yp = /*@__PURE__*/ new J(), Xp = /*@__PURE__*/ new J(), Zp = /*@__PURE__*/ new J(), Qp = /*@__PURE__*/ new J(), $p = /*@__PURE__*/ new J(), em = /*@__PURE__*/ new J(), tm = /*@__PURE__*/ new J(), nm = /*@__PURE__*/ new J(), rm = /*@__PURE__*/ new J(), im = /*@__PURE__*/ new J();
function am(e, t, n, r, i) {
	for (let a = 0, o = e.length - 3; a <= o; a += 3) {
		im.fromArray(e, a);
		let o = i.x * Math.abs(im.x) + i.y * Math.abs(im.y) + i.z * Math.abs(im.z), s = t.dot(im), c = n.dot(im), l = r.dot(im);
		if (Math.max(-Math.max(s, c, l), Math.min(s, c, l)) > o) return !1;
	}
	return !0;
}
var om = /*@__PURE__*/ new J(), sm = /*@__PURE__*/ new Tf(), cm = 0, lm = class extends Zd {
	constructor(e, t, n = !1) {
		if (super(), Array.isArray(e)) throw TypeError("THREE.BufferAttribute: array should be a Typed Array.");
		this.isBufferAttribute = !0, Object.defineProperty(this, "id", { value: cm++ }), this.name = "", this.array = e, this.itemSize = t, this.count = e === void 0 ? 0 : e.length / t, this.normalized = n, this.usage = Rd, this.updateRanges = [], this.gpuType = Eu, this.version = 0;
	}
	onUploadCallback() {}
	set needsUpdate(e) {
		e === !0 && this.version++;
	}
	setUsage(e) {
		return this.usage = e, this;
	}
	addUpdateRange(e, t) {
		this.updateRanges.push({
			start: e,
			count: t
		});
	}
	clearUpdateRanges() {
		this.updateRanges.length = 0;
	}
	copy(e) {
		return this.name = e.name, this.array = new e.array.constructor(e.array), this.itemSize = e.itemSize, this.count = e.count, this.normalized = e.normalized, this.usage = e.usage, this.gpuType = e.gpuType, this;
	}
	copyAt(e, t, n) {
		e *= this.itemSize, n *= t.itemSize;
		for (let r = 0, i = this.itemSize; r < i; r++) this.array[e + r] = t.array[n + r];
		return this;
	}
	copyArray(e) {
		return this.array.set(e), this;
	}
	applyMatrix3(e) {
		if (this.itemSize === 2) for (let t = 0, n = this.count; t < n; t++) sm.fromBufferAttribute(this, t), sm.applyMatrix3(e), this.setXY(t, sm.x, sm.y);
		else if (this.itemSize === 3) for (let t = 0, n = this.count; t < n; t++) om.fromBufferAttribute(this, t), om.applyMatrix3(e), this.setXYZ(t, om.x, om.y, om.z);
		return this;
	}
	applyMatrix4(e) {
		for (let t = 0, n = this.count; t < n; t++) om.fromBufferAttribute(this, t), om.applyMatrix4(e), this.setXYZ(t, om.x, om.y, om.z);
		return this;
	}
	applyNormalMatrix(e) {
		for (let t = 0, n = this.count; t < n; t++) om.fromBufferAttribute(this, t), om.applyNormalMatrix(e), this.setXYZ(t, om.x, om.y, om.z);
		return this;
	}
	transformDirection(e) {
		for (let t = 0, n = this.count; t < n; t++) om.fromBufferAttribute(this, t), om.transformDirection(e), this.setXYZ(t, om.x, om.y, om.z);
		return this;
	}
	set(e, t = 0) {
		return this.array.set(e, t), this;
	}
	getComponent(e, t) {
		let n = this.array[e * this.itemSize + t];
		return this.normalized && (n = Sf(n, this.array)), n;
	}
	setComponent(e, t, n) {
		return this.normalized && (n = Cf(n, this.array)), this.array[e * this.itemSize + t] = n, this;
	}
	getX(e) {
		let t = this.array[e * this.itemSize];
		return this.normalized && (t = Sf(t, this.array)), t;
	}
	setX(e, t) {
		return this.normalized && (t = Cf(t, this.array)), this.array[e * this.itemSize] = t, this;
	}
	getY(e) {
		let t = this.array[e * this.itemSize + 1];
		return this.normalized && (t = Sf(t, this.array)), t;
	}
	setY(e, t) {
		return this.normalized && (t = Cf(t, this.array)), this.array[e * this.itemSize + 1] = t, this;
	}
	getZ(e) {
		let t = this.array[e * this.itemSize + 2];
		return this.normalized && (t = Sf(t, this.array)), t;
	}
	setZ(e, t) {
		return this.normalized && (t = Cf(t, this.array)), this.array[e * this.itemSize + 2] = t, this;
	}
	getW(e) {
		let t = this.array[e * this.itemSize + 3];
		return this.normalized && (t = Sf(t, this.array)), t;
	}
	setW(e, t) {
		return this.normalized && (t = Cf(t, this.array)), this.array[e * this.itemSize + 3] = t, this;
	}
	setXY(e, t, n) {
		return e *= this.itemSize, this.normalized && (t = Cf(t, this.array), n = Cf(n, this.array)), this.array[e + 0] = t, this.array[e + 1] = n, this;
	}
	setXYZ(e, t, n, r) {
		return e *= this.itemSize, this.normalized && (t = Cf(t, this.array), n = Cf(n, this.array), r = Cf(r, this.array)), this.array[e + 0] = t, this.array[e + 1] = n, this.array[e + 2] = r, this;
	}
	setXYZW(e, t, n, r, i) {
		return e *= this.itemSize, this.normalized && (t = Cf(t, this.array), n = Cf(n, this.array), r = Cf(r, this.array), i = Cf(i, this.array)), this.array[e + 0] = t, this.array[e + 1] = n, this.array[e + 2] = r, this.array[e + 3] = i, this;
	}
	onUpload(e) {
		return this.onUploadCallback = e, this;
	}
	clone() {
		return new this.constructor(this.array, this.itemSize).copy(this);
	}
	toJSON() {
		let e = {
			itemSize: this.itemSize,
			type: this.array.constructor.name,
			array: Array.from(this.array),
			normalized: this.normalized
		};
		return this.name !== "" && (e.name = this.name), this.usage !== 35044 && (e.usage = this.usage), e;
	}
	dispose() {
		this.dispatchEvent({ type: "dispose" });
	}
}, um = class extends lm {
	constructor(e, t, n) {
		super(new Uint16Array(e), t, n);
	}
}, dm = class extends lm {
	constructor(e, t, n) {
		super(new Uint32Array(e), t, n);
	}
}, fm = class extends lm {
	constructor(e, t, n) {
		super(new Float32Array(e), t, n);
	}
}, pm = /*@__PURE__*/ new Gp(), mm = /*@__PURE__*/ new J(), hm = /*@__PURE__*/ new J(), gm = class {
	constructor(e = new J(), t = -1) {
		this.isSphere = !0, this.center = e, this.radius = t;
	}
	set(e, t) {
		return this.center.copy(e), this.radius = t, this;
	}
	setFromPoints(e, t) {
		let n = this.center;
		t === void 0 ? pm.setFromPoints(e).getCenter(n) : n.copy(t);
		let r = 0;
		for (let t = 0, i = e.length; t < i; t++) r = Math.max(r, n.distanceToSquared(e[t]));
		return this.radius = Math.sqrt(r), this;
	}
	copy(e) {
		return this.center.copy(e.center), this.radius = e.radius, this;
	}
	isEmpty() {
		return this.radius < 0;
	}
	makeEmpty() {
		return this.center.set(0, 0, 0), this.radius = -1, this;
	}
	containsPoint(e) {
		return e.distanceToSquared(this.center) <= this.radius * this.radius;
	}
	distanceToPoint(e) {
		return e.distanceTo(this.center) - this.radius;
	}
	intersectsSphere(e) {
		let t = this.radius + e.radius;
		return e.center.distanceToSquared(this.center) <= t * t;
	}
	intersectsBox(e) {
		return e.intersectsSphere(this);
	}
	intersectsPlane(e) {
		return Math.abs(e.distanceToPoint(this.center)) <= this.radius;
	}
	clampPoint(e, t) {
		let n = this.center.distanceToSquared(e);
		return t.copy(e), n > this.radius * this.radius && (t.sub(this.center).normalize(), t.multiplyScalar(this.radius).add(this.center)), t;
	}
	getBoundingBox(e) {
		return this.isEmpty() ? (e.makeEmpty(), e) : (e.set(this.center, this.center), e.expandByScalar(this.radius), e);
	}
	applyMatrix4(e) {
		return this.center.applyMatrix4(e), this.radius *= e.getMaxScaleOnAxis(), this;
	}
	translate(e) {
		return this.center.add(e), this;
	}
	expandByPoint(e) {
		if (this.isEmpty()) return this.center.copy(e), this.radius = 0, this;
		mm.subVectors(e, this.center);
		let t = mm.lengthSq();
		if (t > this.radius * this.radius) {
			let e = Math.sqrt(t), n = (e - this.radius) * .5;
			this.center.addScaledVector(mm, n / e), this.radius += n;
		}
		return this;
	}
	union(e) {
		return e.isEmpty() ? this : this.isEmpty() ? (this.copy(e), this) : (this.center.equals(e.center) === !0 ? this.radius = Math.max(this.radius, e.radius) : (hm.subVectors(e.center, this.center).setLength(e.radius), this.expandByPoint(mm.copy(e.center).add(hm)), this.expandByPoint(mm.copy(e.center).sub(hm))), this);
	}
	equals(e) {
		return e.center.equals(this.center) && e.radius === this.radius;
	}
	clone() {
		return new this.constructor().copy(this);
	}
	toJSON() {
		return {
			radius: this.radius,
			center: this.center.toArray()
		};
	}
	fromJSON(e) {
		return this.radius = e.radius, this.center.fromArray(e.center), this;
	}
}, _m = 0, vm = /*@__PURE__*/ new Jf(), ym = /*@__PURE__*/ new xp(), bm = /*@__PURE__*/ new J(), xm = /*@__PURE__*/ new Gp(), Sm = /*@__PURE__*/ new Gp(), Cm = /*@__PURE__*/ new J(), wm = class e extends Zd {
	constructor() {
		super(), this.isBufferGeometry = !0, Object.defineProperty(this, "id", { value: _m++ }), this.uuid = nf(), this.name = "", this.type = "BufferGeometry", this.index = null, this.indirect = null, this.indirectOffset = 0, this.attributes = {}, this.morphAttributes = {}, this.morphTargetsRelative = !1, this.groups = [], this.boundingBox = null, this.boundingSphere = null, this.drawRange = {
			start: 0,
			count: Infinity
		}, this.userData = {}, this._transformed = !1;
	}
	getIndex() {
		return this.index;
	}
	setIndex(e) {
		return Array.isArray(e) ? this.index = new (Vd(e) ? dm : um)(e, 1) : this.index = e, this;
	}
	setIndirect(e, t = 0) {
		return this.indirect = e, this.indirectOffset = t, this;
	}
	getIndirect() {
		return this.indirect;
	}
	getAttribute(e) {
		return this.attributes[e];
	}
	setAttribute(e, t) {
		return this.attributes[e] = t, this;
	}
	deleteAttribute(e) {
		return delete this.attributes[e], this;
	}
	hasAttribute(e) {
		return this.attributes[e] !== void 0;
	}
	addGroup(e, t, n = 0) {
		this.groups.push({
			start: e,
			count: t,
			materialIndex: n
		});
	}
	clearGroups() {
		this.groups = [];
	}
	setDrawRange(e, t) {
		this.drawRange.start = e, this.drawRange.count = t;
	}
	applyMatrix4(e) {
		let t = this.attributes.position;
		t !== void 0 && (t.applyMatrix4(e), t.needsUpdate = !0);
		let n = this.attributes.normal;
		if (n !== void 0) {
			let t = new Y().getNormalMatrix(e);
			n.applyNormalMatrix(t), n.needsUpdate = !0;
		}
		let r = this.attributes.tangent;
		return r !== void 0 && (r.transformDirection(e), r.needsUpdate = !0), this.boundingBox !== null && this.computeBoundingBox(), this.boundingSphere !== null && this.computeBoundingSphere(), this._transformed = !0, this;
	}
	applyQuaternion(e) {
		return vm.makeRotationFromQuaternion(e), this.applyMatrix4(vm), this;
	}
	rotateX(e) {
		return vm.makeRotationX(e), this.applyMatrix4(vm), this;
	}
	rotateY(e) {
		return vm.makeRotationY(e), this.applyMatrix4(vm), this;
	}
	rotateZ(e) {
		return vm.makeRotationZ(e), this.applyMatrix4(vm), this;
	}
	translate(e, t, n) {
		return vm.makeTranslation(e, t, n), this.applyMatrix4(vm), this;
	}
	scale(e, t, n) {
		return vm.makeScale(e, t, n), this.applyMatrix4(vm), this;
	}
	lookAt(e) {
		return ym.lookAt(e), ym.updateMatrix(), this.applyMatrix4(ym.matrix), this;
	}
	center() {
		return this.computeBoundingBox(), this.boundingBox.getCenter(bm).negate(), this.translate(bm.x, bm.y, bm.z), this;
	}
	setFromPoints(e) {
		let t = this.getAttribute("position");
		if (t === void 0) {
			let t = [];
			for (let n = 0, r = e.length; n < r; n++) {
				let r = e[n];
				t.push(r.x, r.y, r.z || 0);
			}
			this.setAttribute("position", new fm(t, 3));
		} else {
			let n = Math.min(e.length, t.count);
			for (let r = 0; r < n; r++) {
				let n = e[r];
				t.setXYZ(r, n.x, n.y, n.z || 0);
			}
			e.length > t.count && G("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."), t.needsUpdate = !0;
		}
		return this;
	}
	computeBoundingBox() {
		this.boundingBox === null && (this.boundingBox = new Gp());
		let e = this.attributes.position, t = this.morphAttributes.position;
		if (e && e.isGLBufferAttribute) {
			K("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.", this), this.boundingBox.set(new J(-Infinity, -Infinity, -Infinity), new J(Infinity, Infinity, Infinity));
			return;
		}
		if (e !== void 0) {
			if (this.boundingBox.setFromBufferAttribute(e), t) for (let e = 0, n = t.length; e < n; e++) {
				let n = t[e];
				xm.setFromBufferAttribute(n), this.morphTargetsRelative ? (Cm.addVectors(this.boundingBox.min, xm.min), this.boundingBox.expandByPoint(Cm), Cm.addVectors(this.boundingBox.max, xm.max), this.boundingBox.expandByPoint(Cm)) : (this.boundingBox.expandByPoint(xm.min), this.boundingBox.expandByPoint(xm.max));
			}
		} else this.boundingBox.makeEmpty();
		(isNaN(this.boundingBox.min.x) || isNaN(this.boundingBox.min.y) || isNaN(this.boundingBox.min.z)) && K("BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The \"position\" attribute is likely to have NaN values.", this);
	}
	computeBoundingSphere() {
		this.boundingSphere === null && (this.boundingSphere = new gm());
		let e = this.attributes.position, t = this.morphAttributes.position;
		if (e && e.isGLBufferAttribute) {
			K("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.", this), this.boundingSphere.set(new J(), Infinity);
			return;
		}
		if (e) {
			let n = this.boundingSphere.center;
			if (xm.setFromBufferAttribute(e), t) for (let e = 0, n = t.length; e < n; e++) {
				let n = t[e];
				Sm.setFromBufferAttribute(n), this.morphTargetsRelative ? (Cm.addVectors(xm.min, Sm.min), xm.expandByPoint(Cm), Cm.addVectors(xm.max, Sm.max), xm.expandByPoint(Cm)) : (xm.expandByPoint(Sm.min), xm.expandByPoint(Sm.max));
			}
			xm.getCenter(n);
			let r = 0;
			for (let t = 0, i = e.count; t < i; t++) Cm.fromBufferAttribute(e, t), r = Math.max(r, n.distanceToSquared(Cm));
			if (t) for (let i = 0, a = t.length; i < a; i++) {
				let a = t[i], o = this.morphTargetsRelative;
				for (let t = 0, i = a.count; t < i; t++) Cm.fromBufferAttribute(a, t), o && (bm.fromBufferAttribute(e, t), Cm.add(bm)), r = Math.max(r, n.distanceToSquared(Cm));
			}
			this.boundingSphere.radius = Math.sqrt(r), isNaN(this.boundingSphere.radius) && K("BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The \"position\" attribute is likely to have NaN values.", this);
		}
	}
	computeTangents() {
		let e = this.index, t = this.attributes;
		if (e === null || t.position === void 0 || t.normal === void 0 || t.uv === void 0) {
			K("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");
			return;
		}
		let n = t.position, r = t.normal, i = t.uv, a = this.getAttribute("tangent");
		(a === void 0 || a.count !== n.count) && (a = new lm(new Float32Array(4 * n.count), 4), this.setAttribute("tangent", a));
		let o = [], s = [];
		for (let e = 0; e < n.count; e++) o[e] = new J(), s[e] = new J();
		let c = new J(), l = new J(), u = new J(), d = new Tf(), f = new Tf(), p = new Tf(), m = new J(), h = new J();
		function g(e, t, r) {
			c.fromBufferAttribute(n, e), l.fromBufferAttribute(n, t), u.fromBufferAttribute(n, r), d.fromBufferAttribute(i, e), f.fromBufferAttribute(i, t), p.fromBufferAttribute(i, r), l.sub(c), u.sub(c), f.sub(d), p.sub(d);
			let a = 1 / (f.x * p.y - p.x * f.y);
			isFinite(a) && (m.copy(l).multiplyScalar(p.y).addScaledVector(u, -f.y).multiplyScalar(a), h.copy(u).multiplyScalar(f.x).addScaledVector(l, -p.x).multiplyScalar(a), o[e].add(m), o[t].add(m), o[r].add(m), s[e].add(h), s[t].add(h), s[r].add(h));
		}
		let _ = this.groups;
		_.length === 0 && (_ = [{
			start: 0,
			count: e.count
		}]);
		for (let t = 0, n = _.length; t < n; ++t) {
			let n = _[t], r = n.start, i = n.count;
			for (let t = r, n = r + i; t < n; t += 3) g(e.getX(t + 0), e.getX(t + 1), e.getX(t + 2));
		}
		let v = new J(), y = new J(), b = new J(), x = new J();
		function S(e) {
			b.fromBufferAttribute(r, e), x.copy(b);
			let t = o[e];
			v.copy(t), v.sub(b.multiplyScalar(b.dot(t))).normalize(), y.crossVectors(x, t);
			let n = y.dot(s[e]) < 0 ? -1 : 1;
			a.setXYZW(e, v.x, v.y, v.z, n);
		}
		for (let t = 0, n = _.length; t < n; ++t) {
			let n = _[t], r = n.start, i = n.count;
			for (let t = r, n = r + i; t < n; t += 3) S(e.getX(t + 0)), S(e.getX(t + 1)), S(e.getX(t + 2));
		}
		this._transformed = !0;
	}
	computeVertexNormals() {
		let e = this.index, t = this.getAttribute("position");
		if (t !== void 0) {
			let n = this.getAttribute("normal");
			if (n === void 0 || n.count !== t.count) n = new lm(new Float32Array(t.count * 3), 3), this.setAttribute("normal", n);
			else for (let e = 0, t = n.count; e < t; e++) n.setXYZ(e, 0, 0, 0);
			let r = new J(), i = new J(), a = new J(), o = new J(), s = new J(), c = new J(), l = new J(), u = new J();
			if (e) for (let d = 0, f = e.count; d < f; d += 3) {
				let f = e.getX(d + 0), p = e.getX(d + 1), m = e.getX(d + 2);
				r.fromBufferAttribute(t, f), i.fromBufferAttribute(t, p), a.fromBufferAttribute(t, m), l.subVectors(a, i), u.subVectors(r, i), l.cross(u), o.fromBufferAttribute(n, f), s.fromBufferAttribute(n, p), c.fromBufferAttribute(n, m), o.add(l), s.add(l), c.add(l), n.setXYZ(f, o.x, o.y, o.z), n.setXYZ(p, s.x, s.y, s.z), n.setXYZ(m, c.x, c.y, c.z);
			}
			else for (let e = 0, o = t.count; e < o; e += 3) r.fromBufferAttribute(t, e + 0), i.fromBufferAttribute(t, e + 1), a.fromBufferAttribute(t, e + 2), l.subVectors(a, i), u.subVectors(r, i), l.cross(u), n.setXYZ(e + 0, l.x, l.y, l.z), n.setXYZ(e + 1, l.x, l.y, l.z), n.setXYZ(e + 2, l.x, l.y, l.z);
			this.normalizeNormals(), n.needsUpdate = !0;
		}
	}
	normalizeNormals() {
		let e = this.attributes.normal;
		for (let t = 0, n = e.count; t < n; t++) Cm.fromBufferAttribute(e, t), Cm.normalize(), e.setXYZ(t, Cm.x, Cm.y, Cm.z);
	}
	toNonIndexed() {
		function t(e, t) {
			let n = e.array, r = e.itemSize, i = e.normalized, a = new n.constructor(t.length * r), o = 0, s = 0;
			for (let i = 0, c = t.length; i < c; i++) {
				o = e.isInterleavedBufferAttribute ? t[i] * e.data.stride + e.offset : t[i] * r;
				for (let e = 0; e < r; e++) a[s++] = n[o++];
			}
			return new lm(a, r, i);
		}
		if (this.index === null) return G("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."), this;
		let n = new e(), r = this.index.array, i = this.attributes;
		for (let e in i) {
			let a = i[e], o = t(a, r);
			n.setAttribute(e, o);
		}
		let a = this.morphAttributes;
		for (let e in a) {
			let i = [], o = a[e];
			for (let e = 0, n = o.length; e < n; e++) {
				let n = o[e], a = t(n, r);
				i.push(a);
			}
			n.morphAttributes[e] = i;
		}
		n.morphTargetsRelative = this.morphTargetsRelative;
		let o = this.groups;
		for (let e = 0, t = o.length; e < t; e++) {
			let t = o[e];
			n.addGroup(t.start, t.count, t.materialIndex);
		}
		return n;
	}
	toJSON() {
		let e = { metadata: {
			version: 4.7,
			type: "BufferGeometry",
			generator: "BufferGeometry.toJSON"
		} };
		if (e.uuid = this.uuid, e.type = this.parameters !== void 0 && this._transformed === !0 ? "BufferGeometry" : this.type, this.name !== "" && (e.name = this.name), Object.keys(this.userData).length > 0 && (e.userData = this.userData), this.parameters !== void 0 && this._transformed !== !0) {
			let t = this.parameters;
			for (let n in t) t[n] !== void 0 && (e[n] = t[n]);
			return e;
		}
		e.data = { attributes: {} };
		let t = this.index;
		t !== null && (e.data.index = {
			type: t.array.constructor.name,
			array: Array.prototype.slice.call(t.array)
		});
		let n = this.attributes;
		for (let t in n) {
			let r = n[t];
			e.data.attributes[t] = r.toJSON(e.data);
		}
		let r = {}, i = !1;
		for (let t in this.morphAttributes) {
			let n = this.morphAttributes[t], a = [];
			for (let t = 0, r = n.length; t < r; t++) {
				let r = n[t];
				a.push(r.toJSON(e.data));
			}
			a.length > 0 && (r[t] = a, i = !0);
		}
		i && (e.data.morphAttributes = r, e.data.morphTargetsRelative = this.morphTargetsRelative);
		let a = this.groups;
		a.length > 0 && (e.data.groups = JSON.parse(JSON.stringify(a)));
		let o = this.boundingSphere;
		return o !== null && (e.data.boundingSphere = o.toJSON()), e;
	}
	clone() {
		return new this.constructor().copy(this);
	}
	copy(e) {
		this.index = null, this.attributes = {}, this.morphAttributes = {}, this.groups = [], this.boundingBox = null, this.boundingSphere = null;
		let t = {};
		this.name = e.name;
		let n = e.index;
		n !== null && this.setIndex(n.clone());
		let r = e.attributes;
		for (let e in r) {
			let n = r[e];
			this.setAttribute(e, n.clone(t));
		}
		let i = e.morphAttributes;
		for (let e in i) {
			let n = [], r = i[e];
			for (let e = 0, i = r.length; e < i; e++) n.push(r[e].clone(t));
			this.morphAttributes[e] = n;
		}
		this.morphTargetsRelative = e.morphTargetsRelative;
		let a = e.groups;
		for (let e = 0, t = a.length; e < t; e++) {
			let t = a[e];
			this.addGroup(t.start, t.count, t.materialIndex);
		}
		let o = e.boundingBox;
		o !== null && (this.boundingBox = o.clone());
		let s = e.boundingSphere;
		return s !== null && (this.boundingSphere = s.clone()), this.drawRange.start = e.drawRange.start, this.drawRange.count = e.drawRange.count, this.userData = e.userData, this._transformed = e._transformed, this;
	}
	dispose() {
		this.dispatchEvent({ type: "dispose" });
	}
}, Tm = 0, Em = class extends Zd {
	constructor() {
		super(), this.isMaterial = !0, Object.defineProperty(this, "id", { value: Tm++ }), this.uuid = nf(), this.name = "", this.type = "Material", this.blending = 1, this.side = 0, this.vertexColors = !1, this.opacity = 1, this.transparent = !1, this.alphaHash = !1, this.blendSrc = 204, this.blendDst = 205, this.blendEquation = 100, this.blendSrcAlpha = null, this.blendDstAlpha = null, this.blendEquationAlpha = null, this.blendColor = new Z(0, 0, 0), this.blendAlpha = 0, this.depthFunc = 3, this.depthTest = !0, this.depthWrite = !0, this.stencilWriteMask = 255, this.stencilFunc = 519, this.stencilRef = 0, this.stencilFuncMask = 255, this.stencilFail = Ld, this.stencilZFail = Ld, this.stencilZPass = Ld, this.stencilWrite = !1, this.clippingPlanes = null, this.clipIntersection = !1, this.clipShadows = !1, this.shadowSide = null, this.colorWrite = !0, this.precision = null, this.polygonOffset = !1, this.polygonOffsetFactor = 0, this.polygonOffsetUnits = 0, this.dithering = !1, this.alphaToCoverage = !1, this.premultipliedAlpha = !1, this.forceSinglePass = !1, this.allowOverride = !0, this.visible = !0, this.toneMapped = !0, this.userData = {}, this.version = 0, this._alphaTest = 0;
	}
	get alphaTest() {
		return this._alphaTest;
	}
	set alphaTest(e) {
		this._alphaTest > 0 != e > 0 && this.version++, this._alphaTest = e;
	}
	onBeforeRender() {}
	onBeforeCompile() {}
	customProgramCacheKey() {
		return this.onBeforeCompile.toString();
	}
	setValues(e) {
		if (e !== void 0) for (let t in e) {
			let n = e[t];
			if (n === void 0) {
				G(`Material: parameter '${t}' has value of undefined.`);
				continue;
			}
			let r = this[t];
			if (r === void 0) {
				G(`Material: '${t}' is not a property of THREE.${this.type}.`);
				continue;
			}
			r && r.isColor ? r.set(n) : r && r.isVector2 && n && n.isVector2 || r && r.isEuler && n && n.isEuler || r && r.isVector3 && n && n.isVector3 ? r.copy(n) : this[t] = n;
		}
	}
	toJSON(e) {
		let t = e === void 0 || typeof e == "string";
		t && (e = {
			textures: {},
			images: {}
		});
		let n = { metadata: {
			version: 4.7,
			type: "Material",
			generator: "Material.toJSON"
		} };
		n.uuid = this.uuid, n.type = this.type, this.name !== "" && (n.name = this.name), this.color && this.color.isColor && (n.color = this.color.getHex()), this.roughness !== void 0 && (n.roughness = this.roughness), this.metalness !== void 0 && (n.metalness = this.metalness), this.sheen !== void 0 && (n.sheen = this.sheen), this.sheenColor && this.sheenColor.isColor && (n.sheenColor = this.sheenColor.getHex()), this.sheenRoughness !== void 0 && (n.sheenRoughness = this.sheenRoughness), this.emissive && this.emissive.isColor && (n.emissive = this.emissive.getHex()), this.emissiveIntensity !== void 0 && this.emissiveIntensity !== 1 && (n.emissiveIntensity = this.emissiveIntensity), this.specular && this.specular.isColor && (n.specular = this.specular.getHex()), this.specularIntensity !== void 0 && (n.specularIntensity = this.specularIntensity), this.specularColor && this.specularColor.isColor && (n.specularColor = this.specularColor.getHex()), this.shininess !== void 0 && (n.shininess = this.shininess), this.clearcoat !== void 0 && (n.clearcoat = this.clearcoat), this.clearcoatRoughness !== void 0 && (n.clearcoatRoughness = this.clearcoatRoughness), this.clearcoatMap && this.clearcoatMap.isTexture && (n.clearcoatMap = this.clearcoatMap.toJSON(e).uuid), this.clearcoatRoughnessMap && this.clearcoatRoughnessMap.isTexture && (n.clearcoatRoughnessMap = this.clearcoatRoughnessMap.toJSON(e).uuid), this.clearcoatNormalMap && this.clearcoatNormalMap.isTexture && (n.clearcoatNormalMap = this.clearcoatNormalMap.toJSON(e).uuid, n.clearcoatNormalScale = this.clearcoatNormalScale.toArray()), this.sheenColorMap && this.sheenColorMap.isTexture && (n.sheenColorMap = this.sheenColorMap.toJSON(e).uuid), this.sheenRoughnessMap && this.sheenRoughnessMap.isTexture && (n.sheenRoughnessMap = this.sheenRoughnessMap.toJSON(e).uuid), this.dispersion !== void 0 && (n.dispersion = this.dispersion), this.iridescence !== void 0 && (n.iridescence = this.iridescence), this.iridescenceIOR !== void 0 && (n.iridescenceIOR = this.iridescenceIOR), this.iridescenceThicknessRange !== void 0 && (n.iridescenceThicknessRange = this.iridescenceThicknessRange), this.iridescenceMap && this.iridescenceMap.isTexture && (n.iridescenceMap = this.iridescenceMap.toJSON(e).uuid), this.iridescenceThicknessMap && this.iridescenceThicknessMap.isTexture && (n.iridescenceThicknessMap = this.iridescenceThicknessMap.toJSON(e).uuid), this.anisotropy !== void 0 && (n.anisotropy = this.anisotropy), this.anisotropyRotation !== void 0 && (n.anisotropyRotation = this.anisotropyRotation), this.anisotropyMap && this.anisotropyMap.isTexture && (n.anisotropyMap = this.anisotropyMap.toJSON(e).uuid), this.map && this.map.isTexture && (n.map = this.map.toJSON(e).uuid), this.matcap && this.matcap.isTexture && (n.matcap = this.matcap.toJSON(e).uuid), this.alphaMap && this.alphaMap.isTexture && (n.alphaMap = this.alphaMap.toJSON(e).uuid), this.lightMap && this.lightMap.isTexture && (n.lightMap = this.lightMap.toJSON(e).uuid, n.lightMapIntensity = this.lightMapIntensity), this.aoMap && this.aoMap.isTexture && (n.aoMap = this.aoMap.toJSON(e).uuid, n.aoMapIntensity = this.aoMapIntensity), this.bumpMap && this.bumpMap.isTexture && (n.bumpMap = this.bumpMap.toJSON(e).uuid, n.bumpScale = this.bumpScale), this.normalMap && this.normalMap.isTexture && (n.normalMap = this.normalMap.toJSON(e).uuid, n.normalMapType = this.normalMapType, n.normalScale = this.normalScale.toArray()), this.displacementMap && this.displacementMap.isTexture && (n.displacementMap = this.displacementMap.toJSON(e).uuid, n.displacementScale = this.displacementScale, n.displacementBias = this.displacementBias), this.roughnessMap && this.roughnessMap.isTexture && (n.roughnessMap = this.roughnessMap.toJSON(e).uuid), this.metalnessMap && this.metalnessMap.isTexture && (n.metalnessMap = this.metalnessMap.toJSON(e).uuid), this.emissiveMap && this.emissiveMap.isTexture && (n.emissiveMap = this.emissiveMap.toJSON(e).uuid), this.specularMap && this.specularMap.isTexture && (n.specularMap = this.specularMap.toJSON(e).uuid), this.specularIntensityMap && this.specularIntensityMap.isTexture && (n.specularIntensityMap = this.specularIntensityMap.toJSON(e).uuid), this.specularColorMap && this.specularColorMap.isTexture && (n.specularColorMap = this.specularColorMap.toJSON(e).uuid), this.envMap && this.envMap.isTexture && (n.envMap = this.envMap.toJSON(e).uuid, this.combine !== void 0 && (n.combine = this.combine)), this.envMapRotation !== void 0 && (n.envMapRotation = this.envMapRotation.toArray()), this.envMapIntensity !== void 0 && (n.envMapIntensity = this.envMapIntensity), this.reflectivity !== void 0 && (n.reflectivity = this.reflectivity), this.refractionRatio !== void 0 && (n.refractionRatio = this.refractionRatio), this.gradientMap && this.gradientMap.isTexture && (n.gradientMap = this.gradientMap.toJSON(e).uuid), this.transmission !== void 0 && (n.transmission = this.transmission), this.transmissionMap && this.transmissionMap.isTexture && (n.transmissionMap = this.transmissionMap.toJSON(e).uuid), this.thickness !== void 0 && (n.thickness = this.thickness), this.thicknessMap && this.thicknessMap.isTexture && (n.thicknessMap = this.thicknessMap.toJSON(e).uuid), this.attenuationDistance !== void 0 && this.attenuationDistance !== Infinity && (n.attenuationDistance = this.attenuationDistance), this.attenuationColor !== void 0 && (n.attenuationColor = this.attenuationColor.getHex()), this.size !== void 0 && (n.size = this.size), this.shadowSide !== null && (n.shadowSide = this.shadowSide), this.sizeAttenuation !== void 0 && (n.sizeAttenuation = this.sizeAttenuation), this.blending !== 1 && (n.blending = this.blending), this.side !== 0 && (n.side = this.side), this.vertexColors === !0 && (n.vertexColors = !0), this.opacity < 1 && (n.opacity = this.opacity), this.transparent === !0 && (n.transparent = !0), this.blendSrc !== 204 && (n.blendSrc = this.blendSrc), this.blendDst !== 205 && (n.blendDst = this.blendDst), this.blendEquation !== 100 && (n.blendEquation = this.blendEquation), this.blendSrcAlpha !== null && (n.blendSrcAlpha = this.blendSrcAlpha), this.blendDstAlpha !== null && (n.blendDstAlpha = this.blendDstAlpha), this.blendEquationAlpha !== null && (n.blendEquationAlpha = this.blendEquationAlpha), this.blendColor && this.blendColor.isColor && (n.blendColor = this.blendColor.getHex()), this.blendAlpha !== 0 && (n.blendAlpha = this.blendAlpha), this.depthFunc !== 3 && (n.depthFunc = this.depthFunc), this.depthTest === !1 && (n.depthTest = this.depthTest), this.depthWrite === !1 && (n.depthWrite = this.depthWrite), this.colorWrite === !1 && (n.colorWrite = this.colorWrite), this.stencilWriteMask !== 255 && (n.stencilWriteMask = this.stencilWriteMask), this.stencilFunc !== 519 && (n.stencilFunc = this.stencilFunc), this.stencilRef !== 0 && (n.stencilRef = this.stencilRef), this.stencilFuncMask !== 255 && (n.stencilFuncMask = this.stencilFuncMask), this.stencilFail !== 7680 && (n.stencilFail = this.stencilFail), this.stencilZFail !== 7680 && (n.stencilZFail = this.stencilZFail), this.stencilZPass !== 7680 && (n.stencilZPass = this.stencilZPass), this.stencilWrite === !0 && (n.stencilWrite = this.stencilWrite), this.rotation !== void 0 && this.rotation !== 0 && (n.rotation = this.rotation), this.polygonOffset === !0 && (n.polygonOffset = !0), this.polygonOffsetFactor !== 0 && (n.polygonOffsetFactor = this.polygonOffsetFactor), this.polygonOffsetUnits !== 0 && (n.polygonOffsetUnits = this.polygonOffsetUnits), this.linewidth !== void 0 && this.linewidth !== 1 && (n.linewidth = this.linewidth), this.dashSize !== void 0 && (n.dashSize = this.dashSize), this.gapSize !== void 0 && (n.gapSize = this.gapSize), this.scale !== void 0 && (n.scale = this.scale), this.dithering === !0 && (n.dithering = !0), this.alphaTest > 0 && (n.alphaTest = this.alphaTest), this.alphaHash === !0 && (n.alphaHash = !0), this.alphaToCoverage === !0 && (n.alphaToCoverage = !0), this.premultipliedAlpha === !0 && (n.premultipliedAlpha = !0), this.forceSinglePass === !0 && (n.forceSinglePass = !0), this.allowOverride === !1 && (n.allowOverride = !1), this.wireframe === !0 && (n.wireframe = !0), this.wireframeLinewidth > 1 && (n.wireframeLinewidth = this.wireframeLinewidth), this.wireframeLinecap !== "round" && (n.wireframeLinecap = this.wireframeLinecap), this.wireframeLinejoin !== "round" && (n.wireframeLinejoin = this.wireframeLinejoin), this.flatShading === !0 && (n.flatShading = !0), this.visible === !1 && (n.visible = !1), this.toneMapped === !1 && (n.toneMapped = !1), this.fog === !1 && (n.fog = !1), Object.keys(this.userData).length > 0 && (n.userData = this.userData);
		function r(e) {
			let t = [];
			for (let n in e) {
				let r = e[n];
				delete r.metadata, t.push(r);
			}
			return t;
		}
		if (t) {
			let t = r(e.textures), i = r(e.images);
			t.length > 0 && (n.textures = t), i.length > 0 && (n.images = i);
		}
		return n;
	}
	fromJSON(e, t) {
		if (e.uuid !== void 0 && (this.uuid = e.uuid), e.name !== void 0 && (this.name = e.name), e.color !== void 0 && this.color !== void 0 && this.color.setHex(e.color), e.roughness !== void 0 && (this.roughness = e.roughness), e.metalness !== void 0 && (this.metalness = e.metalness), e.sheen !== void 0 && (this.sheen = e.sheen), e.sheenColor !== void 0 && (this.sheenColor = new Z().setHex(e.sheenColor)), e.sheenRoughness !== void 0 && (this.sheenRoughness = e.sheenRoughness), e.emissive !== void 0 && this.emissive !== void 0 && this.emissive.setHex(e.emissive), e.specular !== void 0 && this.specular !== void 0 && this.specular.setHex(e.specular), e.specularIntensity !== void 0 && (this.specularIntensity = e.specularIntensity), e.specularColor !== void 0 && this.specularColor !== void 0 && this.specularColor.setHex(e.specularColor), e.shininess !== void 0 && (this.shininess = e.shininess), e.clearcoat !== void 0 && (this.clearcoat = e.clearcoat), e.clearcoatRoughness !== void 0 && (this.clearcoatRoughness = e.clearcoatRoughness), e.dispersion !== void 0 && (this.dispersion = e.dispersion), e.iridescence !== void 0 && (this.iridescence = e.iridescence), e.iridescenceIOR !== void 0 && (this.iridescenceIOR = e.iridescenceIOR), e.iridescenceThicknessRange !== void 0 && (this.iridescenceThicknessRange = e.iridescenceThicknessRange), e.transmission !== void 0 && (this.transmission = e.transmission), e.thickness !== void 0 && (this.thickness = e.thickness), e.attenuationDistance !== void 0 && (this.attenuationDistance = e.attenuationDistance), e.attenuationColor !== void 0 && this.attenuationColor !== void 0 && this.attenuationColor.setHex(e.attenuationColor), e.anisotropy !== void 0 && (this.anisotropy = e.anisotropy), e.anisotropyRotation !== void 0 && (this.anisotropyRotation = e.anisotropyRotation), e.fog !== void 0 && (this.fog = e.fog), e.flatShading !== void 0 && (this.flatShading = e.flatShading), e.blending !== void 0 && (this.blending = e.blending), e.combine !== void 0 && (this.combine = e.combine), e.side !== void 0 && (this.side = e.side), e.shadowSide !== void 0 && (this.shadowSide = e.shadowSide), e.opacity !== void 0 && (this.opacity = e.opacity), e.transparent !== void 0 && (this.transparent = e.transparent), e.alphaTest !== void 0 && (this.alphaTest = e.alphaTest), e.alphaHash !== void 0 && (this.alphaHash = e.alphaHash), e.depthFunc !== void 0 && (this.depthFunc = e.depthFunc), e.depthTest !== void 0 && (this.depthTest = e.depthTest), e.depthWrite !== void 0 && (this.depthWrite = e.depthWrite), e.colorWrite !== void 0 && (this.colorWrite = e.colorWrite), e.blendSrc !== void 0 && (this.blendSrc = e.blendSrc), e.blendDst !== void 0 && (this.blendDst = e.blendDst), e.blendEquation !== void 0 && (this.blendEquation = e.blendEquation), e.blendSrcAlpha !== void 0 && (this.blendSrcAlpha = e.blendSrcAlpha), e.blendDstAlpha !== void 0 && (this.blendDstAlpha = e.blendDstAlpha), e.blendEquationAlpha !== void 0 && (this.blendEquationAlpha = e.blendEquationAlpha), e.blendColor !== void 0 && this.blendColor !== void 0 && this.blendColor.setHex(e.blendColor), e.blendAlpha !== void 0 && (this.blendAlpha = e.blendAlpha), e.stencilWriteMask !== void 0 && (this.stencilWriteMask = e.stencilWriteMask), e.stencilFunc !== void 0 && (this.stencilFunc = e.stencilFunc), e.stencilRef !== void 0 && (this.stencilRef = e.stencilRef), e.stencilFuncMask !== void 0 && (this.stencilFuncMask = e.stencilFuncMask), e.stencilFail !== void 0 && (this.stencilFail = e.stencilFail), e.stencilZFail !== void 0 && (this.stencilZFail = e.stencilZFail), e.stencilZPass !== void 0 && (this.stencilZPass = e.stencilZPass), e.stencilWrite !== void 0 && (this.stencilWrite = e.stencilWrite), e.wireframe !== void 0 && (this.wireframe = e.wireframe), e.wireframeLinewidth !== void 0 && (this.wireframeLinewidth = e.wireframeLinewidth), e.wireframeLinecap !== void 0 && (this.wireframeLinecap = e.wireframeLinecap), e.wireframeLinejoin !== void 0 && (this.wireframeLinejoin = e.wireframeLinejoin), e.rotation !== void 0 && (this.rotation = e.rotation), e.linewidth !== void 0 && (this.linewidth = e.linewidth), e.dashSize !== void 0 && (this.dashSize = e.dashSize), e.gapSize !== void 0 && (this.gapSize = e.gapSize), e.scale !== void 0 && (this.scale = e.scale), e.polygonOffset !== void 0 && (this.polygonOffset = e.polygonOffset), e.polygonOffsetFactor !== void 0 && (this.polygonOffsetFactor = e.polygonOffsetFactor), e.polygonOffsetUnits !== void 0 && (this.polygonOffsetUnits = e.polygonOffsetUnits), e.dithering !== void 0 && (this.dithering = e.dithering), e.alphaToCoverage !== void 0 && (this.alphaToCoverage = e.alphaToCoverage), e.premultipliedAlpha !== void 0 && (this.premultipliedAlpha = e.premultipliedAlpha), e.forceSinglePass !== void 0 && (this.forceSinglePass = e.forceSinglePass), e.allowOverride !== void 0 && (this.allowOverride = e.allowOverride), e.visible !== void 0 && (this.visible = e.visible), e.toneMapped !== void 0 && (this.toneMapped = e.toneMapped), e.userData !== void 0 && (this.userData = e.userData), e.vertexColors !== void 0 && (typeof e.vertexColors == "number" ? this.vertexColors = e.vertexColors > 0 : this.vertexColors = e.vertexColors), e.size !== void 0 && (this.size = e.size), e.sizeAttenuation !== void 0 && (this.sizeAttenuation = e.sizeAttenuation), e.map !== void 0 && (this.map = t[e.map] || null), e.matcap !== void 0 && (this.matcap = t[e.matcap] || null), e.alphaMap !== void 0 && (this.alphaMap = t[e.alphaMap] || null), e.bumpMap !== void 0 && (this.bumpMap = t[e.bumpMap] || null), e.bumpScale !== void 0 && (this.bumpScale = e.bumpScale), e.normalMap !== void 0 && (this.normalMap = t[e.normalMap] || null), e.normalMapType !== void 0 && (this.normalMapType = e.normalMapType), e.normalScale !== void 0) {
			let t = e.normalScale;
			Array.isArray(t) === !1 && (t = [t, t]), this.normalScale = new Tf().fromArray(t);
		}
		return e.displacementMap !== void 0 && (this.displacementMap = t[e.displacementMap] || null), e.displacementScale !== void 0 && (this.displacementScale = e.displacementScale), e.displacementBias !== void 0 && (this.displacementBias = e.displacementBias), e.roughnessMap !== void 0 && (this.roughnessMap = t[e.roughnessMap] || null), e.metalnessMap !== void 0 && (this.metalnessMap = t[e.metalnessMap] || null), e.emissiveMap !== void 0 && (this.emissiveMap = t[e.emissiveMap] || null), e.emissiveIntensity !== void 0 && (this.emissiveIntensity = e.emissiveIntensity), e.specularMap !== void 0 && (this.specularMap = t[e.specularMap] || null), e.specularIntensityMap !== void 0 && (this.specularIntensityMap = t[e.specularIntensityMap] || null), e.specularColorMap !== void 0 && (this.specularColorMap = t[e.specularColorMap] || null), e.envMap !== void 0 && (this.envMap = t[e.envMap] || null), e.envMapRotation !== void 0 && this.envMapRotation.fromArray(e.envMapRotation), e.envMapIntensity !== void 0 && (this.envMapIntensity = e.envMapIntensity), e.reflectivity !== void 0 && (this.reflectivity = e.reflectivity), e.refractionRatio !== void 0 && (this.refractionRatio = e.refractionRatio), e.lightMap !== void 0 && (this.lightMap = t[e.lightMap] || null), e.lightMapIntensity !== void 0 && (this.lightMapIntensity = e.lightMapIntensity), e.aoMap !== void 0 && (this.aoMap = t[e.aoMap] || null), e.aoMapIntensity !== void 0 && (this.aoMapIntensity = e.aoMapIntensity), e.gradientMap !== void 0 && (this.gradientMap = t[e.gradientMap] || null), e.clearcoatMap !== void 0 && (this.clearcoatMap = t[e.clearcoatMap] || null), e.clearcoatRoughnessMap !== void 0 && (this.clearcoatRoughnessMap = t[e.clearcoatRoughnessMap] || null), e.clearcoatNormalMap !== void 0 && (this.clearcoatNormalMap = t[e.clearcoatNormalMap] || null), e.clearcoatNormalScale !== void 0 && (this.clearcoatNormalScale = new Tf().fromArray(e.clearcoatNormalScale)), e.iridescenceMap !== void 0 && (this.iridescenceMap = t[e.iridescenceMap] || null), e.iridescenceThicknessMap !== void 0 && (this.iridescenceThicknessMap = t[e.iridescenceThicknessMap] || null), e.transmissionMap !== void 0 && (this.transmissionMap = t[e.transmissionMap] || null), e.thicknessMap !== void 0 && (this.thicknessMap = t[e.thicknessMap] || null), e.anisotropyMap !== void 0 && (this.anisotropyMap = t[e.anisotropyMap] || null), e.sheenColorMap !== void 0 && (this.sheenColorMap = t[e.sheenColorMap] || null), e.sheenRoughnessMap !== void 0 && (this.sheenRoughnessMap = t[e.sheenRoughnessMap] || null), this;
	}
	clone() {
		return new this.constructor().copy(this);
	}
	copy(e) {
		this.name = e.name, this.blending = e.blending, this.side = e.side, this.vertexColors = e.vertexColors, this.opacity = e.opacity, this.transparent = e.transparent, this.blendSrc = e.blendSrc, this.blendDst = e.blendDst, this.blendEquation = e.blendEquation, this.blendSrcAlpha = e.blendSrcAlpha, this.blendDstAlpha = e.blendDstAlpha, this.blendEquationAlpha = e.blendEquationAlpha, this.blendColor.copy(e.blendColor), this.blendAlpha = e.blendAlpha, this.depthFunc = e.depthFunc, this.depthTest = e.depthTest, this.depthWrite = e.depthWrite, this.stencilWriteMask = e.stencilWriteMask, this.stencilFunc = e.stencilFunc, this.stencilRef = e.stencilRef, this.stencilFuncMask = e.stencilFuncMask, this.stencilFail = e.stencilFail, this.stencilZFail = e.stencilZFail, this.stencilZPass = e.stencilZPass, this.stencilWrite = e.stencilWrite;
		let t = e.clippingPlanes, n = null;
		if (t !== null) {
			let e = t.length;
			n = Array(e);
			for (let r = 0; r !== e; ++r) n[r] = t[r].clone();
		}
		return this.clippingPlanes = n, this.clipIntersection = e.clipIntersection, this.clipShadows = e.clipShadows, this.shadowSide = e.shadowSide, this.colorWrite = e.colorWrite, this.precision = e.precision, this.polygonOffset = e.polygonOffset, this.polygonOffsetFactor = e.polygonOffsetFactor, this.polygonOffsetUnits = e.polygonOffsetUnits, this.dithering = e.dithering, this.alphaTest = e.alphaTest, this.alphaHash = e.alphaHash, this.alphaToCoverage = e.alphaToCoverage, this.premultipliedAlpha = e.premultipliedAlpha, this.forceSinglePass = e.forceSinglePass, this.allowOverride = e.allowOverride, this.visible = e.visible, this.toneMapped = e.toneMapped, this.userData = JSON.parse(JSON.stringify(e.userData)), this;
	}
	dispose() {
		this.dispatchEvent({ type: "dispose" });
	}
	set needsUpdate(e) {
		e === !0 && this.version++;
	}
}, Dm = /*@__PURE__*/ new J(), Om = /*@__PURE__*/ new J(), km = /*@__PURE__*/ new J(), Am = /*@__PURE__*/ new J(), jm = /*@__PURE__*/ new J(), Mm = /*@__PURE__*/ new J(), Nm = /*@__PURE__*/ new J(), Pm = class {
	constructor(e = new J(), t = new J(0, 0, -1)) {
		this.origin = e, this.direction = t;
	}
	set(e, t) {
		return this.origin.copy(e), this.direction.copy(t), this;
	}
	copy(e) {
		return this.origin.copy(e.origin), this.direction.copy(e.direction), this;
	}
	at(e, t) {
		return t.copy(this.origin).addScaledVector(this.direction, e);
	}
	lookAt(e) {
		return this.direction.copy(e).sub(this.origin).normalize(), this;
	}
	recast(e) {
		return this.origin.copy(this.at(e, Dm)), this;
	}
	closestPointToPoint(e, t) {
		t.subVectors(e, this.origin);
		let n = t.dot(this.direction);
		return n < 0 ? t.copy(this.origin) : t.copy(this.origin).addScaledVector(this.direction, n);
	}
	distanceToPoint(e) {
		return Math.sqrt(this.distanceSqToPoint(e));
	}
	distanceSqToPoint(e) {
		let t = Dm.subVectors(e, this.origin).dot(this.direction);
		return t < 0 ? this.origin.distanceToSquared(e) : (Dm.copy(this.origin).addScaledVector(this.direction, t), Dm.distanceToSquared(e));
	}
	distanceSqToSegment(e, t, n, r) {
		Om.copy(e).add(t).multiplyScalar(.5), km.copy(t).sub(e).normalize(), Am.copy(this.origin).sub(Om);
		let i = e.distanceTo(t) * .5, a = -this.direction.dot(km), o = Am.dot(this.direction), s = -Am.dot(km), c = Am.lengthSq(), l = Math.abs(1 - a * a), u, d, f, p;
		if (l > 0) if (u = a * s - o, d = a * o - s, p = i * l, u >= 0) if (d >= -p) if (d <= p) {
			let e = 1 / l;
			u *= e, d *= e, f = u * (u + a * d + 2 * o) + d * (a * u + d + 2 * s) + c;
		} else d = i, u = Math.max(0, -(a * d + o)), f = -u * u + d * (d + 2 * s) + c;
		else d = -i, u = Math.max(0, -(a * d + o)), f = -u * u + d * (d + 2 * s) + c;
		else d <= -p ? (u = Math.max(0, -(-a * i + o)), d = u > 0 ? -i : Math.min(Math.max(-i, -s), i), f = -u * u + d * (d + 2 * s) + c) : d <= p ? (u = 0, d = Math.min(Math.max(-i, -s), i), f = d * (d + 2 * s) + c) : (u = Math.max(0, -(a * i + o)), d = u > 0 ? i : Math.min(Math.max(-i, -s), i), f = -u * u + d * (d + 2 * s) + c);
		else d = a > 0 ? -i : i, u = Math.max(0, -(a * d + o)), f = -u * u + d * (d + 2 * s) + c;
		return n && n.copy(this.origin).addScaledVector(this.direction, u), r && r.copy(Om).addScaledVector(km, d), f;
	}
	intersectSphere(e, t) {
		Dm.subVectors(e.center, this.origin);
		let n = Dm.dot(this.direction), r = Dm.dot(Dm) - n * n, i = e.radius * e.radius;
		if (r > i) return null;
		let a = Math.sqrt(i - r), o = n - a, s = n + a;
		return s < 0 ? null : o < 0 ? this.at(s, t) : this.at(o, t);
	}
	intersectsSphere(e) {
		return e.radius < 0 ? !1 : this.distanceSqToPoint(e.center) <= e.radius * e.radius;
	}
	distanceToPlane(e) {
		let t = e.normal.dot(this.direction);
		if (t === 0) return e.distanceToPoint(this.origin) === 0 ? 0 : null;
		let n = -(this.origin.dot(e.normal) + e.constant) / t;
		return n >= 0 ? n : null;
	}
	intersectPlane(e, t) {
		let n = this.distanceToPlane(e);
		return n === null ? null : this.at(n, t);
	}
	intersectsPlane(e) {
		let t = e.distanceToPoint(this.origin);
		return t === 0 || e.normal.dot(this.direction) * t < 0;
	}
	intersectBox(e, t) {
		let n, r, i, a, o, s, c = 1 / this.direction.x, l = 1 / this.direction.y, u = 1 / this.direction.z, d = this.origin;
		return c >= 0 ? (n = (e.min.x - d.x) * c, r = (e.max.x - d.x) * c) : (n = (e.max.x - d.x) * c, r = (e.min.x - d.x) * c), l >= 0 ? (i = (e.min.y - d.y) * l, a = (e.max.y - d.y) * l) : (i = (e.max.y - d.y) * l, a = (e.min.y - d.y) * l), n > a || i > r || ((i > n || isNaN(n)) && (n = i), (a < r || isNaN(r)) && (r = a), u >= 0 ? (o = (e.min.z - d.z) * u, s = (e.max.z - d.z) * u) : (o = (e.max.z - d.z) * u, s = (e.min.z - d.z) * u), n > s || o > r) || ((o > n || n !== n) && (n = o), (s < r || r !== r) && (r = s), r < 0) ? null : this.at(n >= 0 ? n : r, t);
	}
	intersectsBox(e) {
		return this.intersectBox(e, Dm) !== null;
	}
	intersectTriangle(e, t, n, r, i) {
		jm.subVectors(t, e), Mm.subVectors(n, e), Nm.crossVectors(jm, Mm);
		let a = this.direction.dot(Nm), o;
		if (a > 0) {
			if (r) return null;
			o = 1;
		} else if (a < 0) o = -1, a = -a;
		else return null;
		Am.subVectors(this.origin, e);
		let s = o * this.direction.dot(Mm.crossVectors(Am, Mm));
		if (s < 0) return null;
		let c = o * this.direction.dot(jm.cross(Am));
		if (c < 0 || s + c > a) return null;
		let l = -o * Am.dot(Nm);
		return l < 0 ? null : this.at(l / a, i);
	}
	applyMatrix4(e) {
		return this.origin.applyMatrix4(e), this.direction.transformDirection(e), this;
	}
	equals(e) {
		return e.origin.equals(this.origin) && e.direction.equals(this.direction);
	}
	clone() {
		return new this.constructor().copy(this);
	}
}, Fm = class extends Em {
	constructor(e) {
		super(), this.isMeshBasicMaterial = !0, this.type = "MeshBasicMaterial", this.color = new Z(16777215), this.map = null, this.lightMap = null, this.lightMapIntensity = 1, this.aoMap = null, this.aoMapIntensity = 1, this.specularMap = null, this.alphaMap = null, this.envMap = null, this.envMapRotation = new ip(), this.combine = 0, this.reflectivity = 1, this.refractionRatio = .98, this.wireframe = !1, this.wireframeLinewidth = 1, this.wireframeLinecap = "round", this.wireframeLinejoin = "round", this.fog = !0, this.setValues(e);
	}
	copy(e) {
		return super.copy(e), this.color.copy(e.color), this.map = e.map, this.lightMap = e.lightMap, this.lightMapIntensity = e.lightMapIntensity, this.aoMap = e.aoMap, this.aoMapIntensity = e.aoMapIntensity, this.specularMap = e.specularMap, this.alphaMap = e.alphaMap, this.envMap = e.envMap, this.envMapRotation.copy(e.envMapRotation), this.combine = e.combine, this.reflectivity = e.reflectivity, this.refractionRatio = e.refractionRatio, this.wireframe = e.wireframe, this.wireframeLinewidth = e.wireframeLinewidth, this.wireframeLinecap = e.wireframeLinecap, this.wireframeLinejoin = e.wireframeLinejoin, this.fog = e.fog, this;
	}
}, Im = /*@__PURE__*/ new Jf(), Lm = /*@__PURE__*/ new Pm(), Rm = /*@__PURE__*/ new gm(), zm = /*@__PURE__*/ new J(), Bm = /*@__PURE__*/ new J(), Vm = /*@__PURE__*/ new J(), Hm = /*@__PURE__*/ new J(), Um = /*@__PURE__*/ new J(), Wm = /*@__PURE__*/ new J(), Gm = /*@__PURE__*/ new J(), Km = /*@__PURE__*/ new J(), qm = class extends xp {
	constructor(e = new wm(), t = new Fm()) {
		super(), this.isMesh = !0, this.type = "Mesh", this.geometry = e, this.material = t, this.morphTargetDictionary = void 0, this.morphTargetInfluences = void 0, this.count = 1, this.updateMorphTargets();
	}
	copy(e, t) {
		return super.copy(e, t), e.morphTargetInfluences !== void 0 && (this.morphTargetInfluences = e.morphTargetInfluences.slice()), e.morphTargetDictionary !== void 0 && (this.morphTargetDictionary = Object.assign({}, e.morphTargetDictionary)), this.material = Array.isArray(e.material) ? e.material.slice() : e.material, this.geometry = e.geometry, this;
	}
	updateMorphTargets() {
		let e = this.geometry.morphAttributes, t = Object.keys(e);
		if (t.length > 0) {
			let n = e[t[0]];
			if (n !== void 0) {
				this.morphTargetInfluences = [], this.morphTargetDictionary = {};
				for (let e = 0, t = n.length; e < t; e++) {
					let t = n[e].name || String(e);
					this.morphTargetInfluences.push(0), this.morphTargetDictionary[t] = e;
				}
			}
		}
	}
	getVertexPosition(e, t) {
		let n = this.geometry, r = n.attributes.position, i = n.morphAttributes.position, a = n.morphTargetsRelative;
		t.fromBufferAttribute(r, e);
		let o = this.morphTargetInfluences;
		if (i && o) {
			Wm.set(0, 0, 0);
			for (let n = 0, r = i.length; n < r; n++) {
				let r = o[n], s = i[n];
				r !== 0 && (Um.fromBufferAttribute(s, e), a ? Wm.addScaledVector(Um, r) : Wm.addScaledVector(Um.sub(t), r));
			}
			t.add(Wm);
		}
		return t;
	}
	raycast(e, t) {
		let n = this.geometry, r = this.material, i = this.matrixWorld;
		r !== void 0 && (n.boundingSphere === null && n.computeBoundingSphere(), Rm.copy(n.boundingSphere), Rm.applyMatrix4(i), Lm.copy(e.ray).recast(e.near), !(Rm.containsPoint(Lm.origin) === !1 && (Lm.intersectSphere(Rm, zm) === null || Lm.origin.distanceToSquared(zm) > (e.far - e.near) ** 2)) && (Im.copy(i).invert(), Lm.copy(e.ray).applyMatrix4(Im), !(n.boundingBox !== null && Lm.intersectsBox(n.boundingBox) === !1) && this._computeIntersections(e, t, Lm)));
	}
	_computeIntersections(e, t, n) {
		let r, i = this.geometry, a = this.material, o = i.index, s = i.attributes.position, c = i.attributes.uv, l = i.attributes.uv1, u = i.attributes.normal, d = i.groups, f = i.drawRange;
		if (o !== null) if (Array.isArray(a)) for (let i = 0, s = d.length; i < s; i++) {
			let s = d[i], p = a[s.materialIndex], m = Math.max(s.start, f.start), h = Math.min(o.count, Math.min(s.start + s.count, f.start + f.count));
			for (let i = m, a = h; i < a; i += 3) {
				let a = o.getX(i), d = o.getX(i + 1), f = o.getX(i + 2);
				r = Ym(this, p, e, n, c, l, u, a, d, f), r && (r.faceIndex = Math.floor(i / 3), r.face.materialIndex = s.materialIndex, t.push(r));
			}
		}
		else {
			let i = Math.max(0, f.start), s = Math.min(o.count, f.start + f.count);
			for (let d = i, f = s; d < f; d += 3) {
				let i = o.getX(d), s = o.getX(d + 1), f = o.getX(d + 2);
				r = Ym(this, a, e, n, c, l, u, i, s, f), r && (r.faceIndex = Math.floor(d / 3), t.push(r));
			}
		}
		else if (s !== void 0) if (Array.isArray(a)) for (let i = 0, o = d.length; i < o; i++) {
			let o = d[i], p = a[o.materialIndex], m = Math.max(o.start, f.start), h = Math.min(s.count, Math.min(o.start + o.count, f.start + f.count));
			for (let i = m, a = h; i < a; i += 3) {
				let a = i, s = i + 1, d = i + 2;
				r = Ym(this, p, e, n, c, l, u, a, s, d), r && (r.faceIndex = Math.floor(i / 3), r.face.materialIndex = o.materialIndex, t.push(r));
			}
		}
		else {
			let i = Math.max(0, f.start), o = Math.min(s.count, f.start + f.count);
			for (let s = i, d = o; s < d; s += 3) {
				let i = s, o = s + 1, d = s + 2;
				r = Ym(this, a, e, n, c, l, u, i, o, d), r && (r.faceIndex = Math.floor(s / 3), t.push(r));
			}
		}
	}
};
function Jm(e, t, n, r, i, a, o, s) {
	let c;
	if (c = t.side === 1 ? r.intersectTriangle(o, a, i, !0, s) : r.intersectTriangle(i, a, o, t.side === 0, s), c === null) return null;
	Km.copy(s), Km.applyMatrix4(e.matrixWorld);
	let l = n.ray.origin.distanceTo(Km);
	return l < n.near || l > n.far ? null : {
		distance: l,
		point: Km.clone(),
		object: e
	};
}
function Ym(e, t, n, r, i, a, o, s, c, l) {
	e.getVertexPosition(s, Bm), e.getVertexPosition(c, Vm), e.getVertexPosition(l, Hm);
	let u = Jm(e, t, n, r, Bm, Vm, Hm, Gm);
	if (u) {
		let e = new J();
		Wp.getBarycoord(Gm, Bm, Vm, Hm, e), i && (u.uv = Wp.getInterpolatedAttribute(i, s, c, l, e, new Tf())), a && (u.uv1 = Wp.getInterpolatedAttribute(a, s, c, l, e, new Tf())), o && (u.normal = Wp.getInterpolatedAttribute(o, s, c, l, e, new J()), u.normal.dot(r.direction) > 0 && u.normal.multiplyScalar(-1));
		let t = {
			a: s,
			b: c,
			c: l,
			normal: new J(),
			materialIndex: 0
		};
		Wp.getNormal(Bm, Vm, Hm, t.normal), u.face = t, u.barycoord = e;
	}
	return u;
}
var Xm = class extends Hf {
	constructor(e = null, t = 1, n = 1, r, i, a, o, s, c = mu, l = mu, u, d) {
		super(null, a, o, s, c, l, r, i, u, d), this.isDataTexture = !0, this.image = {
			data: e,
			width: t,
			height: n
		}, this.generateMipmaps = !1, this.flipY = !1, this.unpackAlignment = 1;
	}
}, Zm = /*@__PURE__*/ new J(), Qm = /*@__PURE__*/ new J(), $m = /*@__PURE__*/ new Y(), eh = class {
	constructor(e = new J(1, 0, 0), t = 0) {
		this.isPlane = !0, this.normal = e, this.constant = t;
	}
	set(e, t) {
		return this.normal.copy(e), this.constant = t, this;
	}
	setComponents(e, t, n, r) {
		return this.normal.set(e, t, n), this.constant = r, this;
	}
	setFromNormalAndCoplanarPoint(e, t) {
		return this.normal.copy(e), this.constant = -t.dot(this.normal), this;
	}
	setFromCoplanarPoints(e, t, n) {
		let r = Zm.subVectors(n, t).cross(Qm.subVectors(e, t)).normalize();
		return this.setFromNormalAndCoplanarPoint(r, e), this;
	}
	copy(e) {
		return this.normal.copy(e.normal), this.constant = e.constant, this;
	}
	normalize() {
		let e = 1 / this.normal.length();
		return this.normal.multiplyScalar(e), this.constant *= e, this;
	}
	negate() {
		return this.constant *= -1, this.normal.negate(), this;
	}
	distanceToPoint(e) {
		return this.normal.dot(e) + this.constant;
	}
	distanceToSphere(e) {
		return this.distanceToPoint(e.center) - e.radius;
	}
	projectPoint(e, t) {
		return t.copy(e).addScaledVector(this.normal, -this.distanceToPoint(e));
	}
	intersectLine(e, t, n = !0) {
		let r = e.delta(Zm), i = this.normal.dot(r);
		if (i === 0) return this.distanceToPoint(e.start) === 0 ? t.copy(e.start) : null;
		let a = -(e.start.dot(this.normal) + this.constant) / i;
		return n === !0 && (a < 0 || a > 1) ? null : t.copy(e.start).addScaledVector(r, a);
	}
	intersectsLine(e) {
		let t = this.distanceToPoint(e.start), n = this.distanceToPoint(e.end);
		return t < 0 && n > 0 || n < 0 && t > 0;
	}
	intersectsBox(e) {
		return e.intersectsPlane(this);
	}
	intersectsSphere(e) {
		return e.intersectsPlane(this);
	}
	coplanarPoint(e) {
		return e.copy(this.normal).multiplyScalar(-this.constant);
	}
	applyMatrix4(e, t) {
		let n = t || $m.getNormalMatrix(e), r = this.coplanarPoint(Zm).applyMatrix4(e), i = this.normal.applyMatrix3(n).normalize();
		return this.constant = -r.dot(i), this;
	}
	translate(e) {
		return this.constant -= e.dot(this.normal), this;
	}
	equals(e) {
		return e.normal.equals(this.normal) && e.constant === this.constant;
	}
	clone() {
		return new this.constructor().copy(this);
	}
}, th = /*@__PURE__*/ new gm(), nh = /*@__PURE__*/ new Tf(.5, .5), rh = /*@__PURE__*/ new J(), ih = class {
	constructor(e = new eh(), t = new eh(), n = new eh(), r = new eh(), i = new eh(), a = new eh()) {
		this.planes = [
			e,
			t,
			n,
			r,
			i,
			a
		];
	}
	set(e, t, n, r, i, a) {
		let o = this.planes;
		return o[0].copy(e), o[1].copy(t), o[2].copy(n), o[3].copy(r), o[4].copy(i), o[5].copy(a), this;
	}
	copy(e) {
		let t = this.planes;
		for (let n = 0; n < 6; n++) t[n].copy(e.planes[n]);
		return this;
	}
	setFromProjectionMatrix(e, t = Bd, n = !1) {
		let r = this.planes, i = e.elements, a = i[0], o = i[1], s = i[2], c = i[3], l = i[4], u = i[5], d = i[6], f = i[7], p = i[8], m = i[9], h = i[10], g = i[11], _ = i[12], v = i[13], y = i[14], b = i[15];
		if (r[0].setComponents(c - a, f - l, g - p, b - _).normalize(), r[1].setComponents(c + a, f + l, g + p, b + _).normalize(), r[2].setComponents(c + o, f + u, g + m, b + v).normalize(), r[3].setComponents(c - o, f - u, g - m, b - v).normalize(), n) r[4].setComponents(s, d, h, y).normalize(), r[5].setComponents(c - s, f - d, g - h, b - y).normalize();
		else if (r[4].setComponents(c - s, f - d, g - h, b - y).normalize(), t === 2e3) r[5].setComponents(c + s, f + d, g + h, b + y).normalize();
		else if (t === 2001) r[5].setComponents(s, d, h, y).normalize();
		else throw Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: " + t);
		return this;
	}
	intersectsObject(e) {
		if (e.boundingSphere !== void 0) e.boundingSphere === null && e.computeBoundingSphere(), th.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);
		else {
			let t = e.geometry;
			t.boundingSphere === null && t.computeBoundingSphere(), th.copy(t.boundingSphere).applyMatrix4(e.matrixWorld);
		}
		return this.intersectsSphere(th);
	}
	intersectsSprite(e) {
		return th.center.set(0, 0, 0), th.radius = .7071067811865476 + nh.distanceTo(e.center), th.applyMatrix4(e.matrixWorld), this.intersectsSphere(th);
	}
	intersectsSphere(e) {
		let t = this.planes, n = e.center, r = -e.radius;
		for (let e = 0; e < 6; e++) if (t[e].distanceToPoint(n) < r) return !1;
		return !0;
	}
	intersectsBox(e) {
		let t = this.planes;
		for (let n = 0; n < 6; n++) {
			let r = t[n];
			if (rh.x = r.normal.x > 0 ? e.max.x : e.min.x, rh.y = r.normal.y > 0 ? e.max.y : e.min.y, rh.z = r.normal.z > 0 ? e.max.z : e.min.z, r.distanceToPoint(rh) < 0) return !1;
		}
		return !0;
	}
	containsPoint(e) {
		let t = this.planes;
		for (let n = 0; n < 6; n++) if (t[n].distanceToPoint(e) < 0) return !1;
		return !0;
	}
	clone() {
		return new this.constructor().copy(this);
	}
}, ah = class extends Em {
	constructor(e) {
		super(), this.isLineBasicMaterial = !0, this.type = "LineBasicMaterial", this.color = new Z(16777215), this.map = null, this.linewidth = 1, this.linecap = "round", this.linejoin = "round", this.fog = !0, this.setValues(e);
	}
	copy(e) {
		return super.copy(e), this.color.copy(e.color), this.map = e.map, this.linewidth = e.linewidth, this.linecap = e.linecap, this.linejoin = e.linejoin, this.fog = e.fog, this;
	}
}, oh = /*@__PURE__*/ new J(), sh = /*@__PURE__*/ new J(), ch = /*@__PURE__*/ new Jf(), lh = /*@__PURE__*/ new Pm(), uh = /*@__PURE__*/ new gm(), dh = /*@__PURE__*/ new J(), fh = /*@__PURE__*/ new J(), ph = class extends xp {
	constructor(e = new wm(), t = new ah()) {
		super(), this.isLine = !0, this.type = "Line", this.geometry = e, this.material = t, this.morphTargetDictionary = void 0, this.morphTargetInfluences = void 0, this.updateMorphTargets();
	}
	copy(e, t) {
		return super.copy(e, t), this.material = Array.isArray(e.material) ? e.material.slice() : e.material, this.geometry = e.geometry, this;
	}
	computeLineDistances() {
		let e = this.geometry;
		if (e.index === null) {
			let t = e.attributes.position, n = [0];
			for (let e = 1, r = t.count; e < r; e++) oh.fromBufferAttribute(t, e - 1), sh.fromBufferAttribute(t, e), n[e] = n[e - 1], n[e] += oh.distanceTo(sh);
			e.setAttribute("lineDistance", new fm(n, 1));
		} else G("Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");
		return this;
	}
	raycast(e, t) {
		let n = this.geometry, r = this.matrixWorld, i = e.params.Line.threshold, a = n.drawRange;
		if (n.boundingSphere === null && n.computeBoundingSphere(), uh.copy(n.boundingSphere), uh.applyMatrix4(r), uh.radius += i, e.ray.intersectsSphere(uh) === !1) return;
		ch.copy(r).invert(), lh.copy(e.ray).applyMatrix4(ch);
		let o = i / ((this.scale.x + this.scale.y + this.scale.z) / 3), s = o * o, c = this.isLineSegments ? 2 : 1, l = n.index, u = n.attributes.position;
		if (l !== null) {
			let n = Math.max(0, a.start), r = Math.min(l.count, a.start + a.count);
			for (let i = n, a = r - 1; i < a; i += c) {
				let n = l.getX(i), r = l.getX(i + 1), a = mh(this, e, lh, s, n, r, i);
				a && t.push(a);
			}
			if (this.isLineLoop) {
				let i = l.getX(r - 1), a = l.getX(n), o = mh(this, e, lh, s, i, a, r - 1);
				o && t.push(o);
			}
		} else {
			let n = Math.max(0, a.start), r = Math.min(u.count, a.start + a.count);
			for (let i = n, a = r - 1; i < a; i += c) {
				let n = mh(this, e, lh, s, i, i + 1, i);
				n && t.push(n);
			}
			if (this.isLineLoop) {
				let i = mh(this, e, lh, s, r - 1, n, r - 1);
				i && t.push(i);
			}
		}
	}
	updateMorphTargets() {
		let e = this.geometry.morphAttributes, t = Object.keys(e);
		if (t.length > 0) {
			let n = e[t[0]];
			if (n !== void 0) {
				this.morphTargetInfluences = [], this.morphTargetDictionary = {};
				for (let e = 0, t = n.length; e < t; e++) {
					let t = n[e].name || String(e);
					this.morphTargetInfluences.push(0), this.morphTargetDictionary[t] = e;
				}
			}
		}
	}
};
function mh(e, t, n, r, i, a, o) {
	let s = e.geometry.attributes.position;
	if (oh.fromBufferAttribute(s, i), sh.fromBufferAttribute(s, a), n.distanceSqToSegment(oh, sh, dh, fh) > r) return;
	dh.applyMatrix4(e.matrixWorld);
	let c = t.ray.origin.distanceTo(dh);
	if (!(c < t.near || c > t.far)) return {
		distance: c,
		point: fh.clone().applyMatrix4(e.matrixWorld),
		index: o,
		face: null,
		faceIndex: null,
		barycoord: null,
		object: e
	};
}
var hh = /*@__PURE__*/ new J(), gh = /*@__PURE__*/ new J(), _h = class extends ph {
	constructor(e, t) {
		super(e, t), this.isLineSegments = !0, this.type = "LineSegments";
	}
	computeLineDistances() {
		let e = this.geometry;
		if (e.index === null) {
			let t = e.attributes.position, n = [];
			for (let e = 0, r = t.count; e < r; e += 2) hh.fromBufferAttribute(t, e), gh.fromBufferAttribute(t, e + 1), n[e] = e === 0 ? 0 : n[e - 1], n[e + 1] = n[e] + hh.distanceTo(gh);
			e.setAttribute("lineDistance", new fm(n, 1));
		} else G("LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");
		return this;
	}
}, vh = class extends Em {
	constructor(e) {
		super(), this.isPointsMaterial = !0, this.type = "PointsMaterial", this.color = new Z(16777215), this.map = null, this.alphaMap = null, this.size = 1, this.sizeAttenuation = !0, this.fog = !0, this.setValues(e);
	}
	copy(e) {
		return super.copy(e), this.color.copy(e.color), this.map = e.map, this.alphaMap = e.alphaMap, this.size = e.size, this.sizeAttenuation = e.sizeAttenuation, this.fog = e.fog, this;
	}
}, yh = /*@__PURE__*/ new Jf(), bh = /*@__PURE__*/ new Pm(), xh = /*@__PURE__*/ new gm(), Sh = /*@__PURE__*/ new J(), Ch = class extends xp {
	constructor(e = new wm(), t = new vh()) {
		super(), this.isPoints = !0, this.type = "Points", this.geometry = e, this.material = t, this.morphTargetDictionary = void 0, this.morphTargetInfluences = void 0, this.updateMorphTargets();
	}
	copy(e, t) {
		return super.copy(e, t), this.material = Array.isArray(e.material) ? e.material.slice() : e.material, this.geometry = e.geometry, this;
	}
	raycast(e, t) {
		let n = this.geometry, r = this.matrixWorld, i = e.params.Points.threshold, a = n.drawRange;
		if (n.boundingSphere === null && n.computeBoundingSphere(), xh.copy(n.boundingSphere), xh.applyMatrix4(r), xh.radius += i, e.ray.intersectsSphere(xh) === !1) return;
		yh.copy(r).invert(), bh.copy(e.ray).applyMatrix4(yh);
		let o = i / ((this.scale.x + this.scale.y + this.scale.z) / 3), s = o * o, c = n.index, l = n.attributes.position;
		if (c !== null) {
			let n = Math.max(0, a.start), i = Math.min(c.count, a.start + a.count);
			for (let a = n, o = i; a < o; a++) {
				let n = c.getX(a);
				Sh.fromBufferAttribute(l, n), wh(Sh, n, s, r, e, t, this);
			}
		} else {
			let n = Math.max(0, a.start), i = Math.min(l.count, a.start + a.count);
			for (let a = n, o = i; a < o; a++) Sh.fromBufferAttribute(l, a), wh(Sh, a, s, r, e, t, this);
		}
	}
	updateMorphTargets() {
		let e = this.geometry.morphAttributes, t = Object.keys(e);
		if (t.length > 0) {
			let n = e[t[0]];
			if (n !== void 0) {
				this.morphTargetInfluences = [], this.morphTargetDictionary = {};
				for (let e = 0, t = n.length; e < t; e++) {
					let t = n[e].name || String(e);
					this.morphTargetInfluences.push(0), this.morphTargetDictionary[t] = e;
				}
			}
		}
	}
};
function wh(e, t, n, r, i, a, o) {
	let s = bh.distanceSqToPoint(e);
	if (s < n) {
		let n = new J();
		bh.closestPointToPoint(e, n), n.applyMatrix4(r);
		let c = i.ray.origin.distanceTo(n);
		if (c < i.near || c > i.far) return;
		a.push({
			distance: c,
			distanceToRay: Math.sqrt(s),
			point: n,
			index: t,
			face: null,
			faceIndex: null,
			barycoord: null,
			object: o
		});
	}
}
var Th = class extends Hf {
	constructor(e = [], t = 301, n, r, i, a, o, s, c, l) {
		super(e, t, n, r, i, a, o, s, c, l), this.isCubeTexture = !0, this.flipY = !1;
	}
	get images() {
		return this.image;
	}
	set images(e) {
		this.image = e;
	}
}, Eh = class extends Hf {
	constructor(e, t, n, r, i, a, o, s, c) {
		super(e, t, n, r, i, a, o, s, c), this.isCanvasTexture = !0, this.needsUpdate = !0;
	}
}, Dh = class extends Hf {
	constructor(e, t, n = Tu, r, i, a, o = mu, s = mu, c, l = Iu, u = 1) {
		if (l !== 1026 && l !== 1027) throw Error("THREE.DepthTexture: format must be either THREE.DepthFormat or THREE.DepthStencilFormat");
		super({
			width: e,
			height: t,
			depth: u
		}, r, i, a, o, s, l, n, c), this.isDepthTexture = !0, this.flipY = !1, this.generateMipmaps = !1, this.compareFunction = null;
	}
	copy(e) {
		return super.copy(e), this.source = new Rf(Object.assign({}, e.image)), this.compareFunction = e.compareFunction, this;
	}
	toJSON(e) {
		let t = super.toJSON(e);
		return this.compareFunction !== null && (t.compareFunction = this.compareFunction), t;
	}
}, Oh = class extends Dh {
	constructor(e, t = Tu, n = 301, r, i, a = mu, o = mu, s, c = Iu) {
		let l = {
			width: e,
			height: e,
			depth: 1
		}, u = [
			l,
			l,
			l,
			l,
			l,
			l
		];
		super(e, e, t, n, r, i, a, o, s, c), this.image = u, this.isCubeDepthTexture = !0, this.isCubeTexture = !0;
	}
	get images() {
		return this.image;
	}
	set images(e) {
		this.image = e;
	}
}, kh = class extends Hf {
	constructor(e = null) {
		super(), this.sourceTexture = e, this.isExternalTexture = !0;
	}
	copy(e) {
		return super.copy(e), this.sourceTexture = e.sourceTexture, this;
	}
}, Ah = class e extends wm {
	constructor(e = 1, t = 1, n = 1, r = 1, i = 1, a = 1) {
		super(), this.type = "BoxGeometry", this.parameters = {
			width: e,
			height: t,
			depth: n,
			widthSegments: r,
			heightSegments: i,
			depthSegments: a
		};
		let o = this;
		r = Math.floor(r), i = Math.floor(i), a = Math.floor(a);
		let s = [], c = [], l = [], u = [], d = 0, f = 0;
		p("z", "y", "x", -1, -1, n, t, e, a, i, 0), p("z", "y", "x", 1, -1, n, t, -e, a, i, 1), p("x", "z", "y", 1, 1, e, n, t, r, a, 2), p("x", "z", "y", 1, -1, e, n, -t, r, a, 3), p("x", "y", "z", 1, -1, e, t, n, r, i, 4), p("x", "y", "z", -1, -1, e, t, -n, r, i, 5), this.setIndex(s), this.setAttribute("position", new fm(c, 3)), this.setAttribute("normal", new fm(l, 3)), this.setAttribute("uv", new fm(u, 2));
		function p(e, t, n, r, i, a, p, m, h, g, _) {
			let v = a / h, y = p / g, b = a / 2, x = p / 2, S = m / 2, C = h + 1, w = g + 1, T = 0, E = 0, D = new J();
			for (let a = 0; a < w; a++) {
				let o = a * y - x;
				for (let s = 0; s < C; s++) D[e] = (s * v - b) * r, D[t] = o * i, D[n] = S, c.push(D.x, D.y, D.z), D[e] = 0, D[t] = 0, D[n] = m > 0 ? 1 : -1, l.push(D.x, D.y, D.z), u.push(s / h), u.push(1 - a / g), T += 1;
			}
			for (let e = 0; e < g; e++) for (let t = 0; t < h; t++) {
				let n = d + t + C * e, r = d + t + C * (e + 1), i = d + (t + 1) + C * (e + 1), a = d + (t + 1) + C * e;
				s.push(n, r, a), s.push(r, i, a), E += 6;
			}
			o.addGroup(f, E, _), f += E, d += T;
		}
	}
	copy(e) {
		return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
	}
	static fromJSON(t) {
		return new e(t.width, t.height, t.depth, t.widthSegments, t.heightSegments, t.depthSegments);
	}
}, jh = class e extends wm {
	constructor(e = 1, t = 1, n = 1, r = 1) {
		super(), this.type = "PlaneGeometry", this.parameters = {
			width: e,
			height: t,
			widthSegments: n,
			heightSegments: r
		};
		let i = e / 2, a = t / 2, o = Math.floor(n), s = Math.floor(r), c = o + 1, l = s + 1, u = e / o, d = t / s, f = [], p = [], m = [], h = [];
		for (let e = 0; e < l; e++) {
			let t = e * d - a;
			for (let n = 0; n < c; n++) {
				let r = n * u - i;
				p.push(r, -t, 0), m.push(0, 0, 1), h.push(n / o), h.push(1 - e / s);
			}
		}
		for (let e = 0; e < s; e++) for (let t = 0; t < o; t++) {
			let n = t + c * e, r = t + c * (e + 1), i = t + 1 + c * (e + 1), a = t + 1 + c * e;
			f.push(n, r, a), f.push(r, i, a);
		}
		this.setIndex(f), this.setAttribute("position", new fm(p, 3)), this.setAttribute("normal", new fm(m, 3)), this.setAttribute("uv", new fm(h, 2));
	}
	copy(e) {
		return super.copy(e), this.parameters = Object.assign({}, e.parameters), this;
	}
	static fromJSON(t) {
		return new e(t.width, t.height, t.widthSegments, t.heightSegments);
	}
};
function Mh(e) {
	let t = {};
	for (let n in e) {
		t[n] = {};
		for (let r in e[n]) {
			let i = e[n][r];
			if (Ph(i)) i.isRenderTargetTexture ? (G("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."), t[n][r] = null) : t[n][r] = i.clone();
			else if (Array.isArray(i)) if (Ph(i[0])) {
				let e = [];
				for (let t = 0, n = i.length; t < n; t++) e[t] = i[t].clone();
				t[n][r] = e;
			} else t[n][r] = i.slice();
			else t[n][r] = i;
		}
	}
	return t;
}
function Nh(e) {
	let t = {};
	for (let n = 0; n < e.length; n++) {
		let r = Mh(e[n]);
		for (let e in r) t[e] = r[e];
	}
	return t;
}
function Ph(e) {
	return e && (e.isColor || e.isMatrix3 || e.isMatrix4 || e.isVector2 || e.isVector3 || e.isVector4 || e.isTexture || e.isQuaternion);
}
function Fh(e) {
	let t = [];
	for (let n = 0; n < e.length; n++) t.push(e[n].clone());
	return t;
}
function Ih(e) {
	let t = e.getRenderTarget();
	return t === null ? e.outputColorSpace : t.isXRRenderTarget === !0 ? t.texture.colorSpace : X.workingColorSpace;
}
var Lh = {
	clone: Mh,
	merge: Nh
}, Rh = "void main() {\n	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n}", zh = "void main() {\n	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );\n}", Bh = class extends Em {
	constructor(e) {
		super(), this.isShaderMaterial = !0, this.type = "ShaderMaterial", this.defines = {}, this.uniforms = {}, this.uniformsGroups = [], this.vertexShader = Rh, this.fragmentShader = zh, this.linewidth = 1, this.wireframe = !1, this.wireframeLinewidth = 1, this.fog = !1, this.lights = !1, this.clipping = !1, this.forceSinglePass = !0, this.extensions = {
			clipCullDistance: !1,
			multiDraw: !1
		}, this.defaultAttributeValues = {
			color: [
				1,
				1,
				1
			],
			uv: [0, 0],
			uv1: [0, 0]
		}, this.index0AttributeName = void 0, this.uniformsNeedUpdate = !1, this.glslVersion = null, e !== void 0 && this.setValues(e);
	}
	copy(e) {
		return super.copy(e), this.fragmentShader = e.fragmentShader, this.vertexShader = e.vertexShader, this.uniforms = Mh(e.uniforms), this.uniformsGroups = Fh(e.uniformsGroups), this.defines = Object.assign({}, e.defines), this.wireframe = e.wireframe, this.wireframeLinewidth = e.wireframeLinewidth, this.fog = e.fog, this.lights = e.lights, this.clipping = e.clipping, this.extensions = Object.assign({}, e.extensions), this.glslVersion = e.glslVersion, this.defaultAttributeValues = Object.assign({}, e.defaultAttributeValues), this.index0AttributeName = e.index0AttributeName, this.uniformsNeedUpdate = e.uniformsNeedUpdate, this;
	}
	toJSON(e) {
		let t = super.toJSON(e);
		t.glslVersion = this.glslVersion, t.uniforms = {};
		for (let n in this.uniforms) {
			let r = this.uniforms[n].value;
			r && r.isTexture ? t.uniforms[n] = {
				type: "t",
				value: r.toJSON(e).uuid
			} : r && r.isColor ? t.uniforms[n] = {
				type: "c",
				value: r.getHex()
			} : r && r.isVector2 ? t.uniforms[n] = {
				type: "v2",
				value: r.toArray()
			} : r && r.isVector3 ? t.uniforms[n] = {
				type: "v3",
				value: r.toArray()
			} : r && r.isVector4 ? t.uniforms[n] = {
				type: "v4",
				value: r.toArray()
			} : r && r.isMatrix3 ? t.uniforms[n] = {
				type: "m3",
				value: r.toArray()
			} : r && r.isMatrix4 ? t.uniforms[n] = {
				type: "m4",
				value: r.toArray()
			} : t.uniforms[n] = { value: r };
		}
		Object.keys(this.defines).length > 0 && (t.defines = this.defines), t.vertexShader = this.vertexShader, t.fragmentShader = this.fragmentShader, t.lights = this.lights, t.clipping = this.clipping;
		let n = {};
		for (let e in this.extensions) this.extensions[e] === !0 && (n[e] = !0);
		return Object.keys(n).length > 0 && (t.extensions = n), t;
	}
	fromJSON(e, t) {
		if (super.fromJSON(e, t), e.uniforms !== void 0) for (let n in e.uniforms) {
			let r = e.uniforms[n];
			switch (this.uniforms[n] = {}, r.type) {
				case "t":
					this.uniforms[n].value = t[r.value] || null;
					break;
				case "c":
					this.uniforms[n].value = new Z().setHex(r.value);
					break;
				case "v2":
					this.uniforms[n].value = new Tf().fromArray(r.value);
					break;
				case "v3":
					this.uniforms[n].value = new J().fromArray(r.value);
					break;
				case "v4":
					this.uniforms[n].value = new Uf().fromArray(r.value);
					break;
				case "m3":
					this.uniforms[n].value = new Y().fromArray(r.value);
					break;
				case "m4":
					this.uniforms[n].value = new Jf().fromArray(r.value);
					break;
				default: this.uniforms[n].value = r.value;
			}
		}
		if (e.defines !== void 0 && (this.defines = e.defines), e.vertexShader !== void 0 && (this.vertexShader = e.vertexShader), e.fragmentShader !== void 0 && (this.fragmentShader = e.fragmentShader), e.glslVersion !== void 0 && (this.glslVersion = e.glslVersion), e.extensions !== void 0) for (let t in e.extensions) this.extensions[t] = e.extensions[t];
		return e.lights !== void 0 && (this.lights = e.lights), e.clipping !== void 0 && (this.clipping = e.clipping), this;
	}
}, Vh = class extends Bh {
	constructor(e) {
		super(e), this.isRawShaderMaterial = !0, this.type = "RawShaderMaterial";
	}
}, Hh = class extends Em {
	constructor(e) {
		super(), this.isMeshDepthMaterial = !0, this.type = "MeshDepthMaterial", this.depthPacking = Md, this.map = null, this.alphaMap = null, this.displacementMap = null, this.displacementScale = 1, this.displacementBias = 0, this.wireframe = !1, this.wireframeLinewidth = 1, this.setValues(e);
	}
	copy(e) {
		return super.copy(e), this.depthPacking = e.depthPacking, this.map = e.map, this.alphaMap = e.alphaMap, this.displacementMap = e.displacementMap, this.displacementScale = e.displacementScale, this.displacementBias = e.displacementBias, this.wireframe = e.wireframe, this.wireframeLinewidth = e.wireframeLinewidth, this;
	}
}, Uh = class extends Em {
	constructor(e) {
		super(), this.isMeshDistanceMaterial = !0, this.type = "MeshDistanceMaterial", this.map = null, this.alphaMap = null, this.displacementMap = null, this.displacementScale = 1, this.displacementBias = 0, this.setValues(e);
	}
	copy(e) {
		return super.copy(e), this.map = e.map, this.alphaMap = e.alphaMap, this.displacementMap = e.displacementMap, this.displacementScale = e.displacementScale, this.displacementBias = e.displacementBias, this;
	}
};
function Wh(e, t) {
	return !e || e.constructor === t ? e : typeof t.BYTES_PER_ELEMENT == "number" ? new t(e) : Array.prototype.slice.call(e);
}
var Gh = class {
	constructor(e, t, n, r) {
		this.parameterPositions = e, this._cachedIndex = 0, this.resultBuffer = r === void 0 ? new t.constructor(n) : r, this.sampleValues = t, this.valueSize = n, this.settings = null, this.DefaultSettings_ = {};
	}
	evaluate(e) {
		let t = this.parameterPositions, n = this._cachedIndex, r = t[n], i = t[n - 1];
		validate_interval: {
			seek: {
				let a;
				linear_scan: {
					forward_scan: if (!(e < r)) {
						for (let a = n + 2;;) {
							if (r === void 0) {
								if (e < i) break forward_scan;
								return n = t.length, this._cachedIndex = n, this.copySampleValue_(n - 1);
							}
							if (n === a) break;
							if (i = r, r = t[++n], e < r) break seek;
						}
						a = t.length;
						break linear_scan;
					}
					if (!(e >= i)) {
						let o = t[1];
						e < o && (n = 2, i = o);
						for (let a = n - 2;;) {
							if (i === void 0) return this._cachedIndex = 0, this.copySampleValue_(0);
							if (n === a) break;
							if (r = i, i = t[--n - 1], e >= i) break seek;
						}
						a = n, n = 0;
						break linear_scan;
					}
					break validate_interval;
				}
				for (; n < a;) {
					let r = n + a >>> 1;
					e < t[r] ? a = r : n = r + 1;
				}
				if (r = t[n], i = t[n - 1], i === void 0) return this._cachedIndex = 0, this.copySampleValue_(0);
				if (r === void 0) return n = t.length, this._cachedIndex = n, this.copySampleValue_(n - 1);
			}
			this._cachedIndex = n, this.intervalChanged_(n, i, r);
		}
		return this.interpolate_(n, i, e, r);
	}
	getSettings_() {
		return this.settings || this.DefaultSettings_;
	}
	copySampleValue_(e) {
		let t = this.resultBuffer, n = this.sampleValues, r = this.valueSize, i = e * r;
		for (let e = 0; e !== r; ++e) t[e] = n[i + e];
		return t;
	}
	interpolate_() {
		throw Error("THREE.Interpolant: Call to abstract method.");
	}
	intervalChanged_() {}
}, Kh = class extends Gh {
	constructor(e, t, n, r) {
		super(e, t, n, r), this._weightPrev = -0, this._offsetPrev = -0, this._weightNext = -0, this._offsetNext = -0, this.DefaultSettings_ = {
			endingStart: kd,
			endingEnd: kd
		};
	}
	intervalChanged_(e, t, n) {
		let r = this.parameterPositions, i = e - 2, a = e + 1, o = r[i], s = r[a];
		if (o === void 0) switch (this.getSettings_().endingStart) {
			case Ad:
				i = e, o = 2 * t - n;
				break;
			case jd:
				i = r.length - 2, o = t + r[i] - r[i + 1];
				break;
			default: i = e, o = n;
		}
		if (s === void 0) switch (this.getSettings_().endingEnd) {
			case Ad:
				a = e, s = 2 * n - t;
				break;
			case jd:
				a = 1, s = n + r[1] - r[0];
				break;
			default: a = e - 1, s = t;
		}
		let c = (n - t) * .5, l = this.valueSize;
		this._weightPrev = c / (t - o), this._weightNext = c / (s - n), this._offsetPrev = i * l, this._offsetNext = a * l;
	}
	interpolate_(e, t, n, r) {
		let i = this.resultBuffer, a = this.sampleValues, o = this.valueSize, s = e * o, c = s - o, l = this._offsetPrev, u = this._offsetNext, d = this._weightPrev, f = this._weightNext, p = (n - t) / (r - t), m = p * p, h = m * p, g = -d * h + 2 * d * m - d * p, _ = (1 + d) * h + (-1.5 - 2 * d) * m + (-.5 + d) * p + 1, v = (-1 - f) * h + (1.5 + f) * m + .5 * p, y = f * h - f * m;
		for (let e = 0; e !== o; ++e) i[e] = g * a[l + e] + _ * a[c + e] + v * a[s + e] + y * a[u + e];
		return i;
	}
}, qh = class extends Gh {
	constructor(e, t, n, r) {
		super(e, t, n, r);
	}
	interpolate_(e, t, n, r) {
		let i = this.resultBuffer, a = this.sampleValues, o = this.valueSize, s = e * o, c = s - o, l = (n - t) / (r - t), u = 1 - l;
		for (let e = 0; e !== o; ++e) i[e] = a[c + e] * u + a[s + e] * l;
		return i;
	}
}, Jh = class extends Gh {
	constructor(e, t, n, r) {
		super(e, t, n, r);
	}
	interpolate_(e) {
		return this.copySampleValue_(e - 1);
	}
}, Yh = class extends Gh {
	interpolate_(e, t, n, r) {
		let i = this.resultBuffer, a = this.sampleValues, o = this.valueSize, s = e * o, c = s - o, l = this.inTangents, u = this.outTangents;
		if (!l || !u) {
			let e = (n - t) / (r - t), l = 1 - e;
			for (let t = 0; t !== o; ++t) i[t] = a[c + t] * l + a[s + t] * e;
			return i;
		}
		let d = o * 2, f = e - 1;
		for (let p = 0; p !== o; ++p) {
			let o = a[c + p], m = a[s + p], h = f * d + p * 2, g = u[h], _ = u[h + 1], v = e * d + p * 2, y = l[v], b = l[v + 1], x = (n - t) / (r - t), S, C, w, T, E;
			for (let e = 0; e < 8; e++) {
				S = x * x, C = S * x, w = 1 - x, T = w * w, E = T * w;
				let e = E * t + 3 * T * x * g + 3 * w * S * y + C * r - n;
				if (Math.abs(e) < 1e-10) break;
				let i = 3 * T * (g - t) + 6 * w * x * (y - g) + 3 * S * (r - y);
				if (Math.abs(i) < 1e-10) break;
				x -= e / i, x = Math.max(0, Math.min(1, x));
			}
			i[p] = E * o + 3 * T * x * _ + 3 * w * S * b + C * m;
		}
		return i;
	}
}, Xh = class {
	constructor(e, t, n, r) {
		if (e === void 0) throw Error("THREE.KeyframeTrack: track name is undefined");
		if (t === void 0 || t.length === 0) throw Error("THREE.KeyframeTrack: no keyframes in track named " + e);
		this.name = e, this.times = Wh(t, this.TimeBufferType), this.values = Wh(n, this.ValueBufferType), this.setInterpolation(r || this.DefaultInterpolation);
	}
	static toJSON(e) {
		let t = e.constructor, n;
		if (t.toJSON !== this.toJSON) n = t.toJSON(e);
		else {
			n = {
				name: e.name,
				times: Wh(e.times, Array),
				values: Wh(e.values, Array)
			};
			let t = e.getInterpolation();
			t !== e.DefaultInterpolation && (n.interpolation = t);
		}
		return n.type = e.ValueTypeName, n;
	}
	InterpolantFactoryMethodDiscrete(e) {
		return new Jh(this.times, this.values, this.getValueSize(), e);
	}
	InterpolantFactoryMethodLinear(e) {
		return new qh(this.times, this.values, this.getValueSize(), e);
	}
	InterpolantFactoryMethodSmooth(e) {
		return new Kh(this.times, this.values, this.getValueSize(), e);
	}
	InterpolantFactoryMethodBezier(e) {
		let t = new Yh(this.times, this.values, this.getValueSize(), e);
		return this.settings && (t.inTangents = this.settings.inTangents, t.outTangents = this.settings.outTangents), t;
	}
	setInterpolation(e) {
		let t;
		switch (e) {
			case Td:
				t = this.InterpolantFactoryMethodDiscrete;
				break;
			case Ed:
				t = this.InterpolantFactoryMethodLinear;
				break;
			case Dd:
				t = this.InterpolantFactoryMethodSmooth;
				break;
			case Od:
				t = this.InterpolantFactoryMethodBezier;
				break;
		}
		if (t === void 0) {
			let t = "unsupported interpolation for " + this.ValueTypeName + " keyframe track named " + this.name;
			if (this.createInterpolant === void 0) if (e !== this.DefaultInterpolation) this.setInterpolation(this.DefaultInterpolation);
			else throw Error(t);
			return G("KeyframeTrack:", t), this;
		}
		return this.createInterpolant = t, this;
	}
	getInterpolation() {
		switch (this.createInterpolant) {
			case this.InterpolantFactoryMethodDiscrete: return Td;
			case this.InterpolantFactoryMethodLinear: return Ed;
			case this.InterpolantFactoryMethodSmooth: return Dd;
			case this.InterpolantFactoryMethodBezier: return Od;
		}
	}
	getValueSize() {
		return this.values.length / this.times.length;
	}
	shift(e) {
		if (e !== 0) {
			let t = this.times;
			for (let n = 0, r = t.length; n !== r; ++n) t[n] += e;
		}
		return this;
	}
	scale(e) {
		if (e !== 1) {
			let t = this.times;
			for (let n = 0, r = t.length; n !== r; ++n) t[n] *= e;
		}
		return this;
	}
	trim(e, t) {
		let n = this.times, r = n.length, i = 0, a = r - 1;
		for (; i !== r && n[i] < e;) ++i;
		for (; a !== -1 && n[a] > t;) --a;
		if (++a, i !== 0 || a !== r) {
			i >= a && (a = Math.max(a, 1), i = a - 1);
			let e = this.getValueSize();
			this.times = n.slice(i, a), this.values = this.values.slice(i * e, a * e);
		}
		return this;
	}
	validate() {
		let e = !0, t = this.getValueSize();
		t - Math.floor(t) !== 0 && (K("KeyframeTrack: Invalid value size in track.", this), e = !1);
		let n = this.times, r = this.values, i = n.length;
		i === 0 && (K("KeyframeTrack: Track is empty.", this), e = !1);
		let a = null;
		for (let t = 0; t !== i; t++) {
			let r = n[t];
			if (typeof r == "number" && isNaN(r)) {
				K("KeyframeTrack: Time is not a valid number.", this, t, r), e = !1;
				break;
			}
			if (a !== null && a > r) {
				K("KeyframeTrack: Out of order keys.", this, t, r, a), e = !1;
				break;
			}
			a = r;
		}
		if (r !== void 0 && Hd(r)) for (let t = 0, n = r.length; t !== n; ++t) {
			let n = r[t];
			if (isNaN(n)) {
				K("KeyframeTrack: Value is not a valid number.", this, t, n), e = !1;
				break;
			}
		}
		return e;
	}
	optimize() {
		let e = this.times.slice(), t = this.values.slice(), n = this.getValueSize(), r = this.getInterpolation() === Dd, i = e.length - 1, a = 1;
		for (let o = 1; o < i; ++o) {
			let i = !1, s = e[o];
			if (s !== e[o + 1] && (o !== 1 || s !== e[0])) if (r) i = !0;
			else {
				let e = o * n, r = e - n, a = e + n;
				for (let o = 0; o !== n; ++o) {
					let n = t[e + o];
					if (n !== t[r + o] || n !== t[a + o]) {
						i = !0;
						break;
					}
				}
			}
			if (i) {
				if (o !== a) {
					e[a] = e[o];
					let r = o * n, i = a * n;
					for (let e = 0; e !== n; ++e) t[i + e] = t[r + e];
				}
				++a;
			}
		}
		if (i > 0) {
			e[a] = e[i];
			for (let e = i * n, r = a * n, o = 0; o !== n; ++o) t[r + o] = t[e + o];
			++a;
		}
		return a === e.length ? (this.times = e, this.values = t) : (this.times = e.slice(0, a), this.values = t.slice(0, a * n)), this;
	}
	clone() {
		let e = this.times.slice(), t = this.values.slice(), n = this.constructor, r = new n(this.name, e, t);
		return r.createInterpolant = this.createInterpolant, r;
	}
};
Xh.prototype.ValueTypeName = "", Xh.prototype.TimeBufferType = Float32Array, Xh.prototype.ValueBufferType = Float32Array, Xh.prototype.DefaultInterpolation = Ed;
var Zh = class extends Xh {
	constructor(e, t, n) {
		super(e, t, n);
	}
};
Zh.prototype.ValueTypeName = "bool", Zh.prototype.ValueBufferType = Array, Zh.prototype.DefaultInterpolation = Td, Zh.prototype.InterpolantFactoryMethodLinear = void 0, Zh.prototype.InterpolantFactoryMethodSmooth = void 0;
var Qh = class extends Xh {
	constructor(e, t, n, r) {
		super(e, t, n, r);
	}
};
Qh.prototype.ValueTypeName = "color";
var $h = class extends Xh {
	constructor(e, t, n, r) {
		super(e, t, n, r);
	}
};
$h.prototype.ValueTypeName = "number";
var eg = class extends Gh {
	constructor(e, t, n, r) {
		super(e, t, n, r);
	}
	interpolate_(e, t, n, r) {
		let i = this.resultBuffer, a = this.sampleValues, o = this.valueSize, s = (n - t) / (r - t), c = e * o;
		for (let e = c + o; c !== e; c += 4) Ef.slerpFlat(i, 0, a, c - o, a, c, s);
		return i;
	}
}, tg = class extends Xh {
	constructor(e, t, n, r) {
		super(e, t, n, r);
	}
	InterpolantFactoryMethodLinear(e) {
		return new eg(this.times, this.values, this.getValueSize(), e);
	}
};
tg.prototype.ValueTypeName = "quaternion", tg.prototype.InterpolantFactoryMethodSmooth = void 0;
var ng = class extends Xh {
	constructor(e, t, n) {
		super(e, t, n);
	}
};
ng.prototype.ValueTypeName = "string", ng.prototype.ValueBufferType = Array, ng.prototype.DefaultInterpolation = Td, ng.prototype.InterpolantFactoryMethodLinear = void 0, ng.prototype.InterpolantFactoryMethodSmooth = void 0;
var rg = class extends Xh {
	constructor(e, t, n, r) {
		super(e, t, n, r);
	}
};
rg.prototype.ValueTypeName = "vector";
var ig = /*@__PURE__*/ new class {
	constructor(e, t, n) {
		let r = this, i = !1, a = 0, o = 0, s, c = [];
		this.onStart = void 0, this.onLoad = e, this.onProgress = t, this.onError = n, this._abortController = null, this.itemStart = function(e) {
			o++, i === !1 && r.onStart !== void 0 && r.onStart(e, a, o), i = !0;
		}, this.itemEnd = function(e) {
			a++, r.onProgress !== void 0 && r.onProgress(e, a, o), a === o && (i = !1, r.onLoad !== void 0 && r.onLoad());
		}, this.itemError = function(e) {
			r.onError !== void 0 && r.onError(e);
		}, this.resolveURL = function(e) {
			return e = e.normalize("NFC"), s ? s(e) : e;
		}, this.setURLModifier = function(e) {
			return s = e, this;
		}, this.addHandler = function(e, t) {
			return c.push(e, t), this;
		}, this.removeHandler = function(e) {
			let t = c.indexOf(e);
			return t !== -1 && c.splice(t, 2), this;
		}, this.getHandler = function(e) {
			for (let t = 0, n = c.length; t < n; t += 2) {
				let n = c[t], r = c[t + 1];
				if (n.global && (n.lastIndex = 0), n.test(e)) return r;
			}
			return null;
		}, this.abort = function() {
			return this.abortController.abort(), this._abortController = null, this;
		};
	}
	get abortController() {
		return this._abortController ||= new AbortController(), this._abortController;
	}
}(), ag = class {
	constructor(e) {
		this.manager = e === void 0 ? ig : e, this.crossOrigin = "anonymous", this.withCredentials = !1, this.path = "", this.resourcePath = "", this.requestHeader = {}, typeof __THREE_DEVTOOLS__ < "u" && __THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe", { detail: this }));
	}
	load() {}
	loadAsync(e, t) {
		let n = this;
		return new Promise(function(r, i) {
			n.load(e, r, t, i);
		});
	}
	parse() {}
	setCrossOrigin(e) {
		return this.crossOrigin = e, this;
	}
	setWithCredentials(e) {
		return this.withCredentials = e, this;
	}
	setPath(e) {
		return this.path = e, this;
	}
	setResourcePath(e) {
		return this.resourcePath = e, this;
	}
	setRequestHeader(e) {
		return this.requestHeader = e, this;
	}
	abort() {
		return this;
	}
};
ag.DEFAULT_MATERIAL_NAME = "__DEFAULT";
var og = /*@__PURE__*/ new J(), sg = /*@__PURE__*/ new Ef(), cg = /*@__PURE__*/ new J(), lg = class extends xp {
	constructor() {
		super(), this.isCamera = !0, this.type = "Camera", this.matrixWorldInverse = new Jf(), this.projectionMatrix = new Jf(), this.projectionMatrixInverse = new Jf(), this.coordinateSystem = Bd, this._reversedDepth = !1;
	}
	get reversedDepth() {
		return this._reversedDepth;
	}
	copy(e, t) {
		return super.copy(e, t), this.matrixWorldInverse.copy(e.matrixWorldInverse), this.projectionMatrix.copy(e.projectionMatrix), this.projectionMatrixInverse.copy(e.projectionMatrixInverse), this.coordinateSystem = e.coordinateSystem, this;
	}
	getWorldDirection(e) {
		return super.getWorldDirection(e).negate();
	}
	updateMatrixWorld(e) {
		super.updateMatrixWorld(e), this.matrixWorld.decompose(og, sg, cg), cg.x === 1 && cg.y === 1 && cg.z === 1 ? this.matrixWorldInverse.copy(this.matrixWorld).invert() : this.matrixWorldInverse.compose(og, sg, cg.set(1, 1, 1)).invert();
	}
	updateWorldMatrix(e, t, n = !1) {
		super.updateWorldMatrix(e, t, n), this.matrixWorld.decompose(og, sg, cg), cg.x === 1 && cg.y === 1 && cg.z === 1 ? this.matrixWorldInverse.copy(this.matrixWorld).invert() : this.matrixWorldInverse.compose(og, sg, cg.set(1, 1, 1)).invert();
	}
	clone() {
		return new this.constructor().copy(this);
	}
}, ug = /*@__PURE__*/ new J(), dg = /*@__PURE__*/ new Tf(), fg = /*@__PURE__*/ new Tf(), pg = class extends lg {
	constructor(e = 50, t = 1, n = .1, r = 2e3) {
		super(), this.isPerspectiveCamera = !0, this.type = "PerspectiveCamera", this.fov = e, this.zoom = 1, this.near = n, this.far = r, this.focus = 10, this.aspect = t, this.view = null, this.filmGauge = 35, this.filmOffset = 0, this.updateProjectionMatrix();
	}
	copy(e, t) {
		return super.copy(e, t), this.fov = e.fov, this.zoom = e.zoom, this.near = e.near, this.far = e.far, this.focus = e.focus, this.aspect = e.aspect, this.view = e.view === null ? null : Object.assign({}, e.view), this.filmGauge = e.filmGauge, this.filmOffset = e.filmOffset, this;
	}
	setFocalLength(e) {
		let t = .5 * this.getFilmHeight() / e;
		this.fov = tf * 2 * Math.atan(t), this.updateProjectionMatrix();
	}
	getFocalLength() {
		let e = Math.tan(ef * .5 * this.fov);
		return .5 * this.getFilmHeight() / e;
	}
	getEffectiveFOV() {
		return tf * 2 * Math.atan(Math.tan(ef * .5 * this.fov) / this.zoom);
	}
	getFilmWidth() {
		return this.filmGauge * Math.min(this.aspect, 1);
	}
	getFilmHeight() {
		return this.filmGauge / Math.max(this.aspect, 1);
	}
	getViewBounds(e, t, n) {
		ug.set(-1, -1, .5).applyMatrix4(this.projectionMatrixInverse), t.set(ug.x, ug.y).multiplyScalar(-e / ug.z), ug.set(1, 1, .5).applyMatrix4(this.projectionMatrixInverse), n.set(ug.x, ug.y).multiplyScalar(-e / ug.z);
	}
	getViewSize(e, t) {
		return this.getViewBounds(e, dg, fg), t.subVectors(fg, dg);
	}
	setViewOffset(e, t, n, r, i, a) {
		this.aspect = e / t, this.view === null && (this.view = {
			enabled: !0,
			fullWidth: 1,
			fullHeight: 1,
			offsetX: 0,
			offsetY: 0,
			width: 1,
			height: 1
		}), this.view.enabled = !0, this.view.fullWidth = e, this.view.fullHeight = t, this.view.offsetX = n, this.view.offsetY = r, this.view.width = i, this.view.height = a, this.updateProjectionMatrix();
	}
	clearViewOffset() {
		this.view !== null && (this.view.enabled = !1), this.updateProjectionMatrix();
	}
	updateProjectionMatrix() {
		let e = this.near, t = e * Math.tan(ef * .5 * this.fov) / this.zoom, n = 2 * t, r = this.aspect * n, i = -.5 * r, a = this.view;
		if (this.view !== null && this.view.enabled) {
			let e = a.fullWidth, o = a.fullHeight;
			i += a.offsetX * r / e, t -= a.offsetY * n / o, r *= a.width / e, n *= a.height / o;
		}
		let o = this.filmOffset;
		o !== 0 && (i += e * o / this.getFilmWidth()), this.projectionMatrix.makePerspective(i, i + r, t, t - n, e, this.far, this.coordinateSystem, this.reversedDepth), this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
	}
	toJSON(e) {
		let t = super.toJSON(e);
		return t.object.fov = this.fov, t.object.zoom = this.zoom, t.object.near = this.near, t.object.far = this.far, t.object.focus = this.focus, t.object.aspect = this.aspect, this.view !== null && (t.object.view = Object.assign({}, this.view)), t.object.filmGauge = this.filmGauge, t.object.filmOffset = this.filmOffset, t;
	}
}, mg = class extends lg {
	constructor(e = -1, t = 1, n = 1, r = -1, i = .1, a = 2e3) {
		super(), this.isOrthographicCamera = !0, this.type = "OrthographicCamera", this.zoom = 1, this.view = null, this.left = e, this.right = t, this.top = n, this.bottom = r, this.near = i, this.far = a, this.updateProjectionMatrix();
	}
	copy(e, t) {
		return super.copy(e, t), this.left = e.left, this.right = e.right, this.top = e.top, this.bottom = e.bottom, this.near = e.near, this.far = e.far, this.zoom = e.zoom, this.view = e.view === null ? null : Object.assign({}, e.view), this;
	}
	setViewOffset(e, t, n, r, i, a) {
		this.view === null && (this.view = {
			enabled: !0,
			fullWidth: 1,
			fullHeight: 1,
			offsetX: 0,
			offsetY: 0,
			width: 1,
			height: 1
		}), this.view.enabled = !0, this.view.fullWidth = e, this.view.fullHeight = t, this.view.offsetX = n, this.view.offsetY = r, this.view.width = i, this.view.height = a, this.updateProjectionMatrix();
	}
	clearViewOffset() {
		this.view !== null && (this.view.enabled = !1), this.updateProjectionMatrix();
	}
	updateProjectionMatrix() {
		let e = (this.right - this.left) / (2 * this.zoom), t = (this.top - this.bottom) / (2 * this.zoom), n = (this.right + this.left) / 2, r = (this.top + this.bottom) / 2, i = n - e, a = n + e, o = r + t, s = r - t;
		if (this.view !== null && this.view.enabled) {
			let e = (this.right - this.left) / this.view.fullWidth / this.zoom, t = (this.top - this.bottom) / this.view.fullHeight / this.zoom;
			i += e * this.view.offsetX, a = i + e * this.view.width, o -= t * this.view.offsetY, s = o - t * this.view.height;
		}
		this.projectionMatrix.makeOrthographic(i, a, o, s, this.near, this.far, this.coordinateSystem, this.reversedDepth), this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
	}
	toJSON(e) {
		let t = super.toJSON(e);
		return t.object.zoom = this.zoom, t.object.left = this.left, t.object.right = this.right, t.object.top = this.top, t.object.bottom = this.bottom, t.object.near = this.near, t.object.far = this.far, this.view !== null && (t.object.view = Object.assign({}, this.view)), t;
	}
}, hg = -90, gg = 1, _g = class extends xp {
	constructor(e, t, n) {
		super(), this.type = "CubeCamera", this.renderTarget = n, this.coordinateSystem = null, this.activeMipmapLevel = 0;
		let r = new pg(hg, gg, e, t);
		r.layers = this.layers, this.add(r);
		let i = new pg(hg, gg, e, t);
		i.layers = this.layers, this.add(i);
		let a = new pg(hg, gg, e, t);
		a.layers = this.layers, this.add(a);
		let o = new pg(hg, gg, e, t);
		o.layers = this.layers, this.add(o);
		let s = new pg(hg, gg, e, t);
		s.layers = this.layers, this.add(s);
		let c = new pg(hg, gg, e, t);
		c.layers = this.layers, this.add(c);
	}
	updateCoordinateSystem() {
		let e = this.coordinateSystem, t = this.children.concat(), [n, r, i, a, o, s] = t;
		for (let e of t) this.remove(e);
		if (e === 2e3) n.up.set(0, 1, 0), n.lookAt(1, 0, 0), r.up.set(0, 1, 0), r.lookAt(-1, 0, 0), i.up.set(0, 0, -1), i.lookAt(0, 1, 0), a.up.set(0, 0, 1), a.lookAt(0, -1, 0), o.up.set(0, 1, 0), o.lookAt(0, 0, 1), s.up.set(0, 1, 0), s.lookAt(0, 0, -1);
		else if (e === 2001) n.up.set(0, -1, 0), n.lookAt(-1, 0, 0), r.up.set(0, -1, 0), r.lookAt(1, 0, 0), i.up.set(0, 0, 1), i.lookAt(0, 1, 0), a.up.set(0, 0, -1), a.lookAt(0, -1, 0), o.up.set(0, -1, 0), o.lookAt(0, 0, 1), s.up.set(0, -1, 0), s.lookAt(0, 0, -1);
		else throw Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: " + e);
		for (let e of t) this.add(e), e.updateMatrixWorld();
	}
	update(e, t) {
		this.parent === null && this.updateMatrixWorld();
		let { renderTarget: n, activeMipmapLevel: r } = this;
		this.coordinateSystem !== e.coordinateSystem && (this.coordinateSystem = e.coordinateSystem, this.updateCoordinateSystem());
		let [i, a, o, s, c, l] = this.children, u = e.getRenderTarget(), d = e.getActiveCubeFace(), f = e.getActiveMipmapLevel(), p = e.xr.enabled;
		e.xr.enabled = !1;
		let m = n.texture.generateMipmaps;
		n.texture.generateMipmaps = !1;
		let h = !1;
		h = e.isWebGLRenderer === !0 ? e.state.buffers.depth.getReversed() : e.reversedDepthBuffer, e.setRenderTarget(n, 0, r), h && e.autoClear === !1 && e.clearDepth(), e.render(t, i), e.setRenderTarget(n, 1, r), h && e.autoClear === !1 && e.clearDepth(), e.render(t, a), e.setRenderTarget(n, 2, r), h && e.autoClear === !1 && e.clearDepth(), e.render(t, o), e.setRenderTarget(n, 3, r), h && e.autoClear === !1 && e.clearDepth(), e.render(t, s), e.setRenderTarget(n, 4, r), h && e.autoClear === !1 && e.clearDepth(), e.render(t, c), n.texture.generateMipmaps = m, e.setRenderTarget(n, 5, r), h && e.autoClear === !1 && e.clearDepth(), e.render(t, l), e.setRenderTarget(u, d, f), e.xr.enabled = p, n.texture.needsPMREMUpdate = !0;
	}
}, vg = class extends pg {
	constructor(e = []) {
		super(), this.isArrayCamera = !0, this.isMultiViewCamera = !1, this.cameras = e;
	}
}, yg = "\\[\\]\\.:\\/", bg = /* @__PURE__ */ RegExp("[\\[\\]\\.:\\/]", "g"), xg = "[^\\[\\]\\.:\\/]", Sg = "[^" + yg.replace("\\.", "") + "]", Cg = /*@__PURE__*/ "((?:WC+[\\/:])*)".replace("WC", xg), wg = /*@__PURE__*/ "(WCOD+)?".replace("WCOD", Sg), Tg = /*@__PURE__*/ "(?:\\.(WC+)(?:\\[(.+)\\])?)?".replace("WC", xg), Eg = /*@__PURE__*/ "\\.(WC+)(?:\\[(.+)\\])?".replace("WC", xg), Dg = RegExp("^" + Cg + wg + Tg + Eg + "$"), Og = [
	"material",
	"materials",
	"bones",
	"map"
], kg = class {
	constructor(e, t, n) {
		let r = n || Ag.parseTrackName(t);
		this._targetGroup = e, this._bindings = e.subscribe_(t, r);
	}
	getValue(e, t) {
		this.bind();
		let n = this._targetGroup.nCachedObjects_, r = this._bindings[n];
		r !== void 0 && r.getValue(e, t);
	}
	setValue(e, t) {
		let n = this._bindings;
		for (let r = this._targetGroup.nCachedObjects_, i = n.length; r !== i; ++r) n[r].setValue(e, t);
	}
	bind() {
		let e = this._bindings;
		for (let t = this._targetGroup.nCachedObjects_, n = e.length; t !== n; ++t) e[t].bind();
	}
	unbind() {
		let e = this._bindings;
		for (let t = this._targetGroup.nCachedObjects_, n = e.length; t !== n; ++t) e[t].unbind();
	}
}, Ag = class e {
	constructor(t, n, r) {
		this.path = n, this.parsedPath = r || e.parseTrackName(n), this.node = e.findNode(t, this.parsedPath.nodeName), this.rootNode = t, this.getValue = this._getValue_unbound, this.setValue = this._setValue_unbound;
	}
	static create(t, n, r) {
		return t && t.isAnimationObjectGroup ? new e.Composite(t, n, r) : new e(t, n, r);
	}
	static sanitizeNodeName(e) {
		return e.replace(/\s/g, "_").replace(bg, "");
	}
	static parseTrackName(e) {
		let t = Dg.exec(e);
		if (t === null) throw Error("THREE.PropertyBinding: Cannot parse trackName: " + e);
		let n = {
			nodeName: t[2],
			objectName: t[3],
			objectIndex: t[4],
			propertyName: t[5],
			propertyIndex: t[6]
		}, r = n.nodeName && n.nodeName.lastIndexOf(".");
		if (r !== void 0 && r !== -1) {
			let e = n.nodeName.substring(r + 1);
			Og.indexOf(e) !== -1 && (n.nodeName = n.nodeName.substring(0, r), n.objectName = e);
		}
		if (n.propertyName === null || n.propertyName.length === 0) throw Error("THREE.PropertyBinding: can not parse propertyName from trackName: " + e);
		return n;
	}
	static findNode(e, t) {
		if (t === void 0 || t === "" || t === "." || t === -1 || t === e.name || t === e.uuid) return e;
		if (e.skeleton) {
			let n = e.skeleton.getBoneByName(t);
			if (n !== void 0) return n;
		}
		if (e.children) {
			let n = function(e) {
				for (let r = 0; r < e.length; r++) {
					let i = e[r];
					if (i.name === t || i.uuid === t) return i;
					let a = n(i.children);
					if (a) return a;
				}
				return null;
			}, r = n(e.children);
			if (r) return r;
		}
		return null;
	}
	_getValue_unavailable() {}
	_setValue_unavailable() {}
	_getValue_direct(e, t) {
		e[t] = this.targetObject[this.propertyName];
	}
	_getValue_array(e, t) {
		let n = this.resolvedProperty;
		for (let r = 0, i = n.length; r !== i; ++r) e[t++] = n[r];
	}
	_getValue_arrayElement(e, t) {
		e[t] = this.resolvedProperty[this.propertyIndex];
	}
	_getValue_toArray(e, t) {
		this.resolvedProperty.toArray(e, t);
	}
	_setValue_direct(e, t) {
		this.targetObject[this.propertyName] = e[t];
	}
	_setValue_direct_setNeedsUpdate(e, t) {
		this.targetObject[this.propertyName] = e[t], this.targetObject.needsUpdate = !0;
	}
	_setValue_direct_setMatrixWorldNeedsUpdate(e, t) {
		this.targetObject[this.propertyName] = e[t], this.targetObject.matrixWorldNeedsUpdate = !0;
	}
	_setValue_array(e, t) {
		let n = this.resolvedProperty;
		for (let r = 0, i = n.length; r !== i; ++r) n[r] = e[t++];
	}
	_setValue_array_setNeedsUpdate(e, t) {
		let n = this.resolvedProperty;
		for (let r = 0, i = n.length; r !== i; ++r) n[r] = e[t++];
		this.targetObject.needsUpdate = !0;
	}
	_setValue_array_setMatrixWorldNeedsUpdate(e, t) {
		let n = this.resolvedProperty;
		for (let r = 0, i = n.length; r !== i; ++r) n[r] = e[t++];
		this.targetObject.matrixWorldNeedsUpdate = !0;
	}
	_setValue_arrayElement(e, t) {
		this.resolvedProperty[this.propertyIndex] = e[t];
	}
	_setValue_arrayElement_setNeedsUpdate(e, t) {
		this.resolvedProperty[this.propertyIndex] = e[t], this.targetObject.needsUpdate = !0;
	}
	_setValue_arrayElement_setMatrixWorldNeedsUpdate(e, t) {
		this.resolvedProperty[this.propertyIndex] = e[t], this.targetObject.matrixWorldNeedsUpdate = !0;
	}
	_setValue_fromArray(e, t) {
		this.resolvedProperty.fromArray(e, t);
	}
	_setValue_fromArray_setNeedsUpdate(e, t) {
		this.resolvedProperty.fromArray(e, t), this.targetObject.needsUpdate = !0;
	}
	_setValue_fromArray_setMatrixWorldNeedsUpdate(e, t) {
		this.resolvedProperty.fromArray(e, t), this.targetObject.matrixWorldNeedsUpdate = !0;
	}
	_getValue_unbound(e, t) {
		this.bind(), this.getValue(e, t);
	}
	_setValue_unbound(e, t) {
		this.bind(), this.setValue(e, t);
	}
	bind() {
		let t = this.node, n = this.parsedPath, r = n.objectName, i = n.propertyName, a = n.propertyIndex;
		if (t || (t = e.findNode(this.rootNode, n.nodeName), this.node = t), this.getValue = this._getValue_unavailable, this.setValue = this._setValue_unavailable, !t) {
			G("PropertyBinding: No target node found for track: " + this.path + ".");
			return;
		}
		if (r) {
			let e = n.objectIndex;
			switch (r) {
				case "materials":
					if (!t.material) {
						K("PropertyBinding: Can not bind to material as node does not have a material.", this);
						return;
					}
					if (!t.material.materials) {
						K("PropertyBinding: Can not bind to material.materials as node.material does not have a materials array.", this);
						return;
					}
					t = t.material.materials;
					break;
				case "bones":
					if (!t.skeleton) {
						K("PropertyBinding: Can not bind to bones as node does not have a skeleton.", this);
						return;
					}
					t = t.skeleton.bones;
					for (let n = 0; n < t.length; n++) if (t[n].name === e) {
						e = n;
						break;
					}
					break;
				case "map":
					if ("map" in t) {
						t = t.map;
						break;
					}
					if (!t.material) {
						K("PropertyBinding: Can not bind to material as node does not have a material.", this);
						return;
					}
					if (!t.material.map) {
						K("PropertyBinding: Can not bind to material.map as node.material does not have a map.", this);
						return;
					}
					t = t.material.map;
					break;
				default:
					if (t[r] === void 0) {
						K("PropertyBinding: Can not bind to objectName of node undefined.", this);
						return;
					}
					t = t[r];
			}
			if (e !== void 0) {
				if (t[e] === void 0) {
					K("PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined.", this, t);
					return;
				}
				t = t[e];
			}
		}
		let o = t[i];
		if (o === void 0) {
			let e = n.nodeName;
			K("PropertyBinding: Trying to update property for track: " + e + "." + i + " but it wasn't found.", t);
			return;
		}
		let s = this.Versioning.None;
		this.targetObject = t, t.isMaterial === !0 ? s = this.Versioning.NeedsUpdate : t.isObject3D === !0 && (s = this.Versioning.MatrixWorldNeedsUpdate);
		let c = this.BindingType.Direct;
		if (a !== void 0) {
			if (i === "morphTargetInfluences") {
				if (!t.geometry) {
					K("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.", this);
					return;
				}
				if (!t.geometry.morphAttributes) {
					K("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.morphAttributes.", this);
					return;
				}
				t.morphTargetDictionary[a] !== void 0 && (a = t.morphTargetDictionary[a]);
			}
			c = this.BindingType.ArrayElement, this.resolvedProperty = o, this.propertyIndex = a;
		} else o.fromArray !== void 0 && o.toArray !== void 0 ? (c = this.BindingType.HasFromToArray, this.resolvedProperty = o) : Array.isArray(o) ? (c = this.BindingType.EntireArray, this.resolvedProperty = o) : this.propertyName = i;
		this.getValue = this.GetterByBindingType[c], this.setValue = this.SetterByBindingTypeAndVersioning[c][s];
	}
	unbind() {
		this.node = null, this.getValue = this._getValue_unbound, this.setValue = this._setValue_unbound;
	}
};
Ag.Composite = kg, Ag.prototype.BindingType = {
	Direct: 0,
	EntireArray: 1,
	ArrayElement: 2,
	HasFromToArray: 3
}, Ag.prototype.Versioning = {
	None: 0,
	NeedsUpdate: 1,
	MatrixWorldNeedsUpdate: 2
}, Ag.prototype.GetterByBindingType = [
	Ag.prototype._getValue_direct,
	Ag.prototype._getValue_array,
	Ag.prototype._getValue_arrayElement,
	Ag.prototype._getValue_toArray
], Ag.prototype.SetterByBindingTypeAndVersioning = [
	[
		Ag.prototype._setValue_direct,
		Ag.prototype._setValue_direct_setNeedsUpdate,
		Ag.prototype._setValue_direct_setMatrixWorldNeedsUpdate
	],
	[
		Ag.prototype._setValue_array,
		Ag.prototype._setValue_array_setNeedsUpdate,
		Ag.prototype._setValue_array_setMatrixWorldNeedsUpdate
	],
	[
		Ag.prototype._setValue_arrayElement,
		Ag.prototype._setValue_arrayElement_setNeedsUpdate,
		Ag.prototype._setValue_arrayElement_setMatrixWorldNeedsUpdate
	],
	[
		Ag.prototype._setValue_fromArray,
		Ag.prototype._setValue_fromArray_setNeedsUpdate,
		Ag.prototype._setValue_fromArray_setMatrixWorldNeedsUpdate
	]
];
var jg = /*@__PURE__*/ new Jf(), Mg = class {
	constructor(e, t, n = 0, r = Infinity) {
		this.ray = new Pm(e, t), this.near = n, this.far = r, this.camera = null, this.layers = new ap(), this.params = {
			Mesh: {},
			Line: { threshold: 1 },
			LOD: {},
			Points: { threshold: 1 },
			Sprite: {}
		};
	}
	set(e, t) {
		this.ray.set(e, t);
	}
	setFromCamera(e, t) {
		t.isPerspectiveCamera ? (this.ray.origin.setFromMatrixPosition(t.matrixWorld), this.ray.direction.set(e.x, e.y, .5).unproject(t).sub(this.ray.origin).normalize(), this.camera = t) : t.isOrthographicCamera ? (this.ray.origin.set(e.x, e.y, t.projectionMatrix.elements[14]).unproject(t), this.ray.direction.set(0, 0, -1).transformDirection(t.matrixWorld), this.camera = t) : K("Raycaster: Unsupported camera type: " + t.type);
	}
	setFromXRController(e) {
		return jg.identity().extractRotation(e.matrixWorld), this.ray.origin.setFromMatrixPosition(e.matrixWorld), this.ray.direction.set(0, 0, -1).applyMatrix4(jg), this;
	}
	intersectObject(e, t = !0, n = []) {
		return Pg(e, this, n, t), n.sort(Ng), n;
	}
	intersectObjects(e, t = !0, n = []) {
		for (let r = 0, i = e.length; r < i; r++) Pg(e[r], this, n, t);
		return n.sort(Ng), n;
	}
};
function Ng(e, t) {
	return e.distance - t.distance;
}
function Pg(e, t, n, r) {
	let i = !0;
	if (e.layers.test(t.layers) && e.raycast(t, n) === !1 && (i = !1), i === !0 && r === !0) {
		let r = e.children;
		for (let e = 0, i = r.length; e < i; e++) Pg(r[e], t, n, !0);
	}
}
(class e {
	static {
		e.prototype.isMatrix2 = !0;
	}
	constructor(e, t, n, r) {
		this.elements = [
			1,
			0,
			0,
			1
		], e !== void 0 && this.set(e, t, n, r);
	}
	identity() {
		return this.set(1, 0, 0, 1), this;
	}
	fromArray(e, t = 0) {
		for (let n = 0; n < 4; n++) this.elements[n] = e[n + t];
		return this;
	}
	set(e, t, n, r) {
		let i = this.elements;
		return i[0] = e, i[2] = t, i[1] = n, i[3] = r, this;
	}
});
function Fg(e, t, n, r) {
	let i = Ig(r);
	switch (n) {
		case Nu: return e * t;
		case Ru: return e * t / i.components * i.byteLength;
		case zu: return e * t / i.components * i.byteLength;
		case Bu: return e * t * 2 / i.components * i.byteLength;
		case Vu: return e * t * 2 / i.components * i.byteLength;
		case Pu: return e * t * 3 / i.components * i.byteLength;
		case Fu: return e * t * 4 / i.components * i.byteLength;
		case Hu: return e * t * 4 / i.components * i.byteLength;
		case Uu:
		case Wu: return Math.floor((e + 3) / 4) * Math.floor((t + 3) / 4) * 8;
		case Gu:
		case Ku: return Math.floor((e + 3) / 4) * Math.floor((t + 3) / 4) * 16;
		case Ju:
		case Xu: return Math.max(e, 16) * Math.max(t, 8) / 4;
		case qu:
		case Yu: return Math.max(e, 8) * Math.max(t, 8) / 2;
		case Zu:
		case Qu:
		case ed:
		case td: return Math.floor((e + 3) / 4) * Math.floor((t + 3) / 4) * 8;
		case $u:
		case nd:
		case rd: return Math.floor((e + 3) / 4) * Math.floor((t + 3) / 4) * 16;
		case id: return Math.floor((e + 3) / 4) * Math.floor((t + 3) / 4) * 16;
		case ad: return Math.floor((e + 4) / 5) * Math.floor((t + 3) / 4) * 16;
		case od: return Math.floor((e + 4) / 5) * Math.floor((t + 4) / 5) * 16;
		case sd: return Math.floor((e + 5) / 6) * Math.floor((t + 4) / 5) * 16;
		case cd: return Math.floor((e + 5) / 6) * Math.floor((t + 5) / 6) * 16;
		case ld: return Math.floor((e + 7) / 8) * Math.floor((t + 4) / 5) * 16;
		case ud: return Math.floor((e + 7) / 8) * Math.floor((t + 5) / 6) * 16;
		case dd: return Math.floor((e + 7) / 8) * Math.floor((t + 7) / 8) * 16;
		case fd: return Math.floor((e + 9) / 10) * Math.floor((t + 4) / 5) * 16;
		case pd: return Math.floor((e + 9) / 10) * Math.floor((t + 5) / 6) * 16;
		case md: return Math.floor((e + 9) / 10) * Math.floor((t + 7) / 8) * 16;
		case hd: return Math.floor((e + 9) / 10) * Math.floor((t + 9) / 10) * 16;
		case gd: return Math.floor((e + 11) / 12) * Math.floor((t + 9) / 10) * 16;
		case _d: return Math.floor((e + 11) / 12) * Math.floor((t + 11) / 12) * 16;
		case vd:
		case yd:
		case bd: return Math.ceil(e / 4) * Math.ceil(t / 4) * 16;
		case xd:
		case Sd: return Math.ceil(e / 4) * Math.ceil(t / 4) * 8;
		case Cd:
		case wd: return Math.ceil(e / 4) * Math.ceil(t / 4) * 16;
	}
	throw Error(`Unable to determine texture byte length for ${n} format.`);
}
function Ig(e) {
	switch (e) {
		case bu:
		case xu: return {
			byteLength: 1,
			components: 1
		};
		case Cu:
		case Su:
		case Du: return {
			byteLength: 2,
			components: 1
		};
		case Ou:
		case ku: return {
			byteLength: 2,
			components: 4
		};
		case Tu:
		case wu:
		case Eu: return {
			byteLength: 4,
			components: 1
		};
		case ju:
		case Mu: return {
			byteLength: 4,
			components: 3
		};
	}
	throw Error(`THREE.TextureUtils: Unknown texture type ${e}.`);
}
typeof __THREE_DEVTOOLS__ < "u" && __THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register", { detail: { revision: "185" } })), typeof window < "u" && (window.__THREE__ ? G("WARNING: Multiple instances of Three.js being imported.") : window.__THREE__ = "185");
//#endregion
//#region node_modules/three/build/three.module.js
function Lg() {
	let e = null, t = !1, n = null, r = null;
	function i(t, a) {
		n(t, a), r = e.requestAnimationFrame(i);
	}
	return {
		start: function() {
			t !== !0 && n !== null && e !== null && (r = e.requestAnimationFrame(i), t = !0);
		},
		stop: function() {
			e !== null && e.cancelAnimationFrame(r), t = !1;
		},
		setAnimationLoop: function(e) {
			n = e;
		},
		setContext: function(t) {
			e = t;
		}
	};
}
function Rg(e) {
	let t = /* @__PURE__ */ new WeakMap();
	function n(t, n) {
		let r = t.array, i = t.usage, a = r.byteLength, o = e.createBuffer();
		e.bindBuffer(n, o), e.bufferData(n, r, i), t.onUploadCallback();
		let s;
		if (r instanceof Float32Array) s = e.FLOAT;
		else if (typeof Float16Array < "u" && r instanceof Float16Array) s = e.HALF_FLOAT;
		else if (r instanceof Uint16Array) s = t.isFloat16BufferAttribute ? e.HALF_FLOAT : e.UNSIGNED_SHORT;
		else if (r instanceof Int16Array) s = e.SHORT;
		else if (r instanceof Uint32Array) s = e.UNSIGNED_INT;
		else if (r instanceof Int32Array) s = e.INT;
		else if (r instanceof Int8Array) s = e.BYTE;
		else if (r instanceof Uint8Array) s = e.UNSIGNED_BYTE;
		else if (r instanceof Uint8ClampedArray) s = e.UNSIGNED_BYTE;
		else throw Error("THREE.WebGLAttributes: Unsupported buffer data format: " + r);
		return {
			buffer: o,
			type: s,
			bytesPerElement: r.BYTES_PER_ELEMENT,
			version: t.version,
			size: a
		};
	}
	function r(t, n, r) {
		let i = n.array, a = n.updateRanges;
		if (e.bindBuffer(r, t), a.length === 0) e.bufferSubData(r, 0, i);
		else {
			a.sort((e, t) => e.start - t.start);
			let t = 0;
			for (let e = 1; e < a.length; e++) {
				let n = a[t], r = a[e];
				r.start <= n.start + n.count + 1 ? n.count = Math.max(n.count, r.start + r.count - n.start) : (++t, a[t] = r);
			}
			a.length = t + 1;
			for (let t = 0, n = a.length; t < n; t++) {
				let n = a[t];
				e.bufferSubData(r, n.start * i.BYTES_PER_ELEMENT, i, n.start, n.count);
			}
			n.clearUpdateRanges();
		}
		n.onUploadCallback();
	}
	function i(e) {
		return e.isInterleavedBufferAttribute && (e = e.data), t.get(e);
	}
	function a(n) {
		n.isInterleavedBufferAttribute && (n = n.data);
		let r = t.get(n);
		r && (e.deleteBuffer(r.buffer), t.delete(n));
	}
	function o(e, i) {
		if (e.isInterleavedBufferAttribute && (e = e.data), e.isGLBufferAttribute) {
			let n = t.get(e);
			(!n || n.version < e.version) && t.set(e, {
				buffer: e.buffer,
				type: e.type,
				bytesPerElement: e.elementSize,
				version: e.version
			});
			return;
		}
		let a = t.get(e);
		if (a === void 0) t.set(e, n(e, i));
		else if (a.version < e.version) {
			if (a.size !== e.array.byteLength) throw Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");
			r(a.buffer, e, i), a.version = e.version;
		}
	}
	return {
		get: i,
		remove: a,
		update: o
	};
}
var Q = {
	alphahash_fragment: "#ifdef USE_ALPHAHASH\n	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;\n#endif",
	alphahash_pars_fragment: "#ifdef USE_ALPHAHASH\n	const float ALPHA_HASH_SCALE = 0.05;\n	float hash2D( vec2 value ) {\n		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );\n	}\n	float hash3D( vec3 value ) {\n		return hash2D( vec2( hash2D( value.xy ), value.z ) );\n	}\n	float getAlphaHashThreshold( vec3 position ) {\n		float maxDeriv = max(\n			length( dFdx( position.xyz ) ),\n			length( dFdy( position.xyz ) )\n		);\n		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );\n		vec2 pixScales = vec2(\n			exp2( floor( log2( pixScale ) ) ),\n			exp2( ceil( log2( pixScale ) ) )\n		);\n		vec2 alpha = vec2(\n			hash3D( floor( pixScales.x * position.xyz ) ),\n			hash3D( floor( pixScales.y * position.xyz ) )\n		);\n		float lerpFactor = fract( log2( pixScale ) );\n		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;\n		float a = min( lerpFactor, 1.0 - lerpFactor );\n		vec3 cases = vec3(\n			x * x / ( 2.0 * a * ( 1.0 - a ) ),\n			( x - 0.5 * a ) / ( 1.0 - a ),\n			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )\n		);\n		float threshold = ( x < ( 1.0 - a ) )\n			? ( ( x < a ) ? cases.x : cases.y )\n			: cases.z;\n		return clamp( threshold , 1.0e-6, 1.0 );\n	}\n#endif",
	alphamap_fragment: "#ifdef USE_ALPHAMAP\n	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;\n#endif",
	alphamap_pars_fragment: "#ifdef USE_ALPHAMAP\n	uniform sampler2D alphaMap;\n#endif",
	alphatest_fragment: "#ifdef USE_ALPHATEST\n	#ifdef ALPHA_TO_COVERAGE\n	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );\n	if ( diffuseColor.a == 0.0 ) discard;\n	#else\n	if ( diffuseColor.a < alphaTest ) discard;\n	#endif\n#endif",
	alphatest_pars_fragment: "#ifdef USE_ALPHATEST\n	uniform float alphaTest;\n#endif",
	aomap_fragment: "#ifdef USE_AOMAP\n	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;\n	reflectedLight.indirectDiffuse *= ambientOcclusion;\n	#if defined( USE_CLEARCOAT ) \n		clearcoatSpecularIndirect *= ambientOcclusion;\n	#endif\n	#if defined( USE_SHEEN ) \n		sheenSpecularIndirect *= ambientOcclusion;\n	#endif\n	#if defined( USE_ENVMAP ) && defined( STANDARD )\n		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );\n		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );\n	#endif\n#endif",
	aomap_pars_fragment: "#ifdef USE_AOMAP\n	uniform sampler2D aoMap;\n	uniform float aoMapIntensity;\n#endif",
	batching_pars_vertex: "#ifdef USE_BATCHING\n	#if ! defined( GL_ANGLE_multi_draw )\n	#define gl_DrawID _gl_DrawID\n	uniform int _gl_DrawID;\n	#endif\n	uniform highp sampler2D batchingTexture;\n	uniform highp usampler2D batchingIdTexture;\n	mat4 getBatchingMatrix( const in float i ) {\n		int size = textureSize( batchingTexture, 0 ).x;\n		int j = int( i ) * 4;\n		int x = j % size;\n		int y = j / size;\n		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );\n		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );\n		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );\n		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );\n		return mat4( v1, v2, v3, v4 );\n	}\n	float getIndirectIndex( const in int i ) {\n		int size = textureSize( batchingIdTexture, 0 ).x;\n		int x = i % size;\n		int y = i / size;\n		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );\n	}\n#endif\n#ifdef USE_BATCHING_COLOR\n	uniform sampler2D batchingColorTexture;\n	vec4 getBatchingColor( const in float i ) {\n		int size = textureSize( batchingColorTexture, 0 ).x;\n		int j = int( i );\n		int x = j % size;\n		int y = j / size;\n		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 );\n	}\n#endif",
	batching_vertex: "#ifdef USE_BATCHING\n	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );\n#endif",
	begin_vertex: "vec3 transformed = vec3( position );\n#ifdef USE_ALPHAHASH\n	vPosition = vec3( position );\n#endif",
	beginnormal_vertex: "vec3 objectNormal = vec3( normal );\n#ifdef USE_TANGENT\n	vec3 objectTangent = vec3( tangent.xyz );\n#endif",
	bsdfs: "float G_BlinnPhong_Implicit( ) {\n	return 0.25;\n}\nfloat D_BlinnPhong( const in float shininess, const in float dotNH ) {\n	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );\n}\nvec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {\n	vec3 halfDir = normalize( lightDir + viewDir );\n	float dotNH = saturate( dot( normal, halfDir ) );\n	float dotVH = saturate( dot( viewDir, halfDir ) );\n	vec3 F = F_Schlick( specularColor, 1.0, dotVH );\n	float G = G_BlinnPhong_Implicit( );\n	float D = D_BlinnPhong( shininess, dotNH );\n	return F * ( G * D );\n} // validated",
	iridescence_fragment: "#ifdef USE_IRIDESCENCE\n	const mat3 XYZ_TO_REC709 = mat3(\n		 3.2404542, -0.9692660,  0.0556434,\n		-1.5371385,  1.8760108, -0.2040259,\n		-0.4985314,  0.0415560,  1.0572252\n	);\n	vec3 Fresnel0ToIor( vec3 fresnel0 ) {\n		vec3 sqrtF0 = sqrt( fresnel0 );\n		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );\n	}\n	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {\n		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );\n	}\n	float IorToFresnel0( float transmittedIor, float incidentIor ) {\n		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));\n	}\n	vec3 evalSensitivity( float OPD, vec3 shift ) {\n		float phase = 2.0 * PI * OPD * 1.0e-9;\n		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );\n		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );\n		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );\n		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );\n		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );\n		xyz /= 1.0685e-7;\n		vec3 rgb = XYZ_TO_REC709 * xyz;\n		return rgb;\n	}\n	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {\n		vec3 I;\n		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );\n		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );\n		float cosTheta2Sq = 1.0 - sinTheta2Sq;\n		if ( cosTheta2Sq < 0.0 ) {\n			return vec3( 1.0 );\n		}\n		float cosTheta2 = sqrt( cosTheta2Sq );\n		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );\n		float R12 = F_Schlick( R0, 1.0, cosTheta1 );\n		float T121 = 1.0 - R12;\n		float phi12 = 0.0;\n		if ( iridescenceIOR < outsideIOR ) phi12 = PI;\n		float phi21 = PI - phi12;\n		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );\n		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );\n		vec3 phi23 = vec3( 0.0 );\n		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;\n		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;\n		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;\n		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;\n		vec3 phi = vec3( phi21 ) + phi23;\n		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );\n		vec3 r123 = sqrt( R123 );\n		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );\n		vec3 C0 = R12 + Rs;\n		I = C0;\n		vec3 Cm = Rs - T121;\n		for ( int m = 1; m <= 2; ++ m ) {\n			Cm *= r123;\n			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );\n			I += Cm * Sm;\n		}\n		return max( I, vec3( 0.0 ) );\n	}\n#endif",
	bumpmap_pars_fragment: "#ifdef USE_BUMPMAP\n	uniform sampler2D bumpMap;\n	uniform float bumpScale;\n	vec2 dHdxy_fwd() {\n		vec2 dSTdx = dFdx( vBumpMapUv );\n		vec2 dSTdy = dFdy( vBumpMapUv );\n		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;\n		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;\n		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;\n		return vec2( dBx, dBy );\n	}\n	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {\n		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );\n		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );\n		vec3 vN = surf_norm;\n		vec3 R1 = cross( vSigmaY, vN );\n		vec3 R2 = cross( vN, vSigmaX );\n		float fDet = dot( vSigmaX, R1 ) * faceDirection;\n		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );\n		return normalize( abs( fDet ) * surf_norm - vGrad );\n	}\n#endif",
	clipping_planes_fragment: "#if NUM_CLIPPING_PLANES > 0\n	vec4 plane;\n	#ifdef ALPHA_TO_COVERAGE\n		float distanceToPlane, distanceGradient;\n		float clipOpacity = 1.0;\n		#pragma unroll_loop_start\n		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {\n			plane = clippingPlanes[ i ];\n			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;\n			distanceGradient = fwidth( distanceToPlane ) / 2.0;\n			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );\n			if ( clipOpacity == 0.0 ) discard;\n		}\n		#pragma unroll_loop_end\n		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES\n			float unionClipOpacity = 1.0;\n			#pragma unroll_loop_start\n			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {\n				plane = clippingPlanes[ i ];\n				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;\n				distanceGradient = fwidth( distanceToPlane ) / 2.0;\n				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );\n			}\n			#pragma unroll_loop_end\n			clipOpacity *= 1.0 - unionClipOpacity;\n		#endif\n		diffuseColor.a *= clipOpacity;\n		if ( diffuseColor.a == 0.0 ) discard;\n	#else\n		#pragma unroll_loop_start\n		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {\n			plane = clippingPlanes[ i ];\n			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;\n		}\n		#pragma unroll_loop_end\n		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES\n			bool clipped = true;\n			#pragma unroll_loop_start\n			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {\n				plane = clippingPlanes[ i ];\n				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;\n			}\n			#pragma unroll_loop_end\n			if ( clipped ) discard;\n		#endif\n	#endif\n#endif",
	clipping_planes_pars_fragment: "#if NUM_CLIPPING_PLANES > 0\n	varying vec3 vClipPosition;\n	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];\n#endif",
	clipping_planes_pars_vertex: "#if NUM_CLIPPING_PLANES > 0\n	varying vec3 vClipPosition;\n#endif",
	clipping_planes_vertex: "#if NUM_CLIPPING_PLANES > 0\n	vClipPosition = - mvPosition.xyz;\n#endif",
	color_fragment: "#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )\n	diffuseColor *= vColor;\n#endif",
	color_pars_fragment: "#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )\n	varying vec4 vColor;\n#endif",
	color_pars_vertex: "#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )\n	varying vec4 vColor;\n#endif",
	color_vertex: "#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )\n	vColor = vec4( 1.0 );\n#endif\n#ifdef USE_COLOR_ALPHA\n	vColor *= color;\n#elif defined( USE_COLOR )\n	vColor.rgb *= color;\n#endif\n#ifdef USE_INSTANCING_COLOR\n	vColor.rgb *= instanceColor.rgb;\n#endif\n#ifdef USE_BATCHING_COLOR\n	vColor *= getBatchingColor( getIndirectIndex( gl_DrawID ) );\n#endif",
	common: "#define PI 3.141592653589793\n#define PI2 6.283185307179586\n#define PI_HALF 1.5707963267948966\n#define RECIPROCAL_PI 0.3183098861837907\n#define RECIPROCAL_PI2 0.15915494309189535\n#define EPSILON 1e-6\n#ifndef saturate\n#define saturate( a ) clamp( a, 0.0, 1.0 )\n#endif\n#define whiteComplement( a ) ( 1.0 - saturate( a ) )\nfloat pow2( const in float x ) { return x*x; }\nvec3 pow2( const in vec3 x ) { return x*x; }\nfloat pow3( const in float x ) { return x*x*x; }\nfloat pow4( const in float x ) { float x2 = x*x; return x2*x2; }\nfloat max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }\nfloat average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }\nhighp float rand( const in vec2 uv ) {\n	const highp float a = 12.9898, b = 78.233, c = 43758.5453;\n	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );\n	return fract( sin( sn ) * c );\n}\n#ifdef HIGH_PRECISION\n	float precisionSafeLength( vec3 v ) { return length( v ); }\n#else\n	float precisionSafeLength( vec3 v ) {\n		float maxComponent = max3( abs( v ) );\n		return length( v / maxComponent ) * maxComponent;\n	}\n#endif\nstruct IncidentLight {\n	vec3 color;\n	vec3 direction;\n	bool visible;\n};\nstruct ReflectedLight {\n	vec3 directDiffuse;\n	vec3 directSpecular;\n	vec3 indirectDiffuse;\n	vec3 indirectSpecular;\n};\n#ifdef USE_ALPHAHASH\n	varying vec3 vPosition;\n#endif\nvec3 transformDirection( in vec3 dir, in mat4 matrix ) {\n	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );\n}\n#define inverseTransformDirection transformDirectionByInverseViewMatrix\nvec3 transformNormalByInverseViewMatrix( in vec3 normal, in mat4 viewMatrix ) {\n	return normalize( ( vec4( normal, 0.0 ) * viewMatrix ).xyz );\n}\nvec3 transformDirectionByInverseViewMatrix( in vec3 dir, in mat4 viewMatrix ) {\n	return normalize( ( vec4( dir, 0.0 ) * viewMatrix ).xyz );\n}\nbool isPerspectiveMatrix( mat4 m ) {\n	return m[ 2 ][ 3 ] == - 1.0;\n}\nvec2 equirectUv( in vec3 dir ) {\n	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;\n	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;\n	return vec2( u, v );\n}\nvec3 BRDF_Lambert( const in vec3 diffuseColor ) {\n	return RECIPROCAL_PI * diffuseColor;\n}\nvec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {\n	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );\n	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );\n}\nfloat F_Schlick( const in float f0, const in float f90, const in float dotVH ) {\n	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );\n	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );\n} // validated",
	cube_uv_reflection_fragment: "#ifdef ENVMAP_TYPE_CUBE_UV\n	#define cubeUV_minMipLevel 4.0\n	#define cubeUV_minTileSize 16.0\n	float getFace( vec3 direction ) {\n		vec3 absDirection = abs( direction );\n		float face = - 1.0;\n		if ( absDirection.x > absDirection.z ) {\n			if ( absDirection.x > absDirection.y )\n				face = direction.x > 0.0 ? 0.0 : 3.0;\n			else\n				face = direction.y > 0.0 ? 1.0 : 4.0;\n		} else {\n			if ( absDirection.z > absDirection.y )\n				face = direction.z > 0.0 ? 2.0 : 5.0;\n			else\n				face = direction.y > 0.0 ? 1.0 : 4.0;\n		}\n		return face;\n	}\n	vec2 getUV( vec3 direction, float face ) {\n		vec2 uv;\n		if ( face == 0.0 ) {\n			uv = vec2( direction.z, direction.y ) / abs( direction.x );\n		} else if ( face == 1.0 ) {\n			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );\n		} else if ( face == 2.0 ) {\n			uv = vec2( - direction.x, direction.y ) / abs( direction.z );\n		} else if ( face == 3.0 ) {\n			uv = vec2( - direction.z, direction.y ) / abs( direction.x );\n		} else if ( face == 4.0 ) {\n			uv = vec2( - direction.x, direction.z ) / abs( direction.y );\n		} else {\n			uv = vec2( direction.x, direction.y ) / abs( direction.z );\n		}\n		return 0.5 * ( uv + 1.0 );\n	}\n	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {\n		float face = getFace( direction );\n		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );\n		mipInt = max( mipInt, cubeUV_minMipLevel );\n		float faceSize = exp2( mipInt );\n		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;\n		if ( face > 2.0 ) {\n			uv.y += faceSize;\n			face -= 3.0;\n		}\n		uv.x += face * faceSize;\n		uv.x += filterInt * 3.0 * cubeUV_minTileSize;\n		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );\n		uv.x *= CUBEUV_TEXEL_WIDTH;\n		uv.y *= CUBEUV_TEXEL_HEIGHT;\n		#ifdef texture2DGradEXT\n			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;\n		#else\n			return texture2D( envMap, uv ).rgb;\n		#endif\n	}\n	#define cubeUV_r0 1.0\n	#define cubeUV_m0 - 2.0\n	#define cubeUV_r1 0.8\n	#define cubeUV_m1 - 1.0\n	#define cubeUV_r4 0.4\n	#define cubeUV_m4 2.0\n	#define cubeUV_r5 0.305\n	#define cubeUV_m5 3.0\n	#define cubeUV_r6 0.21\n	#define cubeUV_m6 4.0\n	float roughnessToMip( float roughness ) {\n		float mip = 0.0;\n		if ( roughness >= cubeUV_r1 ) {\n			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;\n		} else if ( roughness >= cubeUV_r4 ) {\n			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;\n		} else if ( roughness >= cubeUV_r5 ) {\n			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;\n		} else if ( roughness >= cubeUV_r6 ) {\n			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;\n		} else {\n			mip = - 2.0 * log2( 1.16 * roughness );		}\n		return mip;\n	}\n	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {\n		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );\n		float mipF = fract( mip );\n		float mipInt = floor( mip );\n		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );\n		if ( mipF == 0.0 ) {\n			return vec4( color0, 1.0 );\n		} else {\n			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );\n			return vec4( mix( color0, color1, mipF ), 1.0 );\n		}\n	}\n#endif",
	defaultnormal_vertex: "vec3 transformedNormal = objectNormal;\n#ifdef USE_TANGENT\n	vec3 transformedTangent = objectTangent;\n#endif\n#ifdef USE_BATCHING\n	mat3 bm = mat3( batchingMatrix );\n	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );\n	transformedNormal = bm * transformedNormal;\n	#ifdef USE_TANGENT\n		transformedTangent = bm * transformedTangent;\n	#endif\n#endif\n#ifdef USE_INSTANCING\n	mat3 im = mat3( instanceMatrix );\n	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );\n	transformedNormal = im * transformedNormal;\n	#ifdef USE_TANGENT\n		transformedTangent = im * transformedTangent;\n	#endif\n#endif\ntransformedNormal = normalMatrix * transformedNormal;\n#ifdef FLIP_SIDED\n	transformedNormal = - transformedNormal;\n#endif\n#ifdef USE_TANGENT\n	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;\n#endif",
	displacementmap_pars_vertex: "#ifdef USE_DISPLACEMENTMAP\n	uniform sampler2D displacementMap;\n	uniform float displacementScale;\n	uniform float displacementBias;\n#endif",
	displacementmap_vertex: "#ifdef USE_DISPLACEMENTMAP\n	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );\n#endif",
	emissivemap_fragment: "#ifdef USE_EMISSIVEMAP\n	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );\n	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE\n		emissiveColor = sRGBTransferEOTF( emissiveColor );\n	#endif\n	totalEmissiveRadiance *= emissiveColor.rgb;\n#endif",
	emissivemap_pars_fragment: "#ifdef USE_EMISSIVEMAP\n	uniform sampler2D emissiveMap;\n#endif",
	colorspace_fragment: "gl_FragColor = linearToOutputTexel( gl_FragColor );",
	colorspace_pars_fragment: "vec4 LinearTransferOETF( in vec4 value ) {\n	return value;\n}\nvec4 sRGBTransferEOTF( in vec4 value ) {\n	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );\n}\nvec4 sRGBTransferOETF( in vec4 value ) {\n	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );\n}",
	envmap_fragment: "#ifdef USE_ENVMAP\n	#ifdef ENV_WORLDPOS\n		vec3 cameraToFrag;\n		if ( isOrthographic ) {\n			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );\n		} else {\n			cameraToFrag = normalize( vWorldPosition - cameraPosition );\n		}\n		vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );\n		#ifdef ENVMAP_MODE_REFLECTION\n			vec3 reflectVec = reflect( cameraToFrag, worldNormal );\n		#else\n			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );\n		#endif\n	#else\n		vec3 reflectVec = vReflect;\n	#endif\n	#ifdef ENVMAP_TYPE_CUBE\n		vec4 envColor = textureCube( envMap, envMapRotation * reflectVec );\n		#ifdef ENVMAP_BLENDING_MULTIPLY\n			outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );\n		#elif defined( ENVMAP_BLENDING_MIX )\n			outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );\n		#elif defined( ENVMAP_BLENDING_ADD )\n			outgoingLight += envColor.xyz * specularStrength * reflectivity;\n		#endif\n	#endif\n#endif",
	envmap_common_pars_fragment: "#ifdef USE_ENVMAP\n	uniform float envMapIntensity;\n	uniform mat3 envMapRotation;\n	#ifdef ENVMAP_TYPE_CUBE\n		uniform samplerCube envMap;\n	#else\n		uniform sampler2D envMap;\n	#endif\n#endif",
	envmap_pars_fragment: "#ifdef USE_ENVMAP\n	uniform float reflectivity;\n	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )\n		#define ENV_WORLDPOS\n	#endif\n	#ifdef ENV_WORLDPOS\n		varying vec3 vWorldPosition;\n		uniform float refractionRatio;\n	#else\n		varying vec3 vReflect;\n	#endif\n#endif",
	envmap_pars_vertex: "#ifdef USE_ENVMAP\n	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )\n		#define ENV_WORLDPOS\n	#endif\n	#ifdef ENV_WORLDPOS\n		\n		varying vec3 vWorldPosition;\n	#else\n		varying vec3 vReflect;\n		uniform float refractionRatio;\n	#endif\n#endif",
	envmap_physical_pars_fragment: "#ifdef USE_ENVMAP\n	vec3 getIBLIrradiance( const in vec3 normal ) {\n		#ifdef ENVMAP_TYPE_CUBE_UV\n			vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );\n			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );\n			return PI * envMapColor.rgb * envMapIntensity;\n		#else\n			return vec3( 0.0 );\n		#endif\n	}\n	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {\n		#ifdef ENVMAP_TYPE_CUBE_UV\n			vec3 reflectVec = reflect( - viewDir, normal );\n			reflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );\n			reflectVec = transformDirectionByInverseViewMatrix( reflectVec, viewMatrix );\n			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );\n			return envMapColor.rgb * envMapIntensity;\n		#else\n			return vec3( 0.0 );\n		#endif\n	}\n	#ifdef USE_ANISOTROPY\n		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {\n			#ifdef ENVMAP_TYPE_CUBE_UV\n				vec3 bentNormal = cross( bitangent, viewDir );\n				bentNormal = normalize( cross( bentNormal, bitangent ) );\n				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );\n				return getIBLRadiance( viewDir, bentNormal, roughness );\n			#else\n				return vec3( 0.0 );\n			#endif\n		}\n	#endif\n#endif",
	envmap_vertex: "#ifdef USE_ENVMAP\n	#ifdef ENV_WORLDPOS\n		vWorldPosition = worldPosition.xyz;\n	#else\n		vec3 cameraToVertex;\n		if ( isOrthographic ) {\n			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );\n		} else {\n			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );\n		}\n		vec3 worldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );\n		#ifdef ENVMAP_MODE_REFLECTION\n			vReflect = reflect( cameraToVertex, worldNormal );\n		#else\n			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );\n		#endif\n	#endif\n#endif",
	fog_vertex: "#ifdef USE_FOG\n	vFogDepth = - mvPosition.z;\n#endif",
	fog_pars_vertex: "#ifdef USE_FOG\n	varying float vFogDepth;\n#endif",
	fog_fragment: "#ifdef USE_FOG\n	#ifdef FOG_EXP2\n		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );\n	#else\n		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );\n	#endif\n	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );\n#endif",
	fog_pars_fragment: "#ifdef USE_FOG\n	uniform vec3 fogColor;\n	varying float vFogDepth;\n	#ifdef FOG_EXP2\n		uniform float fogDensity;\n	#else\n		uniform float fogNear;\n		uniform float fogFar;\n	#endif\n#endif",
	gradientmap_pars_fragment: "#ifdef USE_GRADIENTMAP\n	uniform sampler2D gradientMap;\n#endif\nvec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {\n	float dotNL = dot( normal, lightDirection );\n	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );\n	#ifdef USE_GRADIENTMAP\n		return vec3( texture2D( gradientMap, coord ).r );\n	#else\n		vec2 fw = fwidth( coord ) * 0.5;\n		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );\n	#endif\n}",
	lightmap_pars_fragment: "#ifdef USE_LIGHTMAP\n	uniform sampler2D lightMap;\n	uniform float lightMapIntensity;\n#endif",
	lights_lambert_fragment: "LambertMaterial material;\nmaterial.diffuseColor = diffuseColor.rgb;\nmaterial.specularStrength = specularStrength;",
	lights_lambert_pars_fragment: "varying vec3 vViewPosition;\nstruct LambertMaterial {\n	vec3 diffuseColor;\n	float specularStrength;\n};\nvoid RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {\n	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );\n	vec3 irradiance = dotNL * directLight.color;\n	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n}\nvoid RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {\n	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n}\n#define RE_Direct				RE_Direct_Lambert\n#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert",
	lights_pars_begin: "uniform bool receiveShadow;\nuniform vec3 ambientLightColor;\n#if defined( USE_LIGHT_PROBES )\n	uniform vec3 lightProbe[ 9 ];\n#endif\nvec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {\n	float x = normal.x, y = normal.y, z = normal.z;\n	vec3 result = shCoefficients[ 0 ] * 0.886227;\n	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;\n	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;\n	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;\n	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;\n	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;\n	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );\n	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;\n	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );\n	return result;\n}\nvec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {\n	vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );\n	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );\n	return irradiance;\n}\nvec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {\n	vec3 irradiance = ambientLightColor;\n	return irradiance;\n}\nfloat getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {\n	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );\n	if ( cutoffDistance > 0.0 ) {\n		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );\n	}\n	return distanceFalloff;\n}\nfloat getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {\n	return smoothstep( coneCosine, penumbraCosine, angleCosine );\n}\n#if NUM_DIR_LIGHTS > 0\n	struct DirectionalLight {\n		vec3 direction;\n		vec3 color;\n	};\n	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];\n	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {\n		light.color = directionalLight.color;\n		light.direction = directionalLight.direction;\n		light.visible = true;\n	}\n#endif\n#if NUM_POINT_LIGHTS > 0\n	struct PointLight {\n		vec3 position;\n		vec3 color;\n		float distance;\n		float decay;\n	};\n	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];\n	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {\n		vec3 lVector = pointLight.position - geometryPosition;\n		light.direction = normalize( lVector );\n		float lightDistance = length( lVector );\n		light.color = pointLight.color;\n		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );\n		light.visible = ( light.color != vec3( 0.0 ) );\n	}\n#endif\n#if NUM_SPOT_LIGHTS > 0\n	struct SpotLight {\n		vec3 position;\n		vec3 direction;\n		vec3 color;\n		float distance;\n		float decay;\n		float coneCos;\n		float penumbraCos;\n	};\n	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];\n	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {\n		vec3 lVector = spotLight.position - geometryPosition;\n		light.direction = normalize( lVector );\n		float angleCos = dot( light.direction, spotLight.direction );\n		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );\n		if ( spotAttenuation > 0.0 ) {\n			float lightDistance = length( lVector );\n			light.color = spotLight.color * spotAttenuation;\n			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );\n			light.visible = ( light.color != vec3( 0.0 ) );\n		} else {\n			light.color = vec3( 0.0 );\n			light.visible = false;\n		}\n	}\n#endif\n#if NUM_RECT_AREA_LIGHTS > 0\n	struct RectAreaLight {\n		vec3 color;\n		vec3 position;\n		vec3 halfWidth;\n		vec3 halfHeight;\n	};\n	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;\n	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];\n#endif\n#if NUM_HEMI_LIGHTS > 0\n	struct HemisphereLight {\n		vec3 direction;\n		vec3 skyColor;\n		vec3 groundColor;\n	};\n	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];\n	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {\n		float dotNL = dot( normal, hemiLight.direction );\n		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;\n		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );\n		return irradiance;\n	}\n#endif\n#include <lightprobes_pars_fragment>",
	lights_toon_fragment: "ToonMaterial material;\nmaterial.diffuseColor = diffuseColor.rgb;",
	lights_toon_pars_fragment: "varying vec3 vViewPosition;\nstruct ToonMaterial {\n	vec3 diffuseColor;\n};\nvoid RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {\n	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;\n	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n}\nvoid RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {\n	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n}\n#define RE_Direct				RE_Direct_Toon\n#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon",
	lights_phong_fragment: "BlinnPhongMaterial material;\nmaterial.diffuseColor = diffuseColor.rgb;\nmaterial.specularColor = specular;\nmaterial.specularShininess = shininess;\nmaterial.specularStrength = specularStrength;",
	lights_phong_pars_fragment: "varying vec3 vViewPosition;\nstruct BlinnPhongMaterial {\n	vec3 diffuseColor;\n	vec3 specularColor;\n	float specularShininess;\n	float specularStrength;\n};\nvoid RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {\n	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );\n	vec3 irradiance = dotNL * directLight.color;\n	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;\n}\nvoid RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {\n	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n}\n#define RE_Direct				RE_Direct_BlinnPhong\n#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong",
	lights_physical_fragment: "PhysicalMaterial material;\nmaterial.diffuseColor = diffuseColor.rgb;\nmaterial.diffuseContribution = diffuseColor.rgb * ( 1.0 - metalnessFactor );\nmaterial.metalness = metalnessFactor;\nvec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );\nfloat geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );\nmaterial.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;\nmaterial.roughness = min( material.roughness, 1.0 );\n#ifdef IOR\n	material.ior = ior;\n	#ifdef USE_SPECULAR\n		float specularIntensityFactor = specularIntensity;\n		vec3 specularColorFactor = specularColor;\n		#ifdef USE_SPECULAR_COLORMAP\n			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;\n		#endif\n		#ifdef USE_SPECULAR_INTENSITYMAP\n			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;\n		#endif\n		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );\n	#else\n		float specularIntensityFactor = 1.0;\n		vec3 specularColorFactor = vec3( 1.0 );\n		material.specularF90 = 1.0;\n	#endif\n	material.specularColor = min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor;\n	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );\n#else\n	material.specularColor = vec3( 0.04 );\n	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );\n	material.specularF90 = 1.0;\n#endif\n#ifdef USE_CLEARCOAT\n	material.clearcoat = clearcoat;\n	material.clearcoatRoughness = clearcoatRoughness;\n	material.clearcoatF0 = vec3( 0.04 );\n	material.clearcoatF90 = 1.0;\n	#ifdef USE_CLEARCOATMAP\n		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;\n	#endif\n	#ifdef USE_CLEARCOAT_ROUGHNESSMAP\n		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;\n	#endif\n	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );\n	material.clearcoatRoughness += geometryRoughness;\n	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );\n#endif\n#ifdef USE_DISPERSION\n	material.dispersion = dispersion;\n#endif\n#ifdef USE_IRIDESCENCE\n	material.iridescence = iridescence;\n	material.iridescenceIOR = iridescenceIOR;\n	#ifdef USE_IRIDESCENCEMAP\n		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;\n	#endif\n	#ifdef USE_IRIDESCENCE_THICKNESSMAP\n		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;\n	#else\n		material.iridescenceThickness = iridescenceThicknessMaximum;\n	#endif\n#endif\n#ifdef USE_SHEEN\n	material.sheenColor = sheenColor;\n	#ifdef USE_SHEEN_COLORMAP\n		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;\n	#endif\n	material.sheenRoughness = clamp( sheenRoughness, 0.0001, 1.0 );\n	#ifdef USE_SHEEN_ROUGHNESSMAP\n		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;\n	#endif\n#endif\n#ifdef USE_ANISOTROPY\n	#ifdef USE_ANISOTROPYMAP\n		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );\n		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;\n		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;\n	#else\n		vec2 anisotropyV = anisotropyVector;\n	#endif\n	material.anisotropy = length( anisotropyV );\n	if( material.anisotropy == 0.0 ) {\n		anisotropyV = vec2( 1.0, 0.0 );\n	} else {\n		anisotropyV /= material.anisotropy;\n		material.anisotropy = saturate( material.anisotropy );\n	}\n	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );\n	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;\n	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;\n#endif",
	lights_physical_pars_fragment: "uniform sampler2D dfgLUT;\nstruct PhysicalMaterial {\n	vec3 diffuseColor;\n	vec3 diffuseContribution;\n	vec3 specularColor;\n	vec3 specularColorBlended;\n	float roughness;\n	float metalness;\n	float specularF90;\n	float dispersion;\n	#ifdef USE_CLEARCOAT\n		float clearcoat;\n		float clearcoatRoughness;\n		vec3 clearcoatF0;\n		float clearcoatF90;\n	#endif\n	#ifdef USE_IRIDESCENCE\n		float iridescence;\n		float iridescenceIOR;\n		float iridescenceThickness;\n		vec3 iridescenceFresnel;\n		vec3 iridescenceF0;\n		vec3 iridescenceFresnelDielectric;\n		vec3 iridescenceFresnelMetallic;\n	#endif\n	#ifdef USE_SHEEN\n		vec3 sheenColor;\n		float sheenRoughness;\n	#endif\n	#ifdef IOR\n		float ior;\n	#endif\n	#ifdef USE_TRANSMISSION\n		float transmission;\n		float transmissionAlpha;\n		float thickness;\n		float attenuationDistance;\n		vec3 attenuationColor;\n	#endif\n	#ifdef USE_ANISOTROPY\n		float anisotropy;\n		float alphaT;\n		vec3 anisotropyT;\n		vec3 anisotropyB;\n	#endif\n};\nvec3 clearcoatSpecularDirect = vec3( 0.0 );\nvec3 clearcoatSpecularIndirect = vec3( 0.0 );\nvec3 sheenSpecularDirect = vec3( 0.0 );\nvec3 sheenSpecularIndirect = vec3(0.0 );\nvec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {\n    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );\n    float x2 = x * x;\n    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );\n    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );\n}\nfloat V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {\n	float a2 = pow2( alpha );\n	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );\n	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );\n	return 0.5 / max( gv + gl, EPSILON );\n}\nfloat D_GGX( const in float alpha, const in float dotNH ) {\n	float a2 = pow2( alpha );\n	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;\n	return RECIPROCAL_PI * a2 / pow2( denom );\n}\n#ifdef USE_ANISOTROPY\n	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {\n		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );\n		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );\n		return 0.5 / max( gv + gl, EPSILON );\n	}\n	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {\n		float a2 = alphaT * alphaB;\n		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );\n		highp float v2 = dot( v, v );\n		float w2 = a2 / v2;\n		return RECIPROCAL_PI * a2 * pow2 ( w2 );\n	}\n#endif\n#ifdef USE_CLEARCOAT\n	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {\n		vec3 f0 = material.clearcoatF0;\n		float f90 = material.clearcoatF90;\n		float roughness = material.clearcoatRoughness;\n		float alpha = pow2( roughness );\n		vec3 halfDir = normalize( lightDir + viewDir );\n		float dotNL = saturate( dot( normal, lightDir ) );\n		float dotNV = saturate( dot( normal, viewDir ) );\n		float dotNH = saturate( dot( normal, halfDir ) );\n		float dotVH = saturate( dot( viewDir, halfDir ) );\n		vec3 F = F_Schlick( f0, f90, dotVH );\n		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );\n		float D = D_GGX( alpha, dotNH );\n		return F * ( V * D );\n	}\n#endif\nvec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {\n	vec3 f0 = material.specularColorBlended;\n	float f90 = material.specularF90;\n	float roughness = material.roughness;\n	float alpha = pow2( roughness );\n	vec3 halfDir = normalize( lightDir + viewDir );\n	float dotNL = saturate( dot( normal, lightDir ) );\n	float dotNV = saturate( dot( normal, viewDir ) );\n	float dotNH = saturate( dot( normal, halfDir ) );\n	float dotVH = saturate( dot( viewDir, halfDir ) );\n	vec3 F = F_Schlick( f0, f90, dotVH );\n	#ifdef USE_IRIDESCENCE\n		F = mix( F, material.iridescenceFresnel, material.iridescence );\n	#endif\n	#ifdef USE_ANISOTROPY\n		float dotTL = dot( material.anisotropyT, lightDir );\n		float dotTV = dot( material.anisotropyT, viewDir );\n		float dotTH = dot( material.anisotropyT, halfDir );\n		float dotBL = dot( material.anisotropyB, lightDir );\n		float dotBV = dot( material.anisotropyB, viewDir );\n		float dotBH = dot( material.anisotropyB, halfDir );\n		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );\n		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );\n	#else\n		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );\n		float D = D_GGX( alpha, dotNH );\n	#endif\n	return F * ( V * D );\n}\nvec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {\n	const float LUT_SIZE = 64.0;\n	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;\n	const float LUT_BIAS = 0.5 / LUT_SIZE;\n	float dotNV = saturate( dot( N, V ) );\n	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );\n	uv = uv * LUT_SCALE + LUT_BIAS;\n	return uv;\n}\nfloat LTC_ClippedSphereFormFactor( const in vec3 f ) {\n	float l = length( f );\n	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );\n}\nvec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {\n	float x = dot( v1, v2 );\n	float y = abs( x );\n	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;\n	float b = 3.4175940 + ( 4.1616724 + y ) * y;\n	float v = a / b;\n	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;\n	return cross( v1, v2 ) * theta_sintheta;\n}\nvec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {\n	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];\n	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];\n	vec3 lightNormal = cross( v1, v2 );\n	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );\n	vec3 T1, T2;\n	T1 = normalize( V - N * dot( V, N ) );\n	T2 = - cross( N, T1 );\n	mat3 mat = mInv * transpose( mat3( T1, T2, N ) );\n	vec3 coords[ 4 ];\n	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );\n	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );\n	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );\n	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );\n	coords[ 0 ] = normalize( coords[ 0 ] );\n	coords[ 1 ] = normalize( coords[ 1 ] );\n	coords[ 2 ] = normalize( coords[ 2 ] );\n	coords[ 3 ] = normalize( coords[ 3 ] );\n	vec3 vectorFormFactor = vec3( 0.0 );\n	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );\n	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );\n	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );\n	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );\n	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );\n	return vec3( result );\n}\n#if defined( USE_SHEEN )\nfloat D_Charlie( float roughness, float dotNH ) {\n	float alpha = pow2( roughness );\n	float invAlpha = 1.0 / alpha;\n	float cos2h = dotNH * dotNH;\n	float sin2h = max( 1.0 - cos2h, 0.0078125 );\n	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );\n}\nfloat V_Neubelt( float dotNV, float dotNL ) {\n	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );\n}\nvec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {\n	vec3 halfDir = normalize( lightDir + viewDir );\n	float dotNL = saturate( dot( normal, lightDir ) );\n	float dotNV = saturate( dot( normal, viewDir ) );\n	float dotNH = saturate( dot( normal, halfDir ) );\n	float D = D_Charlie( sheenRoughness, dotNH );\n	float V = V_Neubelt( dotNV, dotNL );\n	return sheenColor * ( D * V );\n}\n#endif\nfloat IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {\n	float dotNV = saturate( dot( normal, viewDir ) );\n	float r2 = roughness * roughness;\n	float rInv = 1.0 / ( roughness + 0.1 );\n	float a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * rInv;\n	float b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * rInv;\n	float DG = exp( a * dotNV + b );\n	return saturate( DG );\n}\nvec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {\n	float dotNV = saturate( dot( normal, viewDir ) );\n	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;\n	return specularColor * fab.x + specularF90 * fab.y;\n}\n#ifdef USE_IRIDESCENCE\nvoid computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {\n#else\nvoid computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {\n#endif\n	float dotNV = saturate( dot( normal, viewDir ) );\n	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;\n	#ifdef USE_IRIDESCENCE\n		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );\n	#else\n		vec3 Fr = specularColor;\n	#endif\n	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;\n	float Ess = fab.x + fab.y;\n	float Ems = 1.0 - Ess;\n	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );\n	singleScatter += FssEss;\n	multiScatter += Fms * Ems;\n}\nvec3 BRDF_GGX_Multiscatter( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {\n	vec3 singleScatter = BRDF_GGX( lightDir, viewDir, normal, material );\n	float dotNL = saturate( dot( normal, lightDir ) );\n	float dotNV = saturate( dot( normal, viewDir ) );\n	vec2 dfgV = texture2D( dfgLUT, vec2( material.roughness, dotNV ) ).rg;\n	vec2 dfgL = texture2D( dfgLUT, vec2( material.roughness, dotNL ) ).rg;\n	vec3 FssEss_V = material.specularColorBlended * dfgV.x + material.specularF90 * dfgV.y;\n	vec3 FssEss_L = material.specularColorBlended * dfgL.x + material.specularF90 * dfgL.y;\n	float Ess_V = dfgV.x + dfgV.y;\n	float Ess_L = dfgL.x + dfgL.y;\n	float Ems_V = 1.0 - Ess_V;\n	float Ems_L = 1.0 - Ess_L;\n	vec3 Favg = material.specularColorBlended + ( 1.0 - material.specularColorBlended ) * 0.047619;\n	vec3 Fms = FssEss_V * FssEss_L * Favg / ( 1.0 - Ems_V * Ems_L * Favg + EPSILON );\n	float compensationFactor = Ems_V * Ems_L;\n	vec3 multiScatter = Fms * compensationFactor;\n	return singleScatter + multiScatter;\n}\n#if NUM_RECT_AREA_LIGHTS > 0\n	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {\n		vec3 normal = geometryNormal;\n		vec3 viewDir = geometryViewDir;\n		vec3 position = geometryPosition;\n		vec3 lightPos = rectAreaLight.position;\n		vec3 halfWidth = rectAreaLight.halfWidth;\n		vec3 halfHeight = rectAreaLight.halfHeight;\n		vec3 lightColor = rectAreaLight.color;\n		float roughness = material.roughness;\n		vec3 rectCoords[ 4 ];\n		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;\n		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;\n		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;\n		vec2 uv = LTC_Uv( normal, viewDir, roughness );\n		vec4 t1 = texture2D( ltc_1, uv );\n		vec4 t2 = texture2D( ltc_2, uv );\n		mat3 mInv = mat3(\n			vec3( t1.x, 0, t1.y ),\n			vec3(    0, 1,    0 ),\n			vec3( t1.z, 0, t1.w )\n		);\n		vec3 fresnel = ( material.specularColorBlended * t2.x + ( material.specularF90 - material.specularColorBlended ) * t2.y );\n		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );\n		reflectedLight.directDiffuse += lightColor * material.diffuseContribution * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );\n		#ifdef USE_CLEARCOAT\n			vec3 Ncc = geometryClearcoatNormal;\n			vec2 uvClearcoat = LTC_Uv( Ncc, viewDir, material.clearcoatRoughness );\n			vec4 t1Clearcoat = texture2D( ltc_1, uvClearcoat );\n			vec4 t2Clearcoat = texture2D( ltc_2, uvClearcoat );\n			mat3 mInvClearcoat = mat3(\n				vec3( t1Clearcoat.x, 0, t1Clearcoat.y ),\n				vec3(             0, 1,             0 ),\n				vec3( t1Clearcoat.z, 0, t1Clearcoat.w )\n			);\n			vec3 fresnelClearcoat = material.clearcoatF0 * t2Clearcoat.x + ( material.clearcoatF90 - material.clearcoatF0 ) * t2Clearcoat.y;\n			clearcoatSpecularDirect += lightColor * fresnelClearcoat * LTC_Evaluate( Ncc, viewDir, position, mInvClearcoat, rectCoords );\n		#endif\n	}\n#endif\nvoid RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {\n	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );\n	vec3 irradiance = dotNL * directLight.color;\n	#ifdef USE_CLEARCOAT\n		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );\n		vec3 ccIrradiance = dotNLcc * directLight.color;\n		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );\n	#endif\n	#ifdef USE_SHEEN\n \n 		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );\n \n 		float sheenAlbedoV = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );\n 		float sheenAlbedoL = IBLSheenBRDF( geometryNormal, directLight.direction, material.sheenRoughness );\n \n 		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * max( sheenAlbedoV, sheenAlbedoL );\n \n 		irradiance *= sheenEnergyComp;\n \n 	#endif\n	reflectedLight.directSpecular += irradiance * BRDF_GGX_Multiscatter( directLight.direction, geometryViewDir, geometryNormal, material );\n	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseContribution );\n}\nvoid RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {\n	vec3 diffuse = irradiance * BRDF_Lambert( material.diffuseContribution );\n	#ifdef USE_SHEEN\n		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );\n		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;\n		diffuse *= sheenEnergyComp;\n	#endif\n	reflectedLight.indirectDiffuse += diffuse;\n}\nvoid RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {\n	#ifdef USE_CLEARCOAT\n		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );\n	#endif\n	#ifdef USE_SHEEN\n		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness ) * RECIPROCAL_PI;\n 	#endif\n	vec3 singleScatteringDielectric = vec3( 0.0 );\n	vec3 multiScatteringDielectric = vec3( 0.0 );\n	vec3 singleScatteringMetallic = vec3( 0.0 );\n	vec3 multiScatteringMetallic = vec3( 0.0 );\n	#ifdef USE_IRIDESCENCE\n		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnelDielectric, material.roughness, singleScatteringDielectric, multiScatteringDielectric );\n		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.iridescence, material.iridescenceFresnelMetallic, material.roughness, singleScatteringMetallic, multiScatteringMetallic );\n	#else\n		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScatteringDielectric, multiScatteringDielectric );\n		computeMultiscattering( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.roughness, singleScatteringMetallic, multiScatteringMetallic );\n	#endif\n	vec3 singleScattering = mix( singleScatteringDielectric, singleScatteringMetallic, material.metalness );\n	vec3 multiScattering = mix( multiScatteringDielectric, multiScatteringMetallic, material.metalness );\n	vec3 totalScatteringDielectric = singleScatteringDielectric + multiScatteringDielectric;\n	vec3 diffuse = material.diffuseContribution * ( 1.0 - totalScatteringDielectric );\n	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;\n	vec3 indirectSpecular = radiance * singleScattering;\n	indirectSpecular += multiScattering * cosineWeightedIrradiance;\n	vec3 indirectDiffuse = diffuse * cosineWeightedIrradiance;\n	#ifdef USE_SHEEN\n		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );\n		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;\n		indirectSpecular *= sheenEnergyComp;\n		indirectDiffuse *= sheenEnergyComp;\n	#endif\n	reflectedLight.indirectSpecular += indirectSpecular;\n	reflectedLight.indirectDiffuse += indirectDiffuse;\n}\n#define RE_Direct				RE_Direct_Physical\n#define RE_Direct_RectArea		RE_Direct_RectArea_Physical\n#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical\n#define RE_IndirectSpecular		RE_IndirectSpecular_Physical\nfloat computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {\n	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );\n}",
	lights_fragment_begin: "\nvec3 geometryPosition = - vViewPosition;\nvec3 geometryNormal = normal;\nvec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );\nvec3 geometryClearcoatNormal = vec3( 0.0 );\n#ifdef USE_CLEARCOAT\n	geometryClearcoatNormal = clearcoatNormal;\n#endif\n#ifdef USE_IRIDESCENCE\n	float dotNVi = saturate( dot( normal, geometryViewDir ) );\n	if ( material.iridescenceThickness == 0.0 ) {\n		material.iridescence = 0.0;\n	} else {\n		material.iridescence = saturate( material.iridescence );\n	}\n	if ( material.iridescence > 0.0 ) {\n		material.iridescenceFresnelDielectric = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );\n		material.iridescenceFresnelMetallic = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.diffuseColor );\n		material.iridescenceFresnel = mix( material.iridescenceFresnelDielectric, material.iridescenceFresnelMetallic, material.metalness );\n		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );\n	}\n#endif\nIncidentLight directLight;\n#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )\n	PointLight pointLight;\n	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0\n	PointLightShadow pointLightShadow;\n	#endif\n	#pragma unroll_loop_start\n	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {\n		pointLight = pointLights[ i ];\n		getPointLightInfo( pointLight, geometryPosition, directLight );\n		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS ) && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )\n		pointLightShadow = pointLightShadows[ i ];\n		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;\n		#endif\n		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );\n	}\n	#pragma unroll_loop_end\n#endif\n#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )\n	SpotLight spotLight;\n	vec4 spotColor;\n	vec3 spotLightCoord;\n	bool inSpotLightMap;\n	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0\n	SpotLightShadow spotLightShadow;\n	#endif\n	#pragma unroll_loop_start\n	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {\n		spotLight = spotLights[ i ];\n		getSpotLightInfo( spotLight, geometryPosition, directLight );\n		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )\n		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX\n		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )\n		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS\n		#else\n		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )\n		#endif\n		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )\n			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;\n			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );\n			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );\n			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;\n		#endif\n		#undef SPOT_LIGHT_MAP_INDEX\n		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )\n		spotLightShadow = spotLightShadows[ i ];\n		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;\n		#endif\n		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );\n	}\n	#pragma unroll_loop_end\n#endif\n#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )\n	DirectionalLight directionalLight;\n	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0\n	DirectionalLightShadow directionalLightShadow;\n	#endif\n	#pragma unroll_loop_start\n	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {\n		directionalLight = directionalLights[ i ];\n		getDirectionalLightInfo( directionalLight, directLight );\n		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )\n		directionalLightShadow = directionalLightShadows[ i ];\n		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;\n		#endif\n		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );\n	}\n	#pragma unroll_loop_end\n#endif\n#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )\n	RectAreaLight rectAreaLight;\n	#pragma unroll_loop_start\n	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {\n		rectAreaLight = rectAreaLights[ i ];\n		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );\n	}\n	#pragma unroll_loop_end\n#endif\n#if defined( RE_IndirectDiffuse )\n	vec3 iblIrradiance = vec3( 0.0 );\n	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );\n	#if defined( USE_LIGHT_PROBES )\n		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );\n	#endif\n	#if ( NUM_HEMI_LIGHTS > 0 )\n		#pragma unroll_loop_start\n		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {\n			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );\n		}\n		#pragma unroll_loop_end\n	#endif\n	#ifdef USE_LIGHT_PROBES_GRID\n		vec3 probeWorldPos = ( ( vec4( geometryPosition, 1.0 ) - viewMatrix[ 3 ] ) * viewMatrix ).xyz;\n		vec3 probeWorldNormal = transformNormalByInverseViewMatrix( geometryNormal, viewMatrix );\n		irradiance += getLightProbeGridIrradiance( probeWorldPos, probeWorldNormal );\n	#endif\n#endif\n#if defined( RE_IndirectSpecular )\n	vec3 radiance = vec3( 0.0 );\n	vec3 clearcoatRadiance = vec3( 0.0 );\n#endif",
	lights_fragment_maps: "#if defined( RE_IndirectDiffuse )\n	#ifdef USE_LIGHTMAP\n		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );\n		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;\n		irradiance += lightMapIrradiance;\n	#endif\n	#if defined( USE_ENVMAP ) && defined( ENVMAP_TYPE_CUBE_UV )\n		#if defined( STANDARD ) || defined( LAMBERT ) || defined( PHONG )\n			iblIrradiance += getIBLIrradiance( geometryNormal );\n		#endif\n	#endif\n#endif\n#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )\n	#ifdef USE_ANISOTROPY\n		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );\n	#else\n		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );\n	#endif\n	#ifdef USE_CLEARCOAT\n		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );\n	#endif\n#endif",
	lights_fragment_end: "#if defined( RE_IndirectDiffuse )\n	#if defined( LAMBERT ) || defined( PHONG )\n		irradiance += iblIrradiance;\n	#endif\n	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );\n#endif\n#if defined( RE_IndirectSpecular )\n	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );\n#endif",
	lightprobes_pars_fragment: "#ifdef USE_LIGHT_PROBES_GRID\nuniform highp sampler3D probesSH;\nuniform vec3 probesMin;\nuniform vec3 probesMax;\nuniform vec3 probesResolution;\nvec3 getLightProbeGridIrradiance( vec3 worldPos, vec3 worldNormal ) {\n	vec3 res = probesResolution;\n	vec3 gridRange = probesMax - probesMin;\n	vec3 resMinusOne = res - 1.0;\n	vec3 probeSpacing = gridRange / resMinusOne;\n	vec3 samplePos = worldPos + worldNormal * probeSpacing * 0.5;\n	vec3 uvw = clamp( ( samplePos - probesMin ) / gridRange, 0.0, 1.0 );\n	uvw = uvw * resMinusOne / res + 0.5 / res;\n	float nz          = res.z;\n	float paddedSlices = nz + 2.0;\n	float atlasDepth  = 7.0 * paddedSlices;\n	float uvZBase     = uvw.z * nz + 1.0;\n	vec4 s0 = texture( probesSH, vec3( uvw.xy, ( uvZBase                       ) / atlasDepth ) );\n	vec4 s1 = texture( probesSH, vec3( uvw.xy, ( uvZBase +       paddedSlices   ) / atlasDepth ) );\n	vec4 s2 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 2.0 * paddedSlices   ) / atlasDepth ) );\n	vec4 s3 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 3.0 * paddedSlices   ) / atlasDepth ) );\n	vec4 s4 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 4.0 * paddedSlices   ) / atlasDepth ) );\n	vec4 s5 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 5.0 * paddedSlices   ) / atlasDepth ) );\n	vec4 s6 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 6.0 * paddedSlices   ) / atlasDepth ) );\n	vec3 c0 = s0.xyz;\n	vec3 c1 = vec3( s0.w, s1.xy );\n	vec3 c2 = vec3( s1.zw, s2.x );\n	vec3 c3 = s2.yzw;\n	vec3 c4 = s3.xyz;\n	vec3 c5 = vec3( s3.w, s4.xy );\n	vec3 c6 = vec3( s4.zw, s5.x );\n	vec3 c7 = s5.yzw;\n	vec3 c8 = s6.xyz;\n	float x = worldNormal.x, y = worldNormal.y, z = worldNormal.z;\n	vec3 result = c0 * 0.886227;\n	result += c1 * 2.0 * 0.511664 * y;\n	result += c2 * 2.0 * 0.511664 * z;\n	result += c3 * 2.0 * 0.511664 * x;\n	result += c4 * 2.0 * 0.429043 * x * y;\n	result += c5 * 2.0 * 0.429043 * y * z;\n	result += c6 * ( 0.743125 * z * z - 0.247708 );\n	result += c7 * 2.0 * 0.429043 * x * z;\n	result += c8 * 0.429043 * ( x * x - y * y );\n	return max( result, vec3( 0.0 ) );\n}\n#endif",
	logdepthbuf_fragment: "#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )\n	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;\n#endif",
	logdepthbuf_pars_fragment: "#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )\n	uniform float logDepthBufFC;\n	varying float vFragDepth;\n	varying float vIsPerspective;\n#endif",
	logdepthbuf_pars_vertex: "#ifdef USE_LOGARITHMIC_DEPTH_BUFFER\n	varying float vFragDepth;\n	varying float vIsPerspective;\n#endif",
	logdepthbuf_vertex: "#ifdef USE_LOGARITHMIC_DEPTH_BUFFER\n	vFragDepth = 1.0 + gl_Position.w;\n	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );\n#endif",
	map_fragment: "#ifdef USE_MAP\n	vec4 sampledDiffuseColor = texture2D( map, vMapUv );\n	#ifdef DECODE_VIDEO_TEXTURE\n		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );\n	#endif\n	diffuseColor *= sampledDiffuseColor;\n#endif",
	map_pars_fragment: "#ifdef USE_MAP\n	uniform sampler2D map;\n#endif",
	map_particle_fragment: "#if defined( USE_MAP ) || defined( USE_ALPHAMAP )\n	#if defined( USE_POINTS_UV )\n		vec2 uv = vUv;\n	#else\n		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;\n	#endif\n#endif\n#ifdef USE_MAP\n	diffuseColor *= texture2D( map, uv );\n#endif\n#ifdef USE_ALPHAMAP\n	diffuseColor.a *= texture2D( alphaMap, uv ).g;\n#endif",
	map_particle_pars_fragment: "#if defined( USE_POINTS_UV )\n	varying vec2 vUv;\n#else\n	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )\n		uniform mat3 uvTransform;\n	#endif\n#endif\n#ifdef USE_MAP\n	uniform sampler2D map;\n#endif\n#ifdef USE_ALPHAMAP\n	uniform sampler2D alphaMap;\n#endif",
	metalnessmap_fragment: "float metalnessFactor = metalness;\n#ifdef USE_METALNESSMAP\n	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );\n	metalnessFactor *= texelMetalness.b;\n#endif",
	metalnessmap_pars_fragment: "#ifdef USE_METALNESSMAP\n	uniform sampler2D metalnessMap;\n#endif",
	morphinstance_vertex: "#ifdef USE_INSTANCING_MORPH\n	float morphTargetInfluences[ MORPHTARGETS_COUNT ];\n	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;\n	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {\n		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;\n	}\n#endif",
	morphcolor_vertex: "#if defined( USE_MORPHCOLORS )\n	vColor *= morphTargetBaseInfluence;\n	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {\n		#if defined( USE_COLOR_ALPHA )\n			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];\n		#elif defined( USE_COLOR )\n			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];\n		#endif\n	}\n#endif",
	morphnormal_vertex: "#ifdef USE_MORPHNORMALS\n	objectNormal *= morphTargetBaseInfluence;\n	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {\n		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];\n	}\n#endif",
	morphtarget_pars_vertex: "#ifdef USE_MORPHTARGETS\n	#ifndef USE_INSTANCING_MORPH\n		uniform float morphTargetBaseInfluence;\n		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];\n	#endif\n	uniform sampler2DArray morphTargetsTexture;\n	uniform ivec2 morphTargetsTextureSize;\n	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {\n		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;\n		int y = texelIndex / morphTargetsTextureSize.x;\n		int x = texelIndex - y * morphTargetsTextureSize.x;\n		ivec3 morphUV = ivec3( x, y, morphTargetIndex );\n		return texelFetch( morphTargetsTexture, morphUV, 0 );\n	}\n#endif",
	morphtarget_vertex: "#ifdef USE_MORPHTARGETS\n	transformed *= morphTargetBaseInfluence;\n	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {\n		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];\n	}\n#endif",
	normal_fragment_begin: "float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;\n#ifdef FLAT_SHADED\n	vec3 fdx = dFdx( vViewPosition );\n	vec3 fdy = dFdy( vViewPosition );\n	vec3 normal = normalize( cross( fdx, fdy ) );\n#else\n	vec3 normal = normalize( vNormal );\n	#ifdef DOUBLE_SIDED\n		normal *= faceDirection;\n	#endif\n#endif\n#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )\n	#ifdef USE_TANGENT\n		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );\n	#else\n		mat3 tbn = getTangentFrame( - vViewPosition, normal,\n		#if defined( USE_NORMALMAP )\n			vNormalMapUv\n		#elif defined( USE_CLEARCOAT_NORMALMAP )\n			vClearcoatNormalMapUv\n		#else\n			vUv\n		#endif\n		);\n	#endif\n	#ifdef DOUBLE_SIDED\n		tbn[0] *= faceDirection;\n		tbn[1] *= faceDirection;\n	#endif\n#endif\n#ifdef USE_CLEARCOAT_NORMALMAP\n	#ifdef USE_TANGENT\n		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );\n	#else\n		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );\n	#endif\n	#ifdef DOUBLE_SIDED\n		tbn2[0] *= faceDirection;\n		tbn2[1] *= faceDirection;\n	#endif\n#endif\nvec3 nonPerturbedNormal = normal;",
	normal_fragment_maps: "#ifdef USE_NORMALMAP_OBJECTSPACE\n	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;\n	#ifdef FLIP_SIDED\n		normal = - normal;\n	#endif\n	#ifdef DOUBLE_SIDED\n		normal = normal * faceDirection;\n	#endif\n	normal = normalize( normalMatrix * normal );\n#elif defined( USE_NORMALMAP_TANGENTSPACE )\n	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;\n	#if defined( USE_PACKED_NORMALMAP )\n		mapN = vec3( mapN.xy, sqrt( saturate( 1.0 - dot( mapN.xy, mapN.xy ) ) ) );\n	#endif\n	mapN.xy *= normalScale;\n	normal = normalize( tbn * mapN );\n#elif defined( USE_BUMPMAP )\n	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );\n#endif",
	normal_pars_fragment: "#ifndef FLAT_SHADED\n	varying vec3 vNormal;\n	#ifdef USE_TANGENT\n		varying vec3 vTangent;\n		varying vec3 vBitangent;\n	#endif\n#endif",
	normal_pars_vertex: "#ifndef FLAT_SHADED\n	varying vec3 vNormal;\n	#ifdef USE_TANGENT\n		varying vec3 vTangent;\n		varying vec3 vBitangent;\n	#endif\n#endif",
	normal_vertex: "#ifndef FLAT_SHADED\n	vNormal = normalize( transformedNormal );\n	#ifdef USE_TANGENT\n		vTangent = normalize( transformedTangent );\n		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );\n		#ifdef FLIP_SIDED\n			vBitangent = - vBitangent;\n		#endif\n	#endif\n#endif",
	normalmap_pars_fragment: "#ifdef USE_NORMALMAP\n	uniform sampler2D normalMap;\n	uniform vec2 normalScale;\n#endif\n#ifdef USE_NORMALMAP_OBJECTSPACE\n	uniform mat3 normalMatrix;\n#endif\n#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )\n	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {\n		vec3 q0 = dFdx( eye_pos.xyz );\n		vec3 q1 = dFdy( eye_pos.xyz );\n		vec2 st0 = dFdx( uv.st );\n		vec2 st1 = dFdy( uv.st );\n		vec3 N = surf_norm;\n		vec3 q1perp = cross( q1, N );\n		vec3 q0perp = cross( N, q0 );\n		vec3 T = q1perp * st0.x + q0perp * st1.x;\n		vec3 B = q1perp * st0.y + q0perp * st1.y;\n		float det = max( dot( T, T ), dot( B, B ) );\n		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );\n		return mat3( T * scale, B * scale, N );\n	}\n#endif",
	clearcoat_normal_fragment_begin: "#ifdef USE_CLEARCOAT\n	vec3 clearcoatNormal = nonPerturbedNormal;\n#endif",
	clearcoat_normal_fragment_maps: "#ifdef USE_CLEARCOAT_NORMALMAP\n	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;\n	clearcoatMapN.xy *= clearcoatNormalScale;\n	clearcoatNormal = normalize( tbn2 * clearcoatMapN );\n#endif",
	clearcoat_pars_fragment: "#ifdef USE_CLEARCOATMAP\n	uniform sampler2D clearcoatMap;\n#endif\n#ifdef USE_CLEARCOAT_NORMALMAP\n	uniform sampler2D clearcoatNormalMap;\n	uniform vec2 clearcoatNormalScale;\n#endif\n#ifdef USE_CLEARCOAT_ROUGHNESSMAP\n	uniform sampler2D clearcoatRoughnessMap;\n#endif",
	iridescence_pars_fragment: "#ifdef USE_IRIDESCENCEMAP\n	uniform sampler2D iridescenceMap;\n#endif\n#ifdef USE_IRIDESCENCE_THICKNESSMAP\n	uniform sampler2D iridescenceThicknessMap;\n#endif",
	opaque_fragment: "#ifdef OPAQUE\ndiffuseColor.a = 1.0;\n#endif\n#ifdef USE_TRANSMISSION\ndiffuseColor.a *= material.transmissionAlpha;\n#endif\ngl_FragColor = vec4( outgoingLight, diffuseColor.a );",
	packing: "vec3 packNormalToRGB( const in vec3 normal ) {\n	return normalize( normal ) * 0.5 + 0.5;\n}\nvec3 unpackRGBToNormal( const in vec3 rgb ) {\n	return 2.0 * rgb.xyz - 1.0;\n}\nconst float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;\nconst float Inv255 = 1. / 255.;\nconst vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );\nconst vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );\nconst vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );\nconst vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );\nvec4 packDepthToRGBA( const in float v ) {\n	if( v <= 0.0 )\n		return vec4( 0., 0., 0., 0. );\n	if( v >= 1.0 )\n		return vec4( 1., 1., 1., 1. );\n	float vuf;\n	float af = modf( v * PackFactors.a, vuf );\n	float bf = modf( vuf * ShiftRight8, vuf );\n	float gf = modf( vuf * ShiftRight8, vuf );\n	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );\n}\nvec3 packDepthToRGB( const in float v ) {\n	if( v <= 0.0 )\n		return vec3( 0., 0., 0. );\n	if( v >= 1.0 )\n		return vec3( 1., 1., 1. );\n	float vuf;\n	float bf = modf( v * PackFactors.b, vuf );\n	float gf = modf( vuf * ShiftRight8, vuf );\n	return vec3( vuf * Inv255, gf * PackUpscale, bf );\n}\nvec2 packDepthToRG( const in float v ) {\n	if( v <= 0.0 )\n		return vec2( 0., 0. );\n	if( v >= 1.0 )\n		return vec2( 1., 1. );\n	float vuf;\n	float gf = modf( v * 256., vuf );\n	return vec2( vuf * Inv255, gf );\n}\nfloat unpackRGBAToDepth( const in vec4 v ) {\n	return dot( v, UnpackFactors4 );\n}\nfloat unpackRGBToDepth( const in vec3 v ) {\n	return dot( v, UnpackFactors3 );\n}\nfloat unpackRGToDepth( const in vec2 v ) {\n	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;\n}\nvec4 pack2HalfToRGBA( const in vec2 v ) {\n	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );\n	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );\n}\nvec2 unpackRGBATo2Half( const in vec4 v ) {\n	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );\n}\nfloat viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {\n	return ( viewZ + near ) / ( near - far );\n}\nfloat orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {\n	#ifdef USE_REVERSED_DEPTH_BUFFER\n	\n		return depth * ( far - near ) - far;\n	#else\n		return depth * ( near - far ) - near;\n	#endif\n}\nfloat viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {\n	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );\n}\nfloat perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {\n	\n	#ifdef USE_REVERSED_DEPTH_BUFFER\n		return ( near * far ) / ( ( near - far ) * depth - near );\n	#else\n		return ( near * far ) / ( ( far - near ) * depth - far );\n	#endif\n}",
	premultiplied_alpha_fragment: "#ifdef PREMULTIPLIED_ALPHA\n	gl_FragColor.rgb *= gl_FragColor.a;\n#endif",
	project_vertex: "vec4 mvPosition = vec4( transformed, 1.0 );\n#ifdef USE_BATCHING\n	mvPosition = batchingMatrix * mvPosition;\n#endif\n#ifdef USE_INSTANCING\n	mvPosition = instanceMatrix * mvPosition;\n#endif\nmvPosition = modelViewMatrix * mvPosition;\ngl_Position = projectionMatrix * mvPosition;",
	dithering_fragment: "#ifdef DITHERING\n	gl_FragColor.rgb = dithering( gl_FragColor.rgb );\n#endif",
	dithering_pars_fragment: "#ifdef DITHERING\n	vec3 dithering( vec3 color ) {\n		float grid_position = rand( gl_FragCoord.xy );\n		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );\n		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );\n		return color + dither_shift_RGB;\n	}\n#endif",
	roughnessmap_fragment: "float roughnessFactor = roughness;\n#ifdef USE_ROUGHNESSMAP\n	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );\n	roughnessFactor *= texelRoughness.g;\n#endif",
	roughnessmap_pars_fragment: "#ifdef USE_ROUGHNESSMAP\n	uniform sampler2D roughnessMap;\n#endif",
	shadowmap_pars_fragment: "#if NUM_SPOT_LIGHT_COORDS > 0\n	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];\n#endif\n#if NUM_SPOT_LIGHT_MAPS > 0\n	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];\n#endif\n#ifdef USE_SHADOWMAP\n	#if NUM_DIR_LIGHT_SHADOWS > 0\n		#if defined( SHADOWMAP_TYPE_PCF )\n			uniform sampler2DShadow directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];\n		#else\n			uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];\n		#endif\n		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];\n		struct DirectionalLightShadow {\n			float shadowIntensity;\n			float shadowBias;\n			float shadowNormalBias;\n			float shadowRadius;\n			vec2 shadowMapSize;\n		};\n		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];\n	#endif\n	#if NUM_SPOT_LIGHT_SHADOWS > 0\n		#if defined( SHADOWMAP_TYPE_PCF )\n			uniform sampler2DShadow spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];\n		#else\n			uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];\n		#endif\n		struct SpotLightShadow {\n			float shadowIntensity;\n			float shadowBias;\n			float shadowNormalBias;\n			float shadowRadius;\n			vec2 shadowMapSize;\n		};\n		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];\n	#endif\n	#if NUM_POINT_LIGHT_SHADOWS > 0\n		#if defined( SHADOWMAP_TYPE_PCF )\n			uniform samplerCubeShadow pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];\n		#elif defined( SHADOWMAP_TYPE_BASIC )\n			uniform samplerCube pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];\n		#endif\n		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];\n		struct PointLightShadow {\n			float shadowIntensity;\n			float shadowBias;\n			float shadowNormalBias;\n			float shadowRadius;\n			vec2 shadowMapSize;\n			float shadowCameraNear;\n			float shadowCameraFar;\n		};\n		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];\n	#endif\n	#if defined( SHADOWMAP_TYPE_PCF )\n		float interleavedGradientNoise( vec2 position ) {\n			return fract( 52.9829189 * fract( dot( position, vec2( 0.06711056, 0.00583715 ) ) ) );\n		}\n		vec2 vogelDiskSample( int sampleIndex, int samplesCount, float phi ) {\n			const float goldenAngle = 2.399963229728653;\n			float r = sqrt( ( float( sampleIndex ) + 0.5 ) / float( samplesCount ) );\n			float theta = float( sampleIndex ) * goldenAngle + phi;\n			return vec2( cos( theta ), sin( theta ) ) * r;\n		}\n	#endif\n	#if defined( SHADOWMAP_TYPE_PCF )\n		float getShadow( sampler2DShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {\n			float shadow = 1.0;\n			shadowCoord.xyz /= shadowCoord.w;\n			shadowCoord.z += shadowBias;\n			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;\n			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;\n			if ( frustumTest ) {\n				vec2 texelSize = vec2( 1.0 ) / shadowMapSize;\n				float radius = shadowRadius * texelSize.x;\n				float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;\n				shadow = (\n					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 0, 5, phi ) * radius, shadowCoord.z ) ) +\n					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 1, 5, phi ) * radius, shadowCoord.z ) ) +\n					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 2, 5, phi ) * radius, shadowCoord.z ) ) +\n					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 3, 5, phi ) * radius, shadowCoord.z ) ) +\n					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 4, 5, phi ) * radius, shadowCoord.z ) )\n				) * 0.2;\n			}\n			return mix( 1.0, shadow, shadowIntensity );\n		}\n	#elif defined( SHADOWMAP_TYPE_VSM )\n		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {\n			float shadow = 1.0;\n			shadowCoord.xyz /= shadowCoord.w;\n			#ifdef USE_REVERSED_DEPTH_BUFFER\n				shadowCoord.z -= shadowBias;\n			#else\n				shadowCoord.z += shadowBias;\n			#endif\n			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;\n			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;\n			if ( frustumTest ) {\n				vec2 distribution = texture2D( shadowMap, shadowCoord.xy ).rg;\n				float mean = distribution.x;\n				float variance = distribution.y * distribution.y;\n				#ifdef USE_REVERSED_DEPTH_BUFFER\n					float hard_shadow = step( mean, shadowCoord.z );\n				#else\n					float hard_shadow = step( shadowCoord.z, mean );\n				#endif\n				\n				if ( hard_shadow == 1.0 ) {\n					shadow = 1.0;\n				} else {\n					variance = max( variance, 0.0000001 );\n					float d = shadowCoord.z - mean;\n					float p_max = variance / ( variance + d * d );\n					p_max = clamp( ( p_max - 0.3 ) / 0.65, 0.0, 1.0 );\n					shadow = max( hard_shadow, p_max );\n				}\n			}\n			return mix( 1.0, shadow, shadowIntensity );\n		}\n	#else\n		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {\n			float shadow = 1.0;\n			shadowCoord.xyz /= shadowCoord.w;\n			#ifdef USE_REVERSED_DEPTH_BUFFER\n				shadowCoord.z -= shadowBias;\n			#else\n				shadowCoord.z += shadowBias;\n			#endif\n			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;\n			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;\n			if ( frustumTest ) {\n				float depth = texture2D( shadowMap, shadowCoord.xy ).r;\n				#ifdef USE_REVERSED_DEPTH_BUFFER\n					shadow = step( depth, shadowCoord.z );\n				#else\n					shadow = step( shadowCoord.z, depth );\n				#endif\n			}\n			return mix( 1.0, shadow, shadowIntensity );\n		}\n	#endif\n	#if NUM_POINT_LIGHT_SHADOWS > 0\n	#if defined( SHADOWMAP_TYPE_PCF )\n	float getPointShadow( samplerCubeShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {\n		float shadow = 1.0;\n		vec3 lightToPosition = shadowCoord.xyz;\n		vec3 bd3D = normalize( lightToPosition );\n		vec3 absVec = abs( lightToPosition );\n		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );\n		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {\n			#ifdef USE_REVERSED_DEPTH_BUFFER\n				float dp = ( shadowCameraNear * ( shadowCameraFar - viewSpaceZ ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );\n				dp -= shadowBias;\n			#else\n				float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );\n				dp += shadowBias;\n			#endif\n			float texelSize = shadowRadius / shadowMapSize.x;\n			vec3 absDir = abs( bd3D );\n			vec3 tangent = absDir.x > absDir.z ? vec3( 0.0, 1.0, 0.0 ) : vec3( 1.0, 0.0, 0.0 );\n			tangent = normalize( cross( bd3D, tangent ) );\n			vec3 bitangent = cross( bd3D, tangent );\n			float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;\n			vec2 sample0 = vogelDiskSample( 0, 5, phi );\n			vec2 sample1 = vogelDiskSample( 1, 5, phi );\n			vec2 sample2 = vogelDiskSample( 2, 5, phi );\n			vec2 sample3 = vogelDiskSample( 3, 5, phi );\n			vec2 sample4 = vogelDiskSample( 4, 5, phi );\n			shadow = (\n				texture( shadowMap, vec4( bd3D + ( tangent * sample0.x + bitangent * sample0.y ) * texelSize, dp ) ) +\n				texture( shadowMap, vec4( bd3D + ( tangent * sample1.x + bitangent * sample1.y ) * texelSize, dp ) ) +\n				texture( shadowMap, vec4( bd3D + ( tangent * sample2.x + bitangent * sample2.y ) * texelSize, dp ) ) +\n				texture( shadowMap, vec4( bd3D + ( tangent * sample3.x + bitangent * sample3.y ) * texelSize, dp ) ) +\n				texture( shadowMap, vec4( bd3D + ( tangent * sample4.x + bitangent * sample4.y ) * texelSize, dp ) )\n			) * 0.2;\n		}\n		return mix( 1.0, shadow, shadowIntensity );\n	}\n	#elif defined( SHADOWMAP_TYPE_BASIC )\n	float getPointShadow( samplerCube shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {\n		float shadow = 1.0;\n		vec3 lightToPosition = shadowCoord.xyz;\n		vec3 absVec = abs( lightToPosition );\n		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );\n		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {\n			float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );\n			dp += shadowBias;\n			vec3 bd3D = normalize( lightToPosition );\n			float depth = textureCube( shadowMap, bd3D ).r;\n			#ifdef USE_REVERSED_DEPTH_BUFFER\n				depth = 1.0 - depth;\n			#endif\n			shadow = step( dp, depth );\n		}\n		return mix( 1.0, shadow, shadowIntensity );\n	}\n	#endif\n	#endif\n#endif",
	shadowmap_pars_vertex: "#if NUM_SPOT_LIGHT_COORDS > 0\n	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];\n	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];\n#endif\n#ifdef USE_SHADOWMAP\n	#if NUM_DIR_LIGHT_SHADOWS > 0\n		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];\n		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];\n		struct DirectionalLightShadow {\n			float shadowIntensity;\n			float shadowBias;\n			float shadowNormalBias;\n			float shadowRadius;\n			vec2 shadowMapSize;\n		};\n		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];\n	#endif\n	#if NUM_SPOT_LIGHT_SHADOWS > 0\n		struct SpotLightShadow {\n			float shadowIntensity;\n			float shadowBias;\n			float shadowNormalBias;\n			float shadowRadius;\n			vec2 shadowMapSize;\n		};\n		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];\n	#endif\n	#if NUM_POINT_LIGHT_SHADOWS > 0\n		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];\n		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];\n		struct PointLightShadow {\n			float shadowIntensity;\n			float shadowBias;\n			float shadowNormalBias;\n			float shadowRadius;\n			vec2 shadowMapSize;\n			float shadowCameraNear;\n			float shadowCameraFar;\n		};\n		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];\n	#endif\n#endif",
	shadowmap_vertex: "#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )\n	#ifdef HAS_NORMAL\n		vec3 shadowWorldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );\n	#else\n		vec3 shadowWorldNormal = vec3( 0.0 );\n	#endif\n	vec4 shadowWorldPosition;\n#endif\n#if defined( USE_SHADOWMAP )\n	#if NUM_DIR_LIGHT_SHADOWS > 0\n		#pragma unroll_loop_start\n		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {\n			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );\n			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;\n		}\n		#pragma unroll_loop_end\n	#endif\n	#if NUM_POINT_LIGHT_SHADOWS > 0\n		#pragma unroll_loop_start\n		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {\n			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );\n			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;\n		}\n		#pragma unroll_loop_end\n	#endif\n#endif\n#if NUM_SPOT_LIGHT_COORDS > 0\n	#pragma unroll_loop_start\n	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {\n		shadowWorldPosition = worldPosition;\n		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )\n			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;\n		#endif\n		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;\n	}\n	#pragma unroll_loop_end\n#endif",
	shadowmask_pars_fragment: "float getShadowMask() {\n	float shadow = 1.0;\n	#ifdef USE_SHADOWMAP\n	#if NUM_DIR_LIGHT_SHADOWS > 0\n	DirectionalLightShadow directionalLight;\n	#pragma unroll_loop_start\n	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {\n		directionalLight = directionalLightShadows[ i ];\n		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;\n	}\n	#pragma unroll_loop_end\n	#endif\n	#if NUM_SPOT_LIGHT_SHADOWS > 0\n	SpotLightShadow spotLight;\n	#pragma unroll_loop_start\n	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {\n		spotLight = spotLightShadows[ i ];\n		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;\n	}\n	#pragma unroll_loop_end\n	#endif\n	#if NUM_POINT_LIGHT_SHADOWS > 0 && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )\n	PointLightShadow pointLight;\n	#pragma unroll_loop_start\n	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {\n		pointLight = pointLightShadows[ i ];\n		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;\n	}\n	#pragma unroll_loop_end\n	#endif\n	#endif\n	return shadow;\n}",
	skinbase_vertex: "#ifdef USE_SKINNING\n	mat4 boneMatX = getBoneMatrix( skinIndex.x );\n	mat4 boneMatY = getBoneMatrix( skinIndex.y );\n	mat4 boneMatZ = getBoneMatrix( skinIndex.z );\n	mat4 boneMatW = getBoneMatrix( skinIndex.w );\n#endif",
	skinning_pars_vertex: "#ifdef USE_SKINNING\n	uniform mat4 bindMatrix;\n	uniform mat4 bindMatrixInverse;\n	uniform highp sampler2D boneTexture;\n	mat4 getBoneMatrix( const in float i ) {\n		int size = textureSize( boneTexture, 0 ).x;\n		int j = int( i ) * 4;\n		int x = j % size;\n		int y = j / size;\n		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );\n		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );\n		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );\n		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );\n		return mat4( v1, v2, v3, v4 );\n	}\n#endif",
	skinning_vertex: "#ifdef USE_SKINNING\n	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );\n	vec4 skinned = vec4( 0.0 );\n	skinned += boneMatX * skinVertex * skinWeight.x;\n	skinned += boneMatY * skinVertex * skinWeight.y;\n	skinned += boneMatZ * skinVertex * skinWeight.z;\n	skinned += boneMatW * skinVertex * skinWeight.w;\n	transformed = ( bindMatrixInverse * skinned ).xyz;\n#endif",
	skinnormal_vertex: "#ifdef USE_SKINNING\n	mat4 skinMatrix = mat4( 0.0 );\n	skinMatrix += skinWeight.x * boneMatX;\n	skinMatrix += skinWeight.y * boneMatY;\n	skinMatrix += skinWeight.z * boneMatZ;\n	skinMatrix += skinWeight.w * boneMatW;\n	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;\n	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;\n	#ifdef USE_TANGENT\n		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;\n	#endif\n#endif",
	specularmap_fragment: "float specularStrength;\n#ifdef USE_SPECULARMAP\n	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );\n	specularStrength = texelSpecular.r;\n#else\n	specularStrength = 1.0;\n#endif",
	specularmap_pars_fragment: "#ifdef USE_SPECULARMAP\n	uniform sampler2D specularMap;\n#endif",
	tonemapping_fragment: "#if defined( TONE_MAPPING )\n	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );\n#endif",
	tonemapping_pars_fragment: "#ifndef saturate\n#define saturate( a ) clamp( a, 0.0, 1.0 )\n#endif\nuniform float toneMappingExposure;\nvec3 LinearToneMapping( vec3 color ) {\n	return saturate( toneMappingExposure * color );\n}\nvec3 ReinhardToneMapping( vec3 color ) {\n	color *= toneMappingExposure;\n	return saturate( color / ( vec3( 1.0 ) + color ) );\n}\nvec3 CineonToneMapping( vec3 color ) {\n	color *= toneMappingExposure;\n	color = max( vec3( 0.0 ), color - 0.004 );\n	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );\n}\nvec3 RRTAndODTFit( vec3 v ) {\n	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;\n	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;\n	return a / b;\n}\nvec3 ACESFilmicToneMapping( vec3 color ) {\n	const mat3 ACESInputMat = mat3(\n		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),\n		vec3( 0.04823, 0.01566, 0.83777 )\n	);\n	const mat3 ACESOutputMat = mat3(\n		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),\n		vec3( -0.07367, -0.00605,  1.07602 )\n	);\n	color *= toneMappingExposure / 0.6;\n	color = ACESInputMat * color;\n	color = RRTAndODTFit( color );\n	color = ACESOutputMat * color;\n	return saturate( color );\n}\nconst mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(\n	vec3( 1.6605, - 0.1246, - 0.0182 ),\n	vec3( - 0.5876, 1.1329, - 0.1006 ),\n	vec3( - 0.0728, - 0.0083, 1.1187 )\n);\nconst mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(\n	vec3( 0.6274, 0.0691, 0.0164 ),\n	vec3( 0.3293, 0.9195, 0.0880 ),\n	vec3( 0.0433, 0.0113, 0.8956 )\n);\nvec3 agxDefaultContrastApprox( vec3 x ) {\n	vec3 x2 = x * x;\n	vec3 x4 = x2 * x2;\n	return + 15.5 * x4 * x2\n		- 40.14 * x4 * x\n		+ 31.96 * x4\n		- 6.868 * x2 * x\n		+ 0.4298 * x2\n		+ 0.1191 * x\n		- 0.00232;\n}\nvec3 AgXToneMapping( vec3 color ) {\n	const mat3 AgXInsetMatrix = mat3(\n		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),\n		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),\n		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )\n	);\n	const mat3 AgXOutsetMatrix = mat3(\n		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),\n		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),\n		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )\n	);\n	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;\n	color *= toneMappingExposure;\n	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;\n	color = AgXInsetMatrix * color;\n	color = max( color, 1e-10 );	color = log2( color );\n	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );\n	color = clamp( color, 0.0, 1.0 );\n	color = agxDefaultContrastApprox( color );\n	color = AgXOutsetMatrix * color;\n	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );\n	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;\n	color = clamp( color, 0.0, 1.0 );\n	return color;\n}\nvec3 NeutralToneMapping( vec3 color ) {\n	const float StartCompression = 0.8 - 0.04;\n	const float Desaturation = 0.15;\n	color *= toneMappingExposure;\n	float x = min( color.r, min( color.g, color.b ) );\n	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;\n	color -= offset;\n	float peak = max( color.r, max( color.g, color.b ) );\n	if ( peak < StartCompression ) return color;\n	float d = 1. - StartCompression;\n	float newPeak = 1. - d * d / ( peak + d - StartCompression );\n	color *= newPeak / peak;\n	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );\n	return mix( color, vec3( newPeak ), g );\n}\nvec3 CustomToneMapping( vec3 color ) { return color; }",
	transmission_fragment: "#ifdef USE_TRANSMISSION\n	material.transmission = transmission;\n	material.transmissionAlpha = 1.0;\n	material.thickness = thickness;\n	material.attenuationDistance = attenuationDistance;\n	material.attenuationColor = attenuationColor;\n	#ifdef USE_TRANSMISSIONMAP\n		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;\n	#endif\n	#ifdef USE_THICKNESSMAP\n		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;\n	#endif\n	vec3 pos = vWorldPosition;\n	vec3 v = normalize( cameraPosition - pos );\n	vec3 n = transformNormalByInverseViewMatrix( normal, viewMatrix );\n	vec4 transmitted = getIBLVolumeRefraction(\n		n, v, material.roughness, material.diffuseContribution, material.specularColorBlended, material.specularF90,\n		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,\n		material.attenuationColor, material.attenuationDistance );\n	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );\n	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );\n#endif",
	transmission_pars_fragment: "#ifdef USE_TRANSMISSION\n	uniform float transmission;\n	uniform float thickness;\n	uniform float attenuationDistance;\n	uniform vec3 attenuationColor;\n	#ifdef USE_TRANSMISSIONMAP\n		uniform sampler2D transmissionMap;\n	#endif\n	#ifdef USE_THICKNESSMAP\n		uniform sampler2D thicknessMap;\n	#endif\n	uniform vec2 transmissionSamplerSize;\n	uniform sampler2D transmissionSamplerMap;\n	uniform mat4 modelMatrix;\n	uniform mat4 projectionMatrix;\n	varying vec3 vWorldPosition;\n	float w0( float a ) {\n		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );\n	}\n	float w1( float a ) {\n		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );\n	}\n	float w2( float a ){\n		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );\n	}\n	float w3( float a ) {\n		return ( 1.0 / 6.0 ) * ( a * a * a );\n	}\n	float g0( float a ) {\n		return w0( a ) + w1( a );\n	}\n	float g1( float a ) {\n		return w2( a ) + w3( a );\n	}\n	float h0( float a ) {\n		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );\n	}\n	float h1( float a ) {\n		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );\n	}\n	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {\n		uv = uv * texelSize.zw + 0.5;\n		vec2 iuv = floor( uv );\n		vec2 fuv = fract( uv );\n		float g0x = g0( fuv.x );\n		float g1x = g1( fuv.x );\n		float h0x = h0( fuv.x );\n		float h1x = h1( fuv.x );\n		float h0y = h0( fuv.y );\n		float h1y = h1( fuv.y );\n		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;\n		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;\n		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;\n		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;\n		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +\n			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );\n	}\n	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {\n		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );\n		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );\n		vec2 fLodSizeInv = 1.0 / fLodSize;\n		vec2 cLodSizeInv = 1.0 / cLodSize;\n		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );\n		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );\n		return mix( fSample, cSample, fract( lod ) );\n	}\n	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {\n		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );\n		vec3 modelScale;\n		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );\n		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );\n		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );\n		return normalize( refractionVector ) * thickness * modelScale;\n	}\n	float applyIorToRoughness( const in float roughness, const in float ior ) {\n		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );\n	}\n	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {\n		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );\n		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );\n	}\n	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {\n		if ( isinf( attenuationDistance ) ) {\n			return vec3( 1.0 );\n		} else {\n			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;\n			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;\n		}\n	}\n	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,\n		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,\n		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,\n		const in vec3 attenuationColor, const in float attenuationDistance ) {\n		vec4 transmittedLight;\n		vec3 transmittance;\n		#ifdef USE_DISPERSION\n			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;\n			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );\n			for ( int i = 0; i < 3; i ++ ) {\n				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );\n				vec3 refractedRayExit = position + transmissionRay;\n				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );\n				vec2 refractionCoords = ndcPos.xy / ndcPos.w;\n				refractionCoords += 1.0;\n				refractionCoords /= 2.0;\n				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );\n				transmittedLight[ i ] = transmissionSample[ i ];\n				transmittedLight.a += transmissionSample.a;\n				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];\n			}\n			transmittedLight.a /= 3.0;\n		#else\n			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );\n			vec3 refractedRayExit = position + transmissionRay;\n			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );\n			vec2 refractionCoords = ndcPos.xy / ndcPos.w;\n			refractionCoords += 1.0;\n			refractionCoords /= 2.0;\n			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );\n			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );\n		#endif\n		vec3 attenuatedColor = transmittance * transmittedLight.rgb;\n		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );\n		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;\n		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );\n	}\n#endif",
	uv_pars_fragment: "#if defined( USE_UV ) || defined( USE_ANISOTROPY )\n	varying vec2 vUv;\n#endif\n#ifdef USE_MAP\n	varying vec2 vMapUv;\n#endif\n#ifdef USE_ALPHAMAP\n	varying vec2 vAlphaMapUv;\n#endif\n#ifdef USE_LIGHTMAP\n	varying vec2 vLightMapUv;\n#endif\n#ifdef USE_AOMAP\n	varying vec2 vAoMapUv;\n#endif\n#ifdef USE_BUMPMAP\n	varying vec2 vBumpMapUv;\n#endif\n#ifdef USE_NORMALMAP\n	varying vec2 vNormalMapUv;\n#endif\n#ifdef USE_EMISSIVEMAP\n	varying vec2 vEmissiveMapUv;\n#endif\n#ifdef USE_METALNESSMAP\n	varying vec2 vMetalnessMapUv;\n#endif\n#ifdef USE_ROUGHNESSMAP\n	varying vec2 vRoughnessMapUv;\n#endif\n#ifdef USE_ANISOTROPYMAP\n	varying vec2 vAnisotropyMapUv;\n#endif\n#ifdef USE_CLEARCOATMAP\n	varying vec2 vClearcoatMapUv;\n#endif\n#ifdef USE_CLEARCOAT_NORMALMAP\n	varying vec2 vClearcoatNormalMapUv;\n#endif\n#ifdef USE_CLEARCOAT_ROUGHNESSMAP\n	varying vec2 vClearcoatRoughnessMapUv;\n#endif\n#ifdef USE_IRIDESCENCEMAP\n	varying vec2 vIridescenceMapUv;\n#endif\n#ifdef USE_IRIDESCENCE_THICKNESSMAP\n	varying vec2 vIridescenceThicknessMapUv;\n#endif\n#ifdef USE_SHEEN_COLORMAP\n	varying vec2 vSheenColorMapUv;\n#endif\n#ifdef USE_SHEEN_ROUGHNESSMAP\n	varying vec2 vSheenRoughnessMapUv;\n#endif\n#ifdef USE_SPECULARMAP\n	varying vec2 vSpecularMapUv;\n#endif\n#ifdef USE_SPECULAR_COLORMAP\n	varying vec2 vSpecularColorMapUv;\n#endif\n#ifdef USE_SPECULAR_INTENSITYMAP\n	varying vec2 vSpecularIntensityMapUv;\n#endif\n#ifdef USE_TRANSMISSIONMAP\n	uniform mat3 transmissionMapTransform;\n	varying vec2 vTransmissionMapUv;\n#endif\n#ifdef USE_THICKNESSMAP\n	uniform mat3 thicknessMapTransform;\n	varying vec2 vThicknessMapUv;\n#endif",
	uv_pars_vertex: "#if defined( USE_UV ) || defined( USE_ANISOTROPY )\n	varying vec2 vUv;\n#endif\n#ifdef USE_MAP\n	uniform mat3 mapTransform;\n	varying vec2 vMapUv;\n#endif\n#ifdef USE_ALPHAMAP\n	uniform mat3 alphaMapTransform;\n	varying vec2 vAlphaMapUv;\n#endif\n#ifdef USE_LIGHTMAP\n	uniform mat3 lightMapTransform;\n	varying vec2 vLightMapUv;\n#endif\n#ifdef USE_AOMAP\n	uniform mat3 aoMapTransform;\n	varying vec2 vAoMapUv;\n#endif\n#ifdef USE_BUMPMAP\n	uniform mat3 bumpMapTransform;\n	varying vec2 vBumpMapUv;\n#endif\n#ifdef USE_NORMALMAP\n	uniform mat3 normalMapTransform;\n	varying vec2 vNormalMapUv;\n#endif\n#ifdef USE_DISPLACEMENTMAP\n	uniform mat3 displacementMapTransform;\n	varying vec2 vDisplacementMapUv;\n#endif\n#ifdef USE_EMISSIVEMAP\n	uniform mat3 emissiveMapTransform;\n	varying vec2 vEmissiveMapUv;\n#endif\n#ifdef USE_METALNESSMAP\n	uniform mat3 metalnessMapTransform;\n	varying vec2 vMetalnessMapUv;\n#endif\n#ifdef USE_ROUGHNESSMAP\n	uniform mat3 roughnessMapTransform;\n	varying vec2 vRoughnessMapUv;\n#endif\n#ifdef USE_ANISOTROPYMAP\n	uniform mat3 anisotropyMapTransform;\n	varying vec2 vAnisotropyMapUv;\n#endif\n#ifdef USE_CLEARCOATMAP\n	uniform mat3 clearcoatMapTransform;\n	varying vec2 vClearcoatMapUv;\n#endif\n#ifdef USE_CLEARCOAT_NORMALMAP\n	uniform mat3 clearcoatNormalMapTransform;\n	varying vec2 vClearcoatNormalMapUv;\n#endif\n#ifdef USE_CLEARCOAT_ROUGHNESSMAP\n	uniform mat3 clearcoatRoughnessMapTransform;\n	varying vec2 vClearcoatRoughnessMapUv;\n#endif\n#ifdef USE_SHEEN_COLORMAP\n	uniform mat3 sheenColorMapTransform;\n	varying vec2 vSheenColorMapUv;\n#endif\n#ifdef USE_SHEEN_ROUGHNESSMAP\n	uniform mat3 sheenRoughnessMapTransform;\n	varying vec2 vSheenRoughnessMapUv;\n#endif\n#ifdef USE_IRIDESCENCEMAP\n	uniform mat3 iridescenceMapTransform;\n	varying vec2 vIridescenceMapUv;\n#endif\n#ifdef USE_IRIDESCENCE_THICKNESSMAP\n	uniform mat3 iridescenceThicknessMapTransform;\n	varying vec2 vIridescenceThicknessMapUv;\n#endif\n#ifdef USE_SPECULARMAP\n	uniform mat3 specularMapTransform;\n	varying vec2 vSpecularMapUv;\n#endif\n#ifdef USE_SPECULAR_COLORMAP\n	uniform mat3 specularColorMapTransform;\n	varying vec2 vSpecularColorMapUv;\n#endif\n#ifdef USE_SPECULAR_INTENSITYMAP\n	uniform mat3 specularIntensityMapTransform;\n	varying vec2 vSpecularIntensityMapUv;\n#endif\n#ifdef USE_TRANSMISSIONMAP\n	uniform mat3 transmissionMapTransform;\n	varying vec2 vTransmissionMapUv;\n#endif\n#ifdef USE_THICKNESSMAP\n	uniform mat3 thicknessMapTransform;\n	varying vec2 vThicknessMapUv;\n#endif",
	uv_vertex: "#if defined( USE_UV ) || defined( USE_ANISOTROPY )\n	vUv = vec3( uv, 1 ).xy;\n#endif\n#ifdef USE_MAP\n	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_ALPHAMAP\n	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_LIGHTMAP\n	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_AOMAP\n	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_BUMPMAP\n	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_NORMALMAP\n	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_DISPLACEMENTMAP\n	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_EMISSIVEMAP\n	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_METALNESSMAP\n	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_ROUGHNESSMAP\n	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_ANISOTROPYMAP\n	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_CLEARCOATMAP\n	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_CLEARCOAT_NORMALMAP\n	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_CLEARCOAT_ROUGHNESSMAP\n	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_IRIDESCENCEMAP\n	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_IRIDESCENCE_THICKNESSMAP\n	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_SHEEN_COLORMAP\n	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_SHEEN_ROUGHNESSMAP\n	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_SPECULARMAP\n	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_SPECULAR_COLORMAP\n	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_SPECULAR_INTENSITYMAP\n	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_TRANSMISSIONMAP\n	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;\n#endif\n#ifdef USE_THICKNESSMAP\n	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;\n#endif",
	worldpos_vertex: "#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0\n	vec4 worldPosition = vec4( transformed, 1.0 );\n	#ifdef USE_BATCHING\n		worldPosition = batchingMatrix * worldPosition;\n	#endif\n	#ifdef USE_INSTANCING\n		worldPosition = instanceMatrix * worldPosition;\n	#endif\n	worldPosition = modelMatrix * worldPosition;\n#endif",
	background_vert: "varying vec2 vUv;\nuniform mat3 uvTransform;\nvoid main() {\n	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;\n	gl_Position = vec4( position.xy, 1.0, 1.0 );\n}",
	background_frag: "uniform sampler2D t2D;\nuniform float backgroundIntensity;\nvarying vec2 vUv;\nvoid main() {\n	vec4 texColor = texture2D( t2D, vUv );\n	#ifdef DECODE_VIDEO_TEXTURE\n		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );\n	#endif\n	texColor.rgb *= backgroundIntensity;\n	gl_FragColor = texColor;\n	#include <tonemapping_fragment>\n	#include <colorspace_fragment>\n}",
	backgroundCube_vert: "varying vec3 vWorldDirection;\n#include <common>\nvoid main() {\n	vWorldDirection = transformDirection( position, modelMatrix );\n	#include <begin_vertex>\n	#include <project_vertex>\n	gl_Position.z = gl_Position.w;\n}",
	backgroundCube_frag: "#ifdef ENVMAP_TYPE_CUBE\n	uniform samplerCube envMap;\n#elif defined( ENVMAP_TYPE_CUBE_UV )\n	uniform sampler2D envMap;\n#endif\nuniform float backgroundBlurriness;\nuniform float backgroundIntensity;\nuniform mat3 backgroundRotation;\nvarying vec3 vWorldDirection;\n#include <cube_uv_reflection_fragment>\nvoid main() {\n	#ifdef ENVMAP_TYPE_CUBE\n		vec4 texColor = textureCube( envMap, backgroundRotation * vWorldDirection );\n	#elif defined( ENVMAP_TYPE_CUBE_UV )\n		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );\n	#else\n		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );\n	#endif\n	texColor.rgb *= backgroundIntensity;\n	gl_FragColor = texColor;\n	#include <tonemapping_fragment>\n	#include <colorspace_fragment>\n}",
	cube_vert: "varying vec3 vWorldDirection;\n#include <common>\nvoid main() {\n	vWorldDirection = transformDirection( position, modelMatrix );\n	#include <begin_vertex>\n	#include <project_vertex>\n	gl_Position.z = gl_Position.w;\n}",
	cube_frag: "uniform samplerCube tCube;\nuniform float tFlip;\nuniform float opacity;\nvarying vec3 vWorldDirection;\nvoid main() {\n	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );\n	gl_FragColor = texColor;\n	gl_FragColor.a *= opacity;\n	#include <tonemapping_fragment>\n	#include <colorspace_fragment>\n}",
	depth_vert: "#include <common>\n#include <batching_pars_vertex>\n#include <uv_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvarying vec2 vHighPrecisionZW;\nvoid main() {\n	#include <uv_vertex>\n	#include <batching_vertex>\n	#include <skinbase_vertex>\n	#include <morphinstance_vertex>\n	#ifdef USE_DISPLACEMENTMAP\n		#include <beginnormal_vertex>\n		#include <morphnormal_vertex>\n		#include <skinnormal_vertex>\n	#endif\n	#include <begin_vertex>\n	#include <morphtarget_vertex>\n	#include <skinning_vertex>\n	#include <displacementmap_vertex>\n	#include <project_vertex>\n	#include <logdepthbuf_vertex>\n	#include <clipping_planes_vertex>\n	vHighPrecisionZW = gl_Position.zw;\n}",
	depth_frag: "#if DEPTH_PACKING == 3200\n	uniform float opacity;\n#endif\n#include <common>\n#include <packing>\n#include <uv_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <alphahash_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvarying vec2 vHighPrecisionZW;\nvoid main() {\n	vec4 diffuseColor = vec4( 1.0 );\n	#include <clipping_planes_fragment>\n	#if DEPTH_PACKING == 3200\n		diffuseColor.a = opacity;\n	#endif\n	#include <map_fragment>\n	#include <alphamap_fragment>\n	#include <alphatest_fragment>\n	#include <alphahash_fragment>\n	#include <logdepthbuf_fragment>\n	#ifdef USE_REVERSED_DEPTH_BUFFER\n		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];\n	#else\n		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;\n	#endif\n	#if DEPTH_PACKING == 3200\n		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );\n	#elif DEPTH_PACKING == 3201\n		gl_FragColor = packDepthToRGBA( fragCoordZ );\n	#elif DEPTH_PACKING == 3202\n		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );\n	#elif DEPTH_PACKING == 3203\n		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );\n	#endif\n}",
	distance_vert: "#define DISTANCE\nvarying vec3 vWorldPosition;\n#include <common>\n#include <batching_pars_vertex>\n#include <uv_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n	#include <uv_vertex>\n	#include <batching_vertex>\n	#include <skinbase_vertex>\n	#include <morphinstance_vertex>\n	#ifdef USE_DISPLACEMENTMAP\n		#include <beginnormal_vertex>\n		#include <morphnormal_vertex>\n		#include <skinnormal_vertex>\n	#endif\n	#include <begin_vertex>\n	#include <morphtarget_vertex>\n	#include <skinning_vertex>\n	#include <displacementmap_vertex>\n	#include <project_vertex>\n	#include <worldpos_vertex>\n	#include <clipping_planes_vertex>\n	vWorldPosition = worldPosition.xyz;\n}",
	distance_frag: "#define DISTANCE\nuniform vec3 referencePosition;\nuniform float nearDistance;\nuniform float farDistance;\nvarying vec3 vWorldPosition;\n#include <common>\n#include <uv_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <alphahash_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n	vec4 diffuseColor = vec4( 1.0 );\n	#include <clipping_planes_fragment>\n	#include <map_fragment>\n	#include <alphamap_fragment>\n	#include <alphatest_fragment>\n	#include <alphahash_fragment>\n	float dist = length( vWorldPosition - referencePosition );\n	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );\n	dist = saturate( dist );\n	gl_FragColor = vec4( dist, 0.0, 0.0, 1.0 );\n}",
	equirect_vert: "varying vec3 vWorldDirection;\n#include <common>\nvoid main() {\n	vWorldDirection = transformDirection( position, modelMatrix );\n	#include <begin_vertex>\n	#include <project_vertex>\n}",
	equirect_frag: "uniform sampler2D tEquirect;\nvarying vec3 vWorldDirection;\n#include <common>\nvoid main() {\n	vec3 direction = normalize( vWorldDirection );\n	vec2 sampleUV = equirectUv( direction );\n	gl_FragColor = texture2D( tEquirect, sampleUV );\n	#include <tonemapping_fragment>\n	#include <colorspace_fragment>\n}",
	linedashed_vert: "uniform float scale;\nattribute float lineDistance;\nvarying float vLineDistance;\n#include <common>\n#include <uv_pars_vertex>\n#include <color_pars_vertex>\n#include <fog_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n	vLineDistance = scale * lineDistance;\n	#include <uv_vertex>\n	#include <color_vertex>\n	#include <morphinstance_vertex>\n	#include <morphcolor_vertex>\n	#include <begin_vertex>\n	#include <morphtarget_vertex>\n	#include <project_vertex>\n	#include <logdepthbuf_vertex>\n	#include <clipping_planes_vertex>\n	#include <fog_vertex>\n}",
	linedashed_frag: "uniform vec3 diffuse;\nuniform float opacity;\nuniform float dashSize;\nuniform float totalSize;\nvarying float vLineDistance;\n#include <common>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <map_pars_fragment>\n#include <fog_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n	vec4 diffuseColor = vec4( diffuse, opacity );\n	#include <clipping_planes_fragment>\n	if ( mod( vLineDistance, totalSize ) > dashSize ) {\n		discard;\n	}\n	vec3 outgoingLight = vec3( 0.0 );\n	#include <logdepthbuf_fragment>\n	#include <map_fragment>\n	#include <color_fragment>\n	outgoingLight = diffuseColor.rgb;\n	#include <opaque_fragment>\n	#include <tonemapping_fragment>\n	#include <colorspace_fragment>\n	#include <fog_fragment>\n	#include <premultiplied_alpha_fragment>\n}",
	meshbasic_vert: "#include <common>\n#include <batching_pars_vertex>\n#include <uv_pars_vertex>\n#include <envmap_pars_vertex>\n#include <color_pars_vertex>\n#include <fog_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n	#include <uv_vertex>\n	#include <color_vertex>\n	#include <morphinstance_vertex>\n	#include <morphcolor_vertex>\n	#include <batching_vertex>\n	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )\n		#include <beginnormal_vertex>\n		#include <morphnormal_vertex>\n		#include <skinbase_vertex>\n		#include <skinnormal_vertex>\n		#include <defaultnormal_vertex>\n	#endif\n	#include <begin_vertex>\n	#include <morphtarget_vertex>\n	#include <skinning_vertex>\n	#include <project_vertex>\n	#include <logdepthbuf_vertex>\n	#include <clipping_planes_vertex>\n	#include <worldpos_vertex>\n	#include <envmap_vertex>\n	#include <fog_vertex>\n}",
	meshbasic_frag: "uniform vec3 diffuse;\nuniform float opacity;\n#ifndef FLAT_SHADED\n	varying vec3 vNormal;\n#endif\n#include <common>\n#include <dithering_pars_fragment>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <alphahash_pars_fragment>\n#include <aomap_pars_fragment>\n#include <lightmap_pars_fragment>\n#include <envmap_common_pars_fragment>\n#include <envmap_pars_fragment>\n#include <fog_pars_fragment>\n#include <specularmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n	vec4 diffuseColor = vec4( diffuse, opacity );\n	#include <clipping_planes_fragment>\n	#include <logdepthbuf_fragment>\n	#include <map_fragment>\n	#include <color_fragment>\n	#include <alphamap_fragment>\n	#include <alphatest_fragment>\n	#include <alphahash_fragment>\n	#include <specularmap_fragment>\n	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );\n	#ifdef USE_LIGHTMAP\n		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );\n		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;\n	#else\n		reflectedLight.indirectDiffuse += vec3( 1.0 );\n	#endif\n	#include <aomap_fragment>\n	reflectedLight.indirectDiffuse *= diffuseColor.rgb;\n	vec3 outgoingLight = reflectedLight.indirectDiffuse;\n	#include <envmap_fragment>\n	#include <opaque_fragment>\n	#include <tonemapping_fragment>\n	#include <colorspace_fragment>\n	#include <fog_fragment>\n	#include <premultiplied_alpha_fragment>\n	#include <dithering_fragment>\n}",
	meshlambert_vert: "#define LAMBERT\nvarying vec3 vViewPosition;\n#include <common>\n#include <batching_pars_vertex>\n#include <uv_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <envmap_pars_vertex>\n#include <color_pars_vertex>\n#include <fog_pars_vertex>\n#include <normal_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <shadowmap_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n	#include <uv_vertex>\n	#include <color_vertex>\n	#include <morphinstance_vertex>\n	#include <morphcolor_vertex>\n	#include <batching_vertex>\n	#include <beginnormal_vertex>\n	#include <morphnormal_vertex>\n	#include <skinbase_vertex>\n	#include <skinnormal_vertex>\n	#include <defaultnormal_vertex>\n	#include <normal_vertex>\n	#include <begin_vertex>\n	#include <morphtarget_vertex>\n	#include <skinning_vertex>\n	#include <displacementmap_vertex>\n	#include <project_vertex>\n	#include <logdepthbuf_vertex>\n	#include <clipping_planes_vertex>\n	vViewPosition = - mvPosition.xyz;\n	#include <worldpos_vertex>\n	#include <envmap_vertex>\n	#include <shadowmap_vertex>\n	#include <fog_vertex>\n}",
	meshlambert_frag: "#define LAMBERT\nuniform vec3 diffuse;\nuniform vec3 emissive;\nuniform float opacity;\n#include <common>\n#include <dithering_pars_fragment>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <alphahash_pars_fragment>\n#include <aomap_pars_fragment>\n#include <lightmap_pars_fragment>\n#include <emissivemap_pars_fragment>\n#include <cube_uv_reflection_fragment>\n#include <envmap_common_pars_fragment>\n#include <envmap_pars_fragment>\n#include <envmap_physical_pars_fragment>\n#include <fog_pars_fragment>\n#include <bsdfs>\n#include <lights_pars_begin>\n#include <normal_pars_fragment>\n#include <lights_lambert_pars_fragment>\n#include <shadowmap_pars_fragment>\n#include <bumpmap_pars_fragment>\n#include <normalmap_pars_fragment>\n#include <specularmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n	vec4 diffuseColor = vec4( diffuse, opacity );\n	#include <clipping_planes_fragment>\n	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );\n	vec3 totalEmissiveRadiance = emissive;\n	#include <logdepthbuf_fragment>\n	#include <map_fragment>\n	#include <color_fragment>\n	#include <alphamap_fragment>\n	#include <alphatest_fragment>\n	#include <alphahash_fragment>\n	#include <specularmap_fragment>\n	#include <normal_fragment_begin>\n	#include <normal_fragment_maps>\n	#include <emissivemap_fragment>\n	#include <lights_lambert_fragment>\n	#include <lights_fragment_begin>\n	#include <lights_fragment_maps>\n	#include <lights_fragment_end>\n	#include <aomap_fragment>\n	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;\n	#include <envmap_fragment>\n	#include <opaque_fragment>\n	#include <tonemapping_fragment>\n	#include <colorspace_fragment>\n	#include <fog_fragment>\n	#include <premultiplied_alpha_fragment>\n	#include <dithering_fragment>\n}",
	meshmatcap_vert: "#define MATCAP\nvarying vec3 vViewPosition;\n#include <common>\n#include <batching_pars_vertex>\n#include <uv_pars_vertex>\n#include <color_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <fog_pars_vertex>\n#include <normal_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n	#include <uv_vertex>\n	#include <color_vertex>\n	#include <morphinstance_vertex>\n	#include <morphcolor_vertex>\n	#include <batching_vertex>\n	#include <beginnormal_vertex>\n	#include <morphnormal_vertex>\n	#include <skinbase_vertex>\n	#include <skinnormal_vertex>\n	#include <defaultnormal_vertex>\n	#include <normal_vertex>\n	#include <begin_vertex>\n	#include <morphtarget_vertex>\n	#include <skinning_vertex>\n	#include <displacementmap_vertex>\n	#include <project_vertex>\n	#include <logdepthbuf_vertex>\n	#include <clipping_planes_vertex>\n	#include <fog_vertex>\n	vViewPosition = - mvPosition.xyz;\n}",
	meshmatcap_frag: "#define MATCAP\nuniform vec3 diffuse;\nuniform float opacity;\nuniform sampler2D matcap;\nvarying vec3 vViewPosition;\n#include <common>\n#include <dithering_pars_fragment>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <alphahash_pars_fragment>\n#include <fog_pars_fragment>\n#include <normal_pars_fragment>\n#include <bumpmap_pars_fragment>\n#include <normalmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n	vec4 diffuseColor = vec4( diffuse, opacity );\n	#include <clipping_planes_fragment>\n	#include <logdepthbuf_fragment>\n	#include <map_fragment>\n	#include <color_fragment>\n	#include <alphamap_fragment>\n	#include <alphatest_fragment>\n	#include <alphahash_fragment>\n	#include <normal_fragment_begin>\n	#include <normal_fragment_maps>\n	vec3 viewDir = normalize( vViewPosition );\n	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );\n	vec3 y = cross( viewDir, x );\n	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;\n	#ifdef USE_MATCAP\n		vec4 matcapColor = texture2D( matcap, uv );\n	#else\n		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );\n	#endif\n	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;\n	#include <opaque_fragment>\n	#include <tonemapping_fragment>\n	#include <colorspace_fragment>\n	#include <fog_fragment>\n	#include <premultiplied_alpha_fragment>\n	#include <dithering_fragment>\n}",
	meshnormal_vert: "#define NORMAL\n#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )\n	varying vec3 vViewPosition;\n#endif\n#include <common>\n#include <batching_pars_vertex>\n#include <uv_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <normal_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n	#include <uv_vertex>\n	#include <batching_vertex>\n	#include <beginnormal_vertex>\n	#include <morphinstance_vertex>\n	#include <morphnormal_vertex>\n	#include <skinbase_vertex>\n	#include <skinnormal_vertex>\n	#include <defaultnormal_vertex>\n	#include <normal_vertex>\n	#include <begin_vertex>\n	#include <morphtarget_vertex>\n	#include <skinning_vertex>\n	#include <displacementmap_vertex>\n	#include <project_vertex>\n	#include <logdepthbuf_vertex>\n	#include <clipping_planes_vertex>\n#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )\n	vViewPosition = - mvPosition.xyz;\n#endif\n}",
	meshnormal_frag: "#define NORMAL\nuniform float opacity;\n#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )\n	varying vec3 vViewPosition;\n#endif\n#include <uv_pars_fragment>\n#include <normal_pars_fragment>\n#include <bumpmap_pars_fragment>\n#include <normalmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );\n	#include <clipping_planes_fragment>\n	#include <logdepthbuf_fragment>\n	#include <normal_fragment_begin>\n	#include <normal_fragment_maps>\n	gl_FragColor = vec4( normalize( normal ) * 0.5 + 0.5, diffuseColor.a );\n	#ifdef OPAQUE\n		gl_FragColor.a = 1.0;\n	#endif\n}",
	meshphong_vert: "#define PHONG\nvarying vec3 vViewPosition;\n#include <common>\n#include <batching_pars_vertex>\n#include <uv_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <envmap_pars_vertex>\n#include <color_pars_vertex>\n#include <fog_pars_vertex>\n#include <normal_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <shadowmap_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n	#include <uv_vertex>\n	#include <color_vertex>\n	#include <morphcolor_vertex>\n	#include <batching_vertex>\n	#include <beginnormal_vertex>\n	#include <morphinstance_vertex>\n	#include <morphnormal_vertex>\n	#include <skinbase_vertex>\n	#include <skinnormal_vertex>\n	#include <defaultnormal_vertex>\n	#include <normal_vertex>\n	#include <begin_vertex>\n	#include <morphtarget_vertex>\n	#include <skinning_vertex>\n	#include <displacementmap_vertex>\n	#include <project_vertex>\n	#include <logdepthbuf_vertex>\n	#include <clipping_planes_vertex>\n	vViewPosition = - mvPosition.xyz;\n	#include <worldpos_vertex>\n	#include <envmap_vertex>\n	#include <shadowmap_vertex>\n	#include <fog_vertex>\n}",
	meshphong_frag: "#define PHONG\nuniform vec3 diffuse;\nuniform vec3 emissive;\nuniform vec3 specular;\nuniform float shininess;\nuniform float opacity;\n#include <common>\n#include <dithering_pars_fragment>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <alphahash_pars_fragment>\n#include <aomap_pars_fragment>\n#include <lightmap_pars_fragment>\n#include <emissivemap_pars_fragment>\n#include <cube_uv_reflection_fragment>\n#include <envmap_common_pars_fragment>\n#include <envmap_pars_fragment>\n#include <envmap_physical_pars_fragment>\n#include <fog_pars_fragment>\n#include <bsdfs>\n#include <lights_pars_begin>\n#include <normal_pars_fragment>\n#include <lights_phong_pars_fragment>\n#include <shadowmap_pars_fragment>\n#include <bumpmap_pars_fragment>\n#include <normalmap_pars_fragment>\n#include <specularmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n	vec4 diffuseColor = vec4( diffuse, opacity );\n	#include <clipping_planes_fragment>\n	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );\n	vec3 totalEmissiveRadiance = emissive;\n	#include <logdepthbuf_fragment>\n	#include <map_fragment>\n	#include <color_fragment>\n	#include <alphamap_fragment>\n	#include <alphatest_fragment>\n	#include <alphahash_fragment>\n	#include <specularmap_fragment>\n	#include <normal_fragment_begin>\n	#include <normal_fragment_maps>\n	#include <emissivemap_fragment>\n	#include <lights_phong_fragment>\n	#include <lights_fragment_begin>\n	#include <lights_fragment_maps>\n	#include <lights_fragment_end>\n	#include <aomap_fragment>\n	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;\n	#include <envmap_fragment>\n	#include <opaque_fragment>\n	#include <tonemapping_fragment>\n	#include <colorspace_fragment>\n	#include <fog_fragment>\n	#include <premultiplied_alpha_fragment>\n	#include <dithering_fragment>\n}",
	meshphysical_vert: "#define STANDARD\nvarying vec3 vViewPosition;\n#ifdef USE_TRANSMISSION\n	varying vec3 vWorldPosition;\n#endif\n#include <common>\n#include <batching_pars_vertex>\n#include <uv_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <color_pars_vertex>\n#include <fog_pars_vertex>\n#include <normal_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <shadowmap_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n	#include <uv_vertex>\n	#include <color_vertex>\n	#include <morphinstance_vertex>\n	#include <morphcolor_vertex>\n	#include <batching_vertex>\n	#include <beginnormal_vertex>\n	#include <morphnormal_vertex>\n	#include <skinbase_vertex>\n	#include <skinnormal_vertex>\n	#include <defaultnormal_vertex>\n	#include <normal_vertex>\n	#include <begin_vertex>\n	#include <morphtarget_vertex>\n	#include <skinning_vertex>\n	#include <displacementmap_vertex>\n	#include <project_vertex>\n	#include <logdepthbuf_vertex>\n	#include <clipping_planes_vertex>\n	vViewPosition = - mvPosition.xyz;\n	#include <worldpos_vertex>\n	#include <shadowmap_vertex>\n	#include <fog_vertex>\n#ifdef USE_TRANSMISSION\n	vWorldPosition = worldPosition.xyz;\n#endif\n}",
	meshphysical_frag: "#define STANDARD\n#ifdef PHYSICAL\n	#define IOR\n	#define USE_SPECULAR\n#endif\nuniform vec3 diffuse;\nuniform vec3 emissive;\nuniform float roughness;\nuniform float metalness;\nuniform float opacity;\n#ifdef IOR\n	uniform float ior;\n#endif\n#ifdef USE_SPECULAR\n	uniform float specularIntensity;\n	uniform vec3 specularColor;\n	#ifdef USE_SPECULAR_COLORMAP\n		uniform sampler2D specularColorMap;\n	#endif\n	#ifdef USE_SPECULAR_INTENSITYMAP\n		uniform sampler2D specularIntensityMap;\n	#endif\n#endif\n#ifdef USE_CLEARCOAT\n	uniform float clearcoat;\n	uniform float clearcoatRoughness;\n#endif\n#ifdef USE_DISPERSION\n	uniform float dispersion;\n#endif\n#ifdef USE_IRIDESCENCE\n	uniform float iridescence;\n	uniform float iridescenceIOR;\n	uniform float iridescenceThicknessMinimum;\n	uniform float iridescenceThicknessMaximum;\n#endif\n#ifdef USE_SHEEN\n	uniform vec3 sheenColor;\n	uniform float sheenRoughness;\n	#ifdef USE_SHEEN_COLORMAP\n		uniform sampler2D sheenColorMap;\n	#endif\n	#ifdef USE_SHEEN_ROUGHNESSMAP\n		uniform sampler2D sheenRoughnessMap;\n	#endif\n#endif\n#ifdef USE_ANISOTROPY\n	uniform vec2 anisotropyVector;\n	#ifdef USE_ANISOTROPYMAP\n		uniform sampler2D anisotropyMap;\n	#endif\n#endif\nvarying vec3 vViewPosition;\n#include <common>\n#include <dithering_pars_fragment>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <alphahash_pars_fragment>\n#include <aomap_pars_fragment>\n#include <lightmap_pars_fragment>\n#include <emissivemap_pars_fragment>\n#include <iridescence_fragment>\n#include <cube_uv_reflection_fragment>\n#include <envmap_common_pars_fragment>\n#include <envmap_physical_pars_fragment>\n#include <fog_pars_fragment>\n#include <lights_pars_begin>\n#include <normal_pars_fragment>\n#include <lights_physical_pars_fragment>\n#include <transmission_pars_fragment>\n#include <shadowmap_pars_fragment>\n#include <bumpmap_pars_fragment>\n#include <normalmap_pars_fragment>\n#include <clearcoat_pars_fragment>\n#include <iridescence_pars_fragment>\n#include <roughnessmap_pars_fragment>\n#include <metalnessmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n	vec4 diffuseColor = vec4( diffuse, opacity );\n	#include <clipping_planes_fragment>\n	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );\n	vec3 totalEmissiveRadiance = emissive;\n	#include <logdepthbuf_fragment>\n	#include <map_fragment>\n	#include <color_fragment>\n	#include <alphamap_fragment>\n	#include <alphatest_fragment>\n	#include <alphahash_fragment>\n	#include <roughnessmap_fragment>\n	#include <metalnessmap_fragment>\n	#include <normal_fragment_begin>\n	#include <normal_fragment_maps>\n	#include <clearcoat_normal_fragment_begin>\n	#include <clearcoat_normal_fragment_maps>\n	#include <emissivemap_fragment>\n	#include <lights_physical_fragment>\n	#include <lights_fragment_begin>\n	#include <lights_fragment_maps>\n	#include <lights_fragment_end>\n	#include <aomap_fragment>\n	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;\n	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;\n	#include <transmission_fragment>\n	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;\n	#ifdef USE_SHEEN\n \n		outgoingLight = outgoingLight + sheenSpecularDirect + sheenSpecularIndirect;\n \n 	#endif\n	#ifdef USE_CLEARCOAT\n		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );\n		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );\n		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;\n	#endif\n	#include <opaque_fragment>\n	#include <tonemapping_fragment>\n	#include <colorspace_fragment>\n	#include <fog_fragment>\n	#include <premultiplied_alpha_fragment>\n	#include <dithering_fragment>\n}",
	meshtoon_vert: "#define TOON\nvarying vec3 vViewPosition;\n#include <common>\n#include <batching_pars_vertex>\n#include <uv_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <color_pars_vertex>\n#include <fog_pars_vertex>\n#include <normal_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <shadowmap_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n	#include <uv_vertex>\n	#include <color_vertex>\n	#include <morphinstance_vertex>\n	#include <morphcolor_vertex>\n	#include <batching_vertex>\n	#include <beginnormal_vertex>\n	#include <morphnormal_vertex>\n	#include <skinbase_vertex>\n	#include <skinnormal_vertex>\n	#include <defaultnormal_vertex>\n	#include <normal_vertex>\n	#include <begin_vertex>\n	#include <morphtarget_vertex>\n	#include <skinning_vertex>\n	#include <displacementmap_vertex>\n	#include <project_vertex>\n	#include <logdepthbuf_vertex>\n	#include <clipping_planes_vertex>\n	vViewPosition = - mvPosition.xyz;\n	#include <worldpos_vertex>\n	#include <shadowmap_vertex>\n	#include <fog_vertex>\n}",
	meshtoon_frag: "#define TOON\nuniform vec3 diffuse;\nuniform vec3 emissive;\nuniform float opacity;\n#include <common>\n#include <dithering_pars_fragment>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <alphahash_pars_fragment>\n#include <aomap_pars_fragment>\n#include <lightmap_pars_fragment>\n#include <emissivemap_pars_fragment>\n#include <gradientmap_pars_fragment>\n#include <fog_pars_fragment>\n#include <bsdfs>\n#include <lights_pars_begin>\n#include <normal_pars_fragment>\n#include <lights_toon_pars_fragment>\n#include <shadowmap_pars_fragment>\n#include <bumpmap_pars_fragment>\n#include <normalmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n	vec4 diffuseColor = vec4( diffuse, opacity );\n	#include <clipping_planes_fragment>\n	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );\n	vec3 totalEmissiveRadiance = emissive;\n	#include <logdepthbuf_fragment>\n	#include <map_fragment>\n	#include <color_fragment>\n	#include <alphamap_fragment>\n	#include <alphatest_fragment>\n	#include <alphahash_fragment>\n	#include <normal_fragment_begin>\n	#include <normal_fragment_maps>\n	#include <emissivemap_fragment>\n	#include <lights_toon_fragment>\n	#include <lights_fragment_begin>\n	#include <lights_fragment_maps>\n	#include <lights_fragment_end>\n	#include <aomap_fragment>\n	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;\n	#include <opaque_fragment>\n	#include <tonemapping_fragment>\n	#include <colorspace_fragment>\n	#include <fog_fragment>\n	#include <premultiplied_alpha_fragment>\n	#include <dithering_fragment>\n}",
	points_vert: "uniform float size;\nuniform float scale;\n#include <common>\n#include <color_pars_vertex>\n#include <fog_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\n#ifdef USE_POINTS_UV\n	varying vec2 vUv;\n	uniform mat3 uvTransform;\n#endif\nvoid main() {\n	#ifdef USE_POINTS_UV\n		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;\n	#endif\n	#include <color_vertex>\n	#include <morphinstance_vertex>\n	#include <morphcolor_vertex>\n	#include <begin_vertex>\n	#include <morphtarget_vertex>\n	#include <project_vertex>\n	gl_PointSize = size;\n	#ifdef USE_SIZEATTENUATION\n		bool isPerspective = isPerspectiveMatrix( projectionMatrix );\n		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );\n	#endif\n	#include <logdepthbuf_vertex>\n	#include <clipping_planes_vertex>\n	#include <worldpos_vertex>\n	#include <fog_vertex>\n}",
	points_frag: "uniform vec3 diffuse;\nuniform float opacity;\n#include <common>\n#include <color_pars_fragment>\n#include <map_particle_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <alphahash_pars_fragment>\n#include <fog_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n	vec4 diffuseColor = vec4( diffuse, opacity );\n	#include <clipping_planes_fragment>\n	vec3 outgoingLight = vec3( 0.0 );\n	#include <logdepthbuf_fragment>\n	#include <map_particle_fragment>\n	#include <color_fragment>\n	#include <alphatest_fragment>\n	#include <alphahash_fragment>\n	outgoingLight = diffuseColor.rgb;\n	#include <opaque_fragment>\n	#include <tonemapping_fragment>\n	#include <colorspace_fragment>\n	#include <fog_fragment>\n	#include <premultiplied_alpha_fragment>\n}",
	shadow_vert: "#include <common>\n#include <batching_pars_vertex>\n#include <fog_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <shadowmap_pars_vertex>\nvoid main() {\n	#include <batching_vertex>\n	#include <beginnormal_vertex>\n	#include <morphinstance_vertex>\n	#include <morphnormal_vertex>\n	#include <skinbase_vertex>\n	#include <skinnormal_vertex>\n	#include <defaultnormal_vertex>\n	#include <begin_vertex>\n	#include <morphtarget_vertex>\n	#include <skinning_vertex>\n	#include <project_vertex>\n	#include <logdepthbuf_vertex>\n	#include <worldpos_vertex>\n	#include <shadowmap_vertex>\n	#include <fog_vertex>\n}",
	shadow_frag: "uniform vec3 color;\nuniform float opacity;\n#include <common>\n#include <fog_pars_fragment>\n#include <bsdfs>\n#include <lights_pars_begin>\n#include <logdepthbuf_pars_fragment>\n#include <shadowmap_pars_fragment>\n#include <shadowmask_pars_fragment>\nvoid main() {\n	#include <logdepthbuf_fragment>\n	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );\n	#include <tonemapping_fragment>\n	#include <colorspace_fragment>\n	#include <fog_fragment>\n	#include <premultiplied_alpha_fragment>\n}",
	sprite_vert: "uniform float rotation;\nuniform vec2 center;\n#include <common>\n#include <uv_pars_vertex>\n#include <fog_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n	#include <uv_vertex>\n	vec4 mvPosition = modelViewMatrix[ 3 ];\n	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );\n	#ifndef USE_SIZEATTENUATION\n		bool isPerspective = isPerspectiveMatrix( projectionMatrix );\n		if ( isPerspective ) scale *= - mvPosition.z;\n	#endif\n	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;\n	vec2 rotatedPosition;\n	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;\n	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;\n	mvPosition.xy += rotatedPosition;\n	gl_Position = projectionMatrix * mvPosition;\n	#include <logdepthbuf_vertex>\n	#include <clipping_planes_vertex>\n	#include <fog_vertex>\n}",
	sprite_frag: "uniform vec3 diffuse;\nuniform float opacity;\n#include <common>\n#include <uv_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <alphahash_pars_fragment>\n#include <fog_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n	vec4 diffuseColor = vec4( diffuse, opacity );\n	#include <clipping_planes_fragment>\n	vec3 outgoingLight = vec3( 0.0 );\n	#include <logdepthbuf_fragment>\n	#include <map_fragment>\n	#include <alphamap_fragment>\n	#include <alphatest_fragment>\n	#include <alphahash_fragment>\n	outgoingLight = diffuseColor.rgb;\n	#include <opaque_fragment>\n	#include <tonemapping_fragment>\n	#include <colorspace_fragment>\n	#include <fog_fragment>\n}"
}, $ = {
	common: {
		diffuse: { value: /*@__PURE__*/ new Z(16777215) },
		opacity: { value: 1 },
		map: { value: null },
		mapTransform: { value: /*@__PURE__*/ new Y() },
		alphaMap: { value: null },
		alphaMapTransform: { value: /*@__PURE__*/ new Y() },
		alphaTest: { value: 0 }
	},
	specularmap: {
		specularMap: { value: null },
		specularMapTransform: { value: /*@__PURE__*/ new Y() }
	},
	envmap: {
		envMap: { value: null },
		envMapRotation: { value: /*@__PURE__*/ new Y() },
		reflectivity: { value: 1 },
		ior: { value: 1.5 },
		refractionRatio: { value: .98 },
		dfgLUT: { value: null }
	},
	aomap: {
		aoMap: { value: null },
		aoMapIntensity: { value: 1 },
		aoMapTransform: { value: /*@__PURE__*/ new Y() }
	},
	lightmap: {
		lightMap: { value: null },
		lightMapIntensity: { value: 1 },
		lightMapTransform: { value: /*@__PURE__*/ new Y() }
	},
	bumpmap: {
		bumpMap: { value: null },
		bumpMapTransform: { value: /*@__PURE__*/ new Y() },
		bumpScale: { value: 1 }
	},
	normalmap: {
		normalMap: { value: null },
		normalMapTransform: { value: /*@__PURE__*/ new Y() },
		normalScale: { value: /*@__PURE__*/ new Tf(1, 1) }
	},
	displacementmap: {
		displacementMap: { value: null },
		displacementMapTransform: { value: /*@__PURE__*/ new Y() },
		displacementScale: { value: 1 },
		displacementBias: { value: 0 }
	},
	emissivemap: {
		emissiveMap: { value: null },
		emissiveMapTransform: { value: /*@__PURE__*/ new Y() }
	},
	metalnessmap: {
		metalnessMap: { value: null },
		metalnessMapTransform: { value: /*@__PURE__*/ new Y() }
	},
	roughnessmap: {
		roughnessMap: { value: null },
		roughnessMapTransform: { value: /*@__PURE__*/ new Y() }
	},
	gradientmap: { gradientMap: { value: null } },
	fog: {
		fogDensity: { value: 25e-5 },
		fogNear: { value: 1 },
		fogFar: { value: 2e3 },
		fogColor: { value: /*@__PURE__*/ new Z(16777215) }
	},
	lights: {
		ambientLightColor: { value: [] },
		lightProbe: { value: [] },
		directionalLights: {
			value: [],
			properties: {
				direction: {},
				color: {}
			}
		},
		directionalLightShadows: {
			value: [],
			properties: {
				shadowIntensity: 1,
				shadowBias: {},
				shadowNormalBias: {},
				shadowRadius: {},
				shadowMapSize: {}
			}
		},
		directionalShadowMatrix: { value: [] },
		spotLights: {
			value: [],
			properties: {
				color: {},
				position: {},
				direction: {},
				distance: {},
				coneCos: {},
				penumbraCos: {},
				decay: {}
			}
		},
		spotLightShadows: {
			value: [],
			properties: {
				shadowIntensity: 1,
				shadowBias: {},
				shadowNormalBias: {},
				shadowRadius: {},
				shadowMapSize: {}
			}
		},
		spotLightMap: { value: [] },
		spotLightMatrix: { value: [] },
		pointLights: {
			value: [],
			properties: {
				color: {},
				position: {},
				decay: {},
				distance: {}
			}
		},
		pointLightShadows: {
			value: [],
			properties: {
				shadowIntensity: 1,
				shadowBias: {},
				shadowNormalBias: {},
				shadowRadius: {},
				shadowMapSize: {},
				shadowCameraNear: {},
				shadowCameraFar: {}
			}
		},
		pointShadowMatrix: { value: [] },
		hemisphereLights: {
			value: [],
			properties: {
				direction: {},
				skyColor: {},
				groundColor: {}
			}
		},
		rectAreaLights: {
			value: [],
			properties: {
				color: {},
				position: {},
				width: {},
				height: {}
			}
		},
		ltc_1: { value: null },
		ltc_2: { value: null },
		probesSH: { value: null },
		probesMin: { value: /*@__PURE__*/ new J() },
		probesMax: { value: /*@__PURE__*/ new J() },
		probesResolution: { value: /*@__PURE__*/ new J() }
	},
	points: {
		diffuse: { value: /*@__PURE__*/ new Z(16777215) },
		opacity: { value: 1 },
		size: { value: 1 },
		scale: { value: 1 },
		map: { value: null },
		alphaMap: { value: null },
		alphaMapTransform: { value: /*@__PURE__*/ new Y() },
		alphaTest: { value: 0 },
		uvTransform: { value: /*@__PURE__*/ new Y() }
	},
	sprite: {
		diffuse: { value: /*@__PURE__*/ new Z(16777215) },
		opacity: { value: 1 },
		center: { value: /*@__PURE__*/ new Tf(.5, .5) },
		rotation: { value: 0 },
		map: { value: null },
		mapTransform: { value: /*@__PURE__*/ new Y() },
		alphaMap: { value: null },
		alphaMapTransform: { value: /*@__PURE__*/ new Y() },
		alphaTest: { value: 0 }
	}
}, zg = {
	basic: {
		uniforms: /*@__PURE__*/ Nh([
			$.common,
			$.specularmap,
			$.envmap,
			$.aomap,
			$.lightmap,
			$.fog
		]),
		vertexShader: Q.meshbasic_vert,
		fragmentShader: Q.meshbasic_frag
	},
	lambert: {
		uniforms: /*@__PURE__*/ Nh([
			$.common,
			$.specularmap,
			$.envmap,
			$.aomap,
			$.lightmap,
			$.emissivemap,
			$.bumpmap,
			$.normalmap,
			$.displacementmap,
			$.fog,
			$.lights,
			{
				emissive: { value: /*@__PURE__*/ new Z(0) },
				envMapIntensity: { value: 1 }
			}
		]),
		vertexShader: Q.meshlambert_vert,
		fragmentShader: Q.meshlambert_frag
	},
	phong: {
		uniforms: /*@__PURE__*/ Nh([
			$.common,
			$.specularmap,
			$.envmap,
			$.aomap,
			$.lightmap,
			$.emissivemap,
			$.bumpmap,
			$.normalmap,
			$.displacementmap,
			$.fog,
			$.lights,
			{
				emissive: { value: /*@__PURE__*/ new Z(0) },
				specular: { value: /*@__PURE__*/ new Z(1118481) },
				shininess: { value: 30 },
				envMapIntensity: { value: 1 }
			}
		]),
		vertexShader: Q.meshphong_vert,
		fragmentShader: Q.meshphong_frag
	},
	standard: {
		uniforms: /*@__PURE__*/ Nh([
			$.common,
			$.envmap,
			$.aomap,
			$.lightmap,
			$.emissivemap,
			$.bumpmap,
			$.normalmap,
			$.displacementmap,
			$.roughnessmap,
			$.metalnessmap,
			$.fog,
			$.lights,
			{
				emissive: { value: /*@__PURE__*/ new Z(0) },
				roughness: { value: 1 },
				metalness: { value: 0 },
				envMapIntensity: { value: 1 }
			}
		]),
		vertexShader: Q.meshphysical_vert,
		fragmentShader: Q.meshphysical_frag
	},
	toon: {
		uniforms: /*@__PURE__*/ Nh([
			$.common,
			$.aomap,
			$.lightmap,
			$.emissivemap,
			$.bumpmap,
			$.normalmap,
			$.displacementmap,
			$.gradientmap,
			$.fog,
			$.lights,
			{ emissive: { value: /*@__PURE__*/ new Z(0) } }
		]),
		vertexShader: Q.meshtoon_vert,
		fragmentShader: Q.meshtoon_frag
	},
	matcap: {
		uniforms: /*@__PURE__*/ Nh([
			$.common,
			$.bumpmap,
			$.normalmap,
			$.displacementmap,
			$.fog,
			{ matcap: { value: null } }
		]),
		vertexShader: Q.meshmatcap_vert,
		fragmentShader: Q.meshmatcap_frag
	},
	points: {
		uniforms: /*@__PURE__*/ Nh([$.points, $.fog]),
		vertexShader: Q.points_vert,
		fragmentShader: Q.points_frag
	},
	dashed: {
		uniforms: /*@__PURE__*/ Nh([
			$.common,
			$.fog,
			{
				scale: { value: 1 },
				dashSize: { value: 1 },
				totalSize: { value: 2 }
			}
		]),
		vertexShader: Q.linedashed_vert,
		fragmentShader: Q.linedashed_frag
	},
	depth: {
		uniforms: /*@__PURE__*/ Nh([$.common, $.displacementmap]),
		vertexShader: Q.depth_vert,
		fragmentShader: Q.depth_frag
	},
	normal: {
		uniforms: /*@__PURE__*/ Nh([
			$.common,
			$.bumpmap,
			$.normalmap,
			$.displacementmap,
			{ opacity: { value: 1 } }
		]),
		vertexShader: Q.meshnormal_vert,
		fragmentShader: Q.meshnormal_frag
	},
	sprite: {
		uniforms: /*@__PURE__*/ Nh([$.sprite, $.fog]),
		vertexShader: Q.sprite_vert,
		fragmentShader: Q.sprite_frag
	},
	background: {
		uniforms: {
			uvTransform: { value: /*@__PURE__*/ new Y() },
			t2D: { value: null },
			backgroundIntensity: { value: 1 }
		},
		vertexShader: Q.background_vert,
		fragmentShader: Q.background_frag
	},
	backgroundCube: {
		uniforms: {
			envMap: { value: null },
			backgroundBlurriness: { value: 0 },
			backgroundIntensity: { value: 1 },
			backgroundRotation: { value: /*@__PURE__*/ new Y() }
		},
		vertexShader: Q.backgroundCube_vert,
		fragmentShader: Q.backgroundCube_frag
	},
	cube: {
		uniforms: {
			tCube: { value: null },
			tFlip: { value: -1 },
			opacity: { value: 1 }
		},
		vertexShader: Q.cube_vert,
		fragmentShader: Q.cube_frag
	},
	equirect: {
		uniforms: { tEquirect: { value: null } },
		vertexShader: Q.equirect_vert,
		fragmentShader: Q.equirect_frag
	},
	distance: {
		uniforms: /*@__PURE__*/ Nh([
			$.common,
			$.displacementmap,
			{
				referencePosition: { value: /*@__PURE__*/ new J() },
				nearDistance: { value: 1 },
				farDistance: { value: 1e3 }
			}
		]),
		vertexShader: Q.distance_vert,
		fragmentShader: Q.distance_frag
	},
	shadow: {
		uniforms: /*@__PURE__*/ Nh([
			$.lights,
			$.fog,
			{
				color: { value: /*@__PURE__*/ new Z(0) },
				opacity: { value: 1 }
			}
		]),
		vertexShader: Q.shadow_vert,
		fragmentShader: Q.shadow_frag
	}
};
zg.physical = {
	uniforms: /*@__PURE__*/ Nh([zg.standard.uniforms, {
		clearcoat: { value: 0 },
		clearcoatMap: { value: null },
		clearcoatMapTransform: { value: /*@__PURE__*/ new Y() },
		clearcoatNormalMap: { value: null },
		clearcoatNormalMapTransform: { value: /*@__PURE__*/ new Y() },
		clearcoatNormalScale: { value: /*@__PURE__*/ new Tf(1, 1) },
		clearcoatRoughness: { value: 0 },
		clearcoatRoughnessMap: { value: null },
		clearcoatRoughnessMapTransform: { value: /*@__PURE__*/ new Y() },
		dispersion: { value: 0 },
		iridescence: { value: 0 },
		iridescenceMap: { value: null },
		iridescenceMapTransform: { value: /*@__PURE__*/ new Y() },
		iridescenceIOR: { value: 1.3 },
		iridescenceThicknessMinimum: { value: 100 },
		iridescenceThicknessMaximum: { value: 400 },
		iridescenceThicknessMap: { value: null },
		iridescenceThicknessMapTransform: { value: /*@__PURE__*/ new Y() },
		sheen: { value: 0 },
		sheenColor: { value: /*@__PURE__*/ new Z(0) },
		sheenColorMap: { value: null },
		sheenColorMapTransform: { value: /*@__PURE__*/ new Y() },
		sheenRoughness: { value: 1 },
		sheenRoughnessMap: { value: null },
		sheenRoughnessMapTransform: { value: /*@__PURE__*/ new Y() },
		transmission: { value: 0 },
		transmissionMap: { value: null },
		transmissionMapTransform: { value: /*@__PURE__*/ new Y() },
		transmissionSamplerSize: { value: /*@__PURE__*/ new Tf() },
		transmissionSamplerMap: { value: null },
		thickness: { value: 0 },
		thicknessMap: { value: null },
		thicknessMapTransform: { value: /*@__PURE__*/ new Y() },
		attenuationDistance: { value: 0 },
		attenuationColor: { value: /*@__PURE__*/ new Z(0) },
		specularColor: { value: /*@__PURE__*/ new Z(1, 1, 1) },
		specularColorMap: { value: null },
		specularColorMapTransform: { value: /*@__PURE__*/ new Y() },
		specularIntensity: { value: 1 },
		specularIntensityMap: { value: null },
		specularIntensityMapTransform: { value: /*@__PURE__*/ new Y() },
		anisotropyVector: { value: /*@__PURE__*/ new Tf() },
		anisotropyMap: { value: null },
		anisotropyMapTransform: { value: /*@__PURE__*/ new Y() }
	}]),
	vertexShader: Q.meshphysical_vert,
	fragmentShader: Q.meshphysical_frag
};
var Bg = {
	r: 0,
	b: 0,
	g: 0
}, Vg = /*@__PURE__*/ new Jf(), Hg = /*@__PURE__*/ new Y();
Hg.set(-1, 0, 0, 0, 1, 0, 0, 0, 1);
function Ug(e, t, n, r, i, a) {
	let o = new Z(0), s = i === !0 ? 0 : 1, c, l, u = null, d = 0, f = null;
	function p(e) {
		let n = e.isScene === !0 ? e.background : null;
		if (n && n.isTexture) {
			let r = e.backgroundBlurriness > 0;
			n = t.get(n, r);
		}
		return n;
	}
	function m(t) {
		let r = !1, i = p(t);
		i === null ? g(o, s) : i && i.isColor && (g(i, 1), r = !0);
		let c = e.xr.getEnvironmentBlendMode();
		c === "additive" ? n.buffers.color.setClear(0, 0, 0, 1, a) : c === "alpha-blend" && n.buffers.color.setClear(0, 0, 0, 0, a), (e.autoClear || r) && (n.buffers.depth.setTest(!0), n.buffers.depth.setMask(!0), n.buffers.color.setMask(!0), e.clear(e.autoClearColor, e.autoClearDepth, e.autoClearStencil));
	}
	function h(t, n) {
		let i = p(n);
		i && (i.isCubeTexture || i.mapping === 306) ? (l === void 0 && (l = new qm(new Ah(1, 1, 1), new Bh({
			name: "BackgroundCubeMaterial",
			uniforms: Mh(zg.backgroundCube.uniforms),
			vertexShader: zg.backgroundCube.vertexShader,
			fragmentShader: zg.backgroundCube.fragmentShader,
			side: 1,
			depthTest: !1,
			depthWrite: !1,
			fog: !1,
			allowOverride: !1
		})), l.geometry.deleteAttribute("normal"), l.geometry.deleteAttribute("uv"), l.onBeforeRender = function(e, t, n) {
			this.matrixWorld.copyPosition(n.matrixWorld);
		}, Object.defineProperty(l.material, "envMap", { get: function() {
			return this.uniforms.envMap.value;
		} }), r.update(l)), l.material.uniforms.envMap.value = i, l.material.uniforms.backgroundBlurriness.value = n.backgroundBlurriness, l.material.uniforms.backgroundIntensity.value = n.backgroundIntensity, l.material.uniforms.backgroundRotation.value.setFromMatrix4(Vg.makeRotationFromEuler(n.backgroundRotation)).transpose(), i.isCubeTexture && i.isRenderTargetTexture === !1 && l.material.uniforms.backgroundRotation.value.premultiply(Hg), l.material.toneMapped = X.getTransfer(i.colorSpace) !== Id, (u !== i || d !== i.version || f !== e.toneMapping) && (l.material.needsUpdate = !0, u = i, d = i.version, f = e.toneMapping), l.layers.enableAll(), t.unshift(l, l.geometry, l.material, 0, 0, null)) : i && i.isTexture && (c === void 0 && (c = new qm(new jh(2, 2), new Bh({
			name: "BackgroundMaterial",
			uniforms: Mh(zg.background.uniforms),
			vertexShader: zg.background.vertexShader,
			fragmentShader: zg.background.fragmentShader,
			side: 0,
			depthTest: !1,
			depthWrite: !1,
			fog: !1,
			allowOverride: !1
		})), c.geometry.deleteAttribute("normal"), Object.defineProperty(c.material, "map", { get: function() {
			return this.uniforms.t2D.value;
		} }), r.update(c)), c.material.uniforms.t2D.value = i, c.material.uniforms.backgroundIntensity.value = n.backgroundIntensity, c.material.toneMapped = X.getTransfer(i.colorSpace) !== Id, i.matrixAutoUpdate === !0 && i.updateMatrix(), c.material.uniforms.uvTransform.value.copy(i.matrix), (u !== i || d !== i.version || f !== e.toneMapping) && (c.material.needsUpdate = !0, u = i, d = i.version, f = e.toneMapping), c.layers.enableAll(), t.unshift(c, c.geometry, c.material, 0, 0, null));
	}
	function g(t, r) {
		t.getRGB(Bg, Ih(e)), n.buffers.color.setClear(Bg.r, Bg.g, Bg.b, r, a);
	}
	function _() {
		l !== void 0 && (l.geometry.dispose(), l.material.dispose(), l = void 0), c !== void 0 && (c.geometry.dispose(), c.material.dispose(), c = void 0);
	}
	return {
		getClearColor: function() {
			return o;
		},
		setClearColor: function(e, t = 1) {
			o.set(e), s = t, g(o, s);
		},
		getClearAlpha: function() {
			return s;
		},
		setClearAlpha: function(e) {
			s = e, g(o, s);
		},
		render: m,
		addToRenderList: h,
		dispose: _
	};
}
function Wg(e, t) {
	let n = e.getParameter(e.MAX_VERTEX_ATTRIBS), r = {}, i = f(null), a = i, o = !1;
	function s(n, r, i, s, c) {
		let u = !1, f = d(n, s, i, r);
		a !== f && (a = f, l(a.object)), u = p(n, s, i, c), u && m(n, s, i, c), c !== null && t.update(c, e.ELEMENT_ARRAY_BUFFER), (u || o) && (o = !1, b(n, r, i, s), c !== null && e.bindBuffer(e.ELEMENT_ARRAY_BUFFER, t.get(c).buffer));
	}
	function c() {
		return e.createVertexArray();
	}
	function l(t) {
		return e.bindVertexArray(t);
	}
	function u(t) {
		return e.deleteVertexArray(t);
	}
	function d(e, t, n, i) {
		let a = i.wireframe === !0, o = r[t.id];
		o === void 0 && (o = {}, r[t.id] = o);
		let s = e.isInstancedMesh === !0 ? e.id : 0, l = o[s];
		l === void 0 && (l = {}, o[s] = l);
		let u = l[n.id];
		u === void 0 && (u = {}, l[n.id] = u);
		let d = u[a];
		return d === void 0 && (d = f(c()), u[a] = d), d;
	}
	function f(e) {
		let t = [], r = [], i = [];
		for (let e = 0; e < n; e++) t[e] = 0, r[e] = 0, i[e] = 0;
		return {
			geometry: null,
			program: null,
			wireframe: !1,
			newAttributes: t,
			enabledAttributes: r,
			attributeDivisors: i,
			object: e,
			attributes: {},
			index: null
		};
	}
	function p(e, t, n, r) {
		let i = a.attributes, o = t.attributes, s = 0, c = n.getAttributes();
		for (let t in c) if (c[t].location >= 0) {
			let n = i[t], r = o[t];
			if (r === void 0 && (t === "instanceMatrix" && e.instanceMatrix && (r = e.instanceMatrix), t === "instanceColor" && e.instanceColor && (r = e.instanceColor)), n === void 0 || n.attribute !== r || r && n.data !== r.data) return !0;
			s++;
		}
		return a.attributesNum !== s || a.index !== r;
	}
	function m(e, t, n, r) {
		let i = {}, o = t.attributes, s = 0, c = n.getAttributes();
		for (let t in c) if (c[t].location >= 0) {
			let n = o[t];
			n === void 0 && (t === "instanceMatrix" && e.instanceMatrix && (n = e.instanceMatrix), t === "instanceColor" && e.instanceColor && (n = e.instanceColor));
			let r = {};
			r.attribute = n, n && n.data && (r.data = n.data), i[t] = r, s++;
		}
		a.attributes = i, a.attributesNum = s, a.index = r;
	}
	function h() {
		let e = a.newAttributes;
		for (let t = 0, n = e.length; t < n; t++) e[t] = 0;
	}
	function g(e) {
		_(e, 0);
	}
	function _(t, n) {
		let r = a.newAttributes, i = a.enabledAttributes, o = a.attributeDivisors;
		r[t] = 1, i[t] === 0 && (e.enableVertexAttribArray(t), i[t] = 1), o[t] !== n && (e.vertexAttribDivisor(t, n), o[t] = n);
	}
	function v() {
		let t = a.newAttributes, n = a.enabledAttributes;
		for (let r = 0, i = n.length; r < i; r++) n[r] !== t[r] && (e.disableVertexAttribArray(r), n[r] = 0);
	}
	function y(t, n, r, i, a, o, s) {
		s === !0 ? e.vertexAttribIPointer(t, n, r, a, o) : e.vertexAttribPointer(t, n, r, i, a, o);
	}
	function b(n, r, i, a) {
		h();
		let o = a.attributes, s = i.getAttributes(), c = r.defaultAttributeValues;
		for (let r in s) {
			let i = s[r];
			if (i.location >= 0) {
				let s = o[r];
				if (s === void 0 && (r === "instanceMatrix" && n.instanceMatrix && (s = n.instanceMatrix), r === "instanceColor" && n.instanceColor && (s = n.instanceColor)), s !== void 0) {
					let r = s.normalized, o = s.itemSize, c = t.get(s);
					if (c === void 0) continue;
					let l = c.buffer, u = c.type, d = c.bytesPerElement, f = u === e.INT || u === e.UNSIGNED_INT || s.gpuType === 1013;
					if (s.isInterleavedBufferAttribute) {
						let t = s.data, c = t.stride, p = s.offset;
						if (t.isInstancedInterleavedBuffer) {
							for (let e = 0; e < i.locationSize; e++) _(i.location + e, t.meshPerAttribute);
							n.isInstancedMesh !== !0 && a._maxInstanceCount === void 0 && (a._maxInstanceCount = t.meshPerAttribute * t.count);
						} else for (let e = 0; e < i.locationSize; e++) g(i.location + e);
						e.bindBuffer(e.ARRAY_BUFFER, l);
						for (let e = 0; e < i.locationSize; e++) y(i.location + e, o / i.locationSize, u, r, c * d, (p + o / i.locationSize * e) * d, f);
					} else {
						if (s.isInstancedBufferAttribute) {
							for (let e = 0; e < i.locationSize; e++) _(i.location + e, s.meshPerAttribute);
							n.isInstancedMesh !== !0 && a._maxInstanceCount === void 0 && (a._maxInstanceCount = s.meshPerAttribute * s.count);
						} else for (let e = 0; e < i.locationSize; e++) g(i.location + e);
						e.bindBuffer(e.ARRAY_BUFFER, l);
						for (let e = 0; e < i.locationSize; e++) y(i.location + e, o / i.locationSize, u, r, o * d, o / i.locationSize * e * d, f);
					}
				} else if (c !== void 0) {
					let t = c[r];
					if (t !== void 0) switch (t.length) {
						case 2:
							e.vertexAttrib2fv(i.location, t);
							break;
						case 3:
							e.vertexAttrib3fv(i.location, t);
							break;
						case 4:
							e.vertexAttrib4fv(i.location, t);
							break;
						default: e.vertexAttrib1fv(i.location, t);
					}
				}
			}
		}
		v();
	}
	function x() {
		T();
		for (let e in r) {
			let t = r[e];
			for (let e in t) {
				let n = t[e];
				for (let e in n) {
					let t = n[e];
					for (let e in t) u(t[e].object), delete t[e];
					delete n[e];
				}
			}
			delete r[e];
		}
	}
	function S(e) {
		if (r[e.id] === void 0) return;
		let t = r[e.id];
		for (let e in t) {
			let n = t[e];
			for (let e in n) {
				let t = n[e];
				for (let e in t) u(t[e].object), delete t[e];
				delete n[e];
			}
		}
		delete r[e.id];
	}
	function C(e) {
		for (let t in r) {
			let n = r[t];
			for (let t in n) {
				let r = n[t];
				if (r[e.id] === void 0) continue;
				let i = r[e.id];
				for (let e in i) u(i[e].object), delete i[e];
				delete r[e.id];
			}
		}
	}
	function w(e) {
		for (let t in r) {
			let n = r[t], i = e.isInstancedMesh === !0 ? e.id : 0, a = n[i];
			if (a !== void 0) {
				for (let e in a) {
					let t = a[e];
					for (let e in t) u(t[e].object), delete t[e];
					delete a[e];
				}
				delete n[i], Object.keys(n).length === 0 && delete r[t];
			}
		}
	}
	function T() {
		E(), o = !0, a !== i && (a = i, l(a.object));
	}
	function E() {
		i.geometry = null, i.program = null, i.wireframe = !1;
	}
	return {
		setup: s,
		reset: T,
		resetDefaultState: E,
		dispose: x,
		releaseStatesOfGeometry: S,
		releaseStatesOfObject: w,
		releaseStatesOfProgram: C,
		initAttributes: h,
		enableAttribute: g,
		disableUnusedAttributes: v
	};
}
function Gg(e, t, n) {
	let r;
	function i(e) {
		r = e;
	}
	function a(t, i) {
		e.drawArrays(r, t, i), n.update(i, r, 1);
	}
	function o(t, i, a) {
		a !== 0 && (e.drawArraysInstanced(r, t, i, a), n.update(i, r, a));
	}
	function s(e, i, a) {
		if (a === 0) return;
		t.get("WEBGL_multi_draw").multiDrawArraysWEBGL(r, e, 0, i, 0, a);
		let o = 0;
		for (let e = 0; e < a; e++) o += i[e];
		n.update(o, r, 1);
	}
	this.setMode = i, this.render = a, this.renderInstances = o, this.renderMultiDraw = s;
}
function Kg(e, t, n, r) {
	let i;
	function a() {
		if (i !== void 0) return i;
		if (t.has("EXT_texture_filter_anisotropic") === !0) {
			let n = t.get("EXT_texture_filter_anisotropic");
			i = e.getParameter(n.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
		} else i = 0;
		return i;
	}
	function o(t) {
		return !(t !== 1023 && r.convert(t) !== e.getParameter(e.IMPLEMENTATION_COLOR_READ_FORMAT));
	}
	function s(n) {
		let i = n === 1016 && (t.has("EXT_color_buffer_half_float") || t.has("EXT_color_buffer_float"));
		return !(n !== 1009 && r.convert(n) !== e.getParameter(e.IMPLEMENTATION_COLOR_READ_TYPE) && n !== 1015 && !i);
	}
	function c(t) {
		if (t === "highp") {
			if (e.getShaderPrecisionFormat(e.VERTEX_SHADER, e.HIGH_FLOAT).precision > 0 && e.getShaderPrecisionFormat(e.FRAGMENT_SHADER, e.HIGH_FLOAT).precision > 0) return "highp";
			t = "mediump";
		}
		return t === "mediump" && e.getShaderPrecisionFormat(e.VERTEX_SHADER, e.MEDIUM_FLOAT).precision > 0 && e.getShaderPrecisionFormat(e.FRAGMENT_SHADER, e.MEDIUM_FLOAT).precision > 0 ? "mediump" : "lowp";
	}
	let l = n.precision === void 0 ? "highp" : n.precision, u = c(l);
	u !== l && (G("WebGLRenderer:", l, "not supported, using", u, "instead."), l = u);
	let d = n.logarithmicDepthBuffer === !0, f = n.reversedDepthBuffer === !0 && t.has("EXT_clip_control");
	n.reversedDepthBuffer === !0 && f === !1 && G("WebGLRenderer: Unable to use reversed depth buffer due to missing EXT_clip_control extension. Fallback to default depth buffer.");
	let p = e.getParameter(e.MAX_TEXTURE_IMAGE_UNITS), m = e.getParameter(e.MAX_VERTEX_TEXTURE_IMAGE_UNITS), h = e.getParameter(e.MAX_TEXTURE_SIZE), g = e.getParameter(e.MAX_CUBE_MAP_TEXTURE_SIZE), _ = e.getParameter(e.MAX_VERTEX_ATTRIBS), v = e.getParameter(e.MAX_VERTEX_UNIFORM_VECTORS), y = e.getParameter(e.MAX_VARYING_VECTORS), b = e.getParameter(e.MAX_FRAGMENT_UNIFORM_VECTORS), x = e.getParameter(e.MAX_SAMPLES), S = e.getParameter(e.SAMPLES);
	return {
		isWebGL2: !0,
		getMaxAnisotropy: a,
		getMaxPrecision: c,
		textureFormatReadable: o,
		textureTypeReadable: s,
		precision: l,
		logarithmicDepthBuffer: d,
		reversedDepthBuffer: f,
		maxTextures: p,
		maxVertexTextures: m,
		maxTextureSize: h,
		maxCubemapSize: g,
		maxAttributes: _,
		maxVertexUniforms: v,
		maxVaryings: y,
		maxFragmentUniforms: b,
		maxSamples: x,
		samples: S
	};
}
function qg(e) {
	let t = this, n = null, r = 0, i = !1, a = !1, o = new eh(), s = new Y(), c = {
		value: null,
		needsUpdate: !1
	};
	this.uniform = c, this.numPlanes = 0, this.numIntersection = 0, this.init = function(e, t) {
		let n = e.length !== 0 || t || r !== 0 || i;
		return i = t, r = e.length, n;
	}, this.beginShadows = function() {
		a = !0, u(null);
	}, this.endShadows = function() {
		a = !1;
	}, this.setGlobalState = function(e, t) {
		n = u(e, t, 0);
	}, this.setState = function(t, o, s) {
		let d = t.clippingPlanes, f = t.clipIntersection, p = t.clipShadows, m = e.get(t);
		if (!i || d === null || d.length === 0 || a && !p) a ? u(null) : l();
		else {
			let e = a ? 0 : r, t = e * 4, i = m.clippingState || null;
			c.value = i, i = u(d, o, t, s);
			for (let e = 0; e !== t; ++e) i[e] = n[e];
			m.clippingState = i, this.numIntersection = f ? this.numPlanes : 0, this.numPlanes += e;
		}
	};
	function l() {
		c.value !== n && (c.value = n, c.needsUpdate = r > 0), t.numPlanes = r, t.numIntersection = 0;
	}
	function u(e, n, r, i) {
		let a = e === null ? 0 : e.length, l = null;
		if (a !== 0) {
			if (l = c.value, i !== !0 || l === null) {
				let t = r + a * 4, i = n.matrixWorldInverse;
				s.getNormalMatrix(i), (l === null || l.length < t) && (l = new Float32Array(t));
				for (let t = 0, n = r; t !== a; ++t, n += 4) o.copy(e[t]).applyMatrix4(i, s), o.normal.toArray(l, n), l[n + 3] = o.constant;
			}
			c.value = l, c.needsUpdate = !0;
		}
		return t.numPlanes = a, t.numIntersection = 0, l;
	}
}
var Jg = 4, Yg = [
	.125,
	.215,
	.35,
	.446,
	.526,
	.582
], Xg = 20, Zg = 256, Qg = /*@__PURE__*/ new mg(), $g = /*@__PURE__*/ new Z(), e_ = null, t_ = 0, n_ = 0, r_ = !1, i_ = /*@__PURE__*/ new J(), a_ = class {
	constructor(e) {
		this._renderer = e, this._pingPongRenderTarget = null, this._lodMax = 0, this._cubeSize = 0, this._sizeLods = [], this._sigmas = [], this._lodMeshes = [], this._backgroundBox = null, this._cubemapMaterial = null, this._equirectMaterial = null, this._blurMaterial = null, this._ggxMaterial = null;
	}
	fromScene(e, t = 0, n = .1, r = 100, i = {}) {
		let { size: a = 256, position: o = i_ } = i;
		e_ = this._renderer.getRenderTarget(), t_ = this._renderer.getActiveCubeFace(), n_ = this._renderer.getActiveMipmapLevel(), r_ = this._renderer.xr.enabled, this._renderer.xr.enabled = !1, this._setSize(a);
		let s = this._allocateTargets();
		return s.depthBuffer = !0, this._sceneToCubeUV(e, n, r, s, o), t > 0 && this._blur(s, 0, 0, t), this._applyPMREM(s), this._cleanup(s), s;
	}
	fromEquirectangular(e, t = null) {
		return this._fromTexture(e, t);
	}
	fromCubemap(e, t = null) {
		return this._fromTexture(e, t);
	}
	compileCubemapShader() {
		this._cubemapMaterial === null && (this._cubemapMaterial = f_(), this._compileMaterial(this._cubemapMaterial));
	}
	compileEquirectangularShader() {
		this._equirectMaterial === null && (this._equirectMaterial = d_(), this._compileMaterial(this._equirectMaterial));
	}
	dispose() {
		this._dispose(), this._cubemapMaterial !== null && this._cubemapMaterial.dispose(), this._equirectMaterial !== null && this._equirectMaterial.dispose(), this._backgroundBox !== null && (this._backgroundBox.geometry.dispose(), this._backgroundBox.material.dispose());
	}
	_setSize(e) {
		this._lodMax = Math.floor(Math.log2(e)), this._cubeSize = 2 ** this._lodMax;
	}
	_dispose() {
		this._blurMaterial !== null && this._blurMaterial.dispose(), this._ggxMaterial !== null && this._ggxMaterial.dispose(), this._pingPongRenderTarget !== null && this._pingPongRenderTarget.dispose();
		for (let e = 0; e < this._lodMeshes.length; e++) this._lodMeshes[e].geometry.dispose();
	}
	_cleanup(e) {
		this._renderer.setRenderTarget(e_, t_, n_), this._renderer.xr.enabled = r_, e.scissorTest = !1, c_(e, 0, 0, e.width, e.height);
	}
	_fromTexture(e, t) {
		e.mapping === 301 || e.mapping === 302 ? this._setSize(e.image.length === 0 ? 16 : e.image[0].width || e.image[0].image.width) : this._setSize(e.image.width / 4), e_ = this._renderer.getRenderTarget(), t_ = this._renderer.getActiveCubeFace(), n_ = this._renderer.getActiveMipmapLevel(), r_ = this._renderer.xr.enabled, this._renderer.xr.enabled = !1;
		let n = t || this._allocateTargets();
		return this._textureToCubeUV(e, n), this._applyPMREM(n), this._cleanup(n), n;
	}
	_allocateTargets() {
		let e = 3 * Math.max(this._cubeSize, 112), t = 4 * this._cubeSize, n = {
			magFilter: _u,
			minFilter: _u,
			generateMipmaps: !1,
			type: Du,
			format: Fu,
			colorSpace: Pd,
			depthBuffer: !1
		}, r = s_(e, t, n);
		if (this._pingPongRenderTarget === null || this._pingPongRenderTarget.width !== e || this._pingPongRenderTarget.height !== t) {
			this._pingPongRenderTarget !== null && this._dispose(), this._pingPongRenderTarget = s_(e, t, n);
			let { _lodMax: r } = this;
			({lodMeshes: this._lodMeshes, sizeLods: this._sizeLods, sigmas: this._sigmas} = o_(r)), this._blurMaterial = u_(r, e, t), this._ggxMaterial = l_(r, e, t);
		}
		return r;
	}
	_compileMaterial(e) {
		let t = new qm(new wm(), e);
		this._renderer.compile(t, Qg);
	}
	_sceneToCubeUV(e, t, n, r, i) {
		let a = new pg(90, 1, t, n), o = [
			1,
			-1,
			1,
			1,
			1,
			1
		], s = [
			1,
			1,
			1,
			-1,
			-1,
			-1
		], c = this._renderer, l = c.autoClear, u = c.toneMapping;
		c.getClearColor($g), c.toneMapping = 0, c.autoClear = !1, c.state.buffers.depth.getReversed() && (c.setRenderTarget(r), c.clearDepth(), c.setRenderTarget(null)), this._backgroundBox === null && (this._backgroundBox = new qm(new Ah(), new Fm({
			name: "PMREM.Background",
			side: 1,
			depthWrite: !1,
			depthTest: !1
		})));
		let d = this._backgroundBox, f = d.material, p = !1, m = e.background;
		m ? m.isColor && (f.color.copy(m), e.background = null, p = !0) : (f.color.copy($g), p = !0);
		for (let t = 0; t < 6; t++) {
			let n = t % 3;
			n === 0 ? (a.up.set(0, o[t], 0), a.position.set(i.x, i.y, i.z), a.lookAt(i.x + s[t], i.y, i.z)) : n === 1 ? (a.up.set(0, 0, o[t]), a.position.set(i.x, i.y, i.z), a.lookAt(i.x, i.y + s[t], i.z)) : (a.up.set(0, o[t], 0), a.position.set(i.x, i.y, i.z), a.lookAt(i.x, i.y, i.z + s[t]));
			let l = this._cubeSize;
			c_(r, n * l, t > 2 ? l : 0, l, l), c.setRenderTarget(r), p && c.render(d, a), c.render(e, a);
		}
		c.toneMapping = u, c.autoClear = l, e.background = m;
	}
	_textureToCubeUV(e, t) {
		let n = this._renderer, r = e.mapping === 301 || e.mapping === 302;
		r ? (this._cubemapMaterial === null && (this._cubemapMaterial = f_()), this._cubemapMaterial.uniforms.flipEnvMap.value = e.isRenderTargetTexture === !1 ? -1 : 1) : this._equirectMaterial === null && (this._equirectMaterial = d_());
		let i = r ? this._cubemapMaterial : this._equirectMaterial, a = this._lodMeshes[0];
		a.material = i;
		let o = i.uniforms;
		o.envMap.value = e;
		let s = this._cubeSize;
		c_(t, 0, 0, 3 * s, 2 * s), n.setRenderTarget(t), n.render(a, Qg);
	}
	_applyPMREM(e) {
		let t = this._renderer, n = t.autoClear;
		t.autoClear = !1;
		let r = this._lodMeshes.length;
		for (let t = 1; t < r; t++) this._applyGGXFilter(e, t - 1, t);
		t.autoClear = n;
	}
	_applyGGXFilter(e, t, n) {
		let r = this._renderer, i = this._pingPongRenderTarget, a = this._ggxMaterial, o = this._lodMeshes[n];
		o.material = a;
		let s = a.uniforms, c = n / (this._lodMeshes.length - 1), l = t / (this._lodMeshes.length - 1), u = Math.sqrt(c * c - l * l) * (0 + c * 1.25), { _lodMax: d } = this, f = this._sizeLods[n], p = 3 * f * (n > d - Jg ? n - d + Jg : 0), m = 4 * (this._cubeSize - f);
		s.envMap.value = e.texture, s.roughness.value = u, s.mipInt.value = d - t, c_(i, p, m, 3 * f, 2 * f), r.setRenderTarget(i), r.render(o, Qg), s.envMap.value = i.texture, s.roughness.value = 0, s.mipInt.value = d - n, c_(e, p, m, 3 * f, 2 * f), r.setRenderTarget(e), r.render(o, Qg);
	}
	_blur(e, t, n, r, i) {
		let a = this._pingPongRenderTarget;
		this._halfBlur(e, a, t, n, r, "latitudinal", i), this._halfBlur(a, e, n, n, r, "longitudinal", i);
	}
	_halfBlur(e, t, n, r, i, a, o) {
		let s = this._renderer, c = this._blurMaterial;
		a !== "latitudinal" && a !== "longitudinal" && K("blur direction must be either latitudinal or longitudinal!");
		let l = this._lodMeshes[r];
		l.material = c;
		let u = c.uniforms, d = this._sizeLods[n] - 1, f = isFinite(i) ? Math.PI / (2 * d) : 2 * Math.PI / (2 * Xg - 1), p = i / f, m = isFinite(i) ? 1 + Math.floor(3 * p) : Xg;
		m > Xg && G(`sigmaRadians, ${i}, is too large and will clip, as it requested ${m} samples when the maximum is set to ${Xg}`);
		let h = [], g = 0;
		for (let e = 0; e < Xg; ++e) {
			let t = e / p, n = Math.exp(-t * t / 2);
			h.push(n), e === 0 ? g += n : e < m && (g += 2 * n);
		}
		for (let e = 0; e < h.length; e++) h[e] = h[e] / g;
		u.envMap.value = e.texture, u.samples.value = m, u.weights.value = h, u.latitudinal.value = a === "latitudinal", o && (u.poleAxis.value = o);
		let { _lodMax: _ } = this;
		u.dTheta.value = f, u.mipInt.value = _ - n;
		let v = this._sizeLods[r];
		c_(t, 3 * v * (r > _ - Jg ? r - _ + Jg : 0), 4 * (this._cubeSize - v), 3 * v, 2 * v), s.setRenderTarget(t), s.render(l, Qg);
	}
};
function o_(e) {
	let t = [], n = [], r = [], i = e, a = e - Jg + 1 + Yg.length;
	for (let o = 0; o < a; o++) {
		let a = 2 ** i;
		t.push(a);
		let s = 1 / a;
		o > e - Jg ? s = Yg[o - e + Jg - 1] : o === 0 && (s = 0), n.push(s);
		let c = 1 / (a - 2), l = -c, u = 1 + c, d = [
			l,
			l,
			u,
			l,
			u,
			u,
			l,
			l,
			u,
			u,
			l,
			u
		], f = /* @__PURE__ */ new Float32Array(108), p = /* @__PURE__ */ new Float32Array(72), m = /* @__PURE__ */ new Float32Array(36);
		for (let e = 0; e < 6; e++) {
			let t = e % 3 * 2 / 3 - 1, n = e > 2 ? 0 : -1, r = [
				t,
				n,
				0,
				t + 2 / 3,
				n,
				0,
				t + 2 / 3,
				n + 1,
				0,
				t,
				n,
				0,
				t + 2 / 3,
				n + 1,
				0,
				t,
				n + 1,
				0
			];
			f.set(r, 18 * e), p.set(d, 12 * e);
			let i = [
				e,
				e,
				e,
				e,
				e,
				e
			];
			m.set(i, 6 * e);
		}
		let h = new wm();
		h.setAttribute("position", new lm(f, 3)), h.setAttribute("uv", new lm(p, 2)), h.setAttribute("faceIndex", new lm(m, 1)), r.push(new qm(h, null)), i > Jg && i--;
	}
	return {
		lodMeshes: r,
		sizeLods: t,
		sigmas: n
	};
}
function s_(e, t, n) {
	let r = new Gf(e, t, n);
	return r.texture.mapping = 306, r.texture.name = "PMREM.cubeUv", r.scissorTest = !0, r;
}
function c_(e, t, n, r, i) {
	e.viewport.set(t, n, r, i), e.scissor.set(t, n, r, i);
}
function l_(e, t, n) {
	return new Bh({
		name: "PMREMGGXConvolution",
		defines: {
			GGX_SAMPLES: Zg,
			CUBEUV_TEXEL_WIDTH: 1 / t,
			CUBEUV_TEXEL_HEIGHT: 1 / n,
			CUBEUV_MAX_MIP: `${e}.0`
		},
		uniforms: {
			envMap: { value: null },
			roughness: { value: 0 },
			mipInt: { value: 0 }
		},
		vertexShader: p_(),
		fragmentShader: "\n\n			precision highp float;\n			precision highp int;\n\n			varying vec3 vOutputDirection;\n\n			uniform sampler2D envMap;\n			uniform float roughness;\n			uniform float mipInt;\n\n			#define ENVMAP_TYPE_CUBE_UV\n			#include <cube_uv_reflection_fragment>\n\n			#define PI 3.14159265359\n\n			// Van der Corput radical inverse\n			float radicalInverse_VdC(uint bits) {\n				bits = (bits << 16u) | (bits >> 16u);\n				bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);\n				bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);\n				bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);\n				bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);\n				return float(bits) * 2.3283064365386963e-10; // / 0x100000000\n			}\n\n			// Hammersley sequence\n			vec2 hammersley(uint i, uint N) {\n				return vec2(float(i) / float(N), radicalInverse_VdC(i));\n			}\n\n			// GGX VNDF importance sampling (Eric Heitz 2018)\n			// \"Sampling the GGX Distribution of Visible Normals\"\n			// https://jcgt.org/published/0007/04/01/\n			vec3 importanceSampleGGX_VNDF(vec2 Xi, vec3 V, float roughness) {\n				float alpha = roughness * roughness;\n\n				// Section 4.1: Orthonormal basis\n				vec3 T1 = vec3(1.0, 0.0, 0.0);\n				vec3 T2 = cross(V, T1);\n\n				// Section 4.2: Parameterization of projected area\n				float r = sqrt(Xi.x);\n				float phi = 2.0 * PI * Xi.y;\n				float t1 = r * cos(phi);\n				float t2 = r * sin(phi);\n				float s = 0.5 * (1.0 + V.z);\n				t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;\n\n				// Section 4.3: Reprojection onto hemisphere\n				vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * V;\n\n				// Section 3.4: Transform back to ellipsoid configuration\n				return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));\n			}\n\n			void main() {\n				vec3 N = normalize(vOutputDirection);\n				vec3 V = N; // Assume view direction equals normal for pre-filtering\n\n				vec3 prefilteredColor = vec3(0.0);\n				float totalWeight = 0.0;\n\n				// For very low roughness, just sample the environment directly\n				if (roughness < 0.001) {\n					gl_FragColor = vec4(bilinearCubeUV(envMap, N, mipInt), 1.0);\n					return;\n				}\n\n				// Tangent space basis for VNDF sampling\n				vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);\n				vec3 tangent = normalize(cross(up, N));\n				vec3 bitangent = cross(N, tangent);\n\n				for(uint i = 0u; i < uint(GGX_SAMPLES); i++) {\n					vec2 Xi = hammersley(i, uint(GGX_SAMPLES));\n\n					// For PMREM, V = N, so in tangent space V is always (0, 0, 1)\n					vec3 H_tangent = importanceSampleGGX_VNDF(Xi, vec3(0.0, 0.0, 1.0), roughness);\n\n					// Transform H back to world space\n					vec3 H = normalize(tangent * H_tangent.x + bitangent * H_tangent.y + N * H_tangent.z);\n					vec3 L = normalize(2.0 * dot(V, H) * H - V);\n\n					float NdotL = max(dot(N, L), 0.0);\n\n					if(NdotL > 0.0) {\n						// Sample environment at fixed mip level\n						// VNDF importance sampling handles the distribution filtering\n						vec3 sampleColor = bilinearCubeUV(envMap, L, mipInt);\n\n						// Weight by NdotL for the split-sum approximation\n						// VNDF PDF naturally accounts for the visible microfacet distribution\n						prefilteredColor += sampleColor * NdotL;\n						totalWeight += NdotL;\n					}\n				}\n\n				if (totalWeight > 0.0) {\n					prefilteredColor = prefilteredColor / totalWeight;\n				}\n\n				gl_FragColor = vec4(prefilteredColor, 1.0);\n			}\n		",
		blending: 0,
		depthTest: !1,
		depthWrite: !1
	});
}
function u_(e, t, n) {
	let r = new Float32Array(Xg), i = new J(0, 1, 0);
	return new Bh({
		name: "SphericalGaussianBlur",
		defines: {
			n: Xg,
			CUBEUV_TEXEL_WIDTH: 1 / t,
			CUBEUV_TEXEL_HEIGHT: 1 / n,
			CUBEUV_MAX_MIP: `${e}.0`
		},
		uniforms: {
			envMap: { value: null },
			samples: { value: 1 },
			weights: { value: r },
			latitudinal: { value: !1 },
			dTheta: { value: 0 },
			mipInt: { value: 0 },
			poleAxis: { value: i }
		},
		vertexShader: p_(),
		fragmentShader: "\n\n			precision mediump float;\n			precision mediump int;\n\n			varying vec3 vOutputDirection;\n\n			uniform sampler2D envMap;\n			uniform int samples;\n			uniform float weights[ n ];\n			uniform bool latitudinal;\n			uniform float dTheta;\n			uniform float mipInt;\n			uniform vec3 poleAxis;\n\n			#define ENVMAP_TYPE_CUBE_UV\n			#include <cube_uv_reflection_fragment>\n\n			vec3 getSample( float theta, vec3 axis ) {\n\n				float cosTheta = cos( theta );\n				// Rodrigues' axis-angle rotation\n				vec3 sampleDirection = vOutputDirection * cosTheta\n					+ cross( axis, vOutputDirection ) * sin( theta )\n					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );\n\n				return bilinearCubeUV( envMap, sampleDirection, mipInt );\n\n			}\n\n			void main() {\n\n				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );\n\n				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {\n\n					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );\n\n				}\n\n				axis = normalize( axis );\n\n				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );\n				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );\n\n				for ( int i = 1; i < n; i++ ) {\n\n					if ( i >= samples ) {\n\n						break;\n\n					}\n\n					float theta = dTheta * float( i );\n					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );\n					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );\n\n				}\n\n			}\n		",
		blending: 0,
		depthTest: !1,
		depthWrite: !1
	});
}
function d_() {
	return new Bh({
		name: "EquirectangularToCubeUV",
		uniforms: { envMap: { value: null } },
		vertexShader: p_(),
		fragmentShader: "\n\n			precision mediump float;\n			precision mediump int;\n\n			varying vec3 vOutputDirection;\n\n			uniform sampler2D envMap;\n\n			#include <common>\n\n			void main() {\n\n				vec3 outputDirection = normalize( vOutputDirection );\n				vec2 uv = equirectUv( outputDirection );\n\n				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );\n\n			}\n		",
		blending: 0,
		depthTest: !1,
		depthWrite: !1
	});
}
function f_() {
	return new Bh({
		name: "CubemapToCubeUV",
		uniforms: {
			envMap: { value: null },
			flipEnvMap: { value: -1 }
		},
		vertexShader: p_(),
		fragmentShader: "\n\n			precision mediump float;\n			precision mediump int;\n\n			uniform float flipEnvMap;\n\n			varying vec3 vOutputDirection;\n\n			uniform samplerCube envMap;\n\n			void main() {\n\n				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );\n\n			}\n		",
		blending: 0,
		depthTest: !1,
		depthWrite: !1
	});
}
function p_() {
	return "\n\n		precision mediump float;\n		precision mediump int;\n\n		attribute float faceIndex;\n\n		varying vec3 vOutputDirection;\n\n		// RH coordinate system; PMREM face-indexing convention\n		vec3 getDirection( vec2 uv, float face ) {\n\n			uv = 2.0 * uv - 1.0;\n\n			vec3 direction = vec3( uv, 1.0 );\n\n			if ( face == 0.0 ) {\n\n				direction = direction.zyx; // ( 1, v, u ) pos x\n\n			} else if ( face == 1.0 ) {\n\n				direction = direction.xzy;\n				direction.xz *= -1.0; // ( -u, 1, -v ) pos y\n\n			} else if ( face == 2.0 ) {\n\n				direction.x *= -1.0; // ( -u, v, 1 ) pos z\n\n			} else if ( face == 3.0 ) {\n\n				direction = direction.zyx;\n				direction.xz *= -1.0; // ( -1, v, -u ) neg x\n\n			} else if ( face == 4.0 ) {\n\n				direction = direction.xzy;\n				direction.xy *= -1.0; // ( -u, -1, v ) neg y\n\n			} else if ( face == 5.0 ) {\n\n				direction.z *= -1.0; // ( u, v, -1 ) neg z\n\n			}\n\n			return direction;\n\n		}\n\n		void main() {\n\n			vOutputDirection = getDirection( uv, faceIndex );\n			gl_Position = vec4( position, 1.0 );\n\n		}\n	";
}
var m_ = class extends Gf {
	constructor(e = 1, t = {}) {
		super(e, e, t), this.isWebGLCubeRenderTarget = !0;
		let n = {
			width: e,
			height: e,
			depth: 1
		}, r = [
			n,
			n,
			n,
			n,
			n,
			n
		];
		this.texture = new Th(r), this._setTextureOptions(t), this.texture.isRenderTargetTexture = !0;
	}
	fromEquirectangularTexture(e, t) {
		this.texture.type = t.type, this.texture.colorSpace = t.colorSpace, this.texture.generateMipmaps = t.generateMipmaps, this.texture.minFilter = t.minFilter, this.texture.magFilter = t.magFilter;
		let n = {
			uniforms: { tEquirect: { value: null } },
			vertexShader: "\n\n				varying vec3 vWorldDirection;\n\n				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {\n\n					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );\n\n				}\n\n				void main() {\n\n					vWorldDirection = transformDirection( position, modelMatrix );\n\n					#include <begin_vertex>\n					#include <project_vertex>\n\n				}\n			",
			fragmentShader: "\n\n				uniform sampler2D tEquirect;\n\n				varying vec3 vWorldDirection;\n\n				#include <common>\n\n				void main() {\n\n					vec3 direction = normalize( vWorldDirection );\n\n					vec2 sampleUV = equirectUv( direction );\n\n					gl_FragColor = texture2D( tEquirect, sampleUV );\n\n				}\n			"
		}, r = new Ah(5, 5, 5), i = new Bh({
			name: "CubemapFromEquirect",
			uniforms: Mh(n.uniforms),
			vertexShader: n.vertexShader,
			fragmentShader: n.fragmentShader,
			side: 1,
			blending: 0
		});
		i.uniforms.tEquirect.value = t;
		let a = new qm(r, i), o = t.minFilter;
		return t.minFilter === 1008 && (t.minFilter = _u), new _g(1, 10, this).update(e, a), t.minFilter = o, a.geometry.dispose(), a.material.dispose(), this;
	}
	clear(e, t = !0, n = !0, r = !0) {
		let i = e.getRenderTarget();
		for (let i = 0; i < 6; i++) e.setRenderTarget(this, i), e.clear(t, n, r);
		e.setRenderTarget(i);
	}
};
function h_(e) {
	let t = /* @__PURE__ */ new WeakMap(), n = /* @__PURE__ */ new WeakMap(), r = null;
	function i(e, t = !1) {
		return e == null ? null : t ? o(e) : a(e);
	}
	function a(n) {
		if (n && n.isTexture) {
			let r = n.mapping;
			if (r === 303 || r === 304) if (t.has(n)) {
				let e = t.get(n).texture;
				return s(e, n.mapping);
			} else {
				let r = n.image;
				if (r && r.height > 0) {
					let i = new m_(r.height);
					return i.fromEquirectangularTexture(e, n), t.set(n, i), n.addEventListener("dispose", l), s(i.texture, n.mapping);
				} else return null;
			}
		}
		return n;
	}
	function o(t) {
		if (t && t.isTexture) {
			let i = t.mapping, a = i === 303 || i === 304, o = i === 301 || i === 302;
			if (a || o) {
				let i = n.get(t), s = i === void 0 ? 0 : i.texture.pmremVersion;
				if (t.isRenderTargetTexture && t.pmremVersion !== s) return r === null && (r = new a_(e)), i = a ? r.fromEquirectangular(t, i) : r.fromCubemap(t, i), i.texture.pmremVersion = t.pmremVersion, n.set(t, i), i.texture;
				if (i !== void 0) return i.texture;
				{
					let s = t.image;
					return a && s && s.height > 0 || o && s && c(s) ? (r === null && (r = new a_(e)), i = a ? r.fromEquirectangular(t) : r.fromCubemap(t), i.texture.pmremVersion = t.pmremVersion, n.set(t, i), t.addEventListener("dispose", u), i.texture) : null;
				}
			}
		}
		return t;
	}
	function s(e, t) {
		return t === 303 ? e.mapping = 301 : t === 304 && (e.mapping = 302), e;
	}
	function c(e) {
		let t = 0;
		for (let n = 0; n < 6; n++) e[n] !== void 0 && t++;
		return t === 6;
	}
	function l(e) {
		let n = e.target;
		n.removeEventListener("dispose", l);
		let r = t.get(n);
		r !== void 0 && (t.delete(n), r.dispose());
	}
	function u(e) {
		let t = e.target;
		t.removeEventListener("dispose", u);
		let r = n.get(t);
		r !== void 0 && (n.delete(t), r.dispose());
	}
	function d() {
		t = /* @__PURE__ */ new WeakMap(), n = /* @__PURE__ */ new WeakMap(), r !== null && (r.dispose(), r = null);
	}
	return {
		get: i,
		dispose: d
	};
}
function g_(e) {
	let t = {};
	function n(n) {
		if (t[n] !== void 0) return t[n];
		let r = e.getExtension(n);
		return t[n] = r, r;
	}
	return {
		has: function(e) {
			return n(e) !== null;
		},
		init: function() {
			n("EXT_color_buffer_float"), n("WEBGL_clip_cull_distance"), n("OES_texture_float_linear"), n("EXT_color_buffer_half_float"), n("WEBGL_multisampled_render_to_texture"), n("WEBGL_render_shared_exponent");
		},
		get: function(e) {
			let t = n(e);
			return t === null && Jd("WebGLRenderer: " + e + " extension not supported."), t;
		}
	};
}
function __(e, t, n, r) {
	let i = {}, a = /* @__PURE__ */ new WeakMap();
	function o(e) {
		let s = e.target;
		s.index !== null && t.remove(s.index);
		for (let e in s.attributes) t.remove(s.attributes[e]);
		s.removeEventListener("dispose", o), delete i[s.id];
		let c = a.get(s);
		c && (t.remove(c), a.delete(s)), r.releaseStatesOfGeometry(s), s.isInstancedBufferGeometry === !0 && delete s._maxInstanceCount, n.memory.geometries--;
	}
	function s(e, t) {
		return i[t.id] === !0 ? t : (t.addEventListener("dispose", o), i[t.id] = !0, n.memory.geometries++, t);
	}
	function c(n) {
		let r = n.attributes;
		for (let n in r) t.update(r[n], e.ARRAY_BUFFER);
	}
	function l(e) {
		let n = [], r = e.index, i = e.attributes.position, o = 0;
		if (i === void 0) return;
		if (r !== null) {
			let e = r.array;
			o = r.version;
			for (let t = 0, r = e.length; t < r; t += 3) {
				let r = e[t + 0], i = e[t + 1], a = e[t + 2];
				n.push(r, i, i, a, a, r);
			}
		} else {
			let e = i.array;
			o = i.version;
			for (let t = 0, r = e.length / 3 - 1; t < r; t += 3) {
				let e = t + 0, r = t + 1, i = t + 2;
				n.push(e, r, r, i, i, e);
			}
		}
		let s = new (i.count >= 65535 ? dm : um)(n, 1);
		s.version = o;
		let c = a.get(e);
		c && t.remove(c), a.set(e, s);
	}
	function u(e) {
		let t = a.get(e);
		if (t) {
			let n = e.index;
			n !== null && t.version < n.version && l(e);
		} else l(e);
		return a.get(e);
	}
	return {
		get: s,
		update: c,
		getWireframeAttribute: u
	};
}
function v_(e, t, n) {
	let r;
	function i(e) {
		r = e;
	}
	let a, o;
	function s(e) {
		a = e.type, o = e.bytesPerElement;
	}
	function c(t, i) {
		e.drawElements(r, i, a, t * o), n.update(i, r, 1);
	}
	function l(t, i, s) {
		s !== 0 && (e.drawElementsInstanced(r, i, a, t * o, s), n.update(i, r, s));
	}
	function u(e, i, o) {
		if (o === 0) return;
		t.get("WEBGL_multi_draw").multiDrawElementsWEBGL(r, i, 0, a, e, 0, o);
		let s = 0;
		for (let e = 0; e < o; e++) s += i[e];
		n.update(s, r, 1);
	}
	this.setMode = i, this.setIndex = s, this.render = c, this.renderInstances = l, this.renderMultiDraw = u;
}
function y_(e) {
	let t = {
		geometries: 0,
		textures: 0
	}, n = {
		frame: 0,
		calls: 0,
		triangles: 0,
		points: 0,
		lines: 0
	};
	function r(t, r, i) {
		switch (n.calls++, r) {
			case e.TRIANGLES:
				n.triangles += t / 3 * i;
				break;
			case e.LINES:
				n.lines += t / 2 * i;
				break;
			case e.LINE_STRIP:
				n.lines += i * (t - 1);
				break;
			case e.LINE_LOOP:
				n.lines += i * t;
				break;
			case e.POINTS:
				n.points += i * t;
				break;
			default:
				K("WebGLInfo: Unknown draw mode:", r);
				break;
		}
	}
	function i() {
		n.calls = 0, n.triangles = 0, n.points = 0, n.lines = 0;
	}
	return {
		memory: t,
		render: n,
		programs: null,
		autoReset: !0,
		reset: i,
		update: r
	};
}
function b_(e, t, n) {
	let r = /* @__PURE__ */ new WeakMap(), i = new Uf();
	function a(a, o, s) {
		let c = a.morphTargetInfluences, l = o.morphAttributes.position || o.morphAttributes.normal || o.morphAttributes.color, u = l === void 0 ? 0 : l.length, d = r.get(o);
		if (d === void 0 || d.count !== u) {
			d !== void 0 && d.texture.dispose();
			let e = o.morphAttributes.position !== void 0, n = o.morphAttributes.normal !== void 0, a = o.morphAttributes.color !== void 0, s = o.morphAttributes.position || [], c = o.morphAttributes.normal || [], l = o.morphAttributes.color || [], f = 0;
			e === !0 && (f = 1), n === !0 && (f = 2), a === !0 && (f = 3);
			let p = o.attributes.position.count * f, m = 1;
			p > t.maxTextureSize && (m = Math.ceil(p / t.maxTextureSize), p = t.maxTextureSize);
			let h = new Float32Array(p * m * 4 * u), g = new Kf(h, p, m, u);
			g.type = Eu, g.needsUpdate = !0;
			let _ = f * 4;
			for (let t = 0; t < u; t++) {
				let r = s[t], o = c[t], u = l[t], d = p * m * 4 * t;
				for (let t = 0; t < r.count; t++) {
					let s = t * _;
					e === !0 && (i.fromBufferAttribute(r, t), h[d + s + 0] = i.x, h[d + s + 1] = i.y, h[d + s + 2] = i.z, h[d + s + 3] = 0), n === !0 && (i.fromBufferAttribute(o, t), h[d + s + 4] = i.x, h[d + s + 5] = i.y, h[d + s + 6] = i.z, h[d + s + 7] = 0), a === !0 && (i.fromBufferAttribute(u, t), h[d + s + 8] = i.x, h[d + s + 9] = i.y, h[d + s + 10] = i.z, h[d + s + 11] = u.itemSize === 4 ? i.w : 1);
				}
			}
			d = {
				count: u,
				texture: g,
				size: new Tf(p, m)
			}, r.set(o, d);
			function v() {
				g.dispose(), r.delete(o), o.removeEventListener("dispose", v);
			}
			o.addEventListener("dispose", v);
		}
		if (a.isInstancedMesh === !0 && a.morphTexture !== null) s.getUniforms().setValue(e, "morphTexture", a.morphTexture, n);
		else {
			let t = 0;
			for (let e = 0; e < c.length; e++) t += c[e];
			let n = o.morphTargetsRelative ? 1 : 1 - t;
			s.getUniforms().setValue(e, "morphTargetBaseInfluence", n), s.getUniforms().setValue(e, "morphTargetInfluences", c);
		}
		s.getUniforms().setValue(e, "morphTargetsTexture", d.texture, n), s.getUniforms().setValue(e, "morphTargetsTextureSize", d.size);
	}
	return { update: a };
}
function x_(e, t, n, r, i) {
	let a = /* @__PURE__ */ new WeakMap();
	function o(r) {
		let o = i.render.frame, s = r.geometry, l = t.get(r, s);
		if (a.get(l) !== o && (t.update(l), a.set(l, o)), r.isInstancedMesh && (r.hasEventListener("dispose", c) === !1 && r.addEventListener("dispose", c), a.get(r) !== o && (n.update(r.instanceMatrix, e.ARRAY_BUFFER), r.instanceColor !== null && n.update(r.instanceColor, e.ARRAY_BUFFER), a.set(r, o))), r.isSkinnedMesh) {
			let e = r.skeleton;
			a.get(e) !== o && (e.update(), a.set(e, o));
		}
		return l;
	}
	function s() {
		a = /* @__PURE__ */ new WeakMap();
	}
	function c(e) {
		let t = e.target;
		t.removeEventListener("dispose", c), r.releaseStatesOfObject(t), n.remove(t.instanceMatrix), t.instanceColor !== null && n.remove(t.instanceColor);
	}
	return {
		update: o,
		dispose: s
	};
}
var S_ = {
	1: "LINEAR_TONE_MAPPING",
	2: "REINHARD_TONE_MAPPING",
	3: "CINEON_TONE_MAPPING",
	4: "ACES_FILMIC_TONE_MAPPING",
	6: "AGX_TONE_MAPPING",
	7: "NEUTRAL_TONE_MAPPING",
	5: "CUSTOM_TONE_MAPPING"
};
function C_(e, t, n, r, i, a) {
	let o = new Gf(t, n, {
		type: e,
		depthBuffer: i,
		stencilBuffer: a,
		samples: r ? 4 : 0,
		depthTexture: i ? new Dh(t, n) : void 0
	}), s = new Gf(t, n, {
		type: Du,
		depthBuffer: !1,
		stencilBuffer: !1
	}), c = new wm();
	c.setAttribute("position", new fm([
		-1,
		3,
		0,
		-1,
		-1,
		0,
		3,
		-1,
		0
	], 3)), c.setAttribute("uv", new fm([
		0,
		2,
		0,
		0,
		2,
		0
	], 2));
	let l = new Vh({
		uniforms: { tDiffuse: { value: null } },
		vertexShader: "\n			precision highp float;\n\n			uniform mat4 modelViewMatrix;\n			uniform mat4 projectionMatrix;\n\n			attribute vec3 position;\n			attribute vec2 uv;\n\n			varying vec2 vUv;\n\n			void main() {\n				vUv = uv;\n				gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n			}",
		fragmentShader: "\n			precision highp float;\n\n			uniform sampler2D tDiffuse;\n\n			varying vec2 vUv;\n\n			#include <tonemapping_pars_fragment>\n			#include <colorspace_pars_fragment>\n\n			void main() {\n				gl_FragColor = texture2D( tDiffuse, vUv );\n\n				#ifdef LINEAR_TONE_MAPPING\n					gl_FragColor.rgb = LinearToneMapping( gl_FragColor.rgb );\n				#elif defined( REINHARD_TONE_MAPPING )\n					gl_FragColor.rgb = ReinhardToneMapping( gl_FragColor.rgb );\n				#elif defined( CINEON_TONE_MAPPING )\n					gl_FragColor.rgb = CineonToneMapping( gl_FragColor.rgb );\n				#elif defined( ACES_FILMIC_TONE_MAPPING )\n					gl_FragColor.rgb = ACESFilmicToneMapping( gl_FragColor.rgb );\n				#elif defined( AGX_TONE_MAPPING )\n					gl_FragColor.rgb = AgXToneMapping( gl_FragColor.rgb );\n				#elif defined( NEUTRAL_TONE_MAPPING )\n					gl_FragColor.rgb = NeutralToneMapping( gl_FragColor.rgb );\n				#elif defined( CUSTOM_TONE_MAPPING )\n					gl_FragColor.rgb = CustomToneMapping( gl_FragColor.rgb );\n				#endif\n\n				#ifdef SRGB_TRANSFER\n					gl_FragColor = sRGBTransferOETF( gl_FragColor );\n				#endif\n			}",
		depthTest: !1,
		depthWrite: !1
	}), u = new qm(c, l), d = new mg(-1, 1, 1, -1, 0, 1), f = null, p = null, m = !1, h, g = null, _ = [], v = !1;
	this.setSize = function(e, t) {
		o.setSize(e, t), s.setSize(e, t);
		for (let n = 0; n < _.length; n++) {
			let r = _[n];
			r.setSize && r.setSize(e, t);
		}
	}, this.setEffects = function(e) {
		_ = e, v = _.length > 0 && _[0].isRenderPass === !0;
		let t = o.width, n = o.height;
		for (let e = 0; e < _.length; e++) {
			let r = _[e];
			r.setSize && r.setSize(t, n);
		}
	}, this.begin = function(e, t) {
		if (m || e.toneMapping === 0 && _.length === 0) return !1;
		if (g = t, t !== null) {
			let e = t.width, n = t.height;
			(o.width !== e || o.height !== n) && this.setSize(e, n);
		}
		return v === !1 && e.setRenderTarget(o), h = e.toneMapping, e.toneMapping = 0, !0;
	}, this.hasRenderPass = function() {
		return v;
	}, this.end = function(e, t) {
		e.toneMapping = h, m = !0;
		let n = o, r = s;
		for (let i = 0; i < _.length; i++) {
			let a = _[i];
			if (a.enabled !== !1 && (a.render(e, r, n, t), a.needsSwap !== !1)) {
				let e = n;
				n = r, r = e;
			}
		}
		if (f !== e.outputColorSpace || p !== e.toneMapping) {
			f = e.outputColorSpace, p = e.toneMapping, l.defines = {}, X.getTransfer(f) === "srgb" && (l.defines.SRGB_TRANSFER = "");
			let t = S_[p];
			t && (l.defines[t] = ""), l.needsUpdate = !0;
		}
		l.uniforms.tDiffuse.value = n.texture, e.setRenderTarget(g), e.render(u, d), g = null, m = !1;
	}, this.isCompositing = function() {
		return m;
	}, this.dispose = function() {
		o.depthTexture && o.depthTexture.dispose(), o.dispose(), s.dispose(), c.dispose(), l.dispose();
	};
}
var w_ = /*@__PURE__*/ new Hf(), T_ = /*@__PURE__*/ new Dh(1, 1), E_ = /*@__PURE__*/ new Kf(), D_ = /*@__PURE__*/ new qf(), O_ = /*@__PURE__*/ new Th(), k_ = [], A_ = [], j_ = /* @__PURE__ */ new Float32Array(16), M_ = /* @__PURE__ */ new Float32Array(9), N_ = /* @__PURE__ */ new Float32Array(4);
function P_(e, t, n) {
	let r = e[0];
	if (r <= 0 || r > 0) return e;
	let i = t * n, a = k_[i];
	if (a === void 0 && (a = new Float32Array(i), k_[i] = a), t !== 0) {
		r.toArray(a, 0);
		for (let r = 1, i = 0; r !== t; ++r) i += n, e[r].toArray(a, i);
	}
	return a;
}
function F_(e, t) {
	if (e.length !== t.length) return !1;
	for (let n = 0, r = e.length; n < r; n++) if (e[n] !== t[n]) return !1;
	return !0;
}
function I_(e, t) {
	for (let n = 0, r = t.length; n < r; n++) e[n] = t[n];
}
function L_(e, t) {
	let n = A_[t];
	n === void 0 && (n = new Int32Array(t), A_[t] = n);
	for (let r = 0; r !== t; ++r) n[r] = e.allocateTextureUnit();
	return n;
}
function R_(e, t) {
	let n = this.cache;
	n[0] !== t && (e.uniform1f(this.addr, t), n[0] = t);
}
function z_(e, t) {
	let n = this.cache;
	if (t.x !== void 0) (n[0] !== t.x || n[1] !== t.y) && (e.uniform2f(this.addr, t.x, t.y), n[0] = t.x, n[1] = t.y);
	else {
		if (F_(n, t)) return;
		e.uniform2fv(this.addr, t), I_(n, t);
	}
}
function B_(e, t) {
	let n = this.cache;
	if (t.x !== void 0) (n[0] !== t.x || n[1] !== t.y || n[2] !== t.z) && (e.uniform3f(this.addr, t.x, t.y, t.z), n[0] = t.x, n[1] = t.y, n[2] = t.z);
	else if (t.r !== void 0) (n[0] !== t.r || n[1] !== t.g || n[2] !== t.b) && (e.uniform3f(this.addr, t.r, t.g, t.b), n[0] = t.r, n[1] = t.g, n[2] = t.b);
	else {
		if (F_(n, t)) return;
		e.uniform3fv(this.addr, t), I_(n, t);
	}
}
function V_(e, t) {
	let n = this.cache;
	if (t.x !== void 0) (n[0] !== t.x || n[1] !== t.y || n[2] !== t.z || n[3] !== t.w) && (e.uniform4f(this.addr, t.x, t.y, t.z, t.w), n[0] = t.x, n[1] = t.y, n[2] = t.z, n[3] = t.w);
	else {
		if (F_(n, t)) return;
		e.uniform4fv(this.addr, t), I_(n, t);
	}
}
function H_(e, t) {
	let n = this.cache, r = t.elements;
	if (r === void 0) {
		if (F_(n, t)) return;
		e.uniformMatrix2fv(this.addr, !1, t), I_(n, t);
	} else {
		if (F_(n, r)) return;
		N_.set(r), e.uniformMatrix2fv(this.addr, !1, N_), I_(n, r);
	}
}
function U_(e, t) {
	let n = this.cache, r = t.elements;
	if (r === void 0) {
		if (F_(n, t)) return;
		e.uniformMatrix3fv(this.addr, !1, t), I_(n, t);
	} else {
		if (F_(n, r)) return;
		M_.set(r), e.uniformMatrix3fv(this.addr, !1, M_), I_(n, r);
	}
}
function W_(e, t) {
	let n = this.cache, r = t.elements;
	if (r === void 0) {
		if (F_(n, t)) return;
		e.uniformMatrix4fv(this.addr, !1, t), I_(n, t);
	} else {
		if (F_(n, r)) return;
		j_.set(r), e.uniformMatrix4fv(this.addr, !1, j_), I_(n, r);
	}
}
function G_(e, t) {
	let n = this.cache;
	n[0] !== t && (e.uniform1i(this.addr, t), n[0] = t);
}
function K_(e, t) {
	let n = this.cache;
	if (t.x !== void 0) (n[0] !== t.x || n[1] !== t.y) && (e.uniform2i(this.addr, t.x, t.y), n[0] = t.x, n[1] = t.y);
	else {
		if (F_(n, t)) return;
		e.uniform2iv(this.addr, t), I_(n, t);
	}
}
function q_(e, t) {
	let n = this.cache;
	if (t.x !== void 0) (n[0] !== t.x || n[1] !== t.y || n[2] !== t.z) && (e.uniform3i(this.addr, t.x, t.y, t.z), n[0] = t.x, n[1] = t.y, n[2] = t.z);
	else {
		if (F_(n, t)) return;
		e.uniform3iv(this.addr, t), I_(n, t);
	}
}
function J_(e, t) {
	let n = this.cache;
	if (t.x !== void 0) (n[0] !== t.x || n[1] !== t.y || n[2] !== t.z || n[3] !== t.w) && (e.uniform4i(this.addr, t.x, t.y, t.z, t.w), n[0] = t.x, n[1] = t.y, n[2] = t.z, n[3] = t.w);
	else {
		if (F_(n, t)) return;
		e.uniform4iv(this.addr, t), I_(n, t);
	}
}
function Y_(e, t) {
	let n = this.cache;
	n[0] !== t && (e.uniform1ui(this.addr, t), n[0] = t);
}
function X_(e, t) {
	let n = this.cache;
	if (t.x !== void 0) (n[0] !== t.x || n[1] !== t.y) && (e.uniform2ui(this.addr, t.x, t.y), n[0] = t.x, n[1] = t.y);
	else {
		if (F_(n, t)) return;
		e.uniform2uiv(this.addr, t), I_(n, t);
	}
}
function Z_(e, t) {
	let n = this.cache;
	if (t.x !== void 0) (n[0] !== t.x || n[1] !== t.y || n[2] !== t.z) && (e.uniform3ui(this.addr, t.x, t.y, t.z), n[0] = t.x, n[1] = t.y, n[2] = t.z);
	else {
		if (F_(n, t)) return;
		e.uniform3uiv(this.addr, t), I_(n, t);
	}
}
function Q_(e, t) {
	let n = this.cache;
	if (t.x !== void 0) (n[0] !== t.x || n[1] !== t.y || n[2] !== t.z || n[3] !== t.w) && (e.uniform4ui(this.addr, t.x, t.y, t.z, t.w), n[0] = t.x, n[1] = t.y, n[2] = t.z, n[3] = t.w);
	else {
		if (F_(n, t)) return;
		e.uniform4uiv(this.addr, t), I_(n, t);
	}
}
function $_(e, t, n) {
	let r = this.cache, i = n.allocateTextureUnit();
	r[0] !== i && (e.uniform1i(this.addr, i), r[0] = i);
	let a;
	this.type === e.SAMPLER_2D_SHADOW ? (T_.compareFunction = n.isReversedDepthBuffer() ? 518 : 515, a = T_) : a = w_, n.setTexture2D(t || a, i);
}
function ev(e, t, n) {
	let r = this.cache, i = n.allocateTextureUnit();
	r[0] !== i && (e.uniform1i(this.addr, i), r[0] = i), n.setTexture3D(t || D_, i);
}
function tv(e, t, n) {
	let r = this.cache, i = n.allocateTextureUnit();
	r[0] !== i && (e.uniform1i(this.addr, i), r[0] = i), n.setTextureCube(t || O_, i);
}
function nv(e, t, n) {
	let r = this.cache, i = n.allocateTextureUnit();
	r[0] !== i && (e.uniform1i(this.addr, i), r[0] = i), n.setTexture2DArray(t || E_, i);
}
function rv(e) {
	switch (e) {
		case 5126: return R_;
		case 35664: return z_;
		case 35665: return B_;
		case 35666: return V_;
		case 35674: return H_;
		case 35675: return U_;
		case 35676: return W_;
		case 5124:
		case 35670: return G_;
		case 35667:
		case 35671: return K_;
		case 35668:
		case 35672: return q_;
		case 35669:
		case 35673: return J_;
		case 5125: return Y_;
		case 36294: return X_;
		case 36295: return Z_;
		case 36296: return Q_;
		case 35678:
		case 36198:
		case 36298:
		case 36306:
		case 35682: return $_;
		case 35679:
		case 36299:
		case 36307: return ev;
		case 35680:
		case 36300:
		case 36308:
		case 36293: return tv;
		case 36289:
		case 36303:
		case 36311:
		case 36292: return nv;
	}
}
function iv(e, t) {
	e.uniform1fv(this.addr, t);
}
function av(e, t) {
	let n = P_(t, this.size, 2);
	e.uniform2fv(this.addr, n);
}
function ov(e, t) {
	let n = P_(t, this.size, 3);
	e.uniform3fv(this.addr, n);
}
function sv(e, t) {
	let n = P_(t, this.size, 4);
	e.uniform4fv(this.addr, n);
}
function cv(e, t) {
	let n = P_(t, this.size, 4);
	e.uniformMatrix2fv(this.addr, !1, n);
}
function lv(e, t) {
	let n = P_(t, this.size, 9);
	e.uniformMatrix3fv(this.addr, !1, n);
}
function uv(e, t) {
	let n = P_(t, this.size, 16);
	e.uniformMatrix4fv(this.addr, !1, n);
}
function dv(e, t) {
	e.uniform1iv(this.addr, t);
}
function fv(e, t) {
	e.uniform2iv(this.addr, t);
}
function pv(e, t) {
	e.uniform3iv(this.addr, t);
}
function mv(e, t) {
	e.uniform4iv(this.addr, t);
}
function hv(e, t) {
	e.uniform1uiv(this.addr, t);
}
function gv(e, t) {
	e.uniform2uiv(this.addr, t);
}
function _v(e, t) {
	e.uniform3uiv(this.addr, t);
}
function vv(e, t) {
	e.uniform4uiv(this.addr, t);
}
function yv(e, t, n) {
	let r = this.cache, i = t.length, a = L_(n, i);
	F_(r, a) || (e.uniform1iv(this.addr, a), I_(r, a));
	let o;
	o = this.type === e.SAMPLER_2D_SHADOW ? T_ : w_;
	for (let e = 0; e !== i; ++e) n.setTexture2D(t[e] || o, a[e]);
}
function bv(e, t, n) {
	let r = this.cache, i = t.length, a = L_(n, i);
	F_(r, a) || (e.uniform1iv(this.addr, a), I_(r, a));
	for (let e = 0; e !== i; ++e) n.setTexture3D(t[e] || D_, a[e]);
}
function xv(e, t, n) {
	let r = this.cache, i = t.length, a = L_(n, i);
	F_(r, a) || (e.uniform1iv(this.addr, a), I_(r, a));
	for (let e = 0; e !== i; ++e) n.setTextureCube(t[e] || O_, a[e]);
}
function Sv(e, t, n) {
	let r = this.cache, i = t.length, a = L_(n, i);
	F_(r, a) || (e.uniform1iv(this.addr, a), I_(r, a));
	for (let e = 0; e !== i; ++e) n.setTexture2DArray(t[e] || E_, a[e]);
}
function Cv(e) {
	switch (e) {
		case 5126: return iv;
		case 35664: return av;
		case 35665: return ov;
		case 35666: return sv;
		case 35674: return cv;
		case 35675: return lv;
		case 35676: return uv;
		case 5124:
		case 35670: return dv;
		case 35667:
		case 35671: return fv;
		case 35668:
		case 35672: return pv;
		case 35669:
		case 35673: return mv;
		case 5125: return hv;
		case 36294: return gv;
		case 36295: return _v;
		case 36296: return vv;
		case 35678:
		case 36198:
		case 36298:
		case 36306:
		case 35682: return yv;
		case 35679:
		case 36299:
		case 36307: return bv;
		case 35680:
		case 36300:
		case 36308:
		case 36293: return xv;
		case 36289:
		case 36303:
		case 36311:
		case 36292: return Sv;
	}
}
var wv = class {
	constructor(e, t, n) {
		this.id = e, this.addr = n, this.cache = [], this.type = t.type, this.setValue = rv(t.type);
	}
}, Tv = class {
	constructor(e, t, n) {
		this.id = e, this.addr = n, this.cache = [], this.type = t.type, this.size = t.size, this.setValue = Cv(t.type);
	}
}, Ev = class {
	constructor(e) {
		this.id = e, this.seq = [], this.map = {};
	}
	setValue(e, t, n) {
		let r = this.seq;
		for (let i = 0, a = r.length; i !== a; ++i) {
			let a = r[i];
			a.setValue(e, t[a.id], n);
		}
	}
}, Dv = /(\w+)(\])?(\[|\.)?/g;
function Ov(e, t) {
	e.seq.push(t), e.map[t.id] = t;
}
function kv(e, t, n) {
	let r = e.name, i = r.length;
	for (Dv.lastIndex = 0;;) {
		let a = Dv.exec(r), o = Dv.lastIndex, s = a[1], c = a[2] === "]", l = a[3];
		if (c && (s |= 0), l === void 0 || l === "[" && o + 2 === i) {
			Ov(n, l === void 0 ? new wv(s, e, t) : new Tv(s, e, t));
			break;
		} else {
			let e = n.map[s];
			e === void 0 && (e = new Ev(s), Ov(n, e)), n = e;
		}
	}
}
var Av = class {
	constructor(e, t) {
		this.seq = [], this.map = {};
		let n = e.getProgramParameter(t, e.ACTIVE_UNIFORMS);
		for (let r = 0; r < n; ++r) {
			let n = e.getActiveUniform(t, r);
			kv(n, e.getUniformLocation(t, n.name), this);
		}
		let r = [], i = [];
		for (let t of this.seq) t.type === e.SAMPLER_2D_SHADOW || t.type === e.SAMPLER_CUBE_SHADOW || t.type === e.SAMPLER_2D_ARRAY_SHADOW ? r.push(t) : i.push(t);
		r.length > 0 && (this.seq = r.concat(i));
	}
	setValue(e, t, n, r) {
		let i = this.map[t];
		i !== void 0 && i.setValue(e, n, r);
	}
	setOptional(e, t, n) {
		let r = t[n];
		r !== void 0 && this.setValue(e, n, r);
	}
	static upload(e, t, n, r) {
		for (let i = 0, a = t.length; i !== a; ++i) {
			let a = t[i], o = n[a.id];
			o.needsUpdate !== !1 && a.setValue(e, o.value, r);
		}
	}
	static seqWithValue(e, t) {
		let n = [];
		for (let r = 0, i = e.length; r !== i; ++r) {
			let i = e[r];
			i.id in t && n.push(i);
		}
		return n;
	}
};
function jv(e, t, n) {
	let r = e.createShader(t);
	return e.shaderSource(r, n), e.compileShader(r), r;
}
var Mv = 37297, Nv = 0;
function Pv(e, t) {
	let n = e.split("\n"), r = [], i = Math.max(t - 6, 0), a = Math.min(t + 6, n.length);
	for (let e = i; e < a; e++) {
		let i = e + 1;
		r.push(`${i === t ? ">" : " "} ${i}: ${n[e]}`);
	}
	return r.join("\n");
}
var Fv = /*@__PURE__*/ new Y();
function Iv(e) {
	X._getMatrix(Fv, X.workingColorSpace, e);
	let t = `mat3( ${Fv.elements.map((e) => e.toFixed(4))} )`;
	switch (X.getTransfer(e)) {
		case Fd: return [t, "LinearTransferOETF"];
		case Id: return [t, "sRGBTransferOETF"];
		default: return G("WebGLProgram: Unsupported color space: ", e), [t, "LinearTransferOETF"];
	}
}
function Lv(e, t, n) {
	let r = e.getShaderParameter(t, e.COMPILE_STATUS), i = (e.getShaderInfoLog(t) || "").trim();
	if (r && i === "") return "";
	let a = /ERROR: 0:(\d+)/.exec(i);
	if (a) {
		let r = parseInt(a[1]);
		return n.toUpperCase() + "\n\n" + i + "\n\n" + Pv(e.getShaderSource(t), r);
	} else return i;
}
function Rv(e, t) {
	let n = Iv(t);
	return [
		`vec4 ${e}( vec4 value ) {`,
		`	return ${n[1]}( vec4( value.rgb * ${n[0]}, value.a ) );`,
		"}"
	].join("\n");
}
var zv = {
	1: "Linear",
	2: "Reinhard",
	3: "Cineon",
	4: "ACESFilmic",
	6: "AgX",
	7: "Neutral",
	5: "Custom"
};
function Bv(e, t) {
	let n = zv[t];
	return n === void 0 ? (G("WebGLProgram: Unsupported toneMapping:", t), "vec3 " + e + "( vec3 color ) { return LinearToneMapping( color ); }") : "vec3 " + e + "( vec3 color ) { return " + n + "ToneMapping( color ); }";
}
var Vv = /*@__PURE__*/ new J();
function Hv() {
	return X.getLuminanceCoefficients(Vv), [
		"float luminance( const in vec3 rgb ) {",
		`	const vec3 weights = vec3( ${Vv.x.toFixed(4)}, ${Vv.y.toFixed(4)}, ${Vv.z.toFixed(4)} );`,
		"	return dot( weights, rgb );",
		"}"
	].join("\n");
}
function Uv(e) {
	return [e.extensionClipCullDistance ? "#extension GL_ANGLE_clip_cull_distance : require" : "", e.extensionMultiDraw ? "#extension GL_ANGLE_multi_draw : require" : ""].filter(Kv).join("\n");
}
function Wv(e) {
	let t = [];
	for (let n in e) {
		let r = e[n];
		r !== !1 && t.push("#define " + n + " " + r);
	}
	return t.join("\n");
}
function Gv(e, t) {
	let n = {}, r = e.getProgramParameter(t, e.ACTIVE_ATTRIBUTES);
	for (let i = 0; i < r; i++) {
		let r = e.getActiveAttrib(t, i), a = r.name, o = 1;
		r.type === e.FLOAT_MAT2 && (o = 2), r.type === e.FLOAT_MAT3 && (o = 3), r.type === e.FLOAT_MAT4 && (o = 4), n[a] = {
			type: r.type,
			location: e.getAttribLocation(t, a),
			locationSize: o
		};
	}
	return n;
}
function Kv(e) {
	return e !== "";
}
function qv(e, t) {
	let n = t.numSpotLightShadows + t.numSpotLightMaps - t.numSpotLightShadowsWithMaps;
	return e.replace(/NUM_DIR_LIGHTS/g, t.numDirLights).replace(/NUM_SPOT_LIGHTS/g, t.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g, t.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g, n).replace(/NUM_RECT_AREA_LIGHTS/g, t.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g, t.numPointLights).replace(/NUM_HEMI_LIGHTS/g, t.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g, t.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g, t.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g, t.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g, t.numPointLightShadows);
}
function Jv(e, t) {
	return e.replace(/NUM_CLIPPING_PLANES/g, t.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g, t.numClippingPlanes - t.numClipIntersection);
}
var Yv = /^[ \t]*#include +<([\w\d./]+)>/gm;
function Xv(e) {
	return e.replace(Yv, Qv);
}
var Zv = /* @__PURE__ */ new Map();
function Qv(e, t) {
	let n = Q[t];
	if (n === void 0) {
		let e = Zv.get(t);
		if (e !== void 0) n = Q[e], G("WebGLRenderer: Shader chunk \"%s\" has been deprecated. Use \"%s\" instead.", t, e);
		else throw Error("THREE.WebGLProgram: Can not resolve #include <" + t + ">");
	}
	return Xv(n);
}
var $v = /#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;
function ey(e) {
	return e.replace($v, ty);
}
function ty(e, t, n, r) {
	let i = "";
	for (let e = parseInt(t); e < parseInt(n); e++) i += r.replace(/\[\s*i\s*\]/g, "[ " + e + " ]").replace(/UNROLLED_LOOP_INDEX/g, e);
	return i;
}
function ny(e) {
	let t = `precision ${e.precision} float;
	precision ${e.precision} int;
	precision ${e.precision} sampler2D;
	precision ${e.precision} samplerCube;
	precision ${e.precision} sampler3D;
	precision ${e.precision} sampler2DArray;
	precision ${e.precision} sampler2DShadow;
	precision ${e.precision} samplerCubeShadow;
	precision ${e.precision} sampler2DArrayShadow;
	precision ${e.precision} isampler2D;
	precision ${e.precision} isampler3D;
	precision ${e.precision} isamplerCube;
	precision ${e.precision} isampler2DArray;
	precision ${e.precision} usampler2D;
	precision ${e.precision} usampler3D;
	precision ${e.precision} usamplerCube;
	precision ${e.precision} usampler2DArray;
	`;
	return e.precision === "highp" ? t += "\n#define HIGH_PRECISION" : e.precision === "mediump" ? t += "\n#define MEDIUM_PRECISION" : e.precision === "lowp" && (t += "\n#define LOW_PRECISION"), t;
}
var ry = {
	1: "SHADOWMAP_TYPE_PCF",
	3: "SHADOWMAP_TYPE_VSM"
};
function iy(e) {
	return ry[e.shadowMapType] || "SHADOWMAP_TYPE_BASIC";
}
var ay = {
	301: "ENVMAP_TYPE_CUBE",
	302: "ENVMAP_TYPE_CUBE",
	306: "ENVMAP_TYPE_CUBE_UV"
};
function oy(e) {
	return e.envMap === !1 ? "ENVMAP_TYPE_CUBE" : ay[e.envMapMode] || "ENVMAP_TYPE_CUBE";
}
var sy = { 302: "ENVMAP_MODE_REFRACTION" };
function cy(e) {
	return e.envMap === !1 ? "ENVMAP_MODE_REFLECTION" : sy[e.envMapMode] || "ENVMAP_MODE_REFLECTION";
}
var ly = {
	0: "ENVMAP_BLENDING_MULTIPLY",
	1: "ENVMAP_BLENDING_MIX",
	2: "ENVMAP_BLENDING_ADD"
};
function uy(e) {
	return e.envMap === !1 ? "ENVMAP_BLENDING_NONE" : ly[e.combine] || "ENVMAP_BLENDING_NONE";
}
function dy(e) {
	let t = e.envMapCubeUVHeight;
	if (t === null) return null;
	let n = Math.log2(t) - 2, r = 1 / t;
	return {
		texelWidth: 1 / (3 * Math.max(2 ** n, 112)),
		texelHeight: r,
		maxMip: n
	};
}
function fy(e, t, n, r) {
	let i = e.getContext(), a = n.defines, o = n.vertexShader, s = n.fragmentShader, c = iy(n), l = oy(n), u = cy(n), d = uy(n), f = dy(n), p = Uv(n), m = Wv(a), h = i.createProgram(), g, _, v = n.glslVersion ? "#version " + n.glslVersion + "\n" : "";
	n.isRawShaderMaterial ? (g = [
		"#define SHADER_TYPE " + n.shaderType,
		"#define SHADER_NAME " + n.shaderName,
		m
	].filter(Kv).join("\n"), g.length > 0 && (g += "\n"), _ = [
		"#define SHADER_TYPE " + n.shaderType,
		"#define SHADER_NAME " + n.shaderName,
		m
	].filter(Kv).join("\n"), _.length > 0 && (_ += "\n")) : (g = [
		ny(n),
		"#define SHADER_TYPE " + n.shaderType,
		"#define SHADER_NAME " + n.shaderName,
		m,
		n.extensionClipCullDistance ? "#define USE_CLIP_DISTANCE" : "",
		n.batching ? "#define USE_BATCHING" : "",
		n.batchingColor ? "#define USE_BATCHING_COLOR" : "",
		n.instancing ? "#define USE_INSTANCING" : "",
		n.instancingColor ? "#define USE_INSTANCING_COLOR" : "",
		n.instancingMorph ? "#define USE_INSTANCING_MORPH" : "",
		n.useFog && n.fog ? "#define USE_FOG" : "",
		n.useFog && n.fogExp2 ? "#define FOG_EXP2" : "",
		n.map ? "#define USE_MAP" : "",
		n.envMap ? "#define USE_ENVMAP" : "",
		n.envMap ? "#define " + u : "",
		n.lightMap ? "#define USE_LIGHTMAP" : "",
		n.aoMap ? "#define USE_AOMAP" : "",
		n.bumpMap ? "#define USE_BUMPMAP" : "",
		n.normalMap ? "#define USE_NORMALMAP" : "",
		n.normalMapObjectSpace ? "#define USE_NORMALMAP_OBJECTSPACE" : "",
		n.normalMapTangentSpace ? "#define USE_NORMALMAP_TANGENTSPACE" : "",
		n.displacementMap ? "#define USE_DISPLACEMENTMAP" : "",
		n.emissiveMap ? "#define USE_EMISSIVEMAP" : "",
		n.anisotropy ? "#define USE_ANISOTROPY" : "",
		n.anisotropyMap ? "#define USE_ANISOTROPYMAP" : "",
		n.clearcoatMap ? "#define USE_CLEARCOATMAP" : "",
		n.clearcoatRoughnessMap ? "#define USE_CLEARCOAT_ROUGHNESSMAP" : "",
		n.clearcoatNormalMap ? "#define USE_CLEARCOAT_NORMALMAP" : "",
		n.iridescenceMap ? "#define USE_IRIDESCENCEMAP" : "",
		n.iridescenceThicknessMap ? "#define USE_IRIDESCENCE_THICKNESSMAP" : "",
		n.specularMap ? "#define USE_SPECULARMAP" : "",
		n.specularColorMap ? "#define USE_SPECULAR_COLORMAP" : "",
		n.specularIntensityMap ? "#define USE_SPECULAR_INTENSITYMAP" : "",
		n.roughnessMap ? "#define USE_ROUGHNESSMAP" : "",
		n.metalnessMap ? "#define USE_METALNESSMAP" : "",
		n.alphaMap ? "#define USE_ALPHAMAP" : "",
		n.alphaHash ? "#define USE_ALPHAHASH" : "",
		n.transmission ? "#define USE_TRANSMISSION" : "",
		n.transmissionMap ? "#define USE_TRANSMISSIONMAP" : "",
		n.thicknessMap ? "#define USE_THICKNESSMAP" : "",
		n.sheenColorMap ? "#define USE_SHEEN_COLORMAP" : "",
		n.sheenRoughnessMap ? "#define USE_SHEEN_ROUGHNESSMAP" : "",
		n.mapUv ? "#define MAP_UV " + n.mapUv : "",
		n.alphaMapUv ? "#define ALPHAMAP_UV " + n.alphaMapUv : "",
		n.lightMapUv ? "#define LIGHTMAP_UV " + n.lightMapUv : "",
		n.aoMapUv ? "#define AOMAP_UV " + n.aoMapUv : "",
		n.emissiveMapUv ? "#define EMISSIVEMAP_UV " + n.emissiveMapUv : "",
		n.bumpMapUv ? "#define BUMPMAP_UV " + n.bumpMapUv : "",
		n.normalMapUv ? "#define NORMALMAP_UV " + n.normalMapUv : "",
		n.displacementMapUv ? "#define DISPLACEMENTMAP_UV " + n.displacementMapUv : "",
		n.metalnessMapUv ? "#define METALNESSMAP_UV " + n.metalnessMapUv : "",
		n.roughnessMapUv ? "#define ROUGHNESSMAP_UV " + n.roughnessMapUv : "",
		n.anisotropyMapUv ? "#define ANISOTROPYMAP_UV " + n.anisotropyMapUv : "",
		n.clearcoatMapUv ? "#define CLEARCOATMAP_UV " + n.clearcoatMapUv : "",
		n.clearcoatNormalMapUv ? "#define CLEARCOAT_NORMALMAP_UV " + n.clearcoatNormalMapUv : "",
		n.clearcoatRoughnessMapUv ? "#define CLEARCOAT_ROUGHNESSMAP_UV " + n.clearcoatRoughnessMapUv : "",
		n.iridescenceMapUv ? "#define IRIDESCENCEMAP_UV " + n.iridescenceMapUv : "",
		n.iridescenceThicknessMapUv ? "#define IRIDESCENCE_THICKNESSMAP_UV " + n.iridescenceThicknessMapUv : "",
		n.sheenColorMapUv ? "#define SHEEN_COLORMAP_UV " + n.sheenColorMapUv : "",
		n.sheenRoughnessMapUv ? "#define SHEEN_ROUGHNESSMAP_UV " + n.sheenRoughnessMapUv : "",
		n.specularMapUv ? "#define SPECULARMAP_UV " + n.specularMapUv : "",
		n.specularColorMapUv ? "#define SPECULAR_COLORMAP_UV " + n.specularColorMapUv : "",
		n.specularIntensityMapUv ? "#define SPECULAR_INTENSITYMAP_UV " + n.specularIntensityMapUv : "",
		n.transmissionMapUv ? "#define TRANSMISSIONMAP_UV " + n.transmissionMapUv : "",
		n.thicknessMapUv ? "#define THICKNESSMAP_UV " + n.thicknessMapUv : "",
		n.vertexTangents && n.flatShading === !1 ? "#define USE_TANGENT" : "",
		n.vertexNormals ? "#define HAS_NORMAL" : "",
		n.vertexColors ? "#define USE_COLOR" : "",
		n.vertexAlphas ? "#define USE_COLOR_ALPHA" : "",
		n.vertexUv1s ? "#define USE_UV1" : "",
		n.vertexUv2s ? "#define USE_UV2" : "",
		n.vertexUv3s ? "#define USE_UV3" : "",
		n.pointsUvs ? "#define USE_POINTS_UV" : "",
		n.flatShading ? "#define FLAT_SHADED" : "",
		n.skinning ? "#define USE_SKINNING" : "",
		n.morphTargets ? "#define USE_MORPHTARGETS" : "",
		n.morphNormals && n.flatShading === !1 ? "#define USE_MORPHNORMALS" : "",
		n.morphColors ? "#define USE_MORPHCOLORS" : "",
		n.morphTargetsCount > 0 ? "#define MORPHTARGETS_TEXTURE_STRIDE " + n.morphTextureStride : "",
		n.morphTargetsCount > 0 ? "#define MORPHTARGETS_COUNT " + n.morphTargetsCount : "",
		n.doubleSided ? "#define DOUBLE_SIDED" : "",
		n.flipSided ? "#define FLIP_SIDED" : "",
		n.shadowMapEnabled ? "#define USE_SHADOWMAP" : "",
		n.shadowMapEnabled ? "#define " + c : "",
		n.sizeAttenuation ? "#define USE_SIZEATTENUATION" : "",
		n.numLightProbes > 0 ? "#define USE_LIGHT_PROBES" : "",
		n.logarithmicDepthBuffer ? "#define USE_LOGARITHMIC_DEPTH_BUFFER" : "",
		n.reversedDepthBuffer ? "#define USE_REVERSED_DEPTH_BUFFER" : "",
		"uniform mat4 modelMatrix;",
		"uniform mat4 modelViewMatrix;",
		"uniform mat4 projectionMatrix;",
		"uniform mat4 viewMatrix;",
		"uniform mat3 normalMatrix;",
		"uniform vec3 cameraPosition;",
		"uniform bool isOrthographic;",
		"#ifdef USE_INSTANCING",
		"	attribute mat4 instanceMatrix;",
		"#endif",
		"#ifdef USE_INSTANCING_COLOR",
		"	attribute vec3 instanceColor;",
		"#endif",
		"#ifdef USE_INSTANCING_MORPH",
		"	uniform sampler2D morphTexture;",
		"#endif",
		"attribute vec3 position;",
		"attribute vec3 normal;",
		"attribute vec2 uv;",
		"#ifdef USE_UV1",
		"	attribute vec2 uv1;",
		"#endif",
		"#ifdef USE_UV2",
		"	attribute vec2 uv2;",
		"#endif",
		"#ifdef USE_UV3",
		"	attribute vec2 uv3;",
		"#endif",
		"#ifdef USE_TANGENT",
		"	attribute vec4 tangent;",
		"#endif",
		"#if defined( USE_COLOR_ALPHA )",
		"	attribute vec4 color;",
		"#elif defined( USE_COLOR )",
		"	attribute vec3 color;",
		"#endif",
		"#ifdef USE_SKINNING",
		"	attribute vec4 skinIndex;",
		"	attribute vec4 skinWeight;",
		"#endif",
		"\n"
	].filter(Kv).join("\n"), _ = [
		ny(n),
		"#define SHADER_TYPE " + n.shaderType,
		"#define SHADER_NAME " + n.shaderName,
		m,
		n.useFog && n.fog ? "#define USE_FOG" : "",
		n.useFog && n.fogExp2 ? "#define FOG_EXP2" : "",
		n.alphaToCoverage ? "#define ALPHA_TO_COVERAGE" : "",
		n.map ? "#define USE_MAP" : "",
		n.matcap ? "#define USE_MATCAP" : "",
		n.envMap ? "#define USE_ENVMAP" : "",
		n.envMap ? "#define " + l : "",
		n.envMap ? "#define " + u : "",
		n.envMap ? "#define " + d : "",
		f ? "#define CUBEUV_TEXEL_WIDTH " + f.texelWidth : "",
		f ? "#define CUBEUV_TEXEL_HEIGHT " + f.texelHeight : "",
		f ? "#define CUBEUV_MAX_MIP " + f.maxMip + ".0" : "",
		n.lightMap ? "#define USE_LIGHTMAP" : "",
		n.aoMap ? "#define USE_AOMAP" : "",
		n.bumpMap ? "#define USE_BUMPMAP" : "",
		n.normalMap ? "#define USE_NORMALMAP" : "",
		n.normalMapObjectSpace ? "#define USE_NORMALMAP_OBJECTSPACE" : "",
		n.normalMapTangentSpace ? "#define USE_NORMALMAP_TANGENTSPACE" : "",
		n.packedNormalMap ? "#define USE_PACKED_NORMALMAP" : "",
		n.emissiveMap ? "#define USE_EMISSIVEMAP" : "",
		n.anisotropy ? "#define USE_ANISOTROPY" : "",
		n.anisotropyMap ? "#define USE_ANISOTROPYMAP" : "",
		n.clearcoat ? "#define USE_CLEARCOAT" : "",
		n.clearcoatMap ? "#define USE_CLEARCOATMAP" : "",
		n.clearcoatRoughnessMap ? "#define USE_CLEARCOAT_ROUGHNESSMAP" : "",
		n.clearcoatNormalMap ? "#define USE_CLEARCOAT_NORMALMAP" : "",
		n.dispersion ? "#define USE_DISPERSION" : "",
		n.iridescence ? "#define USE_IRIDESCENCE" : "",
		n.iridescenceMap ? "#define USE_IRIDESCENCEMAP" : "",
		n.iridescenceThicknessMap ? "#define USE_IRIDESCENCE_THICKNESSMAP" : "",
		n.specularMap ? "#define USE_SPECULARMAP" : "",
		n.specularColorMap ? "#define USE_SPECULAR_COLORMAP" : "",
		n.specularIntensityMap ? "#define USE_SPECULAR_INTENSITYMAP" : "",
		n.roughnessMap ? "#define USE_ROUGHNESSMAP" : "",
		n.metalnessMap ? "#define USE_METALNESSMAP" : "",
		n.alphaMap ? "#define USE_ALPHAMAP" : "",
		n.alphaTest ? "#define USE_ALPHATEST" : "",
		n.alphaHash ? "#define USE_ALPHAHASH" : "",
		n.sheen ? "#define USE_SHEEN" : "",
		n.sheenColorMap ? "#define USE_SHEEN_COLORMAP" : "",
		n.sheenRoughnessMap ? "#define USE_SHEEN_ROUGHNESSMAP" : "",
		n.transmission ? "#define USE_TRANSMISSION" : "",
		n.transmissionMap ? "#define USE_TRANSMISSIONMAP" : "",
		n.thicknessMap ? "#define USE_THICKNESSMAP" : "",
		n.vertexTangents && n.flatShading === !1 ? "#define USE_TANGENT" : "",
		n.vertexColors || n.instancingColor ? "#define USE_COLOR" : "",
		n.vertexAlphas || n.batchingColor ? "#define USE_COLOR_ALPHA" : "",
		n.vertexUv1s ? "#define USE_UV1" : "",
		n.vertexUv2s ? "#define USE_UV2" : "",
		n.vertexUv3s ? "#define USE_UV3" : "",
		n.pointsUvs ? "#define USE_POINTS_UV" : "",
		n.gradientMap ? "#define USE_GRADIENTMAP" : "",
		n.flatShading ? "#define FLAT_SHADED" : "",
		n.doubleSided ? "#define DOUBLE_SIDED" : "",
		n.flipSided ? "#define FLIP_SIDED" : "",
		n.shadowMapEnabled ? "#define USE_SHADOWMAP" : "",
		n.shadowMapEnabled ? "#define " + c : "",
		n.premultipliedAlpha ? "#define PREMULTIPLIED_ALPHA" : "",
		n.numLightProbes > 0 ? "#define USE_LIGHT_PROBES" : "",
		n.numLightProbeGrids > 0 ? "#define USE_LIGHT_PROBES_GRID" : "",
		n.decodeVideoTexture ? "#define DECODE_VIDEO_TEXTURE" : "",
		n.decodeVideoTextureEmissive ? "#define DECODE_VIDEO_TEXTURE_EMISSIVE" : "",
		n.logarithmicDepthBuffer ? "#define USE_LOGARITHMIC_DEPTH_BUFFER" : "",
		n.reversedDepthBuffer ? "#define USE_REVERSED_DEPTH_BUFFER" : "",
		"uniform mat4 viewMatrix;",
		"uniform vec3 cameraPosition;",
		"uniform bool isOrthographic;",
		n.toneMapping === 0 ? "" : "#define TONE_MAPPING",
		n.toneMapping === 0 ? "" : Q.tonemapping_pars_fragment,
		n.toneMapping === 0 ? "" : Bv("toneMapping", n.toneMapping),
		n.dithering ? "#define DITHERING" : "",
		n.opaque ? "#define OPAQUE" : "",
		Q.colorspace_pars_fragment,
		Rv("linearToOutputTexel", n.outputColorSpace),
		Hv(),
		n.useDepthPacking ? "#define DEPTH_PACKING " + n.depthPacking : "",
		"\n"
	].filter(Kv).join("\n")), o = Xv(o), o = qv(o, n), o = Jv(o, n), s = Xv(s), s = qv(s, n), s = Jv(s, n), o = ey(o), s = ey(s), n.isRawShaderMaterial !== !0 && (v = "#version 300 es\n", g = [
		p,
		"#define attribute in",
		"#define varying out",
		"#define texture2D texture"
	].join("\n") + "\n" + g, _ = [
		"#define varying in",
		n.glslVersion === "300 es" ? "" : "layout(location = 0) out highp vec4 pc_fragColor;",
		n.glslVersion === "300 es" ? "" : "#define gl_FragColor pc_fragColor",
		"#define gl_FragDepthEXT gl_FragDepth",
		"#define texture2D texture",
		"#define textureCube texture",
		"#define texture2DProj textureProj",
		"#define texture2DLodEXT textureLod",
		"#define texture2DProjLodEXT textureProjLod",
		"#define textureCubeLodEXT textureLod",
		"#define texture2DGradEXT textureGrad",
		"#define texture2DProjGradEXT textureProjGrad",
		"#define textureCubeGradEXT textureGrad"
	].join("\n") + "\n" + _);
	let y = v + g + o, b = v + _ + s, x = jv(i, i.VERTEX_SHADER, y), S = jv(i, i.FRAGMENT_SHADER, b);
	i.attachShader(h, x), i.attachShader(h, S), n.index0AttributeName === void 0 ? n.hasPositionAttribute === !0 && i.bindAttribLocation(h, 0, "position") : i.bindAttribLocation(h, 0, n.index0AttributeName), i.linkProgram(h);
	function C(t) {
		if (e.debug.checkShaderErrors) {
			let n = i.getProgramInfoLog(h) || "", r = i.getShaderInfoLog(x) || "", a = i.getShaderInfoLog(S) || "", o = n.trim(), s = r.trim(), c = a.trim(), l = !0, u = !0;
			if (i.getProgramParameter(h, i.LINK_STATUS) === !1) if (l = !1, typeof e.debug.onShaderError == "function") e.debug.onShaderError(i, h, x, S);
			else {
				let e = Lv(i, x, "vertex"), n = Lv(i, S, "fragment");
				K("WebGLProgram: Shader Error " + i.getError() + " - VALIDATE_STATUS " + i.getProgramParameter(h, i.VALIDATE_STATUS) + "\n\nMaterial Name: " + t.name + "\nMaterial Type: " + t.type + "\n\nProgram Info Log: " + o + "\n" + e + "\n" + n);
			}
			else o === "" ? (s === "" || c === "") && (u = !1) : G("WebGLProgram: Program Info Log:", o);
			u && (t.diagnostics = {
				runnable: l,
				programLog: o,
				vertexShader: {
					log: s,
					prefix: g
				},
				fragmentShader: {
					log: c,
					prefix: _
				}
			});
		}
		i.deleteShader(x), i.deleteShader(S), w = new Av(i, h), T = Gv(i, h);
	}
	let w;
	this.getUniforms = function() {
		return w === void 0 && C(this), w;
	};
	let T;
	this.getAttributes = function() {
		return T === void 0 && C(this), T;
	};
	let E = n.rendererExtensionParallelShaderCompile === !1;
	return this.isReady = function() {
		return E === !1 && (E = i.getProgramParameter(h, Mv)), E;
	}, this.destroy = function() {
		r.releaseStatesOfProgram(this), i.deleteProgram(h), this.program = void 0;
	}, this.type = n.shaderType, this.name = n.shaderName, this.id = Nv++, this.cacheKey = t, this.usedTimes = 1, this.program = h, this.vertexShader = x, this.fragmentShader = S, this;
}
var py = 0, my = class {
	constructor() {
		this.shaderCache = /* @__PURE__ */ new Map(), this.materialCache = /* @__PURE__ */ new Map();
	}
	update(e, t, n) {
		let r = this._getShaderCacheForMaterial(e);
		return r.has(t) === !1 && (r.add(t), t.usedTimes++), r.has(n) === !1 && (r.add(n), n.usedTimes++), this;
	}
	remove(e) {
		let t = this.materialCache.get(e);
		for (let e of t) e.usedTimes--, e.usedTimes === 0 && this.shaderCache.delete(e.code);
		return this.materialCache.delete(e), this;
	}
	getVertexShaderStage(e) {
		return this._getShaderStage(e.vertexShader);
	}
	getFragmentShaderStage(e) {
		return this._getShaderStage(e.fragmentShader);
	}
	dispose() {
		this.shaderCache.clear(), this.materialCache.clear();
	}
	_getShaderCacheForMaterial(e) {
		let t = this.materialCache, n = t.get(e);
		return n === void 0 && (n = /* @__PURE__ */ new Set(), t.set(e, n)), n;
	}
	_getShaderStage(e) {
		let t = this.shaderCache, n = t.get(e);
		return n === void 0 && (n = new hy(e), t.set(e, n)), n;
	}
}, hy = class {
	constructor(e) {
		this.id = py++, this.code = e, this.usedTimes = 0;
	}
};
function gy(e) {
	return e === 1030 || e === 37490 || e === 36285;
}
function _y(e, t, n, r, i, a) {
	let o = new ap(), s = new my(), c = /* @__PURE__ */ new Set(), l = [], u = /* @__PURE__ */ new Map(), d = r.logarithmicDepthBuffer, f = r.precision, p = {
		MeshDepthMaterial: "depth",
		MeshDistanceMaterial: "distance",
		MeshNormalMaterial: "normal",
		MeshBasicMaterial: "basic",
		MeshLambertMaterial: "lambert",
		MeshPhongMaterial: "phong",
		MeshToonMaterial: "toon",
		MeshStandardMaterial: "physical",
		MeshPhysicalMaterial: "physical",
		MeshMatcapMaterial: "matcap",
		LineBasicMaterial: "basic",
		LineDashedMaterial: "dashed",
		PointsMaterial: "points",
		ShadowMaterial: "shadow",
		SpriteMaterial: "sprite"
	};
	function m(e) {
		return c.add(e), e === 0 ? "uv" : `uv${e}`;
	}
	function h(i, o, l, u, h, g) {
		let _ = u.fog, v = h.geometry, y = i.isMeshStandardMaterial || i.isMeshLambertMaterial || i.isMeshPhongMaterial ? u.environment : null, b = i.isMeshStandardMaterial || i.isMeshLambertMaterial && !i.envMap || i.isMeshPhongMaterial && !i.envMap, x = t.get(i.envMap || y, b), S = x && x.mapping === 306 ? x.image.height : null, C = p[i.type];
		i.precision !== null && (f = r.getMaxPrecision(i.precision), f !== i.precision && G("WebGLProgram.getParameters:", i.precision, "not supported, using", f, "instead."));
		let w = v.morphAttributes.position || v.morphAttributes.normal || v.morphAttributes.color, T = w === void 0 ? 0 : w.length, E = 0;
		v.morphAttributes.position !== void 0 && (E = 1), v.morphAttributes.normal !== void 0 && (E = 2), v.morphAttributes.color !== void 0 && (E = 3);
		let D, O, k, A;
		if (C) {
			let e = zg[C];
			D = e.vertexShader, O = e.fragmentShader;
		} else {
			D = i.vertexShader, O = i.fragmentShader;
			let e = s.getVertexShaderStage(i), t = s.getFragmentShaderStage(i);
			s.update(i, e, t), k = e.id, A = t.id;
		}
		let ee = e.getRenderTarget(), te = e.state.buffers.depth.getReversed(), j = h.isInstancedMesh === !0, ne = h.isBatchedMesh === !0, re = !!i.map, ie = !!i.matcap, ae = !!x, oe = !!i.aoMap, se = !!i.lightMap, ce = !!i.bumpMap && i.wireframe === !1, le = !!i.normalMap, ue = !!i.displacementMap, de = !!i.emissiveMap, fe = !!i.metalnessMap, pe = !!i.roughnessMap, me = i.anisotropy > 0, he = i.clearcoat > 0, ge = i.dispersion > 0, _e = i.iridescence > 0, ve = i.sheen > 0, ye = i.transmission > 0, be = me && !!i.anisotropyMap, xe = he && !!i.clearcoatMap, Se = he && !!i.clearcoatNormalMap, Ce = he && !!i.clearcoatRoughnessMap, we = _e && !!i.iridescenceMap, Te = _e && !!i.iridescenceThicknessMap, M = ve && !!i.sheenColorMap, Ee = ve && !!i.sheenRoughnessMap, De = !!i.specularMap, Oe = !!i.specularColorMap, N = !!i.specularIntensityMap, ke = ye && !!i.transmissionMap, P = ye && !!i.thicknessMap, F = !!i.gradientMap, Ae = !!i.alphaMap, je = i.alphaTest > 0, Me = !!i.alphaHash, Ne = !!i.extensions, Pe = 0;
		i.toneMapped && (ee === null || ee.isXRRenderTarget === !0) && (Pe = e.toneMapping);
		let Fe = {
			shaderID: C,
			shaderType: i.type,
			shaderName: i.name,
			vertexShader: D,
			fragmentShader: O,
			defines: i.defines,
			customVertexShaderID: k,
			customFragmentShaderID: A,
			isRawShaderMaterial: i.isRawShaderMaterial === !0,
			glslVersion: i.glslVersion,
			precision: f,
			batching: ne,
			batchingColor: ne && h._colorsTexture !== null,
			instancing: j,
			instancingColor: j && h.instanceColor !== null,
			instancingMorph: j && h.morphTexture !== null,
			outputColorSpace: ee === null ? e.outputColorSpace : ee.isXRRenderTarget === !0 ? ee.texture.colorSpace : X.workingColorSpace,
			alphaToCoverage: !!i.alphaToCoverage,
			map: re,
			matcap: ie,
			envMap: ae,
			envMapMode: ae && x.mapping,
			envMapCubeUVHeight: S,
			aoMap: oe,
			lightMap: se,
			bumpMap: ce,
			normalMap: le,
			displacementMap: ue,
			emissiveMap: de,
			normalMapObjectSpace: le && i.normalMapType === 1,
			normalMapTangentSpace: le && i.normalMapType === 0,
			packedNormalMap: le && i.normalMapType === 0 && gy(i.normalMap.format),
			metalnessMap: fe,
			roughnessMap: pe,
			anisotropy: me,
			anisotropyMap: be,
			clearcoat: he,
			clearcoatMap: xe,
			clearcoatNormalMap: Se,
			clearcoatRoughnessMap: Ce,
			dispersion: ge,
			iridescence: _e,
			iridescenceMap: we,
			iridescenceThicknessMap: Te,
			sheen: ve,
			sheenColorMap: M,
			sheenRoughnessMap: Ee,
			specularMap: De,
			specularColorMap: Oe,
			specularIntensityMap: N,
			transmission: ye,
			transmissionMap: ke,
			thicknessMap: P,
			gradientMap: F,
			opaque: i.transparent === !1 && i.blending === 1 && i.alphaToCoverage === !1,
			alphaMap: Ae,
			alphaTest: je,
			alphaHash: Me,
			combine: i.combine,
			mapUv: re && m(i.map.channel),
			aoMapUv: oe && m(i.aoMap.channel),
			lightMapUv: se && m(i.lightMap.channel),
			bumpMapUv: ce && m(i.bumpMap.channel),
			normalMapUv: le && m(i.normalMap.channel),
			displacementMapUv: ue && m(i.displacementMap.channel),
			emissiveMapUv: de && m(i.emissiveMap.channel),
			metalnessMapUv: fe && m(i.metalnessMap.channel),
			roughnessMapUv: pe && m(i.roughnessMap.channel),
			anisotropyMapUv: be && m(i.anisotropyMap.channel),
			clearcoatMapUv: xe && m(i.clearcoatMap.channel),
			clearcoatNormalMapUv: Se && m(i.clearcoatNormalMap.channel),
			clearcoatRoughnessMapUv: Ce && m(i.clearcoatRoughnessMap.channel),
			iridescenceMapUv: we && m(i.iridescenceMap.channel),
			iridescenceThicknessMapUv: Te && m(i.iridescenceThicknessMap.channel),
			sheenColorMapUv: M && m(i.sheenColorMap.channel),
			sheenRoughnessMapUv: Ee && m(i.sheenRoughnessMap.channel),
			specularMapUv: De && m(i.specularMap.channel),
			specularColorMapUv: Oe && m(i.specularColorMap.channel),
			specularIntensityMapUv: N && m(i.specularIntensityMap.channel),
			transmissionMapUv: ke && m(i.transmissionMap.channel),
			thicknessMapUv: P && m(i.thicknessMap.channel),
			alphaMapUv: Ae && m(i.alphaMap.channel),
			vertexTangents: !!v.attributes.tangent && (le || me),
			vertexNormals: !!v.attributes.normal,
			vertexColors: i.vertexColors,
			vertexAlphas: i.vertexColors === !0 && !!v.attributes.color && v.attributes.color.itemSize === 4,
			pointsUvs: h.isPoints === !0 && !!v.attributes.uv && (re || Ae),
			fog: !!_,
			useFog: i.fog === !0,
			fogExp2: !!_ && _.isFogExp2,
			flatShading: i.wireframe === !1 && (i.flatShading === !0 || v.attributes.normal === void 0 && le === !1 && (i.isMeshLambertMaterial || i.isMeshPhongMaterial || i.isMeshStandardMaterial || i.isMeshPhysicalMaterial)),
			sizeAttenuation: i.sizeAttenuation === !0,
			logarithmicDepthBuffer: d,
			reversedDepthBuffer: te,
			skinning: h.isSkinnedMesh === !0,
			hasPositionAttribute: v.attributes.position !== void 0,
			morphTargets: v.morphAttributes.position !== void 0,
			morphNormals: v.morphAttributes.normal !== void 0,
			morphColors: v.morphAttributes.color !== void 0,
			morphTargetsCount: T,
			morphTextureStride: E,
			numDirLights: o.directional.length,
			numPointLights: o.point.length,
			numSpotLights: o.spot.length,
			numSpotLightMaps: o.spotLightMap.length,
			numRectAreaLights: o.rectArea.length,
			numHemiLights: o.hemi.length,
			numDirLightShadows: o.directionalShadowMap.length,
			numPointLightShadows: o.pointShadowMap.length,
			numSpotLightShadows: o.spotShadowMap.length,
			numSpotLightShadowsWithMaps: o.numSpotLightShadowsWithMaps,
			numLightProbes: o.numLightProbes,
			numLightProbeGrids: g.length,
			numClippingPlanes: a.numPlanes,
			numClipIntersection: a.numIntersection,
			dithering: i.dithering,
			shadowMapEnabled: e.shadowMap.enabled && l.length > 0,
			shadowMapType: e.shadowMap.type,
			toneMapping: Pe,
			decodeVideoTexture: re && i.map.isVideoTexture === !0 && X.getTransfer(i.map.colorSpace) === "srgb",
			decodeVideoTextureEmissive: de && i.emissiveMap.isVideoTexture === !0 && X.getTransfer(i.emissiveMap.colorSpace) === "srgb",
			premultipliedAlpha: i.premultipliedAlpha,
			doubleSided: i.side === 2,
			flipSided: i.side === 1,
			useDepthPacking: i.depthPacking >= 0,
			depthPacking: i.depthPacking || 0,
			index0AttributeName: i.index0AttributeName,
			extensionClipCullDistance: Ne && i.extensions.clipCullDistance === !0 && n.has("WEBGL_clip_cull_distance"),
			extensionMultiDraw: (Ne && i.extensions.multiDraw === !0 || ne) && n.has("WEBGL_multi_draw"),
			rendererExtensionParallelShaderCompile: n.has("KHR_parallel_shader_compile"),
			customProgramCacheKey: i.customProgramCacheKey()
		};
		return Fe.vertexUv1s = c.has(1), Fe.vertexUv2s = c.has(2), Fe.vertexUv3s = c.has(3), c.clear(), Fe;
	}
	function g(t) {
		let n = [];
		if (t.shaderID ? n.push(t.shaderID) : (n.push(t.customVertexShaderID), n.push(t.customFragmentShaderID)), t.defines !== void 0) for (let e in t.defines) n.push(e), n.push(t.defines[e]);
		return t.isRawShaderMaterial === !1 && (_(n, t), v(n, t), n.push(e.outputColorSpace)), n.push(t.customProgramCacheKey), n.join();
	}
	function _(e, t) {
		e.push(t.precision), e.push(t.outputColorSpace), e.push(t.envMapMode), e.push(t.envMapCubeUVHeight), e.push(t.mapUv), e.push(t.alphaMapUv), e.push(t.lightMapUv), e.push(t.aoMapUv), e.push(t.bumpMapUv), e.push(t.normalMapUv), e.push(t.displacementMapUv), e.push(t.emissiveMapUv), e.push(t.metalnessMapUv), e.push(t.roughnessMapUv), e.push(t.anisotropyMapUv), e.push(t.clearcoatMapUv), e.push(t.clearcoatNormalMapUv), e.push(t.clearcoatRoughnessMapUv), e.push(t.iridescenceMapUv), e.push(t.iridescenceThicknessMapUv), e.push(t.sheenColorMapUv), e.push(t.sheenRoughnessMapUv), e.push(t.specularMapUv), e.push(t.specularColorMapUv), e.push(t.specularIntensityMapUv), e.push(t.transmissionMapUv), e.push(t.thicknessMapUv), e.push(t.combine), e.push(t.fogExp2), e.push(t.sizeAttenuation), e.push(t.morphTargetsCount), e.push(t.morphAttributeCount), e.push(t.numDirLights), e.push(t.numPointLights), e.push(t.numSpotLights), e.push(t.numSpotLightMaps), e.push(t.numHemiLights), e.push(t.numRectAreaLights), e.push(t.numDirLightShadows), e.push(t.numPointLightShadows), e.push(t.numSpotLightShadows), e.push(t.numSpotLightShadowsWithMaps), e.push(t.numLightProbes), e.push(t.shadowMapType), e.push(t.toneMapping), e.push(t.numClippingPlanes), e.push(t.numClipIntersection), e.push(t.depthPacking);
	}
	function v(e, t) {
		o.disableAll(), t.instancing && o.enable(0), t.instancingColor && o.enable(1), t.instancingMorph && o.enable(2), t.matcap && o.enable(3), t.envMap && o.enable(4), t.normalMapObjectSpace && o.enable(5), t.normalMapTangentSpace && o.enable(6), t.clearcoat && o.enable(7), t.iridescence && o.enable(8), t.alphaTest && o.enable(9), t.vertexColors && o.enable(10), t.vertexAlphas && o.enable(11), t.vertexUv1s && o.enable(12), t.vertexUv2s && o.enable(13), t.vertexUv3s && o.enable(14), t.vertexTangents && o.enable(15), t.anisotropy && o.enable(16), t.alphaHash && o.enable(17), t.batching && o.enable(18), t.dispersion && o.enable(19), t.batchingColor && o.enable(20), t.gradientMap && o.enable(21), t.packedNormalMap && o.enable(22), t.vertexNormals && o.enable(23), e.push(o.mask), o.disableAll(), t.fog && o.enable(0), t.useFog && o.enable(1), t.flatShading && o.enable(2), t.logarithmicDepthBuffer && o.enable(3), t.reversedDepthBuffer && o.enable(4), t.skinning && o.enable(5), t.morphTargets && o.enable(6), t.morphNormals && o.enable(7), t.morphColors && o.enable(8), t.premultipliedAlpha && o.enable(9), t.shadowMapEnabled && o.enable(10), t.doubleSided && o.enable(11), t.flipSided && o.enable(12), t.useDepthPacking && o.enable(13), t.dithering && o.enable(14), t.transmission && o.enable(15), t.sheen && o.enable(16), t.opaque && o.enable(17), t.pointsUvs && o.enable(18), t.decodeVideoTexture && o.enable(19), t.decodeVideoTextureEmissive && o.enable(20), t.alphaToCoverage && o.enable(21), t.numLightProbeGrids > 0 && o.enable(22), t.hasPositionAttribute && o.enable(23), e.push(o.mask);
	}
	function y(e) {
		let t = p[e.type], n;
		if (t) {
			let e = zg[t];
			n = Lh.clone(e.uniforms);
		} else n = e.uniforms;
		return n;
	}
	function b(t, n) {
		let r = u.get(n);
		return r === void 0 ? (r = new fy(e, n, t, i), l.push(r), u.set(n, r)) : ++r.usedTimes, r;
	}
	function x(e) {
		if (--e.usedTimes === 0) {
			let t = l.indexOf(e);
			l[t] = l[l.length - 1], l.pop(), u.delete(e.cacheKey), e.destroy();
		}
	}
	function S(e) {
		s.remove(e);
	}
	function C() {
		s.dispose();
	}
	return {
		getParameters: h,
		getProgramCacheKey: g,
		getUniforms: y,
		acquireProgram: b,
		releaseProgram: x,
		releaseShaderCache: S,
		programs: l,
		dispose: C
	};
}
function vy() {
	let e = /* @__PURE__ */ new WeakMap();
	function t(t) {
		return e.has(t);
	}
	function n(t) {
		let n = e.get(t);
		return n === void 0 && (n = {}, e.set(t, n)), n;
	}
	function r(t) {
		e.delete(t);
	}
	function i(t, n, r) {
		e.get(t)[n] = r;
	}
	function a() {
		e = /* @__PURE__ */ new WeakMap();
	}
	return {
		has: t,
		get: n,
		remove: r,
		update: i,
		dispose: a
	};
}
function yy(e, t) {
	return e.groupOrder === t.groupOrder ? e.renderOrder === t.renderOrder ? e.material.id === t.material.id ? e.materialVariant === t.materialVariant ? e.z === t.z ? e.id - t.id : e.z - t.z : e.materialVariant - t.materialVariant : e.material.id - t.material.id : e.renderOrder - t.renderOrder : e.groupOrder - t.groupOrder;
}
function by(e, t) {
	return e.groupOrder === t.groupOrder ? e.renderOrder === t.renderOrder ? e.z === t.z ? e.id - t.id : t.z - e.z : e.renderOrder - t.renderOrder : e.groupOrder - t.groupOrder;
}
function xy() {
	let e = [], t = 0, n = [], r = [], i = [];
	function a() {
		t = 0, n.length = 0, r.length = 0, i.length = 0;
	}
	function o(e) {
		let t = 0;
		return e.isInstancedMesh && (t += 2), e.isSkinnedMesh && (t += 1), t;
	}
	function s(n, r, i, a, s, c) {
		let l = e[t];
		return l === void 0 ? (l = {
			id: n.id,
			object: n,
			geometry: r,
			material: i,
			materialVariant: o(n),
			groupOrder: a,
			renderOrder: n.renderOrder,
			z: s,
			group: c
		}, e[t] = l) : (l.id = n.id, l.object = n, l.geometry = r, l.material = i, l.materialVariant = o(n), l.groupOrder = a, l.renderOrder = n.renderOrder, l.z = s, l.group = c), t++, l;
	}
	function c(e, t, a, o, c, l) {
		let u = s(e, t, a, o, c, l);
		a.transmission > 0 ? r.push(u) : a.transparent === !0 ? i.push(u) : n.push(u);
	}
	function l(e, t, a, o, c, l) {
		let u = s(e, t, a, o, c, l);
		a.transmission > 0 ? r.unshift(u) : a.transparent === !0 ? i.unshift(u) : n.unshift(u);
	}
	function u(e, t, a) {
		n.length > 1 && n.sort(e || yy), r.length > 1 && r.sort(t || by), i.length > 1 && i.sort(t || by), a && (n.reverse(), r.reverse(), i.reverse());
	}
	function d() {
		for (let n = t, r = e.length; n < r; n++) {
			let t = e[n];
			if (t.id === null) break;
			t.id = null, t.object = null, t.geometry = null, t.material = null, t.group = null;
		}
	}
	return {
		opaque: n,
		transmissive: r,
		transparent: i,
		init: a,
		push: c,
		unshift: l,
		finish: d,
		sort: u
	};
}
function Sy() {
	let e = /* @__PURE__ */ new WeakMap();
	function t(t, n) {
		let r = e.get(t), i;
		return r === void 0 ? (i = new xy(), e.set(t, [i])) : n >= r.length ? (i = new xy(), r.push(i)) : i = r[n], i;
	}
	function n() {
		e = /* @__PURE__ */ new WeakMap();
	}
	return {
		get: t,
		dispose: n
	};
}
function Cy() {
	let e = {};
	return { get: function(t) {
		if (e[t.id] !== void 0) return e[t.id];
		let n;
		switch (t.type) {
			case "DirectionalLight":
				n = {
					direction: new J(),
					color: new Z()
				};
				break;
			case "SpotLight":
				n = {
					position: new J(),
					direction: new J(),
					color: new Z(),
					distance: 0,
					coneCos: 0,
					penumbraCos: 0,
					decay: 0
				};
				break;
			case "PointLight":
				n = {
					position: new J(),
					color: new Z(),
					distance: 0,
					decay: 0
				};
				break;
			case "HemisphereLight":
				n = {
					direction: new J(),
					skyColor: new Z(),
					groundColor: new Z()
				};
				break;
			case "RectAreaLight":
				n = {
					color: new Z(),
					position: new J(),
					halfWidth: new J(),
					halfHeight: new J()
				};
				break;
		}
		return e[t.id] = n, n;
	} };
}
function wy() {
	let e = {};
	return { get: function(t) {
		if (e[t.id] !== void 0) return e[t.id];
		let n;
		switch (t.type) {
			case "DirectionalLight":
				n = {
					shadowIntensity: 1,
					shadowBias: 0,
					shadowNormalBias: 0,
					shadowRadius: 1,
					shadowMapSize: new Tf()
				};
				break;
			case "SpotLight":
				n = {
					shadowIntensity: 1,
					shadowBias: 0,
					shadowNormalBias: 0,
					shadowRadius: 1,
					shadowMapSize: new Tf()
				};
				break;
			case "PointLight":
				n = {
					shadowIntensity: 1,
					shadowBias: 0,
					shadowNormalBias: 0,
					shadowRadius: 1,
					shadowMapSize: new Tf(),
					shadowCameraNear: 1,
					shadowCameraFar: 1e3
				};
				break;
		}
		return e[t.id] = n, n;
	} };
}
var Ty = 0;
function Ey(e, t) {
	return (t.castShadow ? 2 : 0) - (e.castShadow ? 2 : 0) + +!!t.map - !!e.map;
}
function Dy(e) {
	let t = new Cy(), n = wy(), r = {
		version: 0,
		hash: {
			directionalLength: -1,
			pointLength: -1,
			spotLength: -1,
			rectAreaLength: -1,
			hemiLength: -1,
			numDirectionalShadows: -1,
			numPointShadows: -1,
			numSpotShadows: -1,
			numSpotMaps: -1,
			numLightProbes: -1
		},
		ambient: [
			0,
			0,
			0
		],
		probe: [],
		directional: [],
		directionalShadow: [],
		directionalShadowMap: [],
		directionalShadowMatrix: [],
		spot: [],
		spotLightMap: [],
		spotShadow: [],
		spotShadowMap: [],
		spotLightMatrix: [],
		rectArea: [],
		rectAreaLTC1: null,
		rectAreaLTC2: null,
		point: [],
		pointShadow: [],
		pointShadowMap: [],
		pointShadowMatrix: [],
		hemi: [],
		numSpotLightShadowsWithMaps: 0,
		numLightProbes: 0
	};
	for (let e = 0; e < 9; e++) r.probe.push(new J());
	let i = new J(), a = new Jf(), o = new Jf();
	function s(i) {
		let a = 0, o = 0, s = 0;
		for (let e = 0; e < 9; e++) r.probe[e].set(0, 0, 0);
		let c = 0, l = 0, u = 0, d = 0, f = 0, p = 0, m = 0, h = 0, g = 0, _ = 0, v = 0;
		i.sort(Ey);
		for (let e = 0, y = i.length; e < y; e++) {
			let y = i[e], b = y.color, x = y.intensity, S = y.distance, C = null;
			if (y.shadow && y.shadow.map && (C = y.shadow.map.texture.format === 1030 ? y.shadow.map.texture : y.shadow.map.depthTexture || y.shadow.map.texture), y.isAmbientLight) a += b.r * x, o += b.g * x, s += b.b * x;
			else if (y.isLightProbe) {
				for (let e = 0; e < 9; e++) r.probe[e].addScaledVector(y.sh.coefficients[e], x);
				v++;
			} else if (y.isDirectionalLight) {
				let e = t.get(y);
				if (e.color.copy(y.color).multiplyScalar(y.intensity), y.castShadow) {
					let e = y.shadow, t = n.get(y);
					t.shadowIntensity = e.intensity, t.shadowBias = e.bias, t.shadowNormalBias = e.normalBias, t.shadowRadius = e.radius, t.shadowMapSize = e.mapSize, r.directionalShadow[c] = t, r.directionalShadowMap[c] = C, r.directionalShadowMatrix[c] = y.shadow.matrix, p++;
				}
				r.directional[c] = e, c++;
			} else if (y.isSpotLight) {
				let e = t.get(y);
				e.position.setFromMatrixPosition(y.matrixWorld), e.color.copy(b).multiplyScalar(x), e.distance = S, e.coneCos = Math.cos(y.angle), e.penumbraCos = Math.cos(y.angle * (1 - y.penumbra)), e.decay = y.decay, r.spot[u] = e;
				let i = y.shadow;
				if (y.map && (r.spotLightMap[g] = y.map, g++, i.updateMatrices(y), y.castShadow && _++), r.spotLightMatrix[u] = i.matrix, y.castShadow) {
					let e = n.get(y);
					e.shadowIntensity = i.intensity, e.shadowBias = i.bias, e.shadowNormalBias = i.normalBias, e.shadowRadius = i.radius, e.shadowMapSize = i.mapSize, r.spotShadow[u] = e, r.spotShadowMap[u] = C, h++;
				}
				u++;
			} else if (y.isRectAreaLight) {
				let e = t.get(y);
				e.color.copy(b).multiplyScalar(x), e.halfWidth.set(y.width * .5, 0, 0), e.halfHeight.set(0, y.height * .5, 0), r.rectArea[d] = e, d++;
			} else if (y.isPointLight) {
				let e = t.get(y);
				if (e.color.copy(y.color).multiplyScalar(y.intensity), e.distance = y.distance, e.decay = y.decay, y.castShadow) {
					let e = y.shadow, t = n.get(y);
					t.shadowIntensity = e.intensity, t.shadowBias = e.bias, t.shadowNormalBias = e.normalBias, t.shadowRadius = e.radius, t.shadowMapSize = e.mapSize, t.shadowCameraNear = e.camera.near, t.shadowCameraFar = e.camera.far, r.pointShadow[l] = t, r.pointShadowMap[l] = C, r.pointShadowMatrix[l] = y.shadow.matrix, m++;
				}
				r.point[l] = e, l++;
			} else if (y.isHemisphereLight) {
				let e = t.get(y);
				e.skyColor.copy(y.color).multiplyScalar(x), e.groundColor.copy(y.groundColor).multiplyScalar(x), r.hemi[f] = e, f++;
			}
		}
		d > 0 && (e.has("OES_texture_float_linear") === !0 ? (r.rectAreaLTC1 = $.LTC_FLOAT_1, r.rectAreaLTC2 = $.LTC_FLOAT_2) : (r.rectAreaLTC1 = $.LTC_HALF_1, r.rectAreaLTC2 = $.LTC_HALF_2)), r.ambient[0] = a, r.ambient[1] = o, r.ambient[2] = s;
		let y = r.hash;
		(y.directionalLength !== c || y.pointLength !== l || y.spotLength !== u || y.rectAreaLength !== d || y.hemiLength !== f || y.numDirectionalShadows !== p || y.numPointShadows !== m || y.numSpotShadows !== h || y.numSpotMaps !== g || y.numLightProbes !== v) && (r.directional.length = c, r.spot.length = u, r.rectArea.length = d, r.point.length = l, r.hemi.length = f, r.directionalShadow.length = p, r.directionalShadowMap.length = p, r.pointShadow.length = m, r.pointShadowMap.length = m, r.spotShadow.length = h, r.spotShadowMap.length = h, r.directionalShadowMatrix.length = p, r.pointShadowMatrix.length = m, r.spotLightMatrix.length = h + g - _, r.spotLightMap.length = g, r.numSpotLightShadowsWithMaps = _, r.numLightProbes = v, y.directionalLength = c, y.pointLength = l, y.spotLength = u, y.rectAreaLength = d, y.hemiLength = f, y.numDirectionalShadows = p, y.numPointShadows = m, y.numSpotShadows = h, y.numSpotMaps = g, y.numLightProbes = v, r.version = Ty++);
	}
	function c(e, t) {
		let n = 0, s = 0, c = 0, l = 0, u = 0, d = t.matrixWorldInverse;
		for (let t = 0, f = e.length; t < f; t++) {
			let f = e[t];
			if (f.isDirectionalLight) {
				let e = r.directional[n];
				e.direction.setFromMatrixPosition(f.matrixWorld), i.setFromMatrixPosition(f.target.matrixWorld), e.direction.sub(i), e.direction.transformDirection(d), n++;
			} else if (f.isSpotLight) {
				let e = r.spot[c];
				e.position.setFromMatrixPosition(f.matrixWorld), e.position.applyMatrix4(d), e.direction.setFromMatrixPosition(f.matrixWorld), i.setFromMatrixPosition(f.target.matrixWorld), e.direction.sub(i), e.direction.transformDirection(d), c++;
			} else if (f.isRectAreaLight) {
				let e = r.rectArea[l];
				e.position.setFromMatrixPosition(f.matrixWorld), e.position.applyMatrix4(d), o.identity(), a.copy(f.matrixWorld), a.premultiply(d), o.extractRotation(a), e.halfWidth.set(f.width * .5, 0, 0), e.halfHeight.set(0, f.height * .5, 0), e.halfWidth.applyMatrix4(o), e.halfHeight.applyMatrix4(o), l++;
			} else if (f.isPointLight) {
				let e = r.point[s];
				e.position.setFromMatrixPosition(f.matrixWorld), e.position.applyMatrix4(d), s++;
			} else if (f.isHemisphereLight) {
				let e = r.hemi[u];
				e.direction.setFromMatrixPosition(f.matrixWorld), e.direction.transformDirection(d), u++;
			}
		}
	}
	return {
		setup: s,
		setupView: c,
		state: r
	};
}
function Oy(e) {
	let t = new Dy(e), n = [], r = [], i = [];
	function a(e) {
		d.camera = e, n.length = 0, r.length = 0, i.length = 0;
	}
	function o(e) {
		n.push(e);
	}
	function s(e) {
		r.push(e);
	}
	function c(e) {
		i.push(e);
	}
	function l() {
		t.setup(n);
	}
	function u(e) {
		t.setupView(n, e);
	}
	let d = {
		lightsArray: n,
		shadowsArray: r,
		lightProbeGridArray: i,
		camera: null,
		lights: t,
		transmissionRenderTarget: {},
		textureUnits: 0
	};
	return {
		init: a,
		state: d,
		setupLights: l,
		setupLightsView: u,
		pushLight: o,
		pushShadow: s,
		pushLightProbeGrid: c
	};
}
function ky(e) {
	let t = /* @__PURE__ */ new WeakMap();
	function n(n, r = 0) {
		let i = t.get(n), a;
		return i === void 0 ? (a = new Oy(e), t.set(n, [a])) : r >= i.length ? (a = new Oy(e), i.push(a)) : a = i[r], a;
	}
	function r() {
		t = /* @__PURE__ */ new WeakMap();
	}
	return {
		get: n,
		dispose: r
	};
}
var Ay = "void main() {\n	gl_Position = vec4( position, 1.0 );\n}", jy = "uniform sampler2D shadow_pass;\nuniform vec2 resolution;\nuniform float radius;\nvoid main() {\n	const float samples = float( VSM_SAMPLES );\n	float mean = 0.0;\n	float squared_mean = 0.0;\n	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );\n	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;\n	for ( float i = 0.0; i < samples; i ++ ) {\n		float uvOffset = uvStart + i * uvStride;\n		#ifdef HORIZONTAL_PASS\n			vec2 distribution = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ).rg;\n			mean += distribution.x;\n			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;\n		#else\n			float depth = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ).r;\n			mean += depth;\n			squared_mean += depth * depth;\n		#endif\n	}\n	mean = mean / samples;\n	squared_mean = squared_mean / samples;\n	float std_dev = sqrt( max( 0.0, squared_mean - mean * mean ) );\n	gl_FragColor = vec4( mean, std_dev, 0.0, 1.0 );\n}", My = [
	/*@__PURE__*/ new J(1, 0, 0),
	/*@__PURE__*/ new J(-1, 0, 0),
	/*@__PURE__*/ new J(0, 1, 0),
	/*@__PURE__*/ new J(0, -1, 0),
	/*@__PURE__*/ new J(0, 0, 1),
	/*@__PURE__*/ new J(0, 0, -1)
], Ny = [
	/*@__PURE__*/ new J(0, -1, 0),
	/*@__PURE__*/ new J(0, -1, 0),
	/*@__PURE__*/ new J(0, 0, 1),
	/*@__PURE__*/ new J(0, 0, -1),
	/*@__PURE__*/ new J(0, -1, 0),
	/*@__PURE__*/ new J(0, -1, 0)
], Py = /*@__PURE__*/ new Jf(), Fy = /*@__PURE__*/ new J(), Iy = /*@__PURE__*/ new J();
function Ly(e, t, n) {
	let r = new ih(), i = new Tf(), a = new Tf(), o = new Uf(), s = new Hh(), c = new Uh(), l = {}, u = n.maxTextureSize, d = {
		0: 1,
		1: 0,
		2: 2
	}, f = new Bh({
		defines: { VSM_SAMPLES: 8 },
		uniforms: {
			shadow_pass: { value: null },
			resolution: { value: new Tf() },
			radius: { value: 4 }
		},
		vertexShader: Ay,
		fragmentShader: jy
	}), p = f.clone();
	p.defines.HORIZONTAL_PASS = 1;
	let m = new wm();
	m.setAttribute("position", new lm(new Float32Array([
		-1,
		-1,
		.5,
		3,
		-1,
		.5,
		-1,
		3,
		.5
	]), 3));
	let h = new qm(m, f), g = this;
	this.enabled = !1, this.autoUpdate = !0, this.needsUpdate = !1, this.type = 1;
	let _ = this.type;
	this.render = function(t, n, s) {
		if (g.enabled === !1 || g.autoUpdate === !1 && g.needsUpdate === !1 || t.length === 0) return;
		this.type === 2 && (G("WebGLShadowMap: PCFSoftShadowMap has been deprecated. Using PCFShadowMap instead."), this.type = 1);
		let c = e.getRenderTarget(), l = e.getActiveCubeFace(), d = e.getActiveMipmapLevel(), f = e.state;
		f.setBlending(0), f.buffers.depth.getReversed() === !0 ? f.buffers.color.setClear(0, 0, 0, 0) : f.buffers.color.setClear(1, 1, 1, 1), f.buffers.depth.setTest(!0), f.setScissorTest(!1);
		let p = _ !== this.type;
		p && n.traverse(function(e) {
			e.material && (Array.isArray(e.material) ? e.material.forEach((e) => e.needsUpdate = !0) : e.material.needsUpdate = !0);
		});
		for (let c = 0, l = t.length; c < l; c++) {
			let l = t[c], d = l.shadow;
			if (d === void 0) {
				G("WebGLShadowMap:", l, "has no shadow.");
				continue;
			}
			if (d.autoUpdate === !1 && d.needsUpdate === !1) continue;
			i.copy(d.mapSize);
			let m = d.getFrameExtents();
			i.multiply(m), a.copy(d.mapSize), (i.x > u || i.y > u) && (i.x > u && (a.x = Math.floor(u / m.x), i.x = a.x * m.x, d.mapSize.x = a.x), i.y > u && (a.y = Math.floor(u / m.y), i.y = a.y * m.y, d.mapSize.y = a.y));
			let h = e.state.buffers.depth.getReversed();
			if (d.camera._reversedDepth = h, d.map === null || p === !0) {
				if (d.map !== null && (d.map.depthTexture !== null && (d.map.depthTexture.dispose(), d.map.depthTexture = null), d.map.dispose()), this.type === 3) {
					if (l.isPointLight) {
						G("WebGLShadowMap: VSM shadow maps are not supported for PointLights. Use PCF or BasicShadowMap instead.");
						continue;
					}
					d.map = new Gf(i.x, i.y, {
						format: Bu,
						type: Du,
						minFilter: _u,
						magFilter: _u,
						generateMipmaps: !1
					}), d.map.texture.name = l.name + ".shadowMap", d.map.depthTexture = new Dh(i.x, i.y, Eu), d.map.depthTexture.name = l.name + ".shadowMapDepth", d.map.depthTexture.format = Iu, d.map.depthTexture.compareFunction = null, d.map.depthTexture.minFilter = mu, d.map.depthTexture.magFilter = mu;
				} else l.isPointLight ? (d.map = new m_(i.x), d.map.depthTexture = new Oh(i.x, Tu)) : (d.map = new Gf(i.x, i.y), d.map.depthTexture = new Dh(i.x, i.y, Tu)), d.map.depthTexture.name = l.name + ".shadowMap", d.map.depthTexture.format = Iu, this.type === 1 ? (d.map.depthTexture.compareFunction = h ? 518 : 515, d.map.depthTexture.minFilter = _u, d.map.depthTexture.magFilter = _u) : (d.map.depthTexture.compareFunction = null, d.map.depthTexture.minFilter = mu, d.map.depthTexture.magFilter = mu);
				d.camera.updateProjectionMatrix();
			}
			let g = d.map.isWebGLCubeRenderTarget ? 6 : 1;
			for (let t = 0; t < g; t++) {
				if (d.map.isWebGLCubeRenderTarget) e.setRenderTarget(d.map, t), e.clear();
				else {
					t === 0 && (e.setRenderTarget(d.map), e.clear());
					let n = d.getViewport(t);
					o.set(a.x * n.x, a.y * n.y, a.x * n.z, a.y * n.w), f.viewport(o);
				}
				if (l.isPointLight) {
					let e = d.camera, n = d.matrix, r = l.distance || e.far;
					r !== e.far && (e.far = r, e.updateProjectionMatrix()), Fy.setFromMatrixPosition(l.matrixWorld), e.position.copy(Fy), Iy.copy(e.position), Iy.add(My[t]), e.up.copy(Ny[t]), e.lookAt(Iy), e.updateMatrixWorld(), n.makeTranslation(-Fy.x, -Fy.y, -Fy.z), Py.multiplyMatrices(e.projectionMatrix, e.matrixWorldInverse), d._frustum.setFromProjectionMatrix(Py, e.coordinateSystem, e.reversedDepth);
				} else d.updateMatrices(l);
				r = d.getFrustum(), b(n, s, d.camera, l, this.type);
			}
			d.isPointLightShadow !== !0 && this.type === 3 && v(d, s), d.needsUpdate = !1;
		}
		_ = this.type, g.needsUpdate = !1, e.setRenderTarget(c, l, d);
	};
	function v(n, r) {
		let a = t.update(h);
		f.defines.VSM_SAMPLES !== n.blurSamples && (f.defines.VSM_SAMPLES = n.blurSamples, p.defines.VSM_SAMPLES = n.blurSamples, f.needsUpdate = !0, p.needsUpdate = !0), n.mapPass === null && (n.mapPass = new Gf(i.x, i.y, {
			format: Bu,
			type: Du
		})), f.uniforms.shadow_pass.value = n.map.depthTexture, f.uniforms.resolution.value = n.mapSize, f.uniforms.radius.value = n.radius, e.setRenderTarget(n.mapPass), e.clear(), e.renderBufferDirect(r, null, a, f, h, null), p.uniforms.shadow_pass.value = n.mapPass.texture, p.uniforms.resolution.value = n.mapSize, p.uniforms.radius.value = n.radius, e.setRenderTarget(n.map), e.clear(), e.renderBufferDirect(r, null, a, p, h, null);
	}
	function y(t, n, r, i) {
		let a = null, o = r.isPointLight === !0 ? t.customDistanceMaterial : t.customDepthMaterial;
		if (o !== void 0) a = o;
		else if (a = r.isPointLight === !0 ? c : s, e.localClippingEnabled && n.clipShadows === !0 && Array.isArray(n.clippingPlanes) && n.clippingPlanes.length !== 0 || n.displacementMap && n.displacementScale !== 0 || n.alphaMap && n.alphaTest > 0 || n.map && n.alphaTest > 0 || n.alphaToCoverage === !0) {
			let e = a.uuid, t = n.uuid, r = l[e];
			r === void 0 && (r = {}, l[e] = r);
			let i = r[t];
			i === void 0 && (i = a.clone(), r[t] = i, n.addEventListener("dispose", x)), a = i;
		}
		if (a.visible = n.visible, a.wireframe = n.wireframe, i === 3 ? a.side = n.shadowSide === null ? n.side : n.shadowSide : a.side = n.shadowSide === null ? d[n.side] : n.shadowSide, a.alphaMap = n.alphaMap, a.alphaTest = n.alphaToCoverage === !0 ? .5 : n.alphaTest, a.map = n.map, a.clipShadows = n.clipShadows, a.clippingPlanes = n.clippingPlanes, a.clipIntersection = n.clipIntersection, a.displacementMap = n.displacementMap, a.displacementScale = n.displacementScale, a.displacementBias = n.displacementBias, a.wireframeLinewidth = n.wireframeLinewidth, a.linewidth = n.linewidth, r.isPointLight === !0 && a.isMeshDistanceMaterial === !0) {
			let t = e.properties.get(a);
			t.light = r;
		}
		return a;
	}
	function b(n, i, a, o, s) {
		if (n.visible === !1) return;
		if (n.layers.test(i.layers) && (n.isMesh || n.isLine || n.isPoints) && (n.castShadow || n.receiveShadow && s === 3) && (!n.frustumCulled || r.intersectsObject(n))) {
			n.modelViewMatrix.multiplyMatrices(a.matrixWorldInverse, n.matrixWorld);
			let r = t.update(n), c = n.material;
			if (Array.isArray(c)) {
				let t = r.groups;
				for (let l = 0, u = t.length; l < u; l++) {
					let u = t[l], d = c[u.materialIndex];
					if (d && d.visible) {
						let t = y(n, d, o, s);
						n.onBeforeShadow(e, n, i, a, r, t, u), e.renderBufferDirect(a, null, r, t, n, u), n.onAfterShadow(e, n, i, a, r, t, u);
					}
				}
			} else if (c.visible) {
				let t = y(n, c, o, s);
				n.onBeforeShadow(e, n, i, a, r, t, null), e.renderBufferDirect(a, null, r, t, n, null), n.onAfterShadow(e, n, i, a, r, t, null);
			}
		}
		let c = n.children;
		for (let e = 0, t = c.length; e < t; e++) b(c[e], i, a, o, s);
	}
	function x(e) {
		e.target.removeEventListener("dispose", x);
		for (let t in l) {
			let n = l[t], r = e.target.uuid;
			r in n && (n[r].dispose(), delete n[r]);
		}
	}
}
function Ry(e, t) {
	function n() {
		let t = !1, n = new Uf(), r = null, i = new Uf(0, 0, 0, 0);
		return {
			setMask: function(n) {
				r !== n && !t && (e.colorMask(n, n, n, n), r = n);
			},
			setLocked: function(e) {
				t = e;
			},
			setClear: function(t, r, a, o, s) {
				s === !0 && (t *= o, r *= o, a *= o), n.set(t, r, a, o), i.equals(n) === !1 && (e.clearColor(t, r, a, o), i.copy(n));
			},
			reset: function() {
				t = !1, r = null, i.set(-1, 0, 0, 0);
			}
		};
	}
	function r() {
		let n = !1, r = !1, i = null, a = null, o = null;
		return {
			setReversed: function(e) {
				if (r !== e) {
					let n = t.get("EXT_clip_control");
					e ? n.clipControlEXT(n.LOWER_LEFT_EXT, n.ZERO_TO_ONE_EXT) : n.clipControlEXT(n.LOWER_LEFT_EXT, n.NEGATIVE_ONE_TO_ONE_EXT), r = e;
					let i = o;
					o = null, this.setClear(i);
				}
			},
			getReversed: function() {
				return r;
			},
			setTest: function(t) {
				t ? fe(e.DEPTH_TEST) : pe(e.DEPTH_TEST);
			},
			setMask: function(t) {
				i !== t && !n && (e.depthMask(t), i = t);
			},
			setFunc: function(t) {
				if (r && (t = Xd[t]), a !== t) {
					switch (t) {
						case 0:
							e.depthFunc(e.NEVER);
							break;
						case 1:
							e.depthFunc(e.ALWAYS);
							break;
						case 2:
							e.depthFunc(e.LESS);
							break;
						case 3:
							e.depthFunc(e.LEQUAL);
							break;
						case 4:
							e.depthFunc(e.EQUAL);
							break;
						case 5:
							e.depthFunc(e.GEQUAL);
							break;
						case 6:
							e.depthFunc(e.GREATER);
							break;
						case 7:
							e.depthFunc(e.NOTEQUAL);
							break;
						default: e.depthFunc(e.LEQUAL);
					}
					a = t;
				}
			},
			setLocked: function(e) {
				n = e;
			},
			setClear: function(t) {
				o !== t && (o = t, r && (t = 1 - t), e.clearDepth(t));
			},
			reset: function() {
				n = !1, i = null, a = null, o = null, r = !1;
			}
		};
	}
	function i() {
		let t = !1, n = null, r = null, i = null, a = null, o = null, s = null, c = null, l = null;
		return {
			setTest: function(n) {
				t || (n ? fe(e.STENCIL_TEST) : pe(e.STENCIL_TEST));
			},
			setMask: function(r) {
				n !== r && !t && (e.stencilMask(r), n = r);
			},
			setFunc: function(t, n, o) {
				(r !== t || i !== n || a !== o) && (e.stencilFunc(t, n, o), r = t, i = n, a = o);
			},
			setOp: function(t, n, r) {
				(o !== t || s !== n || c !== r) && (e.stencilOp(t, n, r), o = t, s = n, c = r);
			},
			setLocked: function(e) {
				t = e;
			},
			setClear: function(t) {
				l !== t && (e.clearStencil(t), l = t);
			},
			reset: function() {
				t = !1, n = null, r = null, i = null, a = null, o = null, s = null, c = null, l = null;
			}
		};
	}
	let a = new n(), o = new r(), s = new i(), c = /* @__PURE__ */ new WeakMap(), l = /* @__PURE__ */ new WeakMap(), u = {}, d = {}, f = {}, p = /* @__PURE__ */ new WeakMap(), m = [], h = null, g = !1, _ = null, v = null, y = null, b = null, x = null, S = null, C = null, w = new Z(0, 0, 0), T = 0, E = !1, D = null, O = null, k = null, A = null, ee = null, te = e.getParameter(e.MAX_COMBINED_TEXTURE_IMAGE_UNITS), j = !1, ne = 0, re = e.getParameter(e.VERSION);
	re.indexOf("WebGL") === -1 ? re.indexOf("OpenGL ES") !== -1 && (ne = parseFloat(/^OpenGL ES (\d)/.exec(re)[1]), j = ne >= 2) : (ne = parseFloat(/^WebGL (\d)/.exec(re)[1]), j = ne >= 1);
	let ie = null, ae = {}, oe = e.getParameter(e.SCISSOR_BOX), se = e.getParameter(e.VIEWPORT), ce = new Uf().fromArray(oe), le = new Uf().fromArray(se);
	function ue(t, n, r, i) {
		let a = /* @__PURE__ */ new Uint8Array(4), o = e.createTexture();
		e.bindTexture(t, o), e.texParameteri(t, e.TEXTURE_MIN_FILTER, e.NEAREST), e.texParameteri(t, e.TEXTURE_MAG_FILTER, e.NEAREST);
		for (let o = 0; o < r; o++) t === e.TEXTURE_3D || t === e.TEXTURE_2D_ARRAY ? e.texImage3D(n, 0, e.RGBA, 1, 1, i, 0, e.RGBA, e.UNSIGNED_BYTE, a) : e.texImage2D(n + o, 0, e.RGBA, 1, 1, 0, e.RGBA, e.UNSIGNED_BYTE, a);
		return o;
	}
	let de = {};
	de[e.TEXTURE_2D] = ue(e.TEXTURE_2D, e.TEXTURE_2D, 1), de[e.TEXTURE_CUBE_MAP] = ue(e.TEXTURE_CUBE_MAP, e.TEXTURE_CUBE_MAP_POSITIVE_X, 6), de[e.TEXTURE_2D_ARRAY] = ue(e.TEXTURE_2D_ARRAY, e.TEXTURE_2D_ARRAY, 1, 1), de[e.TEXTURE_3D] = ue(e.TEXTURE_3D, e.TEXTURE_3D, 1, 1), a.setClear(0, 0, 0, 1), o.setClear(1), s.setClear(0), fe(e.DEPTH_TEST), o.setFunc(3), xe(!1), Se(1), fe(e.CULL_FACE), ye(0);
	function fe(t) {
		u[t] !== !0 && (e.enable(t), u[t] = !0);
	}
	function pe(t) {
		u[t] !== !1 && (e.disable(t), u[t] = !1);
	}
	function me(t, n) {
		return f[t] === n ? !1 : (e.bindFramebuffer(t, n), f[t] = n, t === e.DRAW_FRAMEBUFFER && (f[e.FRAMEBUFFER] = n), t === e.FRAMEBUFFER && (f[e.DRAW_FRAMEBUFFER] = n), !0);
	}
	function he(t, n) {
		let r = m, i = !1;
		if (t) {
			r = p.get(n), r === void 0 && (r = [], p.set(n, r));
			let a = t.textures;
			if (r.length !== a.length || r[0] !== e.COLOR_ATTACHMENT0) {
				for (let t = 0, n = a.length; t < n; t++) r[t] = e.COLOR_ATTACHMENT0 + t;
				r.length = a.length, i = !0;
			}
		} else r[0] !== e.BACK && (r[0] = e.BACK, i = !0);
		i && e.drawBuffers(r);
	}
	function ge(t) {
		return h === t ? !1 : (e.useProgram(t), h = t, !0);
	}
	let _e = {
		100: e.FUNC_ADD,
		101: e.FUNC_SUBTRACT,
		102: e.FUNC_REVERSE_SUBTRACT
	};
	_e[103] = e.MIN, _e[104] = e.MAX;
	let ve = {
		200: e.ZERO,
		201: e.ONE,
		202: e.SRC_COLOR,
		204: e.SRC_ALPHA,
		210: e.SRC_ALPHA_SATURATE,
		208: e.DST_COLOR,
		206: e.DST_ALPHA,
		203: e.ONE_MINUS_SRC_COLOR,
		205: e.ONE_MINUS_SRC_ALPHA,
		209: e.ONE_MINUS_DST_COLOR,
		207: e.ONE_MINUS_DST_ALPHA,
		211: e.CONSTANT_COLOR,
		212: e.ONE_MINUS_CONSTANT_COLOR,
		213: e.CONSTANT_ALPHA,
		214: e.ONE_MINUS_CONSTANT_ALPHA
	};
	function ye(t, n, r, i, a, o, s, c, l, u) {
		if (t === 0) {
			g === !0 && (pe(e.BLEND), g = !1);
			return;
		}
		if (g === !1 && (fe(e.BLEND), g = !0), t !== 5) {
			if (t !== _ || u !== E) {
				if ((v !== 100 || x !== 100) && (e.blendEquation(e.FUNC_ADD), v = 100, x = 100), u) switch (t) {
					case 1:
						e.blendFuncSeparate(e.ONE, e.ONE_MINUS_SRC_ALPHA, e.ONE, e.ONE_MINUS_SRC_ALPHA);
						break;
					case 2:
						e.blendFunc(e.ONE, e.ONE);
						break;
					case 3:
						e.blendFuncSeparate(e.ZERO, e.ONE_MINUS_SRC_COLOR, e.ZERO, e.ONE);
						break;
					case 4:
						e.blendFuncSeparate(e.DST_COLOR, e.ONE_MINUS_SRC_ALPHA, e.ZERO, e.ONE);
						break;
					default:
						K("WebGLState: Invalid blending: ", t);
						break;
				}
				else switch (t) {
					case 1:
						e.blendFuncSeparate(e.SRC_ALPHA, e.ONE_MINUS_SRC_ALPHA, e.ONE, e.ONE_MINUS_SRC_ALPHA);
						break;
					case 2:
						e.blendFuncSeparate(e.SRC_ALPHA, e.ONE, e.ONE, e.ONE);
						break;
					case 3:
						K("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");
						break;
					case 4:
						K("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");
						break;
					default:
						K("WebGLState: Invalid blending: ", t);
						break;
				}
				y = null, b = null, S = null, C = null, w.set(0, 0, 0), T = 0, _ = t, E = u;
			}
			return;
		}
		a ||= n, o ||= r, s ||= i, (n !== v || a !== x) && (e.blendEquationSeparate(_e[n], _e[a]), v = n, x = a), (r !== y || i !== b || o !== S || s !== C) && (e.blendFuncSeparate(ve[r], ve[i], ve[o], ve[s]), y = r, b = i, S = o, C = s), (c.equals(w) === !1 || l !== T) && (e.blendColor(c.r, c.g, c.b, l), w.copy(c), T = l), _ = t, E = !1;
	}
	function be(t, n) {
		t.side === 2 ? pe(e.CULL_FACE) : fe(e.CULL_FACE);
		let r = t.side === 1;
		n && (r = !r), xe(r), t.blending === 1 && t.transparent === !1 ? ye(0) : ye(t.blending, t.blendEquation, t.blendSrc, t.blendDst, t.blendEquationAlpha, t.blendSrcAlpha, t.blendDstAlpha, t.blendColor, t.blendAlpha, t.premultipliedAlpha), o.setFunc(t.depthFunc), o.setTest(t.depthTest), o.setMask(t.depthWrite), a.setMask(t.colorWrite);
		let i = t.stencilWrite;
		s.setTest(i), i && (s.setMask(t.stencilWriteMask), s.setFunc(t.stencilFunc, t.stencilRef, t.stencilFuncMask), s.setOp(t.stencilFail, t.stencilZFail, t.stencilZPass)), we(t.polygonOffset, t.polygonOffsetFactor, t.polygonOffsetUnits), t.alphaToCoverage === !0 ? fe(e.SAMPLE_ALPHA_TO_COVERAGE) : pe(e.SAMPLE_ALPHA_TO_COVERAGE);
	}
	function xe(t) {
		D !== t && (t ? e.frontFace(e.CW) : e.frontFace(e.CCW), D = t);
	}
	function Se(t) {
		t === 0 ? pe(e.CULL_FACE) : (fe(e.CULL_FACE), t !== O && (t === 1 ? e.cullFace(e.BACK) : t === 2 ? e.cullFace(e.FRONT) : e.cullFace(e.FRONT_AND_BACK))), O = t;
	}
	function Ce(t) {
		t !== k && (j && e.lineWidth(t), k = t);
	}
	function we(t, n, r) {
		t ? (fe(e.POLYGON_OFFSET_FILL), (A !== n || ee !== r) && (A = n, ee = r, o.getReversed() && (n = -n), e.polygonOffset(n, r))) : pe(e.POLYGON_OFFSET_FILL);
	}
	function Te(t) {
		t ? fe(e.SCISSOR_TEST) : pe(e.SCISSOR_TEST);
	}
	function M(t) {
		t === void 0 && (t = e.TEXTURE0 + te - 1), ie !== t && (e.activeTexture(t), ie = t);
	}
	function Ee(t, n, r) {
		r === void 0 && (r = ie === null ? e.TEXTURE0 + te - 1 : ie);
		let i = ae[r];
		i === void 0 && (i = {
			type: void 0,
			texture: void 0
		}, ae[r] = i), (i.type !== t || i.texture !== n) && (ie !== r && (e.activeTexture(r), ie = r), e.bindTexture(t, n || de[t]), i.type = t, i.texture = n);
	}
	function De() {
		let t = ae[ie];
		t !== void 0 && t.type !== void 0 && (e.bindTexture(t.type, null), t.type = void 0, t.texture = void 0);
	}
	function Oe() {
		try {
			e.compressedTexImage2D(...arguments);
		} catch (e) {
			K("WebGLState:", e);
		}
	}
	function N() {
		try {
			e.compressedTexImage3D(...arguments);
		} catch (e) {
			K("WebGLState:", e);
		}
	}
	function ke() {
		try {
			e.texSubImage2D(...arguments);
		} catch (e) {
			K("WebGLState:", e);
		}
	}
	function P() {
		try {
			e.texSubImage3D(...arguments);
		} catch (e) {
			K("WebGLState:", e);
		}
	}
	function F() {
		try {
			e.compressedTexSubImage2D(...arguments);
		} catch (e) {
			K("WebGLState:", e);
		}
	}
	function Ae() {
		try {
			e.compressedTexSubImage3D(...arguments);
		} catch (e) {
			K("WebGLState:", e);
		}
	}
	function je() {
		try {
			e.texStorage2D(...arguments);
		} catch (e) {
			K("WebGLState:", e);
		}
	}
	function Me() {
		try {
			e.texStorage3D(...arguments);
		} catch (e) {
			K("WebGLState:", e);
		}
	}
	function Ne() {
		try {
			e.texImage2D(...arguments);
		} catch (e) {
			K("WebGLState:", e);
		}
	}
	function Pe() {
		try {
			e.texImage3D(...arguments);
		} catch (e) {
			K("WebGLState:", e);
		}
	}
	function Fe(t) {
		return d[t] === void 0 ? e.getParameter(t) : d[t];
	}
	function Ie(t, n) {
		d[t] !== n && (e.pixelStorei(t, n), d[t] = n);
	}
	function Le(t) {
		ce.equals(t) === !1 && (e.scissor(t.x, t.y, t.z, t.w), ce.copy(t));
	}
	function Re(t) {
		le.equals(t) === !1 && (e.viewport(t.x, t.y, t.z, t.w), le.copy(t));
	}
	function ze(t, n) {
		let r = l.get(n);
		r === void 0 && (r = /* @__PURE__ */ new WeakMap(), l.set(n, r));
		let i = r.get(t);
		i === void 0 && (i = e.getUniformBlockIndex(n, t.name), r.set(t, i));
	}
	function Be(t, n) {
		let r = l.get(n).get(t);
		c.get(n) !== r && (e.uniformBlockBinding(n, r, t.__bindingPointIndex), c.set(n, r));
	}
	function Ve() {
		e.disable(e.BLEND), e.disable(e.CULL_FACE), e.disable(e.DEPTH_TEST), e.disable(e.POLYGON_OFFSET_FILL), e.disable(e.SCISSOR_TEST), e.disable(e.STENCIL_TEST), e.disable(e.SAMPLE_ALPHA_TO_COVERAGE), e.blendEquation(e.FUNC_ADD), e.blendFunc(e.ONE, e.ZERO), e.blendFuncSeparate(e.ONE, e.ZERO, e.ONE, e.ZERO), e.blendColor(0, 0, 0, 0), e.colorMask(!0, !0, !0, !0), e.clearColor(0, 0, 0, 0), e.depthMask(!0), e.depthFunc(e.LESS), o.setReversed(!1), e.clearDepth(1), e.stencilMask(4294967295), e.stencilFunc(e.ALWAYS, 0, 4294967295), e.stencilOp(e.KEEP, e.KEEP, e.KEEP), e.clearStencil(0), e.cullFace(e.BACK), e.frontFace(e.CCW), e.polygonOffset(0, 0), e.activeTexture(e.TEXTURE0), e.bindFramebuffer(e.FRAMEBUFFER, null), e.bindFramebuffer(e.DRAW_FRAMEBUFFER, null), e.bindFramebuffer(e.READ_FRAMEBUFFER, null), e.useProgram(null), e.lineWidth(1), e.scissor(0, 0, e.canvas.width, e.canvas.height), e.viewport(0, 0, e.canvas.width, e.canvas.height), e.pixelStorei(e.PACK_ALIGNMENT, 4), e.pixelStorei(e.UNPACK_ALIGNMENT, 4), e.pixelStorei(e.UNPACK_FLIP_Y_WEBGL, !1), e.pixelStorei(e.UNPACK_PREMULTIPLY_ALPHA_WEBGL, !1), e.pixelStorei(e.UNPACK_COLORSPACE_CONVERSION_WEBGL, e.BROWSER_DEFAULT_WEBGL), e.pixelStorei(e.PACK_ROW_LENGTH, 0), e.pixelStorei(e.PACK_SKIP_PIXELS, 0), e.pixelStorei(e.PACK_SKIP_ROWS, 0), e.pixelStorei(e.UNPACK_ROW_LENGTH, 0), e.pixelStorei(e.UNPACK_IMAGE_HEIGHT, 0), e.pixelStorei(e.UNPACK_SKIP_PIXELS, 0), e.pixelStorei(e.UNPACK_SKIP_ROWS, 0), e.pixelStorei(e.UNPACK_SKIP_IMAGES, 0), u = {}, d = {}, ie = null, ae = {}, f = {}, p = /* @__PURE__ */ new WeakMap(), m = [], h = null, g = !1, _ = null, v = null, y = null, b = null, x = null, S = null, C = null, w = new Z(0, 0, 0), T = 0, E = !1, D = null, O = null, k = null, A = null, ee = null, ce.set(0, 0, e.canvas.width, e.canvas.height), le.set(0, 0, e.canvas.width, e.canvas.height), a.reset(), o.reset(), s.reset();
	}
	return {
		buffers: {
			color: a,
			depth: o,
			stencil: s
		},
		enable: fe,
		disable: pe,
		bindFramebuffer: me,
		drawBuffers: he,
		useProgram: ge,
		setBlending: ye,
		setMaterial: be,
		setFlipSided: xe,
		setCullFace: Se,
		setLineWidth: Ce,
		setPolygonOffset: we,
		setScissorTest: Te,
		activeTexture: M,
		bindTexture: Ee,
		unbindTexture: De,
		compressedTexImage2D: Oe,
		compressedTexImage3D: N,
		texImage2D: Ne,
		texImage3D: Pe,
		pixelStorei: Ie,
		getParameter: Fe,
		updateUBOMapping: ze,
		uniformBlockBinding: Be,
		texStorage2D: je,
		texStorage3D: Me,
		texSubImage2D: ke,
		texSubImage3D: P,
		compressedTexSubImage2D: F,
		compressedTexSubImage3D: Ae,
		scissor: Le,
		viewport: Re,
		reset: Ve
	};
}
function zy(e, t, n, r, i, a, o) {
	let s = t.has("WEBGL_multisampled_render_to_texture") ? t.get("WEBGL_multisampled_render_to_texture") : null, c = typeof navigator > "u" ? !1 : /OculusBrowser/g.test(navigator.userAgent), l = new Tf(), u = /* @__PURE__ */ new WeakMap(), d = /* @__PURE__ */ new Set(), f, p = /* @__PURE__ */ new WeakMap(), m = !1;
	try {
		m = typeof OffscreenCanvas < "u" && new OffscreenCanvas(1, 1).getContext("2d") !== null;
	} catch {}
	function h(e, t) {
		return m ? new OffscreenCanvas(e, t) : Ud("canvas");
	}
	function g(e, t, n) {
		let r = 1, i = Oe(e);
		if ((i.width > n || i.height > n) && (r = n / Math.max(i.width, i.height)), r < 1) if (typeof HTMLImageElement < "u" && e instanceof HTMLImageElement || typeof HTMLCanvasElement < "u" && e instanceof HTMLCanvasElement || typeof ImageBitmap < "u" && e instanceof ImageBitmap || typeof VideoFrame < "u" && e instanceof VideoFrame) {
			let n = Math.floor(r * i.width), a = Math.floor(r * i.height);
			f === void 0 && (f = h(n, a));
			let o = t ? h(n, a) : f;
			return o.width = n, o.height = a, o.getContext("2d").drawImage(e, 0, 0, n, a), G("WebGLRenderer: Texture has been resized from (" + i.width + "x" + i.height + ") to (" + n + "x" + a + ")."), o;
		} else return "data" in e && G("WebGLRenderer: Image in DataTexture is too big (" + i.width + "x" + i.height + ")."), e;
		return e;
	}
	function _(e) {
		return e.generateMipmaps;
	}
	function v(t) {
		e.generateMipmap(t);
	}
	function y(t) {
		return t.isWebGLCubeRenderTarget ? e.TEXTURE_CUBE_MAP : t.isWebGL3DRenderTarget ? e.TEXTURE_3D : t.isWebGLArrayRenderTarget || t.isCompressedArrayTexture ? e.TEXTURE_2D_ARRAY : e.TEXTURE_2D;
	}
	function b(n, r, i, a, o, s = !1) {
		if (n !== null) {
			if (e[n] !== void 0) return e[n];
			G("WebGLRenderer: Attempt to use non-existing WebGL internal format '" + n + "'");
		}
		let c;
		a && (c = t.get("EXT_texture_norm16"), c || G("WebGLRenderer: Unable to use normalized textures without EXT_texture_norm16 extension"));
		let l = r;
		if (r === e.RED && (i === e.FLOAT && (l = e.R32F), i === e.HALF_FLOAT && (l = e.R16F), i === e.UNSIGNED_BYTE && (l = e.R8), i === e.UNSIGNED_SHORT && c && (l = c.R16_EXT), i === e.SHORT && c && (l = c.R16_SNORM_EXT)), r === e.RED_INTEGER && (i === e.UNSIGNED_BYTE && (l = e.R8UI), i === e.UNSIGNED_SHORT && (l = e.R16UI), i === e.UNSIGNED_INT && (l = e.R32UI), i === e.BYTE && (l = e.R8I), i === e.SHORT && (l = e.R16I), i === e.INT && (l = e.R32I)), r === e.RG && (i === e.FLOAT && (l = e.RG32F), i === e.HALF_FLOAT && (l = e.RG16F), i === e.UNSIGNED_BYTE && (l = e.RG8), i === e.UNSIGNED_SHORT && c && (l = c.RG16_EXT), i === e.SHORT && c && (l = c.RG16_SNORM_EXT)), r === e.RG_INTEGER && (i === e.UNSIGNED_BYTE && (l = e.RG8UI), i === e.UNSIGNED_SHORT && (l = e.RG16UI), i === e.UNSIGNED_INT && (l = e.RG32UI), i === e.BYTE && (l = e.RG8I), i === e.SHORT && (l = e.RG16I), i === e.INT && (l = e.RG32I)), r === e.RGB_INTEGER && (i === e.UNSIGNED_BYTE && (l = e.RGB8UI), i === e.UNSIGNED_SHORT && (l = e.RGB16UI), i === e.UNSIGNED_INT && (l = e.RGB32UI), i === e.BYTE && (l = e.RGB8I), i === e.SHORT && (l = e.RGB16I), i === e.INT && (l = e.RGB32I)), r === e.RGBA_INTEGER && (i === e.UNSIGNED_BYTE && (l = e.RGBA8UI), i === e.UNSIGNED_SHORT && (l = e.RGBA16UI), i === e.UNSIGNED_INT && (l = e.RGBA32UI), i === e.BYTE && (l = e.RGBA8I), i === e.SHORT && (l = e.RGBA16I), i === e.INT && (l = e.RGBA32I)), r === e.RGB && (i === e.UNSIGNED_SHORT && c && (l = c.RGB16_EXT), i === e.SHORT && c && (l = c.RGB16_SNORM_EXT), i === e.UNSIGNED_INT_5_9_9_9_REV && (l = e.RGB9_E5), i === e.UNSIGNED_INT_10F_11F_11F_REV && (l = e.R11F_G11F_B10F)), r === e.RGBA) {
			let t = s ? Fd : X.getTransfer(o);
			i === e.FLOAT && (l = e.RGBA32F), i === e.HALF_FLOAT && (l = e.RGBA16F), i === e.UNSIGNED_BYTE && (l = t === "srgb" ? e.SRGB8_ALPHA8 : e.RGBA8), i === e.UNSIGNED_SHORT && c && (l = c.RGBA16_EXT), i === e.SHORT && c && (l = c.RGBA16_SNORM_EXT), i === e.UNSIGNED_SHORT_4_4_4_4 && (l = e.RGBA4), i === e.UNSIGNED_SHORT_5_5_5_1 && (l = e.RGB5_A1);
		}
		return (l === e.R16F || l === e.R32F || l === e.RG16F || l === e.RG32F || l === e.RGBA16F || l === e.RGBA32F) && t.get("EXT_color_buffer_float"), l;
	}
	function x(t, n) {
		let r;
		return t ? n === null || n === 1014 || n === 1020 ? r = e.DEPTH24_STENCIL8 : n === 1015 ? r = e.DEPTH32F_STENCIL8 : n === 1012 && (r = e.DEPTH24_STENCIL8, G("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")) : n === null || n === 1014 || n === 1020 ? r = e.DEPTH_COMPONENT24 : n === 1015 ? r = e.DEPTH_COMPONENT32F : n === 1012 && (r = e.DEPTH_COMPONENT16), r;
	}
	function S(e, t) {
		return _(e) === !0 || e.isFramebufferTexture && e.minFilter !== 1003 && e.minFilter !== 1006 ? Math.log2(Math.max(t.width, t.height)) + 1 : e.mipmaps !== void 0 && e.mipmaps.length > 0 ? e.mipmaps.length : e.isCompressedTexture && Array.isArray(e.image) ? t.mipmaps.length : 1;
	}
	function C(e) {
		let t = e.target;
		t.removeEventListener("dispose", C), T(t), t.isVideoTexture && u.delete(t), t.isHTMLTexture && d.delete(t);
	}
	function w(e) {
		let t = e.target;
		t.removeEventListener("dispose", w), D(t);
	}
	function T(e) {
		let t = r.get(e);
		if (t.__webglInit === void 0) return;
		let n = e.source, i = p.get(n);
		if (i) {
			let r = i[t.__cacheKey];
			r.usedTimes--, r.usedTimes === 0 && E(e), Object.keys(i).length === 0 && p.delete(n);
		}
		r.remove(e);
	}
	function E(t) {
		let n = r.get(t);
		e.deleteTexture(n.__webglTexture);
		let i = t.source, a = p.get(i);
		delete a[n.__cacheKey], o.memory.textures--;
	}
	function D(t) {
		let n = r.get(t);
		if (t.depthTexture && (t.depthTexture.dispose(), r.remove(t.depthTexture)), t.isWebGLCubeRenderTarget) for (let t = 0; t < 6; t++) {
			if (Array.isArray(n.__webglFramebuffer[t])) for (let r = 0; r < n.__webglFramebuffer[t].length; r++) e.deleteFramebuffer(n.__webglFramebuffer[t][r]);
			else e.deleteFramebuffer(n.__webglFramebuffer[t]);
			n.__webglDepthbuffer && e.deleteRenderbuffer(n.__webglDepthbuffer[t]);
		}
		else {
			if (Array.isArray(n.__webglFramebuffer)) for (let t = 0; t < n.__webglFramebuffer.length; t++) e.deleteFramebuffer(n.__webglFramebuffer[t]);
			else e.deleteFramebuffer(n.__webglFramebuffer);
			if (n.__webglDepthbuffer && e.deleteRenderbuffer(n.__webglDepthbuffer), n.__webglMultisampledFramebuffer && e.deleteFramebuffer(n.__webglMultisampledFramebuffer), n.__webglColorRenderbuffer) for (let t = 0; t < n.__webglColorRenderbuffer.length; t++) n.__webglColorRenderbuffer[t] && e.deleteRenderbuffer(n.__webglColorRenderbuffer[t]);
			n.__webglDepthRenderbuffer && e.deleteRenderbuffer(n.__webglDepthRenderbuffer);
		}
		let i = t.textures;
		for (let t = 0, n = i.length; t < n; t++) {
			let n = r.get(i[t]);
			n.__webglTexture && (e.deleteTexture(n.__webglTexture), o.memory.textures--), r.remove(i[t]);
		}
		r.remove(t);
	}
	let O = 0;
	function k() {
		O = 0;
	}
	function A() {
		return O;
	}
	function ee(e) {
		O = e;
	}
	function te() {
		let e = O;
		return e >= i.maxTextures && G("WebGLTextures: Trying to use " + e + " texture units while this GPU supports only " + i.maxTextures), O += 1, e;
	}
	function j(e) {
		let t = [];
		return t.push(e.wrapS), t.push(e.wrapT), t.push(e.wrapR || 0), t.push(e.magFilter), t.push(e.minFilter), t.push(e.anisotropy), t.push(e.internalFormat), t.push(e.format), t.push(e.type), t.push(e.generateMipmaps), t.push(e.premultiplyAlpha), t.push(e.flipY), t.push(e.unpackAlignment), t.push(e.colorSpace), t.join();
	}
	function ne(t, i) {
		let a = r.get(t);
		if (t.isVideoTexture && Ee(t), t.isRenderTargetTexture === !1 && t.isExternalTexture !== !0 && t.version > 0 && a.__version !== t.version) {
			let e = t.image;
			if (e === null) G("WebGLRenderer: Texture marked for update but no image data found.");
			else if (e.complete === !1) G("WebGLRenderer: Texture marked for update but image is incomplete");
			else {
				pe(a, t, i);
				return;
			}
		} else t.isExternalTexture && (a.__webglTexture = t.sourceTexture ? t.sourceTexture : null);
		n.bindTexture(e.TEXTURE_2D, a.__webglTexture, e.TEXTURE0 + i);
	}
	function re(t, i) {
		let a = r.get(t);
		if (t.isRenderTargetTexture === !1 && t.version > 0 && a.__version !== t.version) {
			pe(a, t, i);
			return;
		} else t.isExternalTexture && (a.__webglTexture = t.sourceTexture ? t.sourceTexture : null);
		n.bindTexture(e.TEXTURE_2D_ARRAY, a.__webglTexture, e.TEXTURE0 + i);
	}
	function ie(t, i) {
		let a = r.get(t);
		if (t.isRenderTargetTexture === !1 && t.version > 0 && a.__version !== t.version) {
			pe(a, t, i);
			return;
		}
		n.bindTexture(e.TEXTURE_3D, a.__webglTexture, e.TEXTURE0 + i);
	}
	function ae(t, i) {
		let a = r.get(t);
		if (t.isCubeDepthTexture !== !0 && t.version > 0 && a.__version !== t.version) {
			me(a, t, i);
			return;
		}
		n.bindTexture(e.TEXTURE_CUBE_MAP, a.__webglTexture, e.TEXTURE0 + i);
	}
	let oe = {
		[du]: e.REPEAT,
		[fu]: e.CLAMP_TO_EDGE,
		[pu]: e.MIRRORED_REPEAT
	}, se = {
		[mu]: e.NEAREST,
		[hu]: e.NEAREST_MIPMAP_NEAREST,
		[gu]: e.NEAREST_MIPMAP_LINEAR,
		[_u]: e.LINEAR,
		[vu]: e.LINEAR_MIPMAP_NEAREST,
		[yu]: e.LINEAR_MIPMAP_LINEAR
	}, ce = {
		512: e.NEVER,
		519: e.ALWAYS,
		513: e.LESS,
		515: e.LEQUAL,
		514: e.EQUAL,
		518: e.GEQUAL,
		516: e.GREATER,
		517: e.NOTEQUAL
	};
	function le(n, a) {
		if (a.type === 1015 && t.has("OES_texture_float_linear") === !1 && (a.magFilter === 1006 || a.magFilter === 1007 || a.magFilter === 1005 || a.magFilter === 1008 || a.minFilter === 1006 || a.minFilter === 1007 || a.minFilter === 1005 || a.minFilter === 1008) && G("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."), e.texParameteri(n, e.TEXTURE_WRAP_S, oe[a.wrapS]), e.texParameteri(n, e.TEXTURE_WRAP_T, oe[a.wrapT]), (n === e.TEXTURE_3D || n === e.TEXTURE_2D_ARRAY) && e.texParameteri(n, e.TEXTURE_WRAP_R, oe[a.wrapR]), e.texParameteri(n, e.TEXTURE_MAG_FILTER, se[a.magFilter]), e.texParameteri(n, e.TEXTURE_MIN_FILTER, se[a.minFilter]), a.compareFunction && (e.texParameteri(n, e.TEXTURE_COMPARE_MODE, e.COMPARE_REF_TO_TEXTURE), e.texParameteri(n, e.TEXTURE_COMPARE_FUNC, ce[a.compareFunction])), t.has("EXT_texture_filter_anisotropic") === !0) {
			if (a.magFilter === 1003 || a.minFilter !== 1005 && a.minFilter !== 1008 || a.type === 1015 && t.has("OES_texture_float_linear") === !1) return;
			if (a.anisotropy > 1 || r.get(a).__currentAnisotropy) {
				let o = t.get("EXT_texture_filter_anisotropic");
				e.texParameterf(n, o.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(a.anisotropy, i.getMaxAnisotropy())), r.get(a).__currentAnisotropy = a.anisotropy;
			}
		}
	}
	function ue(t, n) {
		let r = !1;
		t.__webglInit === void 0 && (t.__webglInit = !0, n.addEventListener("dispose", C));
		let i = n.source, a = p.get(i);
		a === void 0 && (a = {}, p.set(i, a));
		let s = j(n);
		if (s !== t.__cacheKey) {
			a[s] === void 0 && (a[s] = {
				texture: e.createTexture(),
				usedTimes: 0
			}, o.memory.textures++, r = !0), a[s].usedTimes++;
			let i = a[t.__cacheKey];
			i !== void 0 && (a[t.__cacheKey].usedTimes--, i.usedTimes === 0 && E(n)), t.__cacheKey = s, t.__webglTexture = a[s].texture;
		}
		return r;
	}
	function de(e, t, n) {
		return Math.floor(Math.floor(e / n) / t);
	}
	function fe(t, r, i, a) {
		let o = t.updateRanges;
		if (o.length === 0) n.texSubImage2D(e.TEXTURE_2D, 0, 0, 0, r.width, r.height, i, a, r.data);
		else {
			o.sort((e, t) => e.start - t.start);
			let s = 0;
			for (let e = 1; e < o.length; e++) {
				let t = o[s], n = o[e], i = t.start + t.count, a = de(n.start, r.width, 4), c = de(t.start, r.width, 4);
				n.start <= i + 1 && a === c && de(n.start + n.count - 1, r.width, 4) === a ? t.count = Math.max(t.count, n.start + n.count - t.start) : (++s, o[s] = n);
			}
			o.length = s + 1;
			let c = n.getParameter(e.UNPACK_ROW_LENGTH), l = n.getParameter(e.UNPACK_SKIP_PIXELS), u = n.getParameter(e.UNPACK_SKIP_ROWS);
			n.pixelStorei(e.UNPACK_ROW_LENGTH, r.width);
			for (let t = 0, s = o.length; t < s; t++) {
				let s = o[t], c = Math.floor(s.start / 4), l = Math.ceil(s.count / 4), u = c % r.width, d = Math.floor(c / r.width), f = l;
				n.pixelStorei(e.UNPACK_SKIP_PIXELS, u), n.pixelStorei(e.UNPACK_SKIP_ROWS, d), n.texSubImage2D(e.TEXTURE_2D, 0, u, d, f, 1, i, a, r.data);
			}
			t.clearUpdateRanges(), n.pixelStorei(e.UNPACK_ROW_LENGTH, c), n.pixelStorei(e.UNPACK_SKIP_PIXELS, l), n.pixelStorei(e.UNPACK_SKIP_ROWS, u);
		}
	}
	function pe(t, o, s) {
		let c = e.TEXTURE_2D;
		(o.isDataArrayTexture || o.isCompressedArrayTexture) && (c = e.TEXTURE_2D_ARRAY), o.isData3DTexture && (c = e.TEXTURE_3D);
		let l = ue(t, o), u = o.source;
		n.bindTexture(c, t.__webglTexture, e.TEXTURE0 + s);
		let f = r.get(u);
		if (u.version !== f.__version || l === !0) {
			if (n.activeTexture(e.TEXTURE0 + s), !(typeof ImageBitmap < "u" && o.image instanceof ImageBitmap)) {
				let t = X.getPrimaries(X.workingColorSpace), r = o.colorSpace === "" ? null : X.getPrimaries(o.colorSpace), i = o.colorSpace === "" || t === r ? e.NONE : e.BROWSER_DEFAULT_WEBGL;
				n.pixelStorei(e.UNPACK_FLIP_Y_WEBGL, o.flipY), n.pixelStorei(e.UNPACK_PREMULTIPLY_ALPHA_WEBGL, o.premultiplyAlpha), n.pixelStorei(e.UNPACK_COLORSPACE_CONVERSION_WEBGL, i);
			}
			n.pixelStorei(e.UNPACK_ALIGNMENT, o.unpackAlignment);
			let t = g(o.image, !1, i.maxTextureSize);
			t = De(o, t);
			let r = a.convert(o.format, o.colorSpace), p = a.convert(o.type), m = b(o.internalFormat, r, p, o.normalized, o.colorSpace, o.isVideoTexture);
			le(c, o);
			let h, y = o.mipmaps, C = o.isVideoTexture !== !0, w = f.__version === void 0 || l === !0, T = u.dataReady, E = S(o, t);
			if (o.isDepthTexture) m = x(o.format === Lu, o.type), w && (C ? n.texStorage2D(e.TEXTURE_2D, 1, m, t.width, t.height) : n.texImage2D(e.TEXTURE_2D, 0, m, t.width, t.height, 0, r, p, null));
			else if (o.isDataTexture) if (y.length > 0) {
				C && w && n.texStorage2D(e.TEXTURE_2D, E, m, y[0].width, y[0].height);
				for (let t = 0, i = y.length; t < i; t++) h = y[t], C ? T && n.texSubImage2D(e.TEXTURE_2D, t, 0, 0, h.width, h.height, r, p, h.data) : n.texImage2D(e.TEXTURE_2D, t, m, h.width, h.height, 0, r, p, h.data);
				o.generateMipmaps = !1;
			} else C ? (w && n.texStorage2D(e.TEXTURE_2D, E, m, t.width, t.height), T && fe(o, t, r, p)) : n.texImage2D(e.TEXTURE_2D, 0, m, t.width, t.height, 0, r, p, t.data);
			else if (o.isCompressedTexture) if (o.isCompressedArrayTexture) {
				C && w && n.texStorage3D(e.TEXTURE_2D_ARRAY, E, m, y[0].width, y[0].height, t.depth);
				for (let i = 0, a = y.length; i < a; i++) if (h = y[i], o.format !== 1023) if (r !== null) if (C) {
					if (T) if (o.layerUpdates.size > 0) {
						let t = Fg(h.width, h.height, o.format, o.type);
						for (let a of o.layerUpdates) {
							let o = h.data.subarray(a * t / h.data.BYTES_PER_ELEMENT, (a + 1) * t / h.data.BYTES_PER_ELEMENT);
							n.compressedTexSubImage3D(e.TEXTURE_2D_ARRAY, i, 0, 0, a, h.width, h.height, 1, r, o);
						}
						o.clearLayerUpdates();
					} else n.compressedTexSubImage3D(e.TEXTURE_2D_ARRAY, i, 0, 0, 0, h.width, h.height, t.depth, r, h.data);
				} else n.compressedTexImage3D(e.TEXTURE_2D_ARRAY, i, m, h.width, h.height, t.depth, 0, h.data, 0, 0);
				else G("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");
				else C ? T && n.texSubImage3D(e.TEXTURE_2D_ARRAY, i, 0, 0, 0, h.width, h.height, t.depth, r, p, h.data) : n.texImage3D(e.TEXTURE_2D_ARRAY, i, m, h.width, h.height, t.depth, 0, r, p, h.data);
			} else {
				C && w && n.texStorage2D(e.TEXTURE_2D, E, m, y[0].width, y[0].height);
				for (let t = 0, i = y.length; t < i; t++) h = y[t], o.format === 1023 ? C ? T && n.texSubImage2D(e.TEXTURE_2D, t, 0, 0, h.width, h.height, r, p, h.data) : n.texImage2D(e.TEXTURE_2D, t, m, h.width, h.height, 0, r, p, h.data) : r === null ? G("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()") : C ? T && n.compressedTexSubImage2D(e.TEXTURE_2D, t, 0, 0, h.width, h.height, r, h.data) : n.compressedTexImage2D(e.TEXTURE_2D, t, m, h.width, h.height, 0, h.data);
			}
			else if (o.isDataArrayTexture) if (C) {
				if (w && n.texStorage3D(e.TEXTURE_2D_ARRAY, E, m, t.width, t.height, t.depth), T) if (o.layerUpdates.size > 0) {
					let i = Fg(t.width, t.height, o.format, o.type);
					for (let a of o.layerUpdates) {
						let o = t.data.subarray(a * i / t.data.BYTES_PER_ELEMENT, (a + 1) * i / t.data.BYTES_PER_ELEMENT);
						n.texSubImage3D(e.TEXTURE_2D_ARRAY, 0, 0, 0, a, t.width, t.height, 1, r, p, o);
					}
					o.clearLayerUpdates();
				} else n.texSubImage3D(e.TEXTURE_2D_ARRAY, 0, 0, 0, 0, t.width, t.height, t.depth, r, p, t.data);
			} else n.texImage3D(e.TEXTURE_2D_ARRAY, 0, m, t.width, t.height, t.depth, 0, r, p, t.data);
			else if (o.isData3DTexture) C ? (w && n.texStorage3D(e.TEXTURE_3D, E, m, t.width, t.height, t.depth), T && n.texSubImage3D(e.TEXTURE_3D, 0, 0, 0, 0, t.width, t.height, t.depth, r, p, t.data)) : n.texImage3D(e.TEXTURE_3D, 0, m, t.width, t.height, t.depth, 0, r, p, t.data);
			else if (o.isFramebufferTexture) {
				if (w) if (C) n.texStorage2D(e.TEXTURE_2D, E, m, t.width, t.height);
				else {
					let i = t.width, a = t.height;
					for (let t = 0; t < E; t++) n.texImage2D(e.TEXTURE_2D, t, m, i, a, 0, r, p, null), i >>= 1, a >>= 1;
				}
			} else if (o.isHTMLTexture) {
				if ("texElementImage2D" in e) {
					let n = e.canvas;
					if (n.hasAttribute("layoutsubtree") || n.setAttribute("layoutsubtree", "true"), t.parentNode !== n) {
						n.appendChild(t), d.add(o), n.onpaint = (e) => {
							let t = e.changedElements;
							for (let e of d) t.includes(e.image) && (e.needsUpdate = !0);
						}, n.requestPaint();
						return;
					}
					if (e.texElementImage2D.length === 3) e.texElementImage2D(e.TEXTURE_2D, e.RGBA8, t);
					else {
						let n = e.RGBA, r = e.RGBA, i = e.UNSIGNED_BYTE;
						e.texElementImage2D(e.TEXTURE_2D, 0, n, r, i, t);
					}
					e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.LINEAR), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, e.CLAMP_TO_EDGE);
				}
			} else if (y.length > 0) {
				if (C && w) {
					let t = Oe(y[0]);
					n.texStorage2D(e.TEXTURE_2D, E, m, t.width, t.height);
				}
				for (let t = 0, i = y.length; t < i; t++) h = y[t], C ? T && n.texSubImage2D(e.TEXTURE_2D, t, 0, 0, r, p, h) : n.texImage2D(e.TEXTURE_2D, t, m, r, p, h);
				o.generateMipmaps = !1;
			} else if (C) {
				if (w) {
					let r = Oe(t);
					n.texStorage2D(e.TEXTURE_2D, E, m, r.width, r.height);
				}
				T && n.texSubImage2D(e.TEXTURE_2D, 0, 0, 0, r, p, t);
			} else n.texImage2D(e.TEXTURE_2D, 0, m, r, p, t);
			_(o) && v(c), f.__version = u.version, o.onUpdate && o.onUpdate(o);
		}
		t.__version = o.version;
	}
	function me(t, o, s) {
		if (o.image.length !== 6) return;
		let c = ue(t, o), l = o.source;
		n.bindTexture(e.TEXTURE_CUBE_MAP, t.__webglTexture, e.TEXTURE0 + s);
		let u = r.get(l);
		if (l.version !== u.__version || c === !0) {
			n.activeTexture(e.TEXTURE0 + s);
			let t = X.getPrimaries(X.workingColorSpace), r = o.colorSpace === "" ? null : X.getPrimaries(o.colorSpace), d = o.colorSpace === "" || t === r ? e.NONE : e.BROWSER_DEFAULT_WEBGL;
			n.pixelStorei(e.UNPACK_FLIP_Y_WEBGL, o.flipY), n.pixelStorei(e.UNPACK_PREMULTIPLY_ALPHA_WEBGL, o.premultiplyAlpha), n.pixelStorei(e.UNPACK_ALIGNMENT, o.unpackAlignment), n.pixelStorei(e.UNPACK_COLORSPACE_CONVERSION_WEBGL, d);
			let f = o.isCompressedTexture || o.image[0].isCompressedTexture, p = o.image[0] && o.image[0].isDataTexture, m = [];
			for (let e = 0; e < 6; e++) !f && !p ? m[e] = g(o.image[e], !0, i.maxCubemapSize) : m[e] = p ? o.image[e].image : o.image[e], m[e] = De(o, m[e]);
			let h = m[0], y = a.convert(o.format, o.colorSpace), x = a.convert(o.type), C = b(o.internalFormat, y, x, o.normalized, o.colorSpace), w = o.isVideoTexture !== !0, T = u.__version === void 0 || c === !0, E = l.dataReady, D = S(o, h);
			le(e.TEXTURE_CUBE_MAP, o);
			let O;
			if (f) {
				w && T && n.texStorage2D(e.TEXTURE_CUBE_MAP, D, C, h.width, h.height);
				for (let t = 0; t < 6; t++) {
					O = m[t].mipmaps;
					for (let r = 0; r < O.length; r++) {
						let i = O[r];
						o.format === 1023 ? w ? E && n.texSubImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + t, r, 0, 0, i.width, i.height, y, x, i.data) : n.texImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + t, r, C, i.width, i.height, 0, y, x, i.data) : y === null ? G("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()") : w ? E && n.compressedTexSubImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + t, r, 0, 0, i.width, i.height, y, i.data) : n.compressedTexImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + t, r, C, i.width, i.height, 0, i.data);
					}
				}
			} else {
				if (O = o.mipmaps, w && T) {
					O.length > 0 && D++;
					let t = Oe(m[0]);
					n.texStorage2D(e.TEXTURE_CUBE_MAP, D, C, t.width, t.height);
				}
				for (let t = 0; t < 6; t++) if (p) {
					w ? E && n.texSubImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + t, 0, 0, 0, m[t].width, m[t].height, y, x, m[t].data) : n.texImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + t, 0, C, m[t].width, m[t].height, 0, y, x, m[t].data);
					for (let r = 0; r < O.length; r++) {
						let i = O[r].image[t].image;
						w ? E && n.texSubImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + t, r + 1, 0, 0, i.width, i.height, y, x, i.data) : n.texImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + t, r + 1, C, i.width, i.height, 0, y, x, i.data);
					}
				} else {
					w ? E && n.texSubImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + t, 0, 0, 0, y, x, m[t]) : n.texImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + t, 0, C, y, x, m[t]);
					for (let r = 0; r < O.length; r++) {
						let i = O[r];
						w ? E && n.texSubImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + t, r + 1, 0, 0, y, x, i.image[t]) : n.texImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + t, r + 1, C, y, x, i.image[t]);
					}
				}
			}
			_(o) && v(e.TEXTURE_CUBE_MAP), u.__version = l.version, o.onUpdate && o.onUpdate(o);
		}
		t.__version = o.version;
	}
	function he(t, i, o, c, l, u) {
		let d = a.convert(o.format, o.colorSpace), f = a.convert(o.type), p = b(o.internalFormat, d, f, o.normalized, o.colorSpace), m = r.get(i), h = r.get(o);
		if (h.__renderTarget = i, !m.__hasExternalTextures) {
			let t = Math.max(1, i.width >> u), r = Math.max(1, i.height >> u);
			l === e.TEXTURE_3D || l === e.TEXTURE_2D_ARRAY ? n.texImage3D(l, u, p, t, r, i.depth, 0, d, f, null) : n.texImage2D(l, u, p, t, r, 0, d, f, null);
		}
		n.bindFramebuffer(e.FRAMEBUFFER, t), M(i) ? s.framebufferTexture2DMultisampleEXT(e.FRAMEBUFFER, c, l, h.__webglTexture, 0, Te(i)) : (l === e.TEXTURE_2D || l >= e.TEXTURE_CUBE_MAP_POSITIVE_X && l <= e.TEXTURE_CUBE_MAP_NEGATIVE_Z) && e.framebufferTexture2D(e.FRAMEBUFFER, c, l, h.__webglTexture, u), n.bindFramebuffer(e.FRAMEBUFFER, null);
	}
	function ge(t, n, r) {
		if (e.bindRenderbuffer(e.RENDERBUFFER, t), n.depthBuffer) {
			let i = n.depthTexture, a = i && i.isDepthTexture ? i.type : null, o = x(n.stencilBuffer, a), c = n.stencilBuffer ? e.DEPTH_STENCIL_ATTACHMENT : e.DEPTH_ATTACHMENT;
			M(n) ? s.renderbufferStorageMultisampleEXT(e.RENDERBUFFER, Te(n), o, n.width, n.height) : r ? e.renderbufferStorageMultisample(e.RENDERBUFFER, Te(n), o, n.width, n.height) : e.renderbufferStorage(e.RENDERBUFFER, o, n.width, n.height), e.framebufferRenderbuffer(e.FRAMEBUFFER, c, e.RENDERBUFFER, t);
		} else {
			let t = n.textures;
			for (let i = 0; i < t.length; i++) {
				let o = t[i], c = a.convert(o.format, o.colorSpace), l = a.convert(o.type), u = b(o.internalFormat, c, l, o.normalized, o.colorSpace);
				M(n) ? s.renderbufferStorageMultisampleEXT(e.RENDERBUFFER, Te(n), u, n.width, n.height) : r ? e.renderbufferStorageMultisample(e.RENDERBUFFER, Te(n), u, n.width, n.height) : e.renderbufferStorage(e.RENDERBUFFER, u, n.width, n.height);
			}
		}
		e.bindRenderbuffer(e.RENDERBUFFER, null);
	}
	function _e(t, i, o) {
		let c = i.isWebGLCubeRenderTarget === !0;
		if (n.bindFramebuffer(e.FRAMEBUFFER, t), !(i.depthTexture && i.depthTexture.isDepthTexture)) throw Error("THREE.WebGLTextures: renderTarget.depthTexture must be an instance of THREE.DepthTexture.");
		let l = r.get(i.depthTexture);
		if (l.__renderTarget = i, (!l.__webglTexture || i.depthTexture.image.width !== i.width || i.depthTexture.image.height !== i.height) && (i.depthTexture.image.width = i.width, i.depthTexture.image.height = i.height, i.depthTexture.needsUpdate = !0), c) {
			if (l.__webglInit === void 0 && (l.__webglInit = !0, i.depthTexture.addEventListener("dispose", C)), l.__webglTexture === void 0) {
				l.__webglTexture = e.createTexture(), n.bindTexture(e.TEXTURE_CUBE_MAP, l.__webglTexture), le(e.TEXTURE_CUBE_MAP, i.depthTexture);
				let t = a.convert(i.depthTexture.format), r = a.convert(i.depthTexture.type), o;
				i.depthTexture.format === 1026 ? o = e.DEPTH_COMPONENT24 : i.depthTexture.format === 1027 && (o = e.DEPTH24_STENCIL8);
				for (let n = 0; n < 6; n++) e.texImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + n, 0, o, i.width, i.height, 0, t, r, null);
			}
		} else ne(i.depthTexture, 0);
		let u = l.__webglTexture, d = Te(i), f = c ? e.TEXTURE_CUBE_MAP_POSITIVE_X + o : e.TEXTURE_2D, p = i.depthTexture.format === 1027 ? e.DEPTH_STENCIL_ATTACHMENT : e.DEPTH_ATTACHMENT;
		if (i.depthTexture.format === 1026) M(i) ? s.framebufferTexture2DMultisampleEXT(e.FRAMEBUFFER, p, f, u, 0, d) : e.framebufferTexture2D(e.FRAMEBUFFER, p, f, u, 0);
		else if (i.depthTexture.format === 1027) M(i) ? s.framebufferTexture2DMultisampleEXT(e.FRAMEBUFFER, p, f, u, 0, d) : e.framebufferTexture2D(e.FRAMEBUFFER, p, f, u, 0);
		else throw Error("THREE.WebGLTextures: Unknown depthTexture format.");
	}
	function ve(t) {
		let i = r.get(t), a = t.isWebGLCubeRenderTarget === !0;
		if (i.__boundDepthTexture !== t.depthTexture) {
			let e = t.depthTexture;
			if (i.__depthDisposeCallback && i.__depthDisposeCallback(), e) {
				let t = () => {
					delete i.__boundDepthTexture, delete i.__depthDisposeCallback, e.removeEventListener("dispose", t);
				};
				e.addEventListener("dispose", t), i.__depthDisposeCallback = t;
			}
			i.__boundDepthTexture = e;
		}
		if (t.depthTexture && !i.__autoAllocateDepthBuffer) if (a) for (let e = 0; e < 6; e++) _e(i.__webglFramebuffer[e], t, e);
		else {
			let e = t.texture.mipmaps;
			e && e.length > 0 ? _e(i.__webglFramebuffer[0], t, 0) : _e(i.__webglFramebuffer, t, 0);
		}
		else if (a) {
			i.__webglDepthbuffer = [];
			for (let r = 0; r < 6; r++) if (n.bindFramebuffer(e.FRAMEBUFFER, i.__webglFramebuffer[r]), i.__webglDepthbuffer[r] === void 0) i.__webglDepthbuffer[r] = e.createRenderbuffer(), ge(i.__webglDepthbuffer[r], t, !1);
			else {
				let n = t.stencilBuffer ? e.DEPTH_STENCIL_ATTACHMENT : e.DEPTH_ATTACHMENT, a = i.__webglDepthbuffer[r];
				e.bindRenderbuffer(e.RENDERBUFFER, a), e.framebufferRenderbuffer(e.FRAMEBUFFER, n, e.RENDERBUFFER, a);
			}
		} else {
			let r = t.texture.mipmaps;
			if (r && r.length > 0 ? n.bindFramebuffer(e.FRAMEBUFFER, i.__webglFramebuffer[0]) : n.bindFramebuffer(e.FRAMEBUFFER, i.__webglFramebuffer), i.__webglDepthbuffer === void 0) i.__webglDepthbuffer = e.createRenderbuffer(), ge(i.__webglDepthbuffer, t, !1);
			else {
				let n = t.stencilBuffer ? e.DEPTH_STENCIL_ATTACHMENT : e.DEPTH_ATTACHMENT, r = i.__webglDepthbuffer;
				e.bindRenderbuffer(e.RENDERBUFFER, r), e.framebufferRenderbuffer(e.FRAMEBUFFER, n, e.RENDERBUFFER, r);
			}
		}
		n.bindFramebuffer(e.FRAMEBUFFER, null);
	}
	function ye(t, n, i) {
		let a = r.get(t);
		n !== void 0 && he(a.__webglFramebuffer, t, t.texture, e.COLOR_ATTACHMENT0, e.TEXTURE_2D, 0), i !== void 0 && ve(t);
	}
	function be(t) {
		let i = t.texture, s = r.get(t), c = r.get(i);
		t.addEventListener("dispose", w);
		let l = t.textures, u = t.isWebGLCubeRenderTarget === !0, d = l.length > 1;
		if (d || (c.__webglTexture === void 0 && (c.__webglTexture = e.createTexture()), c.__version = i.version, o.memory.textures++), u) {
			s.__webglFramebuffer = [];
			for (let t = 0; t < 6; t++) if (i.mipmaps && i.mipmaps.length > 0) {
				s.__webglFramebuffer[t] = [];
				for (let n = 0; n < i.mipmaps.length; n++) s.__webglFramebuffer[t][n] = e.createFramebuffer();
			} else s.__webglFramebuffer[t] = e.createFramebuffer();
		} else {
			if (i.mipmaps && i.mipmaps.length > 0) {
				s.__webglFramebuffer = [];
				for (let t = 0; t < i.mipmaps.length; t++) s.__webglFramebuffer[t] = e.createFramebuffer();
			} else s.__webglFramebuffer = e.createFramebuffer();
			if (d) for (let t = 0, n = l.length; t < n; t++) {
				let n = r.get(l[t]);
				n.__webglTexture === void 0 && (n.__webglTexture = e.createTexture(), o.memory.textures++);
			}
			if (t.samples > 0 && M(t) === !1) {
				s.__webglMultisampledFramebuffer = e.createFramebuffer(), s.__webglColorRenderbuffer = [], n.bindFramebuffer(e.FRAMEBUFFER, s.__webglMultisampledFramebuffer);
				for (let n = 0; n < l.length; n++) {
					let r = l[n];
					s.__webglColorRenderbuffer[n] = e.createRenderbuffer(), e.bindRenderbuffer(e.RENDERBUFFER, s.__webglColorRenderbuffer[n]);
					let i = a.convert(r.format, r.colorSpace), o = a.convert(r.type), c = b(r.internalFormat, i, o, r.normalized, r.colorSpace, t.isXRRenderTarget === !0), u = Te(t);
					e.renderbufferStorageMultisample(e.RENDERBUFFER, u, c, t.width, t.height), e.framebufferRenderbuffer(e.FRAMEBUFFER, e.COLOR_ATTACHMENT0 + n, e.RENDERBUFFER, s.__webglColorRenderbuffer[n]);
				}
				e.bindRenderbuffer(e.RENDERBUFFER, null), t.depthBuffer && (s.__webglDepthRenderbuffer = e.createRenderbuffer(), ge(s.__webglDepthRenderbuffer, t, !0)), n.bindFramebuffer(e.FRAMEBUFFER, null);
			}
		}
		if (u) {
			n.bindTexture(e.TEXTURE_CUBE_MAP, c.__webglTexture), le(e.TEXTURE_CUBE_MAP, i);
			for (let n = 0; n < 6; n++) if (i.mipmaps && i.mipmaps.length > 0) for (let r = 0; r < i.mipmaps.length; r++) he(s.__webglFramebuffer[n][r], t, i, e.COLOR_ATTACHMENT0, e.TEXTURE_CUBE_MAP_POSITIVE_X + n, r);
			else he(s.__webglFramebuffer[n], t, i, e.COLOR_ATTACHMENT0, e.TEXTURE_CUBE_MAP_POSITIVE_X + n, 0);
			_(i) && v(e.TEXTURE_CUBE_MAP), n.unbindTexture();
		} else if (d) {
			for (let i = 0, a = l.length; i < a; i++) {
				let a = l[i], o = r.get(a), c = e.TEXTURE_2D;
				(t.isWebGL3DRenderTarget || t.isWebGLArrayRenderTarget) && (c = t.isWebGL3DRenderTarget ? e.TEXTURE_3D : e.TEXTURE_2D_ARRAY), n.bindTexture(c, o.__webglTexture), le(c, a), he(s.__webglFramebuffer, t, a, e.COLOR_ATTACHMENT0 + i, c, 0), _(a) && v(c);
			}
			n.unbindTexture();
		} else {
			let r = e.TEXTURE_2D;
			if ((t.isWebGL3DRenderTarget || t.isWebGLArrayRenderTarget) && (r = t.isWebGL3DRenderTarget ? e.TEXTURE_3D : e.TEXTURE_2D_ARRAY), n.bindTexture(r, c.__webglTexture), le(r, i), i.mipmaps && i.mipmaps.length > 0) for (let n = 0; n < i.mipmaps.length; n++) he(s.__webglFramebuffer[n], t, i, e.COLOR_ATTACHMENT0, r, n);
			else he(s.__webglFramebuffer, t, i, e.COLOR_ATTACHMENT0, r, 0);
			_(i) && v(r), n.unbindTexture();
		}
		t.depthBuffer && ve(t);
	}
	function xe(e) {
		let t = e.textures;
		for (let i = 0, a = t.length; i < a; i++) {
			let a = t[i];
			if (_(a)) {
				let t = y(e), i = r.get(a).__webglTexture;
				n.bindTexture(t, i), v(t), n.unbindTexture();
			}
		}
	}
	let Se = [], Ce = [];
	function we(t) {
		if (t.samples > 0) {
			if (M(t) === !1) {
				let i = t.textures, a = t.width, o = t.height, s = e.COLOR_BUFFER_BIT, l = t.stencilBuffer ? e.DEPTH_STENCIL_ATTACHMENT : e.DEPTH_ATTACHMENT, u = r.get(t), d = i.length > 1;
				if (d) for (let t = 0; t < i.length; t++) n.bindFramebuffer(e.FRAMEBUFFER, u.__webglMultisampledFramebuffer), e.framebufferRenderbuffer(e.FRAMEBUFFER, e.COLOR_ATTACHMENT0 + t, e.RENDERBUFFER, null), n.bindFramebuffer(e.FRAMEBUFFER, u.__webglFramebuffer), e.framebufferTexture2D(e.DRAW_FRAMEBUFFER, e.COLOR_ATTACHMENT0 + t, e.TEXTURE_2D, null, 0);
				n.bindFramebuffer(e.READ_FRAMEBUFFER, u.__webglMultisampledFramebuffer);
				let f = t.texture.mipmaps;
				f && f.length > 0 ? n.bindFramebuffer(e.DRAW_FRAMEBUFFER, u.__webglFramebuffer[0]) : n.bindFramebuffer(e.DRAW_FRAMEBUFFER, u.__webglFramebuffer);
				for (let n = 0; n < i.length; n++) {
					if (t.resolveDepthBuffer && (t.depthBuffer && (s |= e.DEPTH_BUFFER_BIT), t.stencilBuffer && t.resolveStencilBuffer && (s |= e.STENCIL_BUFFER_BIT)), d) {
						e.framebufferRenderbuffer(e.READ_FRAMEBUFFER, e.COLOR_ATTACHMENT0, e.RENDERBUFFER, u.__webglColorRenderbuffer[n]);
						let t = r.get(i[n]).__webglTexture;
						e.framebufferTexture2D(e.DRAW_FRAMEBUFFER, e.COLOR_ATTACHMENT0, e.TEXTURE_2D, t, 0);
					}
					e.blitFramebuffer(0, 0, a, o, 0, 0, a, o, s, e.NEAREST), c === !0 && (Se.length = 0, Ce.length = 0, Se.push(e.COLOR_ATTACHMENT0 + n), t.depthBuffer && t.resolveDepthBuffer === !1 && (Se.push(l), Ce.push(l), e.invalidateFramebuffer(e.DRAW_FRAMEBUFFER, Ce)), e.invalidateFramebuffer(e.READ_FRAMEBUFFER, Se));
				}
				if (n.bindFramebuffer(e.READ_FRAMEBUFFER, null), n.bindFramebuffer(e.DRAW_FRAMEBUFFER, null), d) for (let t = 0; t < i.length; t++) {
					n.bindFramebuffer(e.FRAMEBUFFER, u.__webglMultisampledFramebuffer), e.framebufferRenderbuffer(e.FRAMEBUFFER, e.COLOR_ATTACHMENT0 + t, e.RENDERBUFFER, u.__webglColorRenderbuffer[t]);
					let a = r.get(i[t]).__webglTexture;
					n.bindFramebuffer(e.FRAMEBUFFER, u.__webglFramebuffer), e.framebufferTexture2D(e.DRAW_FRAMEBUFFER, e.COLOR_ATTACHMENT0 + t, e.TEXTURE_2D, a, 0);
				}
				n.bindFramebuffer(e.DRAW_FRAMEBUFFER, u.__webglMultisampledFramebuffer);
			} else if (t.depthBuffer && t.resolveDepthBuffer === !1 && c) {
				let n = t.stencilBuffer ? e.DEPTH_STENCIL_ATTACHMENT : e.DEPTH_ATTACHMENT;
				e.invalidateFramebuffer(e.DRAW_FRAMEBUFFER, [n]);
			}
		}
	}
	function Te(e) {
		return Math.min(i.maxSamples, e.samples);
	}
	function M(e) {
		let n = r.get(e);
		return e.samples > 0 && t.has("WEBGL_multisampled_render_to_texture") === !0 && n.__useRenderToTexture !== !1;
	}
	function Ee(e) {
		let t = o.render.frame;
		u.get(e) !== t && (u.set(e, t), e.update());
	}
	function De(e, t) {
		let n = e.colorSpace, r = e.format, i = e.type;
		return e.isCompressedTexture === !0 || e.isVideoTexture === !0 || n !== "srgb-linear" && n !== "" && (X.getTransfer(n) === "srgb" ? (r !== 1023 || i !== 1009) && G("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType.") : K("WebGLTextures: Unsupported texture color space:", n)), t;
	}
	function Oe(e) {
		return typeof HTMLImageElement < "u" && e instanceof HTMLImageElement ? (l.width = e.naturalWidth || e.width, l.height = e.naturalHeight || e.height) : typeof VideoFrame < "u" && e instanceof VideoFrame ? (l.width = e.displayWidth, l.height = e.displayHeight) : (l.width = e.width, l.height = e.height), l;
	}
	this.allocateTextureUnit = te, this.resetTextureUnits = k, this.getTextureUnits = A, this.setTextureUnits = ee, this.setTexture2D = ne, this.setTexture2DArray = re, this.setTexture3D = ie, this.setTextureCube = ae, this.rebindTextures = ye, this.setupRenderTarget = be, this.updateRenderTargetMipmap = xe, this.updateMultisampleRenderTarget = we, this.setupDepthRenderbuffer = ve, this.setupFrameBufferTexture = he, this.useMultisampledRTT = M, this.isReversedDepthBuffer = function() {
		return n.buffers.depth.getReversed();
	};
}
function By(e, t) {
	function n(n, r = "") {
		let i, a = X.getTransfer(r);
		if (n === 1009) return e.UNSIGNED_BYTE;
		if (n === 1017) return e.UNSIGNED_SHORT_4_4_4_4;
		if (n === 1018) return e.UNSIGNED_SHORT_5_5_5_1;
		if (n === 35902) return e.UNSIGNED_INT_5_9_9_9_REV;
		if (n === 35899) return e.UNSIGNED_INT_10F_11F_11F_REV;
		if (n === 1010) return e.BYTE;
		if (n === 1011) return e.SHORT;
		if (n === 1012) return e.UNSIGNED_SHORT;
		if (n === 1013) return e.INT;
		if (n === 1014) return e.UNSIGNED_INT;
		if (n === 1015) return e.FLOAT;
		if (n === 1016) return e.HALF_FLOAT;
		if (n === 1021) return e.ALPHA;
		if (n === 1022) return e.RGB;
		if (n === 1023) return e.RGBA;
		if (n === 1026) return e.DEPTH_COMPONENT;
		if (n === 1027) return e.DEPTH_STENCIL;
		if (n === 1028) return e.RED;
		if (n === 1029) return e.RED_INTEGER;
		if (n === 1030) return e.RG;
		if (n === 1031) return e.RG_INTEGER;
		if (n === 1033) return e.RGBA_INTEGER;
		if (n === 33776 || n === 33777 || n === 33778 || n === 33779) if (a === "srgb") if (i = t.get("WEBGL_compressed_texture_s3tc_srgb"), i !== null) {
			if (n === 33776) return i.COMPRESSED_SRGB_S3TC_DXT1_EXT;
			if (n === 33777) return i.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
			if (n === 33778) return i.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
			if (n === 33779) return i.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
		} else return null;
		else if (i = t.get("WEBGL_compressed_texture_s3tc"), i !== null) {
			if (n === 33776) return i.COMPRESSED_RGB_S3TC_DXT1_EXT;
			if (n === 33777) return i.COMPRESSED_RGBA_S3TC_DXT1_EXT;
			if (n === 33778) return i.COMPRESSED_RGBA_S3TC_DXT3_EXT;
			if (n === 33779) return i.COMPRESSED_RGBA_S3TC_DXT5_EXT;
		} else return null;
		if (n === 35840 || n === 35841 || n === 35842 || n === 35843) if (i = t.get("WEBGL_compressed_texture_pvrtc"), i !== null) {
			if (n === 35840) return i.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;
			if (n === 35841) return i.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;
			if (n === 35842) return i.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;
			if (n === 35843) return i.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;
		} else return null;
		if (n === 36196 || n === 37492 || n === 37496 || n === 37488 || n === 37489 || n === 37490 || n === 37491) if (i = t.get("WEBGL_compressed_texture_etc"), i !== null) {
			if (n === 36196 || n === 37492) return a === "srgb" ? i.COMPRESSED_SRGB8_ETC2 : i.COMPRESSED_RGB8_ETC2;
			if (n === 37496) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC : i.COMPRESSED_RGBA8_ETC2_EAC;
			if (n === 37488) return i.COMPRESSED_R11_EAC;
			if (n === 37489) return i.COMPRESSED_SIGNED_R11_EAC;
			if (n === 37490) return i.COMPRESSED_RG11_EAC;
			if (n === 37491) return i.COMPRESSED_SIGNED_RG11_EAC;
		} else return null;
		if (n === 37808 || n === 37809 || n === 37810 || n === 37811 || n === 37812 || n === 37813 || n === 37814 || n === 37815 || n === 37816 || n === 37817 || n === 37818 || n === 37819 || n === 37820 || n === 37821) if (i = t.get("WEBGL_compressed_texture_astc"), i !== null) {
			if (n === 37808) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR : i.COMPRESSED_RGBA_ASTC_4x4_KHR;
			if (n === 37809) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR : i.COMPRESSED_RGBA_ASTC_5x4_KHR;
			if (n === 37810) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR : i.COMPRESSED_RGBA_ASTC_5x5_KHR;
			if (n === 37811) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR : i.COMPRESSED_RGBA_ASTC_6x5_KHR;
			if (n === 37812) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR : i.COMPRESSED_RGBA_ASTC_6x6_KHR;
			if (n === 37813) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR : i.COMPRESSED_RGBA_ASTC_8x5_KHR;
			if (n === 37814) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR : i.COMPRESSED_RGBA_ASTC_8x6_KHR;
			if (n === 37815) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR : i.COMPRESSED_RGBA_ASTC_8x8_KHR;
			if (n === 37816) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR : i.COMPRESSED_RGBA_ASTC_10x5_KHR;
			if (n === 37817) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR : i.COMPRESSED_RGBA_ASTC_10x6_KHR;
			if (n === 37818) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR : i.COMPRESSED_RGBA_ASTC_10x8_KHR;
			if (n === 37819) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR : i.COMPRESSED_RGBA_ASTC_10x10_KHR;
			if (n === 37820) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR : i.COMPRESSED_RGBA_ASTC_12x10_KHR;
			if (n === 37821) return a === "srgb" ? i.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR : i.COMPRESSED_RGBA_ASTC_12x12_KHR;
		} else return null;
		if (n === 36492 || n === 36494 || n === 36495) if (i = t.get("EXT_texture_compression_bptc"), i !== null) {
			if (n === 36492) return a === "srgb" ? i.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT : i.COMPRESSED_RGBA_BPTC_UNORM_EXT;
			if (n === 36494) return i.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;
			if (n === 36495) return i.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT;
		} else return null;
		if (n === 36283 || n === 36284 || n === 36285 || n === 36286) if (i = t.get("EXT_texture_compression_rgtc"), i !== null) {
			if (n === 36283) return i.COMPRESSED_RED_RGTC1_EXT;
			if (n === 36284) return i.COMPRESSED_SIGNED_RED_RGTC1_EXT;
			if (n === 36285) return i.COMPRESSED_RED_GREEN_RGTC2_EXT;
			if (n === 36286) return i.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
		} else return null;
		return n === 1020 ? e.UNSIGNED_INT_24_8 : e[n] === void 0 ? null : e[n];
	}
	return { convert: n };
}
var Vy = "\nvoid main() {\n\n	gl_Position = vec4( position, 1.0 );\n\n}", Hy = "\nuniform sampler2DArray depthColor;\nuniform float depthWidth;\nuniform float depthHeight;\n\nvoid main() {\n\n	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );\n\n	if ( coord.x >= 1.0 ) {\n\n		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;\n\n	} else {\n\n		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;\n\n	}\n\n}", Uy = class {
	constructor() {
		this.texture = null, this.mesh = null, this.depthNear = 0, this.depthFar = 0;
	}
	init(e, t) {
		if (this.texture === null) {
			let n = new kh(e.texture);
			(e.depthNear !== t.depthNear || e.depthFar !== t.depthFar) && (this.depthNear = e.depthNear, this.depthFar = e.depthFar), this.texture = n;
		}
	}
	getMesh(e) {
		if (this.texture !== null && this.mesh === null) {
			let t = e.cameras[0].viewport, n = new Bh({
				vertexShader: Vy,
				fragmentShader: Hy,
				uniforms: {
					depthColor: { value: this.texture },
					depthWidth: { value: t.z },
					depthHeight: { value: t.w }
				}
			});
			this.mesh = new qm(new jh(20, 20), n);
		}
		return this.mesh;
	}
	reset() {
		this.texture = null, this.mesh = null;
	}
	getDepthTexture() {
		return this.texture;
	}
}, Wy = class extends Zd {
	constructor(e, t) {
		super();
		let n = this, r = null, i = 1, a = null, o = "local-floor", s = 1, c = null, l = null, u = null, d = null, f = null, p = null, m = typeof XRWebGLBinding < "u", h = new Uy(), g = {}, _ = t.getContextAttributes(), v = null, y = null, b = [], x = [], S = new Tf(), C = null, w = new pg();
		w.viewport = new Uf();
		let T = new pg();
		T.viewport = new Uf();
		let E = [w, T], D = new vg(), O = null, k = null;
		this.cameraAutoUpdate = !0, this.enabled = !1, this.isPresenting = !1, this.getController = function(e) {
			let t = b[e];
			return t === void 0 && (t = new wp(), b[e] = t), t.getTargetRaySpace();
		}, this.getControllerGrip = function(e) {
			let t = b[e];
			return t === void 0 && (t = new wp(), b[e] = t), t.getGripSpace();
		}, this.getHand = function(e) {
			let t = b[e];
			return t === void 0 && (t = new wp(), b[e] = t), t.getHandSpace();
		};
		function A(e) {
			let t = x.indexOf(e.inputSource);
			if (t === -1) return;
			let n = b[t];
			n !== void 0 && (n.update(e.inputSource, e.frame, c || a), n.dispatchEvent({
				type: e.type,
				data: e.inputSource
			}));
		}
		function ee() {
			r.removeEventListener("select", A), r.removeEventListener("selectstart", A), r.removeEventListener("selectend", A), r.removeEventListener("squeeze", A), r.removeEventListener("squeezestart", A), r.removeEventListener("squeezeend", A), r.removeEventListener("end", ee), r.removeEventListener("inputsourceschange", te);
			for (let e = 0; e < b.length; e++) {
				let t = x[e];
				t !== null && (x[e] = null, b[e].disconnect(t));
			}
			O = null, k = null, h.reset();
			for (let e in g) delete g[e];
			e.setRenderTarget(v), f = null, d = null, u = null, r = null, y = null, ce.stop(), n.isPresenting = !1, e.setPixelRatio(C), e.setSize(S.width, S.height, !1), n.dispatchEvent({ type: "sessionend" });
		}
		this.setFramebufferScaleFactor = function(e) {
			i = e, n.isPresenting === !0 && G("WebXRManager: Cannot change framebuffer scale while presenting.");
		}, this.setReferenceSpaceType = function(e) {
			o = e, n.isPresenting === !0 && G("WebXRManager: Cannot change reference space type while presenting.");
		}, this.getReferenceSpace = function() {
			return c || a;
		}, this.setReferenceSpace = function(e) {
			c = e;
		}, this.getBaseLayer = function() {
			return d === null ? f : d;
		}, this.getBinding = function() {
			return u === null && m && (u = new XRWebGLBinding(r, t)), u;
		}, this.getFrame = function() {
			return p;
		}, this.getSession = function() {
			return r;
		}, this.setSession = async function(l) {
			if (r = l, r !== null) {
				if (v = e.getRenderTarget(), r.addEventListener("select", A), r.addEventListener("selectstart", A), r.addEventListener("selectend", A), r.addEventListener("squeeze", A), r.addEventListener("squeezestart", A), r.addEventListener("squeezeend", A), r.addEventListener("end", ee), r.addEventListener("inputsourceschange", te), _.xrCompatible !== !0 && await t.makeXRCompatible(), C = e.getPixelRatio(), e.getSize(S), m && "createProjectionLayer" in XRWebGLBinding.prototype) {
					let n = null, a = null, o = null;
					_.depth && (o = _.stencil ? t.DEPTH24_STENCIL8 : t.DEPTH_COMPONENT24, n = _.stencil ? Lu : Iu, a = _.stencil ? Au : Tu);
					let s = {
						colorFormat: t.RGBA8,
						depthFormat: o,
						scaleFactor: i
					};
					u = this.getBinding(), d = u.createProjectionLayer(s), r.updateRenderState({ layers: [d] }), e.setPixelRatio(1), e.setSize(d.textureWidth, d.textureHeight, !1), y = new Gf(d.textureWidth, d.textureHeight, {
						format: Fu,
						type: bu,
						depthTexture: new Dh(d.textureWidth, d.textureHeight, a, void 0, void 0, void 0, void 0, void 0, void 0, n),
						stencilBuffer: _.stencil,
						colorSpace: e.outputColorSpace,
						samples: _.antialias ? 4 : 0,
						resolveDepthBuffer: d.ignoreDepthValues === !1,
						resolveStencilBuffer: d.ignoreDepthValues === !1
					});
				} else {
					let n = {
						antialias: _.antialias,
						alpha: !0,
						depth: _.depth,
						stencil: _.stencil,
						framebufferScaleFactor: i
					};
					f = new XRWebGLLayer(r, t, n), r.updateRenderState({ baseLayer: f }), e.setPixelRatio(1), e.setSize(f.framebufferWidth, f.framebufferHeight, !1), y = new Gf(f.framebufferWidth, f.framebufferHeight, {
						format: Fu,
						type: bu,
						colorSpace: e.outputColorSpace,
						stencilBuffer: _.stencil,
						resolveDepthBuffer: f.ignoreDepthValues === !1,
						resolveStencilBuffer: f.ignoreDepthValues === !1
					});
				}
				y.isXRRenderTarget = !0, this.setFoveation(s), c = null, a = await r.requestReferenceSpace(o), ce.setContext(r), ce.start(), n.isPresenting = !0, n.dispatchEvent({ type: "sessionstart" });
			}
		}, this.getEnvironmentBlendMode = function() {
			if (r !== null) return r.environmentBlendMode;
		}, this.getDepthTexture = function() {
			return h.getDepthTexture();
		};
		function te(e) {
			for (let t = 0; t < e.removed.length; t++) {
				let n = e.removed[t], r = x.indexOf(n);
				r >= 0 && (x[r] = null, b[r].disconnect(n));
			}
			for (let t = 0; t < e.added.length; t++) {
				let n = e.added[t], r = x.indexOf(n);
				if (r === -1) {
					for (let e = 0; e < b.length; e++) if (e >= x.length) {
						x.push(n), r = e;
						break;
					} else if (x[e] === null) {
						x[e] = n, r = e;
						break;
					}
					if (r === -1) break;
				}
				let i = b[r];
				i && i.connect(n);
			}
		}
		let j = new J(), ne = new J();
		function re(e, t, n) {
			j.setFromMatrixPosition(t.matrixWorld), ne.setFromMatrixPosition(n.matrixWorld);
			let r = j.distanceTo(ne), i = t.projectionMatrix.elements, a = n.projectionMatrix.elements, o = i[14] / (i[10] - 1), s = i[14] / (i[10] + 1), c = (i[9] + 1) / i[5], l = (i[9] - 1) / i[5], u = (i[8] - 1) / i[0], d = (a[8] + 1) / a[0], f = o * u, p = o * d, m = r / (-u + d), h = m * -u;
			if (t.matrixWorld.decompose(e.position, e.quaternion, e.scale), e.translateX(h), e.translateZ(m), e.matrixWorld.compose(e.position, e.quaternion, e.scale), e.matrixWorldInverse.copy(e.matrixWorld).invert(), i[10] === -1) e.projectionMatrix.copy(t.projectionMatrix), e.projectionMatrixInverse.copy(t.projectionMatrixInverse);
			else {
				let t = o + m, n = s + m, i = f - h, a = p + (r - h), u = c * s / n * t, d = l * s / n * t;
				e.projectionMatrix.makePerspective(i, a, u, d, t, n), e.projectionMatrixInverse.copy(e.projectionMatrix).invert();
			}
		}
		function ie(e, t) {
			t === null ? e.matrixWorld.copy(e.matrix) : e.matrixWorld.multiplyMatrices(t.matrixWorld, e.matrix), e.matrixWorldInverse.copy(e.matrixWorld).invert();
		}
		this.updateCamera = function(e) {
			if (r === null) return;
			let t = e.near, n = e.far;
			h.texture !== null && (h.depthNear > 0 && (t = h.depthNear), h.depthFar > 0 && (n = h.depthFar)), D.near = T.near = w.near = t, D.far = T.far = w.far = n, (O !== D.near || k !== D.far) && (r.updateRenderState({
				depthNear: D.near,
				depthFar: D.far
			}), O = D.near, k = D.far), D.layers.mask = e.layers.mask | 6, w.layers.mask = D.layers.mask & -5, T.layers.mask = D.layers.mask & -3;
			let i = e.parent, a = D.cameras;
			ie(D, i);
			for (let e = 0; e < a.length; e++) ie(a[e], i);
			a.length === 2 ? re(D, w, T) : D.projectionMatrix.copy(w.projectionMatrix), ae(e, D, i);
		};
		function ae(e, t, n) {
			n === null ? e.matrix.copy(t.matrixWorld) : (e.matrix.copy(n.matrixWorld), e.matrix.invert(), e.matrix.multiply(t.matrixWorld)), e.matrix.decompose(e.position, e.quaternion, e.scale), e.updateMatrixWorld(!0), e.projectionMatrix.copy(t.projectionMatrix), e.projectionMatrixInverse.copy(t.projectionMatrixInverse), e.isPerspectiveCamera && (e.fov = tf * 2 * Math.atan(1 / e.projectionMatrix.elements[5]), e.zoom = 1);
		}
		this.getCamera = function() {
			return D;
		}, this.getFoveation = function() {
			if (!(d === null && f === null)) return s;
		}, this.setFoveation = function(e) {
			s = e, d !== null && (d.fixedFoveation = e), f !== null && f.fixedFoveation !== void 0 && (f.fixedFoveation = e);
		}, this.hasDepthSensing = function() {
			return h.texture !== null;
		}, this.getDepthSensingMesh = function() {
			return h.getMesh(D);
		}, this.getCameraTexture = function(e) {
			return g[e];
		};
		let oe = null;
		function se(t, i) {
			if (l = i.getViewerPose(c || a), p = i, l !== null) {
				let t = l.views;
				f !== null && (e.setRenderTargetFramebuffer(y, f.framebuffer), e.setRenderTarget(y));
				let i = !1;
				t.length !== D.cameras.length && (D.cameras.length = 0, i = !0);
				for (let n = 0; n < t.length; n++) {
					let r = t[n], a = null;
					if (f !== null) a = f.getViewport(r);
					else {
						let t = u.getViewSubImage(d, r);
						a = t.viewport, n === 0 && (e.setRenderTargetTextures(y, t.colorTexture, t.depthStencilTexture), e.setRenderTarget(y));
					}
					let o = E[n];
					o === void 0 && (o = new pg(), o.layers.enable(n), o.viewport = new Uf(), E[n] = o), o.matrix.fromArray(r.transform.matrix), o.matrix.decompose(o.position, o.quaternion, o.scale), o.projectionMatrix.fromArray(r.projectionMatrix), o.projectionMatrixInverse.copy(o.projectionMatrix).invert(), o.viewport.set(a.x, a.y, a.width, a.height), n === 0 && (D.matrix.copy(o.matrix), D.matrix.decompose(D.position, D.quaternion, D.scale)), i === !0 && D.cameras.push(o);
				}
				let a = r.enabledFeatures;
				if (a && a.includes("depth-sensing") && r.depthUsage == "gpu-optimized" && m) {
					u = n.getBinding();
					let e = u.getDepthInformation(t[0]);
					e && e.isValid && e.texture && h.init(e, r.renderState);
				}
				if (a && a.includes("camera-access") && m) {
					e.state.unbindTexture(), u = n.getBinding();
					for (let e = 0; e < t.length; e++) {
						let n = t[e].camera;
						if (n) {
							let e = g[n];
							e || (e = new kh(), g[n] = e);
							let t = u.getCameraImage(n);
							e.sourceTexture = t;
						}
					}
				}
			}
			for (let e = 0; e < b.length; e++) {
				let t = x[e], n = b[e];
				t !== null && n !== void 0 && n.update(t, i, c || a);
			}
			oe && oe(t, i), i.detectedPlanes && n.dispatchEvent({
				type: "planesdetected",
				data: i
			}), p = null;
		}
		let ce = new Lg();
		ce.setAnimationLoop(se), this.setAnimationLoop = function(e) {
			oe = e;
		}, this.dispose = function() {};
	}
}, Gy = /*@__PURE__*/ new Jf(), Ky = /*@__PURE__*/ new Y();
Ky.set(-1, 0, 0, 0, 1, 0, 0, 0, 1);
function qy(e, t) {
	function n(e, t) {
		e.matrixAutoUpdate === !0 && e.updateMatrix(), t.value.copy(e.matrix);
	}
	function r(t, n) {
		n.color.getRGB(t.fogColor.value, Ih(e)), n.isFog ? (t.fogNear.value = n.near, t.fogFar.value = n.far) : n.isFogExp2 && (t.fogDensity.value = n.density);
	}
	function i(e, t, n, r, i) {
		t.isNodeMaterial ? t.uniformsNeedUpdate = !1 : t.isMeshBasicMaterial ? a(e, t) : t.isMeshLambertMaterial ? (a(e, t), t.envMap && (e.envMapIntensity.value = t.envMapIntensity)) : t.isMeshToonMaterial ? (a(e, t), d(e, t)) : t.isMeshPhongMaterial ? (a(e, t), u(e, t), t.envMap && (e.envMapIntensity.value = t.envMapIntensity)) : t.isMeshStandardMaterial ? (a(e, t), f(e, t), t.isMeshPhysicalMaterial && p(e, t, i)) : t.isMeshMatcapMaterial ? (a(e, t), m(e, t)) : t.isMeshDepthMaterial ? a(e, t) : t.isMeshDistanceMaterial ? (a(e, t), h(e, t)) : t.isMeshNormalMaterial ? a(e, t) : t.isLineBasicMaterial ? (o(e, t), t.isLineDashedMaterial && s(e, t)) : t.isPointsMaterial ? c(e, t, n, r) : t.isSpriteMaterial ? l(e, t) : t.isShadowMaterial ? (e.color.value.copy(t.color), e.opacity.value = t.opacity) : t.isShaderMaterial && (t.uniformsNeedUpdate = !1);
	}
	function a(e, r) {
		e.opacity.value = r.opacity, r.color && e.diffuse.value.copy(r.color), r.emissive && e.emissive.value.copy(r.emissive).multiplyScalar(r.emissiveIntensity), r.map && (e.map.value = r.map, n(r.map, e.mapTransform)), r.alphaMap && (e.alphaMap.value = r.alphaMap, n(r.alphaMap, e.alphaMapTransform)), r.bumpMap && (e.bumpMap.value = r.bumpMap, n(r.bumpMap, e.bumpMapTransform), e.bumpScale.value = r.bumpScale, r.side === 1 && (e.bumpScale.value *= -1)), r.normalMap && (e.normalMap.value = r.normalMap, n(r.normalMap, e.normalMapTransform), e.normalScale.value.copy(r.normalScale), r.side === 1 && e.normalScale.value.negate()), r.displacementMap && (e.displacementMap.value = r.displacementMap, n(r.displacementMap, e.displacementMapTransform), e.displacementScale.value = r.displacementScale, e.displacementBias.value = r.displacementBias), r.emissiveMap && (e.emissiveMap.value = r.emissiveMap, n(r.emissiveMap, e.emissiveMapTransform)), r.specularMap && (e.specularMap.value = r.specularMap, n(r.specularMap, e.specularMapTransform)), r.alphaTest > 0 && (e.alphaTest.value = r.alphaTest);
		let i = t.get(r), a = i.envMap, o = i.envMapRotation;
		a && (e.envMap.value = a, e.envMapRotation.value.setFromMatrix4(Gy.makeRotationFromEuler(o)).transpose(), a.isCubeTexture && a.isRenderTargetTexture === !1 && e.envMapRotation.value.premultiply(Ky), e.reflectivity.value = r.reflectivity, e.ior.value = r.ior, e.refractionRatio.value = r.refractionRatio), r.lightMap && (e.lightMap.value = r.lightMap, e.lightMapIntensity.value = r.lightMapIntensity, n(r.lightMap, e.lightMapTransform)), r.aoMap && (e.aoMap.value = r.aoMap, e.aoMapIntensity.value = r.aoMapIntensity, n(r.aoMap, e.aoMapTransform));
	}
	function o(e, t) {
		e.diffuse.value.copy(t.color), e.opacity.value = t.opacity, t.map && (e.map.value = t.map, n(t.map, e.mapTransform));
	}
	function s(e, t) {
		e.dashSize.value = t.dashSize, e.totalSize.value = t.dashSize + t.gapSize, e.scale.value = t.scale;
	}
	function c(e, t, r, i) {
		e.diffuse.value.copy(t.color), e.opacity.value = t.opacity, e.size.value = t.size * r, e.scale.value = i * .5, t.map && (e.map.value = t.map, n(t.map, e.uvTransform)), t.alphaMap && (e.alphaMap.value = t.alphaMap, n(t.alphaMap, e.alphaMapTransform)), t.alphaTest > 0 && (e.alphaTest.value = t.alphaTest);
	}
	function l(e, t) {
		e.diffuse.value.copy(t.color), e.opacity.value = t.opacity, e.rotation.value = t.rotation, t.map && (e.map.value = t.map, n(t.map, e.mapTransform)), t.alphaMap && (e.alphaMap.value = t.alphaMap, n(t.alphaMap, e.alphaMapTransform)), t.alphaTest > 0 && (e.alphaTest.value = t.alphaTest);
	}
	function u(e, t) {
		e.specular.value.copy(t.specular), e.shininess.value = Math.max(t.shininess, 1e-4);
	}
	function d(e, t) {
		t.gradientMap && (e.gradientMap.value = t.gradientMap);
	}
	function f(e, t) {
		e.metalness.value = t.metalness, t.metalnessMap && (e.metalnessMap.value = t.metalnessMap, n(t.metalnessMap, e.metalnessMapTransform)), e.roughness.value = t.roughness, t.roughnessMap && (e.roughnessMap.value = t.roughnessMap, n(t.roughnessMap, e.roughnessMapTransform)), t.envMap && (e.envMapIntensity.value = t.envMapIntensity);
	}
	function p(e, t, r) {
		e.ior.value = t.ior, t.sheen > 0 && (e.sheenColor.value.copy(t.sheenColor).multiplyScalar(t.sheen), e.sheenRoughness.value = t.sheenRoughness, t.sheenColorMap && (e.sheenColorMap.value = t.sheenColorMap, n(t.sheenColorMap, e.sheenColorMapTransform)), t.sheenRoughnessMap && (e.sheenRoughnessMap.value = t.sheenRoughnessMap, n(t.sheenRoughnessMap, e.sheenRoughnessMapTransform))), t.clearcoat > 0 && (e.clearcoat.value = t.clearcoat, e.clearcoatRoughness.value = t.clearcoatRoughness, t.clearcoatMap && (e.clearcoatMap.value = t.clearcoatMap, n(t.clearcoatMap, e.clearcoatMapTransform)), t.clearcoatRoughnessMap && (e.clearcoatRoughnessMap.value = t.clearcoatRoughnessMap, n(t.clearcoatRoughnessMap, e.clearcoatRoughnessMapTransform)), t.clearcoatNormalMap && (e.clearcoatNormalMap.value = t.clearcoatNormalMap, n(t.clearcoatNormalMap, e.clearcoatNormalMapTransform), e.clearcoatNormalScale.value.copy(t.clearcoatNormalScale), t.side === 1 && e.clearcoatNormalScale.value.negate())), t.dispersion > 0 && (e.dispersion.value = t.dispersion), t.iridescence > 0 && (e.iridescence.value = t.iridescence, e.iridescenceIOR.value = t.iridescenceIOR, e.iridescenceThicknessMinimum.value = t.iridescenceThicknessRange[0], e.iridescenceThicknessMaximum.value = t.iridescenceThicknessRange[1], t.iridescenceMap && (e.iridescenceMap.value = t.iridescenceMap, n(t.iridescenceMap, e.iridescenceMapTransform)), t.iridescenceThicknessMap && (e.iridescenceThicknessMap.value = t.iridescenceThicknessMap, n(t.iridescenceThicknessMap, e.iridescenceThicknessMapTransform))), t.transmission > 0 && (e.transmission.value = t.transmission, e.transmissionSamplerMap.value = r.texture, e.transmissionSamplerSize.value.set(r.width, r.height), t.transmissionMap && (e.transmissionMap.value = t.transmissionMap, n(t.transmissionMap, e.transmissionMapTransform)), e.thickness.value = t.thickness, t.thicknessMap && (e.thicknessMap.value = t.thicknessMap, n(t.thicknessMap, e.thicknessMapTransform)), e.attenuationDistance.value = t.attenuationDistance, e.attenuationColor.value.copy(t.attenuationColor)), t.anisotropy > 0 && (e.anisotropyVector.value.set(t.anisotropy * Math.cos(t.anisotropyRotation), t.anisotropy * Math.sin(t.anisotropyRotation)), t.anisotropyMap && (e.anisotropyMap.value = t.anisotropyMap, n(t.anisotropyMap, e.anisotropyMapTransform))), e.specularIntensity.value = t.specularIntensity, e.specularColor.value.copy(t.specularColor), t.specularColorMap && (e.specularColorMap.value = t.specularColorMap, n(t.specularColorMap, e.specularColorMapTransform)), t.specularIntensityMap && (e.specularIntensityMap.value = t.specularIntensityMap, n(t.specularIntensityMap, e.specularIntensityMapTransform));
	}
	function m(e, t) {
		t.matcap && (e.matcap.value = t.matcap);
	}
	function h(e, n) {
		let r = t.get(n).light;
		e.referencePosition.value.setFromMatrixPosition(r.matrixWorld), e.nearDistance.value = r.shadow.camera.near, e.farDistance.value = r.shadow.camera.far;
	}
	return {
		refreshFogUniforms: r,
		refreshMaterialUniforms: i
	};
}
function Jy(e, t, n, r) {
	let i = {}, a = {}, o = [], s = e.getParameter(e.MAX_UNIFORM_BUFFER_BINDINGS);
	function c(e, t) {
		let n = t.program;
		r.uniformBlockBinding(e, n);
	}
	function l(e, n) {
		let o = i[e.id];
		o === void 0 && (g(e), o = u(e), i[e.id] = o, e.addEventListener("dispose", v));
		let s = n.program;
		r.updateUBOMapping(e, s);
		let c = t.render.frame;
		a[e.id] !== c && (f(e), a[e.id] = c);
	}
	function u(t) {
		let n = d();
		t.__bindingPointIndex = n;
		let r = e.createBuffer(), i = t.__size, a = t.usage;
		return e.bindBuffer(e.UNIFORM_BUFFER, r), e.bufferData(e.UNIFORM_BUFFER, i, a), e.bindBuffer(e.UNIFORM_BUFFER, null), e.bindBufferBase(e.UNIFORM_BUFFER, n, r), r;
	}
	function d() {
		for (let e = 0; e < s; e++) if (o.indexOf(e) === -1) return o.push(e), e;
		return K("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."), 0;
	}
	function f(t) {
		let n = i[t.id], r = t.uniforms, a = t.__cache;
		e.bindBuffer(e.UNIFORM_BUFFER, n);
		for (let e = 0, t = r.length; e < t; e++) {
			let t = r[e];
			if (Array.isArray(t)) for (let n = 0, r = t.length; n < r; n++) p(t[n], e, n, a);
			else p(t, e, 0, a);
		}
		e.bindBuffer(e.UNIFORM_BUFFER, null);
	}
	function p(t, n, r, i) {
		if (h(t, n, r, i) === !0) {
			let n = t.__offset, r = t.value;
			if (Array.isArray(r)) {
				let e = 0;
				for (let n = 0; n < r.length; n++) {
					let i = r[n], a = _(i);
					m(i, t.__data, e), typeof i != "number" && typeof i != "boolean" && !i.isMatrix3 && !ArrayBuffer.isView(i) && (e += a.storage / Float32Array.BYTES_PER_ELEMENT);
				}
			} else m(r, t.__data, 0);
			e.bufferSubData(e.UNIFORM_BUFFER, n, t.__data);
		}
	}
	function m(e, t, n) {
		typeof e == "number" || typeof e == "boolean" ? t[0] = e : e.isMatrix3 ? (t[0] = e.elements[0], t[1] = e.elements[1], t[2] = e.elements[2], t[3] = 0, t[4] = e.elements[3], t[5] = e.elements[4], t[6] = e.elements[5], t[7] = 0, t[8] = e.elements[6], t[9] = e.elements[7], t[10] = e.elements[8], t[11] = 0) : ArrayBuffer.isView(e) ? t.set(new e.constructor(e.buffer, e.byteOffset, t.length)) : e.toArray(t, n);
	}
	function h(e, t, n, r) {
		let i = e.value, a = t + "_" + n;
		if (r[a] === void 0) return typeof i == "number" || typeof i == "boolean" ? r[a] = i : ArrayBuffer.isView(i) ? r[a] = i.slice() : r[a] = i.clone(), !0;
		{
			let e = r[a];
			if (typeof i == "number" || typeof i == "boolean") {
				if (e !== i) return r[a] = i, !0;
			} else if (ArrayBuffer.isView(i)) return !0;
			else if (e.equals(i) === !1) return e.copy(i), !0;
		}
		return !1;
	}
	function g(e) {
		let t = e.uniforms, n = 0;
		for (let e = 0, r = t.length; e < r; e++) {
			let r = Array.isArray(t[e]) ? t[e] : [t[e]];
			for (let e = 0, t = r.length; e < t; e++) {
				let t = r[e], i = Array.isArray(t.value) ? t.value : [t.value];
				for (let e = 0, r = i.length; e < r; e++) {
					let r = i[e], a = _(r), o = n % 16, s = o % a.boundary, c = o + s;
					n += s, c !== 0 && 16 - c < a.storage && (n += 16 - c), t.__data = new Float32Array(a.storage / Float32Array.BYTES_PER_ELEMENT), t.__offset = n, n += a.storage;
				}
			}
		}
		let r = n % 16;
		return r > 0 && (n += 16 - r), e.__size = n, e.__cache = {}, this;
	}
	function _(e) {
		let t = {
			boundary: 0,
			storage: 0
		};
		return typeof e == "number" || typeof e == "boolean" ? (t.boundary = 4, t.storage = 4) : e.isVector2 ? (t.boundary = 8, t.storage = 8) : e.isVector3 || e.isColor ? (t.boundary = 16, t.storage = 12) : e.isVector4 ? (t.boundary = 16, t.storage = 16) : e.isMatrix3 ? (t.boundary = 48, t.storage = 48) : e.isMatrix4 ? (t.boundary = 64, t.storage = 64) : e.isTexture ? G("WebGLRenderer: Texture samplers can not be part of an uniforms group.") : ArrayBuffer.isView(e) ? (t.boundary = 16, t.storage = e.byteLength) : G("WebGLRenderer: Unsupported uniform value type.", e), t;
	}
	function v(t) {
		let n = t.target;
		n.removeEventListener("dispose", v);
		let r = o.indexOf(n.__bindingPointIndex);
		o.splice(r, 1), e.deleteBuffer(i[n.id]), delete i[n.id], delete a[n.id];
	}
	function y() {
		for (let t in i) e.deleteBuffer(i[t]);
		o = [], i = {}, a = {};
	}
	return {
		bind: c,
		update: l,
		dispose: y
	};
}
var Yy = new Uint16Array([
	12469,
	15057,
	12620,
	14925,
	13266,
	14620,
	13807,
	14376,
	14323,
	13990,
	14545,
	13625,
	14713,
	13328,
	14840,
	12882,
	14931,
	12528,
	14996,
	12233,
	15039,
	11829,
	15066,
	11525,
	15080,
	11295,
	15085,
	10976,
	15082,
	10705,
	15073,
	10495,
	13880,
	14564,
	13898,
	14542,
	13977,
	14430,
	14158,
	14124,
	14393,
	13732,
	14556,
	13410,
	14702,
	12996,
	14814,
	12596,
	14891,
	12291,
	14937,
	11834,
	14957,
	11489,
	14958,
	11194,
	14943,
	10803,
	14921,
	10506,
	14893,
	10278,
	14858,
	9960,
	14484,
	14039,
	14487,
	14025,
	14499,
	13941,
	14524,
	13740,
	14574,
	13468,
	14654,
	13106,
	14743,
	12678,
	14818,
	12344,
	14867,
	11893,
	14889,
	11509,
	14893,
	11180,
	14881,
	10751,
	14852,
	10428,
	14812,
	10128,
	14765,
	9754,
	14712,
	9466,
	14764,
	13480,
	14764,
	13475,
	14766,
	13440,
	14766,
	13347,
	14769,
	13070,
	14786,
	12713,
	14816,
	12387,
	14844,
	11957,
	14860,
	11549,
	14868,
	11215,
	14855,
	10751,
	14825,
	10403,
	14782,
	10044,
	14729,
	9651,
	14666,
	9352,
	14599,
	9029,
	14967,
	12835,
	14966,
	12831,
	14963,
	12804,
	14954,
	12723,
	14936,
	12564,
	14917,
	12347,
	14900,
	11958,
	14886,
	11569,
	14878,
	11247,
	14859,
	10765,
	14828,
	10401,
	14784,
	10011,
	14727,
	9600,
	14660,
	9289,
	14586,
	8893,
	14508,
	8533,
	15111,
	12234,
	15110,
	12234,
	15104,
	12216,
	15092,
	12156,
	15067,
	12010,
	15028,
	11776,
	14981,
	11500,
	14942,
	11205,
	14902,
	10752,
	14861,
	10393,
	14812,
	9991,
	14752,
	9570,
	14682,
	9252,
	14603,
	8808,
	14519,
	8445,
	14431,
	8145,
	15209,
	11449,
	15208,
	11451,
	15202,
	11451,
	15190,
	11438,
	15163,
	11384,
	15117,
	11274,
	15055,
	10979,
	14994,
	10648,
	14932,
	10343,
	14871,
	9936,
	14803,
	9532,
	14729,
	9218,
	14645,
	8742,
	14556,
	8381,
	14461,
	8020,
	14365,
	7603,
	15273,
	10603,
	15272,
	10607,
	15267,
	10619,
	15256,
	10631,
	15231,
	10614,
	15182,
	10535,
	15118,
	10389,
	15042,
	10167,
	14963,
	9787,
	14883,
	9447,
	14800,
	9115,
	14710,
	8665,
	14615,
	8318,
	14514,
	7911,
	14411,
	7507,
	14279,
	7198,
	15314,
	9675,
	15313,
	9683,
	15309,
	9712,
	15298,
	9759,
	15277,
	9797,
	15229,
	9773,
	15166,
	9668,
	15084,
	9487,
	14995,
	9274,
	14898,
	8910,
	14800,
	8539,
	14697,
	8234,
	14590,
	7790,
	14479,
	7409,
	14367,
	7067,
	14178,
	6621,
	15337,
	8619,
	15337,
	8631,
	15333,
	8677,
	15325,
	8769,
	15305,
	8871,
	15264,
	8940,
	15202,
	8909,
	15119,
	8775,
	15022,
	8565,
	14916,
	8328,
	14804,
	8009,
	14688,
	7614,
	14569,
	7287,
	14448,
	6888,
	14321,
	6483,
	14088,
	6171,
	15350,
	7402,
	15350,
	7419,
	15347,
	7480,
	15340,
	7613,
	15322,
	7804,
	15287,
	7973,
	15229,
	8057,
	15148,
	8012,
	15046,
	7846,
	14933,
	7611,
	14810,
	7357,
	14682,
	7069,
	14552,
	6656,
	14421,
	6316,
	14251,
	5948,
	14007,
	5528,
	15356,
	5942,
	15356,
	5977,
	15353,
	6119,
	15348,
	6294,
	15332,
	6551,
	15302,
	6824,
	15249,
	7044,
	15171,
	7122,
	15070,
	7050,
	14949,
	6861,
	14818,
	6611,
	14679,
	6349,
	14538,
	6067,
	14398,
	5651,
	14189,
	5311,
	13935,
	4958,
	15359,
	4123,
	15359,
	4153,
	15356,
	4296,
	15353,
	4646,
	15338,
	5160,
	15311,
	5508,
	15263,
	5829,
	15188,
	6042,
	15088,
	6094,
	14966,
	6001,
	14826,
	5796,
	14678,
	5543,
	14527,
	5287,
	14377,
	4985,
	14133,
	4586,
	13869,
	4257,
	15360,
	1563,
	15360,
	1642,
	15358,
	2076,
	15354,
	2636,
	15341,
	3350,
	15317,
	4019,
	15273,
	4429,
	15203,
	4732,
	15105,
	4911,
	14981,
	4932,
	14836,
	4818,
	14679,
	4621,
	14517,
	4386,
	14359,
	4156,
	14083,
	3795,
	13808,
	3437,
	15360,
	122,
	15360,
	137,
	15358,
	285,
	15355,
	636,
	15344,
	1274,
	15322,
	2177,
	15281,
	2765,
	15215,
	3223,
	15120,
	3451,
	14995,
	3569,
	14846,
	3567,
	14681,
	3466,
	14511,
	3305,
	14344,
	3121,
	14037,
	2800,
	13753,
	2467,
	15360,
	0,
	15360,
	1,
	15359,
	21,
	15355,
	89,
	15346,
	253,
	15325,
	479,
	15287,
	796,
	15225,
	1148,
	15133,
	1492,
	15008,
	1749,
	14856,
	1882,
	14685,
	1886,
	14506,
	1783,
	14324,
	1608,
	13996,
	1398,
	13702,
	1183
]), Xy = null;
function Zy() {
	return Xy === null && (Xy = new Xm(Yy, 16, 16, Bu, Du), Xy.name = "DFG_LUT", Xy.minFilter = _u, Xy.magFilter = _u, Xy.wrapS = fu, Xy.wrapT = fu, Xy.generateMipmaps = !1, Xy.needsUpdate = !0), Xy;
}
var Qy = class {
	constructor(e = {}) {
		let { canvas: t = Wd(), context: n = null, depth: r = !0, stencil: i = !1, alpha: a = !1, antialias: o = !1, premultipliedAlpha: s = !0, preserveDrawingBuffer: c = !1, powerPreference: l = "default", failIfMajorPerformanceCaveat: u = !1, reversedDepthBuffer: d = !1, outputBufferType: f = bu } = e;
		this.isWebGLRenderer = !0;
		let p;
		if (n !== null) {
			if (typeof WebGLRenderingContext < "u" && n instanceof WebGLRenderingContext) throw Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");
			p = n.getContextAttributes().alpha;
		} else p = a;
		let m = f, h = /* @__PURE__ */ new Set([
			Hu,
			Vu,
			zu
		]), g = /* @__PURE__ */ new Set([
			bu,
			Tu,
			Cu,
			Au,
			Ou,
			ku
		]), _ = /* @__PURE__ */ new Uint32Array(4), v = /* @__PURE__ */ new Int32Array(4), y = new J(), b = null, x = null, S = [], C = [], w = null;
		this.domElement = t, this.debug = {
			checkShaderErrors: !0,
			onShaderError: null
		}, this.autoClear = !0, this.autoClearColor = !0, this.autoClearDepth = !0, this.autoClearStencil = !0, this.sortObjects = !0, this.clippingPlanes = [], this.localClippingEnabled = !1, this.toneMapping = 0, this.toneMappingExposure = 1, this.transmissionResolutionScale = 1;
		let T = this, E = !1, D = null, O = null, k = null, A = null;
		this._outputColorSpace = Nd;
		let ee = 0, te = 0, j = null, ne = -1, re = null, ie = new Uf(), ae = new Uf(), oe = null, se = new Z(0), ce = 0, le = t.width, ue = t.height, de = 1, fe = null, pe = null, me = new Uf(0, 0, le, ue), he = new Uf(0, 0, le, ue), ge = !1, _e = new ih(), ve = !1, ye = !1, be = new Jf(), xe = new J(), Se = new Uf(), Ce = {
			background: null,
			fog: null,
			environment: null,
			overrideMaterial: null,
			isScene: !0
		}, we = !1;
		function Te() {
			return j === null ? de : 1;
		}
		let M = n;
		function Ee(e, n) {
			return t.getContext(e, n);
		}
		try {
			let e = {
				alpha: !0,
				depth: r,
				stencil: i,
				antialias: o,
				premultipliedAlpha: s,
				preserveDrawingBuffer: c,
				powerPreference: l,
				failIfMajorPerformanceCaveat: u
			};
			if ("setAttribute" in t && t.setAttribute("data-engine", "three.js r185"), t.addEventListener("webglcontextlost", Ye, !1), t.addEventListener("webglcontextrestored", Xe, !1), t.addEventListener("webglcontextcreationerror", Ze, !1), M === null) {
				let t = "webgl2";
				if (M = Ee(t, e), M === null) throw Ee(t) ? Error("THREE.WebGLRenderer: Error creating WebGL context with your selected attributes.") : Error("THREE.WebGLRenderer: Error creating WebGL context.");
			}
		} catch (e) {
			throw K("WebGLRenderer: " + e.message), e;
		}
		let De, Oe, N, ke, P, F, Ae, je, Me, Ne, Pe, Fe, Ie, Le, Re, ze, Be, Ve, He, Ue, We, Ge, Ke;
		function qe() {
			De = new g_(M), De.init(), We = new By(M, De), Oe = new Kg(M, De, e, We), N = new Ry(M, De), Oe.reversedDepthBuffer && d && N.buffers.depth.setReversed(!0), O = M.createFramebuffer(), k = M.createFramebuffer(), A = M.createFramebuffer(), ke = new y_(M), P = new vy(), F = new zy(M, De, N, P, Oe, We, ke), Ae = new h_(T), je = new Rg(M), Ge = new Wg(M, je), Me = new __(M, je, ke, Ge), Ne = new x_(M, Me, je, Ge, ke), Ve = new b_(M, Oe, F), Re = new qg(P), Pe = new _y(T, Ae, De, Oe, Ge, Re), Fe = new qy(T, P), Ie = new Sy(), Le = new ky(De), Be = new Ug(T, Ae, N, Ne, p, s), ze = new Ly(T, Ne, Oe), Ke = new Jy(M, ke, Oe, N), He = new Gg(M, De, ke), Ue = new v_(M, De, ke), ke.programs = Pe.programs, T.capabilities = Oe, T.extensions = De, T.properties = P, T.renderLists = Ie, T.shadowMap = ze, T.state = N, T.info = ke;
		}
		qe(), m !== 1009 && (w = new C_(m, t.width, t.height, o, r, i));
		let Je = new Wy(T, M);
		this.xr = Je, this.getContext = function() {
			return M;
		}, this.getContextAttributes = function() {
			return M.getContextAttributes();
		}, this.forceContextLoss = function() {
			let e = De.get("WEBGL_lose_context");
			e && e.loseContext();
		}, this.forceContextRestore = function() {
			let e = De.get("WEBGL_lose_context");
			e && e.restoreContext();
		}, this.getPixelRatio = function() {
			return de;
		}, this.setPixelRatio = function(e) {
			e !== void 0 && (de = e, this.setSize(le, ue, !1));
		}, this.getSize = function(e) {
			return e.set(le, ue);
		}, this.setSize = function(e, n, r = !0) {
			if (Je.isPresenting) {
				G("WebGLRenderer: Can't change size while VR device is presenting.");
				return;
			}
			le = e, ue = n, t.width = Math.floor(e * de), t.height = Math.floor(n * de), r === !0 && (t.style.width = e + "px", t.style.height = n + "px"), w !== null && w.setSize(t.width, t.height), this.setViewport(0, 0, e, n);
		}, this.getDrawingBufferSize = function(e) {
			return e.set(le * de, ue * de).floor();
		}, this.setDrawingBufferSize = function(e, n, r) {
			le = e, ue = n, de = r, t.width = Math.floor(e * r), t.height = Math.floor(n * r), this.setViewport(0, 0, e, n);
		}, this.setEffects = function(e) {
			if (m === 1009) {
				K("WebGLRenderer: setEffects() requires outputBufferType set to HalfFloatType or FloatType.");
				return;
			}
			if (e) {
				for (let t = 0; t < e.length; t++) if (e[t].isOutputPass === !0) {
					G("WebGLRenderer: OutputPass is not needed in setEffects(). Tone mapping and color space conversion are applied automatically.");
					break;
				}
			}
			w.setEffects(e || []);
		}, this.getCurrentViewport = function(e) {
			return e.copy(ie);
		}, this.getViewport = function(e) {
			return e.copy(me);
		}, this.setViewport = function(e, t, n, r) {
			e.isVector4 ? me.set(e.x, e.y, e.z, e.w) : me.set(e, t, n, r), N.viewport(ie.copy(me).multiplyScalar(de).round());
		}, this.getScissor = function(e) {
			return e.copy(he);
		}, this.setScissor = function(e, t, n, r) {
			e.isVector4 ? he.set(e.x, e.y, e.z, e.w) : he.set(e, t, n, r), N.scissor(ae.copy(he).multiplyScalar(de).round());
		}, this.getScissorTest = function() {
			return ge;
		}, this.setScissorTest = function(e) {
			N.setScissorTest(ge = e);
		}, this.setOpaqueSort = function(e) {
			fe = e;
		}, this.setTransparentSort = function(e) {
			pe = e;
		}, this.getClearColor = function(e) {
			return e.copy(Be.getClearColor());
		}, this.setClearColor = function() {
			Be.setClearColor(...arguments);
		}, this.getClearAlpha = function() {
			return Be.getClearAlpha();
		}, this.setClearAlpha = function() {
			Be.setClearAlpha(...arguments);
		}, this.clear = function(e = !0, t = !0, n = !0) {
			let r = 0;
			if (e) {
				let e = !1;
				if (j !== null) {
					let t = j.texture.format;
					e = h.has(t);
				}
				if (e) {
					let e = j.texture.type, t = g.has(e), n = Be.getClearColor(), r = Be.getClearAlpha(), i = n.r, a = n.g, o = n.b;
					t ? (_[0] = i, _[1] = a, _[2] = o, _[3] = r, M.clearBufferuiv(M.COLOR, 0, _)) : (v[0] = i, v[1] = a, v[2] = o, v[3] = r, M.clearBufferiv(M.COLOR, 0, v));
				} else r |= M.COLOR_BUFFER_BIT;
			}
			t && (r |= M.DEPTH_BUFFER_BIT, this.state.buffers.depth.setMask(!0)), n && (r |= M.STENCIL_BUFFER_BIT, this.state.buffers.stencil.setMask(4294967295)), r !== 0 && M.clear(r);
		}, this.clearColor = function() {
			this.clear(!0, !1, !1);
		}, this.clearDepth = function() {
			this.clear(!1, !0, !1);
		}, this.clearStencil = function() {
			this.clear(!1, !1, !0);
		}, this.setNodesHandler = function(e) {
			e.setRenderer(this), D = e;
		}, this.dispose = function() {
			t.removeEventListener("webglcontextlost", Ye, !1), t.removeEventListener("webglcontextrestored", Xe, !1), t.removeEventListener("webglcontextcreationerror", Ze, !1), Be.dispose(), Ie.dispose(), Le.dispose(), P.dispose(), Ae.dispose(), Ne.dispose(), Ge.dispose(), Ke.dispose(), Pe.dispose(), Je.dispose(), Je.removeEventListener("sessionstart", it), Je.removeEventListener("sessionend", at), ot.stop();
		};
		function Ye(e) {
			e.preventDefault(), Kd("WebGLRenderer: Context Lost."), E = !0;
		}
		function Xe() {
			Kd("WebGLRenderer: Context Restored."), E = !1;
			let e = ke.autoReset, t = ze.enabled, n = ze.autoUpdate, r = ze.needsUpdate, i = ze.type;
			qe(), ke.autoReset = e, ze.enabled = t, ze.autoUpdate = n, ze.needsUpdate = r, ze.type = i;
		}
		function Ze(e) {
			K("WebGLRenderer: A WebGL context could not be created. Reason: ", e.statusMessage);
		}
		function Qe(e) {
			let t = e.target;
			t.removeEventListener("dispose", Qe), $e(t);
		}
		function $e(e) {
			et(e), P.remove(e);
		}
		function et(e) {
			let t = P.get(e).programs;
			t !== void 0 && (t.forEach(function(e) {
				Pe.releaseProgram(e);
			}), e.isShaderMaterial && Pe.releaseShaderCache(e));
		}
		this.renderBufferDirect = function(e, t, n, r, i, a) {
			t === null && (t = Ce);
			let o = i.isMesh && i.matrixWorld.determinantAffine() < 0, s = ht(e, t, n, r, i);
			N.setMaterial(r, o);
			let c = n.index, l = 1;
			if (r.wireframe === !0) {
				if (c = Me.getWireframeAttribute(n), c === void 0) return;
				l = 2;
			}
			let u = n.drawRange, d = n.attributes.position, f = u.start * l, p = (u.start + u.count) * l;
			a !== null && (f = Math.max(f, a.start * l), p = Math.min(p, (a.start + a.count) * l)), c === null ? d != null && (f = Math.max(f, 0), p = Math.min(p, d.count)) : (f = Math.max(f, 0), p = Math.min(p, c.count));
			let m = p - f;
			if (m < 0 || m === Infinity) return;
			Ge.setup(i, r, s, n, c);
			let h, g = He;
			if (c !== null && (h = je.get(c), g = Ue, g.setIndex(h)), i.isMesh) r.wireframe === !0 ? (N.setLineWidth(r.wireframeLinewidth * Te()), g.setMode(M.LINES)) : g.setMode(M.TRIANGLES);
			else if (i.isLine) {
				let e = r.linewidth;
				e === void 0 && (e = 1), N.setLineWidth(e * Te()), i.isLineSegments ? g.setMode(M.LINES) : i.isLineLoop ? g.setMode(M.LINE_LOOP) : g.setMode(M.LINE_STRIP);
			} else i.isPoints ? g.setMode(M.POINTS) : i.isSprite && g.setMode(M.TRIANGLES);
			if (i.isBatchedMesh) if (De.get("WEBGL_multi_draw")) g.renderMultiDraw(i._multiDrawStarts, i._multiDrawCounts, i._multiDrawCount);
			else {
				let e = i._multiDrawStarts, t = i._multiDrawCounts, n = i._multiDrawCount, a = c ? je.get(c).bytesPerElement : 1, o = P.get(r).currentProgram.getUniforms();
				for (let r = 0; r < n; r++) o.setValue(M, "_gl_DrawID", r), g.render(e[r] / a, t[r]);
			}
			else if (i.isInstancedMesh) g.renderInstances(f, m, i.count);
			else if (n.isInstancedBufferGeometry) {
				let e = n._maxInstanceCount === void 0 ? Infinity : n._maxInstanceCount, t = Math.min(n.instanceCount, e);
				g.renderInstances(f, m, t);
			} else g.render(f, m);
		};
		function tt(e, t, n) {
			e.transparent === !0 && e.side === 2 && e.forceSinglePass === !1 ? (e.side = 1, e.needsUpdate = !0, dt(e, t, n), e.side = 0, e.needsUpdate = !0, dt(e, t, n), e.side = 2) : dt(e, t, n);
		}
		this.compile = function(e, t, n = null) {
			n === null && (n = e), x = Le.get(n), x.init(t), C.push(x), n.traverseVisible(function(e) {
				e.isLight && e.layers.test(t.layers) && (x.pushLight(e), e.castShadow && x.pushShadow(e));
			}), e !== n && e.traverseVisible(function(e) {
				e.isLight && e.layers.test(t.layers) && (x.pushLight(e), e.castShadow && x.pushShadow(e));
			}), x.setupLights();
			let r = /* @__PURE__ */ new Set();
			return e.traverse(function(e) {
				if (!(e.isMesh || e.isPoints || e.isLine || e.isSprite)) return;
				let t = e.material;
				if (t) if (Array.isArray(t)) for (let i = 0; i < t.length; i++) {
					let a = t[i];
					tt(a, n, e), r.add(a);
				}
				else tt(t, n, e), r.add(t);
			}), x = C.pop(), r;
		}, this.compileAsync = function(e, t, n = null) {
			let r = this.compile(e, t, n);
			return new Promise((t) => {
				function n() {
					if (r.forEach(function(e) {
						P.get(e).currentProgram.isReady() && r.delete(e);
					}), r.size === 0) {
						t(e);
						return;
					}
					setTimeout(n, 10);
				}
				De.get("KHR_parallel_shader_compile") === null ? setTimeout(n, 10) : n();
			});
		};
		let nt = null;
		function rt(e) {
			nt && nt(e);
		}
		function it() {
			ot.stop();
		}
		function at() {
			ot.start();
		}
		let ot = new Lg();
		ot.setAnimationLoop(rt), typeof self < "u" && ot.setContext(self), this.setAnimationLoop = function(e) {
			nt = e, Je.setAnimationLoop(e), e === null ? ot.stop() : ot.start();
		}, Je.addEventListener("sessionstart", it), Je.addEventListener("sessionend", at), this.render = function(e, t) {
			if (t !== void 0 && t.isCamera !== !0) {
				K("WebGLRenderer.render: camera is not an instance of THREE.Camera.");
				return;
			}
			if (E === !0) return;
			D !== null && D.renderStart(e, t);
			let n = Je.enabled === !0 && Je.isPresenting === !0, r = w !== null && (j === null || n) && w.begin(T, j);
			if (e.matrixWorldAutoUpdate === !0 && e.updateMatrixWorld(), t.parent === null && t.matrixWorldAutoUpdate === !0 && t.updateMatrixWorld(), Je.enabled === !0 && Je.isPresenting === !0 && (w === null || w.isCompositing() === !1) && (Je.cameraAutoUpdate === !0 && Je.updateCamera(t), t = Je.getCamera()), e.isScene === !0 && e.onBeforeRender(T, e, t, j), x = Le.get(e, C.length), x.init(t), x.state.textureUnits = F.getTextureUnits(), C.push(x), be.multiplyMatrices(t.projectionMatrix, t.matrixWorldInverse), _e.setFromProjectionMatrix(be, Bd, t.reversedDepth), ye = this.localClippingEnabled, ve = Re.init(this.clippingPlanes, ye), b = Ie.get(e, S.length), b.init(), S.push(b), Je.enabled === !0 && Je.isPresenting === !0) {
				let e = T.xr.getDepthSensingMesh();
				e !== null && st(e, t, -Infinity, T.sortObjects);
			}
			st(e, t, 0, T.sortObjects), b.finish(), T.sortObjects === !0 && b.sort(fe, pe, t.reversedDepth), we = Je.enabled === !1 || Je.isPresenting === !1 || Je.hasDepthSensing() === !1, we && Be.addToRenderList(b, e), this.info.render.frame++, this.info.autoReset === !0 && this.info.reset(), ve === !0 && Re.beginShadows();
			let i = x.state.shadowsArray;
			if (ze.render(i, e, t), ve === !0 && Re.endShadows(), (r && w.hasRenderPass()) === !1) {
				let n = b.opaque, r = b.transmissive;
				if (x.setupLights(), t.isArrayCamera) {
					let i = t.cameras;
					if (r.length > 0) for (let t = 0, a = i.length; t < a; t++) {
						let a = i[t];
						I(n, r, e, a);
					}
					we && Be.render(e);
					for (let t = 0, n = i.length; t < n; t++) {
						let n = i[t];
						ct(b, e, n, n.viewport);
					}
				} else r.length > 0 && I(n, r, e, t), we && Be.render(e), ct(b, e, t);
			}
			j !== null && te === 0 && (F.updateMultisampleRenderTarget(j), F.updateRenderTargetMipmap(j)), r && w.end(T), e.isScene === !0 && e.onAfterRender(T, e, t), Ge.resetDefaultState(), ne = -1, re = null, C.pop(), C.length > 0 ? (x = C[C.length - 1], F.setTextureUnits(x.state.textureUnits), ve === !0 && Re.setGlobalState(T.clippingPlanes, x.state.camera)) : x = null, S.pop(), b = S.length > 0 ? S[S.length - 1] : null, D !== null && D.renderEnd();
		};
		function st(e, t, n, r) {
			if (e.visible === !1) return;
			if (e.layers.test(t.layers)) {
				if (e.isGroup) n = e.renderOrder;
				else if (e.isLOD) e.autoUpdate === !0 && e.update(t);
				else if (e.isLightProbeGrid) x.pushLightProbeGrid(e);
				else if (e.isLight) x.pushLight(e), e.castShadow && x.pushShadow(e);
				else if (e.isSprite) {
					if (!e.frustumCulled || _e.intersectsSprite(e)) {
						r && Se.setFromMatrixPosition(e.matrixWorld).applyMatrix4(be);
						let t = Ne.update(e), i = e.material;
						i.visible && b.push(e, t, i, n, Se.z, null);
					}
				} else if ((e.isMesh || e.isLine || e.isPoints) && (!e.frustumCulled || _e.intersectsObject(e))) {
					let t = Ne.update(e), i = e.material;
					if (r && (e.boundingSphere === void 0 ? (t.boundingSphere === null && t.computeBoundingSphere(), Se.copy(t.boundingSphere.center)) : (e.boundingSphere === null && e.computeBoundingSphere(), Se.copy(e.boundingSphere.center)), Se.applyMatrix4(e.matrixWorld).applyMatrix4(be)), Array.isArray(i)) {
						let r = t.groups;
						for (let a = 0, o = r.length; a < o; a++) {
							let o = r[a], s = i[o.materialIndex];
							s && s.visible && b.push(e, t, s, n, Se.z, o);
						}
					} else i.visible && b.push(e, t, i, n, Se.z, null);
				}
			}
			let i = e.children;
			for (let e = 0, a = i.length; e < a; e++) st(i[e], t, n, r);
		}
		function ct(e, t, n, r) {
			let { opaque: i, transmissive: a, transparent: o } = e;
			x.setupLightsView(n), ve === !0 && Re.setGlobalState(T.clippingPlanes, n), r && N.viewport(ie.copy(r)), i.length > 0 && lt(i, t, n), a.length > 0 && lt(a, t, n), o.length > 0 && lt(o, t, n), N.buffers.depth.setTest(!0), N.buffers.depth.setMask(!0), N.buffers.color.setMask(!0), N.setPolygonOffset(!1);
		}
		function I(e, t, n, r) {
			if ((n.isScene === !0 ? n.overrideMaterial : null) !== null) return;
			if (x.state.transmissionRenderTarget[r.id] === void 0) {
				let e = De.has("EXT_color_buffer_half_float") || De.has("EXT_color_buffer_float");
				x.state.transmissionRenderTarget[r.id] = new Gf(1, 1, {
					generateMipmaps: !0,
					type: e ? Du : bu,
					minFilter: yu,
					samples: Math.max(4, Oe.samples),
					stencilBuffer: i,
					resolveDepthBuffer: !1,
					resolveStencilBuffer: !1,
					colorSpace: X.workingColorSpace
				});
			}
			let a = x.state.transmissionRenderTarget[r.id], o = r.viewport || ie;
			a.setSize(o.z * T.transmissionResolutionScale, o.w * T.transmissionResolutionScale);
			let s = T.getRenderTarget(), c = T.getActiveCubeFace(), l = T.getActiveMipmapLevel();
			T.setRenderTarget(a), T.getClearColor(se), ce = T.getClearAlpha(), ce < 1 && T.setClearColor(16777215, .5), T.clear(), we && Be.render(n);
			let u = T.toneMapping;
			T.toneMapping = 0;
			let d = r.viewport;
			if (r.viewport !== void 0 && (r.viewport = void 0), x.setupLightsView(r), ve === !0 && Re.setGlobalState(T.clippingPlanes, r), lt(e, n, r), F.updateMultisampleRenderTarget(a), F.updateRenderTargetMipmap(a), De.has("WEBGL_multisampled_render_to_texture") === !1) {
				let e = !1;
				for (let i = 0, a = t.length; i < a; i++) {
					let { object: a, geometry: o, material: s, group: c } = t[i];
					if (s.side === 2 && a.layers.test(r.layers)) {
						let t = s.side;
						s.side = 1, s.needsUpdate = !0, ut(a, n, r, o, s, c), s.side = t, s.needsUpdate = !0, e = !0;
					}
				}
				e === !0 && (F.updateMultisampleRenderTarget(a), F.updateRenderTargetMipmap(a));
			}
			T.setRenderTarget(s, c, l), T.setClearColor(se, ce), d !== void 0 && (r.viewport = d), T.toneMapping = u;
		}
		function lt(e, t, n) {
			let r = t.isScene === !0 ? t.overrideMaterial : null;
			for (let i = 0, a = e.length; i < a; i++) {
				let a = e[i], { object: o, geometry: s, group: c } = a, l = a.material;
				l.allowOverride === !0 && r !== null && (l = r), o.layers.test(n.layers) && ut(o, t, n, s, l, c);
			}
		}
		function ut(e, t, n, r, i, a) {
			e.onBeforeRender(T, t, n, r, i, a), e.modelViewMatrix.multiplyMatrices(n.matrixWorldInverse, e.matrixWorld), e.normalMatrix.getNormalMatrix(e.modelViewMatrix), i.onBeforeRender(T, t, n, r, e, a), i.transparent === !0 && i.side === 2 && i.forceSinglePass === !1 ? (i.side = 1, i.needsUpdate = !0, T.renderBufferDirect(n, t, r, i, e, a), i.side = 0, i.needsUpdate = !0, T.renderBufferDirect(n, t, r, i, e, a), i.side = 2) : T.renderBufferDirect(n, t, r, i, e, a), e.onAfterRender(T, t, n, r, i, a);
		}
		function dt(e, t, n) {
			t.isScene !== !0 && (t = Ce);
			let r = P.get(e), i = x.state.lights, a = x.state.shadowsArray, o = i.state.version, s = Pe.getParameters(e, i.state, a, t, n, x.state.lightProbeGridArray), c = Pe.getProgramCacheKey(s), l = r.programs;
			r.environment = e.isMeshStandardMaterial || e.isMeshLambertMaterial || e.isMeshPhongMaterial ? t.environment : null, r.fog = t.fog;
			let u = e.isMeshStandardMaterial || e.isMeshLambertMaterial && !e.envMap || e.isMeshPhongMaterial && !e.envMap;
			r.envMap = Ae.get(e.envMap || r.environment, u), r.envMapRotation = r.environment !== null && e.envMap === null ? t.environmentRotation : e.envMapRotation, l === void 0 && (e.addEventListener("dispose", Qe), l = /* @__PURE__ */ new Map(), r.programs = l);
			let d = l.get(c);
			if (d !== void 0) {
				if (r.currentProgram === d && r.lightsStateVersion === o) return pt(e, s), d;
			} else s.uniforms = Pe.getUniforms(e), D !== null && e.isNodeMaterial && D.build(e, n, s), e.onBeforeCompile(s, T), d = Pe.acquireProgram(s, c), l.set(c, d), r.uniforms = s.uniforms;
			let f = r.uniforms;
			return (!e.isShaderMaterial && !e.isRawShaderMaterial || e.clipping === !0) && (f.clippingPlanes = Re.uniform), pt(e, s), r.needsLights = _t(e), r.lightsStateVersion = o, r.needsLights && (f.ambientLightColor.value = i.state.ambient, f.lightProbe.value = i.state.probe, f.directionalLights.value = i.state.directional, f.directionalLightShadows.value = i.state.directionalShadow, f.spotLights.value = i.state.spot, f.spotLightShadows.value = i.state.spotShadow, f.rectAreaLights.value = i.state.rectArea, f.ltc_1.value = i.state.rectAreaLTC1, f.ltc_2.value = i.state.rectAreaLTC2, f.pointLights.value = i.state.point, f.pointLightShadows.value = i.state.pointShadow, f.hemisphereLights.value = i.state.hemi, f.directionalShadowMatrix.value = i.state.directionalShadowMatrix, f.spotLightMatrix.value = i.state.spotLightMatrix, f.spotLightMap.value = i.state.spotLightMap, f.pointShadowMatrix.value = i.state.pointShadowMatrix), r.lightProbeGrid = x.state.lightProbeGridArray.length > 0, r.currentProgram = d, r.uniformsList = null, d;
		}
		function ft(e) {
			if (e.uniformsList === null) {
				let t = e.currentProgram.getUniforms();
				e.uniformsList = Av.seqWithValue(t.seq, e.uniforms);
			}
			return e.uniformsList;
		}
		function pt(e, t) {
			let n = P.get(e);
			n.outputColorSpace = t.outputColorSpace, n.batching = t.batching, n.batchingColor = t.batchingColor, n.instancing = t.instancing, n.instancingColor = t.instancingColor, n.instancingMorph = t.instancingMorph, n.skinning = t.skinning, n.morphTargets = t.morphTargets, n.morphNormals = t.morphNormals, n.morphColors = t.morphColors, n.morphTargetsCount = t.morphTargetsCount, n.numClippingPlanes = t.numClippingPlanes, n.numIntersection = t.numClipIntersection, n.vertexAlphas = t.vertexAlphas, n.vertexTangents = t.vertexTangents, n.toneMapping = t.toneMapping;
		}
		function mt(e, t) {
			if (e.length === 0) return null;
			if (e.length === 1) return e[0].texture === null ? null : e[0];
			y.setFromMatrixPosition(t.matrixWorld);
			for (let t = 0, n = e.length; t < n; t++) {
				let n = e[t];
				if (n.texture !== null && n.boundingBox.containsPoint(y)) return n;
			}
			return null;
		}
		function ht(e, t, n, r, i) {
			t.isScene !== !0 && (t = Ce), F.resetTextureUnits();
			let a = t.fog, o = r.isMeshStandardMaterial || r.isMeshLambertMaterial || r.isMeshPhongMaterial ? t.environment : null, s = j === null ? T.outputColorSpace : j.isXRRenderTarget === !0 ? j.texture.colorSpace : X.workingColorSpace, c = r.isMeshStandardMaterial || r.isMeshLambertMaterial && !r.envMap || r.isMeshPhongMaterial && !r.envMap, l = Ae.get(r.envMap || o, c), u = r.vertexColors === !0 && !!n.attributes.color && n.attributes.color.itemSize === 4, d = !!n.attributes.tangent && (!!r.normalMap || r.anisotropy > 0), f = !!n.morphAttributes.position, p = !!n.morphAttributes.normal, m = !!n.morphAttributes.color, h = 0;
			r.toneMapped && (j === null || j.isXRRenderTarget === !0) && (h = T.toneMapping);
			let g = n.morphAttributes.position || n.morphAttributes.normal || n.morphAttributes.color, _ = g === void 0 ? 0 : g.length, v = P.get(r), y = x.state.lights;
			if (ve === !0 && (ye === !0 || e !== re)) {
				let t = e === re && r.id === ne;
				Re.setState(r, e, t);
			}
			let b = !1;
			r.version === v.__version ? v.needsLights && v.lightsStateVersion !== y.state.version ? b = !0 : v.outputColorSpace === s ? i.isBatchedMesh && v.batching === !1 || !i.isBatchedMesh && v.batching === !0 || i.isBatchedMesh && v.batchingColor === !0 && i.colorTexture === null || i.isBatchedMesh && v.batchingColor === !1 && i.colorTexture !== null || i.isInstancedMesh && v.instancing === !1 || !i.isInstancedMesh && v.instancing === !0 || i.isSkinnedMesh && v.skinning === !1 || !i.isSkinnedMesh && v.skinning === !0 || i.isInstancedMesh && v.instancingColor === !0 && i.instanceColor === null || i.isInstancedMesh && v.instancingColor === !1 && i.instanceColor !== null || i.isInstancedMesh && v.instancingMorph === !0 && i.morphTexture === null || i.isInstancedMesh && v.instancingMorph === !1 && i.morphTexture !== null ? b = !0 : v.envMap === l ? r.fog === !0 && v.fog !== a || v.numClippingPlanes !== void 0 && (v.numClippingPlanes !== Re.numPlanes || v.numIntersection !== Re.numIntersection) ? b = !0 : v.vertexAlphas === u && v.vertexTangents === d && v.morphTargets === f && v.morphNormals === p && v.morphColors === m && v.toneMapping === h && v.morphTargetsCount === _ ? !!v.lightProbeGrid != x.state.lightProbeGridArray.length > 0 && (b = !0) : b = !0 : b = !0 : b = !0 : (b = !0, v.__version = r.version);
			let S = v.currentProgram;
			b === !0 && (S = dt(r, t, i), D && r.isNodeMaterial && D.onUpdateProgram(r, S, v));
			let C = !1, w = !1, E = !1, O = S.getUniforms(), k = v.uniforms;
			if (N.useProgram(S.program) && (C = !0, w = !0, E = !0), r.id !== ne && (ne = r.id, w = !0), v.needsLights) {
				let e = mt(x.state.lightProbeGridArray, i);
				v.lightProbeGrid !== e && (v.lightProbeGrid = e, w = !0);
			}
			if (C || re !== e) {
				N.buffers.depth.getReversed() && e.reversedDepth !== !0 && (e._reversedDepth = !0, e.updateProjectionMatrix()), O.setValue(M, "projectionMatrix", e.projectionMatrix), O.setValue(M, "viewMatrix", e.matrixWorldInverse);
				let t = O.map.cameraPosition;
				t !== void 0 && t.setValue(M, xe.setFromMatrixPosition(e.matrixWorld)), Oe.logarithmicDepthBuffer && O.setValue(M, "logDepthBufFC", 2 / (Math.log(e.far + 1) / Math.LN2)), (r.isMeshPhongMaterial || r.isMeshToonMaterial || r.isMeshLambertMaterial || r.isMeshBasicMaterial || r.isMeshStandardMaterial || r.isShaderMaterial) && O.setValue(M, "isOrthographic", e.isOrthographicCamera === !0), re !== e && (re = e, w = !0, E = !0);
			}
			if (v.needsLights && (y.state.directionalShadowMap.length > 0 && O.setValue(M, "directionalShadowMap", y.state.directionalShadowMap, F), y.state.spotShadowMap.length > 0 && O.setValue(M, "spotShadowMap", y.state.spotShadowMap, F), y.state.pointShadowMap.length > 0 && O.setValue(M, "pointShadowMap", y.state.pointShadowMap, F)), i.isSkinnedMesh) {
				O.setOptional(M, i, "bindMatrix"), O.setOptional(M, i, "bindMatrixInverse");
				let e = i.skeleton;
				e && (e.boneTexture === null && e.computeBoneTexture(), O.setValue(M, "boneTexture", e.boneTexture, F));
			}
			i.isBatchedMesh && (O.setOptional(M, i, "batchingTexture"), O.setValue(M, "batchingTexture", i._matricesTexture, F), O.setOptional(M, i, "batchingIdTexture"), O.setValue(M, "batchingIdTexture", i._indirectTexture, F), O.setOptional(M, i, "batchingColorTexture"), i._colorsTexture !== null && O.setValue(M, "batchingColorTexture", i._colorsTexture, F));
			let A = n.morphAttributes;
			if ((A.position !== void 0 || A.normal !== void 0 || A.color !== void 0) && Ve.update(i, n, S), (w || v.receiveShadow !== i.receiveShadow) && (v.receiveShadow = i.receiveShadow, O.setValue(M, "receiveShadow", i.receiveShadow)), (r.isMeshStandardMaterial || r.isMeshLambertMaterial || r.isMeshPhongMaterial) && r.envMap === null && t.environment !== null && (k.envMapIntensity.value = t.environmentIntensity), k.dfgLUT !== void 0 && (k.dfgLUT.value = Zy()), w) {
				if (O.setValue(M, "toneMappingExposure", T.toneMappingExposure), v.needsLights && gt(k, E), a && r.fog === !0 && Fe.refreshFogUniforms(k, a), Fe.refreshMaterialUniforms(k, r, de, ue, x.state.transmissionRenderTarget[e.id]), v.needsLights && v.lightProbeGrid) {
					let e = v.lightProbeGrid;
					k.probesSH.value = e.texture, k.probesMin.value.copy(e.boundingBox.min), k.probesMax.value.copy(e.boundingBox.max), k.probesResolution.value.copy(e.resolution);
				}
				Av.upload(M, ft(v), k, F);
			}
			if (r.isShaderMaterial && r.uniformsNeedUpdate === !0 && (Av.upload(M, ft(v), k, F), r.uniformsNeedUpdate = !1), r.isSpriteMaterial && O.setValue(M, "center", i.center), O.setValue(M, "modelViewMatrix", i.modelViewMatrix), O.setValue(M, "normalMatrix", i.normalMatrix), O.setValue(M, "modelMatrix", i.matrixWorld), r.uniformsGroups !== void 0) {
				let e = r.uniformsGroups;
				for (let t = 0, n = e.length; t < n; t++) {
					let n = e[t];
					Ke.update(n, S), Ke.bind(n, S);
				}
			}
			return S;
		}
		function gt(e, t) {
			e.ambientLightColor.needsUpdate = t, e.lightProbe.needsUpdate = t, e.directionalLights.needsUpdate = t, e.directionalLightShadows.needsUpdate = t, e.pointLights.needsUpdate = t, e.pointLightShadows.needsUpdate = t, e.spotLights.needsUpdate = t, e.spotLightShadows.needsUpdate = t, e.rectAreaLights.needsUpdate = t, e.hemisphereLights.needsUpdate = t;
		}
		function _t(e) {
			return e.isMeshLambertMaterial || e.isMeshToonMaterial || e.isMeshPhongMaterial || e.isMeshStandardMaterial || e.isShadowMaterial || e.isShaderMaterial && e.lights === !0;
		}
		this.getActiveCubeFace = function() {
			return ee;
		}, this.getActiveMipmapLevel = function() {
			return te;
		}, this.getRenderTarget = function() {
			return j;
		}, this.setRenderTargetTextures = function(e, t, n) {
			let r = P.get(e);
			r.__autoAllocateDepthBuffer = e.resolveDepthBuffer === !1, r.__autoAllocateDepthBuffer === !1 && (r.__useRenderToTexture = !1), P.get(e.texture).__webglTexture = t, P.get(e.depthTexture).__webglTexture = r.__autoAllocateDepthBuffer ? void 0 : n, r.__hasExternalTextures = !0;
		}, this.setRenderTargetFramebuffer = function(e, t) {
			let n = P.get(e);
			n.__webglFramebuffer = t, n.__useDefaultFramebuffer = t === void 0;
		}, this.setRenderTarget = function(e, t = 0, n = 0) {
			j = e, ee = t, te = n;
			let r = null, i = !1, a = !1;
			if (e) {
				let o = P.get(e);
				if (o.__useDefaultFramebuffer !== void 0) {
					N.bindFramebuffer(M.FRAMEBUFFER, o.__webglFramebuffer), ie.copy(e.viewport), ae.copy(e.scissor), oe = e.scissorTest, N.viewport(ie), N.scissor(ae), N.setScissorTest(oe), ne = -1;
					return;
				} else if (o.__webglFramebuffer === void 0) F.setupRenderTarget(e);
				else if (o.__hasExternalTextures) F.rebindTextures(e, P.get(e.texture).__webglTexture, P.get(e.depthTexture).__webglTexture);
				else if (e.depthBuffer) {
					let t = e.depthTexture;
					if (o.__boundDepthTexture !== t) {
						if (t !== null && P.has(t) && (e.width !== t.image.width || e.height !== t.image.height)) throw Error("THREE.WebGLRenderer: Attached DepthTexture is initialized to the incorrect size.");
						F.setupDepthRenderbuffer(e);
					}
				}
				let s = e.texture;
				(s.isData3DTexture || s.isDataArrayTexture || s.isCompressedArrayTexture) && (a = !0);
				let c = P.get(e).__webglFramebuffer;
				e.isWebGLCubeRenderTarget ? (r = Array.isArray(c[t]) ? c[t][n] : c[t], i = !0) : r = e.samples > 0 && F.useMultisampledRTT(e) === !1 ? P.get(e).__webglMultisampledFramebuffer : Array.isArray(c) ? c[n] : c, ie.copy(e.viewport), ae.copy(e.scissor), oe = e.scissorTest;
			} else ie.copy(me).multiplyScalar(de).floor(), ae.copy(he).multiplyScalar(de).floor(), oe = ge;
			if (n !== 0 && (r = O), N.bindFramebuffer(M.FRAMEBUFFER, r) && N.drawBuffers(e, r), N.viewport(ie), N.scissor(ae), N.setScissorTest(oe), i) {
				let r = P.get(e.texture);
				M.framebufferTexture2D(M.FRAMEBUFFER, M.COLOR_ATTACHMENT0, M.TEXTURE_CUBE_MAP_POSITIVE_X + t, r.__webglTexture, n);
			} else if (a) {
				let r = t;
				for (let t = 0; t < e.textures.length; t++) {
					let i = P.get(e.textures[t]);
					M.framebufferTextureLayer(M.FRAMEBUFFER, M.COLOR_ATTACHMENT0 + t, i.__webglTexture, n, r);
				}
			} else if (e !== null && n !== 0) {
				let t = P.get(e.texture);
				M.framebufferTexture2D(M.FRAMEBUFFER, M.COLOR_ATTACHMENT0, M.TEXTURE_2D, t.__webglTexture, n);
			}
			ne = -1;
		}, this.readRenderTargetPixels = function(e, t, n, r, i, a, o, s = 0) {
			if (!(e && e.isWebGLRenderTarget)) {
				K("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");
				return;
			}
			let c = P.get(e).__webglFramebuffer;
			if (e.isWebGLCubeRenderTarget && o !== void 0 && (c = c[o]), c) {
				N.bindFramebuffer(M.FRAMEBUFFER, c);
				try {
					let o = e.textures[s], c = o.format, l = o.type;
					if (e.textures.length > 1 && M.readBuffer(M.COLOR_ATTACHMENT0 + s), !Oe.textureFormatReadable(c)) {
						K("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");
						return;
					}
					if (!Oe.textureTypeReadable(l)) {
						K("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");
						return;
					}
					t >= 0 && t <= e.width - r && n >= 0 && n <= e.height - i && M.readPixels(t, n, r, i, We.convert(c), We.convert(l), a);
				} finally {
					let e = j === null ? null : P.get(j).__webglFramebuffer;
					N.bindFramebuffer(M.FRAMEBUFFER, e);
				}
			}
		}, this.readRenderTargetPixelsAsync = async function(e, t, n, r, i, a, o, s = 0) {
			if (!(e && e.isWebGLRenderTarget)) throw Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");
			let c = P.get(e).__webglFramebuffer;
			if (e.isWebGLCubeRenderTarget && o !== void 0 && (c = c[o]), c) if (t >= 0 && t <= e.width - r && n >= 0 && n <= e.height - i) {
				N.bindFramebuffer(M.FRAMEBUFFER, c);
				let o = e.textures[s], l = o.format, u = o.type;
				if (e.textures.length > 1 && M.readBuffer(M.COLOR_ATTACHMENT0 + s), !Oe.textureFormatReadable(l)) throw Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");
				if (!Oe.textureTypeReadable(u)) throw Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");
				let d = M.createBuffer();
				M.bindBuffer(M.PIXEL_PACK_BUFFER, d), M.bufferData(M.PIXEL_PACK_BUFFER, a.byteLength, M.STREAM_READ), M.readPixels(t, n, r, i, We.convert(l), We.convert(u), 0);
				let f = j === null ? null : P.get(j).__webglFramebuffer;
				N.bindFramebuffer(M.FRAMEBUFFER, f);
				let p = M.fenceSync(M.SYNC_GPU_COMMANDS_COMPLETE, 0);
				return M.flush(), await Yd(M, p, 4), M.bindBuffer(M.PIXEL_PACK_BUFFER, d), M.getBufferSubData(M.PIXEL_PACK_BUFFER, 0, a), M.deleteBuffer(d), M.deleteSync(p), a;
			} else throw Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.");
		}, this.copyFramebufferToTexture = function(e, t = null, n = 0) {
			let r = 2 ** -n, i = Math.floor(e.image.width * r), a = Math.floor(e.image.height * r), o = t === null ? 0 : t.x, s = t === null ? 0 : t.y;
			F.setTexture2D(e, 0), M.copyTexSubImage2D(M.TEXTURE_2D, n, 0, 0, o, s, i, a), N.unbindTexture();
		}, this.copyTextureToTexture = function(e, t, n = null, r = null, i = 0, a = 0) {
			let o, s, c, l, u, d, f, p, m, h = e.isCompressedTexture ? e.mipmaps[a] : e.image;
			if (n !== null) o = n.max.x - n.min.x, s = n.max.y - n.min.y, c = n.isBox3 ? n.max.z - n.min.z : 1, l = n.min.x, u = n.min.y, d = n.isBox3 ? n.min.z : 0;
			else {
				let t = 2 ** -i;
				o = Math.floor(h.width * t), s = Math.floor(h.height * t), c = e.isDataArrayTexture ? h.depth : e.isData3DTexture ? Math.floor(h.depth * t) : 1, l = 0, u = 0, d = 0;
			}
			r === null ? (f = 0, p = 0, m = 0) : (f = r.x, p = r.y, m = r.z);
			let g = We.convert(t.format), _ = We.convert(t.type), v;
			t.isData3DTexture ? (F.setTexture3D(t, 0), v = M.TEXTURE_3D) : t.isDataArrayTexture || t.isCompressedArrayTexture ? (F.setTexture2DArray(t, 0), v = M.TEXTURE_2D_ARRAY) : (F.setTexture2D(t, 0), v = M.TEXTURE_2D), N.activeTexture(M.TEXTURE0), N.pixelStorei(M.UNPACK_FLIP_Y_WEBGL, t.flipY), N.pixelStorei(M.UNPACK_PREMULTIPLY_ALPHA_WEBGL, t.premultiplyAlpha), N.pixelStorei(M.UNPACK_ALIGNMENT, t.unpackAlignment);
			let y = N.getParameter(M.UNPACK_ROW_LENGTH), b = N.getParameter(M.UNPACK_IMAGE_HEIGHT), x = N.getParameter(M.UNPACK_SKIP_PIXELS), S = N.getParameter(M.UNPACK_SKIP_ROWS), C = N.getParameter(M.UNPACK_SKIP_IMAGES);
			N.pixelStorei(M.UNPACK_ROW_LENGTH, h.width), N.pixelStorei(M.UNPACK_IMAGE_HEIGHT, h.height), N.pixelStorei(M.UNPACK_SKIP_PIXELS, l), N.pixelStorei(M.UNPACK_SKIP_ROWS, u), N.pixelStorei(M.UNPACK_SKIP_IMAGES, d);
			let w = e.isDataArrayTexture || e.isData3DTexture, T = t.isDataArrayTexture || t.isData3DTexture;
			if (e.isDepthTexture) {
				let n = P.get(e), r = P.get(t), h = P.get(n.__renderTarget), g = P.get(r.__renderTarget);
				N.bindFramebuffer(M.READ_FRAMEBUFFER, h.__webglFramebuffer), N.bindFramebuffer(M.DRAW_FRAMEBUFFER, g.__webglFramebuffer);
				for (let n = 0; n < c; n++) w && (M.framebufferTextureLayer(M.READ_FRAMEBUFFER, M.COLOR_ATTACHMENT0, P.get(e).__webglTexture, i, d + n), M.framebufferTextureLayer(M.DRAW_FRAMEBUFFER, M.COLOR_ATTACHMENT0, P.get(t).__webglTexture, a, m + n)), M.blitFramebuffer(l, u, o, s, f, p, o, s, M.DEPTH_BUFFER_BIT, M.NEAREST);
				N.bindFramebuffer(M.READ_FRAMEBUFFER, null), N.bindFramebuffer(M.DRAW_FRAMEBUFFER, null);
			} else if (i !== 0 || e.isRenderTargetTexture || P.has(e)) {
				let n = P.get(e), r = P.get(t);
				N.bindFramebuffer(M.READ_FRAMEBUFFER, k), N.bindFramebuffer(M.DRAW_FRAMEBUFFER, A);
				for (let e = 0; e < c; e++) w ? M.framebufferTextureLayer(M.READ_FRAMEBUFFER, M.COLOR_ATTACHMENT0, n.__webglTexture, i, d + e) : M.framebufferTexture2D(M.READ_FRAMEBUFFER, M.COLOR_ATTACHMENT0, M.TEXTURE_2D, n.__webglTexture, i), T ? M.framebufferTextureLayer(M.DRAW_FRAMEBUFFER, M.COLOR_ATTACHMENT0, r.__webglTexture, a, m + e) : M.framebufferTexture2D(M.DRAW_FRAMEBUFFER, M.COLOR_ATTACHMENT0, M.TEXTURE_2D, r.__webglTexture, a), i === 0 ? T ? M.copyTexSubImage3D(v, a, f, p, m + e, l, u, o, s) : M.copyTexSubImage2D(v, a, f, p, l, u, o, s) : M.blitFramebuffer(l, u, o, s, f, p, o, s, M.COLOR_BUFFER_BIT, M.NEAREST);
				N.bindFramebuffer(M.READ_FRAMEBUFFER, null), N.bindFramebuffer(M.DRAW_FRAMEBUFFER, null);
			} else T ? e.isDataTexture || e.isData3DTexture ? M.texSubImage3D(v, a, f, p, m, o, s, c, g, _, h.data) : t.isCompressedArrayTexture ? M.compressedTexSubImage3D(v, a, f, p, m, o, s, c, g, h.data) : M.texSubImage3D(v, a, f, p, m, o, s, c, g, _, h) : e.isDataTexture ? M.texSubImage2D(M.TEXTURE_2D, a, f, p, o, s, g, _, h.data) : e.isCompressedTexture ? M.compressedTexSubImage2D(M.TEXTURE_2D, a, f, p, h.width, h.height, g, h.data) : M.texSubImage2D(M.TEXTURE_2D, a, f, p, o, s, g, _, h);
			N.pixelStorei(M.UNPACK_ROW_LENGTH, y), N.pixelStorei(M.UNPACK_IMAGE_HEIGHT, b), N.pixelStorei(M.UNPACK_SKIP_PIXELS, x), N.pixelStorei(M.UNPACK_SKIP_ROWS, S), N.pixelStorei(M.UNPACK_SKIP_IMAGES, C), a === 0 && t.generateMipmaps && M.generateMipmap(v), N.unbindTexture();
		}, this.initRenderTarget = function(e) {
			P.get(e).__webglFramebuffer === void 0 && F.setupRenderTarget(e);
		}, this.initTexture = function(e) {
			e.isCubeTexture ? F.setTextureCube(e, 0) : e.isData3DTexture ? F.setTexture3D(e, 0) : e.isDataArrayTexture || e.isCompressedArrayTexture ? F.setTexture2DArray(e, 0) : F.setTexture2D(e, 0), N.unbindTexture();
		}, this.resetState = function() {
			ee = 0, te = 0, j = null, N.reset(), Ge.reset();
		}, typeof __THREE_DEVTOOLS__ < "u" && __THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe", { detail: this }));
	}
	get coordinateSystem() {
		return Bd;
	}
	get outputColorSpace() {
		return this._outputColorSpace;
	}
	set outputColorSpace(e) {
		this._outputColorSpace = e;
		let t = this.getContext();
		t.drawingBufferColorSpace = X._getDrawingBufferColorSpace(e), t.unpackColorSpace = X._getUnpackColorSpace();
	}
}, $y = 2048, eb = 64, tb = $y / eb, nb = tb * tb, rb = 40, ib = 13, ab = "\n  attribute vec2 aOffset;\n  attribute float aSize;\n  attribute vec4 aUvRect;\n  attribute vec3 aColor;\n  attribute float aOpacity;\n  uniform vec2 uResolution;\n  uniform float uPixelRatio;\n  varying vec4 vUvRect;\n  varying vec3 vColor;\n  varying float vOpacity;\n  void main() {\n    vec4 clip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n    clip.xy += vec2(aOffset.x, -aOffset.y) * 2.0 / uResolution * clip.w;\n    gl_Position = clip;\n    gl_PointSize = aSize * uPixelRatio;\n    vUvRect = aUvRect;\n    vColor = aColor;\n    vOpacity = aOpacity;\n  }\n", ob = "\n  uniform sampler2D uAtlas;\n  varying vec4 vUvRect;\n  varying vec3 vColor;\n  varying float vOpacity;\n  void main() {\n    vec2 glyphUv = mix(vUvRect.xy, vUvRect.zw, vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y));\n    float alpha = texture2D(uAtlas, glyphUv).a;\n    alpha = smoothstep(0.08, 0.62, alpha) * vOpacity;\n    if (alpha < 0.02) discard;\n    gl_FragColor = vec4(vColor, alpha);\n  }\n", sb = class {
	scene;
	resolution;
	glyphs = /* @__PURE__ */ new Map();
	pages = [];
	visibleLabelCount = 0;
	visibleGlyphCount = 0;
	constructor(e, t) {
		this.scene = e, this.resolution = t;
	}
	addPage() {
		let e = document.createElement("canvas");
		e.width = $y, e.height = $y;
		let t = e.getContext("2d", { alpha: !0 });
		if (!t) throw Error("Canvas 2D is required for graph labels");
		t.clearRect(0, 0, $y, $y), t.fillStyle = "#ffffff", t.font = `600 ${rb}px Inter, "Microsoft YaHei", "Noto Sans SC", sans-serif`, t.textAlign = "center", t.textBaseline = "middle";
		let n = new Eh(e);
		n.colorSpace = Nd, n.minFilter = _u, n.magFilter = _u, n.generateMipmaps = !1;
		let r = {
			canvas: e,
			context: t,
			texture: n,
			material: new Bh({
				vertexShader: ab,
				fragmentShader: ob,
				transparent: !0,
				depthTest: !1,
				depthWrite: !1,
				uniforms: {
					uAtlas: { value: n },
					uResolution: { value: this.resolution },
					uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 1.75) }
				}
			}),
			mesh: null,
			count: 0
		};
		return this.pages.push(r), r;
	}
	ensureGlyph(e) {
		let t = this.glyphs.get(e);
		if (t) return t;
		let n = this.pages[this.pages.length - 1];
		(!n || n.count >= nb) && (n = this.addPage());
		let r = n.count++, i = r % tb, a = Math.floor(r / tb), o = i * eb, s = a * eb;
		n.context.fillText(e, o + eb / 2, s + eb / 2 + 1), n.texture.needsUpdate = !0;
		let c = n.context.measureText(e).width, l = o / $y, u = (o + eb) / $y, d = 1 - (s + eb) / $y, f = 1 - s / $y, p = {
			page: this.pages.length - 1,
			uv: [
				l,
				d,
				u,
				f
			],
			advance: c
		};
		return this.glyphs.set(e, p), p;
	}
	update(e, t) {
		this.visibleLabelCount = e.length;
		let n = /* @__PURE__ */ new Map();
		e.forEach((e) => {
			let r = [...e.text].map((e) => this.ensureGlyph(e)), i = ib / rb, a = r.map((e) => Math.max(4, e.advance * i)), o = a.reduce((e, t) => e + t, 0), s = e.offsetX + Math.max(0, (e.width - o) / 2);
			r.forEach((r, o) => {
				let c = Math.max(12, eb * i), l = eb * i, u = a[o], d = n.get(r.page) ?? [];
				d.push({
					anchor: [
						t[e.nodeIndex * 3],
						t[e.nodeIndex * 3 + 1],
						t[e.nodeIndex * 3 + 2]
					],
					offset: [s + u / 2, e.offsetY + e.height / 2],
					size: [c, l],
					uv: r.uv,
					color: e.color,
					opacity: e.opacity
				}), n.set(r.page, d), s += u;
			});
		}), this.visibleGlyphCount = [...n.values()].reduce((e, t) => e + t.length, 0), this.pages.forEach((e, t) => {
			e.mesh &&= (this.scene.remove(e.mesh), e.mesh.geometry.dispose(), null);
			let r = n.get(t) ?? [];
			if (!r.length) return;
			let i = new wm();
			i.setAttribute("position", new lm(new Float32Array(r.flatMap((e) => e.anchor)), 3)), i.setAttribute("aOffset", new lm(new Float32Array(r.flatMap((e) => e.offset)), 2)), i.setAttribute("aSize", new lm(new Float32Array(r.map((e) => Math.max(e.size[0], e.size[1]))), 1)), i.setAttribute("aUvRect", new lm(new Float32Array(r.flatMap((e) => e.uv)), 4)), i.setAttribute("aColor", new lm(new Float32Array(r.flatMap((e) => e.color)), 3)), i.setAttribute("aOpacity", new lm(new Float32Array(r.map((e) => e.opacity)), 1)), e.mesh = new Ch(i, e.material), e.mesh.frustumCulled = !1, e.mesh.renderOrder = 10, this.scene.add(e.mesh);
		});
	}
	resize(e, t) {
		this.resolution.set(e, t);
	}
	dispose() {
		this.pages.forEach((e) => {
			e.mesh && (this.scene.remove(e.mesh), e.mesh.geometry.dispose()), e.material.dispose(), e.texture.dispose();
		}), this.pages = [], this.glyphs.clear();
	}
}, cb = [
	.85,
	1.6,
	3,
	5
], lb = [
	0,
	.03,
	.15,
	.5,
	1
], ub = .12, db = class {
	level = 1;
	update(e) {
		for (; this.level < cb.length && e >= cb[this.level] * 1.12;) this.level += 1;
		for (; this.level > 0 && e < cb[this.level - 1] * (1 - ub);) --this.level;
		return this.level;
	}
	reset(e = 1) {
		return this.level = fb(e), this.level;
	}
	get currentLevel() {
		return this.level;
	}
};
function fb(e) {
	return e < cb[0] ? 0 : e < cb[1] ? 1 : e < cb[2] ? 2 : e < cb[3] ? 3 : 4;
}
function pb(e) {
	return lb[Math.max(0, Math.min(lb.length - 1, e))];
}
function mb(e) {
	return [
		6,
		4,
		3,
		5,
		2,
		1
	][e] ?? 0;
}
function hb(e, t, n, r) {
	let i = t.filter((e) => e.visible), a = pb(n), o = n === 0 ? 0 : Math.max(1, Math.ceil(i.length * a));
	return i.map((t) => {
		let n = e.nodes[t.nodeIndex], i = r.has(t.nodeIndex);
		return {
			nodeIndex: t.nodeIndex,
			text: n.label,
			anchorX: t.x,
			anchorY: t.y,
			nodeRadius: n.size + 3,
			priority: (i ? 1e6 : 0) + n.degree * 100 + mb(n.group) * 10 - t.nodeIndex * 1e-6,
			color: i ? [
				.78,
				1,
				.98
			] : [
				.88,
				.93,
				.98
			],
			forced: i
		};
	}).sort((e, t) => t.priority - e.priority).filter((e, t) => e.forced || t < o);
}
function gb(e, t = 13) {
	let n = 0;
	for (let t of e) n += t.charCodeAt(0) <= 127 ? .56 : 1;
	return Math.max(18, n * t + 4);
}
function _b(e, t) {
	let n = e.anchorX + e.offsetX, r = e.anchorY + e.offsetY, i = t.anchorX + t.offsetX, a = t.anchorY + t.offsetY;
	return n < i + t.width && n + e.width > i && r < a + t.height && r + e.height > a;
}
function vb(e, t, n) {
	let r = [], i = (e, t, n) => [
		[e, -n / 2],
		[-t - e, -n / 2],
		[-t / 2, -e - n],
		[-t / 2, e],
		[e, -e - n],
		[-t - e, -e - n],
		[e, e],
		[-t - e, e]
	];
	return e.forEach((e) => {
		let a = gb(e.text), o = null;
		for (let [s, c] of i(e.nodeRadius, a, 18)) {
			let i = {
				...e,
				offsetX: s,
				offsetY: c,
				width: a,
				height: 18,
				opacity: 1
			}, l = i.anchorX + i.offsetX, u = i.anchorY + i.offsetY;
			if (!(l < 8 || u < 8 || l + a > t - 8 || u + 18 > n - 8) && !r.some((e) => _b(i, e))) {
				o = i;
				break;
			}
		}
		!o && e.forced && (o = {
			...e,
			offsetX: Math.max(8, Math.min(t - a - 8, e.anchorX + e.nodeRadius)) - e.anchorX,
			offsetY: Math.max(8, Math.min(n - 18 - 8, e.anchorY - 18 / 2)) - e.anchorY,
			width: a,
			height: 18,
			opacity: 1
		}), o && r.push(o);
	}), r;
}
//#endregion
//#region frontend/src/visual-depth.ts
function yb(e, t) {
	let n = Math.min(t.length, Math.floor(e.length / 3)), r = new Float32Array(n);
	if (!n) return {
		depths: r,
		maximumDepth: 0
	};
	let i = Infinity, a = -Infinity, o = Infinity, s = -Infinity;
	for (let t = 0; t < n; t += 1) i = Math.min(i, e[t * 3]), a = Math.max(a, e[t * 3]), o = Math.min(o, e[t * 3 + 1]), s = Math.max(s, e[t * 3 + 1]);
	let c = (i + a) / 2, l = (o + s) / 2, u = 1;
	for (let t = 0; t < n; t += 1) u = Math.max(u, Math.hypot(e[t * 3] - c, e[t * 3 + 1] - l));
	let d = u * .34;
	for (let i = 0; i < n; i += 1) {
		let n = e[i * 3] - c, a = e[i * 3 + 1] - l, o = Math.min(1, Math.hypot(n, a) / u), s = Math.sqrt(Math.max(0, 1 - o * o)) * d, f = t[i] >>> 0, p = f & 1 ? 1 : -1, m = .38 + (f >>> 8 & 65535) / 65535 * .62;
		r[i] = p * s * m;
	}
	return {
		depths: r,
		maximumDepth: d
	};
}
//#endregion
//#region frontend/src/star-map-renderer.ts
var bb = "\n  attribute vec3 aColor;\n  attribute float aSize;\n  attribute float aState;\n  uniform float uPixelRatio;\n  uniform float uSemanticScale;\n  varying vec3 vColor;\n  varying float vState;\n  void main() {\n    vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);\n    gl_Position = projectionMatrix * viewPosition;\n    float stateScale = aState > 2.5 ? 1.7 : aState > 1.5 ? 1.28 : aState > 0.5 ? 0.88 : 1.0;\n    gl_PointSize = clamp(aSize * uPixelRatio * uSemanticScale * stateScale, 2.0, 34.0);\n    vColor = aColor;\n    vState = aState;\n  }\n", xb = "\n  varying vec3 vColor;\n  varying float vState;\n  void main() {\n    float distanceFromCenter = length(gl_PointCoord - vec2(0.5));\n    float aa = max(fwidth(distanceFromCenter), 0.006);\n    float disc = 1.0 - smoothstep(0.49 - aa, 0.49 + aa, distanceFromCenter);\n    if (disc <= 0.0) discard;\n    float alpha = vState > 0.5 && vState < 1.5 ? 0.18 : 0.96;\n    vec3 color = vColor;\n    if (vState > 2.5) {\n      float inner = 1.0 - smoothstep(0.34 - aa, 0.34 + aa, distanceFromCenter);\n      float ring = smoothstep(0.35 - aa, 0.35 + aa, distanceFromCenter) * disc;\n      color = mix(vec3(0.94, 1.0, 1.0), vec3(0.39, 0.94, 1.0), ring);\n      alpha = max(inner, ring);\n    } else if (vState > 1.5) {\n      color = mix(vColor, vec3(0.83, 1.0, 0.98), 0.42);\n    }\n    gl_FragColor = vec4(color, alpha * disc);\n  }\n", Sb = "\n  attribute vec3 aPickColor;\n  attribute float aSize;\n  uniform float uPixelRatio;\n  varying vec3 vPickColor;\n  void main() {\n    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n    gl_PointSize = max(14.0, aSize * uPixelRatio * 1.5);\n    vPickColor = aPickColor;\n  }\n", Cb = "\n  varying vec3 vPickColor;\n  void main() {\n    if (length(gl_PointCoord - vec2(0.5)) > 0.5) discard;\n    gl_FragColor = vec4(vPickColor, 1.0);\n  }\n", wb = "\n  attribute vec3 aColor;\n  attribute float aState;\n  attribute float aLineT;\n  attribute float aDashed;\n  varying vec3 vColor;\n  varying float vState;\n  varying float vLineT;\n  varying float vDashed;\n  void main() {\n    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n    vColor = aColor;\n    vState = aState;\n    vLineT = aLineT;\n    vDashed = aDashed;\n  }\n", Tb = "\n  varying vec3 vColor;\n  varying float vState;\n  varying float vLineT;\n  varying float vDashed;\n  void main() {\n    if (vDashed > 0.5 && mod(vLineT * 22.0, 1.0) > 0.58) discard;\n    float alpha = vState > 2.5 ? 0.98 : vState > 1.5 ? 0.78 : vState > 0.5 ? 0.025 : 0.105;\n    vec3 color = vState > 2.5 ? mix(vColor, vec3(0.94, 1.0, 1.0), 0.72) : vColor;\n    gl_FragColor = vec4(color, alpha);\n  }\n", Eb = class {
	canvas;
	host;
	renderer;
	scene = new Ap();
	pickingScene = new Ap();
	camera = new mg(-1, 1, 1, -1, .1, 2e3);
	orbitTarget = new J();
	orbitRadius = 1e3;
	orbitAzimuth = 0;
	orbitPolar = 0;
	automaticOrbitPolar = wf.degToRad(13);
	raycaster = new Mg();
	viewPlane = new eh();
	resolution = new Tf(1, 1);
	labels;
	model = null;
	basePositions = /* @__PURE__ */ new Float32Array();
	renderPositions = /* @__PURE__ */ new Float32Array();
	nodePoints = null;
	pickingPoints = null;
	edgeLines = null;
	pickingTarget = new Gf(1, 1, {
		minFilter: mu,
		magFilter: mu,
		depthBuffer: !0,
		stencilBuffer: !1
	});
	width = 1;
	height = 1;
	fitZoom = 1;
	selectedIndex = -1;
	selectedEdgeIndex = -1;
	hoveredIndex = -1;
	searchIndex = -1;
	focusIndices = /* @__PURE__ */ new Uint32Array();
	focusAnimation = null;
	focusAnchorIndex = -1;
	labelPolicy = new db();
	labelsDirty = !0;
	lastLabelUpdate = 0;
	running = !1;
	animationFrame = 0;
	lastFrameAt = performance.now();
	fps = 60;
	frameSamples = [];
	viewChangedCallback = null;
	frameCallback = null;
	rendererName = "WebGL2";
	constructor(e) {
		this.host = e, this.renderer = new Qy({
			antialias: !0,
			alpha: !1,
			powerPreference: "high-performance"
		}), this.renderer.setClearColor(329485, 1), this.renderer.outputColorSpace = Nd, this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75)), this.canvas = this.renderer.domElement;
		let t = this.renderer.getContext(), n = t.getExtension("WEBGL_debug_renderer_info");
		this.rendererName = String(n ? t.getParameter(n.UNMASKED_RENDERER_WEBGL) : t.getParameter(t.RENDERER)), this.canvas.className = "star-map-canvas", this.canvas.tabIndex = 0, this.canvas.setAttribute("role", "application"), this.canvas.setAttribute("aria-label", "二维语义缩放知识星图。使用滚轮缩放，拖动画布平移，点击节点或关系边查看证据。"), e.replaceChildren(this.canvas), this.applyCameraOrbit(), this.labels = new sb(this.scene, this.resolution), new ResizeObserver(() => this.resize()).observe(e), this.resize();
	}
	disposeGraphObjects() {
		let e = /* @__PURE__ */ new Set();
		[
			this.nodePoints,
			this.pickingPoints,
			this.edgeLines
		].forEach((t) => {
			t && (t.parent?.remove(t), e.has(t.geometry) || (t.geometry.dispose(), e.add(t.geometry)), Array.isArray(t.material) ? t.material.forEach((e) => e.dispose()) : t.material.dispose());
		}), this.nodePoints = null, this.pickingPoints = null, this.edgeLines = null;
	}
	setGraph(e, t) {
		this.disposeGraphObjects(), this.model = e, this.basePositions = t.slice(), this.renderPositions = t.slice(), this.focusIndices = /* @__PURE__ */ new Uint32Array(), this.focusAnimation = null, this.focusAnchorIndex = -1;
		let n = yb(this.basePositions, e.hashes);
		for (let t = 0; t < e.nodes.length; t += 1) this.renderPositions[t * 3 + 2] = n.depths[t];
		let r = new wm();
		r.setAttribute("position", new lm(this.renderPositions, 3).setUsage(zd)), r.setAttribute("aColor", new lm(e.colors, 3)), r.setAttribute("aSize", new lm(e.sizes, 1)), r.setAttribute("aState", new lm(new Float32Array(e.nodes.length), 1).setUsage(zd));
		let i = new Float32Array(e.nodes.length * 3);
		for (let t = 0; t < e.nodes.length; t += 1) {
			let e = t + 1;
			i[t * 3] = (e & 255) / 255, i[t * 3 + 1] = (e >>> 8 & 255) / 255, i[t * 3 + 2] = (e >>> 16 & 255) / 255;
		}
		r.setAttribute("aPickColor", new lm(i, 3));
		let a = new Bh({
			vertexShader: bb,
			fragmentShader: xb,
			transparent: !0,
			depthTest: !1,
			depthWrite: !1,
			uniforms: {
				uPixelRatio: { value: this.renderer.getPixelRatio() },
				uSemanticScale: { value: 1 }
			}
		});
		this.nodePoints = new Ch(r, a), this.nodePoints.frustumCulled = !1, this.nodePoints.renderOrder = 5, this.scene.add(this.nodePoints);
		let o = new Bh({
			vertexShader: Sb,
			fragmentShader: Cb,
			depthTest: !1,
			depthWrite: !1,
			uniforms: { uPixelRatio: { value: this.renderer.getPixelRatio() } }
		});
		this.pickingPoints = new Ch(r, o), this.pickingPoints.frustumCulled = !1, this.pickingScene.add(this.pickingPoints);
		let s = new Float32Array(e.edges.length * 6), c = new Float32Array(e.edges.length * 6), l = new Float32Array(e.edges.length * 2), u = new Float32Array(e.edges.length * 2), d = new Float32Array(e.edges.length * 2);
		e.edges.forEach((e, t) => {
			let n = e.isPromoted ? [
				.73,
				.58,
				1
			] : [
				.54,
				.69,
				.8
			];
			c.set(n, t * 6), c.set(n, t * 6 + 3), u[t * 2] = 0, u[t * 2 + 1] = 1, d[t * 2] = +!!e.isPromoted, d[t * 2 + 1] = +!!e.isPromoted;
		});
		let f = new wm();
		f.setAttribute("position", new lm(s, 3).setUsage(zd)), f.setAttribute("aColor", new lm(c, 3)), f.setAttribute("aState", new lm(l, 1).setUsage(zd)), f.setAttribute("aLineT", new lm(u, 1)), f.setAttribute("aDashed", new lm(d, 1));
		let p = new Bh({
			vertexShader: wb,
			fragmentShader: Tb,
			transparent: !0,
			depthTest: !1,
			depthWrite: !1
		});
		this.edgeLines = new _h(f, p), this.edgeLines.frustumCulled = !1, this.edgeLines.renderOrder = 1, this.scene.add(this.edgeLines), this.syncEdgePositions(), this.clearSelection(!1), this.fitToGraph(!1), this.labelsDirty = !0, this.frameSamples = [], this.fps = 60, this.lastFrameAt = performance.now();
	}
	syncEdgePositions() {
		if (!this.model || !this.edgeLines) return;
		let e = this.edgeLines.geometry.getAttribute("position"), t = e.array;
		this.model.edges.forEach((e, n) => {
			t[n * 6] = this.renderPositions[e.source * 3], t[n * 6 + 1] = this.renderPositions[e.source * 3 + 1], t[n * 6 + 2] = this.renderPositions[e.source * 3 + 2], t[n * 6 + 3] = this.renderPositions[e.target * 3], t[n * 6 + 4] = this.renderPositions[e.target * 3 + 1], t[n * 6 + 5] = this.renderPositions[e.target * 3 + 2];
		}), e.needsUpdate = !0;
	}
	syncNodePositions() {
		this.nodePoints && (this.nodePoints.geometry.getAttribute("position").needsUpdate = !0, this.syncEdgePositions(), this.labelsDirty = !0);
	}
	setSelection(e) {
		!this.model || !this.nodePoints || !this.edgeLines || (this.selectedIndex = e, this.selectedEdgeIndex = -1, this.refreshSelectionStates());
	}
	setEdgeSelection(e) {
		!this.model || !this.nodePoints || !this.edgeLines || (this.selectedIndex = -1, this.selectedEdgeIndex = e >= 0 && e < this.model.edges.length ? e : -1, this.refreshSelectionStates());
	}
	refreshSelectionStates() {
		if (!this.model || !this.nodePoints || !this.edgeLines) return;
		let e = this.selectedIndex, t = this.selectedEdgeIndex, n = e >= 0 ? new Set(ae(this.model, e)) : /* @__PURE__ */ new Set(), r = t >= 0 ? this.model.edges[t] : void 0, i = this.nodePoints.geometry.getAttribute("aState").array;
		for (let t = 0; t < i.length; t += 1) e >= 0 ? i[t] = t === e ? 3 : n.has(t) ? 2 : 1 : r ? i[t] = t === r.source || t === r.target ? 3 : 1 : i[t] = 0;
		this.hoveredIndex >= 0 && this.hoveredIndex !== e && (i[this.hoveredIndex] = 4), this.nodePoints.geometry.getAttribute("aState").needsUpdate = !0;
		let a = this.edgeLines.geometry.getAttribute("aState").array;
		this.model.edges.forEach((n, r) => {
			let i = e >= 0 ? n.source === e || n.target === e ? 2 : 1 : t >= 0 ? r === t ? 3 : 1 : 0;
			a[r * 2] = i, a[r * 2 + 1] = i;
		}), this.edgeLines.geometry.getAttribute("aState").needsUpdate = !0, this.labelsDirty = !0;
	}
	clearSelection(e = !0) {
		this.selectedIndex = -1, this.selectedEdgeIndex = -1, this.refreshSelectionStates(), e && this.clearFocus(280);
	}
	setHovered(e) {
		!this.model || !this.nodePoints || e === this.hoveredIndex || (this.hoveredIndex = e, this.refreshSelectionStates());
	}
	setSearchMatch(e) {
		this.searchIndex = e, this.labelsDirty = !0;
	}
	beginFocus(e) {
		e < 0 || e >= this.basePositions.length / 3 || (this.focusAnchorIndex = e, this.focusIndices = Uint32Array.of(e), this.focusAnimation = null);
	}
	animateFocus(e, t, n = 350) {
		let r = new Set(this.focusIndices);
		e.forEach((e) => r.add(e));
		let i = Uint32Array.from(r), a = new Float32Array(i.length * 2), o = new Float32Array(i.length * 2), s = /* @__PURE__ */ new Map(), c = this.focusAnchorIndex, l = c >= 0 ? this.renderPositions[c * 3] - this.basePositions[c * 3] : 0, u = c >= 0 ? this.renderPositions[c * 3 + 1] - this.basePositions[c * 3 + 1] : 0;
		e.forEach((e, n) => s.set(e, [t[n * 2] + l, t[n * 2 + 1] + u])), i.forEach((e, t) => {
			a[t * 2] = this.renderPositions[e * 3], a[t * 2 + 1] = this.renderPositions[e * 3 + 1];
			let n = s.get(e);
			o[t * 2] = n?.[0] ?? this.basePositions[e * 3], o[t * 2 + 1] = n?.[1] ?? this.basePositions[e * 3 + 1];
		}), this.focusIndices = e.slice(), this.focusAnimation = {
			indices: i,
			starts: a,
			targets: o,
			startedAt: performance.now(),
			duration: n,
			clearing: !1
		};
	}
	clearFocus(e = 280) {
		if (!this.focusIndices.length) return;
		let t = new Float32Array(this.focusIndices.length * 2), n = new Float32Array(this.focusIndices.length * 2);
		this.focusIndices.forEach((e, r) => {
			t[r * 2] = this.renderPositions[e * 3], t[r * 2 + 1] = this.renderPositions[e * 3 + 1], n[r * 2] = this.basePositions[e * 3], n[r * 2 + 1] = this.basePositions[e * 3 + 1];
		}), this.focusAnimation = {
			indices: this.focusIndices.slice(),
			starts: t,
			targets: n,
			startedAt: performance.now(),
			duration: e,
			clearing: !0
		};
	}
	setNodeTransientPosition(e, t, n) {
		if (e < 0 || e >= this.renderPositions.length / 3) return;
		let r = t - this.renderPositions[e * 3], i = n - this.renderPositions[e * 3 + 1];
		if (!(!Number.isFinite(r) || !Number.isFinite(i))) {
			if ((this.focusIndices.length ? this.focusIndices : Uint32Array.of(e)).forEach((e) => {
				this.renderPositions[e * 3] += r, this.renderPositions[e * 3 + 1] += i;
			}), this.renderPositions[e * 3] = t, this.renderPositions[e * 3 + 1] = n, this.focusAnimation && !this.focusAnimation.clearing) for (let e = 0; e < this.focusAnimation.indices.length; e += 1) this.focusAnimation.starts[e * 2] += r, this.focusAnimation.starts[e * 2 + 1] += i, this.focusAnimation.targets[e * 2] += r, this.focusAnimation.targets[e * 2 + 1] += i;
			this.syncNodePositions();
		}
	}
	setNodeWorldPosition(e, t, n) {
		this.basePositions[e * 3] = t, this.basePositions[e * 3 + 1] = n, this.renderPositions[e * 3] = t, this.renderPositions[e * 3 + 1] = n, this.syncNodePositions();
	}
	getNodeWorldPosition(e) {
		return {
			x: this.renderPositions[e * 3],
			y: this.renderPositions[e * 3 + 1],
			z: this.renderPositions[e * 3 + 2]
		};
	}
	fitToGraph(e = !0) {
		if (!this.model?.nodes.length) return;
		let t = Infinity, n = -Infinity, r = Infinity, i = -Infinity;
		for (let e = 0; e < this.basePositions.length; e += 3) t = Math.min(t, this.basePositions[e]), n = Math.max(n, this.basePositions[e]), r = Math.min(r, this.basePositions[e + 1]), i = Math.max(i, this.basePositions[e + 1]);
		this.orbitTarget.set((t + n) / 2, (r + i) / 2, 0), this.orbitAzimuth = 0, this.orbitPolar = this.automaticOrbitPolar;
		let a = Math.max(160, this.width - 96), o = Math.max(160, this.height - 112);
		this.camera.zoom = Math.max(.02, Math.min(a / Math.max(160, n - t), o / Math.max(160, i - r))), this.fitZoom = this.camera.zoom, this.camera.updateProjectionMatrix(), this.applyCameraOrbit(), this.labelPolicy.reset(1), this.labelsDirty = !0, e && this.viewChangedCallback?.();
	}
	zoomAt(e, t, n) {
		let r = this.screenToWorld(e, t, this.orbitTarget);
		this.camera.zoom = wf.clamp(this.camera.zoom * n, this.fitZoom * .28, this.fitZoom * 20), this.camera.updateProjectionMatrix();
		let i = this.screenToWorld(e, t, this.orbitTarget), a = new J(r.x - i.x, r.y - i.y, r.z - i.z);
		this.orbitTarget.add(a), this.camera.position.add(a), this.labelsDirty = !0, this.viewChangedCallback?.();
	}
	panBy(e, t) {
		this.camera.updateMatrixWorld();
		let n = new J().setFromMatrixColumn(this.camera.matrixWorld, 0), r = new J().setFromMatrixColumn(this.camera.matrixWorld, 1), i = n.multiplyScalar(-e / this.camera.zoom).add(r.multiplyScalar(t / this.camera.zoom));
		this.orbitTarget.add(i), this.camera.position.add(i), this.labelsDirty = !0, this.viewChangedCallback?.();
	}
	centerOnNode(e, t = 2.1) {
		let n = this.getNodeWorldPosition(e);
		this.orbitTarget.set(n.x, n.y, n.z), this.applyCameraOrbit(), this.zoomRatio < t && (this.camera.zoom = this.fitZoom * t), this.camera.updateProjectionMatrix(), this.labelsDirty = !0, this.viewChangedCallback?.();
	}
	screenToWorld(e, t, n = this.orbitTarget, r) {
		let i = new Tf(e / this.width * 2 - 1, -(t / this.height) * 2 + 1);
		this.camera.updateMatrixWorld(), this.raycaster.setFromCamera(i, this.camera);
		let a = r ? new J(r.x, r.y, r.z).normalize() : this.camera.getWorldDirection(new J());
		this.viewPlane.setFromNormalAndCoplanarPoint(a, new J(n.x, n.y, n.z));
		let o = this.raycaster.ray.intersectPlane(this.viewPlane, new J());
		return o ? {
			x: o.x,
			y: o.y,
			z: o.z
		} : {
			x: n.x,
			y: n.y,
			z: n.z
		};
	}
	projectNode(e) {
		let t = new J(this.renderPositions[e * 3], this.renderPositions[e * 3 + 1], this.renderPositions[e * 3 + 2]).project(this.camera);
		return {
			x: (t.x + 1) * this.width / 2,
			y: (1 - t.y) * this.height / 2,
			visible: Math.abs(t.x) <= 1.08 && Math.abs(t.y) <= 1.08
		};
	}
	pick(e, t) {
		if (!this.pickingPoints || !this.model || e < 0 || t < 0 || e > this.width || t > this.height) return -1;
		let n = /* @__PURE__ */ new Uint8Array(4);
		this.camera.setViewOffset(this.width, this.height, Math.floor(e), Math.floor(t), 1, 1), this.renderer.setRenderTarget(this.pickingTarget), this.renderer.setClearColor(0, 1), this.renderer.clear(), this.renderer.render(this.pickingScene, this.camera), this.renderer.readRenderTargetPixels(this.pickingTarget, 0, 0, 1, 1, n), this.renderer.setRenderTarget(null), this.renderer.setClearColor(329485, 1), this.camera.clearViewOffset(), this.camera.updateProjectionMatrix();
		let r = n[0] + (n[1] << 8) + (n[2] << 16) - 1;
		return r >= 0 && r < this.model.nodes.length ? r : -1;
	}
	pickEdge(e, t, n = 7) {
		if (!this.model || e < 0 || t < 0 || e > this.width || t > this.height) return -1;
		let r = -1, i = n * n;
		return this.model.edges.forEach((a, o) => {
			let s = this.projectNode(a.source), c = this.projectNode(a.target), l = Math.min(s.x, c.x) - n, u = Math.max(s.x, c.x) + n, d = Math.min(s.y, c.y) - n, f = Math.max(s.y, c.y) + n;
			if (e < l || e > u || t < d || t > f) return;
			let p = c.x - s.x, m = c.y - s.y, h = p * p + m * m;
			if (h < 1) return;
			let g = wf.clamp(((e - s.x) * p + (t - s.y) * m) / h, 0, 1), _ = s.x + p * g, v = s.y + m * g, y = (e - _) ** 2 + (t - v) ** 2;
			y <= i && (i = y, r = o);
		}), r;
	}
	getEdgeScreenMidpoint(e) {
		let t = this.model?.edges[e];
		if (!t) return null;
		let n = this.projectNode(t.source), r = this.projectNode(t.target);
		return {
			x: (n.x + r.x) / 2,
			y: (n.y + r.y) / 2
		};
	}
	refreshLabels(e) {
		if (!this.model || !this.labelsDirty || e - this.lastLabelUpdate < 70) return;
		let t = this.labelPolicy.update(this.zoomRatio), n = this.model.nodes.map((e, t) => ({
			nodeIndex: t,
			...this.projectNode(t)
		})), r = this.selectedEdgeIndex >= 0 ? this.model.edges[this.selectedEdgeIndex] : void 0, i = new Set([
			this.selectedIndex,
			r?.source ?? -1,
			r?.target ?? -1,
			this.hoveredIndex,
			this.searchIndex
		].filter((e) => e >= 0)), a = vb(hb(this.model, n, t, i), this.width, this.height);
		this.labels.update(a, this.renderPositions), this.labelsDirty = !1, this.lastLabelUpdate = e;
	}
	tickFocus(e) {
		let t = this.focusAnimation;
		if (!t) return;
		let n = Math.min(1, (e - t.startedAt) / t.duration), r = 1 - (1 - n) ** 3;
		t.indices.forEach((e, n) => {
			this.renderPositions[e * 3] = wf.lerp(t.starts[n * 2], t.targets[n * 2], r), this.renderPositions[e * 3 + 1] = wf.lerp(t.starts[n * 2 + 1], t.targets[n * 2 + 1], r);
		}), this.syncNodePositions(), n >= 1 && (t.clearing && (this.focusIndices = /* @__PURE__ */ new Uint32Array(), this.focusAnchorIndex = -1), this.focusAnimation = null);
	}
	frame = (e) => {
		if (!this.running) return;
		let t = Math.max(1, e - this.lastFrameAt);
		this.frameCallback?.(e, Math.min(.1, t / 1e3)), this.tickFocus(e);
		let n = this.zoomRatio;
		if (this.nodePoints) {
			let e = this.nodePoints.material;
			e.uniforms.uSemanticScale.value = wf.clamp(.8 + Math.log2(Math.max(.35, n)) * .14, .65, 1.65);
		}
		if (this.refreshLabels(e), this.renderer.render(this.scene, this.camera), t < 250) {
			this.frameSamples.push(t), this.frameSamples.length > 60 && this.frameSamples.shift();
			let e = this.frameSamples.reduce((e, t) => e + t, 0) / this.frameSamples.length;
			this.fps = 1e3 / Math.max(1, e);
		}
		this.lastFrameAt = e, this.animationFrame = requestAnimationFrame(this.frame);
	};
	start() {
		this.running || (this.running = !0, this.lastFrameAt = performance.now(), this.animationFrame = requestAnimationFrame(this.frame));
	}
	stop() {
		this.running = !1, cancelAnimationFrame(this.animationFrame);
	}
	resize() {
		let e = this.host.getBoundingClientRect();
		this.width = Math.max(1, Math.round(e.width)), this.height = Math.max(1, Math.round(e.height)), this.camera.left = -this.width / 2, this.camera.right = this.width / 2, this.camera.top = this.height / 2, this.camera.bottom = -this.height / 2, this.camera.updateProjectionMatrix(), this.renderer.setSize(this.width, this.height, !1), this.resolution.set(this.width, this.height), this.labels.resize(this.width, this.height), this.labelsDirty = !0;
	}
	onViewChanged(e) {
		this.viewChangedCallback = e;
	}
	onFrame(e) {
		this.frameCallback = e;
	}
	advanceAutomaticOrbit(e, t) {
		let n = 1 - Math.exp(-Math.max(0, e) * 2.6);
		this.orbitPolar = wf.lerp(this.orbitPolar, this.automaticOrbitPolar, n), this.orbitAzimuth = (this.orbitAzimuth + t * e) % (Math.PI * 2), this.applyCameraOrbit(), this.labelsDirty = !0;
	}
	getCameraOrbitState() {
		return {
			position: this.camera.position.toArray(),
			target: this.orbitTarget.toArray(),
			zoom: this.camera.zoom,
			zoomRatio: this.zoomRatio,
			azimuth: this.orbitAzimuth,
			polar: this.orbitPolar
		};
	}
	applyCameraOrbit() {
		let e = Math.sin(this.orbitPolar), t = new J(this.orbitRadius * e * Math.cos(this.orbitAzimuth), this.orbitRadius * e * Math.sin(this.orbitAzimuth), this.orbitRadius * Math.cos(this.orbitPolar));
		this.camera.position.copy(this.orbitTarget).add(t), this.camera.up.set(0, 1, 0), this.camera.lookAt(this.orbitTarget), this.camera.updateMatrixWorld();
	}
	get zoomRatio() {
		return this.fitZoom > 0 ? this.camera.zoom / this.fitZoom : 1;
	}
	get selectedNodeIndex() {
		return this.selectedIndex;
	}
	get selectedRelationIndex() {
		return this.selectedEdgeIndex;
	}
	getStats() {
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
			renderer: this.rendererName
		};
	}
	dispose() {
		this.stop(), this.disposeGraphObjects(), this.labels.dispose(), this.pickingTarget.dispose(), this.renderer.dispose();
	}
};
//#endregion
//#region frontend/src/main.ts
function Db(e) {
	let t = document.querySelector(e);
	if (!t) throw Error(`Missing required element: ${e}`);
	return t;
}
function Ob(e) {
	return String(e.id ?? e.session_id ?? "");
}
function kb(e) {
	let t = Ob(e), n = String(e.title ?? e.name ?? "未命名会话");
	return t ? `${t} | ${n}` : n;
}
var Ab = "docthinker.promo.autoRotate", jb = "current_session_id", Mb = "docthinker.promo.session";
function Nb() {
	try {
		return localStorage.getItem(jb) || localStorage.getItem(Mb) || "";
	} catch {
		return "";
	}
}
function Pb(e) {
	try {
		localStorage.setItem(jb, e), localStorage.setItem(Mb, e);
	} catch {}
}
function Fb() {
	try {
		let e = localStorage.getItem(Ab);
		if (e === "true") return !0;
		if (e === "false") return !1;
	} catch {
		return;
	}
}
var Ib = new class {
	root = Db("#promo-app");
	stage = Db("#star-map-stage");
	sessionSelect = Db("#session-select");
	searchInput = Db("#node-search");
	searchResults = Db("#search-results");
	panelRoot = Db("#node-panel");
	panelToggle = Db("#toggle-panel");
	rotationButton = Db("#toggle-auto-rotation");
	gestureButton = Db("#toggle-gesture");
	gestureLayer = Db("#gesture-layer");
	gestureVideo = Db("#gesture-video");
	gestureCursor = Db("#gesture-cursor");
	loading = Db("#loading-state");
	loadingDetail = Db("#loading-detail");
	errorState = Db("#error-state");
	errorMessage = Db("#error-message");
	status = Db("#graph-status");
	counts = Db("#graph-counts");
	zoomStatus = Db("#zoom-status");
	apiPrefix = this.root.dataset.apiPrefix || "/api/v1";
	renderer;
	layout = new uu();
	autoRotation = new S({}, Fb());
	input;
	gesture;
	detail = new ce(this.panelRoot);
	model = null;
	currentSessionId = "";
	graphAbort = null;
	chunksAbort = null;
	selectedIndex = -1;
	selectedEdgeIndex = -1;
	graphReady = !1;
	selectedSearchResult = -1;
	statsTimer = 0;
	lastRotationUiKey = "";
	mouseOrbitInteracting = !1;
	gestureInteracting = !1;
	mouseNodeDragging = !1;
	gestureNodeDragging = !1;
	attractionNode = -1;
	attractionRequest = 0;
	constructor() {
		this.renderer = new Eb(this.stage), this.gesture = new ru(this.stage, this.gestureLayer, this.gestureVideo, this.gestureCursor, this.renderer, {
			onSelect: (e) => this.selectNode(e),
			onClearSelection: () => this.clearSelection(),
			onNodeAttractionStart: (e) => this.beginNodeAttraction(e),
			onNodeAttractionEnd: () => this.endNodeAttraction(),
			onHover: (e) => this.onHover(e),
			onNodeMoved: (e) => this.setStatus("节点位置已固定", this.model?.nodes[e]?.label),
			onGestureActiveChange: (e) => {
				this.gestureInteracting = e, this.updateCombinedInteractionState();
			},
			onNodeDraggingChange: (e) => {
				this.gestureNodeDragging = e, this.updateCombinedInteractionState();
			}
		}), this.input = new iu(this.renderer, {
			onSelect: (e) => this.selectNode(e),
			onSelectEdge: (e) => this.selectEdge(e),
			onNodeAttractionStart: (e) => this.beginNodeAttraction(e),
			onNodeAttractionEnd: () => this.endNodeAttraction(),
			onHover: (e) => this.onHover(e),
			onEscape: () => this.clearSelection(),
			onNodeMoved: (e) => this.setStatus("节点位置已固定", this.model?.nodes[e]?.label),
			onPointerInsideChange: (e) => this.autoRotation.setPointerInside(e),
			onPointerDownChange: (e) => this.autoRotation.setPointerDown(e),
			onOrbitInteractionChange: (e) => {
				this.mouseOrbitInteracting = e, this.updateCombinedInteractionState();
			},
			onNodeDraggingChange: (e) => {
				this.mouseNodeDragging = e, this.updateCombinedInteractionState();
			},
			onWheel: () => this.autoRotation.noteWheel(),
			onKeyboardInteraction: () => this.autoRotation.markInteraction()
		}), this.bindControls(), this.renderer.onViewChanged(() => this.updateStats()), this.renderer.onFrame((e, t) => {
			this.tickAutoRotation(e, t), this.gesture.tick(e);
		}), this.autoRotation.setDocumentVisible(!document.hidden), document.addEventListener("visibilitychange", this.onVisibilityChange), this.root.addEventListener("pointerdown", this.onRootPointerDown, !0), document.hidden || this.renderer.start(), this.statsTimer = window.setInterval(() => this.updateStats(), 250), window.__PROMO_GRAPH_DEBUG__ = {
			stats: () => ({
				...this.renderer.getStats(),
				nodes: this.model?.nodes.length ?? 0,
				edges: this.model?.edges.length ?? 0
			}),
			zoom: (e) => {
				let t = this.renderer.zoomRatio;
				this.renderer.zoomAt(this.stage.clientWidth / 2, this.stage.clientHeight / 2, e / Math.max(.001, t));
			},
			select: (e) => this.selectNode(e),
			selectEdge: (e) => this.selectEdge(e),
			fit: () => this.renderer.fitToGraph(),
			autoRotate: (e) => this.setAutoRotationEnabled(e),
			rotation: () => ({
				decision: this.autoRotation.evaluate(),
				state: this.rotationDebugState()
			}),
			gesture: () => this.gesture.getState(),
			attract: (e) => this.beginNodeAttraction(e),
			releaseAttraction: () => this.endNodeAttraction(),
			position: (e) => this.renderer.getNodeWorldPosition(e),
			edgePosition: (e) => this.renderer.getEdgeScreenMidpoint(e)
		};
	}
	bindControls() {
		Db("#zoom-in").addEventListener("click", () => this.renderer.zoomAt(this.stage.clientWidth / 2, this.stage.clientHeight / 2, 1.28)), Db("#zoom-out").addEventListener("click", () => this.renderer.zoomAt(this.stage.clientWidth / 2, this.stage.clientHeight / 2, .78)), Db("#fit-graph").addEventListener("click", () => this.renderer.fitToGraph()), this.rotationButton.addEventListener("click", () => this.setAutoRotationEnabled(!this.autoRotation.snapshot.enabled)), this.gestureButton.addEventListener("click", () => void this.toggleGestureControl()), Db("#close-panel").addEventListener("click", () => this.closePanel()), Db("#retry-load").addEventListener("click", () => void this.loadGraph()), this.panelToggle.addEventListener("click", () => {
			this.selectedIndex < 0 && this.selectedEdgeIndex < 0 || (this.root.classList.contains("is-panel-open") ? this.closePanel() : this.openPanel());
		}), this.sessionSelect.addEventListener("change", () => {
			this.currentSessionId = this.sessionSelect.value, Pb(this.currentSessionId);
			let e = new URL(window.location.href);
			e.searchParams.set("session_id", this.currentSessionId), window.history.replaceState({}, "", e), this.loadGraph();
		}), this.searchInput.addEventListener("input", () => this.updateSearchResults()), this.searchInput.addEventListener("keydown", (e) => this.onSearchKeyDown(e)), this.searchInput.addEventListener("blur", () => window.setTimeout(() => this.hideSearchResults(), 120));
	}
	async init() {
		b({ icons: {
			ArrowLeft: u,
			Hand: d,
			Maximize2: f,
			PanelRight: p,
			RotateCw: m,
			Search: h,
			TriangleAlert: g,
			X: _,
			ZoomIn: v,
			ZoomOut: y
		} }), this.updateRotationButton(), this.updateGestureButton(), await this.loadSessions(), this.currentSessionId && await this.loadGraph();
	}
	async loadSessions() {
		this.showLoading("读取会话列表");
		let e = await fetch(`${this.apiPrefix}/sessions`, { cache: "no-store" });
		if (!e.ok) throw Error(`会话列表读取失败（HTTP ${e.status}）`);
		let t = await e.json(), n = Array.isArray(t) ? t : Array.isArray(t.sessions) ? t.sessions : [];
		this.sessionSelect.replaceChildren(), n.forEach((e) => {
			let t = Ob(e);
			if (!t) return;
			let n = document.createElement("option");
			n.value = t, n.textContent = kb(e), this.sessionSelect.append(n);
		});
		let r = new URLSearchParams(window.location.search).get("session_id") || "", i = Nb(), a = new Set(n.map(Ob));
		if (this.currentSessionId = [
			r,
			i,
			Ob(n[0] ?? {})
		].find((e) => e && a.has(e)) || "", this.sessionSelect.value = this.currentSessionId, this.currentSessionId && Pb(this.currentSessionId), !this.currentSessionId) throw Error("当前没有可显示的知识图谱会话");
	}
	async loadGraph() {
		if (this.currentSessionId) {
			this.graphAbort?.abort(), this.graphAbort = new AbortController(), this.graphReady = !1, this.autoRotation.setLayoutStable(!1), this.clearSelection(), this.showLoading("读取完整节点与关系");
			try {
				let e = `${this.apiPrefix}/knowledge-graph/data?session_id=${encodeURIComponent(this.currentSessionId)}&scope=full`, t = await fetch(e, {
					cache: "no-store",
					signal: this.graphAbort.signal
				});
				if (!t.ok) throw Error(`图谱接口返回 HTTP ${t.status}`);
				let n = await t.json(), r = String(n.metadata?.error ?? "");
				if (r) throw Error(r);
				let i = oe(n);
				if (!i.nodes.length) throw Error("该会话还没有可显示的知识图谱节点");
				this.model = i, this.showLoading(i.nodes.length >= 2e3 ? "正在计算多层 ForceAtlas2 布局" : "正在计算二维力导向布局");
				let a = await this.layout.layout(i);
				this.renderer.setGraph(i, a.positions), this.graphReady = !0, this.autoRotation.setLayoutStable(!0), this.loading.hidden = !0, this.errorState.hidden = !0;
				let o = Number(i.metadata.total_nodes ?? i.nodes.length), s = !!i.metadata.truncated || o > i.nodes.length;
				this.setStatus(s ? "图谱数据被截断" : a.cached ? "知识星图已从缓存恢复" : "知识星图已就绪"), this.updateStats();
			} catch (e) {
				if (this.graphReady = !1, this.autoRotation.setLayoutStable(!1), e.name === "AbortError" || e.message === "Layout request cancelled") return;
				this.showError(e.message || "未知错误");
			}
		}
	}
	selectNode(e) {
		if (!this.model || e < 0) {
			this.clearSelection();
			return;
		}
		if (e === this.selectedIndex) {
			this.clearSelection();
			return;
		}
		this.selectedIndex = e, this.selectedEdgeIndex = -1, this.autoRotation.setHasSelection(!0);
		let t = this.model.nodes[e];
		this.renderer.setSelection(e), this.detail.show(t), this.openPanel(), this.setStatus("已聚焦节点", t.label), this.loadChunks(t.id);
	}
	selectEdge(e) {
		if (!this.model || e < 0 || e >= this.model.edges.length) {
			this.clearSelection();
			return;
		}
		if (e === this.selectedEdgeIndex) {
			this.clearSelection();
			return;
		}
		this.selectedIndex = -1, this.selectedEdgeIndex = e, this.autoRotation.setHasSelection(!0);
		let t = this.model.edges[e], n = this.model.nodes[t.source], r = this.model.nodes[t.target];
		this.renderer.setEdgeSelection(e), this.detail.showEdge(t, n, r), this.openPanel(), this.setStatus(t.kind === "eclrr_v4" ? "已聚焦 ECLRR-v4 推断关系" : "已聚焦事实关系", `${n.label} → ${r.label}`), this.loadEdgeChunks(e);
	}
	beginNodeAttraction(e) {
		if (!this.model || e < 0 || e >= this.model.nodes.length) return;
		this.attractionNode = e;
		let t = ++this.attractionRequest;
		this.renderer.setSelection(e), this.renderer.beginFocus(e), this.layout.focus(e).then((n) => {
			t !== this.attractionRequest || this.attractionNode !== e || !n.indices.length || this.renderer.animateFocus(n.indices, n.positions, 350);
		}).catch(() => void 0);
	}
	endNodeAttraction() {
		this.attractionNode < 0 || (this.attractionNode = -1, this.attractionRequest += 1, this.renderer.clearFocus(320), this.selectedEdgeIndex >= 0 ? this.renderer.setEdgeSelection(this.selectedEdgeIndex) : this.renderer.setSelection(this.selectedIndex));
	}
	clearSelection() {
		this.attractionNode = -1, this.attractionRequest += 1, this.selectedIndex = -1, this.selectedEdgeIndex = -1, this.autoRotation.setHasSelection(!1), this.chunksAbort?.abort(), this.renderer.clearSelection(!0), this.detail.hide(), this.closePanel(), this.setStatus(this.model ? "知识星图已就绪" : "准备加载");
	}
	async loadChunks(e) {
		this.chunksAbort?.abort();
		let t = new AbortController();
		this.chunksAbort = t;
		try {
			let n = `${this.apiPrefix}/knowledge-graph/entity-chunks?session_id=${encodeURIComponent(this.currentSessionId)}&entity_id=${encodeURIComponent(e)}&max_chunks=0`, r = await fetch(n, {
				cache: "no-store",
				signal: t.signal
			}), i = await r.json();
			if (!r.ok || i.error) throw Error(i.error || `HTTP ${r.status}`);
			this.model?.nodes[this.selectedIndex]?.id === e && this.detail.showChunks(i.chunks ?? []);
		} catch (e) {
			e.name !== "AbortError" && this.detail.showChunks([], `原文读取失败：${e.message}`);
		}
	}
	async loadEdgeChunks(e) {
		let t = this.model?.edges[e];
		if (!t) return;
		this.chunksAbort?.abort();
		let n = new AbortController();
		this.chunksAbort = n;
		try {
			let r = new URLSearchParams({
				session_id: this.currentSessionId,
				source_id: t.sourceId,
				edge_id: t.id,
				max_chunks: "0"
			}), i = await fetch(`${this.apiPrefix}/knowledge-graph/edge-chunks?${r}`, {
				cache: "no-store",
				signal: n.signal
			}), a = await i.json();
			if (!i.ok || a.error) throw Error(a.error || `HTTP ${i.status}`);
			this.selectedEdgeIndex === e && this.model?.edges[e]?.id === t.id && this.detail.showChunks(a.chunks ?? []);
		} catch (t) {
			t.name !== "AbortError" && this.selectedEdgeIndex === e && this.detail.showChunks([], `原文读取失败：${t.message}`);
		}
	}
	updateSearchResults() {
		if (!this.model) return;
		let e = this.searchInput.value.trim().toLocaleLowerCase();
		if (this.searchResults.replaceChildren(), this.selectedSearchResult = -1, !e) {
			this.renderer.setSearchMatch(-1), this.hideSearchResults();
			return;
		}
		let t = this.model.nodes.map((e, t) => ({
			node: e,
			nodeIndex: t,
			label: e.label.toLocaleLowerCase()
		})).filter((t) => t.label.includes(e) || t.node.type.toLocaleLowerCase().includes(e)).sort((t, n) => Number(n.label.startsWith(e)) - Number(t.label.startsWith(e)) || n.node.degree - t.node.degree).slice(0, 12);
		t.forEach((e, t) => {
			let n = document.createElement("button");
			n.type = "button", n.className = "search-result", n.role = "option", n.dataset.nodeIndex = String(e.nodeIndex), n.dataset.resultIndex = String(t);
			let r = document.createElement("strong");
			r.textContent = e.node.label;
			let i = document.createElement("small");
			i.textContent = `${e.node.type} · ${e.node.degree} 条连接`, n.append(r, i), n.addEventListener("pointerdown", (e) => e.preventDefault()), n.addEventListener("click", () => this.chooseSearchResult(e.nodeIndex)), this.searchResults.append(n);
		}), this.searchResults.hidden = t.length === 0, this.searchInput.setAttribute("aria-expanded", String(t.length > 0)), this.renderer.setSearchMatch(t[0]?.nodeIndex ?? -1);
	}
	onSearchKeyDown(e) {
		let t = [...this.searchResults.querySelectorAll(".search-result")];
		if (t.length) {
			if (e.key === "ArrowDown") this.selectedSearchResult = Math.min(t.length - 1, this.selectedSearchResult + 1);
			else if (e.key === "ArrowUp") this.selectedSearchResult = Math.max(0, this.selectedSearchResult - 1);
			else if (e.key === "Enter") {
				let n = t[Math.max(0, this.selectedSearchResult)];
				n && this.chooseSearchResult(Number(n.dataset.nodeIndex)), e.preventDefault();
				return;
			} else if (e.key === "Escape") {
				this.hideSearchResults();
				return;
			} else return;
			t.forEach((e, t) => e.setAttribute("aria-selected", String(t === this.selectedSearchResult))), t[this.selectedSearchResult]?.scrollIntoView({ block: "nearest" }), e.preventDefault();
		}
	}
	chooseSearchResult(e) {
		this.selectNode(e), this.renderer.centerOnNode(e, 2.1), this.searchInput.value = this.model?.nodes[e]?.label ?? "", this.renderer.setSearchMatch(e), this.hideSearchResults();
	}
	hideSearchResults() {
		this.searchResults.hidden = !0, this.searchInput.setAttribute("aria-expanded", "false");
	}
	onHover(e) {
		e >= 0 && this.model ? this.status.textContent = this.model.nodes[e].label : this.selectedIndex < 0 && this.selectedEdgeIndex < 0 && this.model && (this.status.textContent = "知识星图已就绪");
	}
	openPanel() {
		this.selectedIndex < 0 && this.selectedEdgeIndex < 0 || (this.panelRoot.hidden = !1, requestAnimationFrame(() => this.root.classList.add("is-panel-open")), this.panelToggle.setAttribute("aria-expanded", "true"));
	}
	closePanel() {
		this.root.classList.remove("is-panel-open"), this.panelToggle.setAttribute("aria-expanded", "false"), window.setTimeout(() => {
			this.root.classList.contains("is-panel-open") || (this.panelRoot.hidden = !0);
		}, 230);
	}
	showLoading(e) {
		this.loading.hidden = !1, this.loadingDetail.textContent = e, this.errorState.hidden = !0;
	}
	showError(e) {
		this.loading.hidden = !0, this.errorState.hidden = !1, this.errorMessage.textContent = e, this.setStatus("图谱加载失败");
	}
	setStatus(e, t = "") {
		this.status.textContent = t ? `${e} · ${t}` : e;
	}
	updateStats() {
		if (document.hidden) return;
		let e = this.renderer.getStats();
		this.counts.textContent = `${this.model?.nodes.length ?? 0} 节点 · ${this.model?.edges.length ?? 0} 关系`, this.zoomStatus.textContent = `${e.zoomRatio.toFixed(2)}×`, this.stage.dataset.webglReady = String(this.graphReady), this.stage.dataset.nodeCount = String(this.model?.nodes.length ?? 0), this.stage.dataset.edgeCount = String(this.model?.edges.length ?? 0), this.stage.dataset.selectedNode = String(this.selectedIndex), this.stage.dataset.selectedEdge = String(this.selectedEdgeIndex), this.stage.dataset.labelLevel = String(e.labelLevel), this.stage.dataset.zoomRatio = e.zoomRatio.toFixed(3), this.stage.dataset.fps = e.fps.toFixed(1), this.stage.dataset.drawCalls = String(e.calls), this.stage.dataset.labelCount = String(e.labels), this.stage.dataset.glyphCount = String(e.glyphs), this.stage.dataset.triangles = String(e.triangles);
	}
	setAutoRotationEnabled(e) {
		this.autoRotation.setEnabled(e);
		try {
			localStorage.setItem(Ab, String(e));
		} catch {}
		this.updateRotationButton();
	}
	async toggleGestureControl() {
		if (this.gesture.getState().enabled) {
			this.gesture.stop(), this.gestureInteracting = !1, this.gestureNodeDragging = !1, this.updateCombinedInteractionState(), this.updateGestureButton();
			return;
		}
		this.gestureButton.disabled = !0, this.gestureButton.title = "正在启动手势控制", this.gestureButton.setAttribute("aria-label", "正在启动手势控制");
		let e = await this.gesture.start();
		this.gestureButton.disabled = !1, this.updateGestureButton(e);
	}
	updateGestureButton(e = this.gesture.getState().enabled) {
		let t = e ? "关闭手势控制" : "开启手势控制";
		this.gestureButton.title = t, this.gestureButton.setAttribute("aria-label", t), this.gestureButton.setAttribute("aria-pressed", String(e));
	}
	tickAutoRotation(e, t) {
		let n = this.autoRotation.evaluate(e);
		n.shouldRotate && this.renderer.advanceAutomaticOrbit(t, this.autoRotation.config.angularSpeedRadPerSecond);
		let r = `${this.autoRotation.snapshot.enabled}:${n.pausedReason ?? "rotating"}`;
		r !== this.lastRotationUiKey && (this.lastRotationUiKey = r, this.updateRotationButton(n.pausedReason)), this.stage.dataset.autoRotating = String(n.shouldRotate), this.stage.dataset.rotationPauseReason = n.pausedReason ?? "";
	}
	updateRotationButton(e = this.autoRotation.evaluate().pausedReason) {
		let t = this.autoRotation.snapshot.enabled, n = t ? e === "selection" ? "自动旋转已开启，当前因选中图谱元素而暂停" : "自动旋转已开启" : "自动旋转已关闭";
		this.rotationButton.title = n, this.rotationButton.setAttribute("aria-label", n), this.rotationButton.setAttribute("aria-pressed", String(t));
	}
	rotationDebugState() {
		return {
			...this.autoRotation.snapshot,
			camera: this.renderer.getCameraOrbitState()
		};
	}
	updateCombinedInteractionState() {
		this.autoRotation.setOrbitInteracting(this.mouseOrbitInteracting || this.gestureInteracting), this.autoRotation.setNodeDragging(this.mouseNodeDragging || this.gestureNodeDragging);
	}
	onRootPointerDown = () => this.autoRotation.markInteraction();
	onVisibilityChange = () => {
		let e = !document.hidden;
		this.autoRotation.setDocumentVisible(e), this.gesture.setSuspended(!e), e ? this.renderer.start() : this.renderer.stop(), this.updateRotationButton();
	};
	dispose() {
		window.clearInterval(this.statsTimer), this.graphAbort?.abort(), this.chunksAbort?.abort(), document.removeEventListener("visibilitychange", this.onVisibilityChange), this.root.removeEventListener("pointerdown", this.onRootPointerDown, !0), this.input.dispose(), this.gesture.dispose(), this.layout.dispose(), this.renderer.dispose(), delete window.__PROMO_GRAPH_DEBUG__;
	}
}();
Ib.init().catch((e) => {
	let t = document.querySelector("#loading-state"), n = document.querySelector("#error-state"), r = document.querySelector("#error-message");
	t && (t.hidden = !0), n && (n.hidden = !1), r && (r.textContent = e.message);
}), window.addEventListener("beforeunload", () => Ib.dispose(), { once: !0 });
//#endregion
