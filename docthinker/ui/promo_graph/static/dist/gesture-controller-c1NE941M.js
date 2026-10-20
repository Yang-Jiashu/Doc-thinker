//#region frontend/src/gesture/gesture-preferences.ts
var e = Object.freeze({
	mode: "inspect",
	dominantHand: "right",
	smoothingMs: 70,
	dwellMs: 650,
	resetHoldMs: 1e3,
	activationMs: 180,
	movementThresholdPx: 5,
	pinchEnterRatio: .24,
	pinchReleaseRatio: .34,
	panGain: 1,
	zoomGain: 1,
	orbitGain: 1
});
function t(t) {
	let n = t && typeof t == "object" ? t : {}, r = (t, r, i) => {
		let a = n[t];
		return typeof a == "number" && Number.isFinite(a) ? Math.min(i, Math.max(r, a)) : e[t];
	}, i = r("pinchEnterRatio", .1, .4);
	return {
		mode: n.mode === "presentation" ? "presentation" : "inspect",
		dominantHand: n.dominantHand === "left" || n.dominantHand === "auto" ? n.dominantHand : "right",
		smoothingMs: r("smoothingMs", 0, 250),
		dwellMs: r("dwellMs", 250, 2e3),
		resetHoldMs: r("resetHoldMs", 500, 3e3),
		activationMs: r("activationMs", 80, 600),
		movementThresholdPx: r("movementThresholdPx", 2, 30),
		pinchEnterRatio: i,
		pinchReleaseRatio: Math.max(i + .05, r("pinchReleaseRatio", .15, .6)),
		panGain: r("panGain", .25, 3),
		zoomGain: r("zoomGain", .25, 3),
		orbitGain: r("orbitGain", .25, 3)
	};
}
//#endregion
//#region frontend/src/gesture/gesture-metrics.ts
function n(e, t) {
	let n = String(e ?? "").trim().toLowerCase();
	return n === "left" ? "left" : n === "right" ? "right" : t >= .5 ? "left" : "right";
}
function r(e, t) {
	return Math.hypot(e.x - t.x, e.y - t.y, (e.z ?? 0) - (t.z ?? 0));
}
function i(t, n = !1, i = e) {
	if (t.length < 21 || t.slice(0, 21).some((e) => !e || !Number.isFinite(e.x) || !Number.isFinite(e.y) || !Number.isFinite(e.z ?? 0))) return null;
	let a = t[0], o = t[4], s = t[8], c = Math.max(r(a, t[9]), .035), l = r(o, s) / c, u = n ? l < i.pinchReleaseRatio : l < i.pinchEnterRatio, d = [
		8,
		12,
		16,
		20
	], f = [
		6,
		10,
		14,
		18
	], ee = d.map((e, n) => r(t[e], a) > r(t[f[n]], a) * 1.08), te = ee.filter(Boolean).length, ne = [
		0,
		5,
		9,
		13,
		17
	].reduce((e, n) => ({
		x: e.x + t[n].x / 5,
		y: e.y + t[n].y / 5
	}), {
		x: 0,
		y: 0
	});
	return {
		point: {
			x: 1 - s.x,
			y: s.y
		},
		thumbPoint: {
			x: 1 - o.x,
			y: o.y
		},
		palmPoint: {
			x: 1 - ne.x,
			y: ne.y
		},
		pinch: u,
		pinchDistance: l,
		openPalm: !u && te >= 4,
		extendedFingers: te,
		pointing: !u && ee[0] && te === 1
	};
}
function a(e, t, n, r = "right") {
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
function o(e, t, n = .34) {
	return e ? {
		x: e.x * (1 - n) + t.x * n,
		y: e.y * (1 - n) + t.y * n
	} : { ...t };
}
//#endregion
//#region node_modules/@mediapipe/tasks-vision/vision_bundle.mjs
var s = typeof self < "u" ? self : {};
function c(e, t) {
	t: {
		for (var n = ["CLOSURE_FLAGS"], r = s, i = 0; i < n.length; i++) if ((r = r[n[i]]) == null) {
			n = null;
			break t;
		}
		n = r;
	}
	return (e = n && n[e]) ?? t;
}
function l() {
	throw Error("Invalid UTF8");
}
function u(e, t) {
	return t = String.fromCharCode.apply(null, t), e == null ? t : e + t;
}
var d, f, ee = typeof TextDecoder < "u", te, ne = typeof TextEncoder < "u";
function re(e) {
	if (ne) e = (te ||= new TextEncoder()).encode(e);
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
function ie(e) {
	s.setTimeout((() => {
		throw e;
	}), 0);
}
var ae, oe = c(610401301, !1), se = c(748402147, !0);
function ce() {
	var e = s.navigator;
	return (e &&= e.userAgent) ? e : "";
}
var le = s.navigator;
function ue(e) {
	return ue[" "](e), e;
}
ae = le && le.userAgentData || null, ue[" "] = function() {};
var de = {}, fe = null;
function pe(e) {
	let t = e.length, n = 3 * t / 4;
	n % 3 ? n = Math.floor(n) : "=.".indexOf(e[t - 1]) != -1 && (n = "=.".indexOf(e[t - 2]) == -1 ? n - 1 : n - 2);
	let r = new Uint8Array(n), i = 0;
	return function(e, t) {
		function n(t) {
			for (; r < e.length;) {
				let t = e.charAt(r++), n = fe[t];
				if (n != null) return n;
				if (!/^[\s\xa0]*$/.test(t)) throw Error("Unknown base64 encoding at char: " + t);
			}
			return t;
		}
		me();
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
function me() {
	if (!fe) {
		fe = {};
		var e = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".split(""), t = [
			"+/=",
			"+/",
			"-_=",
			"-_.",
			"-_"
		];
		for (let n = 0; n < 5; n++) {
			let r = e.concat(t[n].split(""));
			de[n] = r;
			for (let e = 0; e < r.length; e++) {
				let t = r[e];
				fe[t] === void 0 && (fe[t] = e);
			}
		}
	}
}
var he = typeof Uint8Array < "u", ge = !(!(oe && ae && ae.brands.length > 0) && (ce().indexOf("Trident") != -1 || ce().indexOf("MSIE") != -1)) && typeof btoa == "function", _e = /[-_.]/g, ve = {
	"-": "+",
	_: "/",
	".": "="
};
function ye(e) {
	return ve[e] || "";
}
function be(e) {
	if (!ge) return pe(e);
	e = _e.test(e) ? e.replace(_e, ye) : e, e = atob(e);
	let t = new Uint8Array(e.length);
	for (let n = 0; n < e.length; n++) t[n] = e.charCodeAt(n);
	return t;
}
function xe(e) {
	return he && e != null && e instanceof Uint8Array;
}
var Se = {};
function Ce() {
	return Ee ||= new Te(null, Se);
}
function we(e) {
	Oe(Se);
	var t = e.g;
	return (t = t == null || xe(t) ? t : typeof t == "string" ? be(t) : null) == null ? t : e.g = t;
}
var Te = class {
	h() {
		return new Uint8Array(we(this) || 0);
	}
	constructor(e, t) {
		if (Oe(t), this.g = e, e != null && e.length === 0) throw Error("ByteString should be constructed with non-empty values");
	}
}, Ee, De;
function Oe(e) {
	if (e !== Se) throw Error("illegal external caller");
}
function ke(e, t) {
	e.__closure__error__context__984382 ||= {}, e.__closure__error__context__984382.severity = t;
}
function Ae(e) {
	return ke(e = Error(e), "warning"), e;
}
function je(e, t) {
	if (e != null) {
		var n = De ??= {}, r = n[e] || 0;
		r >= t || (n[e] = r + 1, ke(e = Error(), "incident"), ie(e));
	}
}
function Me() {
	return typeof BigInt == "function";
}
var Ne = typeof Symbol == "function" && typeof Symbol() == "symbol";
function Pe(e, t, n = !1) {
	return typeof Symbol == "function" && typeof Symbol() == "symbol" ? n && Symbol.for && e ? Symbol.for(e) : e == null ? Symbol() : Symbol(e) : t;
}
var Fe = Pe("jas", void 0, !0), Ie = Pe(void 0, "0di"), Le = Pe(void 0, "1oa"), Re = Pe(void 0, Symbol()), ze = Pe(void 0, "0ub"), Be = Pe(void 0, "0ubs"), Ve = Pe(void 0, "0ubsb"), He = Pe(void 0, "0actk"), Ue = Pe("m_m", "Pa", !0), We = Pe(), Ge = { Ga: {
	value: 0,
	configurable: !0,
	writable: !0,
	enumerable: !1
} }, Ke = Object.defineProperties, p = Ne ? Fe : "Ga", qe, Je = [];
function Ye(e, t) {
	Ne || p in e || Ke(e, Ge), e[p] |= t;
}
function m(e, t) {
	Ne || p in e || Ke(e, Ge), e[p] = t;
}
function Xe(e) {
	return Ye(e, 34), e;
}
function Ze(e) {
	return Ye(e, 8192), e;
}
m(Je, 7), qe = Object.freeze(Je);
var Qe = {};
function $e(e, t) {
	return t === void 0 ? e.h !== et && !!(2 & (0 | e.v[p])) : !!(2 & t) && e.h !== et;
}
var et = {};
function tt(e, t) {
	if (e != null) {
		if (typeof e == "string") e = e ? new Te(e, Se) : Ce();
		else if (e.constructor !== Te) if (xe(e)) e = e.length ? new Te(new Uint8Array(e), Se) : Ce();
		else {
			if (!t) throw Error();
			e = void 0;
		}
	}
	return e;
}
var nt = class {
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
}, rt = Object.freeze({});
function it(e, t, n) {
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
var at = {};
function ot(e) {
	return 128 & e ? at : void 0;
}
function st(e) {
	return e.Na = !0, e;
}
var ct = st(((e) => typeof e == "number")), lt = st(((e) => typeof e == "string")), ut = st(((e) => typeof e == "boolean")), dt = typeof s.BigInt == "function" && typeof s.BigInt(0) == "bigint";
function ft(e) {
	var t = e;
	if (lt(t)) {
		if (!/^\s*(?:-?[1-9]\d*|0)?\s*$/.test(t)) throw Error(String(t));
	} else if (ct(t) && !Number.isSafeInteger(t)) throw Error(String(t));
	return dt ? BigInt(e) : e = ut(e) ? e ? "1" : "0" : lt(e) ? e.trim() || "0" : String(e);
}
var pt = st(((e) => dt ? e >= ht && e <= _t : e[0] === "-" ? vt(e, mt) : vt(e, gt))), mt = (-(2 ** 53 - 1)).toString(), ht = dt ? BigInt(-(2 ** 53 - 1)) : void 0, gt = (2 ** 53 - 1).toString(), _t = dt ? BigInt(2 ** 53 - 1) : void 0;
function vt(e, t) {
	if (e.length > t.length) return !1;
	if (e.length < t.length || e === t) return !0;
	for (let n = 0; n < e.length; n++) {
		let r = e[n], i = t[n];
		if (r > i) return !1;
		if (r < i) return !0;
	}
}
var yt = typeof Uint8Array.prototype.slice == "function", bt, h = 0, g = 0;
function xt(e) {
	let t = e >>> 0;
	h = t, g = (e - t) / 4294967296 >>> 0;
}
function St(e) {
	if (e < 0) {
		xt(-e);
		let [t, n] = jt(h, g);
		h = t >>> 0, g = n >>> 0;
	} else xt(e);
}
function Ct(e) {
	let t = bt ||= /* @__PURE__ */ new DataView(/* @__PURE__ */ new ArrayBuffer(8));
	t.setFloat32(0, +e, !0), g = 0, h = t.getUint32(0, !0);
}
function wt(e, t) {
	let n = 4294967296 * t + (e >>> 0);
	return Number.isSafeInteger(n) ? n : Dt(e, t);
}
function Tt(e, t) {
	return ft(Me() ? BigInt.asUintN(64, (BigInt(t >>> 0) << BigInt(32)) + BigInt(e >>> 0)) : Dt(e, t));
}
function Et(e, t) {
	return Me() ? ft(BigInt.asIntN(64, (BigInt.asUintN(32, BigInt(t)) << BigInt(32)) + BigInt.asUintN(32, BigInt(e)))) : ft(kt(e, t));
}
function Dt(e, t) {
	if (e >>>= 0, (t >>>= 0) <= 2097151) var n = "" + (4294967296 * t + e);
	else Me() ? n = "" + (BigInt(t) << BigInt(32) | BigInt(e)) : (e = (16777215 & e) + 6777216 * (n = 16777215 & (e >>> 24 | t << 8)) + 6710656 * (t = t >> 16 & 65535), n += 8147497 * t, t *= 2, e >= 1e7 && (n += e / 1e7 >>> 0, e %= 1e7), n >= 1e7 && (t += n / 1e7 >>> 0, n %= 1e7), n = t + Ot(n) + Ot(e));
	return n;
}
function Ot(e) {
	return e = String(e), "0000000".slice(e.length) + e;
}
function kt(e, t) {
	if (2147483648 & t) if (Me()) e = "" + (BigInt(0 | t) << BigInt(32) | BigInt(e >>> 0));
	else {
		let [n, r] = jt(e, t);
		e = "-" + Dt(n, r);
	}
	else e = Dt(e, t);
	return e;
}
function At(e) {
	if (e.length < 16) St(Number(e));
	else if (Me()) e = BigInt(e), h = Number(e & BigInt(4294967295)) >>> 0, g = Number(e >> BigInt(32) & BigInt(4294967295));
	else {
		let t = +(e[0] === "-");
		g = h = 0;
		let n = e.length;
		for (let r = t, i = (n - t) % 6 + t; i <= n; r = i, i += 6) {
			let t = Number(e.slice(r, i));
			g *= 1e6, h = 1e6 * h + t, h >= 4294967296 && (g += Math.trunc(h / 4294967296), g >>>= 0, h >>>= 0);
		}
		if (t) {
			let [e, t] = jt(h, g);
			h = e, g = t;
		}
	}
}
function jt(e, t) {
	return t = ~t, e ? e = 1 + ~e : t += 1, [e, t];
}
function Mt(e) {
	return Array.prototype.slice.call(e);
}
var Nt = typeof BigInt == "function" ? BigInt.asIntN : void 0, Pt = typeof BigInt == "function" ? BigInt.asUintN : void 0, Ft = Number.isSafeInteger, It = Number.isFinite, Lt = Math.trunc, Rt = ft(0);
function zt(e) {
	if (e != null && typeof e != "number") throw Error(`Value of float/double field must be a number, found ${typeof e}: ${e}`);
	return e;
}
function Bt(e) {
	return e == null || typeof e == "number" ? e : e === "NaN" || e === "Infinity" || e === "-Infinity" ? Number(e) : void 0;
}
function Vt(e) {
	if (e != null && typeof e != "boolean") {
		var t = typeof e;
		throw Error(`Expected boolean but got ${t == "object" ? e ? Array.isArray(e) ? "array" : t : "null" : t}: ${e}`);
	}
	return e;
}
function Ht(e) {
	return e == null || typeof e == "boolean" ? e : typeof e == "number" ? !!e : void 0;
}
var Ut = /^-?([1-9][0-9]*|0)(\.[0-9]+)?$/;
function Wt(e) {
	switch (typeof e) {
		case "bigint": return !0;
		case "number": return It(e);
		case "string": return Ut.test(e);
		default: return !1;
	}
}
function Gt(e) {
	if (e == null) return e;
	if (typeof e == "string" && e) e = +e;
	else if (typeof e != "number") return;
	return It(e) ? 0 | e : void 0;
}
function Kt(e) {
	if (e == null) return e;
	if (typeof e == "string" && e) e = +e;
	else if (typeof e != "number") return;
	return It(e) ? e >>> 0 : void 0;
}
function qt(e) {
	let t = e.length;
	return (e[0] === "-" ? t < 20 || t === 20 && e <= "-9223372036854775808" : t < 19 || t === 19 && e <= "9223372036854775807") ? e : (At(e), kt(h, g));
}
function Jt(e) {
	if (e = Lt(e), !Ft(e)) {
		St(e);
		var t = h, n = g;
		(e = 2147483648 & n) && (n = ~n >>> 0, (t = 1 + ~t >>> 0) == 0 && (n = n + 1 >>> 0)), e = typeof (t = wt(t, n)) == "number" ? e ? -t : t : e ? "-" + t : t;
	}
	return e;
}
function Yt(e) {
	var t = Lt(Number(e));
	return Ft(t) ? String(t) : ((t = e.indexOf(".")) !== -1 && (e = e.substring(0, t)), qt(e));
}
function Xt(e) {
	var t = Lt(Number(e));
	return Ft(t) ? ft(t) : ((t = e.indexOf(".")) !== -1 && (e = e.substring(0, t)), Me() ? ft(Nt(64, BigInt(e))) : ft(qt(e)));
}
function Zt(e) {
	return Ft(e) ? e = ft(Jt(e)) : (e = Lt(e), Ft(e) ? e = String(e) : (St(e), e = kt(h, g)), e = ft(e)), e;
}
function Qt(e) {
	let t = typeof e;
	return e == null ? e : t === "bigint" ? ft(Nt(64, e)) : Wt(e) ? t === "string" ? Xt(e) : Zt(e) : void 0;
}
function $t(e) {
	if (typeof e != "string") throw Error();
	return e;
}
function en(e) {
	if (e != null && typeof e != "string") throw Error();
	return e;
}
function _(e) {
	return e == null || typeof e == "string" ? e : void 0;
}
function tn(e, t, n, r) {
	return e != null && e[Ue] === Qe ? e : Array.isArray(e) ? ((r = (n = 0 | e[p]) | 32 & r | 2 & r) !== n && m(e, r), new t(e)) : (n ? 2 & r ? ((e = t[Ie]) || (Xe((e = new t()).v), e = t[Ie] = e), t = e) : t = new t() : t = void 0, t);
}
function nn(e, t, n) {
	if (t) t: {
		if (!Wt(t = e)) throw Ae("int64");
		switch (typeof t) {
			case "string":
				t = Xt(t);
				break t;
			case "bigint":
				t = ft(Nt(64, t));
				break t;
			default: t = Zt(t);
		}
	}
	else t = Qt(e);
	return (e = t) ?? (n ? Rt : void 0);
}
var rn = {}, an = function() {
	try {
		return ue(new class extends Map {
			constructor() {
				super();
			}
		}()), !1;
	} catch {
		return !0;
	}
}(), on = class {
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
}, sn = an ? (Object.setPrototypeOf(on.prototype, Map.prototype), Object.defineProperties(on.prototype, { size: {
	value: 0,
	configurable: !0,
	enumerable: !0,
	writable: !0
} }), on) : class extends Map {
	constructor() {
		super();
	}
};
function cn(e) {
	return e;
}
function ln(e) {
	if (2 & e.J) throw Error("Cannot mutate an immutable Map");
}
var un = class extends sn {
	constructor(e, t, n = cn, r = cn) {
		super(), this.J = 0 | e[p], this.K = t, this.S = n, this.fa = this.K ? dn : r;
		for (let i = 0; i < e.length; i++) {
			let a = e[i], o = n(a[0], !1, !0), s = a[1];
			t ? s === void 0 && (s = null) : s = r(a[1], !1, !0, void 0, void 0, this.J), super.set(o, s);
		}
	}
	V(e) {
		return Ze(Array.from(super.entries(), e));
	}
	clear() {
		ln(this), super.clear();
	}
	delete(e) {
		return ln(this), super.delete(this.S(e, !0, !1));
	}
	entries() {
		if (this.K) {
			var e = super.keys();
			e = new nt(e, fn, this);
		} else e = super.entries();
		return e;
	}
	values() {
		if (this.K) {
			var e = super.keys();
			e = new nt(e, un.prototype.get, this);
		} else e = super.values();
		return e;
	}
	forEach(e, t) {
		this.K ? super.forEach(((n, r, i) => {
			e.call(t, i.get(r), r, i);
		})) : super.forEach(e, t);
	}
	set(e, t) {
		return ln(this), (e = this.S(e, !0, !1)) == null ? this : t == null ? (super.delete(e), this) : super.set(e, this.fa(t, !0, !0, this.K, !1, this.J));
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
function dn(e, t, n, r, i, a) {
	return e = tn(e, r, n, a), i && (e = Nn(e)), e;
}
function fn(e) {
	return [e, this.get(e)];
}
var pn;
function mn() {
	return pn ||= new un(Xe([]), void 0, void 0, void 0, rn);
}
function hn(e) {
	return Re ? e[Re] : void 0;
}
function gn(e, t) {
	for (let n in e) !isNaN(n) && t(e, +n, e[n]);
}
un.prototype.toJSON = void 0;
var _n = class {}, vn = { Ka: !0 };
function yn(e, t) {
	t < 100 || je(Be, 1);
}
function bn(e, t, n, r) {
	let i = r !== void 0;
	r = !!r;
	var a, o = Re;
	!i && Ne && o && (a = e[o]) && gn(a, yn), o = [];
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
	return t && (l ? o.push(t) : o[a] = t), i && Re && (e = hn(e)) && e instanceof _n && (o[Re] = function(e) {
		let t = new _n();
		return gn(e, ((e, n, r) => {
			t[n] = Mt(r);
		})), t.da = e.da, t;
	}(e)), o;
}
function xn(e) {
	return e[0] = Sn(e[0]), e[1] = Sn(e[1]), e;
}
function Sn(e) {
	switch (typeof e) {
		case "number": return Number.isFinite(e) ? e : "" + e;
		case "bigint": return pt(e) ? Number(e) : "" + e;
		case "boolean": return +!!e;
		case "object":
			if (Array.isArray(e)) {
				var t = 0 | e[p];
				return e.length === 0 && 1 & t ? void 0 : bn(e, t, Sn);
			}
			if (e != null && e[Ue] === Qe) return Tn(e);
			if (e instanceof Te) {
				if ((t = e.g) == null) e = "";
				else if (typeof t == "string") e = t;
				else {
					if (ge) {
						for (var n = "", r = 0, i = t.length - 10240; r < i;) n += String.fromCharCode.apply(null, t.subarray(r, r += 10240));
						n += String.fromCharCode.apply(null, r ? t.subarray(r) : t), t = btoa(n);
					} else {
						n === void 0 && (n = 0), me(), n = de[n], r = Array(Math.floor(t.length / 3)), i = n[64] || "";
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
			return e instanceof un ? e = e.size === 0 ? void 0 : e.V(xn) : void 0;
	}
	return e;
}
var Cn, wn;
function Tn(e) {
	return bn(e = e.v, 0 | e[p], Sn);
}
function En(e, t) {
	return Dn(e, t[0], t[1]);
}
function Dn(e, t, n, r = 0) {
	if (e == null) {
		var i = 32;
		n ? (e = [n], i |= 128) : e = [], t && (i = -16760833 & i | (1023 & t) << 14);
	} else {
		if (!Array.isArray(e)) throw Error("narr");
		if (i = 0 | e[p], se && 1 & i) throw Error("rfarr");
		if (2048 & i && !(2 & i) && function() {
			if (se) throw Error("carr");
			je(He, 5);
		}(), 256 & i) throw Error("farr");
		if (64 & i) return (i | r) !== i && m(e, i | r), e;
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
	return m(e, 64 | i | r), e;
}
function On(e, t) {
	if (typeof e != "object") return e;
	if (Array.isArray(e)) {
		var n = 0 | e[p];
		return e.length === 0 && 1 & n ? void 0 : kn(e, n, t);
	}
	if (e != null && e[Ue] === Qe) return jn(e);
	if (e instanceof un) {
		if (2 & (t = e.J)) return e;
		if (!e.size) return;
		if (n = Xe(e.V()), e.K) for (e = 0; e < n.length; e++) {
			let r = n[e], i = r[1];
			i = typeof i != "object" || !i ? void 0 : i != null && i[Ue] === Qe ? jn(i) : Array.isArray(i) ? kn(i, 0 | i[p], !!(32 & t)) : void 0, r[1] = i;
		}
		return n;
	}
	return e instanceof Te ? e : void 0;
}
function kn(e, t, n) {
	return 2 & t || (!n || 4096 & t || 16 & t ? e = Mn(e, t, !1, n && !(16 & t)) : (Ye(e, 34), 4 & t && Object.freeze(e))), e;
}
function An(e, t, n) {
	return e = new e.constructor(t), n && (e.h = et), e.m = et, e;
}
function jn(e) {
	let t = e.v, n = 0 | t[p];
	return $e(e, n) ? e : Ln(e, t, n) ? An(e, t) : Mn(t, n);
}
function Mn(e, t, n, r) {
	return r ??= !!(34 & t), e = bn(e, t, On, r), r = 32, n && (r |= 2), m(e, t = 16769217 & t | r), e;
}
function Nn(e) {
	let t = e.v, n = 0 | t[p];
	return $e(e, n) ? Ln(e, t, n) ? An(e, t, !0) : new e.constructor(Mn(t, n, !1)) : e;
}
function Pn(e) {
	if (e.h !== et) return !1;
	var t = e.v;
	return Ye(t = Mn(t, 0 | t[p]), 2048), e.v = t, e.h = void 0, e.m = void 0, !0;
}
function Fn(e) {
	if (!Pn(e) && $e(e, 0 | e.v[p])) throw Error();
}
function In(e, t) {
	t === void 0 && (t = 0 | e[p]), 32 & t && !(4096 & t) && m(e, 4096 | t);
}
function Ln(e, t, n) {
	return !!(2 & n) || !(!(32 & n) || 4096 & n) && (m(t, 2 | n), e.h = et, !0);
}
var Rn = ft(0), zn = {};
function v(e, t, n, r, i) {
	if ((t = Bn(e.v, t, n, i)) !== null || r && e.m !== et) return t;
}
function Bn(e, t, n, r) {
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
function y(e, t, n, r) {
	Fn(e), b(e = e.v, 0 | e[p], t, n, r);
}
function b(e, t, n, r, i) {
	let a = n + (i ? 0 : -1);
	var o = e.length - 1;
	if (o >= 1 + (i ? 0 : -1) && a >= o) {
		let i = e[o];
		if (typeof i == "object" && i && i.constructor === Object) return i[n] = r, t;
	}
	return a <= o ? (e[a] = r, t) : (r !== void 0 && (n >= (o = (t ??= 0 | e[p]) >> 14 & 1023 || 536870912) ? r != null && (e[o + (i ? 0 : -1)] = { [n]: r }) : e[a] = r), t);
}
function Vn() {
	return rt === void 0 ? 2 : 4;
}
function Hn(e, t, n, r, i) {
	let a = e.v, o = 0 | a[p];
	r = $e(e, o) ? 1 : r, i = !!i || r === 3, r === 2 && Pn(e) && (a = e.v, o = 0 | a[p]);
	let s = (e = Wn(a, t)) === qe ? 7 : 0 | e[p], c = Gn(s, o);
	var l = !(4 & c);
	if (l) {
		4 & c && (e = Mt(e), s = 0, c = sr(c, o), o = b(a, o, t, e));
		let r = 0, i = 0;
		for (; r < e.length; r++) {
			let t = n(e[r]);
			t != null && (e[i++] = t);
		}
		i < r && (e.length = i), n = -513 & (4 | c), c = n &= -1025, c &= -4097;
	}
	return c !== s && (m(e, c), 2 & c && Object.freeze(e)), Un(e, c, a, o, t, r, l, i);
}
function Un(e, t, n, r, i, a, o, s) {
	let c = t;
	return a === 1 || a === 4 && (2 & t || !(16 & t) && 32 & r) ? Kn(t) || ((t |= !e.length || o && !(4096 & t) || 32 & r && !(4096 & t || 16 & t) ? 2 : 256) !== c && m(e, t), Object.freeze(e)) : (a === 2 && Kn(t) && (e = Mt(e), c = 0, t = sr(t, r), r = b(n, r, i, e)), Kn(t) || (s || (t |= 16), t !== c && m(e, t))), 2 & t || !(4096 & t || 16 & t) || In(n, r), e;
}
function Wn(e, t, n) {
	return e = Bn(e, t, n), Array.isArray(e) ? e : qe;
}
function Gn(e, t) {
	return 2 & t && (e |= 2), 1 | e;
}
function Kn(e) {
	return !!(2 & e) && !!(4 & e) || !!(256 & e);
}
function qn(e) {
	return tt(e, !0);
}
function Jn(e) {
	e = Mt(e);
	for (let t = 0; t < e.length; t++) {
		let n = e[t] = Mt(e[t]);
		Array.isArray(n[1]) && (n[1] = Xe(n[1]));
	}
	return Ze(e);
}
function Yn(e, t, n, r) {
	Fn(e), b(e = e.v, 0 | e[p], t, (r === "0" ? Number(n) === 0 : n === r) ? void 0 : n);
}
function Xn(e, t, n) {
	if (2 & t) throw Error();
	let r = ot(t), i = Wn(e, n, r), a = i === qe ? 7 : 0 | i[p], o = Gn(a, t);
	return (2 & o || Kn(o) || 16 & o) && (o === a || Kn(o) || m(i, o), i = Mt(i), a = 0, o = sr(o, t), b(e, t, n, i, r)), o &= -13, o !== a && m(i, o), i;
}
function Zn(e, t) {
	var n = $a;
	return er(Qn(e = e.v), e, void 0, n) === t ? t : -1;
}
function Qn(e) {
	if (Ne) return e[Le] ?? (e[Le] = /* @__PURE__ */ new Map());
	if (Le in e) return e[Le];
	let t = /* @__PURE__ */ new Map();
	return Object.defineProperty(e, Le, { value: t }), t;
}
function $n(e, t, n, r, i) {
	let a = Qn(e), o = er(a, e, t, n, i);
	return o !== r && (o && (t = b(e, t, o, void 0, i)), a.set(n, r)), t;
}
function er(e, t, n, r, i) {
	let a = e.get(r);
	if (a != null) return a;
	a = 0;
	for (let e = 0; e < r.length; e++) {
		let o = r[e];
		Bn(t, o, i) != null && (a !== 0 && (n = b(t, n, a, void 0, i)), a = o);
	}
	return e.set(r, a), a;
}
function tr(e, t, n) {
	let r = 0 | e[p], i = ot(r), a = Bn(e, n, i), o;
	if (a != null && a[Ue] === Qe) {
		if (!$e(a)) return Pn(a), a.v;
		o = a.v;
	} else Array.isArray(a) && (o = a);
	if (o) {
		let e = 0 | o[p];
		2 & e && (o = Mn(o, e));
	}
	return o = En(o, t), o !== a && b(e, r, n, o, i), o;
}
function nr(e, t, n, r, i) {
	let a = !1;
	if ((r = Bn(e, r, i, ((e) => {
		let r = tn(e, n, !1, t);
		return a = r !== e && r != null, r;
	}))) != null) return a && !$e(r) && In(e, t), r;
}
function x(e, t, n, r) {
	let i = e.v, a = 0 | i[p];
	if ((t = nr(i, a, t, n, r)) == null) return t;
	if (a = 0 | i[p], !$e(e, a)) {
		let o = Nn(t);
		o !== t && (Pn(e) && (i = e.v, a = 0 | i[p]), a = b(i, a, n, t = o, r), In(i, a));
	}
	return t;
}
function rr(e, t, n, r, i, a, o, s) {
	var c = $e(e, n);
	a = c ? 1 : a, o = !!o || a === 3, c = s && !c, (a === 2 || c) && Pn(e) && (n = 0 | (t = e.v)[p]);
	var l = (e = Wn(t, i)) === qe ? 7 : 0 | e[p], u = Gn(l, n);
	if (s = !(4 & u)) {
		var d = e, f = n;
		let t = !!(2 & u);
		t && (f |= 2);
		let i = !t, a = !0, o = 0, s = 0;
		for (; o < d.length; o++) {
			let e = tn(d[o], r, !1, f);
			if (e instanceof r) {
				if (!t) {
					let t = $e(e);
					i &&= !t, a &&= t;
				}
				d[s++] = e;
			}
		}
		s < o && (d.length = s), u |= 4, u = a ? -4097 & u : 4096 | u, u = i ? 8 | u : -9 & u;
	}
	if (u !== l && (m(e, u), 2 & u && Object.freeze(e)), c && !(8 & u || !e.length && (a === 1 || a === 4 && (2 & u || !(16 & u) && 32 & n)))) {
		for (Kn(u) && (e = Mt(e), u = sr(u, n), n = b(t, n, i, e)), r = e, c = u, l = 0; l < r.length; l++) (d = r[l]) !== (u = Nn(d)) && (r[l] = u);
		c |= 8, m(e, u = c = r.length ? 4096 | c : -4097 & c);
	}
	return Un(e, u, t, n, i, a, s, o);
}
function ir(e, t, n) {
	let r = e.v;
	return rr(e, r, 0 | r[p], t, n, Vn(), !1, !0);
}
function ar(e) {
	return e ??= void 0, e;
}
function S(e, t, n, r, i) {
	return y(e, n, r = ar(r), i), r && !$e(r) && In(e.v), e;
}
function or(e, t, n, r) {
	t: {
		var i = r = ar(r);
		Fn(e);
		let a = e.v, o = 0 | a[p];
		if (i == null) {
			let e = Qn(a);
			if (er(e, a, o, n) !== t) break t;
			e.set(n, 0);
		} else o = $n(a, o, n, t);
		b(a, o, t, i);
	}
	r && !$e(r) && In(e.v);
}
function sr(e, t) {
	return -273 & (2 & t ? 2 | e : -3 & e);
}
function cr(e, t, n, r) {
	var i = r;
	Fn(e), e = rr(e, r = e.v, 0 | r[p], n, t, 2, !0), i ??= new n(), e.push(i), t = n = e === qe ? 7 : 0 | e[p], (i = $e(i)) ? (n &= -9, e.length === 1 && (n &= -4097)) : n |= 4096, n !== t && m(e, n), i || In(r);
}
function lr(e, t, n) {
	return Gt(v(e, t, void 0, n));
}
function C(e, t) {
	return v(e, t, void 0, void 0, Bt) ?? 0;
}
function ur(e, t, n) {
	if (n != null) {
		if (typeof n != "number" || !It(n)) throw Ae("int32");
		n |= 0;
	}
	y(e, t, n);
}
function w(e, t, n) {
	y(e, t, zt(n));
}
function dr(e, t, n) {
	Yn(e, t, en(n), "");
}
function fr(e, t, n) {
	{
		Fn(e);
		let o = e.v, s = 0 | o[p];
		if (n == null) b(o, s, t);
		else {
			var r = e = n === qe ? 7 : 0 | n[p], i = Kn(e), a = i || Object.isFrozen(n);
			for (i || (e = 0), a ||= (n = Mt(n), r = 0, e = sr(e, s), !1), e |= 5, e |= (4 & e ? 512 & e ? 512 : 1024 & e ? 1024 : 0 : void 0) ?? 1024, i = 0; i < n.length; i++) {
				let t = n[i], o = $t(t);
				Object.is(t, o) || (a &&= (n = Mt(n), r = 0, e = sr(e, s), !1), n[i] = o);
			}
			e !== r && (a && (n = Mt(n), e = sr(e, s)), m(n, e)), b(o, s, t, n);
		}
	}
}
function pr(e, t, n) {
	Fn(e), Hn(e, t, _, 2, !0).push($t(n));
}
var mr = class {
	constructor(e, t, n) {
		if (this.buffer = e, n && !t) throw Error();
		this.g = t;
	}
};
function hr(e, t) {
	if (typeof e == "string") return new mr(be(e), t);
	if (Array.isArray(e)) return new mr(new Uint8Array(e), t);
	if (e.constructor === Uint8Array) return new mr(e, !1);
	if (e.constructor === ArrayBuffer) return e = new Uint8Array(e), new mr(e, !1);
	if (e.constructor === Te) return t = we(e) || /* @__PURE__ */ new Uint8Array(), new mr(t, !0, e);
	if (e instanceof Uint8Array) return e = e.constructor === Uint8Array ? e : new Uint8Array(e.buffer, e.byteOffset, e.byteLength), new mr(e, !1);
	throw Error();
}
function gr(e, t) {
	let n, r = 0, i = 0, a = 0, o = e.h, s = e.g;
	do
		n = o[s++], r |= (127 & n) << a, a += 7;
	while (a < 32 && 128 & n);
	if (a > 32) for (i |= (127 & n) >> 4, a = 3; a < 32 && 128 & n; a += 7) n = o[s++], i |= (127 & n) << a;
	if (Sr(e, s), !(128 & n)) return t(r >>> 0, i >>> 0);
	throw Error();
}
function _r(e) {
	let t = 0, n = e.g, r = n + 10, i = e.h;
	for (; n < r;) {
		let r = i[n++];
		if (t |= r, !(128 & r)) return Sr(e, n), !!(127 & t);
	}
	throw Error();
}
function vr(e) {
	let t = e.h, n = e.g, r = t[n++], i = 127 & r;
	if (128 & r && (r = t[n++], i |= (127 & r) << 7, 128 & r && (r = t[n++], i |= (127 & r) << 14, 128 & r && (r = t[n++], i |= (127 & r) << 21, 128 & r && (r = t[n++], i |= r << 28, 128 & r && 128 & t[n++] && 128 & t[n++] && 128 & t[n++] && 128 & t[n++] && 128 & t[n++]))))) throw Error();
	return Sr(e, n), i;
}
function yr(e) {
	return vr(e) >>> 0;
}
function br(e) {
	var t = e.h;
	let n = e.g;
	var r = t[n], i = t[n + 1];
	let a = t[n + 2];
	return t = t[n + 3], Sr(e, e.g + 4), e = 2 * ((i = (r << 0 | i << 8 | a << 16 | t << 24) >>> 0) >> 31) + 1, r = i >>> 23 & 255, i &= 8388607, r == 255 ? i ? NaN : e * Infinity : r == 0 ? 1401298464324817e-60 * e * i : e * 2 ** (r - 150) * (i + 8388608);
}
function xr(e) {
	return vr(e);
}
function Sr(e, t) {
	if (e.g = t, t > e.l) throw Error();
}
function Cr(e, t) {
	if (t < 0) throw Error();
	let n = e.g;
	if ((t = n + t) > e.l) throw Error();
	return e.g = t, n;
}
function wr(e, t) {
	if (t == 0) return Ce();
	var n = Cr(e, t);
	return e.Y && e.j ? n = e.h.subarray(n, n + t) : (e = e.h, n = n === (t = n + t) ? /* @__PURE__ */ new Uint8Array() : yt ? e.slice(n, t) : new Uint8Array(e.subarray(n, t))), n.length == 0 ? Ce() : new Te(n, Se);
}
var Tr = [];
function Er(e, t, n, r) {
	if (Fr.length) {
		let i = Fr.pop();
		return i.o(r), i.g.init(e, t, n, r), i;
	}
	return new Pr(e, t, n, r);
}
function Dr(e) {
	e.g.clear(), e.l = -1, e.h = -1, Fr.length < 100 && Fr.push(e);
}
function Or(e) {
	var t = e.g;
	if (t.g == t.l) return !1;
	e.m = e.g.g;
	var n = yr(e.g);
	if (t = n >>> 3, !((n &= 7) >= 0 && n <= 5) || t < 1) throw Error();
	return e.l = t, e.h = n, !0;
}
function kr(e) {
	switch (e.h) {
		case 0:
			e.h == 0 ? _r(e.g) : kr(e);
			break;
		case 1:
			Sr(e = e.g, e.g + 8);
			break;
		case 2:
			if (e.h != 2) kr(e);
			else {
				var t = yr(e.g);
				Sr(e = e.g, e.g + t);
			}
			break;
		case 5:
			Sr(e = e.g, e.g + 4);
			break;
		case 3:
			for (t = e.l;;) {
				if (!Or(e)) throw Error();
				if (e.h == 4) {
					if (e.l != t) throw Error();
					break;
				}
				kr(e);
			}
			break;
		default: throw Error();
	}
}
function Ar(e, t, n) {
	let r = e.g.l;
	var i = yr(e.g);
	let a = (i = e.g.g + i) - r;
	if (a <= 0 && (e.g.l = i, n(t, e, void 0, void 0, void 0), a = i - e.g.g), a) throw Error();
	return e.g.g = i, e.g.l = r, t;
}
function jr(e) {
	var t = yr(e.g), n = Cr(e = e.g, t);
	if (e = e.h, ee) {
		var r, i = e;
		(r = f) || (r = f = new TextDecoder("utf-8", { fatal: !0 })), t = n + t, i = n === 0 && t === i.length ? i : i.subarray(n, t);
		try {
			var a = r.decode(i);
		} catch (e) {
			if (d === void 0) {
				try {
					r.decode(new Uint8Array([128]));
				} catch {}
				try {
					r.decode(new Uint8Array([97])), d = !0;
				} catch {
					d = !1;
				}
			}
			throw !d && (f = void 0), e;
		}
	} else {
		t = (a = n) + t, n = [];
		let s, c = null;
		for (; a < t;) {
			var o = e[a++];
			o < 128 ? n.push(o) : o < 224 ? a >= t ? l() : (s = e[a++], o < 194 || (192 & s) != 128 ? (a--, l()) : n.push((31 & o) << 6 | 63 & s)) : o < 240 ? a >= t - 1 ? l() : (s = e[a++], (192 & s) != 128 || o === 224 && s < 160 || o === 237 && s >= 160 || (192 & (r = e[a++])) != 128 ? (a--, l()) : n.push((15 & o) << 12 | (63 & s) << 6 | 63 & r)) : o <= 244 ? a >= t - 2 ? l() : (s = e[a++], (192 & s) != 128 || s - 144 + (o << 28) >> 30 || (192 & (r = e[a++])) != 128 || (192 & (i = e[a++])) != 128 ? (a--, l()) : (o = (7 & o) << 18 | (63 & s) << 12 | (63 & r) << 6 | 63 & i, o -= 65536, n.push(55296 + (o >> 10 & 1023), 56320 + (1023 & o)))) : l(), n.length >= 8192 && (c = u(c, n), n.length = 0);
		}
		a = u(c, n);
	}
	return a;
}
function Mr(e) {
	let t = yr(e.g);
	return wr(e.g, t);
}
function Nr(e, t, n) {
	var r = yr(e.g);
	for (r = e.g.g + r; e.g.g < r;) n.push(t(e.g));
}
var Pr = class {
	constructor(e, t, n, r) {
		if (Tr.length) {
			let i = Tr.pop();
			i.init(e, t, n, r), e = i;
		} else e = new class {
			constructor(e, t, n, r) {
				this.h = null, this.j = !1, this.g = this.l = this.m = 0, this.init(e, t, n, r);
			}
			init(e, t, n, { Y: r = !1, ea: i = !1 } = {}) {
				this.Y = r, this.ea = i, e && (e = hr(e, this.ea), this.h = e.buffer, this.j = e.g, this.m = t || 0, this.l = n === void 0 ? this.h.length : this.m + n, this.g = this.m);
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
}, Fr = [];
function Ir(e) {
	return e ? /^\d+$/.test(e) ? (At(e), new Lr(h, g)) : null : Rr ||= new Lr(0, 0);
}
var Lr = class {
	constructor(e, t) {
		this.h = e >>> 0, this.g = t >>> 0;
	}
}, Rr;
function zr(e) {
	return e ? /^-?\d+$/.test(e) ? (At(e), new Br(h, g)) : null : Vr ||= new Br(0, 0);
}
var Br = class {
	constructor(e, t) {
		this.h = e >>> 0, this.g = t >>> 0;
	}
}, Vr;
function Hr(e, t, n) {
	for (; n > 0 || t > 127;) e.g.push(127 & t | 128), t = (t >>> 7 | n << 25) >>> 0, n >>>= 7;
	e.g.push(t);
}
function Ur(e, t) {
	for (; t > 127;) e.g.push(127 & t | 128), t >>>= 7;
	e.g.push(t);
}
function Wr(e, t) {
	if (t >= 0) Ur(e, t);
	else {
		for (let n = 0; n < 9; n++) e.g.push(127 & t | 128), t >>= 7;
		e.g.push(1);
	}
}
function Gr(e) {
	var t = h;
	e.g.push(t >>> 0 & 255), e.g.push(t >>> 8 & 255), e.g.push(t >>> 16 & 255), e.g.push(t >>> 24 & 255);
}
function Kr(e, t) {
	t.length !== 0 && (e.l.push(t), e.h += t.length);
}
function qr(e, t, n) {
	Ur(e.g, 8 * t + n);
}
function Jr(e, t) {
	return qr(e, t, 2), t = e.g.end(), Kr(e, t), t.push(e.h), t;
}
function Yr(e, t) {
	var n = t.pop();
	for (n = e.h + e.g.length() - n; n > 127;) t.push(127 & n | 128), n >>>= 7, e.h++;
	t.push(n), e.h++;
}
function Xr(e, t, n) {
	qr(e, t, 2), Ur(e.g, n.length), Kr(e, e.g.end()), Kr(e, n);
}
function Zr(e, t, n, r) {
	n != null && (t = Jr(e, t), r(n, e), Yr(e, t));
}
function Qr() {
	let e = class {
		constructor() {
			throw Error();
		}
	};
	return Object.setPrototypeOf(e, e.prototype), e;
}
var $r = Qr(), ei = Qr(), ti = Qr(), ni = Qr(), ri = Qr(), ii = Qr(), ai = Qr(), oi = Qr(), si = Qr(), ci = Qr();
function li(e, t, n) {
	var r = e.v;
	Re && Re in r && (r = r[Re]) && delete r[t.g], t.h ? t.j(e, t.h, t.g, n, t.l) : t.j(e, t.g, n, t.l);
}
var T = class {
	constructor(e, t) {
		this.v = Dn(e, t, void 0, 2048);
	}
	toJSON() {
		return Tn(this);
	}
	j() {
		var e = Wo, t = this.v, n = e.g, r = Re;
		if (Ne && r && t[r]?.[n] != null && je(ze, 3), t = e.g, We && Re && We === void 0 && (r = (n = this.v)[Re]) && (r = r.da)) try {
			r(n, t, vn);
		} catch (e) {
			ie(e);
		}
		return e.h ? e.m(this, e.h, e.g, e.l) : e.m(this, e.g, e.defaultValue, e.l);
	}
	clone() {
		let e = this.v, t = 0 | e[p];
		return Ln(this, e, t) ? An(this, e, !0) : new this.constructor(Mn(e, t, !1));
	}
};
T.prototype[Ue] = Qe, T.prototype.toString = function() {
	return this.v.toString();
};
var ui = class {
	constructor(e, t, n) {
		this.g = e, this.h = t, e = $r, this.l = !!e && n === e || !1;
	}
};
function di(e, t) {
	return new ui(e, t, $r);
}
function fi(e, t, n, r, i) {
	Zr(e, n, wi(t, r), i);
}
var pi = di((function(e, t, n, r, i) {
	return e.h === 2 && (Ar(e, tr(t, r, n), i), !0);
}), fi), mi = di((function(e, t, n, r, i) {
	return e.h === 2 && (Ar(e, tr(t, r, n), i), !0);
}), fi), hi = Symbol(), gi = Symbol(), _i = Symbol(), vi = Symbol(), yi = Symbol(), bi, xi;
function Si(e, t, n, r) {
	var i = r[e];
	if (i) return i;
	(i = {}).qa = r, i.T = function(e) {
		switch (typeof e) {
			case "boolean": return Cn ||= [
				0,
				void 0,
				!0
			];
			case "number": return e > 0 ? void 0 : e === 0 ? wn ||= [0, void 0] : [-e, void 0];
			case "string": return [0, e];
			case "object": return e;
		}
	}(r[0]);
	var a = r[1];
	let o = 1;
	a && a.constructor === Object && (i.ba = a, typeof (a = r[++o]) == "function" && (i.ma = !0, bi ??= a, xi ??= r[o + 1], a = r[o += 2]));
	let s = {};
	for (; a && Array.isArray(a) && a.length && typeof a[0] == "number" && a[0] > 0;) {
		for (var c = 0; c < a.length; c++) s[a[c]] = a;
		a = r[++o];
	}
	for (c = 1; a !== void 0;) {
		let e;
		typeof a == "number" && (c += a, a = r[++o]);
		var l = void 0;
		if (a instanceof ui ? e = a : (e = pi, o--), e?.l) {
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
function Ci(e) {
	return Array.isArray(e) ? e[0] instanceof ui ? e : [mi, e] : [e, void 0];
}
function wi(e, t) {
	return e instanceof T ? e.v : Array.isArray(e) ? En(e, t) : void 0;
}
function Ti(e, t, n, r) {
	let i = n.g;
	e[t] = r ? (e, t, n) => i(e, t, n, r) : i;
}
function Ei(e, t, n, r, i) {
	let a = n.g, o, s;
	e[t] = (e, t, n) => a(e, t, n, s ||= Si(gi, Ti, Ei, r).T, o ||= Di(r), i);
}
function Di(e) {
	let t = e[_i];
	if (t != null) return t;
	let n = Si(gi, Ti, Ei, e);
	return t = n.ma ? (e, t) => bi(e, t, n) : (e, t) => {
		for (; Or(t) && t.h != 4;) {
			var r = t.l, i = n[r];
			if (i == null) {
				var a = n.ba;
				(a &&= a[r]) && (a = ki(a)) != null && (i = n[r] = a);
			}
			if (i == null || !i(t, e, r)) {
				if (i = (a = t).m, kr(a), a.ha) var o = void 0;
				else o = a.g.g - i, a.g.g = i, o = wr(a.g, o);
				i = void 0, a = e, o && ((i = a[Re] ?? (a[Re] = new _n()))[r] ?? (i[r] = [])).push(o);
			}
		}
		return (e = hn(e)) && (e.da = n.qa[yi]), !0;
	}, e[_i] = t, e[yi] = Oi.bind(e), t;
}
function Oi(e, t, n, r) {
	var i = this[gi];
	let a = this[_i], o = En(void 0, i.T), s = hn(e);
	if (s) {
		var c = !1, l = i.ba;
		if (l) {
			if (i = (t, n, i) => {
				if (i.length !== 0) if (l[n]) for (let e of i) {
					t = Er(e);
					try {
						c = !0, a(o, t);
					} finally {
						Dr(t);
					}
				}
				else r?.(e, n, i);
			}, t == null) gn(s, i);
			else if (s != null) {
				let e = s[t];
				e && i(s, t, e);
			}
			if (c) {
				let r = 0 | e[p];
				if (2 & r && 2048 & r && !n?.Ka) throw Error();
				let i = ot(r), a = (t, a) => {
					if (Bn(e, t, i) != null) {
						if (n?.Qa === 1) return;
						throw Error();
					}
					a != null && (r = b(e, r, t, a, i)), delete s[t];
				};
				t == null ? it(o, 0 | o[p], ((e, t) => {
					a(e, t);
				})) : a(t, Bn(o, t, i));
			}
		}
	}
}
function ki(e) {
	let t = (e = Ci(e))[0].g;
	if (e = e[1]) {
		let n = Di(e), r = Si(gi, Ti, Ei, e).T;
		return (e, i, a) => t(e, i, a, r, n);
	}
	return t;
}
function Ai(e, t, n) {
	e[t] = n.h;
}
function ji(e, t, n, r) {
	let i, a, o = n.h;
	e[t] = (e, t, n) => o(e, t, n, a ||= Si(hi, Ai, ji, r).T, i ||= Mi(r));
}
function Mi(e) {
	let t = e[vi];
	if (!t) {
		let n = Si(hi, Ai, ji, e);
		t = (e, t) => Ni(e, t, n), e[vi] = t;
	}
	return t;
}
function Ni(e, t, n) {
	it(e, 0 | e[p], ((e, r) => {
		if (r != null) {
			var i = function(e, t) {
				var n = e[t];
				if (n) return n;
				if ((n = e.ba) && (n = n[t])) {
					var r = (n = Ci(n))[0].h;
					if (n = n[1]) {
						let t = Mi(n), i = Si(hi, Ai, ji, n).T;
						n = e.ma ? xi(i, t) : (e, n, a) => r(e, n, a, i, t);
					} else n = r;
					return e[t] = n;
				}
			}(n, e);
			i ? i(t, r, e) : e < 500 || je(Ve, 3);
		}
	})), (e = hn(e)) && gn(e, ((e, n, r) => {
		for (Kr(t, t.g.end()), e = 0; e < r.length; e++) Kr(t, we(r[e]) || /* @__PURE__ */ new Uint8Array());
	}));
}
var Pi = ft(0);
function Fi(e, t) {
	if (Array.isArray(t)) {
		var n = 0 | t[p];
		if (4 & n) return t;
		for (var r = 0, i = 0; r < t.length; r++) {
			let n = e(t[r]);
			n != null && (t[i++] = n);
		}
		return i < r && (t.length = i), (e = -1537 & (5 | n)) !== n && m(t, e), 2 & e && Object.freeze(t), t;
	}
}
function E(e, t, n) {
	return new ui(e, t, n);
}
function Ii(e, t, n) {
	return new ui(e, t, n);
}
function D(e, t, n) {
	b(e, 0 | e[p], t, n, ot(0 | e[p]));
}
var Li = di((function(e, t, n, r, i) {
	if (e.h !== 2) return !1;
	if (e = Mt(e = Ar(e, En([void 0, void 0], r), i)), i = ot(r = 0 | t[p]), 2 & r) throw Error();
	let a = Bn(t, n, i);
	if (a instanceof un) 2 & a.J ? (a = a.V(), a.push(e), b(t, r, n, a, i)) : a.Ma(e);
	else if (Array.isArray(a)) {
		var o = 0 | a[p];
		8192 & o || m(a, o |= 8192), 2 & o && (a = Jn(a), b(t, r, n, a, i)), a.push(e);
	} else b(t, r, n, Ze([e]), i);
	return !0;
}), (function(e, t, n, r, i) {
	if (t instanceof un) t.forEach(((t, a) => {
		Zr(e, n, En([a, t], r), i);
	}));
	else if (Array.isArray(t)) {
		for (let a = 0; a < t.length; a++) {
			let o = t[a];
			Array.isArray(o) && Zr(e, n, En(o, r), i);
		}
		Ze(t);
	}
}));
function Ri(e, t, n) {
	(t = Bt(t)) != null && (qr(e, n, 5), e = e.g, Ct(t), Gr(e));
}
function zi(e, t, n) {
	if (t = function(e) {
		if (e == null) return e;
		let t = typeof e;
		if (t === "bigint") return String(Nt(64, e));
		if (Wt(e)) {
			if (t === "string") return Yt(e);
			if (t === "number") return Jt(e);
		}
	}(t), t != null && (typeof t == "string" && zr(t), t != null)) switch (qr(e, n, 0), typeof t) {
		case "number":
			e = e.g, St(t), Hr(e, h, g);
			break;
		case "bigint":
			n = BigInt.asUintN(64, t), n = new Br(Number(n & BigInt(4294967295)), Number(n >> BigInt(32))), Hr(e.g, n.h, n.g);
			break;
		default: n = zr(t), Hr(e.g, n.h, n.g);
	}
}
function Bi(e, t, n) {
	(t = Gt(t)) != null && t != null && (qr(e, n, 0), Wr(e.g, t));
}
function Vi(e, t, n) {
	(t = Ht(t)) != null && (qr(e, n, 0), e.g.g.push(+!!t));
}
function Hi(e, t, n) {
	(t = _(t)) != null && Xr(e, n, re(t));
}
function Ui(e, t, n, r, i) {
	Zr(e, n, wi(t, r), i);
}
function Wi(e, t, n) {
	(t = t == null || typeof t == "string" || t instanceof Te ? t : void 0) != null && Xr(e, n, hr(t, !0).buffer);
}
function Gi(e, t, n) {
	(t = Kt(t)) != null && t != null && (qr(e, n, 0), Ur(e.g, t));
}
function Ki(e, t, n) {
	return (e.h === 5 || e.h === 2) && (t = Xn(t, 0 | t[p], n), e.h == 2 ? Nr(e, br, t) : t.push(br(e.g)), !0);
}
var O = E((function(e, t, n) {
	return e.h === 5 && (D(t, n, br(e.g)), !0);
}), Ri, oi), qi = Ii(Ki, (function(e, t, n) {
	if ((t = Fi(Bt, t)) != null) for (let o = 0; o < t.length; o++) {
		var r = e, i = n, a = t[o];
		a != null && (qr(r, i, 5), r = r.g, Ct(a), Gr(r));
	}
}), oi), Ji = Ii(Ki, (function(e, t, n) {
	if ((t = Fi(Bt, t)) != null && t.length) {
		qr(e, n, 2), Ur(e.g, 4 * t.length);
		for (let r = 0; r < t.length; r++) n = e.g, Ct(t[r]), Gr(n);
	}
}), oi), Yi = E((function(e, t, n) {
	return e.h === 5 && (D(t, n, (e = br(e.g)) === 0 ? void 0 : e), !0);
}), Ri, oi), Xi = E((function(e, t, n) {
	return e.h === 0 ? (D(t, n, gr(e.g, Et)), e = !0) : e = !1, e;
}), zi, ii), Zi = E((function(e, t, n) {
	return e.h === 0 ? (D(t, n, (e = gr(e.g, Et)) === Pi ? void 0 : e), t = !0) : t = !1, t;
}), zi, ii), Qi = E((function(e, t, n) {
	return e.h === 0 ? (D(t, n, gr(e.g, Tt)), e = !0) : e = !1, e;
}), (function(e, t, n) {
	if (t = function(e) {
		if (e == null) return e;
		var t = typeof e;
		if (t === "bigint") return String(Pt(64, e));
		if (Wt(e)) {
			if (t === "string") return t = Lt(Number(e)), Ft(t) && t >= 0 ? e = String(t) : ((t = e.indexOf(".")) !== -1 && (e = e.substring(0, t)), (t = e[0] !== "-" && ((t = e.length) < 20 || t === 20 && e <= "18446744073709551615")) || (At(e), e = Dt(h, g))), e;
			if (t === "number") return (e = Lt(e)) >= 0 && Ft(e) || (St(e), e = wt(h, g)), e;
		}
	}(t), t != null && (typeof t == "string" && Ir(t), t != null)) switch (qr(e, n, 0), typeof t) {
		case "number":
			e = e.g, St(t), Hr(e, h, g);
			break;
		case "bigint":
			n = BigInt.asUintN(64, t), n = new Lr(Number(n & BigInt(4294967295)), Number(n >> BigInt(32))), Hr(e.g, n.h, n.g);
			break;
		default: n = Ir(t), Hr(e.g, n.h, n.g);
	}
}), ai), k = E((function(e, t, n) {
	return e.h === 0 && (D(t, n, vr(e.g)), !0);
}), Bi, ni), $i = Ii((function(e, t, n) {
	return (e.h === 0 || e.h === 2) && (t = Xn(t, 0 | t[p], n), e.h == 2 ? Nr(e, vr, t) : t.push(vr(e.g)), !0);
}), (function(e, t, n) {
	if ((t = Fi(Gt, t)) != null && t.length) {
		n = Jr(e, n);
		for (let n = 0; n < t.length; n++) Wr(e.g, t[n]);
		Yr(e, n);
	}
}), ni), ea = E((function(e, t, n) {
	return e.h === 0 && (D(t, n, (e = vr(e.g)) === 0 ? void 0 : e), !0);
}), Bi, ni), A = E((function(e, t, n) {
	return e.h === 0 && (D(t, n, _r(e.g)), !0);
}), Vi, ei), ta = E((function(e, t, n) {
	return e.h === 0 && (D(t, n, !1 === (e = _r(e.g)) ? void 0 : e), !0);
}), Vi, ei), j = Ii((function(e, t, n) {
	return e.h === 2 && (e = jr(e), Xn(t, 0 | t[p], n).push(e), !0);
}), (function(e, t, n) {
	if ((t = Fi(_, t)) != null) for (let o = 0; o < t.length; o++) {
		var r = e, i = n, a = t[o];
		a != null && Xr(r, i, re(a));
	}
}), ti), na = E((function(e, t, n) {
	return e.h === 2 && (D(t, n, (e = jr(e)) === "" ? void 0 : e), !0);
}), Hi, ti), M = E((function(e, t, n) {
	return e.h === 2 && (D(t, n, jr(e)), !0);
}), Hi, ti), N = function(e, t, n = $r) {
	return new ui(e, t, n);
}((function(e, t, n, r, i) {
	return e.h === 2 && (r = En(void 0, r), Xn(t, 0 | t[p], n).push(r), Ar(e, r, i), !0);
}), (function(e, t, n, r, i) {
	if (Array.isArray(t)) {
		for (let a = 0; a < t.length; a++) Ui(e, t[a], n, r, i);
		1 & (e = 0 | t[p]) || m(t, 1 | e);
	}
})), P = di((function(e, t, n, r, i, a) {
	if (e.h !== 2) return !1;
	let o = 0 | t[p];
	return $n(t, o, a, n, ot(o)), Ar(e, t = tr(t, r, n), i), !0;
}), Ui), ra = E((function(e, t, n) {
	return e.h === 2 && (D(t, n, Mr(e)), !0);
}), Wi, si), ia = Ii((function(e, t, n) {
	return (e.h === 0 || e.h === 2) && (t = Xn(t, 0 | t[p], n), e.h == 2 ? Nr(e, yr, t) : t.push(yr(e.g)), !0);
}), (function(e, t, n) {
	if ((t = Fi(Kt, t)) != null) for (let o = 0; o < t.length; o++) {
		var r = e, i = n, a = t[o];
		a != null && (qr(r, i, 0), Ur(r.g, a));
	}
}), ri), aa = E((function(e, t, n) {
	return e.h === 0 && (D(t, n, (e = yr(e.g)) === 0 ? void 0 : e), !0);
}), Gi, ri), F = E((function(e, t, n) {
	return e.h === 0 && (D(t, n, vr(e.g)), !0);
}), (function(e, t, n) {
	(t = Gt(t)) != null && (t = parseInt(t, 10), qr(e, n, 0), Wr(e.g, t));
}), ci), oa = class {
	constructor(e, t) {
		var n = wa;
		this.g = e, this.h = t, this.m = x, this.j = S, this.defaultValue = void 0, this.l = n.Oa == null ? void 0 : at;
	}
	register() {
		ue(this);
	}
};
function sa(e, t) {
	return new oa(e, t);
}
function ca(e, t) {
	return (n, r) => {
		{
			let a = { ea: !0 };
			r && Object.assign(a, r), n = Er(n, void 0, void 0, a);
			try {
				let r = new e(), a = r.v;
				Di(t)(a, n);
				var i = r;
			} finally {
				Dr(n);
			}
		}
		return i;
	};
}
function la(e) {
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
		Ni(this.v, t, Si(hi, Ai, ji, e)), Kr(t, t.g.end());
		let n = new Uint8Array(t.h), r = t.l, i = r.length, a = 0;
		for (let e = 0; e < i; e++) {
			let t = r[e];
			n.set(t, a), a += t.length;
		}
		return t.l = [n], n;
	};
}
var ua = class extends T {
	constructor(e) {
		super(e);
	}
}, da = [
	0,
	na,
	E((function(e, t, n) {
		return e.h === 2 && (D(t, n, (e = Mr(e)) === Ce() ? void 0 : e), !0);
	}), (function(e, t, n) {
		if (t != null) {
			if (t instanceof T) {
				let r = t.Ra;
				r ? (t = r(t), t != null && Xr(e, n, hr(t, !0).buffer)) : je(Ve, 3);
				return;
			}
			if (Array.isArray(t)) return void je(Ve, 3);
		}
		Wi(e, t, n);
	}), si)
], fa, pa = globalThis.trustedTypes;
function ma(e) {
	var t;
	return fa === void 0 && (fa = function() {
		let e = null;
		if (!pa) return e;
		try {
			let t = (e) => e;
			e = pa.createPolicy("goog#html", {
				createHTML: t,
				createScript: t,
				createScriptURL: t
			});
		} catch {}
		return e;
	}()), e = (t = fa) ? t.createScriptURL(e) : e, new class {
		constructor(e) {
			this.g = e;
		}
		toString() {
			return this.g + "";
		}
	}(e);
}
function ha(e, ...t) {
	if (t.length === 0) return ma(e[0]);
	let n = e[0];
	for (let r = 0; r < t.length; r++) n += encodeURIComponent(t[r]) + e[r + 1];
	return ma(n);
}
var ga = [
	0,
	k,
	F,
	A,
	-1,
	$i,
	F,
	-1,
	A
], _a = class extends T {
	constructor(e) {
		super(e);
	}
}, va = [
	0,
	A,
	M,
	A,
	F,
	-1,
	Ii((function(e, t, n) {
		return (e.h === 0 || e.h === 2) && (t = Xn(t, 0 | t[p], n), e.h == 2 ? Nr(e, xr, t) : t.push(vr(e.g)), !0);
	}), (function(e, t, n) {
		if ((t = Fi(Gt, t)) != null && t.length) {
			n = Jr(e, n);
			for (let n = 0; n < t.length; n++) Wr(e.g, t[n]);
			Yr(e, n);
		}
	}), ci),
	M,
	-1,
	[
		0,
		A,
		-1
	],
	F,
	A,
	-1
], ya = [
	0,
	3,
	A,
	-1,
	2,
	[
		0,
		[2],
		k,
		P,
		[0, E((function(e, t, n) {
			return e.h === 0 && (D(t, n, yr(e.g)), !0);
		}), Gi, ri)]
	],
	[
		0,
		F,
		A,
		F,
		A,
		F,
		A,
		M,
		-1
	],
	[
		0,
		[3, 4],
		M,
		-1,
		P,
		[0, k],
		P,
		[0, F]
	],
	[0]
], ba = [
	0,
	M,
	-2
], xa = class extends T {
	constructor(e) {
		super(e);
	}
}, Sa = [0], Ca = [
	0,
	k,
	A,
	1,
	A,
	-4
], wa = class extends T {
	constructor(e) {
		super(e, 2);
	}
}, I = {};
I[336783863] = [
	0,
	M,
	A,
	-1,
	k,
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
		P,
		Sa,
		P,
		va,
		P,
		ba,
		P,
		Ca,
		P,
		ga,
		P,
		[
			0,
			M,
			-2
		],
		P,
		[
			0,
			M,
			F
		],
		P,
		ya,
		P,
		[
			0,
			F,
			-1,
			A
		]
	],
	[0, M],
	A,
	[
		0,
		[1, 3],
		[2, 4],
		P,
		[0, $i],
		-1,
		P,
		[0, j],
		-1,
		N,
		[
			0,
			M,
			-1
		]
	],
	M
];
var Ta = [
	0,
	Zi,
	-1,
	ta,
	-3,
	Zi,
	$i,
	na,
	ea,
	Zi,
	-1,
	ta,
	ea,
	ta,
	-2,
	na
];
function L(e, t) {
	pr(e, 3, t);
}
function R(e, t) {
	pr(e, 4, t);
}
var z = class extends T {
	constructor(e) {
		super(e, 500);
	}
	o(e) {
		return S(this, 0, 7, e);
	}
}, Ea = [-1, {}], Da = [
	0,
	M,
	1,
	Ea
], Oa = [
	0,
	M,
	j,
	Ea
];
function ka(e, t) {
	cr(e, 1, z, t);
}
function B(e, t) {
	pr(e, 10, t);
}
function V(e, t) {
	pr(e, 15, t);
}
var Aa = class extends T {
	constructor(e) {
		super(e, 500);
	}
	o(e) {
		return S(this, 0, 1001, e);
	}
}, ja = [
	-500,
	N,
	[
		-500,
		na,
		-1,
		j,
		-3,
		[
			-2,
			I,
			A
		],
		N,
		da,
		ea,
		-1,
		Da,
		Oa,
		N,
		[
			0,
			na,
			ta
		],
		na,
		Ta,
		ea,
		j,
		987,
		j
	],
	4,
	N,
	[
		-500,
		M,
		-1,
		[-1, {}],
		998,
		M
	],
	N,
	[
		-500,
		M,
		j,
		-1,
		[
			-2,
			{},
			A
		],
		997,
		j,
		-1
	],
	ea,
	N,
	[
		-500,
		M,
		j,
		Ea,
		998,
		j
	],
	j,
	ea,
	Da,
	Oa,
	N,
	[
		0,
		na,
		-1,
		Ea
	],
	j,
	-2,
	Ta,
	na,
	-1,
	ta,
	[
		0,
		ta,
		aa
	],
	978,
	Ea,
	N,
	da
];
Aa.prototype.g = la(ja);
var Ma = ca(Aa, ja), Na = class extends T {
	constructor(e) {
		super(e);
	}
}, Pa = class extends T {
	constructor(e) {
		super(e);
	}
	g() {
		return ir(this, Na, 1);
	}
}, Fa = [
	0,
	N,
	[
		0,
		k,
		O,
		M,
		-1
	]
], Ia = ca(Pa, Fa), La = class extends T {
	constructor(e) {
		super(e);
	}
}, Ra = class extends T {
	constructor(e) {
		super(e);
	}
}, za = class extends T {
	constructor(e) {
		super(e);
	}
	l() {
		return x(this, La, 2);
	}
	g() {
		return ir(this, Ra, 5);
	}
}, Ba = ca(class extends T {
	constructor(e) {
		super(e);
	}
}, [
	0,
	j,
	$i,
	Ji,
	[
		0,
		F,
		[
			0,
			k,
			-3
		],
		[
			0,
			O,
			-3
		],
		[
			0,
			k,
			-1,
			[
				0,
				N,
				[
					0,
					k,
					-2
				]
			]
		],
		N,
		[
			0,
			O,
			-1,
			M,
			O
		]
	],
	M,
	-1,
	Xi,
	N,
	[
		0,
		k,
		O
	],
	j,
	Xi
]), Va = class extends T {
	constructor(e) {
		super(e);
	}
}, Ha = ca(class extends T {
	constructor(e) {
		super(e);
	}
}, [
	0,
	N,
	[
		0,
		O,
		-4
	]
]), Ua = class extends T {
	constructor(e) {
		super(e);
	}
}, Wa = ca(class extends T {
	constructor(e) {
		super(e);
	}
}, [
	0,
	N,
	[
		0,
		O,
		-4
	]
]), Ga = class extends T {
	constructor(e) {
		super(e);
	}
}, Ka = [
	0,
	k,
	-1,
	Ji,
	F
], qa = class extends T {
	constructor(e) {
		super(e);
	}
};
qa.prototype.g = la([
	0,
	O,
	-4,
	Xi
]);
var Ja = class extends T {
	constructor(e) {
		super(e);
	}
}, Ya = ca(class extends T {
	constructor(e) {
		super(e);
	}
}, [
	0,
	N,
	[
		0,
		1,
		k,
		M,
		Fa
	],
	Xi
]), Xa = class extends T {
	constructor(e) {
		super(e);
	}
}, Za = class extends T {
	constructor(e) {
		super(e);
	}
	na() {
		return v(this, 1, void 0, void 0, qn) ?? Ce();
	}
}, Qa = class extends T {
	constructor(e) {
		super(e);
	}
}, $a = [1, 2], eo = ca(class extends T {
	constructor(e) {
		super(e);
	}
}, [
	0,
	N,
	[
		0,
		$a,
		P,
		[0, Ji],
		P,
		[0, ra],
		k,
		M
	],
	Xi
]), to = class extends T {
	constructor(e) {
		super(e);
	}
}, no = [
	0,
	M,
	k,
	O,
	j,
	-1
], ro = class extends T {
	constructor(e) {
		super(e);
	}
}, io = [
	0,
	A,
	-1
], ao = class extends T {
	constructor(e) {
		super(e);
	}
}, oo = [
	1,
	2,
	3,
	4,
	5,
	6
], so = class extends T {
	constructor(e) {
		super(e);
	}
	g() {
		return v(this, 1, void 0, void 0, qn) != null;
	}
	l() {
		return _(v(this, 2)) != null;
	}
}, H = class extends T {
	constructor(e) {
		super(e);
	}
	g() {
		return Ht(v(this, 2)) ?? !1;
	}
}, co = [
	0,
	ra,
	M,
	[
		0,
		k,
		Xi,
		-1
	],
	[
		0,
		Qi,
		Xi
	]
], U = [
	0,
	co,
	A,
	[
		0,
		oo,
		P,
		Ca,
		P,
		va,
		P,
		ga,
		P,
		Sa,
		P,
		ba,
		P,
		ya
	],
	F
], lo = class extends T {
	constructor(e) {
		super(e);
	}
}, uo = [
	0,
	U,
	O,
	-1,
	k
], fo = sa(502141897, lo);
I[502141897] = uo;
var po = ca(class extends T {
	constructor(e) {
		super(e);
	}
}, [
	0,
	[
		0,
		F,
		-1,
		qi,
		ia
	],
	Ka
]), mo = class extends T {
	constructor(e) {
		super(e);
	}
}, ho = class extends T {
	constructor(e) {
		super(e);
	}
}, go = [
	0,
	U,
	O,
	[0, U],
	A
], _o = sa(508968150, ho);
I[508968150] = [
	0,
	U,
	uo,
	go,
	O,
	[0, [0, co]]
], I[508968149] = go;
var vo = class extends T {
	constructor(e) {
		super(e);
	}
	l() {
		return x(this, to, 2);
	}
	g() {
		y(this, 2);
	}
}, yo = [
	0,
	U,
	no
];
I[478825465] = yo;
var bo = class extends T {
	constructor(e) {
		super(e);
	}
}, xo = class extends T {
	constructor(e) {
		super(e);
	}
}, So = class extends T {
	constructor(e) {
		super(e);
	}
}, Co = class extends T {
	constructor(e) {
		super(e);
	}
}, wo = class extends T {
	constructor(e) {
		super(e);
	}
}, To = [
	0,
	U,
	[0, U],
	yo,
	-1
], Eo = [
	0,
	U,
	O,
	k
], Do = [
	0,
	U,
	O
], Oo = [
	0,
	U,
	Eo,
	Do,
	O
], ko = sa(479097054, wo);
I[479097054] = [
	0,
	U,
	Oo,
	To
], I[463370452] = To, I[464864288] = Eo;
var Ao = sa(462713202, Co);
I[462713202] = Oo, I[474472470] = Do;
var jo = class extends T {
	constructor(e) {
		super(e);
	}
}, Mo = class extends T {
	constructor(e) {
		super(e);
	}
}, No = class extends T {
	constructor(e) {
		super(e);
	}
}, Po = class extends T {
	constructor(e) {
		super(e);
	}
}, Fo = [
	0,
	U,
	O,
	-1,
	k
], Io = [
	0,
	U,
	O,
	A
];
Po.prototype.g = la([
	0,
	U,
	Do,
	[0, U],
	uo,
	go,
	Fo,
	Io
]);
var Lo = class extends T {
	constructor(e) {
		super(e);
	}
}, Ro = sa(456383383, Lo);
I[456383383] = [
	0,
	U,
	no
];
var zo = class extends T {
	constructor(e) {
		super(e);
	}
}, Bo = sa(476348187, zo);
I[476348187] = [
	0,
	U,
	io
];
var Vo = class extends T {
	constructor(e) {
		super(e);
	}
}, Ho = class extends T {
	constructor(e) {
		super(e);
	}
}, Uo = [
	0,
	F,
	-1
], Wo = sa(458105876, class extends T {
	constructor(e) {
		super(e);
	}
	g() {
		let e;
		var t = this.v;
		let n = 0 | t[p];
		return e = $e(this, n), t = function(e, t, n, r) {
			var i = Ho;
			!r && Pn(e) && (n = 0 | (t = e.v)[p]);
			var a = Bn(t, 2);
			if (e = !1, a == null) {
				if (r) return mn();
				a = [];
			} else if (a.constructor === un) {
				if (!(2 & a.J) || r) return a;
				a = a.V();
			} else Array.isArray(a) ? e = !!(2 & (0 | a[p])) : a = [];
			if (r) {
				if (!a.length) return mn();
				e || (e = !0, Xe(a));
			} else e && (e = !1, Ze(a), a = Jn(a));
			return !e && 32 & n && Ye(a, 32), n = b(t, n, 2, r = new un(a, i, nn, void 0)), e || In(t, n), r;
		}(this, t, n, e), !e && Ho && (t.ra = !0), t;
	}
});
I[458105876] = [
	0,
	Uo,
	Li,
	[
		!0,
		Xi,
		[
			0,
			M,
			-1,
			j
		]
	],
	[
		0,
		$i,
		A,
		F
	]
];
var Go = class extends T {
	constructor(e) {
		super(e);
	}
}, Ko = sa(458105758, Go);
I[458105758] = [
	0,
	U,
	M,
	Uo
];
var qo = class extends T {
	constructor(e) {
		super(e);
	}
}, Jo = [
	0,
	Yi,
	-1,
	ta
], Yo = class extends T {
	constructor(e) {
		super(e);
	}
}, Xo = class extends T {
	constructor(e) {
		super(e);
	}
}, Zo = [1, 2];
Xo.prototype.g = la([
	0,
	Zo,
	P,
	Jo,
	P,
	[
		0,
		N,
		Jo
	]
]);
var Qo = class extends T {
	constructor(e) {
		super(e);
	}
}, $o = sa(443442058, Qo);
I[443442058] = [
	0,
	U,
	M,
	k,
	O,
	j,
	-1,
	A,
	O
], I[514774813] = Fo;
var es = class extends T {
	constructor(e) {
		super(e);
	}
}, ts = sa(516587230, es);
function ns(e, t) {
	return t = t ? t.clone() : new to(), e.displayNamesLocale === void 0 ? e.displayNamesLocale === void 0 && y(t, 1) : y(t, 1, en(e.displayNamesLocale)), e.maxResults === void 0 ? "maxResults" in e && y(t, 2) : ur(t, 2, e.maxResults), e.scoreThreshold === void 0 ? "scoreThreshold" in e && y(t, 3) : w(t, 3, e.scoreThreshold), e.categoryAllowlist === void 0 ? "categoryAllowlist" in e && y(t, 4) : fr(t, 4, e.categoryAllowlist), e.categoryDenylist === void 0 ? "categoryDenylist" in e && y(t, 5) : fr(t, 5, e.categoryDenylist), t;
}
function rs(e) {
	let t = Number(e);
	return Number.isSafeInteger(t) ? t : String(e);
}
function is(e, t = -1, n = "") {
	return {
		categories: e.map(((e) => ({
			index: lr(e, 1) ?? 0 ?? -1,
			score: C(e, 2) ?? 0,
			categoryName: _(v(e, 3)) ?? "" ?? "",
			displayName: _(v(e, 4)) ?? "" ?? ""
		}))),
		headIndex: t,
		headName: n
	};
}
function as(e) {
	let t = { classifications: ir(e, Ja, 1).map(((e) => is(x(e, Pa, 4)?.g() ?? [], lr(e, 2) ?? 0, _(v(e, 3)) ?? ""))) };
	return function(e) {
		return e == null ? e : typeof e == "bigint" ? (pt(e) ? e = Number(e) : (e = Nt(64, e), e = pt(e) ? Number(e) : String(e)), e) : Wt(e) ? typeof e == "number" ? Jt(e) : Yt(e) : void 0;
	}(v(e, 2, void 0, void 0, Qt)) != null && (t.timestampMs = rs(v(e, 2, void 0, void 0, Qt) ?? Rn)), t;
}
function os(e) {
	var t = Hn(e, 3, Bt, Vn()), n = Hn(e, 2, Gt, Vn()), r = Hn(e, 1, _, Vn()), i = Hn(e, 9, _, Vn());
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
	if ((t = x(e, za, 4)?.l()) && (a.boundingBox = {
		originX: lr(t, 1, zn) ?? 0,
		originY: lr(t, 2, zn) ?? 0,
		width: lr(t, 3, zn) ?? 0,
		height: lr(t, 4, zn) ?? 0,
		angle: 0
	}), x(e, za, 4)?.g().length) for (let t of x(e, za, 4).g()) a.keypoints.push({
		x: v(t, 1, void 0, zn, Bt) ?? 0,
		y: v(t, 2, void 0, zn, Bt) ?? 0,
		score: v(t, 4, void 0, zn, Bt) ?? 0,
		label: _(v(t, 3, void 0, zn)) ?? ""
	});
	return a;
}
function ss(e) {
	let t = [];
	for (let n of ir(e, Ua, 1)) t.push({
		x: C(n, 1) ?? 0,
		y: C(n, 2) ?? 0,
		z: C(n, 3) ?? 0,
		visibility: C(n, 4) ?? 0
	});
	return t;
}
function cs(e) {
	let t = [];
	for (let n of ir(e, Va, 1)) t.push({
		x: C(n, 1) ?? 0,
		y: C(n, 2) ?? 0,
		z: C(n, 3) ?? 0,
		visibility: C(n, 4) ?? 0
	});
	return t;
}
function ls(e) {
	return Array.from(e, ((e) => e > 127 ? e - 256 : e));
}
function us(e, t) {
	if (e.length !== t.length) throw Error(`Cannot compute cosine similarity between embeddings of different sizes (${e.length} vs. ${t.length}).`);
	let n = 0, r = 0, i = 0;
	for (let a = 0; a < e.length; a++) n += e[a] * t[a], r += e[a] * e[a], i += t[a] * t[a];
	if (r <= 0 || i <= 0) throw Error("Cannot compute cosine similarity on embedding with 0 norm.");
	return n / Math.sqrt(r * i);
}
var ds;
I[516587230] = [
	0,
	U,
	Fo,
	Io,
	O
], I[518928384] = Io;
var fs = new Uint8Array([
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
async function ps(e) {
	if (e) return !0;
	if (ds === void 0) try {
		await WebAssembly.instantiate(fs), ds = !0;
	} catch {
		ds = !1;
	}
	return ds;
}
async function ms(e, t, n) {
	return {
		wasmLoaderPath: `${t}/${e}_${n = `wasm${n ? "_module" : ""}${await ps(n) ? "" : "_nosimd"}_internal`}.js`,
		wasmBinaryPath: `${t}/${e}_${n}.wasm`
	};
}
var hs = class {};
function gs() {
	var e = navigator;
	return typeof OffscreenCanvas < "u" && (!function(e = navigator) {
		return (e = e.userAgent).includes("Safari") && !e.includes("Chrome");
	}(e) || !!((e = e.userAgent.match(/Version\/([\d]+).*Safari/)) && e.length >= 1 && Number(e[1]) >= 17));
}
async function _s(e) {
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
function vs(e) {
	return e.videoWidth === void 0 ? e.naturalWidth === void 0 ? e.displayWidth === void 0 ? [e.width, e.height] : [e.displayWidth, e.displayHeight] : [e.naturalWidth, e.naturalHeight] : [e.videoWidth, e.videoHeight];
}
function W(e, t, n) {
	e.m || console.error("No wasm multistream support detected: ensure dependency inclusion of :gl_graph_runner_internal_multi_input target"), n(t = e.i.stringToNewUTF8(t)), e.i._free(t);
}
function ys(e, t, n) {
	if (!e.i.canvas) throw Error("No OpenGL canvas configured.");
	if (n ? e.i._bindTextureToStream(n) : e.i._bindTextureToCanvas(), !(n = e.i.canvas.getContext("webgl2") || e.i.canvas.getContext("webgl"))) throw Error("Failed to obtain WebGL context from the provided canvas. `getContext()` should only be invoked with `webgl` or `webgl2`.");
	e.i.gpuOriginForWebTexturesIsBottomLeft && n.pixelStorei(n.UNPACK_FLIP_Y_WEBGL, !0), n.texImage2D(n.TEXTURE_2D, 0, n.RGBA, n.RGBA, n.UNSIGNED_BYTE, t), e.i.gpuOriginForWebTexturesIsBottomLeft && n.pixelStorei(n.UNPACK_FLIP_Y_WEBGL, !1);
	let [r, i] = vs(t);
	return !e.l || r === e.i.canvas.width && i === e.i.canvas.height || (e.i.canvas.width = r, e.i.canvas.height = i), [r, i];
}
function bs(e, t, n) {
	e.m || console.error("No wasm multistream support detected: ensure dependency inclusion of :gl_graph_runner_internal_multi_input target");
	let r = new Uint32Array(t.length);
	for (let n = 0; n < t.length; n++) r[n] = e.i.stringToNewUTF8(t[n]);
	t = e.i._malloc(4 * r.length), e.i.HEAPU32.set(r, t >> 2), n(t);
	for (let t of r) e.i._free(t);
	e.i._free(t);
}
function xs(e, t, n) {
	e.i.simpleListeners = e.i.simpleListeners || {}, e.i.simpleListeners[t] = n;
}
function Ss(e, t, n) {
	let r = [];
	e.i.simpleListeners = e.i.simpleListeners || {}, e.i.simpleListeners[t] = (e, t, i) => {
		t ? (n(r, i), r = []) : r.push(e);
	};
}
hs.forVisionTasks = function(e, t = !1) {
	return ms("vision", e ?? ha``, t);
}, hs.forTextTasks = function(e, t = !1) {
	return ms("text", e ?? ha``, t);
}, hs.forGenAiTasks = function(e, t = !1) {
	return ms("genai", e ?? ha``, t);
}, hs.forAudioTasks = function(e, t = !1) {
	return ms("audio", e ?? ha``, t);
}, hs.isSimdSupported = function(e = !1) {
	return ps(e);
};
async function Cs(e, t, n, r) {
	return e = await (async (e, t, n, r, i) => {
		if (t && await _s(t), !self.ModuleFactory || n && (await _s(n), !self.ModuleFactory)) throw Error("ModuleFactory not set.");
		return self.Module && i && ((t = self.Module).locateFile = i.locateFile, i.mainScriptUrlOrBlob && (t.mainScriptUrlOrBlob = i.mainScriptUrlOrBlob)), i = await self.ModuleFactory(self.Module || i), self.ModuleFactory = self.Module = void 0, new e(i, r);
	})(e, n.wasmLoaderPath, n.assetLoaderPath, t, { locateFile: (e) => e.endsWith(".wasm") ? n.wasmBinaryPath.toString() : n.assetBinaryPath && e.endsWith(".data") ? n.assetBinaryPath.toString() : e }), await e.o(r), e;
}
function ws(e, t) {
	let n = x(e.baseOptions, so, 1) || new so();
	typeof t == "string" ? (y(n, 2, en(t)), y(n, 1)) : t instanceof Uint8Array && (y(n, 1, tt(t, !1)), y(n, 2)), S(e.baseOptions, 0, 1, n);
}
function Ts(e) {
	try {
		let t = e.H.length;
		if (t === 1) throw Error(e.H[0].message);
		if (t > 1) throw Error("Encountered multiple errors: " + e.H.map(((e) => e.message)).join(", "));
	} finally {
		e.H = [];
	}
}
function G(e, t) {
	e.C = Math.max(e.C, t);
}
function Es(e, t) {
	e.B = new z(), dr(e.B, 2, "PassThroughCalculator"), L(e.B, "free_memory"), R(e.B, "free_memory_unused_out"), B(t, "free_memory"), ka(t, e.B);
}
function Ds(e, t) {
	L(e.B, t), R(e.B, t + "_unused_out");
}
function Os(e) {
	e.g.addBoolToStream(!0, "free_memory", e.C);
}
var ks = class {
	constructor(e) {
		this.g = e, this.H = [], this.C = 0, this.g.setAutoRenderToScreen(!1);
	}
	l(e, t = !0) {
		if (t) {
			let t = e.baseOptions || {};
			if (e.baseOptions?.modelAssetBuffer && e.baseOptions?.modelAssetPath) throw Error("Cannot set both baseOptions.modelAssetPath and baseOptions.modelAssetBuffer");
			if (!(x(this.baseOptions, so, 1)?.g() || x(this.baseOptions, so, 1)?.l() || e.baseOptions?.modelAssetBuffer || e.baseOptions?.modelAssetPath)) throw Error("Either baseOptions.modelAssetPath or baseOptions.modelAssetBuffer must be set");
			if (function(e, t) {
				let n = x(e.baseOptions, ao, 3);
				if (!n) {
					var r = n = new ao(), i = new xa();
					or(r, 4, oo, i);
				}
				"delegate" in t && (t.delegate === "GPU" ? (t = n, r = new _a(), or(t, 2, oo, r)) : (t = n, r = new xa(), or(t, 4, oo, r))), S(e.baseOptions, 0, 3, n);
			}(this, t), t.modelAssetPath) return fetch(t.modelAssetPath.toString()).then(((e) => {
				if (e.ok) return e.arrayBuffer();
				throw Error(`Failed to fetch model: ${t.modelAssetPath} (${e.status})`);
			})).then(((e) => {
				try {
					this.g.i.FS_unlink("/model.dat");
				} catch {}
				this.g.i.FS_createDataFile("/", "model.dat", new Uint8Array(e), !0, !1, !1), ws(this, "/model.dat"), this.m(), this.L();
			}));
			if (t.modelAssetBuffer instanceof Uint8Array) ws(this, t.modelAssetBuffer);
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
				ws(this, e), this.m(), this.L();
			}));
		}
		return this.m(), this.L(), Promise.resolve();
	}
	L() {}
	ca() {
		let e;
		if (this.g.ca(((t) => {
			e = Ma(t);
		})), !e) throw Error("Failed to retrieve CalculatorGraphConfig");
		return e;
	}
	setGraph(e, t) {
		this.g.attachErrorListener(((e, t) => {
			this.H.push(Error(t));
		})), this.g.Ja(), this.g.setGraph(e, t), this.B = void 0, Ts(this);
	}
	finishProcessing() {
		this.g.finishProcessing(), Ts(this);
	}
	close() {
		this.B = void 0, this.g.closeGraph();
	}
};
function K(e, t) {
	if (!e) throw Error(`Unable to obtain required WebGL resource: ${t}`);
	return e;
}
ks.prototype.close = ks.prototype.close;
var As = class {
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
function js(e, t, n) {
	let r = e.g;
	if (n = K(r.createShader(n), "Failed to create WebGL shader"), r.shaderSource(n, t), r.compileShader(n), !r.getShaderParameter(n, r.COMPILE_STATUS)) throw Error(`Could not compile WebGL shader: ${r.getShaderInfoLog(n)}`);
	return r.attachShader(e.h, n), n;
}
function Ms(e, t) {
	let n = e.g, r = K(n.createVertexArray(), "Failed to create vertex array");
	n.bindVertexArray(r);
	let i = K(n.createBuffer(), "Failed to create buffer");
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
	let a = K(n.createBuffer(), "Failed to create buffer");
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
	]), n.STATIC_DRAW), n.bindBuffer(n.ARRAY_BUFFER, null), n.bindVertexArray(null), new As(n, r, i, a);
}
function Ns(e, t) {
	if (e.g) {
		if (t !== e.g) throw Error("Cannot change GL context once initialized");
	} else e.g = t;
}
function Ps(e, t, n, r) {
	return Ns(e, t), e.h || (e.m(), e.D()), n ? (e.u ||= Ms(e, !0), n = e.u) : (e.A ||= Ms(e, !1), n = e.A), t.useProgram(e.h), n.bind(), e.l(), e = r(), n.g.bindVertexArray(null), e;
}
function Fs(e, t, n) {
	return Ns(e, t), e = K(t.createTexture(), "Failed to create texture"), t.bindTexture(t.TEXTURE_2D, e), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_S, t.CLAMP_TO_EDGE), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_WRAP_T, t.CLAMP_TO_EDGE), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_MIN_FILTER, n ?? t.LINEAR), t.texParameteri(t.TEXTURE_2D, t.TEXTURE_MAG_FILTER, n ?? t.LINEAR), t.bindTexture(t.TEXTURE_2D, null), e;
}
function Is(e, t, n) {
	Ns(e, t), e.B ||= K(t.createFramebuffer(), "Failed to create framebuffe."), t.bindFramebuffer(t.FRAMEBUFFER, e.B), t.framebufferTexture2D(t.FRAMEBUFFER, t.COLOR_ATTACHMENT0, t.TEXTURE_2D, n, 0);
}
function Ls(e) {
	e.g?.bindFramebuffer(e.g.FRAMEBUFFER, null);
}
var Rs = class {
	H() {
		return "\n  precision mediump float;\n  varying vec2 vTex;\n  uniform sampler2D inputTexture;\n  void main() {\n    gl_FragColor = texture2D(inputTexture, vTex);\n  }\n ";
	}
	m() {
		let e = this.g;
		if (this.h = K(e.createProgram(), "Failed to create WebGL program"), this.X = js(this, "\n  attribute vec2 aVertex;\n  attribute vec2 aTex;\n  varying vec2 vTex;\n  void main(void) {\n    gl_Position = vec4(aVertex, 0.0, 1.0);\n    vTex = aTex;\n  }", e.VERTEX_SHADER), this.W = js(this, this.H(), e.FRAGMENT_SHADER), e.linkProgram(this.h), !e.getProgramParameter(this.h, e.LINK_STATUS)) throw Error(`Error during program linking: ${e.getProgramInfoLog(this.h)}`);
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
}, zs = class extends Rs {
	H() {
		return "\n  precision mediump float;\n  uniform sampler2D backgroundTexture;\n  uniform sampler2D maskTexture;\n  uniform sampler2D colorMappingTexture;\n  varying vec2 vTex;\n  void main() {\n    vec4 backgroundColor = texture2D(backgroundTexture, vTex);\n    float category = texture2D(maskTexture, vTex).r;\n    vec4 categoryColor = texture2D(colorMappingTexture, vec2(category, 0.0));\n    gl_FragColor = mix(backgroundColor, categoryColor, categoryColor.a);\n  }\n ";
	}
	D() {
		let e = this.g;
		e.activeTexture(e.TEXTURE1), this.C = Fs(this, e, e.LINEAR), e.activeTexture(e.TEXTURE2), this.j = Fs(this, e, e.NEAREST);
	}
	m() {
		super.m();
		let e = this.g;
		this.P = K(e.getUniformLocation(this.h, "backgroundTexture"), "Uniform location"), this.U = K(e.getUniformLocation(this.h, "colorMappingTexture"), "Uniform location"), this.M = K(e.getUniformLocation(this.h, "maskTexture"), "Uniform location");
	}
	l() {
		super.l();
		let e = this.g;
		e.uniform1i(this.M, 0), e.uniform1i(this.P, 1), e.uniform1i(this.U, 2);
	}
	close() {
		this.C && this.g.deleteTexture(this.C), this.j && this.g.deleteTexture(this.j), super.close();
	}
}, Bs = class extends Rs {
	H() {
		return "\n  precision mediump float;\n  uniform sampler2D maskTexture;\n  uniform sampler2D defaultTexture;\n  uniform sampler2D overlayTexture;\n  varying vec2 vTex;\n  void main() {\n    float confidence = texture2D(maskTexture, vTex).r;\n    vec4 defaultColor = texture2D(defaultTexture, vTex);\n    vec4 overlayColor = texture2D(overlayTexture, vTex);\n    // Apply the alpha from the overlay and merge in the default color\n    overlayColor = mix(defaultColor, overlayColor, overlayColor.a);\n    gl_FragColor = mix(defaultColor, overlayColor, confidence);\n  }\n ";
	}
	D() {
		let e = this.g;
		e.activeTexture(e.TEXTURE1), this.j = Fs(this, e), e.activeTexture(e.TEXTURE2), this.C = Fs(this, e);
	}
	m() {
		super.m();
		let e = this.g;
		this.M = K(e.getUniformLocation(this.h, "defaultTexture"), "Uniform location"), this.P = K(e.getUniformLocation(this.h, "overlayTexture"), "Uniform location"), this.I = K(e.getUniformLocation(this.h, "maskTexture"), "Uniform location");
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
function Vs(e, t) {
	switch (t) {
		case 0: return e.g.find(((e) => e instanceof Uint8Array));
		case 1: return e.g.find(((e) => e instanceof Float32Array));
		case 2: return e.g.find(((e) => typeof WebGLTexture < "u" && e instanceof WebGLTexture));
		default: throw Error(`Type is not supported: ${t}`);
	}
}
function Hs(e) {
	var t = Vs(e, 1);
	if (!t) {
		if (t = Vs(e, 0)) t = new Float32Array(t).map(((e) => e / 255));
		else {
			t = new Float32Array(e.width * e.height);
			let r = Ws(e);
			var n = Ks(e);
			if (Is(n, r, Us(e)), "iPad Simulator;iPhone Simulator;iPod Simulator;iPad;iPhone;iPod".split(";").includes(navigator.platform) || navigator.userAgent.includes("Mac") && "document" in self && "ontouchend" in self.document) {
				n = new Float32Array(e.width * e.height * 4), r.readPixels(0, 0, e.width, e.height, r.RGBA, r.FLOAT, n);
				for (let e = 0, r = 0; e < t.length; ++e, r += 4) t[e] = n[r];
			} else r.readPixels(0, 0, e.width, e.height, r.RED, r.FLOAT, t);
		}
		e.g.push(t);
	}
	return t;
}
function Us(e) {
	let t = Vs(e, 2);
	if (!t) {
		let n = Ws(e);
		t = qs(e);
		let r = Hs(e), i = Gs(e);
		n.texImage2D(n.TEXTURE_2D, 0, i, e.width, e.height, 0, n.RED, n.FLOAT, r), Js(e);
	}
	return t;
}
function Ws(e) {
	if (!e.canvas) throw Error("Conversion to different image formats require that a canvas is passed when initializing the image.");
	return e.h ||= K(e.canvas.getContext("webgl2"), "You cannot use a canvas that is already bound to a different type of rendering context."), e.h;
}
function Gs(e) {
	if (e = Ws(e), !Ys) if (e.getExtension("EXT_color_buffer_float") && e.getExtension("OES_texture_float_linear") && e.getExtension("EXT_float_blend")) Ys = e.R32F;
	else {
		if (!e.getExtension("EXT_color_buffer_half_float")) throw Error("GPU does not fully support 4-channel float32 or float16 formats");
		Ys = e.R16F;
	}
	return Ys;
}
function Ks(e) {
	return e.l ||= new Rs(), e.l;
}
function qs(e) {
	let t = Ws(e);
	t.viewport(0, 0, e.width, e.height), t.activeTexture(t.TEXTURE0);
	let n = Vs(e, 2);
	return n || (n = Fs(Ks(e), t, e.m ? t.LINEAR : t.NEAREST), e.g.push(n), e.j = !0), t.bindTexture(t.TEXTURE_2D, n), n;
}
function Js(e) {
	e.h.bindTexture(e.h.TEXTURE_2D, null);
}
var Ys, q = class {
	constructor(e, t, n, r, i, a, o) {
		this.g = e, this.m = t, this.j = n, this.canvas = r, this.l = i, this.width = a, this.height = o, this.j && --Xs === 0 && console.error("You seem to be creating MPMask instances without invoking .close(). This leaks resources.");
	}
	Fa() {
		return !!Vs(this, 0);
	}
	ka() {
		return !!Vs(this, 1);
	}
	R() {
		return !!Vs(this, 2);
	}
	ja() {
		return (t = Vs(e = this, 0)) || (t = Hs(e), t = new Uint8Array(t.map(((e) => Math.round(255 * e)))), e.g.push(t)), t;
		var e, t;
	}
	ia() {
		return Hs(this);
	}
	N() {
		return Us(this);
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
					let e = Ws(this), t = Ks(this);
					e.activeTexture(e.TEXTURE1), n = Fs(t, e, this.m ? e.LINEAR : e.NEAREST), e.bindTexture(e.TEXTURE_2D, n);
					let r = Gs(this);
					e.texImage2D(e.TEXTURE_2D, 0, r, this.width, this.height, 0, e.RED, e.FLOAT, null), e.bindTexture(e.TEXTURE_2D, null), Is(t, e, n), Ps(t, e, !1, (() => {
						qs(this), e.clearColor(0, 0, 0, 0), e.clear(e.COLOR_BUFFER_BIT), e.drawArrays(e.TRIANGLE_FAN, 0, 4), Js(this);
					})), Ls(t), Js(this);
				}
			}
			e.push(n);
		}
		return new q(e, this.m, this.R(), this.canvas, this.l, this.width, this.height);
	}
	close() {
		this.j && Ws(this).deleteTexture(Vs(this, 2)), Xs = -1;
	}
};
q.prototype.close = q.prototype.close, q.prototype.clone = q.prototype.clone, q.prototype.getAsWebGLTexture = q.prototype.N, q.prototype.getAsFloat32Array = q.prototype.ia, q.prototype.getAsUint8Array = q.prototype.ja, q.prototype.hasWebGLTexture = q.prototype.R, q.prototype.hasFloat32Array = q.prototype.ka, q.prototype.hasUint8Array = q.prototype.Fa;
var Xs = 250, Zs = {
	color: "white",
	lineWidth: 4,
	radius: 6
};
function Qs(e) {
	return {
		...Zs,
		fillColor: (e ||= {}).color,
		...e
	};
}
function $s(e, t) {
	return e instanceof Function ? e(t) : e;
}
function ec(e, t, n) {
	return Math.max(Math.min(t, n), Math.min(Math.max(t, n), e));
}
function tc(e) {
	if (!e.l) throw Error("CPU rendering requested but CanvasRenderingContext2D not provided.");
	return e.l;
}
function nc(e) {
	if (!e.j) throw Error("GPU rendering requested but WebGL2RenderingContext not provided.");
	return e.j;
}
function rc(e, t, n) {
	if (t.R()) n(t.N());
	else {
		let r = t.ka() ? t.ia() : t.ja();
		e.m = e.m ?? new Rs();
		let i = nc(e);
		n((e = new q([r], t.m, !1, i.canvas, e.m, t.width, t.height)).N()), e.close();
	}
}
function ic(e, t, n, r) {
	let i = function(e) {
		return e.g ||= new zs(), e.g;
	}(e), a = nc(e), o = Array.isArray(n) ? new ImageData(new Uint8ClampedArray(n), 1, 1) : n;
	Ps(i, a, !0, (() => {
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
function ac(e, t, n, r) {
	let i = nc(e), a = function(e) {
		return e.h ||= new Bs(), e.h;
	}(e), o = Array.isArray(n) ? new ImageData(new Uint8ClampedArray(n), 1, 1) : n, s = Array.isArray(r) ? new ImageData(new Uint8ClampedArray(r), 1, 1) : r;
	Ps(a, i, !0, (() => {
		var e = a.g;
		e.activeTexture(e.TEXTURE0), e.bindTexture(e.TEXTURE_2D, t), e.activeTexture(e.TEXTURE1), e.bindTexture(e.TEXTURE_2D, a.j), e.texImage2D(e.TEXTURE_2D, 0, e.RGBA, e.RGBA, e.UNSIGNED_BYTE, o), e.activeTexture(e.TEXTURE2), e.bindTexture(e.TEXTURE_2D, a.C), e.texImage2D(e.TEXTURE_2D, 0, e.RGBA, e.RGBA, e.UNSIGNED_BYTE, s), i.clearColor(0, 0, 0, 0), i.clear(i.COLOR_BUFFER_BIT), i.drawArrays(i.TRIANGLE_FAN, 0, 4), i.bindTexture(i.TEXTURE_2D, null), (e = a.g).activeTexture(e.TEXTURE0), e.bindTexture(e.TEXTURE_2D, null), e.activeTexture(e.TEXTURE1), e.bindTexture(e.TEXTURE_2D, null), e.activeTexture(e.TEXTURE2), e.bindTexture(e.TEXTURE_2D, null);
	}));
}
var J = class {
	constructor(e, t) {
		typeof CanvasRenderingContext2D < "u" && e instanceof CanvasRenderingContext2D || e instanceof OffscreenCanvasRenderingContext2D ? (this.l = e, this.j = t) : this.j = e;
	}
	ya(e, t) {
		if (e) {
			var n = tc(this);
			t = Qs(t), n.save();
			var r = n.canvas, i = 0;
			for (let a of e) n.fillStyle = $s(t.fillColor, {
				index: i,
				from: a
			}), n.strokeStyle = $s(t.color, {
				index: i,
				from: a
			}), n.lineWidth = $s(t.lineWidth, {
				index: i,
				from: a
			}), (e = new Path2D()).arc(a.x * r.width, a.y * r.height, $s(t.radius, {
				index: i,
				from: a
			}), 0, 2 * Math.PI), n.fill(e), n.stroke(e), ++i;
			n.restore();
		}
	}
	xa(e, t, n) {
		if (e && t) {
			var r = tc(this);
			n = Qs(n), r.save();
			var i = r.canvas, a = 0;
			for (let o of t) {
				r.beginPath(), t = e[o.start];
				let s = e[o.end];
				t && s && (r.strokeStyle = $s(n.color, {
					index: a,
					from: t,
					to: s
				}), r.lineWidth = $s(n.lineWidth, {
					index: a,
					from: t,
					to: s
				}), r.moveTo(t.x * i.width, t.y * i.height), r.lineTo(s.x * i.width, s.y * i.height)), ++a, r.stroke();
			}
			r.restore();
		}
	}
	ua(e, t) {
		let n = tc(this);
		t = Qs(t), n.save(), n.beginPath(), n.lineWidth = $s(t.lineWidth, {}), n.strokeStyle = $s(t.color, {}), n.fillStyle = $s(t.fillColor, {}), n.moveTo(e.originX, e.originY), n.lineTo(e.originX + e.width, e.originY), n.lineTo(e.originX + e.width, e.originY + e.height), n.lineTo(e.originX, e.originY + e.height), n.lineTo(e.originX, e.originY), n.stroke(), n.fill(), n.restore();
	}
	va(e, t, n = [
		0,
		0,
		0,
		255
	]) {
		this.l ? function(e, t, n, r) {
			let i = nc(e);
			rc(e, t, ((t) => {
				ic(e, t, n, r), (t = tc(e)).drawImage(i.canvas, 0, 0, t.canvas.width, t.canvas.height);
			}));
		}(this, e, n, t) : ic(this, e.N(), n, t);
	}
	wa(e, t, n) {
		this.l ? function(e, t, n, r) {
			let i = nc(e);
			rc(e, t, ((t) => {
				ac(e, t, n, r), (t = tc(e)).drawImage(i.canvas, 0, 0, t.canvas.width, t.canvas.height);
			}));
		}(this, e, t, n) : ac(this, e.N(), t, n);
	}
	close() {
		this.g?.close(), this.g = void 0, this.h?.close(), this.h = void 0, this.m?.close(), this.m = void 0;
	}
};
function oc(e, t) {
	switch (t) {
		case 0: return e.g.find(((e) => e instanceof ImageData));
		case 1: return e.g.find(((e) => typeof ImageBitmap < "u" && e instanceof ImageBitmap));
		case 2: return e.g.find(((e) => typeof WebGLTexture < "u" && e instanceof WebGLTexture));
		default: throw Error(`Type is not supported: ${t}`);
	}
}
function sc(e) {
	var t = oc(e, 0);
	if (!t) {
		t = lc(e);
		let n = uc(e), r = new Uint8Array(e.width * e.height * 4);
		Is(n, t, cc(e)), t.readPixels(0, 0, e.width, e.height, t.RGBA, t.UNSIGNED_BYTE, r), Ls(n), t = new ImageData(new Uint8ClampedArray(r.buffer), e.width, e.height), e.g.push(t);
	}
	return t;
}
function cc(e) {
	let t = oc(e, 2);
	if (!t) {
		let n = lc(e);
		t = dc(e);
		let r = oc(e, 1) || sc(e);
		n.texImage2D(n.TEXTURE_2D, 0, n.RGBA, n.RGBA, n.UNSIGNED_BYTE, r), fc(e);
	}
	return t;
}
function lc(e) {
	if (!e.canvas) throw Error("Conversion to different image formats require that a canvas is passed when initializing the image.");
	return e.h ||= K(e.canvas.getContext("webgl2"), "You cannot use a canvas that is already bound to a different type of rendering context."), e.h;
}
function uc(e) {
	return e.l ||= new Rs(), e.l;
}
function dc(e) {
	let t = lc(e);
	t.viewport(0, 0, e.width, e.height), t.activeTexture(t.TEXTURE0);
	let n = oc(e, 2);
	return n || (n = Fs(uc(e), t), e.g.push(n), e.m = !0), t.bindTexture(t.TEXTURE_2D, n), n;
}
function fc(e) {
	e.h.bindTexture(e.h.TEXTURE_2D, null);
}
function pc(e) {
	let t = lc(e);
	return Ps(uc(e), t, !0, (() => function(e, t) {
		let n = e.canvas;
		if (n.width === e.width && n.height === e.height) return t();
		let r = n.width, i = n.height;
		return n.width = e.width, n.height = e.height, e = t(), n.width = r, n.height = i, e;
	}(e, (() => {
		if (t.bindFramebuffer(t.FRAMEBUFFER, null), t.clearColor(0, 0, 0, 0), t.clear(t.COLOR_BUFFER_BIT), t.drawArrays(t.TRIANGLE_FAN, 0, 4), !(e.canvas instanceof OffscreenCanvas)) throw Error("Conversion to ImageBitmap requires that the MediaPipe Tasks is initialized with an OffscreenCanvas");
		return e.canvas.transferToImageBitmap();
	}))));
}
J.prototype.close = J.prototype.close, J.prototype.drawConfidenceMask = J.prototype.wa, J.prototype.drawCategoryMask = J.prototype.va, J.prototype.drawBoundingBox = J.prototype.ua, J.prototype.drawConnectors = J.prototype.xa, J.prototype.drawLandmarks = J.prototype.ya, J.lerp = function(e, t, n, r, i) {
	return ec(r * (1 - (e - t) / (n - t)) + i * (1 - (n - e) / (n - t)), r, i);
}, J.clamp = ec;
var Y = class {
	constructor(e, t, n, r, i, a, o) {
		this.g = e, this.j = t, this.m = n, this.canvas = r, this.l = i, this.width = a, this.height = o, (this.j || this.m) && --mc === 0 && console.error("You seem to be creating MPImage instances without invoking .close(). This leaks resources.");
	}
	Ea() {
		return !!oc(this, 0);
	}
	la() {
		return !!oc(this, 1);
	}
	R() {
		return !!oc(this, 2);
	}
	Ca() {
		return sc(this);
	}
	Ba() {
		var e = oc(this, 1);
		return e || (cc(this), dc(this), e = pc(this), fc(this), this.g.push(e), this.j = !0), e;
	}
	N() {
		return cc(this);
	}
	clone() {
		let e = [];
		for (let t of this.g) {
			let n;
			if (t instanceof ImageData) n = new ImageData(t.data, this.width, this.height);
			else if (t instanceof WebGLTexture) {
				let e = lc(this), t = uc(this);
				e.activeTexture(e.TEXTURE1), n = Fs(t, e), e.bindTexture(e.TEXTURE_2D, n), e.texImage2D(e.TEXTURE_2D, 0, e.RGBA, this.width, this.height, 0, e.RGBA, e.UNSIGNED_BYTE, null), e.bindTexture(e.TEXTURE_2D, null), Is(t, e, n), Ps(t, e, !1, (() => {
					dc(this), e.clearColor(0, 0, 0, 0), e.clear(e.COLOR_BUFFER_BIT), e.drawArrays(e.TRIANGLE_FAN, 0, 4), fc(this);
				})), Ls(t), fc(this);
			} else {
				if (!(t instanceof ImageBitmap)) throw Error(`Type is not supported: ${t}`);
				cc(this), dc(this), n = pc(this), fc(this);
			}
			e.push(n);
		}
		return new Y(e, this.la(), this.R(), this.canvas, this.l, this.width, this.height);
	}
	close() {
		this.j && oc(this, 1).close(), this.m && lc(this).deleteTexture(oc(this, 2)), mc = -1;
	}
};
Y.prototype.close = Y.prototype.close, Y.prototype.clone = Y.prototype.clone, Y.prototype.getAsWebGLTexture = Y.prototype.N, Y.prototype.getAsImageBitmap = Y.prototype.Ba, Y.prototype.getAsImageData = Y.prototype.Ca, Y.prototype.hasWebGLTexture = Y.prototype.R, Y.prototype.hasImageBitmap = Y.prototype.la, Y.prototype.hasImageData = Y.prototype.Ea;
var mc = 250;
function hc(...e) {
	return e.map((([e, t]) => ({
		start: e,
		end: t
	})));
}
var gc = function(e) {
	return class extends e {
		Ja() {
			this.i._registerModelResourcesGraphService();
		}
	};
}((_c = class {
	constructor(e, t) {
		this.l = !0, this.i = e, this.g = null, this.h = 0, this.m = typeof this.i._addIntToInputStream == "function", t === void 0 ? gs() ? this.i.canvas = new OffscreenCanvas(1, 1) : (console.warn("OffscreenCanvas not supported and GraphRunner constructor glCanvas parameter is undefined. Creating backup canvas."), this.i.canvas = document.createElement("canvas")) : this.i.canvas = t;
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
		this.i._configureAudio || console.warn("Attempting to use configureAudio without support for input audio. Is build dep \":gl_graph_runner_audio\" missing?"), W(this, r || "input_audio", ((r) => {
			W(this, i ||= "audio_header", ((i) => {
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
		xs(this, "__graph_config__", ((t) => {
			e(t);
		})), W(this, "__graph_config__", ((e) => {
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
		this.h !== a && (this.g && this.i._free(this.g), this.g = this.i._malloc(a), this.h = a), this.i.HEAPF32.set(e, this.g / 4), W(this, r, ((e) => {
			this.i._addAudioToInputStream(this.g, t, n, e, i);
		}));
	}
	addGpuBufferToStream(e, t, n) {
		W(this, t, ((t) => {
			let [r, i] = ys(this, e, t);
			this.i._addBoundTextureToStream(t, r, i, n);
		}));
	}
	addBoolToStream(e, t, n) {
		W(this, t, ((t) => {
			this.i._addBoolToInputStream(e, t, n);
		}));
	}
	addDoubleToStream(e, t, n) {
		W(this, t, ((t) => {
			this.i._addDoubleToInputStream(e, t, n);
		}));
	}
	addFloatToStream(e, t, n) {
		W(this, t, ((t) => {
			this.i._addFloatToInputStream(e, t, n);
		}));
	}
	addIntToStream(e, t, n) {
		W(this, t, ((t) => {
			this.i._addIntToInputStream(e, t, n);
		}));
	}
	addUintToStream(e, t, n) {
		W(this, t, ((t) => {
			this.i._addUintToInputStream(e, t, n);
		}));
	}
	addStringToStream(e, t, n) {
		W(this, t, ((t) => {
			W(this, e, ((e) => {
				this.i._addStringToInputStream(e, t, n);
			}));
		}));
	}
	addStringRecordToStream(e, t, n) {
		W(this, t, ((t) => {
			bs(this, Object.keys(e), ((r) => {
				bs(this, Object.values(e), ((i) => {
					this.i._addFlatHashMapToInputStream(r, i, Object.keys(e).length, t, n);
				}));
			}));
		}));
	}
	addProtoToStream(e, t, n, r) {
		W(this, n, ((n) => {
			W(this, t, ((t) => {
				let i = this.i._malloc(e.length);
				this.i.HEAPU8.set(e, i), this.i._addProtoToInputStream(i, e.length, t, n, r), this.i._free(i);
			}));
		}));
	}
	addEmptyPacketToStream(e, t) {
		W(this, e, ((e) => {
			this.i._addEmptyPacketToInputStream(e, t);
		}));
	}
	addBoolVectorToStream(e, t, n) {
		W(this, t, ((t) => {
			let r = this.i._allocateBoolVector(e.length);
			if (!r) throw Error("Unable to allocate new bool vector on heap.");
			for (let t of e) this.i._addBoolVectorEntry(r, t);
			this.i._addBoolVectorToInputStream(r, t, n);
		}));
	}
	addDoubleVectorToStream(e, t, n) {
		W(this, t, ((t) => {
			let r = this.i._allocateDoubleVector(e.length);
			if (!r) throw Error("Unable to allocate new double vector on heap.");
			for (let t of e) this.i._addDoubleVectorEntry(r, t);
			this.i._addDoubleVectorToInputStream(r, t, n);
		}));
	}
	addFloatVectorToStream(e, t, n) {
		W(this, t, ((t) => {
			let r = this.i._allocateFloatVector(e.length);
			if (!r) throw Error("Unable to allocate new float vector on heap.");
			for (let t of e) this.i._addFloatVectorEntry(r, t);
			this.i._addFloatVectorToInputStream(r, t, n);
		}));
	}
	addIntVectorToStream(e, t, n) {
		W(this, t, ((t) => {
			let r = this.i._allocateIntVector(e.length);
			if (!r) throw Error("Unable to allocate new int vector on heap.");
			for (let t of e) this.i._addIntVectorEntry(r, t);
			this.i._addIntVectorToInputStream(r, t, n);
		}));
	}
	addUintVectorToStream(e, t, n) {
		W(this, t, ((t) => {
			let r = this.i._allocateUintVector(e.length);
			if (!r) throw Error("Unable to allocate new unsigned int vector on heap.");
			for (let t of e) this.i._addUintVectorEntry(r, t);
			this.i._addUintVectorToInputStream(r, t, n);
		}));
	}
	addStringVectorToStream(e, t, n) {
		W(this, t, ((t) => {
			let r = this.i._allocateStringVector(e.length);
			if (!r) throw Error("Unable to allocate new string vector on heap.");
			for (let t of e) W(this, t, ((e) => {
				this.i._addStringVectorEntry(r, e);
			}));
			this.i._addStringVectorToInputStream(r, t, n);
		}));
	}
	addBoolToInputSidePacket(e, t) {
		W(this, t, ((t) => {
			this.i._addBoolToInputSidePacket(e, t);
		}));
	}
	addDoubleToInputSidePacket(e, t) {
		W(this, t, ((t) => {
			this.i._addDoubleToInputSidePacket(e, t);
		}));
	}
	addFloatToInputSidePacket(e, t) {
		W(this, t, ((t) => {
			this.i._addFloatToInputSidePacket(e, t);
		}));
	}
	addIntToInputSidePacket(e, t) {
		W(this, t, ((t) => {
			this.i._addIntToInputSidePacket(e, t);
		}));
	}
	addUintToInputSidePacket(e, t) {
		W(this, t, ((t) => {
			this.i._addUintToInputSidePacket(e, t);
		}));
	}
	addStringToInputSidePacket(e, t) {
		W(this, t, ((t) => {
			W(this, e, ((e) => {
				this.i._addStringToInputSidePacket(e, t);
			}));
		}));
	}
	addProtoToInputSidePacket(e, t, n) {
		W(this, n, ((n) => {
			W(this, t, ((t) => {
				let r = this.i._malloc(e.length);
				this.i.HEAPU8.set(e, r), this.i._addProtoToInputSidePacket(r, e.length, t, n), this.i._free(r);
			}));
		}));
	}
	addBoolVectorToInputSidePacket(e, t) {
		W(this, t, ((t) => {
			let n = this.i._allocateBoolVector(e.length);
			if (!n) throw Error("Unable to allocate new bool vector on heap.");
			for (let t of e) this.i._addBoolVectorEntry(n, t);
			this.i._addBoolVectorToInputSidePacket(n, t);
		}));
	}
	addDoubleVectorToInputSidePacket(e, t) {
		W(this, t, ((t) => {
			let n = this.i._allocateDoubleVector(e.length);
			if (!n) throw Error("Unable to allocate new double vector on heap.");
			for (let t of e) this.i._addDoubleVectorEntry(n, t);
			this.i._addDoubleVectorToInputSidePacket(n, t);
		}));
	}
	addFloatVectorToInputSidePacket(e, t) {
		W(this, t, ((t) => {
			let n = this.i._allocateFloatVector(e.length);
			if (!n) throw Error("Unable to allocate new float vector on heap.");
			for (let t of e) this.i._addFloatVectorEntry(n, t);
			this.i._addFloatVectorToInputSidePacket(n, t);
		}));
	}
	addIntVectorToInputSidePacket(e, t) {
		W(this, t, ((t) => {
			let n = this.i._allocateIntVector(e.length);
			if (!n) throw Error("Unable to allocate new int vector on heap.");
			for (let t of e) this.i._addIntVectorEntry(n, t);
			this.i._addIntVectorToInputSidePacket(n, t);
		}));
	}
	addUintVectorToInputSidePacket(e, t) {
		W(this, t, ((t) => {
			let n = this.i._allocateUintVector(e.length);
			if (!n) throw Error("Unable to allocate new unsigned int vector on heap.");
			for (let t of e) this.i._addUintVectorEntry(n, t);
			this.i._addUintVectorToInputSidePacket(n, t);
		}));
	}
	addStringVectorToInputSidePacket(e, t) {
		W(this, t, ((t) => {
			let n = this.i._allocateStringVector(e.length);
			if (!n) throw Error("Unable to allocate new string vector on heap.");
			for (let t of e) W(this, t, ((e) => {
				this.i._addStringVectorEntry(n, e);
			}));
			this.i._addStringVectorToInputSidePacket(n, t);
		}));
	}
	attachBoolListener(e, t) {
		xs(this, e, t), W(this, e, ((e) => {
			this.i._attachBoolListener(e);
		}));
	}
	attachBoolVectorListener(e, t) {
		Ss(this, e, t), W(this, e, ((e) => {
			this.i._attachBoolVectorListener(e);
		}));
	}
	attachIntListener(e, t) {
		xs(this, e, t), W(this, e, ((e) => {
			this.i._attachIntListener(e);
		}));
	}
	attachIntVectorListener(e, t) {
		Ss(this, e, t), W(this, e, ((e) => {
			this.i._attachIntVectorListener(e);
		}));
	}
	attachUintListener(e, t) {
		xs(this, e, t), W(this, e, ((e) => {
			this.i._attachUintListener(e);
		}));
	}
	attachUintVectorListener(e, t) {
		Ss(this, e, t), W(this, e, ((e) => {
			this.i._attachUintVectorListener(e);
		}));
	}
	attachDoubleListener(e, t) {
		xs(this, e, t), W(this, e, ((e) => {
			this.i._attachDoubleListener(e);
		}));
	}
	attachDoubleVectorListener(e, t) {
		Ss(this, e, t), W(this, e, ((e) => {
			this.i._attachDoubleVectorListener(e);
		}));
	}
	attachFloatListener(e, t) {
		xs(this, e, t), W(this, e, ((e) => {
			this.i._attachFloatListener(e);
		}));
	}
	attachFloatVectorListener(e, t) {
		Ss(this, e, t), W(this, e, ((e) => {
			this.i._attachFloatVectorListener(e);
		}));
	}
	attachStringListener(e, t) {
		xs(this, e, t), W(this, e, ((e) => {
			this.i._attachStringListener(e);
		}));
	}
	attachStringVectorListener(e, t) {
		Ss(this, e, t), W(this, e, ((e) => {
			this.i._attachStringVectorListener(e);
		}));
	}
	attachProtoListener(e, t, n) {
		xs(this, e, t), W(this, e, ((e) => {
			this.i._attachProtoListener(e, n || !1);
		}));
	}
	attachProtoVectorListener(e, t, n) {
		Ss(this, e, t), W(this, e, ((e) => {
			this.i._attachProtoVectorListener(e, n || !1);
		}));
	}
	attachAudioListener(e, t, n) {
		this.i._attachAudioListener || console.warn("Attempting to use attachAudioListener without support for output audio. Is build dep \":gl_graph_runner_audio_out\" missing?"), xs(this, e, ((e, n) => {
			e = new Float32Array(e.buffer, e.byteOffset, e.length / 4), t(e, n);
		})), W(this, e, ((e) => {
			this.i._attachAudioListener(e, n || !1);
		}));
	}
	finishProcessing() {
		this.i._waitUntilIdle();
	}
	closeGraph() {
		this.i._closeGraph(), this.i.simpleListeners = void 0, this.i.emptyPacketListeners = void 0;
	}
}, class extends _c {
	get ga() {
		return this.i;
	}
	pa(e, t, n) {
		W(this, t, ((t) => {
			let [r, i] = ys(this, e, t);
			this.ga._addBoundTextureAsImageToStream(t, r, i, n);
		}));
	}
	Z(e, t) {
		xs(this, e, t), W(this, e, ((e) => {
			this.ga._attachImageListener(e);
		}));
	}
	aa(e, t) {
		Ss(this, e, t), W(this, e, ((e) => {
			this.ga._attachImageVectorListener(e);
		}));
	}
})), _c, vc = class extends gc {};
async function X(e, t, n) {
	return async function(e, t, n, r) {
		return Cs(e, t, n, r);
	}(e, n.canvas ?? (gs() ? void 0 : document.createElement("canvas")), t, n);
}
function yc(e, t, n, r) {
	if (e.U) {
		let a = new qa();
		if (n?.regionOfInterest) {
			if (!e.oa) throw Error("This task doesn't support region-of-interest.");
			var i = n.regionOfInterest;
			if (i.left >= i.right || i.top >= i.bottom) throw Error("Expected RectF with left < right and top < bottom.");
			if (i.left < 0 || i.top < 0 || i.right > 1 || i.bottom > 1) throw Error("Expected RectF values to be in [0,1].");
			w(a, 1, (i.left + i.right) / 2), w(a, 2, (i.top + i.bottom) / 2), w(a, 4, i.right - i.left), w(a, 3, i.bottom - i.top);
		} else w(a, 1, .5), w(a, 2, .5), w(a, 4, 1), w(a, 3, 1);
		if (n?.rotationDegrees) {
			if (n?.rotationDegrees % 90 != 0) throw Error("Expected rotation to be a multiple of 90°.");
			if (w(a, 5, -Math.PI * n.rotationDegrees / 180), n?.rotationDegrees % 180 != 0) {
				let [e, r] = vs(t);
				n = C(a, 3) * r / e, i = C(a, 4) * e / r, w(a, 4, n), w(a, 3, i);
			}
		}
		e.g.addProtoToStream(a.g(), "mediapipe.NormalizedRect", e.U, r);
	}
	e.g.pa(t, e.X, r ?? performance.now()), e.finishProcessing();
}
function bc(e, t, n) {
	if (e.baseOptions?.g()) throw Error("Task is not initialized with image mode. 'runningMode' must be set to 'IMAGE'.");
	yc(e, t, n, e.C + 1);
}
function xc(e, t, n, r) {
	if (!e.baseOptions?.g()) throw Error("Task is not initialized with video mode. 'runningMode' must be set to 'VIDEO'.");
	yc(e, t, n, r);
}
function Sc(e, t, n, r) {
	var i = t.data;
	let a = t.width, o = a * (t = t.height);
	if ((i instanceof Uint8Array || i instanceof Float32Array) && i.length !== o) throw Error("Unsupported channel count: " + i.length / o);
	return e = new q([i], n, !1, e.g.i.canvas, e.P, a, t), r ? e.clone() : e;
}
var Cc = class extends ks {
	constructor(e, t, n, r) {
		super(e), this.g = e, this.X = t, this.U = n, this.oa = r, this.P = new Rs();
	}
	l(e, t = !0) {
		if ("runningMode" in e && y(this.baseOptions, 2, Vt(!!e.runningMode && e.runningMode !== "IMAGE")), e.canvas !== void 0 && this.g.i.canvas !== e.canvas) throw Error("You must create a new task to reset the canvas.");
		return super.l(e, t);
	}
	close() {
		this.P.close(), super.close();
	}
};
Cc.prototype.close = Cc.prototype.close;
var wc = class extends Cc {
	constructor(e, t) {
		super(new vc(e, t), "image_in", "norm_rect_in", !1), this.j = { detections: [] }, S(e = this.h = new lo(), 0, 1, t = new H()), w(this.h, 2, .5), w(this.h, 3, .3);
	}
	get baseOptions() {
		return x(this.h, H, 1);
	}
	set baseOptions(e) {
		S(this.h, 0, 1, e);
	}
	o(e) {
		return "minDetectionConfidence" in e && w(this.h, 2, e.minDetectionConfidence ?? .5), "minSuppressionThreshold" in e && w(this.h, 3, e.minSuppressionThreshold ?? .3), this.l(e);
	}
	F(e, t) {
		return this.j = { detections: [] }, bc(this, e, t), this.j;
	}
	G(e, t, n) {
		return this.j = { detections: [] }, xc(this, e, n, t), this.j;
	}
	m() {
		var e = new Aa();
		B(e, "image_in"), B(e, "norm_rect_in"), V(e, "detections");
		let t = new wa();
		li(t, fo, this.h);
		let n = new z();
		dr(n, 2, "mediapipe.tasks.vision.face_detector.FaceDetectorGraph"), L(n, "IMAGE:image_in"), L(n, "NORM_RECT:norm_rect_in"), R(n, "DETECTIONS:detections"), n.o(t), ka(e, n), this.g.attachProtoVectorListener("detections", ((e, t) => {
			for (let t of e) e = Ba(t), this.j.detections.push(os(e));
			G(this, t);
		})), this.g.attachEmptyPacketListener("detections", ((e) => {
			G(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
wc.prototype.detectForVideo = wc.prototype.G, wc.prototype.detect = wc.prototype.F, wc.prototype.setOptions = wc.prototype.o, wc.createFromModelPath = async function(e, t) {
	return X(wc, e, { baseOptions: { modelAssetPath: t } });
}, wc.createFromModelBuffer = function(e, t) {
	return X(wc, e, { baseOptions: { modelAssetBuffer: t } });
}, wc.createFromOptions = function(e, t) {
	return X(wc, e, t);
};
var Tc = hc([61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314], [314, 405], [405, 321], [321, 375], [375, 291], [61, 185], [185, 40], [40, 39], [39, 37], [37, 0], [0, 267], [267, 269], [269, 270], [270, 409], [409, 291], [78, 95], [95, 88], [88, 178], [178, 87], [87, 14], [14, 317], [317, 402], [402, 318], [318, 324], [324, 308], [78, 191], [191, 80], [80, 81], [81, 82], [82, 13], [13, 312], [312, 311], [311, 310], [310, 415], [415, 308]), Ec = hc([263, 249], [249, 390], [390, 373], [373, 374], [374, 380], [380, 381], [381, 382], [382, 362], [263, 466], [466, 388], [388, 387], [387, 386], [386, 385], [385, 384], [384, 398], [398, 362]), Dc = hc([276, 283], [283, 282], [282, 295], [295, 285], [300, 293], [293, 334], [334, 296], [296, 336]), Oc = hc([474, 475], [475, 476], [476, 477], [477, 474]), kc = hc([33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155], [155, 133], [33, 246], [246, 161], [161, 160], [160, 159], [159, 158], [158, 157], [157, 173], [173, 133]), Ac = hc([46, 53], [53, 52], [52, 65], [65, 55], [70, 63], [63, 105], [105, 66], [66, 107]), jc = hc([469, 470], [470, 471], [471, 472], [472, 469]), Mc = hc([10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397], [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152], [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172], [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162], [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10]), Nc = [
	...Tc,
	...Ec,
	...Dc,
	...kc,
	...Ac,
	...Mc
], Pc = hc([127, 34], [34, 139], [139, 127], [11, 0], [0, 37], [37, 11], [232, 231], [231, 120], [120, 232], [72, 37], [37, 39], [39, 72], [128, 121], [121, 47], [47, 128], [232, 121], [121, 128], [128, 232], [104, 69], [69, 67], [67, 104], [175, 171], [171, 148], [148, 175], [118, 50], [50, 101], [101, 118], [73, 39], [39, 40], [40, 73], [9, 151], [151, 108], [108, 9], [48, 115], [115, 131], [131, 48], [194, 204], [204, 211], [211, 194], [74, 40], [40, 185], [185, 74], [80, 42], [42, 183], [183, 80], [40, 92], [92, 186], [186, 40], [230, 229], [229, 118], [118, 230], [202, 212], [212, 214], [214, 202], [83, 18], [18, 17], [17, 83], [76, 61], [61, 146], [146, 76], [160, 29], [29, 30], [30, 160], [56, 157], [157, 173], [173, 56], [106, 204], [204, 194], [194, 106], [135, 214], [214, 192], [192, 135], [203, 165], [165, 98], [98, 203], [21, 71], [71, 68], [68, 21], [51, 45], [45, 4], [4, 51], [144, 24], [24, 23], [23, 144], [77, 146], [146, 91], [91, 77], [205, 50], [50, 187], [187, 205], [201, 200], [200, 18], [18, 201], [91, 106], [106, 182], [182, 91], [90, 91], [91, 181], [181, 90], [85, 84], [84, 17], [17, 85], [206, 203], [203, 36], [36, 206], [148, 171], [171, 140], [140, 148], [92, 40], [40, 39], [39, 92], [193, 189], [189, 244], [244, 193], [159, 158], [158, 28], [28, 159], [247, 246], [246, 161], [161, 247], [236, 3], [3, 196], [196, 236], [54, 68], [68, 104], [104, 54], [193, 168], [168, 8], [8, 193], [117, 228], [228, 31], [31, 117], [189, 193], [193, 55], [55, 189], [98, 97], [97, 99], [99, 98], [126, 47], [47, 100], [100, 126], [166, 79], [79, 218], [218, 166], [155, 154], [154, 26], [26, 155], [209, 49], [49, 131], [131, 209], [135, 136], [136, 150], [150, 135], [47, 126], [126, 217], [217, 47], [223, 52], [52, 53], [53, 223], [45, 51], [51, 134], [134, 45], [211, 170], [170, 140], [140, 211], [67, 69], [69, 108], [108, 67], [43, 106], [106, 91], [91, 43], [230, 119], [119, 120], [120, 230], [226, 130], [130, 247], [247, 226], [63, 53], [53, 52], [52, 63], [238, 20], [20, 242], [242, 238], [46, 70], [70, 156], [156, 46], [78, 62], [62, 96], [96, 78], [46, 53], [53, 63], [63, 46], [143, 34], [34, 227], [227, 143], [123, 117], [117, 111], [111, 123], [44, 125], [125, 19], [19, 44], [236, 134], [134, 51], [51, 236], [216, 206], [206, 205], [205, 216], [154, 153], [153, 22], [22, 154], [39, 37], [37, 167], [167, 39], [200, 201], [201, 208], [208, 200], [36, 142], [142, 100], [100, 36], [57, 212], [212, 202], [202, 57], [20, 60], [60, 99], [99, 20], [28, 158], [158, 157], [157, 28], [35, 226], [226, 113], [113, 35], [160, 159], [159, 27], [27, 160], [204, 202], [202, 210], [210, 204], [113, 225], [225, 46], [46, 113], [43, 202], [202, 204], [204, 43], [62, 76], [76, 77], [77, 62], [137, 123], [123, 116], [116, 137], [41, 38], [38, 72], [72, 41], [203, 129], [129, 142], [142, 203], [64, 98], [98, 240], [240, 64], [49, 102], [102, 64], [64, 49], [41, 73], [73, 74], [74, 41], [212, 216], [216, 207], [207, 212], [42, 74], [74, 184], [184, 42], [169, 170], [170, 211], [211, 169], [170, 149], [149, 176], [176, 170], [105, 66], [66, 69], [69, 105], [122, 6], [6, 168], [168, 122], [123, 147], [147, 187], [187, 123], [96, 77], [77, 90], [90, 96], [65, 55], [55, 107], [107, 65], [89, 90], [90, 180], [180, 89], [101, 100], [100, 120], [120, 101], [63, 105], [105, 104], [104, 63], [93, 137], [137, 227], [227, 93], [15, 86], [86, 85], [85, 15], [129, 102], [102, 49], [49, 129], [14, 87], [87, 86], [86, 14], [55, 8], [8, 9], [9, 55], [100, 47], [47, 121], [121, 100], [145, 23], [23, 22], [22, 145], [88, 89], [89, 179], [179, 88], [6, 122], [122, 196], [196, 6], [88, 95], [95, 96], [96, 88], [138, 172], [172, 136], [136, 138], [215, 58], [58, 172], [172, 215], [115, 48], [48, 219], [219, 115], [42, 80], [80, 81], [81, 42], [195, 3], [3, 51], [51, 195], [43, 146], [146, 61], [61, 43], [171, 175], [175, 199], [199, 171], [81, 82], [82, 38], [38, 81], [53, 46], [46, 225], [225, 53], [144, 163], [163, 110], [110, 144], [52, 65], [65, 66], [66, 52], [229, 228], [228, 117], [117, 229], [34, 127], [127, 234], [234, 34], [107, 108], [108, 69], [69, 107], [109, 108], [108, 151], [151, 109], [48, 64], [64, 235], [235, 48], [62, 78], [78, 191], [191, 62], [129, 209], [209, 126], [126, 129], [111, 35], [35, 143], [143, 111], [117, 123], [123, 50], [50, 117], [222, 65], [65, 52], [52, 222], [19, 125], [125, 141], [141, 19], [221, 55], [55, 65], [65, 221], [3, 195], [195, 197], [197, 3], [25, 7], [7, 33], [33, 25], [220, 237], [237, 44], [44, 220], [70, 71], [71, 139], [139, 70], [122, 193], [193, 245], [245, 122], [247, 130], [130, 33], [33, 247], [71, 21], [21, 162], [162, 71], [170, 169], [169, 150], [150, 170], [188, 174], [174, 196], [196, 188], [216, 186], [186, 92], [92, 216], [2, 97], [97, 167], [167, 2], [141, 125], [125, 241], [241, 141], [164, 167], [167, 37], [37, 164], [72, 38], [38, 12], [12, 72], [38, 82], [82, 13], [13, 38], [63, 68], [68, 71], [71, 63], [226, 35], [35, 111], [111, 226], [101, 50], [50, 205], [205, 101], [206, 92], [92, 165], [165, 206], [209, 198], [198, 217], [217, 209], [165, 167], [167, 97], [97, 165], [220, 115], [115, 218], [218, 220], [133, 112], [112, 243], [243, 133], [239, 238], [238, 241], [241, 239], [214, 135], [135, 169], [169, 214], [190, 173], [173, 133], [133, 190], [171, 208], [208, 32], [32, 171], [125, 44], [44, 237], [237, 125], [86, 87], [87, 178], [178, 86], [85, 86], [86, 179], [179, 85], [84, 85], [85, 180], [180, 84], [83, 84], [84, 181], [181, 83], [201, 83], [83, 182], [182, 201], [137, 93], [93, 132], [132, 137], [76, 62], [62, 183], [183, 76], [61, 76], [76, 184], [184, 61], [57, 61], [61, 185], [185, 57], [212, 57], [57, 186], [186, 212], [214, 207], [207, 187], [187, 214], [34, 143], [143, 156], [156, 34], [79, 239], [239, 237], [237, 79], [123, 137], [137, 177], [177, 123], [44, 1], [1, 4], [4, 44], [201, 194], [194, 32], [32, 201], [64, 102], [102, 129], [129, 64], [213, 215], [215, 138], [138, 213], [59, 166], [166, 219], [219, 59], [242, 99], [99, 97], [97, 242], [2, 94], [94, 141], [141, 2], [75, 59], [59, 235], [235, 75], [24, 110], [110, 228], [228, 24], [25, 130], [130, 226], [226, 25], [23, 24], [24, 229], [229, 23], [22, 23], [23, 230], [230, 22], [26, 22], [22, 231], [231, 26], [112, 26], [26, 232], [232, 112], [189, 190], [190, 243], [243, 189], [221, 56], [56, 190], [190, 221], [28, 56], [56, 221], [221, 28], [27, 28], [28, 222], [222, 27], [29, 27], [27, 223], [223, 29], [30, 29], [29, 224], [224, 30], [247, 30], [30, 225], [225, 247], [238, 79], [79, 20], [20, 238], [166, 59], [59, 75], [75, 166], [60, 75], [75, 240], [240, 60], [147, 177], [177, 215], [215, 147], [20, 79], [79, 166], [166, 20], [187, 147], [147, 213], [213, 187], [112, 233], [233, 244], [244, 112], [233, 128], [128, 245], [245, 233], [128, 114], [114, 188], [188, 128], [114, 217], [217, 174], [174, 114], [131, 115], [115, 220], [220, 131], [217, 198], [198, 236], [236, 217], [198, 131], [131, 134], [134, 198], [177, 132], [132, 58], [58, 177], [143, 35], [35, 124], [124, 143], [110, 163], [163, 7], [7, 110], [228, 110], [110, 25], [25, 228], [356, 389], [389, 368], [368, 356], [11, 302], [302, 267], [267, 11], [452, 350], [350, 349], [349, 452], [302, 303], [303, 269], [269, 302], [357, 343], [343, 277], [277, 357], [452, 453], [453, 357], [357, 452], [333, 332], [332, 297], [297, 333], [175, 152], [152, 377], [377, 175], [347, 348], [348, 330], [330, 347], [303, 304], [304, 270], [270, 303], [9, 336], [336, 337], [337, 9], [278, 279], [279, 360], [360, 278], [418, 262], [262, 431], [431, 418], [304, 408], [408, 409], [409, 304], [310, 415], [415, 407], [407, 310], [270, 409], [409, 410], [410, 270], [450, 348], [348, 347], [347, 450], [422, 430], [430, 434], [434, 422], [313, 314], [314, 17], [17, 313], [306, 307], [307, 375], [375, 306], [387, 388], [388, 260], [260, 387], [286, 414], [414, 398], [398, 286], [335, 406], [406, 418], [418, 335], [364, 367], [367, 416], [416, 364], [423, 358], [358, 327], [327, 423], [251, 284], [284, 298], [298, 251], [281, 5], [5, 4], [4, 281], [373, 374], [374, 253], [253, 373], [307, 320], [320, 321], [321, 307], [425, 427], [427, 411], [411, 425], [421, 313], [313, 18], [18, 421], [321, 405], [405, 406], [406, 321], [320, 404], [404, 405], [405, 320], [315, 16], [16, 17], [17, 315], [426, 425], [425, 266], [266, 426], [377, 400], [400, 369], [369, 377], [322, 391], [391, 269], [269, 322], [417, 465], [465, 464], [464, 417], [386, 257], [257, 258], [258, 386], [466, 260], [260, 388], [388, 466], [456, 399], [399, 419], [419, 456], [284, 332], [332, 333], [333, 284], [417, 285], [285, 8], [8, 417], [346, 340], [340, 261], [261, 346], [413, 441], [441, 285], [285, 413], [327, 460], [460, 328], [328, 327], [355, 371], [371, 329], [329, 355], [392, 439], [439, 438], [438, 392], [382, 341], [341, 256], [256, 382], [429, 420], [420, 360], [360, 429], [364, 394], [394, 379], [379, 364], [277, 343], [343, 437], [437, 277], [443, 444], [444, 283], [283, 443], [275, 440], [440, 363], [363, 275], [431, 262], [262, 369], [369, 431], [297, 338], [338, 337], [337, 297], [273, 375], [375, 321], [321, 273], [450, 451], [451, 349], [349, 450], [446, 342], [342, 467], [467, 446], [293, 334], [334, 282], [282, 293], [458, 461], [461, 462], [462, 458], [276, 353], [353, 383], [383, 276], [308, 324], [324, 325], [325, 308], [276, 300], [300, 293], [293, 276], [372, 345], [345, 447], [447, 372], [352, 345], [345, 340], [340, 352], [274, 1], [1, 19], [19, 274], [456, 248], [248, 281], [281, 456], [436, 427], [427, 425], [425, 436], [381, 256], [256, 252], [252, 381], [269, 391], [391, 393], [393, 269], [200, 199], [199, 428], [428, 200], [266, 330], [330, 329], [329, 266], [287, 273], [273, 422], [422, 287], [250, 462], [462, 328], [328, 250], [258, 286], [286, 384], [384, 258], [265, 353], [353, 342], [342, 265], [387, 259], [259, 257], [257, 387], [424, 431], [431, 430], [430, 424], [342, 353], [353, 276], [276, 342], [273, 335], [335, 424], [424, 273], [292, 325], [325, 307], [307, 292], [366, 447], [447, 345], [345, 366], [271, 303], [303, 302], [302, 271], [423, 266], [266, 371], [371, 423], [294, 455], [455, 460], [460, 294], [279, 278], [278, 294], [294, 279], [271, 272], [272, 304], [304, 271], [432, 434], [434, 427], [427, 432], [272, 407], [407, 408], [408, 272], [394, 430], [430, 431], [431, 394], [395, 369], [369, 400], [400, 395], [334, 333], [333, 299], [299, 334], [351, 417], [417, 168], [168, 351], [352, 280], [280, 411], [411, 352], [325, 319], [319, 320], [320, 325], [295, 296], [296, 336], [336, 295], [319, 403], [403, 404], [404, 319], [330, 348], [348, 349], [349, 330], [293, 298], [298, 333], [333, 293], [323, 454], [454, 447], [447, 323], [15, 16], [16, 315], [315, 15], [358, 429], [429, 279], [279, 358], [14, 15], [15, 316], [316, 14], [285, 336], [336, 9], [9, 285], [329, 349], [349, 350], [350, 329], [374, 380], [380, 252], [252, 374], [318, 402], [402, 403], [403, 318], [6, 197], [197, 419], [419, 6], [318, 319], [319, 325], [325, 318], [367, 364], [364, 365], [365, 367], [435, 367], [367, 397], [397, 435], [344, 438], [438, 439], [439, 344], [272, 271], [271, 311], [311, 272], [195, 5], [5, 281], [281, 195], [273, 287], [287, 291], [291, 273], [396, 428], [428, 199], [199, 396], [311, 271], [271, 268], [268, 311], [283, 444], [444, 445], [445, 283], [373, 254], [254, 339], [339, 373], [282, 334], [334, 296], [296, 282], [449, 347], [347, 346], [346, 449], [264, 447], [447, 454], [454, 264], [336, 296], [296, 299], [299, 336], [338, 10], [10, 151], [151, 338], [278, 439], [439, 455], [455, 278], [292, 407], [407, 415], [415, 292], [358, 371], [371, 355], [355, 358], [340, 345], [345, 372], [372, 340], [346, 347], [347, 280], [280, 346], [442, 443], [443, 282], [282, 442], [19, 94], [94, 370], [370, 19], [441, 442], [442, 295], [295, 441], [248, 419], [419, 197], [197, 248], [263, 255], [255, 359], [359, 263], [440, 275], [275, 274], [274, 440], [300, 383], [383, 368], [368, 300], [351, 412], [412, 465], [465, 351], [263, 467], [467, 466], [466, 263], [301, 368], [368, 389], [389, 301], [395, 378], [378, 379], [379, 395], [412, 351], [351, 419], [419, 412], [436, 426], [426, 322], [322, 436], [2, 164], [164, 393], [393, 2], [370, 462], [462, 461], [461, 370], [164, 0], [0, 267], [267, 164], [302, 11], [11, 12], [12, 302], [268, 12], [12, 13], [13, 268], [293, 300], [300, 301], [301, 293], [446, 261], [261, 340], [340, 446], [330, 266], [266, 425], [425, 330], [426, 423], [423, 391], [391, 426], [429, 355], [355, 437], [437, 429], [391, 327], [327, 326], [326, 391], [440, 457], [457, 438], [438, 440], [341, 382], [382, 362], [362, 341], [459, 457], [457, 461], [461, 459], [434, 430], [430, 394], [394, 434], [414, 463], [463, 362], [362, 414], [396, 369], [369, 262], [262, 396], [354, 461], [461, 457], [457, 354], [316, 403], [403, 402], [402, 316], [315, 404], [404, 403], [403, 315], [314, 405], [405, 404], [404, 314], [313, 406], [406, 405], [405, 313], [421, 418], [418, 406], [406, 421], [366, 401], [401, 361], [361, 366], [306, 408], [408, 407], [407, 306], [291, 409], [409, 408], [408, 291], [287, 410], [410, 409], [409, 287], [432, 436], [436, 410], [410, 432], [434, 416], [416, 411], [411, 434], [264, 368], [368, 383], [383, 264], [309, 438], [438, 457], [457, 309], [352, 376], [376, 401], [401, 352], [274, 275], [275, 4], [4, 274], [421, 428], [428, 262], [262, 421], [294, 327], [327, 358], [358, 294], [433, 416], [416, 367], [367, 433], [289, 455], [455, 439], [439, 289], [462, 370], [370, 326], [326, 462], [2, 326], [326, 370], [370, 2], [305, 460], [460, 455], [455, 305], [254, 449], [449, 448], [448, 254], [255, 261], [261, 446], [446, 255], [253, 450], [450, 449], [449, 253], [252, 451], [451, 450], [450, 252], [256, 452], [452, 451], [451, 256], [341, 453], [453, 452], [452, 341], [413, 464], [464, 463], [463, 413], [441, 413], [413, 414], [414, 441], [258, 442], [442, 441], [441, 258], [257, 443], [443, 442], [442, 257], [259, 444], [444, 443], [443, 259], [260, 445], [445, 444], [444, 260], [467, 342], [342, 445], [445, 467], [459, 458], [458, 250], [250, 459], [289, 392], [392, 290], [290, 289], [290, 328], [328, 460], [460, 290], [376, 433], [433, 435], [435, 376], [250, 290], [290, 392], [392, 250], [411, 416], [416, 433], [433, 411], [341, 463], [463, 464], [464, 341], [453, 464], [464, 465], [465, 453], [357, 465], [465, 412], [412, 357], [343, 412], [412, 399], [399, 343], [360, 363], [363, 440], [440, 360], [437, 399], [399, 456], [456, 437], [420, 456], [456, 363], [363, 420], [401, 435], [435, 288], [288, 401], [372, 383], [383, 353], [353, 372], [339, 255], [255, 249], [249, 339], [448, 261], [261, 255], [255, 448], [133, 243], [243, 190], [190, 133], [133, 155], [155, 112], [112, 133], [33, 246], [246, 247], [247, 33], [33, 130], [130, 25], [25, 33], [398, 384], [384, 286], [286, 398], [362, 398], [398, 414], [414, 362], [362, 463], [463, 341], [341, 362], [263, 359], [359, 467], [467, 263], [263, 249], [249, 255], [255, 263], [466, 467], [467, 260], [260, 466], [75, 60], [60, 166], [166, 75], [238, 239], [239, 79], [79, 238], [162, 127], [127, 139], [139, 162], [72, 11], [11, 37], [37, 72], [121, 232], [232, 120], [120, 121], [73, 72], [72, 39], [39, 73], [114, 128], [128, 47], [47, 114], [233, 232], [232, 128], [128, 233], [103, 104], [104, 67], [67, 103], [152, 175], [175, 148], [148, 152], [119, 118], [118, 101], [101, 119], [74, 73], [73, 40], [40, 74], [107, 9], [9, 108], [108, 107], [49, 48], [48, 131], [131, 49], [32, 194], [194, 211], [211, 32], [184, 74], [74, 185], [185, 184], [191, 80], [80, 183], [183, 191], [185, 40], [40, 186], [186, 185], [119, 230], [230, 118], [118, 119], [210, 202], [202, 214], [214, 210], [84, 83], [83, 17], [17, 84], [77, 76], [76, 146], [146, 77], [161, 160], [160, 30], [30, 161], [190, 56], [56, 173], [173, 190], [182, 106], [106, 194], [194, 182], [138, 135], [135, 192], [192, 138], [129, 203], [203, 98], [98, 129], [54, 21], [21, 68], [68, 54], [5, 51], [51, 4], [4, 5], [145, 144], [144, 23], [23, 145], [90, 77], [77, 91], [91, 90], [207, 205], [205, 187], [187, 207], [83, 201], [201, 18], [18, 83], [181, 91], [91, 182], [182, 181], [180, 90], [90, 181], [181, 180], [16, 85], [85, 17], [17, 16], [205, 206], [206, 36], [36, 205], [176, 148], [148, 140], [140, 176], [165, 92], [92, 39], [39, 165], [245, 193], [193, 244], [244, 245], [27, 159], [159, 28], [28, 27], [30, 247], [247, 161], [161, 30], [174, 236], [236, 196], [196, 174], [103, 54], [54, 104], [104, 103], [55, 193], [193, 8], [8, 55], [111, 117], [117, 31], [31, 111], [221, 189], [189, 55], [55, 221], [240, 98], [98, 99], [99, 240], [142, 126], [126, 100], [100, 142], [219, 166], [166, 218], [218, 219], [112, 155], [155, 26], [26, 112], [198, 209], [209, 131], [131, 198], [169, 135], [135, 150], [150, 169], [114, 47], [47, 217], [217, 114], [224, 223], [223, 53], [53, 224], [220, 45], [45, 134], [134, 220], [32, 211], [211, 140], [140, 32], [109, 67], [67, 108], [108, 109], [146, 43], [43, 91], [91, 146], [231, 230], [230, 120], [120, 231], [113, 226], [226, 247], [247, 113], [105, 63], [63, 52], [52, 105], [241, 238], [238, 242], [242, 241], [124, 46], [46, 156], [156, 124], [95, 78], [78, 96], [96, 95], [70, 46], [46, 63], [63, 70], [116, 143], [143, 227], [227, 116], [116, 123], [123, 111], [111, 116], [1, 44], [44, 19], [19, 1], [3, 236], [236, 51], [51, 3], [207, 216], [216, 205], [205, 207], [26, 154], [154, 22], [22, 26], [165, 39], [39, 167], [167, 165], [199, 200], [200, 208], [208, 199], [101, 36], [36, 100], [100, 101], [43, 57], [57, 202], [202, 43], [242, 20], [20, 99], [99, 242], [56, 28], [28, 157], [157, 56], [124, 35], [35, 113], [113, 124], [29, 160], [160, 27], [27, 29], [211, 204], [204, 210], [210, 211], [124, 113], [113, 46], [46, 124], [106, 43], [43, 204], [204, 106], [96, 62], [62, 77], [77, 96], [227, 137], [137, 116], [116, 227], [73, 41], [41, 72], [72, 73], [36, 203], [203, 142], [142, 36], [235, 64], [64, 240], [240, 235], [48, 49], [49, 64], [64, 48], [42, 41], [41, 74], [74, 42], [214, 212], [212, 207], [207, 214], [183, 42], [42, 184], [184, 183], [210, 169], [169, 211], [211, 210], [140, 170], [170, 176], [176, 140], [104, 105], [105, 69], [69, 104], [193, 122], [122, 168], [168, 193], [50, 123], [123, 187], [187, 50], [89, 96], [96, 90], [90, 89], [66, 65], [65, 107], [107, 66], [179, 89], [89, 180], [180, 179], [119, 101], [101, 120], [120, 119], [68, 63], [63, 104], [104, 68], [234, 93], [93, 227], [227, 234], [16, 15], [15, 85], [85, 16], [209, 129], [129, 49], [49, 209], [15, 14], [14, 86], [86, 15], [107, 55], [55, 9], [9, 107], [120, 100], [100, 121], [121, 120], [153, 145], [145, 22], [22, 153], [178, 88], [88, 179], [179, 178], [197, 6], [6, 196], [196, 197], [89, 88], [88, 96], [96, 89], [135, 138], [138, 136], [136, 135], [138, 215], [215, 172], [172, 138], [218, 115], [115, 219], [219, 218], [41, 42], [42, 81], [81, 41], [5, 195], [195, 51], [51, 5], [57, 43], [43, 61], [61, 57], [208, 171], [171, 199], [199, 208], [41, 81], [81, 38], [38, 41], [224, 53], [53, 225], [225, 224], [24, 144], [144, 110], [110, 24], [105, 52], [52, 66], [66, 105], [118, 229], [229, 117], [117, 118], [227, 34], [34, 234], [234, 227], [66, 107], [107, 69], [69, 66], [10, 109], [109, 151], [151, 10], [219, 48], [48, 235], [235, 219], [183, 62], [62, 191], [191, 183], [142, 129], [129, 126], [126, 142], [116, 111], [111, 143], [143, 116], [118, 117], [117, 50], [50, 118], [223, 222], [222, 52], [52, 223], [94, 19], [19, 141], [141, 94], [222, 221], [221, 65], [65, 222], [196, 3], [3, 197], [197, 196], [45, 220], [220, 44], [44, 45], [156, 70], [70, 139], [139, 156], [188, 122], [122, 245], [245, 188], [139, 71], [71, 162], [162, 139], [149, 170], [170, 150], [150, 149], [122, 188], [188, 196], [196, 122], [206, 216], [216, 92], [92, 206], [164, 2], [2, 167], [167, 164], [242, 141], [141, 241], [241, 242], [0, 164], [164, 37], [37, 0], [11, 72], [72, 12], [12, 11], [12, 38], [38, 13], [13, 12], [70, 63], [63, 71], [71, 70], [31, 226], [226, 111], [111, 31], [36, 101], [101, 205], [205, 36], [203, 206], [206, 165], [165, 203], [126, 209], [209, 217], [217, 126], [98, 165], [165, 97], [97, 98], [237, 220], [220, 218], [218, 237], [237, 239], [239, 241], [241, 237], [210, 214], [214, 169], [169, 210], [140, 171], [171, 32], [32, 140], [241, 125], [125, 237], [237, 241], [179, 86], [86, 178], [178, 179], [180, 85], [85, 179], [179, 180], [181, 84], [84, 180], [180, 181], [182, 83], [83, 181], [181, 182], [194, 201], [201, 182], [182, 194], [177, 137], [137, 132], [132, 177], [184, 76], [76, 183], [183, 184], [185, 61], [61, 184], [184, 185], [186, 57], [57, 185], [185, 186], [216, 212], [212, 186], [186, 216], [192, 214], [214, 187], [187, 192], [139, 34], [34, 156], [156, 139], [218, 79], [79, 237], [237, 218], [147, 123], [123, 177], [177, 147], [45, 44], [44, 4], [4, 45], [208, 201], [201, 32], [32, 208], [98, 64], [64, 129], [129, 98], [192, 213], [213, 138], [138, 192], [235, 59], [59, 219], [219, 235], [141, 242], [242, 97], [97, 141], [97, 2], [2, 141], [141, 97], [240, 75], [75, 235], [235, 240], [229, 24], [24, 228], [228, 229], [31, 25], [25, 226], [226, 31], [230, 23], [23, 229], [229, 230], [231, 22], [22, 230], [230, 231], [232, 26], [26, 231], [231, 232], [233, 112], [112, 232], [232, 233], [244, 189], [189, 243], [243, 244], [189, 221], [221, 190], [190, 189], [222, 28], [28, 221], [221, 222], [223, 27], [27, 222], [222, 223], [224, 29], [29, 223], [223, 224], [225, 30], [30, 224], [224, 225], [113, 247], [247, 225], [225, 113], [99, 60], [60, 240], [240, 99], [213, 147], [147, 215], [215, 213], [60, 20], [20, 166], [166, 60], [192, 187], [187, 213], [213, 192], [243, 112], [112, 244], [244, 243], [244, 233], [233, 245], [245, 244], [245, 128], [128, 188], [188, 245], [188, 114], [114, 174], [174, 188], [134, 131], [131, 220], [220, 134], [174, 217], [217, 236], [236, 174], [236, 198], [198, 134], [134, 236], [215, 177], [177, 58], [58, 215], [156, 143], [143, 124], [124, 156], [25, 110], [110, 7], [7, 25], [31, 228], [228, 25], [25, 31], [264, 356], [356, 368], [368, 264], [0, 11], [11, 267], [267, 0], [451, 452], [452, 349], [349, 451], [267, 302], [302, 269], [269, 267], [350, 357], [357, 277], [277, 350], [350, 452], [452, 357], [357, 350], [299, 333], [333, 297], [297, 299], [396, 175], [175, 377], [377, 396], [280, 347], [347, 330], [330, 280], [269, 303], [303, 270], [270, 269], [151, 9], [9, 337], [337, 151], [344, 278], [278, 360], [360, 344], [424, 418], [418, 431], [431, 424], [270, 304], [304, 409], [409, 270], [272, 310], [310, 407], [407, 272], [322, 270], [270, 410], [410, 322], [449, 450], [450, 347], [347, 449], [432, 422], [422, 434], [434, 432], [18, 313], [313, 17], [17, 18], [291, 306], [306, 375], [375, 291], [259, 387], [387, 260], [260, 259], [424, 335], [335, 418], [418, 424], [434, 364], [364, 416], [416, 434], [391, 423], [423, 327], [327, 391], [301, 251], [251, 298], [298, 301], [275, 281], [281, 4], [4, 275], [254, 373], [373, 253], [253, 254], [375, 307], [307, 321], [321, 375], [280, 425], [425, 411], [411, 280], [200, 421], [421, 18], [18, 200], [335, 321], [321, 406], [406, 335], [321, 320], [320, 405], [405, 321], [314, 315], [315, 17], [17, 314], [423, 426], [426, 266], [266, 423], [396, 377], [377, 369], [369, 396], [270, 322], [322, 269], [269, 270], [413, 417], [417, 464], [464, 413], [385, 386], [386, 258], [258, 385], [248, 456], [456, 419], [419, 248], [298, 284], [284, 333], [333, 298], [168, 417], [417, 8], [8, 168], [448, 346], [346, 261], [261, 448], [417, 413], [413, 285], [285, 417], [326, 327], [327, 328], [328, 326], [277, 355], [355, 329], [329, 277], [309, 392], [392, 438], [438, 309], [381, 382], [382, 256], [256, 381], [279, 429], [429, 360], [360, 279], [365, 364], [364, 379], [379, 365], [355, 277], [277, 437], [437, 355], [282, 443], [443, 283], [283, 282], [281, 275], [275, 363], [363, 281], [395, 431], [431, 369], [369, 395], [299, 297], [297, 337], [337, 299], [335, 273], [273, 321], [321, 335], [348, 450], [450, 349], [349, 348], [359, 446], [446, 467], [467, 359], [283, 293], [293, 282], [282, 283], [250, 458], [458, 462], [462, 250], [300, 276], [276, 383], [383, 300], [292, 308], [308, 325], [325, 292], [283, 276], [276, 293], [293, 283], [264, 372], [372, 447], [447, 264], [346, 352], [352, 340], [340, 346], [354, 274], [274, 19], [19, 354], [363, 456], [456, 281], [281, 363], [426, 436], [436, 425], [425, 426], [380, 381], [381, 252], [252, 380], [267, 269], [269, 393], [393, 267], [421, 200], [200, 428], [428, 421], [371, 266], [266, 329], [329, 371], [432, 287], [287, 422], [422, 432], [290, 250], [250, 328], [328, 290], [385, 258], [258, 384], [384, 385], [446, 265], [265, 342], [342, 446], [386, 387], [387, 257], [257, 386], [422, 424], [424, 430], [430, 422], [445, 342], [342, 276], [276, 445], [422, 273], [273, 424], [424, 422], [306, 292], [292, 307], [307, 306], [352, 366], [366, 345], [345, 352], [268, 271], [271, 302], [302, 268], [358, 423], [423, 371], [371, 358], [327, 294], [294, 460], [460, 327], [331, 279], [279, 294], [294, 331], [303, 271], [271, 304], [304, 303], [436, 432], [432, 427], [427, 436], [304, 272], [272, 408], [408, 304], [395, 394], [394, 431], [431, 395], [378, 395], [395, 400], [400, 378], [296, 334], [334, 299], [299, 296], [6, 351], [351, 168], [168, 6], [376, 352], [352, 411], [411, 376], [307, 325], [325, 320], [320, 307], [285, 295], [295, 336], [336, 285], [320, 319], [319, 404], [404, 320], [329, 330], [330, 349], [349, 329], [334, 293], [293, 333], [333, 334], [366, 323], [323, 447], [447, 366], [316, 15], [15, 315], [315, 316], [331, 358], [358, 279], [279, 331], [317, 14], [14, 316], [316, 317], [8, 285], [285, 9], [9, 8], [277, 329], [329, 350], [350, 277], [253, 374], [374, 252], [252, 253], [319, 318], [318, 403], [403, 319], [351, 6], [6, 419], [419, 351], [324, 318], [318, 325], [325, 324], [397, 367], [367, 365], [365, 397], [288, 435], [435, 397], [397, 288], [278, 344], [344, 439], [439, 278], [310, 272], [272, 311], [311, 310], [248, 195], [195, 281], [281, 248], [375, 273], [273, 291], [291, 375], [175, 396], [396, 199], [199, 175], [312, 311], [311, 268], [268, 312], [276, 283], [283, 445], [445, 276], [390, 373], [373, 339], [339, 390], [295, 282], [282, 296], [296, 295], [448, 449], [449, 346], [346, 448], [356, 264], [264, 454], [454, 356], [337, 336], [336, 299], [299, 337], [337, 338], [338, 151], [151, 337], [294, 278], [278, 455], [455, 294], [308, 292], [292, 415], [415, 308], [429, 358], [358, 355], [355, 429], [265, 340], [340, 372], [372, 265], [352, 346], [346, 280], [280, 352], [295, 442], [442, 282], [282, 295], [354, 19], [19, 370], [370, 354], [285, 441], [441, 295], [295, 285], [195, 248], [248, 197], [197, 195], [457, 440], [440, 274], [274, 457], [301, 300], [300, 368], [368, 301], [417, 351], [351, 465], [465, 417], [251, 301], [301, 389], [389, 251], [394, 395], [395, 379], [379, 394], [399, 412], [412, 419], [419, 399], [410, 436], [436, 322], [322, 410], [326, 2], [2, 393], [393, 326], [354, 370], [370, 461], [461, 354], [393, 164], [164, 267], [267, 393], [268, 302], [302, 12], [12, 268], [312, 268], [268, 13], [13, 312], [298, 293], [293, 301], [301, 298], [265, 446], [446, 340], [340, 265], [280, 330], [330, 425], [425, 280], [322, 426], [426, 391], [391, 322], [420, 429], [429, 437], [437, 420], [393, 391], [391, 326], [326, 393], [344, 440], [440, 438], [438, 344], [458, 459], [459, 461], [461, 458], [364, 434], [434, 394], [394, 364], [428, 396], [396, 262], [262, 428], [274, 354], [354, 457], [457, 274], [317, 316], [316, 402], [402, 317], [316, 315], [315, 403], [403, 316], [315, 314], [314, 404], [404, 315], [314, 313], [313, 405], [405, 314], [313, 421], [421, 406], [406, 313], [323, 366], [366, 361], [361, 323], [292, 306], [306, 407], [407, 292], [306, 291], [291, 408], [408, 306], [291, 287], [287, 409], [409, 291], [287, 432], [432, 410], [410, 287], [427, 434], [434, 411], [411, 427], [372, 264], [264, 383], [383, 372], [459, 309], [309, 457], [457, 459], [366, 352], [352, 401], [401, 366], [1, 274], [274, 4], [4, 1], [418, 421], [421, 262], [262, 418], [331, 294], [294, 358], [358, 331], [435, 433], [433, 367], [367, 435], [392, 289], [289, 439], [439, 392], [328, 462], [462, 326], [326, 328], [94, 2], [2, 370], [370, 94], [289, 305], [305, 455], [455, 289], [339, 254], [254, 448], [448, 339], [359, 255], [255, 446], [446, 359], [254, 253], [253, 449], [449, 254], [253, 252], [252, 450], [450, 253], [252, 256], [256, 451], [451, 252], [256, 341], [341, 452], [452, 256], [414, 413], [413, 463], [463, 414], [286, 441], [441, 414], [414, 286], [286, 258], [258, 441], [441, 286], [258, 257], [257, 442], [442, 258], [257, 259], [259, 443], [443, 257], [259, 260], [260, 444], [444, 259], [260, 467], [467, 445], [445, 260], [309, 459], [459, 250], [250, 309], [305, 289], [289, 290], [290, 305], [305, 290], [290, 460], [460, 305], [401, 376], [376, 435], [435, 401], [309, 250], [250, 392], [392, 309], [376, 411], [411, 433], [433, 376], [453, 341], [341, 464], [464, 453], [357, 453], [453, 465], [465, 357], [343, 357], [357, 412], [412, 343], [437, 343], [343, 399], [399, 437], [344, 360], [360, 440], [440, 344], [420, 437], [437, 456], [456, 420], [360, 420], [420, 363], [363, 360], [361, 401], [401, 288], [288, 361], [265, 372], [372, 353], [353, 265], [390, 339], [339, 249], [249, 390], [339, 448], [448, 255], [255, 339]);
function Fc(e) {
	e.j = {
		faceLandmarks: [],
		faceBlendshapes: [],
		facialTransformationMatrixes: []
	};
}
var Z = class extends Cc {
	constructor(e, t) {
		super(new vc(e, t), "image_in", "norm_rect", !1), this.j = {
			faceLandmarks: [],
			faceBlendshapes: [],
			facialTransformationMatrixes: []
		}, this.outputFacialTransformationMatrixes = this.outputFaceBlendshapes = !1, S(e = this.h = new ho(), 0, 1, t = new H()), this.A = new mo(), S(this.h, 0, 3, this.A), this.u = new lo(), S(this.h, 0, 2, this.u), ur(this.u, 4, 1), w(this.u, 2, .5), w(this.A, 2, .5), w(this.h, 4, .5);
	}
	get baseOptions() {
		return x(this.h, H, 1);
	}
	set baseOptions(e) {
		S(this.h, 0, 1, e);
	}
	o(e) {
		return "numFaces" in e && ur(this.u, 4, e.numFaces ?? 1), "minFaceDetectionConfidence" in e && w(this.u, 2, e.minFaceDetectionConfidence ?? .5), "minTrackingConfidence" in e && w(this.h, 4, e.minTrackingConfidence ?? .5), "minFacePresenceConfidence" in e && w(this.A, 2, e.minFacePresenceConfidence ?? .5), "outputFaceBlendshapes" in e && (this.outputFaceBlendshapes = !!e.outputFaceBlendshapes), "outputFacialTransformationMatrixes" in e && (this.outputFacialTransformationMatrixes = !!e.outputFacialTransformationMatrixes), this.l(e);
	}
	F(e, t) {
		return Fc(this), bc(this, e, t), this.j;
	}
	G(e, t, n) {
		return Fc(this), xc(this, e, n, t), this.j;
	}
	m() {
		var e = new Aa();
		B(e, "image_in"), B(e, "norm_rect"), V(e, "face_landmarks");
		let t = new wa();
		li(t, _o, this.h);
		let n = new z();
		dr(n, 2, "mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph"), L(n, "IMAGE:image_in"), L(n, "NORM_RECT:norm_rect"), R(n, "NORM_LANDMARKS:face_landmarks"), n.o(t), ka(e, n), this.g.attachProtoVectorListener("face_landmarks", ((e, t) => {
			for (let t of e) e = Wa(t), this.j.faceLandmarks.push(ss(e));
			G(this, t);
		})), this.g.attachEmptyPacketListener("face_landmarks", ((e) => {
			G(this, e);
		})), this.outputFaceBlendshapes && (V(e, "blendshapes"), R(n, "BLENDSHAPES:blendshapes"), this.g.attachProtoVectorListener("blendshapes", ((e, t) => {
			if (this.outputFaceBlendshapes) for (let t of e) e = Ia(t), this.j.faceBlendshapes.push(is(e.g() ?? []));
			G(this, t);
		})), this.g.attachEmptyPacketListener("blendshapes", ((e) => {
			G(this, e);
		}))), this.outputFacialTransformationMatrixes && (V(e, "face_geometry"), R(n, "FACE_GEOMETRY:face_geometry"), this.g.attachProtoVectorListener("face_geometry", ((e, t) => {
			if (this.outputFacialTransformationMatrixes) for (let t of e) (e = x(e = po(t), Ga, 2)) && this.j.facialTransformationMatrixes.push({
				rows: lr(e, 1) ?? 0 ?? 0,
				columns: lr(e, 2) ?? 0 ?? 0,
				data: Hn(e, 3, Bt, Vn()).slice() ?? []
			});
			G(this, t);
		})), this.g.attachEmptyPacketListener("face_geometry", ((e) => {
			G(this, e);
		}))), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
Z.prototype.detectForVideo = Z.prototype.G, Z.prototype.detect = Z.prototype.F, Z.prototype.setOptions = Z.prototype.o, Z.createFromModelPath = function(e, t) {
	return X(Z, e, { baseOptions: { modelAssetPath: t } });
}, Z.createFromModelBuffer = function(e, t) {
	return X(Z, e, { baseOptions: { modelAssetBuffer: t } });
}, Z.createFromOptions = function(e, t) {
	return X(Z, e, t);
}, Z.FACE_LANDMARKS_LIPS = Tc, Z.FACE_LANDMARKS_LEFT_EYE = Ec, Z.FACE_LANDMARKS_LEFT_EYEBROW = Dc, Z.FACE_LANDMARKS_LEFT_IRIS = Oc, Z.FACE_LANDMARKS_RIGHT_EYE = kc, Z.FACE_LANDMARKS_RIGHT_EYEBROW = Ac, Z.FACE_LANDMARKS_RIGHT_IRIS = jc, Z.FACE_LANDMARKS_FACE_OVAL = Mc, Z.FACE_LANDMARKS_CONTOURS = Nc, Z.FACE_LANDMARKS_TESSELATION = Pc;
var Ic = hc([0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]);
function Lc(e) {
	e.gestures = [], e.landmarks = [], e.worldLandmarks = [], e.handedness = [];
}
function Rc(e) {
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
function zc(e, t = !0) {
	let n = [];
	for (let i of e) {
		var r = Ia(i);
		e = [];
		for (let n of r.g()) r = t && lr(n, 1) != null ? lr(n, 1) ?? 0 : -1, e.push({
			score: C(n, 2) ?? 0,
			index: r,
			categoryName: _(v(n, 3)) ?? "" ?? "",
			displayName: _(v(n, 4)) ?? "" ?? ""
		});
		n.push(e);
	}
	return n;
}
var Bc = class extends Cc {
	constructor(e, t) {
		super(new vc(e, t), "image_in", "norm_rect", !1), this.gestures = [], this.landmarks = [], this.worldLandmarks = [], this.handedness = [], S(e = this.j = new wo(), 0, 1, t = new H()), this.u = new Co(), S(this.j, 0, 2, this.u), this.D = new So(), S(this.u, 0, 3, this.D), this.A = new xo(), S(this.u, 0, 2, this.A), this.h = new bo(), S(this.j, 0, 3, this.h), w(this.A, 2, .5), w(this.u, 4, .5), w(this.D, 2, .5);
	}
	get baseOptions() {
		return x(this.j, H, 1);
	}
	set baseOptions(e) {
		S(this.j, 0, 1, e);
	}
	o(e) {
		if (ur(this.A, 3, e.numHands ?? 1), "minHandDetectionConfidence" in e && w(this.A, 2, e.minHandDetectionConfidence ?? .5), "minTrackingConfidence" in e && w(this.u, 4, e.minTrackingConfidence ?? .5), "minHandPresenceConfidence" in e && w(this.D, 2, e.minHandPresenceConfidence ?? .5), e.cannedGesturesClassifierOptions) {
			var t = new vo(), n = t, r = ns(e.cannedGesturesClassifierOptions, x(this.h, vo, 3)?.l());
			S(n, 0, 2, r), S(this.h, 0, 3, t);
		} else e.cannedGesturesClassifierOptions === void 0 && x(this.h, vo, 3)?.g();
		return e.customGesturesClassifierOptions ? (S(n = t = new vo(), 0, 2, r = ns(e.customGesturesClassifierOptions, x(this.h, vo, 4)?.l())), S(this.h, 0, 4, t)) : e.customGesturesClassifierOptions === void 0 && x(this.h, vo, 4)?.g(), this.l(e);
	}
	Ha(e, t) {
		return Lc(this), bc(this, e, t), Rc(this);
	}
	Ia(e, t, n) {
		return Lc(this), xc(this, e, n, t), Rc(this);
	}
	m() {
		var e = new Aa();
		B(e, "image_in"), B(e, "norm_rect"), V(e, "hand_gestures"), V(e, "hand_landmarks"), V(e, "world_hand_landmarks"), V(e, "handedness");
		let t = new wa();
		li(t, ko, this.j);
		let n = new z();
		dr(n, 2, "mediapipe.tasks.vision.gesture_recognizer.GestureRecognizerGraph"), L(n, "IMAGE:image_in"), L(n, "NORM_RECT:norm_rect"), R(n, "HAND_GESTURES:hand_gestures"), R(n, "LANDMARKS:hand_landmarks"), R(n, "WORLD_LANDMARKS:world_hand_landmarks"), R(n, "HANDEDNESS:handedness"), n.o(t), ka(e, n), this.g.attachProtoVectorListener("hand_landmarks", ((e, t) => {
			for (let t of e) {
				e = Wa(t);
				let n = [];
				for (let t of ir(e, Ua, 1)) n.push({
					x: C(t, 1) ?? 0,
					y: C(t, 2) ?? 0,
					z: C(t, 3) ?? 0,
					visibility: C(t, 4) ?? 0
				});
				this.landmarks.push(n);
			}
			G(this, t);
		})), this.g.attachEmptyPacketListener("hand_landmarks", ((e) => {
			G(this, e);
		})), this.g.attachProtoVectorListener("world_hand_landmarks", ((e, t) => {
			for (let t of e) {
				e = Ha(t);
				let n = [];
				for (let t of ir(e, Va, 1)) n.push({
					x: C(t, 1) ?? 0,
					y: C(t, 2) ?? 0,
					z: C(t, 3) ?? 0,
					visibility: C(t, 4) ?? 0
				});
				this.worldLandmarks.push(n);
			}
			G(this, t);
		})), this.g.attachEmptyPacketListener("world_hand_landmarks", ((e) => {
			G(this, e);
		})), this.g.attachProtoVectorListener("hand_gestures", ((e, t) => {
			this.gestures.push(...zc(e, !1)), G(this, t);
		})), this.g.attachEmptyPacketListener("hand_gestures", ((e) => {
			G(this, e);
		})), this.g.attachProtoVectorListener("handedness", ((e, t) => {
			this.handedness.push(...zc(e)), G(this, t);
		})), this.g.attachEmptyPacketListener("handedness", ((e) => {
			G(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
function Vc(e) {
	return {
		landmarks: e.landmarks,
		worldLandmarks: e.worldLandmarks,
		handednesses: e.handedness,
		handedness: e.handedness
	};
}
Bc.prototype.recognizeForVideo = Bc.prototype.Ia, Bc.prototype.recognize = Bc.prototype.Ha, Bc.prototype.setOptions = Bc.prototype.o, Bc.createFromModelPath = function(e, t) {
	return X(Bc, e, { baseOptions: { modelAssetPath: t } });
}, Bc.createFromModelBuffer = function(e, t) {
	return X(Bc, e, { baseOptions: { modelAssetBuffer: t } });
}, Bc.createFromOptions = function(e, t) {
	return X(Bc, e, t);
}, Bc.HAND_CONNECTIONS = Ic;
var Q = class extends Cc {
	constructor(e, t) {
		super(new vc(e, t), "image_in", "norm_rect", !1), this.landmarks = [], this.worldLandmarks = [], this.handedness = [], S(e = this.h = new Co(), 0, 1, t = new H()), this.u = new So(), S(this.h, 0, 3, this.u), this.j = new xo(), S(this.h, 0, 2, this.j), ur(this.j, 3, 1), w(this.j, 2, .5), w(this.u, 2, .5), w(this.h, 4, .5);
	}
	get baseOptions() {
		return x(this.h, H, 1);
	}
	set baseOptions(e) {
		S(this.h, 0, 1, e);
	}
	o(e) {
		return "numHands" in e && ur(this.j, 3, e.numHands ?? 1), "minHandDetectionConfidence" in e && w(this.j, 2, e.minHandDetectionConfidence ?? .5), "minTrackingConfidence" in e && w(this.h, 4, e.minTrackingConfidence ?? .5), "minHandPresenceConfidence" in e && w(this.u, 2, e.minHandPresenceConfidence ?? .5), this.l(e);
	}
	F(e, t) {
		return this.landmarks = [], this.worldLandmarks = [], this.handedness = [], bc(this, e, t), Vc(this);
	}
	G(e, t, n) {
		return this.landmarks = [], this.worldLandmarks = [], this.handedness = [], xc(this, e, n, t), Vc(this);
	}
	m() {
		var e = new Aa();
		B(e, "image_in"), B(e, "norm_rect"), V(e, "hand_landmarks"), V(e, "world_hand_landmarks"), V(e, "handedness");
		let t = new wa();
		li(t, Ao, this.h);
		let n = new z();
		dr(n, 2, "mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph"), L(n, "IMAGE:image_in"), L(n, "NORM_RECT:norm_rect"), R(n, "LANDMARKS:hand_landmarks"), R(n, "WORLD_LANDMARKS:world_hand_landmarks"), R(n, "HANDEDNESS:handedness"), n.o(t), ka(e, n), this.g.attachProtoVectorListener("hand_landmarks", ((e, t) => {
			for (let t of e) e = Wa(t), this.landmarks.push(ss(e));
			G(this, t);
		})), this.g.attachEmptyPacketListener("hand_landmarks", ((e) => {
			G(this, e);
		})), this.g.attachProtoVectorListener("world_hand_landmarks", ((e, t) => {
			for (let t of e) e = Ha(t), this.worldLandmarks.push(cs(e));
			G(this, t);
		})), this.g.attachEmptyPacketListener("world_hand_landmarks", ((e) => {
			G(this, e);
		})), this.g.attachProtoVectorListener("handedness", ((e, t) => {
			var n = this.handedness, r = n.push;
			let i = [];
			for (let t of e) {
				e = Ia(t);
				let n = [];
				for (let t of e.g()) n.push({
					score: C(t, 2) ?? 0,
					index: lr(t, 1) ?? 0 ?? -1,
					categoryName: _(v(t, 3)) ?? "" ?? "",
					displayName: _(v(t, 4)) ?? "" ?? ""
				});
				i.push(n);
			}
			r.call(n, ...i), G(this, t);
		})), this.g.attachEmptyPacketListener("handedness", ((e) => {
			G(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
Q.prototype.detectForVideo = Q.prototype.G, Q.prototype.detect = Q.prototype.F, Q.prototype.setOptions = Q.prototype.o, Q.createFromModelPath = function(e, t) {
	return X(Q, e, { baseOptions: { modelAssetPath: t } });
}, Q.createFromModelBuffer = function(e, t) {
	return X(Q, e, { baseOptions: { modelAssetBuffer: t } });
}, Q.createFromOptions = function(e, t) {
	return X(Q, e, t);
}, Q.HAND_CONNECTIONS = Ic;
var Hc = hc([0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20], [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32]);
function Uc(e) {
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
function Wc(e) {
	try {
		if (!e.D) return e.h;
		e.D(e.h);
	} finally {
		Os(e);
	}
}
function Gc(e, t) {
	e = Wa(e), t.push(ss(e));
}
var $ = class extends Cc {
	constructor(e, t) {
		super(new vc(e, t), "input_frames_image", null, !1), this.h = {
			faceLandmarks: [],
			faceBlendshapes: [],
			poseLandmarks: [],
			poseWorldLandmarks: [],
			poseSegmentationMasks: [],
			leftHandLandmarks: [],
			leftHandWorldLandmarks: [],
			rightHandLandmarks: [],
			rightHandWorldLandmarks: []
		}, this.outputPoseSegmentationMasks = this.outputFaceBlendshapes = !1, S(e = this.j = new Po(), 0, 1, t = new H()), this.I = new So(), S(this.j, 0, 2, this.I), this.W = new jo(), S(this.j, 0, 3, this.W), this.u = new lo(), S(this.j, 0, 4, this.u), this.O = new mo(), S(this.j, 0, 5, this.O), this.A = new Mo(), S(this.j, 0, 6, this.A), this.M = new No(), S(this.j, 0, 7, this.M), w(this.u, 2, .5), w(this.u, 3, .3), w(this.O, 2, .5), w(this.A, 2, .5), w(this.A, 3, .3), w(this.M, 2, .5), w(this.I, 2, .5);
	}
	get baseOptions() {
		return x(this.j, H, 1);
	}
	set baseOptions(e) {
		S(this.j, 0, 1, e);
	}
	o(e) {
		return "minFaceDetectionConfidence" in e && w(this.u, 2, e.minFaceDetectionConfidence ?? .5), "minFaceSuppressionThreshold" in e && w(this.u, 3, e.minFaceSuppressionThreshold ?? .3), "minFacePresenceConfidence" in e && w(this.O, 2, e.minFacePresenceConfidence ?? .5), "outputFaceBlendshapes" in e && (this.outputFaceBlendshapes = !!e.outputFaceBlendshapes), "minPoseDetectionConfidence" in e && w(this.A, 2, e.minPoseDetectionConfidence ?? .5), "minPoseSuppressionThreshold" in e && w(this.A, 3, e.minPoseSuppressionThreshold ?? .3), "minPosePresenceConfidence" in e && w(this.M, 2, e.minPosePresenceConfidence ?? .5), "outputPoseSegmentationMasks" in e && (this.outputPoseSegmentationMasks = !!e.outputPoseSegmentationMasks), "minHandLandmarksConfidence" in e && w(this.I, 2, e.minHandLandmarksConfidence ?? .5), this.l(e);
	}
	F(e, t, n) {
		let r = typeof t == "function" ? {} : t;
		return this.D = typeof t == "function" ? t : n, Uc(this), bc(this, e, r), Wc(this);
	}
	G(e, t, n, r) {
		let i = typeof n == "function" ? {} : n;
		return this.D = typeof n == "function" ? n : r, Uc(this), xc(this, e, i, t), Wc(this);
	}
	m() {
		var e = new Aa();
		B(e, "input_frames_image"), V(e, "pose_landmarks"), V(e, "pose_world_landmarks"), V(e, "face_landmarks"), V(e, "left_hand_landmarks"), V(e, "left_hand_world_landmarks"), V(e, "right_hand_landmarks"), V(e, "right_hand_world_landmarks");
		let t = new wa(), n = new ua();
		dr(n, 1, "type.googleapis.com/mediapipe.tasks.vision.holistic_landmarker.proto.HolisticLandmarkerGraphOptions"), function(e, t) {
			if (t != null) if (Array.isArray(t)) y(e, 2, bn(t, 0, Sn));
			else {
				if (!(typeof t == "string" || t instanceof Te || xe(t))) throw Error("invalid value in Any.value field: " + t + " expected a ByteString, a base64 encoded string, a Uint8Array or a jspb array");
				Yn(e, 2, tt(t, !1), Ce());
			}
		}(n, this.j.g());
		let r = new z();
		dr(r, 2, "mediapipe.tasks.vision.holistic_landmarker.HolisticLandmarkerGraph"), cr(r, 8, ua, n), L(r, "IMAGE:input_frames_image"), R(r, "POSE_LANDMARKS:pose_landmarks"), R(r, "POSE_WORLD_LANDMARKS:pose_world_landmarks"), R(r, "FACE_LANDMARKS:face_landmarks"), R(r, "LEFT_HAND_LANDMARKS:left_hand_landmarks"), R(r, "LEFT_HAND_WORLD_LANDMARKS:left_hand_world_landmarks"), R(r, "RIGHT_HAND_LANDMARKS:right_hand_landmarks"), R(r, "RIGHT_HAND_WORLD_LANDMARKS:right_hand_world_landmarks"), r.o(t), ka(e, r), Es(this, e), this.g.attachProtoListener("pose_landmarks", ((e, t) => {
			Gc(e, this.h.poseLandmarks), G(this, t);
		})), this.g.attachEmptyPacketListener("pose_landmarks", ((e) => {
			G(this, e);
		})), this.g.attachProtoListener("pose_world_landmarks", ((e, t) => {
			var n = this.h.poseWorldLandmarks;
			e = Ha(e), n.push(cs(e)), G(this, t);
		})), this.g.attachEmptyPacketListener("pose_world_landmarks", ((e) => {
			G(this, e);
		})), this.outputPoseSegmentationMasks && (R(r, "POSE_SEGMENTATION_MASK:pose_segmentation_mask"), Ds(this, "pose_segmentation_mask"), this.g.Z("pose_segmentation_mask", ((e, t) => {
			this.h.poseSegmentationMasks = [Sc(this, e, !0, !this.D)], G(this, t);
		})), this.g.attachEmptyPacketListener("pose_segmentation_mask", ((e) => {
			this.h.poseSegmentationMasks = [], G(this, e);
		}))), this.g.attachProtoListener("face_landmarks", ((e, t) => {
			Gc(e, this.h.faceLandmarks), G(this, t);
		})), this.g.attachEmptyPacketListener("face_landmarks", ((e) => {
			G(this, e);
		})), this.outputFaceBlendshapes && (V(e, "extra_blendshapes"), R(r, "FACE_BLENDSHAPES:extra_blendshapes"), this.g.attachProtoListener("extra_blendshapes", ((e, t) => {
			var n = this.h.faceBlendshapes;
			this.outputFaceBlendshapes && (e = Ia(e), n.push(is(e.g() ?? []))), G(this, t);
		})), this.g.attachEmptyPacketListener("extra_blendshapes", ((e) => {
			G(this, e);
		}))), this.g.attachProtoListener("left_hand_landmarks", ((e, t) => {
			Gc(e, this.h.leftHandLandmarks), G(this, t);
		})), this.g.attachEmptyPacketListener("left_hand_landmarks", ((e) => {
			G(this, e);
		})), this.g.attachProtoListener("left_hand_world_landmarks", ((e, t) => {
			var n = this.h.leftHandWorldLandmarks;
			e = Ha(e), n.push(cs(e)), G(this, t);
		})), this.g.attachEmptyPacketListener("left_hand_world_landmarks", ((e) => {
			G(this, e);
		})), this.g.attachProtoListener("right_hand_landmarks", ((e, t) => {
			Gc(e, this.h.rightHandLandmarks), G(this, t);
		})), this.g.attachEmptyPacketListener("right_hand_landmarks", ((e) => {
			G(this, e);
		})), this.g.attachProtoListener("right_hand_world_landmarks", ((e, t) => {
			var n = this.h.rightHandWorldLandmarks;
			e = Ha(e), n.push(cs(e)), G(this, t);
		})), this.g.attachEmptyPacketListener("right_hand_world_landmarks", ((e) => {
			G(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
$.prototype.detectForVideo = $.prototype.G, $.prototype.detect = $.prototype.F, $.prototype.setOptions = $.prototype.o, $.createFromModelPath = function(e, t) {
	return X($, e, { baseOptions: { modelAssetPath: t } });
}, $.createFromModelBuffer = function(e, t) {
	return X($, e, { baseOptions: { modelAssetBuffer: t } });
}, $.createFromOptions = function(e, t) {
	return X($, e, t);
}, $.HAND_CONNECTIONS = Ic, $.POSE_CONNECTIONS = Hc, $.FACE_LANDMARKS_LIPS = Tc, $.FACE_LANDMARKS_LEFT_EYE = Ec, $.FACE_LANDMARKS_LEFT_EYEBROW = Dc, $.FACE_LANDMARKS_LEFT_IRIS = Oc, $.FACE_LANDMARKS_RIGHT_EYE = kc, $.FACE_LANDMARKS_RIGHT_EYEBROW = Ac, $.FACE_LANDMARKS_RIGHT_IRIS = jc, $.FACE_LANDMARKS_FACE_OVAL = Mc, $.FACE_LANDMARKS_CONTOURS = Nc, $.FACE_LANDMARKS_TESSELATION = Pc;
var Kc = class extends Cc {
	constructor(e, t) {
		super(new vc(e, t), "input_image", "norm_rect", !0), this.j = { classifications: [] }, S(e = this.h = new Lo(), 0, 1, t = new H());
	}
	get baseOptions() {
		return x(this.h, H, 1);
	}
	set baseOptions(e) {
		S(this.h, 0, 1, e);
	}
	o(e) {
		return S(this.h, 0, 2, ns(e, x(this.h, to, 2))), this.l(e);
	}
	sa(e, t) {
		return this.j = { classifications: [] }, bc(this, e, t), this.j;
	}
	ta(e, t, n) {
		return this.j = { classifications: [] }, xc(this, e, n, t), this.j;
	}
	m() {
		var e = new Aa();
		B(e, "input_image"), B(e, "norm_rect"), V(e, "classifications");
		let t = new wa();
		li(t, Ro, this.h);
		let n = new z();
		dr(n, 2, "mediapipe.tasks.vision.image_classifier.ImageClassifierGraph"), L(n, "IMAGE:input_image"), L(n, "NORM_RECT:norm_rect"), R(n, "CLASSIFICATIONS:classifications"), n.o(t), ka(e, n), this.g.attachProtoListener("classifications", ((e, t) => {
			this.j = as(Ya(e)), G(this, t);
		})), this.g.attachEmptyPacketListener("classifications", ((e) => {
			G(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
Kc.prototype.classifyForVideo = Kc.prototype.ta, Kc.prototype.classify = Kc.prototype.sa, Kc.prototype.setOptions = Kc.prototype.o, Kc.createFromModelPath = function(e, t) {
	return X(Kc, e, { baseOptions: { modelAssetPath: t } });
}, Kc.createFromModelBuffer = function(e, t) {
	return X(Kc, e, { baseOptions: { modelAssetBuffer: t } });
}, Kc.createFromOptions = function(e, t) {
	return X(Kc, e, t);
};
var qc = class extends Cc {
	constructor(e, t) {
		super(new vc(e, t), "image_in", "norm_rect", !0), this.h = new zo(), this.embeddings = { embeddings: [] }, S(e = this.h, 0, 1, t = new H());
	}
	get baseOptions() {
		return x(this.h, H, 1);
	}
	set baseOptions(e) {
		S(this.h, 0, 1, e);
	}
	o(e) {
		var t = this.h, n = x(this.h, ro, 2);
		return n = n ? n.clone() : new ro(), e.l2Normalize === void 0 ? "l2Normalize" in e && y(n, 1) : y(n, 1, Vt(e.l2Normalize)), e.quantize === void 0 ? "quantize" in e && y(n, 2) : y(n, 2, Vt(e.quantize)), S(t, 0, 2, n), this.l(e);
	}
	za(e, t) {
		return bc(this, e, t), this.embeddings;
	}
	Aa(e, t, n) {
		return xc(this, e, n, t), this.embeddings;
	}
	m() {
		var e = new Aa();
		B(e, "image_in"), B(e, "norm_rect"), V(e, "embeddings_out");
		let t = new wa();
		li(t, Bo, this.h);
		let n = new z();
		dr(n, 2, "mediapipe.tasks.vision.image_embedder.ImageEmbedderGraph"), L(n, "IMAGE:image_in"), L(n, "NORM_RECT:norm_rect"), R(n, "EMBEDDINGS:embeddings_out"), n.o(t), ka(e, n), this.g.attachProtoListener("embeddings_out", ((e, t) => {
			e = eo(e), this.embeddings = function(e) {
				return {
					embeddings: ir(e, Qa, 1).map(((e) => {
						let t = {
							headIndex: lr(e, 3) ?? 0 ?? -1,
							headName: _(v(e, 4)) ?? "" ?? ""
						};
						var n = e.v;
						return nr(n, 0 | n[p], Xa, Zn(e, 1)) === void 0 ? (n = /* @__PURE__ */ new Uint8Array(), t.quantizedEmbedding = x(e, Za, Zn(e, 2), void 0)?.na()?.h() ?? n) : (e = Hn(e = x(e, Xa, Zn(e, 1), void 0), 1, Bt, Vn()), t.floatEmbedding = e.slice()), t;
					})),
					timestampMs: rs(v(e, 2, void 0, void 0, Qt) ?? Rn)
				};
			}(e), G(this, t);
		})), this.g.attachEmptyPacketListener("embeddings_out", ((e) => {
			G(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
qc.cosineSimilarity = function(e, t) {
	if (e.floatEmbedding && t.floatEmbedding) e = us(e.floatEmbedding, t.floatEmbedding);
	else {
		if (!e.quantizedEmbedding || !t.quantizedEmbedding) throw Error("Cannot compute cosine similarity between quantized and float embeddings.");
		e = us(ls(e.quantizedEmbedding), ls(t.quantizedEmbedding));
	}
	return e;
}, qc.prototype.embedForVideo = qc.prototype.Aa, qc.prototype.embed = qc.prototype.za, qc.prototype.setOptions = qc.prototype.o, qc.createFromModelPath = function(e, t) {
	return X(qc, e, { baseOptions: { modelAssetPath: t } });
}, qc.createFromModelBuffer = function(e, t) {
	return X(qc, e, { baseOptions: { modelAssetBuffer: t } });
}, qc.createFromOptions = function(e, t) {
	return X(qc, e, t);
};
var Jc = class {
	constructor(e, t, n) {
		this.confidenceMasks = e, this.categoryMask = t, this.qualityScores = n;
	}
	close() {
		this.confidenceMasks?.forEach(((e) => {
			e.close();
		})), this.categoryMask?.close();
	}
};
function Yc(e) {
	let t = function(e) {
		return ir(e, z, 1);
	}(e.ca()).filter(((e) => (_(v(e, 1)) ?? "").includes("mediapipe.tasks.TensorsToSegmentationCalculator")));
	if (e.u = [], t.length > 1) throw Error("The graph has more than one mediapipe.tasks.TensorsToSegmentationCalculator.");
	t.length === 1 && (x(t[0], wa, 7)?.j()?.g() ?? /* @__PURE__ */ new Map()).forEach(((t, n) => {
		e.u[Number(n)] = _(v(t, 1)) ?? "";
	}));
}
function Xc(e) {
	e.categoryMask = void 0, e.confidenceMasks = void 0, e.qualityScores = void 0;
}
function Zc(e) {
	try {
		let t = new Jc(e.confidenceMasks, e.categoryMask, e.qualityScores);
		if (!e.j) return t;
		e.j(t);
	} finally {
		Os(e);
	}
}
Jc.prototype.close = Jc.prototype.close;
var Qc = class extends Cc {
	constructor(e, t) {
		super(new vc(e, t), "image_in", "norm_rect", !1), this.u = [], this.outputCategoryMask = !1, this.outputConfidenceMasks = !0, this.h = new Go(), this.A = new Vo(), S(this.h, 0, 3, this.A), S(e = this.h, 0, 1, t = new H());
	}
	get baseOptions() {
		return x(this.h, H, 1);
	}
	set baseOptions(e) {
		S(this.h, 0, 1, e);
	}
	o(e) {
		return e.displayNamesLocale === void 0 ? "displayNamesLocale" in e && y(this.h, 2) : y(this.h, 2, en(e.displayNamesLocale)), "outputCategoryMask" in e && (this.outputCategoryMask = e.outputCategoryMask ?? !1), "outputConfidenceMasks" in e && (this.outputConfidenceMasks = e.outputConfidenceMasks ?? !0), super.l(e);
	}
	L() {
		Yc(this);
	}
	segment(e, t, n) {
		let r = typeof t == "function" ? {} : t;
		return this.j = typeof t == "function" ? t : n, Xc(this), bc(this, e, r), Zc(this);
	}
	La(e, t, n, r) {
		let i = typeof n == "function" ? {} : n;
		return this.j = typeof n == "function" ? n : r, Xc(this), xc(this, e, i, t), Zc(this);
	}
	Da() {
		return this.u;
	}
	m() {
		var e = new Aa();
		B(e, "image_in"), B(e, "norm_rect");
		let t = new wa();
		li(t, Ko, this.h);
		let n = new z();
		dr(n, 2, "mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph"), L(n, "IMAGE:image_in"), L(n, "NORM_RECT:norm_rect"), n.o(t), ka(e, n), Es(this, e), this.outputConfidenceMasks && (V(e, "confidence_masks"), R(n, "CONFIDENCE_MASKS:confidence_masks"), Ds(this, "confidence_masks"), this.g.aa("confidence_masks", ((e, t) => {
			this.confidenceMasks = e.map(((e) => Sc(this, e, !0, !this.j))), G(this, t);
		})), this.g.attachEmptyPacketListener("confidence_masks", ((e) => {
			this.confidenceMasks = [], G(this, e);
		}))), this.outputCategoryMask && (V(e, "category_mask"), R(n, "CATEGORY_MASK:category_mask"), Ds(this, "category_mask"), this.g.Z("category_mask", ((e, t) => {
			this.categoryMask = Sc(this, e, !1, !this.j), G(this, t);
		})), this.g.attachEmptyPacketListener("category_mask", ((e) => {
			this.categoryMask = void 0, G(this, e);
		}))), V(e, "quality_scores"), R(n, "QUALITY_SCORES:quality_scores"), this.g.attachFloatVectorListener("quality_scores", ((e, t) => {
			this.qualityScores = e, G(this, t);
		})), this.g.attachEmptyPacketListener("quality_scores", ((e) => {
			this.categoryMask = void 0, G(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
Qc.prototype.getLabels = Qc.prototype.Da, Qc.prototype.segmentForVideo = Qc.prototype.La, Qc.prototype.segment = Qc.prototype.segment, Qc.prototype.setOptions = Qc.prototype.o, Qc.createFromModelPath = function(e, t) {
	return X(Qc, e, { baseOptions: { modelAssetPath: t } });
}, Qc.createFromModelBuffer = function(e, t) {
	return X(Qc, e, { baseOptions: { modelAssetBuffer: t } });
}, Qc.createFromOptions = function(e, t) {
	return X(Qc, e, t);
};
var $c = class {
	constructor(e, t, n) {
		this.confidenceMasks = e, this.categoryMask = t, this.qualityScores = n;
	}
	close() {
		this.confidenceMasks?.forEach(((e) => {
			e.close();
		})), this.categoryMask?.close();
	}
};
$c.prototype.close = $c.prototype.close;
var el = class extends Cc {
	constructor(e, t) {
		super(new vc(e, t), "image_in", "norm_rect_in", !1), this.outputCategoryMask = !1, this.outputConfidenceMasks = !0, this.h = new Go(), this.u = new Vo(), S(this.h, 0, 3, this.u), S(e = this.h, 0, 1, t = new H());
	}
	get baseOptions() {
		return x(this.h, H, 1);
	}
	set baseOptions(e) {
		S(this.h, 0, 1, e);
	}
	o(e) {
		return "outputCategoryMask" in e && (this.outputCategoryMask = e.outputCategoryMask ?? !1), "outputConfidenceMasks" in e && (this.outputConfidenceMasks = e.outputConfidenceMasks ?? !0), super.l(e);
	}
	segment(e, t, n, r) {
		let i = typeof n == "function" ? {} : n;
		if (this.j = typeof n == "function" ? n : r, this.qualityScores = this.categoryMask = this.confidenceMasks = void 0, n = this.C + 1, r = new Xo(), t.keypoint && t.scribble) throw Error("Cannot provide both keypoint and scribble.");
		if (t.keypoint) {
			var a = new qo();
			Yn(a, 3, Vt(!0), !1), Yn(a, 1, zt(t.keypoint.x), 0), Yn(a, 2, zt(t.keypoint.y), 0), or(r, 1, Zo, a);
		} else {
			if (!t.scribble) throw Error("Must provide either a keypoint or a scribble.");
			{
				let e = new Yo();
				for (a of t.scribble) Yn(t = new qo(), 3, Vt(!0), !1), Yn(t, 1, zt(a.x), 0), Yn(t, 2, zt(a.y), 0), cr(e, 1, qo, t);
				or(r, 2, Zo, e);
			}
		}
		this.g.addProtoToStream(r.g(), "mediapipe.tasks.vision.interactive_segmenter.proto.RegionOfInterest", "roi_in", n), bc(this, e, i);
		t: {
			try {
				let e = new $c(this.confidenceMasks, this.categoryMask, this.qualityScores);
				if (!this.j) {
					var o = e;
					break t;
				}
				this.j(e);
			} finally {
				Os(this);
			}
			o = void 0;
		}
		return o;
	}
	m() {
		var e = new Aa();
		B(e, "image_in"), B(e, "roi_in"), B(e, "norm_rect_in");
		let t = new wa();
		li(t, Ko, this.h);
		let n = new z();
		dr(n, 2, "mediapipe.tasks.vision.interactive_segmenter.InteractiveSegmenterGraphV2"), L(n, "IMAGE:image_in"), L(n, "ROI:roi_in"), L(n, "NORM_RECT:norm_rect_in"), n.o(t), ka(e, n), Es(this, e), this.outputConfidenceMasks && (V(e, "confidence_masks"), R(n, "CONFIDENCE_MASKS:confidence_masks"), Ds(this, "confidence_masks"), this.g.aa("confidence_masks", ((e, t) => {
			this.confidenceMasks = e.map(((e) => Sc(this, e, !0, !this.j))), G(this, t);
		})), this.g.attachEmptyPacketListener("confidence_masks", ((e) => {
			this.confidenceMasks = [], G(this, e);
		}))), this.outputCategoryMask && (V(e, "category_mask"), R(n, "CATEGORY_MASK:category_mask"), Ds(this, "category_mask"), this.g.Z("category_mask", ((e, t) => {
			this.categoryMask = Sc(this, e, !1, !this.j), G(this, t);
		})), this.g.attachEmptyPacketListener("category_mask", ((e) => {
			this.categoryMask = void 0, G(this, e);
		}))), V(e, "quality_scores"), R(n, "QUALITY_SCORES:quality_scores"), this.g.attachFloatVectorListener("quality_scores", ((e, t) => {
			this.qualityScores = e, G(this, t);
		})), this.g.attachEmptyPacketListener("quality_scores", ((e) => {
			this.categoryMask = void 0, G(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
el.prototype.segment = el.prototype.segment, el.prototype.setOptions = el.prototype.o, el.createFromModelPath = function(e, t) {
	return X(el, e, { baseOptions: { modelAssetPath: t } });
}, el.createFromModelBuffer = function(e, t) {
	return X(el, e, { baseOptions: { modelAssetBuffer: t } });
}, el.createFromOptions = function(e, t) {
	return X(el, e, t);
};
var tl = class extends Cc {
	constructor(e, t) {
		super(new vc(e, t), "input_frame_gpu", "norm_rect", !1), this.j = { detections: [] }, S(e = this.h = new Qo(), 0, 1, t = new H());
	}
	get baseOptions() {
		return x(this.h, H, 1);
	}
	set baseOptions(e) {
		S(this.h, 0, 1, e);
	}
	o(e) {
		return e.displayNamesLocale === void 0 ? "displayNamesLocale" in e && y(this.h, 2) : y(this.h, 2, en(e.displayNamesLocale)), e.maxResults === void 0 ? "maxResults" in e && y(this.h, 3) : ur(this.h, 3, e.maxResults), e.scoreThreshold === void 0 ? "scoreThreshold" in e && y(this.h, 4) : w(this.h, 4, e.scoreThreshold), e.categoryAllowlist === void 0 ? "categoryAllowlist" in e && y(this.h, 5) : fr(this.h, 5, e.categoryAllowlist), e.categoryDenylist === void 0 ? "categoryDenylist" in e && y(this.h, 6) : fr(this.h, 6, e.categoryDenylist), this.l(e);
	}
	F(e, t) {
		return this.j = { detections: [] }, bc(this, e, t), this.j;
	}
	G(e, t, n) {
		return this.j = { detections: [] }, xc(this, e, n, t), this.j;
	}
	m() {
		var e = new Aa();
		B(e, "input_frame_gpu"), B(e, "norm_rect"), V(e, "detections");
		let t = new wa();
		li(t, $o, this.h);
		let n = new z();
		dr(n, 2, "mediapipe.tasks.vision.ObjectDetectorGraph"), L(n, "IMAGE:input_frame_gpu"), L(n, "NORM_RECT:norm_rect"), R(n, "DETECTIONS:detections"), n.o(t), ka(e, n), this.g.attachProtoVectorListener("detections", ((e, t) => {
			for (let t of e) e = Ba(t), this.j.detections.push(os(e));
			G(this, t);
		})), this.g.attachEmptyPacketListener("detections", ((e) => {
			G(this, e);
		})), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
tl.prototype.detectForVideo = tl.prototype.G, tl.prototype.detect = tl.prototype.F, tl.prototype.setOptions = tl.prototype.o, tl.createFromModelPath = async function(e, t) {
	return X(tl, e, { baseOptions: { modelAssetPath: t } });
}, tl.createFromModelBuffer = function(e, t) {
	return X(tl, e, { baseOptions: { modelAssetBuffer: t } });
}, tl.createFromOptions = function(e, t) {
	return X(tl, e, t);
};
var nl = class {
	constructor(e, t, n) {
		this.landmarks = e, this.worldLandmarks = t, this.segmentationMasks = n;
	}
	close() {
		this.segmentationMasks?.forEach(((e) => {
			e.close();
		}));
	}
};
function rl(e) {
	e.landmarks = [], e.worldLandmarks = [], e.segmentationMasks = void 0;
}
function il(e) {
	try {
		let t = new nl(e.landmarks, e.worldLandmarks, e.segmentationMasks);
		if (!e.u) return t;
		e.u(t);
	} finally {
		Os(e);
	}
}
nl.prototype.close = nl.prototype.close;
var al = class extends Cc {
	constructor(e, t) {
		super(new vc(e, t), "image_in", "norm_rect", !1), this.landmarks = [], this.worldLandmarks = [], this.outputSegmentationMasks = !1, S(e = this.h = new es(), 0, 1, t = new H()), this.A = new No(), S(this.h, 0, 3, this.A), this.j = new Mo(), S(this.h, 0, 2, this.j), ur(this.j, 4, 1), w(this.j, 2, .5), w(this.A, 2, .5), w(this.h, 4, .5);
	}
	get baseOptions() {
		return x(this.h, H, 1);
	}
	set baseOptions(e) {
		S(this.h, 0, 1, e);
	}
	o(e) {
		return "numPoses" in e && ur(this.j, 4, e.numPoses ?? 1), "minPoseDetectionConfidence" in e && w(this.j, 2, e.minPoseDetectionConfidence ?? .5), "minTrackingConfidence" in e && w(this.h, 4, e.minTrackingConfidence ?? .5), "minPosePresenceConfidence" in e && w(this.A, 2, e.minPosePresenceConfidence ?? .5), "outputSegmentationMasks" in e && (this.outputSegmentationMasks = e.outputSegmentationMasks ?? !1), this.l(e);
	}
	F(e, t, n) {
		let r = typeof t == "function" ? {} : t;
		return this.u = typeof t == "function" ? t : n, rl(this), bc(this, e, r), il(this);
	}
	G(e, t, n, r) {
		let i = typeof n == "function" ? {} : n;
		return this.u = typeof n == "function" ? n : r, rl(this), xc(this, e, i, t), il(this);
	}
	m() {
		var e = new Aa();
		B(e, "image_in"), B(e, "norm_rect"), V(e, "normalized_landmarks"), V(e, "world_landmarks"), V(e, "segmentation_masks");
		let t = new wa();
		li(t, ts, this.h);
		let n = new z();
		dr(n, 2, "mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph"), L(n, "IMAGE:image_in"), L(n, "NORM_RECT:norm_rect"), R(n, "NORM_LANDMARKS:normalized_landmarks"), R(n, "WORLD_LANDMARKS:world_landmarks"), n.o(t), ka(e, n), Es(this, e), this.g.attachProtoVectorListener("normalized_landmarks", ((e, t) => {
			this.landmarks = [];
			for (let t of e) e = Wa(t), this.landmarks.push(ss(e));
			G(this, t);
		})), this.g.attachEmptyPacketListener("normalized_landmarks", ((e) => {
			this.landmarks = [], G(this, e);
		})), this.g.attachProtoVectorListener("world_landmarks", ((e, t) => {
			this.worldLandmarks = [];
			for (let t of e) e = Ha(t), this.worldLandmarks.push(cs(e));
			G(this, t);
		})), this.g.attachEmptyPacketListener("world_landmarks", ((e) => {
			this.worldLandmarks = [], G(this, e);
		})), this.outputSegmentationMasks && (R(n, "SEGMENTATION_MASK:segmentation_masks"), Ds(this, "segmentation_masks"), this.g.aa("segmentation_masks", ((e, t) => {
			this.segmentationMasks = e.map(((e) => Sc(this, e, !0, !this.u))), G(this, t);
		})), this.g.attachEmptyPacketListener("segmentation_masks", ((e) => {
			this.segmentationMasks = [], G(this, e);
		}))), e = e.g(), this.setGraph(new Uint8Array(e), !0);
	}
};
al.prototype.detectForVideo = al.prototype.G, al.prototype.detect = al.prototype.F, al.prototype.setOptions = al.prototype.o, al.createFromModelPath = function(e, t) {
	return X(al, e, { baseOptions: { modelAssetPath: t } });
}, al.createFromModelBuffer = function(e, t) {
	return X(al, e, { baseOptions: { modelAssetBuffer: t } });
}, al.createFromOptions = function(e, t) {
	return X(al, e, t);
}, al.POSE_CONNECTIONS = Hc;
//#endregion
//#region frontend/src/gesture/mediapipe-adapter.ts
function ol(e) {
	let t = import.meta.url.slice(0, import.meta.url.lastIndexOf("/") + 1);
	return new URL(e, t).href;
}
async function sl() {
	let e = ol("mediapipe/wasm").replace(/\/$/, ""), t = ol("mediapipe/hand_landmarker.task"), n = await hs.forVisionTasks(e), r = {
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
		return await Q.createFromOptions(n, r);
	} catch {
		return await Q.createFromOptions(n, {
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
var cl = (e, t) => Math.hypot(e.x - t.x, e.y - t.y), ll = (e, t) => ({
	x: (e.x + t.x) / 2,
	y: (e.y + t.y) / 2
}), ul = (e) => Number.isFinite(e.x) && Number.isFinite(e.y), dl = 300, fl = class {
	port;
	preferences;
	primaryHand = null;
	lastAt = null;
	smoothed = /* @__PURE__ */ new Map();
	rawPalm = /* @__PURE__ */ new Map();
	hoveredNode = -1;
	hoverSince = 0;
	hoverAnchor = null;
	dwellSelectedNode = -1;
	grab = null;
	pinchNeedsRelease = !1;
	pairDistance = null;
	fistSince = null;
	fistResetDone = !1;
	presentationPose = null;
	presentationOrigin = null;
	presentationLast = null;
	presentationSince = 0;
	constructor(e, n = {}) {
		this.port = e, this.preferences = t(n);
	}
	getPreferences() {
		return { ...this.preferences };
	}
	setPreferences(e) {
		this.reset(), this.preferences = t({
			...this.preferences,
			...e
		});
	}
	process(e, t) {
		let n = e.filter((e) => ul(e.point) && ul(e.thumbPoint) && ul(e.palmPoint) && Number.isFinite(e.pinchDistance) && Number.isInteger(e.extendedFingers) && e.extendedFingers >= 0 && e.extendedFingers <= 4);
		if (!Number.isFinite(t) || !n.length || new Set(n.map((e) => e.handedness)).size !== n.length) {
			this.reset();
			return;
		}
		if (this.lastAt !== null && (t < this.lastAt || t - this.lastAt > dl) && this.reset(), this.lastAt === t) return;
		let r = this.lastAt === null ? 0 : t - this.lastAt;
		n.some((e) => {
			let t = this.rawPalm.get(e.handedness);
			return t && cl(t, e.palmPoint) > Math.max(120, r * 2.5);
		}) && (this.reset(), r = 0), this.lastAt = t;
		let i = this.preferences.smoothingMs === 0 ? 1 : -Math.expm1(-r / this.preferences.smoothingMs), a = new Set(n.map((e) => e.handedness));
		for (let e of this.smoothed.keys()) a.has(e) || this.smoothed.delete(e);
		for (let e of this.rawPalm.keys()) a.has(e) || this.rawPalm.delete(e);
		for (let e of n) this.rawPalm.set(e.handedness, { ...e.palmPoint });
		let s = n.map((e) => {
			let t = this.smoothed.get(e.handedness), n = {
				...e,
				point: o(t?.point ?? null, e.point, i),
				thumbPoint: o(t?.thumbPoint ?? null, e.thumbPoint, i),
				palmPoint: o(t?.palmPoint ?? null, e.palmPoint, i)
			};
			return this.smoothed.set(e.handedness, n), n;
		}), c = this.preferences.dominantHand === "auto" ? this.primaryHand : this.preferences.dominantHand, l = s.find((e) => e.handedness === c) ?? s.find((e) => e.handedness === "right") ?? s[0];
		this.primaryHand && this.primaryHand !== l.handedness && (this.clearInteraction(), this.pinchNeedsRelease = !0), this.primaryHand = l.handedness, this.port.setGestureActive(!0);
		let u = s.find((e) => e.handedness === "left"), d = s.find((e) => e.handedness === "right");
		if (this.preferences.mode === "inspect" && u?.pinch && d?.pinch) {
			this.resetFist(), this.processPairZoom(u, d);
			return;
		}
		if (this.pairDistance !== null && (this.pairDistance = null, this.pinchNeedsRelease = !0), !this.processFist(l, t)) {
			if (this.preferences.mode === "presentation") {
				this.processPresentation(l, t, r);
				return;
			}
			if (l.pinch) {
				this.clearHover(), this.pinchNeedsRelease ? this.port.setCursor(l.point, "idle") : this.processGrab(l);
				return;
			}
			this.pinchNeedsRelease = !1, this.releaseGrab(), l.pointing ? this.processPointer(l, t) : (this.clearHover(), this.port.setCursor(l.point, "idle"));
		}
	}
	reset() {
		this.clearInteraction(), this.primaryHand = null, this.lastAt = null, this.smoothed.clear(), this.rawPalm.clear(), this.pinchNeedsRelease = !0, this.port.setCursor(null, "idle"), this.port.setGestureActive(!1);
	}
	processFist(e, t) {
		return e.extendedFingers !== 0 || e.openPalm || e.pinch ? (this.resetFist(), !1) : (this.releaseGrab(), this.clearHover(), this.resetPresentation(), this.port.setCursor(e.palmPoint, "idle"), this.fistSince === null && (this.fistSince = t), !this.fistResetDone && t - this.fistSince >= this.preferences.resetHoldMs && (this.port.clearSelection(), this.port.fitToGraph(), this.fistResetDone = !0), !0);
	}
	processGrab(e) {
		let t = ll(e.point, e.thumbPoint);
		if (!this.grab) {
			let e = this.port.pick(t);
			this.grab = {
				node: e,
				origin: { ...t },
				last: { ...t },
				moved: !1
			}, e >= 0 && (this.port.select(e), this.port.beginNodeDrag(e), this.port.setNodeDragging(!0));
		}
		let n = this.grab;
		this.port.setCursor(t, n.node >= 0 ? "node-captured" : "canvas-pan"), !(!n.moved && cl(t, n.origin) < this.preferences.movementThresholdPx) && cl(t, n.last) !== 0 && (n.moved = !0, n.node >= 0 ? this.port.moveNode(n.node, t) : this.port.panBy((t.x - n.last.x) * this.preferences.panGain, (t.y - n.last.y) * this.preferences.panGain), n.last = { ...t });
	}
	releaseGrab() {
		this.grab && this.grab.node >= 0 && (this.port.endNodeDrag(this.grab.node, this.grab.moved), this.port.setNodeDragging(!1)), this.grab = null;
	}
	processPairZoom(e, t) {
		this.releaseGrab(), this.clearHover(), this.resetPresentation();
		let n = ll(e.point, e.thumbPoint), r = ll(t.point, t.thumbPoint), i = cl(n, r), a = ll(n, r);
		if (i < 32) {
			this.pairDistance = 0, this.port.setCursor(a, "idle");
			return;
		}
		if (!this.pairDistance) {
			this.pairDistance = i, this.port.setCursor(a, "idle");
			return;
		}
		let o = this.pairDistance;
		if (!(Math.abs(i - o) < this.preferences.movementThresholdPx)) {
			if (i / o > 2 || i / o < .5) {
				this.pairDistance = i;
				return;
			}
			this.port.zoomAt(a, (i / o) ** +this.preferences.zoomGain), this.port.setCursor(a, i > o ? "zoom-in" : "zoom-out"), this.pairDistance = i;
		}
	}
	processPointer(e, t) {
		let n = this.port.pick(e.point);
		if (this.port.setCursor(e.point, "idle"), n < 0) {
			this.clearHover();
			return;
		}
		let r = n !== this.hoveredNode;
		(r || !this.hoverAnchor || cl(e.point, this.hoverAnchor) > this.preferences.movementThresholdPx) && (this.hoverSince = t, this.hoverAnchor = { ...e.point }, r && (this.dwellSelectedNode = -1)), this.hoveredNode = n, this.port.hover(n), this.dwellSelectedNode !== n && t - this.hoverSince >= this.preferences.dwellMs && (this.port.select(n), this.dwellSelectedNode = n);
	}
	processPresentation(e, t, n) {
		if (this.releaseGrab(), e.pinch) {
			if (this.clearHover(), this.pinchNeedsRelease) {
				this.port.setCursor(e.point, "idle");
				return;
			}
			this.presentationPose !== "zoom-out" && this.startPresentation("zoom-out", e.palmPoint, t), this.applyTimedZoom(e.palmPoint, -1, t, n);
			return;
		}
		if (this.pinchNeedsRelease = !1, !e.openPalm) {
			this.resetPresentation(), e.pointing ? this.processPointer(e, t) : (this.clearHover(), this.port.setCursor(e.point, "idle"));
			return;
		}
		if (this.clearHover(), (!this.presentationPose || this.presentationPose === "zoom-out") && this.startPresentation("pending", e.palmPoint, t), this.presentationPose === "pending") {
			let n = this.presentationOrigin, r = e.palmPoint.x - n.x, i = e.palmPoint.y - n.y;
			Math.abs(r) >= this.preferences.movementThresholdPx && Math.abs(r) >= Math.abs(i) * .8 ? this.presentationPose = "orbit" : cl(e.palmPoint, n) >= this.preferences.movementThresholdPx ? this.startPresentation("pending", e.palmPoint, t) : t - this.presentationSince >= this.preferences.activationMs && (this.presentationPose = "zoom-in");
		}
		if (this.presentationPose === "orbit") {
			let t = e.palmPoint.x - this.presentationLast.x;
			t && this.port.orbitBy(t * .006 * this.preferences.orbitGain), this.presentationLast = { ...e.palmPoint }, this.port.setCursor(e.palmPoint, "orbit-swing");
		} else this.presentationPose === "zoom-in" ? this.applyTimedZoom(e.palmPoint, 1, t, n) : this.port.setCursor(e.palmPoint, "idle");
	}
	applyTimedZoom(e, t, n, r) {
		this.port.setCursor(e, t > 0 ? "zoom-in" : "zoom-out");
		let i = Math.max(0, n - Math.max(n - r, this.presentationSince + this.preferences.activationMs));
		i && this.port.zoomAt(e, Math.exp(t * i / 1600 * this.preferences.zoomGain));
	}
	startPresentation(e, t, n) {
		this.presentationPose = e, this.presentationOrigin = { ...t }, this.presentationLast = { ...t }, this.presentationSince = n;
	}
	clearHover() {
		this.hoveredNode >= 0 && this.port.hover(-1), this.hoveredNode = -1, this.hoverAnchor = null, this.hoverSince = 0, this.dwellSelectedNode = -1;
	}
	resetFist() {
		this.fistSince = null, this.fistResetDone = !1;
	}
	resetPresentation() {
		this.presentationPose = null, this.presentationOrigin = null, this.presentationLast = null;
	}
	clearInteraction() {
		this.releaseGrab(), this.clearHover(), this.resetPresentation(), this.resetFist(), this.pairDistance = null;
	}
};
//#endregion
//#region frontend/src/gesture/gesture-controller.ts
function pl(e, t) {
	return t && ml(e);
}
function ml(e) {
	let t = e.getState();
	return t.suspended || !t.enabled && !t.starting ? !1 : (e.setSuspended(!0), !0);
}
var hl = class {
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
	starting = !1;
	startPromise = null;
	generation = 0;
	disposed = !1;
	error = "";
	lastInferenceAt = 0;
	lastVideoTime = -1;
	lastFrameAt = null;
	previousPinch = {
		left: !1,
		right: !1
	};
	dragAnchors = /* @__PURE__ */ new Map();
	constructor(e, t, n, r, i, a, o = {}) {
		this.stage = e, this.layer = t, this.video = n, this.cursor = r, this.renderer = i, this.callbacks = a, this.stateMachine = new fl(this.createInputPort(), o);
	}
	async start() {
		if (this.disposed) return !1;
		if (this.enabled) return !0;
		if (this.startPromise) return this.startPromise;
		if (!navigator.mediaDevices?.getUserMedia) return this.layer.dataset.state = "unavailable", this.error = "当前浏览器无法使用摄像头，请使用 HTTPS 或本机访问；鼠标操作仍可用。", this.notifyState(), !1;
		let e = ++this.generation;
		this.starting = !0, this.error = "", this.layer.dataset.state = "starting", this.notifyState();
		let t = this.startCamera(e);
		this.startPromise = t;
		try {
			return await t;
		} finally {
			this.startPromise === t && (this.startPromise = null);
		}
	}
	async startCamera(e) {
		let t = null, n = null, r = !1, i = () => e === this.generation && !this.disposed;
		try {
			return t = await navigator.mediaDevices.getUserMedia({
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
			}), !i() || (this.stream = t, n = await sl(), !i()) || (this.landmarker = n, r = !0, this.video.srcObject = t, await this.video.play(), !i()) ? !1 : (this.enabled = !0, this.starting = !1, this.suspended = !1, this.layer.hidden = !1, this.layer.dataset.state = "active", t.getVideoTracks().forEach((e) => e.addEventListener?.("ended", () => {
				i() && (this.stop(), this.error = "摄像头连接已结束，可重新开启；鼠标操作仍可用。", this.notifyState());
			}, { once: !0 })), this.notifyState(), !0);
		} catch (e) {
			if (i()) {
				this.stop(), this.layer.dataset.state = "unavailable";
				let t = e?.name;
				this.error = t === "NotAllowedError" || t === "SecurityError" ? "未获得摄像头权限。可在浏览器设置中允许后重试；鼠标操作仍可用。" : t === "NotFoundError" ? "没有找到摄像头；鼠标操作仍可用。" : t === "NotReadableError" ? "摄像头被占用或无法读取，请关闭其他占用程序后重试。" : "手势识别加载失败，请检查摄像头和本地模型资源后重试；鼠标操作仍可用。", this.notifyState();
			}
			return !1;
		} finally {
			i() || (t?.getTracks().forEach((e) => e.stop()), n && !r && n.close());
		}
	}
	tick(e) {
		if (!(!this.enabled || this.suspended || !this.landmarker) && (this.lastFrameAt !== null && e - this.lastFrameAt > 300 && (this.stateMachine.reset(), this.previousPinch = {
			left: !1,
			right: !1
		}, this.lastFrameAt = null), !(this.video.readyState < 2) && !(e - this.lastInferenceAt < 1e3 / 24 || this.video.currentTime === this.lastVideoTime))) {
			this.lastInferenceAt = e, this.lastVideoTime = this.video.currentTime;
			try {
				let t = this.landmarker.detectForVideo(this.video, e);
				this.lastFrameAt = e, this.processLandmarks(t.landmarks, t.handedness);
			} catch {
				this.setSuspended(!0), this.error = "手势识别已暂停：当前帧处理失败。可关闭后重试；鼠标操作仍可用。", this.notifyState();
			}
		}
	}
	setSuspended(e) {
		if (e && this.starting) {
			this.stop();
			return;
		}
		this.enabled && (this.suspended = e, this.stream?.getVideoTracks().forEach((t) => {
			t.enabled = !e;
		}), this.stateMachine.reset(), this.previousPinch = {
			left: !1,
			right: !1
		}, this.lastVideoTime = -1, this.lastFrameAt = null, this.layer.dataset.state = e ? "paused" : "active", this.error = "", this.notifyState());
	}
	setPreferences(e) {
		this.stateMachine.setPreferences(e), this.previousPinch = {
			left: !1,
			right: !1
		};
	}
	getPreferences() {
		return this.stateMachine.getPreferences();
	}
	stop() {
		this.generation += 1, this.startPromise = null, this.starting = !1, this.enabled = !1, this.suspended = !1, this.error = "", this.stateMachine.reset(), this.stream?.getTracks().forEach((e) => e.stop()), this.stream = null, this.video.pause(), this.video.srcObject = null, this.landmarker?.close(), this.landmarker = null, this.layer.hidden = !0, this.layer.dataset.state = "idle", this.cursor.style.opacity = "0", this.dragAnchors.clear(), this.previousPinch = {
			left: !1,
			right: !1
		}, this.lastInferenceAt = 0, this.lastVideoTime = -1, this.lastFrameAt = null, this.notifyState();
	}
	dispose() {
		this.disposed = !0, this.stop();
	}
	getState() {
		return {
			enabled: this.enabled,
			suspended: this.suspended,
			starting: this.starting,
			layerState: this.layer.dataset.state || "idle",
			error: this.error
		};
	}
	notifyState() {
		this.callbacks.onStateChange?.(this.getState());
	}
	processLandmarks(e, t) {
		let r = this.stage.clientWidth, o = this.stage.clientHeight, s = e.map((e) => 1 - (e[8]?.x ?? .5)), c = e.map((e, r) => n(t[r]?.[0]?.categoryName, s[r]));
		e.length >= 2 && new Set(c).size < 2 && (c = s.map((e) => n(void 0, e)));
		let l = e.flatMap((e, t) => {
			let n = c[t], s = i(e, this.previousPinch[n], this.getPreferences());
			if (!s) return [];
			let l = a(s, r, o, n);
			return this.previousPinch[n] = l.pinch, [l];
		});
		for (let e of ["left", "right"]) l.some((t) => t.handedness === e) || (this.previousPinch[e] = !1);
		this.stateMachine.process(l, performance.now());
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
			orbitBy: (e) => this.renderer.orbitBy(e),
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
};
//#endregion
export { t as a, e as i, ml as n, pl as r, hl as t };
