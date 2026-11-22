/**
 * StateSet 3D Weight Studio v2 — Custom WebGL2 Physically-Based (PBR) 3D Engine.
 *
 * Graphics Features:
 * - Real-time 1024x1024 PCF soft shadow maps
 * - Real-time planar optical-bench floor reflections (half-res mirrored camera pass)
 * - Screen-space depth-crevice SSAO + Optical Bokeh Depth-of-Field
 * - Multi-pass HDR FBO + Dual-tier Gaussian Bloom + ACES Filmic Tonemapping
 * - Smooth 60 FPS GPU-uploaded liquid-crystal morphing across edits, checkpoints & live finetuning
 * - Supports 128x128 (16,384 crystals!) binary Float32Array Microscope grids + GQA head dividers
 * - Screen-projected 3D spatial rulers (Layer IDs, Head H0..Hn dividers, SVD stage markers)
 */

import {
  add3,
  clamp,
  cross3,
  lerp,
  mat4Invert,
  mat4LookAt,
  mat4Multiply,
  mat4Ortho,
  mat4Perspective,
  normalize3,
  projectPoint,
  rayIntersectsAABB,
  rayIntersectsPlaneY,
  scale3,
  screenToRay,
  sub3,
  vec3,
} from './math3d.js';
import {
  BLUR_FS,
  BRIGHT_PASS_FS,
  COMPOSITE_FS,
  CONDUIT_FS,
  CONDUIT_VS,
  FLOOR_FS,
  FLOOR_VS,
  INSTANCED_PBR_FS,
  INSTANCED_PBR_VS,
  POST_VS,
  SHADOW_FS,
  SHADOW_VS,
  SKYBOX_FS,
  SKYBOX_VS,
  STUDIO_RIGS,
} from './shaders.js';

function createChamferedBoxGeometry(bevel = 0.085) {
  const positions = [];
  const normals = [];
  const h = 0.5;
  const b = h - bevel;

  const pushTri = (a, bPt, c, n) => {
    positions.push(a[0], a[1], a[2], bPt[0], bPt[1], bPt[2], c[0], c[1], c[2]);
    normals.push(n[0], n[1], n[2], n[0], n[1], n[2], n[0], n[1], n[2]);
  };
  const pushQuad = (a, bPt, c, d, n) => {
    pushTri(a, bPt, c, n);
    pushTri(a, c, d, n);
  };

  pushQuad([-b, h, -b], [-b, h, b], [b, h, b], [b, h, -b], [0, 1, 0]);
  pushQuad([-b, -h, b], [-b, -h, -b], [b, -h, -b], [b, -h, b], [0, -1, 0]);
  pushQuad([h, -b, b], [h, -b, -b], [h, b, -b], [h, b, b], [1, 0, 0]);
  pushQuad([-h, -b, -b], [-h, -b, b], [-h, b, b], [-h, b, -b], [-1, 0, 0]);
  pushQuad([-b, -b, h], [b, -b, h], [b, b, h], [-b, b, h], [0, 0, 1]);
  pushQuad([b, -b, -h], [-b, -b, -h], [-b, b, -h], [b, b, -h], [0, 0, -1]);

  const s = Math.SQRT1_2;
  pushQuad([-b, h, b], [-b, b, h], [b, b, h], [b, h, b], [0, s, s]);
  pushQuad([b, h, -b], [b, b, -h], [-b, b, -h], [-b, h, -b], [0, s, -s]);
  pushQuad([b, h, b], [h, b, b], [h, b, -b], [b, h, -b], [s, s, 0]);
  pushQuad([-b, h, -b], [-h, b, -b], [-h, b, b], [-b, h, b], [-s, s, 0]);

  pushQuad([-b, -b, h], [-b, -h, b], [b, -h, b], [b, -b, h], [0, -s, s]);
  pushQuad([b, -b, -h], [b, -h, -b], [-b, -h, -b], [-b, -b, -h], [0, -s, -s]);
  pushQuad([h, -b, b], [b, -h, b], [b, -h, -b], [h, -b, -b], [s, -s, 0]);
  pushQuad([-h, -b, -b], [-b, -h, -b], [-b, -h, b], [-h, -b, b], [-s, -s, 0]);

  pushQuad([b, -b, h], [h, -b, b], [h, b, b], [b, b, h], [s, 0, s]);
  pushQuad([h, -b, -b], [b, -b, -h], [b, b, -h], [h, b, -b], [s, 0, -s]);
  pushQuad([-b, -b, -h], [-h, -b, -b], [-h, b, -b], [-b, b, -h], [-s, 0, -s]);
  pushQuad([-h, -b, b], [-b, -b, h], [-b, b, h], [-h, b, b], [-s, 0, s]);

  const cNorm = 1.0 / Math.sqrt(3);
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const px = [sx * h, sy * b, sz * b];
        const py = [sx * b, sy * h, sz * b];
        const pz = [sx * b, sy * b, sz * h];
        const n = [sx * cNorm, sy * cNorm, sz * cNorm];
        if (sx * sy * sz > 0) pushTri(px, py, pz, n);
        else pushTri(px, pz, py, n);
      }
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    vertexCount: positions.length / 3,
  };
}

function createConduitGeometry(curves) {
  const pos = [];
  const norm = [];
  const meta = [];
  const radialSegs = 8;
  const pathSegs = 16;

  const evalBezier = (p0, p1, p2, p3, t) => {
    const u = 1 - t;
    const uu = u * u, tt = t * t;
    return [
      uu * u * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + tt * t * p3[0],
      uu * u * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + tt * t * p3[1],
      uu * u * p0[2] + 3 * uu * t * p1[2] + 3 * u * tt * p2[2] + tt * t * p3[2],
    ];
  };

  for (const curve of curves) {
    const { p0, p1, p2, p3, radius = 0.045, lane = 0, intensity = 0.7 } = curve;
    const rings = [];
    for (let i = 0; i <= pathSegs; i++) {
      const t = i / pathSegs;
      const pt = evalBezier(p0, p1, p2, p3, t);
      const ptNext = evalBezier(p0, p1, p2, p3, Math.min(1, t + 0.01));
      const ptPrev = evalBezier(p0, p1, p2, p3, Math.max(0, t - 0.01));
      const tangent = normalize3(sub3(ptNext, ptPrev));
      const refUp = Math.abs(tangent[1]) > 0.92 ? vec3(1, 0, 0) : vec3(0, 1, 0);
      const right = normalize3(cross3(tangent, refUp));
      const up = normalize3(cross3(right, tangent));

      const ring = [];
      for (let j = 0; j < radialSegs; j++) {
        const ang = (j / radialSegs) * Math.PI * 2;
        const ca = Math.cos(ang), sa = Math.sin(ang);
        const nx = right[0] * ca + up[0] * sa;
        const ny = right[1] * ca + up[1] * sa;
        const nz = right[2] * ca + up[2] * sa;
        ring.push({
          p: [pt[0] + nx * radius, pt[1] + ny * radius, pt[2] + nz * radius],
          n: [nx, ny, nz],
          t,
        });
      }
      rings.push(ring);
    }

    for (let i = 0; i < pathSegs; i++) {
      const r0 = rings[i];
      const r1 = rings[i + 1];
      for (let j = 0; j < radialSegs; j++) {
        const jn = (j + 1) % radialSegs;
        const v00 = r0[j], v01 = r0[jn], v10 = r1[j], v11 = r1[jn];
        for (const v of [v00, v10, v11, v00, v11, v01]) {
          pos.push(v.p[0], v.p[1], v.p[2]);
          norm.push(v.n[0], v.n[1], v.n[2]);
          meta.push(v.t, lane, intensity);
        }
      }
    }
  }

  return {
    positions: new Float32Array(pos),
    normals: new Float32Array(norm),
    meta: new Float32Array(meta),
    vertexCount: pos.length / 3,
  };
}

const ROLE_STYLES = {
  linear_attn: { base: [0.16, 0.56, 0.88], metal: 0.65, rough: 0.24 },
  self_attn: { base: [0.82, 0.55, 0.18], metal: 0.78, rough: 0.22 },
  mlp: { base: [0.18, 0.72, 0.56], metal: 0.58, rough: 0.26 },
  norm: { base: [0.58, 0.64, 0.76], metal: 0.85, rough: 0.18 },
  embedding: { base: [0.64, 0.42, 0.88], metal: 0.55, rough: 0.28 },
  other: { base: [0.46, 0.54, 0.66], metal: 0.60, rough: 0.30 },
};

function mapWeightColor(
  value,
  absMax,
  colorMode = 'signed',
  deltaVal = 0,
  stdVal = 0.05,
  importanceVal = 0,
  conflictVal = 0
) {
  const norm = clamp(value / Math.max(absMax, 1e-8), -1, 1);
  const mag = Math.abs(norm);

  if (colorMode === 'importance') {
    const t = clamp(importanceVal, 0, 1.2) / 1.2;
    if (t < 0.28) {
      const k = t / 0.28;
      return { rgb: [lerp(0.05, 0.12, k), lerp(0.07, 0.14, k), lerp(0.12, 0.24, k)], emit: 0.01 + k * 0.05 };
    }
    if (t < 0.68) {
      const k = (t - 0.28) / 0.40;
      return { rgb: [lerp(0.12, 0.92, k), lerp(0.14, 0.42, k), lerp(0.24, 0.08, k)], emit: 0.06 + k * 0.38 };
    }
    const k = (t - 0.68) / 0.32;
    return { rgb: [lerp(0.92, 1.0, k), lerp(0.42, 0.90, k), lerp(0.08, 0.28, k)], emit: 0.44 + k * 0.95 };
  }

  if (colorMode === 'conflict') {
    const cMag = clamp(Math.abs(conflictVal) / Math.max(absMax * 0.55, 1e-6), 0, 1);
    if (cMag < 0.03) return { rgb: [0.08, 0.11, 0.17], emit: 0.02 };
    if (conflictVal < 0) {
      // Sign conflict between Adapter A and Adapter B -> blazing electric magenta/rose!
      return {
        rgb: [lerp(0.55, 1.0, cMag), lerp(0.08, 0.18, cMag), lerp(0.36, 0.72, cMag)],
        emit: 0.22 + cMag * 1.15,
      };
    }
    // Sign aligned -> calm luminous emerald
    return {
      rgb: [lerp(0.06, 0.14, cMag), lerp(0.35, 0.86, cMag), lerp(0.26, 0.58, cMag)],
      emit: 0.08 + cMag * 0.48,
    };
  }

  if (colorMode === 'delta') {
    const dMag = clamp(Math.abs(deltaVal) / Math.max(absMax * 0.45, 1e-6), 0, 1);
    if (dMag < 0.02) return { rgb: [0.09, 0.12, 0.18], emit: 0.02 };
    if (deltaVal >= 0) {
      return {
        rgb: [lerp(0.08, 0.16, dMag), lerp(0.45, 0.96, dMag), lerp(0.32, 0.64, dMag)],
        emit: 0.22 + dMag * 1.05,
      };
    }
    return {
      rgb: [lerp(0.55, 0.98, dMag), lerp(0.14, 0.24, dMag), lerp(0.34, 0.56, dMag)],
      emit: 0.22 + dMag * 1.05,
    };
  }

  if (colorMode === 'magnitude') {
    const t = Math.pow(mag, 0.85);
    if (t < 0.35) {
      const k = t / 0.35;
      return { rgb: [lerp(0.04, 0.05, k), lerp(0.08, 0.30, k), lerp(0.18, 0.58, k)], emit: 0.02 + k * 0.14 };
    }
    if (t < 0.72) {
      const k = (t - 0.35) / 0.37;
      return { rgb: [lerp(0.05, 0.78, k), lerp(0.30, 0.48, k), lerp(0.58, 0.10, k)], emit: 0.16 + k * 0.34 };
    }
    const k = (t - 0.72) / 0.28;
    return { rgb: [lerp(0.78, 1.0, k), lerp(0.48, 0.82, k), lerp(0.10, 0.26, k)], emit: 0.50 + k * 0.65 };
  }

  if (colorMode === 'gradient') {
    const t = Math.pow(mag, 0.80);
    if (t < 0.5) {
      const k = t * 2;
      return { rgb: [lerp(0.08, 0.56, k), lerp(0.05, 0.12, k), lerp(0.28, 0.68, k)], emit: 0.06 + k * 0.28 };
    }
    const k = (t - 0.5) * 2;
    return { rgb: [lerp(0.56, 0.98, k), lerp(0.12, 0.68, k), lerp(0.68, 0.14, k)], emit: 0.34 + k * 0.72 };
  }

  if (colorMode === 'outlier') {
    const z = Math.abs(value) / Math.max(stdVal, 1e-6);
    if (z < 2.2) {
      const k = clamp(z / 2.2, 0, 1);
      return { rgb: [lerp(0.06, 0.14, k), lerp(0.09, 0.19, k), lerp(0.16, 0.30, k)], emit: 0.02 + k * 0.06 };
    }
    const k = clamp((z - 2.2) / 3.2, 0, 1);
    return { rgb: [lerp(0.88, 1.0, k), lerp(0.12, 0.52, k), lerp(0.44, 0.78, k)], emit: 0.42 + k * 1.15 };
  }

  if (mag < 0.04) return { rgb: [0.06, 0.08, 0.13], emit: 0.01 };
  const t = Math.pow(mag, 0.84);
  if (norm < 0) {
    return {
      rgb: [lerp(0.03, 0.10, t), lerp(0.14, 0.62, t), lerp(0.38, 0.96, t)],
      emit: 0.04 + Math.pow(t, 1.5) * 0.68,
    };
  }
  return {
    rgb: [lerp(0.36, 0.98, t), lerp(0.15, 0.58, t), lerp(0.03, 0.12, t)],
    emit: 0.04 + Math.pow(t, 1.5) * 0.72,
  };
}

function compileShader(gl, type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error: ${log}`);
  }
  return shader;
}

function createProgram(gl, vsSrc, fsSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`Program link error: ${log}`);
  }
  return prog;
}

function createFramebuffer(gl, width, height, { withDepthTex = false, withDepthRb = false, hdr = false } = {}) {
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);

  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  if (hdr) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, width, height, 0, gl.RGBA, gl.HALF_FLOAT, null);
  } else {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  }
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

  let depthTex = null;
  let depthRb = null;
  if (withDepthTex) {
    depthTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, depthTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, width, height, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depthTex, 0);
  } else if (withDepthRb) {
    depthRb = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depthRb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRb);
  }

  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteTexture(tex);
    if (depthTex) gl.deleteTexture(depthTex);
    if (depthRb) gl.deleteRenderbuffer(depthRb);
    gl.deleteFramebuffer(fb);
    return null;
  }
  return { fb, tex, depthTex, depthRb, width, height };
}

function createShadowFramebuffer(gl, size = 1024) {
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);

  const depthTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, depthTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, size, size, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depthTex, 0);
  gl.drawBuffers([gl.NONE]);
  gl.readBuffer(gl.NONE);

  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteTexture(depthTex);
    gl.deleteFramebuffer(fb);
    return null;
  }
  return { fb, depthTex, width: size, height: size };
}

function destroyFramebuffer(gl, target) {
  if (!target) return;
  if (target.tex) gl.deleteTexture(target.tex);
  if (target.depthTex) gl.deleteTexture(target.depthTex);
  if (target.depthRb) gl.deleteRenderbuffer(target.depthRb);
  if (target.fb) gl.deleteFramebuffer(target.fb);
}

export class WeightStudioRenderer {
  constructor(canvas, callbacks = {}) {
    this.canvas = canvas;
    this.callbacks = callbacks;

    const gl = canvas.getContext('webgl2', {
      antialias: true,
      alpha: false,
      depth: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    if (!gl) {
      throw new Error('WebGL2 is required for the 3D Weight Studio.');
    }
    this.gl = gl;
    this.hdrSupported = Boolean(gl.getExtension('EXT_color_buffer_float'));

    // Visual realism toggles & settings
    this.rigKey = 'obsidian';
    this.exposureOverride = null;
    this.bloomOverride = null;
    this.enableReflections = true;
    this.enableShadows = true;
    this.ssaoStrength = 0.65;
    this.dofStrength = 0.0; // toggleable cinematic bokeh DoF!
    this.showSpatialRulers = true;
    this.showHeadDividers = true;
    this.colorMode = 'signed';
    this.viewMode = 'stack';
    this.layerSpread = 1.0;
    this.showConduits = true;
    this.showGhostOverlay = true;
    this.autoOrbit = false;
    this.brushMode = false;
    this.loraTargetHighlight = new Set();

    // Spring-damped spherical camera
    this.cam = {
      yaw: 0.62,
      pitch: 0.42,
      distance: 26,
      target: vec3(0, 1.5, 0),
      goalYaw: 0.62,
      goalPitch: 0.42,
      goalDistance: 26,
      goalTarget: vec3(0, 1.5, 0),
    };
    this.viewProj = new Float32Array(16);
    this.invViewProj = new Float32Array(16);
    this.lightViewProj = new Float32Array(16);
    this.cameraPos = vec3(0, 10, 22);

    // Scene state
    this.structure = null;
    this.tensorDetail = null;
    this.binaryGrid = null; // optional 128x128 Float32Array binary payload
    this.trajectory = null;
    this.selectedTensorName = null;
    this.hoveredHit = null;
    this.svdCutoffRank = 16;
    this.activeCheckpointIdx = 0;

    // 3D Microscope Brush selection
    this.brushRect = null;
    this._brushDragging = false;
    this._brushAnchor = null;

    // Live activation / backprop pulse state
    this.pulseWave = -1.0;
    this.pulseMode = 0.0;
    this.pulseSpeed = 0.35;
    this.layerHeat = new Map();

    // GPU Liquid-Crystal Morphing buffers
    this._morphPrev = null;
    this._morphGoal = null;
    this._morphWork = null;
    this._morphT = 1.0;
    this._morphDuration = 0.38;

    this.opaqueCount = 0;
    this.transCount = 0;
    this.conduitVertexCount = 0;
    this.hitBoxes = [];
    this.spatialAnchors = []; // 3D world-space ruler anchors projected every frame
    this.microscopeMeta = null;
    this.footprintSize = [12, 12];
    this.glowCenter = [0, 0, 0];
    this.glowColor = [0.22, 0.58, 0.95];

    this.fps = 60;
    this._frameCount = 0;
    this._fpsTimer = performance.now();
    this._lastTime = performance.now();
    this._elapsed = 0;

    this._initPrograms();
    this._initGeometry();
    this._initPostBuffers();
    this._bindEvents();
    this._startLoop();
  }

  _initPrograms() {
    const gl = this.gl;
    this.progShadow = createProgram(gl, SHADOW_VS, SHADOW_FS);
    this.progSkybox = createProgram(gl, SKYBOX_VS, SKYBOX_FS);
    this.progFloor = createProgram(gl, FLOOR_VS, FLOOR_FS);
    this.progPbr = createProgram(gl, INSTANCED_PBR_VS, INSTANCED_PBR_FS);
    this.progConduit = createProgram(gl, CONDUIT_VS, CONDUIT_FS);
    this.progBright = createProgram(gl, POST_VS, BRIGHT_PASS_FS);
    this.progBlur = createProgram(gl, POST_VS, BLUR_FS);
    this.progComposite = createProgram(gl, POST_VS, COMPOSITE_FS);
  }

  _initGeometry() {
    const gl = this.gl;
    this.emptyVao = gl.createVertexArray();

    const floorVerts = new Float32Array([
      -120, 0, -120, 120, 0, -120, 120, 0, 120,
      -120, 0, -120, 120, 0, 120, -120, 0, 120,
    ]);
    this.floorVao = gl.createVertexArray();
    gl.bindVertexArray(this.floorVao);
    const floorBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, floorBuf);
    gl.bufferData(gl.ARRAY_BUFFER, floorVerts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    const box = createChamferedBoxGeometry(0.085);
    this.boxVertexCount = box.vertexCount;
    this.boxPosBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.boxPosBuf);
    gl.bufferData(gl.ARRAY_BUFFER, box.positions, gl.STATIC_DRAW);
    this.boxNormBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.boxNormBuf);
    gl.bufferData(gl.ARRAY_BUFFER, box.normals, gl.STATIC_DRAW);

    const setupInstancedVao = () => {
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.boxPosBuf);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.boxNormBuf);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

      const instBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
      const stride = 16 * 4;
      for (let loc = 2; loc <= 5; loc++) {
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, stride, (loc - 2) * 16);
        gl.vertexAttribDivisor(loc, 1);
      }
      return { vao, instBuf };
    };

    const opq = setupInstancedVao();
    this.opaqueVao = opq.vao;
    this.opaqueInstBuf = opq.instBuf;

    const trn = setupInstancedVao();
    this.transVao = trn.vao;
    this.transInstBuf = trn.instBuf;

    this.conduitVao = gl.createVertexArray();
    gl.bindVertexArray(this.conduitVao);
    this.conduitPosBuf = gl.createBuffer();
    this.conduitNormBuf = gl.createBuffer();
    this.conduitMetaBuf = gl.createBuffer();
    gl.bindVertexArray(null);
  }

  _initPostBuffers() {
    const gl = this.gl;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(2, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(2, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }

    for (const key of ['fboScene', 'fboReflect', 'fboShadow', 'fboBright', 'fboBlur1A', 'fboBlur1B', 'fboBlur2A', 'fboBlur2B']) {
      destroyFramebuffer(gl, this[key]);
      this[key] = null;
    }

    const hw = Math.max(2, Math.floor(w / 2));
    const hh = Math.max(2, Math.floor(h / 2));
    const qw = Math.max(2, Math.floor(w / 4));
    const qh = Math.max(2, Math.floor(h / 4));

    const tryHdr = this.hdrSupported;
    this.fboScene = createFramebuffer(gl, w, h, { withDepthTex: true, hdr: tryHdr }) ||
      createFramebuffer(gl, w, h, { withDepthTex: true, hdr: false });
    this.fboReflect = createFramebuffer(gl, hw, hh, { withDepthRb: true, hdr: tryHdr }) ||
      createFramebuffer(gl, hw, hh, { withDepthRb: true, hdr: false });
    this.fboShadow = createShadowFramebuffer(gl, 1024);
    this.fboBright = createFramebuffer(gl, hw, hh, { hdr: tryHdr }) ||
      createFramebuffer(gl, hw, hh, { hdr: false });
    this.fboBlur1A = createFramebuffer(gl, hw, hh, { hdr: tryHdr }) ||
      createFramebuffer(gl, hw, hh, { hdr: false });
    this.fboBlur1B = createFramebuffer(gl, hw, hh, { hdr: tryHdr }) ||
      createFramebuffer(gl, hw, hh, { hdr: false });
    this.fboBlur2A = createFramebuffer(gl, qw, qh, { hdr: tryHdr }) ||
      createFramebuffer(gl, qw, qh, { hdr: false });
    this.fboBlur2B = createFramebuffer(gl, qw, qh, { hdr: tryHdr }) ||
      createFramebuffer(gl, qw, qh, { hdr: false });

    this.postReady = Boolean(
      this.fboScene && this.fboScene.depthTex && this.fboBright && this.fboBlur1A && this.fboBlur1B && this.fboBlur2A && this.fboBlur2B
    );
  }

  resize() {
    this._initPostBuffers();
  }

  setCameraPreset(preset) {
    if (preset === 'iso') {
      this.cam.goalYaw = 0.65;
      this.cam.goalPitch = 0.42;
      this.cam.goalDistance = this.viewMode === 'microscope' ? 22 : 28;
    } else if (preset === 'top') {
      this.cam.goalYaw = 0.0;
      this.cam.goalPitch = 1.38;
      this.cam.goalDistance = this.viewMode === 'microscope' ? 23 : 32;
    } else if (preset === 'front') {
      this.cam.goalYaw = 0.0;
      this.cam.goalPitch = 0.20;
      this.cam.goalDistance = this.viewMode === 'microscope' ? 21 : 27;
    } else if (preset === 'side') {
      this.cam.goalYaw = Math.PI * 0.5;
      this.cam.goalPitch = 0.24;
      this.cam.goalDistance = 30;
    } else if (preset === 'reset') {
      this._fitCameraToViewMode();
    }
  }

  _fitCameraToViewMode() {
    if (this.viewMode === 'stack') {
      const nLayers = this.structure?.layers?.length || 12;
      this.cam.goalTarget = vec3(0, 1.3, 0);
      this.cam.goalYaw = 0.62;
      this.cam.goalPitch = 0.40;
      this.cam.goalDistance = clamp(18 + nLayers * 0.68 * this.layerSpread, 20, 56);
    } else if (this.viewMode === 'microscope') {
      this.cam.goalTarget = vec3(0, 1.2, 0);
      this.cam.goalYaw = 0.72;
      this.cam.goalPitch = 0.58;
      this.cam.goalDistance = 23.5;
    } else if (this.viewMode === 'spectral') {
      this.cam.goalTarget = vec3(0, 1.6, 0.5);
      this.cam.goalYaw = 0.36;
      this.cam.goalPitch = 0.44;
      this.cam.goalDistance = 26.5;
    } else if (this.viewMode === 'trajectory') {
      this.cam.goalTarget = vec3(0, 1.8, 0);
      this.cam.goalYaw = 0.74;
      this.cam.goalPitch = 0.48;
      this.cam.goalDistance = 28.0;
    }
  }

  focusOnTensor(tensorName) {
    this.selectedTensorName = tensorName;
    if (this.viewMode === 'stack') {
      const hit = this.hitBoxes.find((h) => h.tensorName === tensorName);
      if (hit) {
        this.cam.goalTarget = vec3(
          (hit.min[0] + hit.max[0]) * 0.5,
          (hit.min[1] + hit.max[1]) * 0.5 + 0.3,
          (hit.min[2] + hit.max[2]) * 0.5
        );
        this.cam.goalDistance = clamp(this.cam.goalDistance, 9.5, 16.5);
      }
      this.rebuildScene(false);
    }
  }

  triggerPulse(mode = 'forward', layerIntensities = null) {
    this.pulseMode = mode === 'backprop' ? 1.0 : 0.0;
    this.pulseWave = mode === 'backprop' ? 1.15 : -0.15;
    this.pulseSpeed = mode === 'backprop' ? 0.75 : 0.65;
    if (layerIntensities) {
      this.layerHeat.clear();
      for (const [k, v] of Object.entries(layerIntensities)) {
        this.layerHeat.set(Number(k), clamp(Number(v), 0, 1.5));
      }
      this.rebuildScene(true);
    }
  }

  setRig(rigKey) {
    if (STUDIO_RIGS[rigKey]) this.rigKey = rigKey;
  }

  setViewMode(mode) {
    this.viewMode = mode;
    this.brushRect = null;
    this._morphPrev = null; // hard reset morph across distinct topologies
    this.rebuildScene(false);
    this._fitCameraToViewMode();
  }

  setData({ structure, tensorDetail, binaryGrid, trajectory, selectedTensorName }) {
    const animateMorph = (tensorDetail !== undefined || structure !== undefined);
    if (structure !== undefined) this.structure = structure;
    if (tensorDetail !== undefined) {
      this.tensorDetail = tensorDetail;
      if (tensorDetail?.svd?.rank) {
        this.svdCutoffRank = Math.min(this.svdCutoffRank || tensorDetail.svd.rank, tensorDetail.svd.rank);
      }
    }
    if (binaryGrid !== undefined) this.binaryGrid = binaryGrid;
    if (trajectory !== undefined) this.trajectory = trajectory;
    if (selectedTensorName !== undefined) this.selectedTensorName = selectedTensorName;
    this.rebuildScene(animateMorph);
  }

  setSvdCutoff(rank) {
    this.svdCutoffRank = Math.max(1, rank);
    if (this.viewMode === 'spectral') this.rebuildScene(true);
  }

  setActiveCheckpointIndex(idx) {
    this.activeCheckpointIdx = idx;
    if (this.viewMode === 'trajectory') this.rebuildScene(true);
  }

  setLoraTargets(targetsList) {
    this.loraTargetHighlight = new Set(targetsList || []);
    if (this.viewMode === 'stack') this.rebuildScene(true);
  }

  captureScreenshotDataUrl() {
    this._renderFrame();
    return this.canvas.toDataURL('image/png');
  }

  // -------------------------------------------------------------------------
  // Scene Builders
  // -------------------------------------------------------------------------

  rebuildScene(animateMorph = true) {
    const opaque = [];
    const trans = [];
    const curves = [];
    this.hitBoxes = [];
    this.spatialAnchors = [];
    this.microscopeMeta = null;

    const pushInst = (list, x, y, z, sx, sy, sz, rough, metal, r, g, b, emit, flag = 0, alpha = 1, phase = 0, heat = 0) => {
      list.push(x, y, z, sx, sy, sz, rough, metal, r, g, b, emit, flag, alpha, phase, heat);
    };

    if (this.viewMode === 'microscope' && (this.binaryGrid || this.tensorDetail)) {
      this._buildMicroscopeScene(opaque, trans, curves, pushInst);
    } else if (this.viewMode === 'spectral' && this.tensorDetail?.svd) {
      this._buildSpectralScene(opaque, trans, curves, pushInst);
    } else if (this.viewMode === 'trajectory' && this.trajectory) {
      this._buildTrajectoryScene(opaque, trans, curves, pushInst);
    } else if (this.structure) {
      this._buildArchitectureScene(opaque, trans, curves, pushInst);
    }

    this._uploadInstances(opaque, trans, curves, animateMorph);
  }

  _buildArchitectureScene(opaque, trans, curves, pushInst) {
    const layers = this.structure.layers || [];
    const globals = this.structure.global_tensors || [];
    const nLayers = Math.max(1, layers.length);
    const maxLayerNorm = Math.max(1e-5, this.structure.summary?.max_layer_norm || 1);

    const zStep = 2.05 * this.layerSpread;
    const totalZ = (nLayers - 1) * zStep;
    const zStart = -totalZ * 0.5;
    const waferW = 13.8;
    const waferD = 1.48;
    const waferH = 0.24;

    this.footprintSize = [waferW * 0.58, Math.max(6, totalZ * 0.56 + 2.5)];
    this.glowCenter = [0, 0, 0];
    this.glowColor = this.pulseMode > 0.5 ? [0.95, 0.42, 0.14] : [0.22, 0.62, 0.96];

    const railLen = totalZ + 4.4;
    for (const rx of [-waferW * 0.46, waferW * 0.46]) {
      pushInst(opaque, rx, 0.16, 0, 0.24, 0.18, railLen, 0.22, 0.92, 0.85, 0.64, 0.28, 0.18, 0, 1, 0.5, 0);
      curves.push({
        p0: [rx, 0.30, zStart - 2.0],
        p1: [rx, 0.34, zStart + totalZ * 0.33],
        p2: [rx, 0.34, zStart + totalZ * 0.66],
        p3: [rx, 0.30, zStart + totalZ + 2.0],
        radius: 0.055,
        lane: 0,
        intensity: 0.95,
      });
    }

    globals.forEach((gt, gIdx) => {
      const isHead = gt.name.includes('head') || gIdx === globals.length - 1;
      const gz = isHead ? zStart + totalZ + 2.3 : zStart - 2.3;
      const gx = (gIdx - (globals.length - 1) * 0.5) * 3.8;
      const style = ROLE_STYLES[gt.role] || ROLE_STYLES.embedding;
      const isSel = gt.name === this.selectedTensorName;
      const isHov = this.hoveredHit?.tensorName === gt.name;
      const flag = isSel ? 2 : isHov ? 1 : gt.edited ? 4 : 0;

      pushInst(opaque, gx, 0.28, gz, 3.1, 0.28, 1.35, 0.25, 0.72, 0.14, 0.17, 0.24, 0.06, flag, 1, isHead ? 1 : 0, 0);
      pushInst(opaque, gx, 0.46, gz, 2.8, 0.10, 1.15, style.rough, style.metal, style.base[0], style.base[1], style.base[2], 0.42, flag, 1, isHead ? 1 : 0, 0);
      this.hitBoxes.push({
        min: [gx - 1.55, 0.05, gz - 0.72],
        max: [gx + 1.55, 1.35, gz + 0.72],
        tensorName: gt.name,
        label: gt.label,
        layer: null,
        role: gt.role,
        rms: gt.rms,
        frob: gt.frobenius_norm,
        shape: gt.shape,
      });
    });

    const labelStep = nLayers > 16 ? 4 : nLayers > 8 ? 2 : 1;
    layers.forEach((layer, li) => {
      const phase = nLayers > 1 ? li / (nLayers - 1) : 0.5;
      const z = zStart + li * zStep;
      const normRatio = clamp(layer.frobenius_norm / maxLayerNorm, 0.1, 1.0);
      const actHeat = this.layerHeat.get(layer.index) || 0;
      const isHybrid = layer.layer_type === 'hybrid_linear_attn';

      if (li % labelStep === 0 || li === nLayers - 1) {
        this.spatialAnchors.push({
          pos: [-waferW * 0.5 - 1.05, 0.52, z],
          text: `L${String(layer.index).padStart(2, '0')}`,
          kind: isHybrid ? 'cyan' : 'amber',
        });
      }

      const chassisTint = isHybrid ? [0.10, 0.14, 0.21] : [0.14, 0.13, 0.17];
      pushInst(
        opaque,
        0, 0.44, z,
        waferW, waferH, waferD,
        0.26, 0.82,
        chassisTint[0], chassisTint[1], chassisTint[2],
        0.04 + actHeat * 0.22,
        layer.has_edits ? 4 : 0, 1.0, phase, actHeat
      );

      const ledCol = layer.has_edits
        ? [0.18, 0.95, 0.65]
        : isHybrid
          ? [0.22, 0.76, 1.0]
          : [0.98, 0.68, 0.22];
      const ledEmit = 0.45 + normRatio * 0.75 + actHeat * 0.9;
      pushInst(opaque, -waferW * 0.503, 0.46, z, 0.09, 0.16, waferD * 0.86, 0.15, 0.3, ledCol[0], ledCol[1], ledCol[2], ledEmit, 0, 1, phase, actHeat);
      pushInst(opaque, waferW * 0.503, 0.46, z, 0.09, 0.16, waferD * 0.86, 0.15, 0.3, ledCol[0], ledCol[1], ledCol[2], ledEmit, 0, 1, phase, actHeat);

      const rawTensors = layer.tensors || [];
      const maxSlots = 12;
      const stepT = rawTensors.length > maxSlots ? Math.ceil(rawTensors.length / maxSlots) : 1;
      const visibleTensors = rawTensors.filter((_, idx) => idx % stepT === 0).slice(0, maxSlots);
      const mCount = Math.max(1, visibleTensors.length);
      const usableW = waferW - 1.35;
      const slotW = usableW / mCount;
      const modW = Math.min(1.65, slotW * 0.84);
      const modD = 1.08;

      visibleTensors.forEach((tensor, mi) => {
        const mx = -usableW * 0.5 + (mi + 0.5) * slotW;
        const style = ROLE_STYLES[tensor.role] || ROLE_STYLES.other;
        const isSel = tensor.name === this.selectedTensorName;
        const isHov = this.hoveredHit?.tensorName === tensor.name;
        const isLoraMatch =
          this.loraTargetHighlight.size > 0 &&
          [...this.loraTargetHighlight].some((tok) => tensor.name.includes(tok));
        const dimmed = this.loraTargetHighlight.size > 0 && !isLoraMatch;
        const flag = isSel ? 2 : isHov ? 1 : tensor.edited ? 4 : 0;

        const plinthAlpha = dimmed ? 0.32 : 1.0;
        const plinthList = dimmed ? trans : opaque;
        pushInst(
          plinthList,
          mx, 0.62, z,
          modW, 0.14, modD,
          style.rough, style.metal,
          dimmed ? 0.16 : style.base[0] * 0.42,
          dimmed ? 0.18 : style.base[1] * 0.42,
          dimmed ? 0.24 : style.base[2] * 0.42,
          isLoraMatch ? 0.62 : 0.10,
          flag, plinthAlpha, phase, actHeat
        );

        const mg = tensor.mini_grid || [];
        const mr = tensor.mini_rows || 8;
        const mc = tensor.mini_cols || 8;
        const subStep = mCount > 8 ? 2 : 1;
        const outR = Math.max(1, Math.floor(mr / subStep));
        const outC = Math.max(1, Math.floor(mc / subStep));
        const cellW = (modW * 0.86) / outC;
        const cellD = (modD * 0.82) / outR;
        const mAbs = Math.max(tensor.mini_abs_max || 0.05, 1e-6);

        let peakY = 0.72;
        for (let r = 0; r < outR; r++) {
          for (let c = 0; c < outC; c++) {
            const val = mg[(r * subStep) * mc + (c * subStep)] || 0;
            const mag = clamp(Math.abs(val) / mAbs, 0, 1);
            const h = 0.06 + Math.pow(mag, 0.82) * (0.68 + normRatio * 0.55) * (dimmed ? 0.45 : 1.0);
            const cx = mx - modW * 0.43 + (c + 0.5) * cellW;
            const cz = z - modD * 0.41 + (r + 0.5) * cellD;
            const cy = 0.69 + h * 0.5;
            if (0.69 + h > peakY) peakY = 0.69 + h;

            const mapped = mapWeightColor(
              val, mAbs, this.colorMode, tensor.edited ? val * 0.5 : 0, tensor.std || 0.05, mag, val
            );
            const colR = dimmed ? mapped.rgb[0] * 0.35 : mapped.rgb[0];
            const colG = dimmed ? mapped.rgb[1] * 0.35 : mapped.rgb[1];
            const colB = dimmed ? mapped.rgb[2] * 0.35 : mapped.rgb[2];
            const em = dimmed ? 0.02 : mapped.emit * (isLoraMatch ? 1.35 : 0.92);

            pushInst(
              plinthList,
              cx, cy, cz,
              cellW * 0.82, h, cellD * 0.82,
              0.20, 0.42,
              colR, colG, colB, em,
              flag, plinthAlpha, phase, actHeat
            );
          }
        }

        this.hitBoxes.push({
          min: [mx - modW * 0.52, 0.50, z - modD * 0.54],
          max: [mx + modW * 0.52, Math.max(1.35, peakY + 0.15), z + modD * 0.54],
          tensorName: tensor.name,
          label: tensor.label,
          layer: layer.index,
          role: tensor.role,
          rms: tensor.rms,
          frob: tensor.frobenius_norm,
          shape: tensor.shape,
          edited: tensor.edited,
        });
      });

      if (li < nLayers - 1) {
        const zNext = zStart + (li + 1) * zStep;
        const arcH = 1.95 + normRatio * 0.95;
        const laneX = (li % 2 === 0 ? -1 : 1) * (waferW * 0.24);
        curves.push({
          p0: [laneX, 0.95, z],
          p1: [laneX * 0.9, arcH, z + zStep * 0.3],
          p2: [laneX * 0.9, arcH, z + zStep * 0.7],
          p3: [laneX, 0.95, zNext],
          radius: 0.036,
          lane: 1,
          intensity: 0.65 + actHeat * 0.85,
        });
      }
    });
  }

  _buildMicroscopeScene(opaque, trans, curves, pushInst) {
    // Prefer binary Float32Array payload (supports up to 128x128 = 16,384 crystals!)
    const bin = this.binaryGrid;
    const detail = this.tensorDetail;
    const rows = bin ? bin.rows : detail.grid.rows;
    const cols = bin ? bin.cols : detail.grid.cols;
    const values = bin ? bin.values : detail.grid.values;
    const dValues = bin ? bin.delta : (detail?.delta_grid?.values || null);
    const impValues = bin ? bin.importance : (detail?.importance_grid?.values || null);
    const confValues = bin ? bin.conflict : (detail?.conflict_grid?.values || null);
    const absMax = Math.max((bin ? bin.abs_max : detail.grid.abs_max) || 0.05, 1e-6);
    const deltaAbsMax = Math.max((bin ? bin.delta_abs_max : detail?.delta_grid?.abs_max) || 0.01, 1e-6);
    const cropBox = bin ? bin.crop : detail.grid.crop;
    const matShape = bin ? bin.matrix_shape : detail.grid.matrix_shape;
    const stdVal = detail?.stats?.std || 0.05;

    const stageW = 16.4;
    const stageD = 16.4;
    this.footprintSize = [stageW * 0.56, stageD * 0.56];
    this.glowCenter = [0, 0, 0];
    this.glowColor = detail?.edited ? [0.16, 0.88, 0.62] : [0.30, 0.68, 1.0];

    pushInst(opaque, 0, 0.22, 0, stageW + 1.1, 0.36, stageD + 1.1, 0.26, 0.84, 0.10, 0.13, 0.18, 0.04, 0, 1, 0.5, 0);
    pushInst(opaque, 0, 0.44, -stageD * 0.5 - 0.35, stageW + 0.6, 0.14, 0.22, 0.20, 0.94, 0.90, 0.68, 0.28, 0.25, 0, 1, 0, 0);
    pushInst(opaque, -stageW * 0.5 - 0.35, 0.44, 0, 0.22, 0.14, stageD + 0.6, 0.20, 0.94, 0.28, 0.78, 0.96, 0.25, 0, 1, 0, 0);

    // Spatial Vernier Row/Col ticks along micrometer rails
    const [cr0, cr1, cc0, cc1] = cropBox;
    for (let t = 0; t <= 4; t++) {
      const frac = t / 4;
      const rTick = Math.round(lerp(cr0, cr1, frac));
      const cTick = Math.round(lerp(cc0, cc1, frac));
      this.spatialAnchors.push({
        pos: [-stageW * 0.5 - 0.95, 0.56, -stageD * 0.5 + frac * stageD],
        text: `r${rTick}`,
        kind: 'cyan',
      });
      this.spatialAnchors.push({
        pos: [-stageW * 0.5 + frac * stageW, 0.56, -stageD * 0.5 - 0.95],
        text: `c${cTick}`,
        kind: 'amber',
      });
    }

    // GQA Attention-Head Partition Divider Rails right across the 3D crystal stage!
    const headInfo = detail?.head_info;
    if (this.showHeadDividers && headInfo?.supports_heads && headInfo.num_heads > 1) {
      const nh = Math.min(16, headInfo.num_heads);
      const axis = headInfo.axis;
      for (let hIdx = 0; hIdx < nh; hIdx++) {
        const fracStart = hIdx / nh;
        const fracMid = (hIdx + 0.5) / nh;
        if (axis === 0) {
          const zLine = -stageD * 0.5 + fracStart * stageD;
          if (hIdx > 0) {
            pushInst(trans, 0, 0.95, zLine, stageW + 0.2, 0.9, 0.04, 0.1, 0.2, 0.35, 0.85, 1.0, 1.1, 0, 0.24, 0.5, 0);
          }
          if (nh <= 8 || hIdx % 2 === 0) {
            this.spatialAnchors.push({
              pos: [stageW * 0.5 + 0.85, 0.65, -stageD * 0.5 + fracMid * stageD],
              text: `H${hIdx}`,
              kind: 'emerald',
            });
          }
        } else {
          const xLine = -stageW * 0.5 + fracStart * stageW;
          if (hIdx > 0) {
            pushInst(trans, xLine, 0.95, 0, 0.04, 0.9, stageD + 0.2, 0.1, 0.2, 0.35, 0.85, 1.0, 1.1, 0, 0.24, 0.5, 0);
          }
          if (nh <= 8 || hIdx % 2 === 0) {
            this.spatialAnchors.push({
              pos: [-stageW * 0.5 + fracMid * stageW, 0.65, stageD * 0.5 + 0.85],
              text: `H${hIdx}`,
              kind: 'emerald',
            });
          }
        }
      }
    }

    const cellW = stageW / cols;
    const cellD = stageD / rows;
    const baseY = 0.42;
    const maxHeight = 4.6;

    this.microscopeMeta = {
      rows,
      cols,
      stageW,
      stageD,
      cellW,
      cellD,
      baseY,
      matrixShape: matShape,
      crop: cropBox,
      values,
      dValues,
    };

    const br = this.brushRect;
    const hovRow = this.hoveredHit?.gridRow ?? -1;
    const hovCol = this.hoveredHit?.gridCol ?? -1;

    for (let r = 0; r < rows; r++) {
      const cz = -stageD * 0.5 + (r + 0.5) * cellD;
      const phase = r / Math.max(rows - 1, 1);
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const val = values[idx] || 0;
        const dVal = dValues ? (dValues[idx] || 0) : 0;
        const impVal = impValues ? (impValues[idx] || 0) : 0;
        const confVal = confValues ? (confValues[idx] || 0) : 0;
        const cx = -stageW * 0.5 + (c + 0.5) * cellW;

        let metricMag = clamp(Math.abs(val) / absMax, 0, 1);
        if (this.colorMode === 'delta' && dValues) {
          metricMag = clamp(Math.abs(dVal) / deltaAbsMax, 0, 1);
        } else if (this.colorMode === 'importance' && impValues) {
          metricMag = clamp(impVal / 1.2, 0, 1);
        } else if (this.colorMode === 'conflict' && confValues) {
          metricMag = clamp(Math.abs(confVal) / Math.max(absMax * 0.55, 1e-6), 0, 1);
        }

        const h = 0.07 + Math.pow(metricMag, 0.78) * maxHeight;
        const cy = baseY + h * 0.5;

        const inBrush =
          br &&
          r >= Math.min(br.r0, br.r1) &&
          r <= Math.max(br.r0, br.r1) &&
          c >= Math.min(br.c0, br.c1) &&
          c <= Math.max(br.c0, br.c1);
        const isHov = r === hovRow && c === hovCol;
        const isEditedCell = Math.abs(dVal) > 1e-6;
        const flag = inBrush ? 3 : isHov ? 1 : (isEditedCell && this.colorMode !== 'delta' ? 4 : 0);

        const mapped = mapWeightColor(val, absMax, this.colorMode, dVal, stdVal, impVal, confVal);
        pushInst(
          opaque,
          cx, cy, cz,
          cellW * 0.86, h, cellD * 0.86,
          0.18, 0.36,
          mapped.rgb[0], mapped.rgb[1], mapped.rgb[2],
          mapped.emit,
          flag, 1.0, phase, 0
        );

        if (this.showGhostOverlay && dValues && rows <= 64 && Math.abs(dVal) > absMax * 0.03) {
          const origVal = val - dVal;
          const origH = 0.07 + Math.pow(clamp(Math.abs(origVal) / absMax, 0, 1), 0.78) * maxHeight;
          if (Math.abs(origH - h) > 0.14) {
            pushInst(
              trans,
              cx, baseY + origH * 0.5, cz,
              cellW * 0.90, origH, cellD * 0.90,
              0.12, 0.15,
              0.72, 0.86, 1.0, 0.55,
              0, 0.22, phase, 0
            );
          }
        }
      }
    }

    for (const out of (detail?.grid?.outliers || []).slice(0, 4)) {
      const gx = clamp(out.grid_col, 0, cols - 1);
      const gz = clamp(out.grid_row, 0, rows - 1);
      const cx = -stageW * 0.5 + (gx + 0.5) * cellW;
      const cz = -stageD * 0.5 + (gz + 0.5) * cellD;
      const mag = clamp(Math.abs(out.value) / absMax, 0, 1);
      const topY = baseY + 0.07 + Math.pow(mag, 0.78) * maxHeight;
      pushInst(trans, cx, topY + 0.65, cz, 0.08, 1.2, 0.08, 0.1, 0.2, 1.0, 0.38, 0.72, 1.8, 0, 0.55, 0.5, 0);
    }

    curves.push({
      p0: [-stageW * 0.52, 0.55, -stageD * 0.52],
      p1: [0, 0.55, -stageD * 0.55],
      p2: [stageW * 0.55, 0.55, -stageD * 0.55],
      p3: [stageW * 0.52, 0.55, stageD * 0.52],
      radius: 0.04,
      lane: 0,
      intensity: 0.85,
    });
  }

  _buildSpectralScene(opaque, trans, curves, pushInst) {
    const svd = this.tensorDetail.svd;
    const sv = svd.singular_values || [];
    const rank = Math.max(1, sv.length);
    const cutoff = clamp(this.svdCutoffRank || rank, 1, rank);
    const maxS = Math.max(sv[0] || 1e-6, 1e-6);

    this.footprintSize = [13.0, 9.0];
    this.glowCenter = [0, 0, 0];
    this.glowColor = [0.95, 0.68, 0.24];

    this.spatialAnchors.push(
      { pos: [-8.2, 2.5, 3.2], text: 'FACTOR A / Vᵀ', kind: 'cyan' },
      { pos: [8.2, 2.5, 3.2], text: 'FACTOR B / U', kind: 'amber' },
      { pos: [0, 6.4, -4.0], text: `SPECTRUM Σ (k=${cutoff}/${rank})`, kind: 'emerald' },
      { pos: [0, 3.8, 3.2], text: 'RANK-k RECONSTRUCTION', kind: 'amber' }
    );

    const specW = 14.5;
    pushInst(opaque, 0, 0.20, -4.0, specW + 1.2, 0.30, 2.8, 0.25, 0.86, 0.11, 0.14, 0.20, 0.05, 0, 1, 0.5, 0);
    const rodStep = specW / rank;
    for (let i = 0; i < rank; i++) {
      const kept = i < cutoff;
      const frac = sv[i] / maxS;
      const h = 0.25 + Math.pow(frac, 0.72) * 5.6;
      const x = -specW * 0.5 + (i + 0.5) * rodStep;
      const col = kept
        ? [lerp(0.14, 0.98, frac), lerp(0.62, 0.66, frac), lerp(0.96, 0.16, frac)]
        : [0.16, 0.20, 0.28];
      const list = kept ? opaque : trans;
      pushInst(
        list,
        x, 0.36 + h * 0.5, -4.0,
        Math.min(0.55, rodStep * 0.68), h, Math.min(0.55, rodStep * 0.68),
        kept ? 0.18 : 0.45, kept ? 0.68 : 0.25,
        col[0], col[1], col[2],
        kept ? (0.28 + frac * 0.95) : 0.04,
        0, kept ? 1.0 : 0.28, i / Math.max(rank - 1, 1), 0
      );
    }

    const cutX = -specW * 0.5 + cutoff * rodStep;
    pushInst(trans, cutX, 3.2, -4.0, 0.09, 6.2, 3.0, 0.1, 0.1, 1.0, 0.22, 0.48, 1.65, 0, 0.38, 0.5, 0);

    this._renderMiniFactorPlate(opaque, pushInst, svd.factor_a, -8.2, 3.2, 5.4, 5.4, 1.8, [0.20, 0.68, 0.98]);
    this._renderMiniFactorPlate(opaque, pushInst, svd.factor_b, 8.2, 3.2, 5.4, 5.4, 1.8, [0.95, 0.62, 0.22]);

    const uBasis = svd.u_basis;
    const vBasis = svd.v_basis;
    if (uBasis && vBasis) {
      const rOut = uBasis.rows;
      const cOut = vBasis.cols;
      const kMax = Math.min(cutoff, uBasis.cols, vBasis.rows, sv.length);
      const recon = new Float32Array(rOut * cOut);
      let rAbsMax = 1e-6;
      for (let r = 0; r < rOut; r++) {
        for (let c = 0; c < cOut; c++) {
          let acc = 0;
          for (let k = 0; k < kMax; k++) {
            const uk = uBasis.values[r * uBasis.cols + k] || 0;
            const vk = vBasis.values[k * vBasis.cols + c] || 0;
            acc += sv[k] * uk * vk;
          }
          recon[r * cOut + c] = acc;
          if (Math.abs(acc) > rAbsMax) rAbsMax = Math.abs(acc);
        }
      }

      const stageW = 8.4, stageD = 8.4;
      pushInst(opaque, 0, 0.22, 3.2, stageW + 0.7, 0.32, stageD + 0.7, 0.24, 0.85, 0.11, 0.14, 0.20, 0.06, 0, 1, 0.5, 0);
      const cw = stageW / cOut;
      const cd = stageD / rOut;
      for (let r = 0; r < rOut; r++) {
        for (let c = 0; c < cOut; c++) {
          const val = recon[r * cOut + c];
          const mag = clamp(Math.abs(val) / rAbsMax, 0, 1);
          const h = 0.06 + Math.pow(mag, 0.78) * 3.1;
          const mapped = mapWeightColor(val, rAbsMax, this.colorMode);
          pushInst(
            opaque,
            -stageW * 0.5 + (c + 0.5) * cw,
            0.40 + h * 0.5,
            3.2 - stageD * 0.5 + (r + 0.5) * cd,
            cw * 0.82, h, cd * 0.82,
            0.18, 0.42,
            mapped.rgb[0], mapped.rgb[1], mapped.rgb[2],
            mapped.emit, 0, 1, r / Math.max(rOut - 1, 1), 0
          );
        }
      }
    }

    curves.push({
      p0: [-8.2, 0.5, 0.5], p1: [-7.5, 2.2, -3.5], p2: [-3.5, 2.2, -4.0], p3: [-1.0, 0.8, -4.0],
      radius: 0.045, lane: 0, intensity: 0.9,
    });
    curves.push({
      p0: [8.2, 0.5, 0.5], p1: [7.5, 2.2, -3.5], p2: [3.5, 2.2, -4.0], p3: [1.0, 0.8, -4.0],
      radius: 0.045, lane: 0, intensity: 0.9,
    });
    curves.push({
      p0: [0, 0.8, -2.5], p1: [0, 2.0, -1.2], p2: [0, 2.0, -0.2], p3: [0, 0.8, -0.9],
      radius: 0.055, lane: 1, intensity: 1.1,
    });
  }

  _renderMiniFactorPlate(opaque, pushInst, grid, cx0, cz0, w, d, maxH, trimCol) {
    if (!grid || !grid.values) return;
    pushInst(opaque, cx0, 0.22, cz0, w + 0.55, 0.30, d + 0.55, 0.25, 0.82, 0.11, 0.14, 0.20, 0.04, 0, 1, 0.3, 0);
    pushInst(opaque, cx0, 0.39, cz0 - d * 0.5 - 0.18, w + 0.3, 0.08, 0.12, 0.18, 0.5, trimCol[0], trimCol[1], trimCol[2], 0.9, 0, 1, 0.3, 0);

    const stepR = Math.max(1, Math.floor(grid.rows / 16));
    const stepC = Math.max(1, Math.floor(grid.cols / 16));
    const outR = Math.max(1, Math.floor(grid.rows / stepR));
    const outC = Math.max(1, Math.floor(grid.cols / stepC));
    const cw = w / outC, cd = d / outR;
    const absMax = Math.max(grid.abs_max || 0.05, 1e-6);

    for (let r = 0; r < outR; r++) {
      for (let c = 0; c < outC; c++) {
        const val = grid.values[(r * stepR) * grid.cols + (c * stepC)] || 0;
        const mag = clamp(Math.abs(val) / absMax, 0, 1);
        const h = 0.05 + Math.pow(mag, 0.82) * maxH;
        const mapped = mapWeightColor(val, absMax, 'signed');
        pushInst(
          opaque,
          cx0 - w * 0.5 + (c + 0.5) * cw,
          0.38 + h * 0.5,
          cz0 - d * 0.5 + (r + 0.5) * cd,
          cw * 0.82, h, cd * 0.82,
          0.20, 0.45,
          mapped.rgb[0], mapped.rgb[1], mapped.rgb[2],
          mapped.emit * 0.85, 0, 1, 0.4, 0
        );
      }
    }
  }

  _buildTrajectoryScene(opaque, trans, curves, pushInst) {
    const steps = this.trajectory.steps || [];
    const layerNames = this.trajectory.layer_names || [];
    const nS = Math.max(1, steps.length);
    const nL = Math.max(1, layerNames.length);

    const useDelta = this.colorMode === 'delta';
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (const s of steps) {
      const arr = useDelta ? s.layer_deltas : s.layer_norms;
      for (const v of arr) {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }
    if (!Number.isFinite(minVal)) minVal = 0;
    if (!Number.isFinite(maxVal) || maxVal <= minVal) maxVal = minVal + 1e-4;
    const span = Math.max(1e-5, maxVal - minVal);

    const stageW = 16.5;
    const stageD = Math.min(18.0, Math.max(8.5, nS * 2.1));
    this.footprintSize = [stageW * 0.56, stageD * 0.56];
    this.glowCenter = [0, 0, 0];
    this.glowColor = [0.20, 0.78, 0.96];

    pushInst(opaque, 0, 0.20, 0, stageW + 1.2, 0.34, stageD + 1.2, 0.25, 0.84, 0.08, 0.11, 0.16, 0.03, 0, 1, 0.5, 0);

    const cw = stageW / nL;
    const cd = stageD / nS;
    const activeIdx = clamp(this.activeCheckpointIdx, 0, nS - 1);

    for (let si = 0; si < nS; si++) {
      const st = steps[si];
      const cz = -stageD * 0.5 + (si + 0.5) * cd;
      const isAct = si === activeIdx;
      const arr = useDelta ? st.layer_deltas : st.layer_norms;

      this.spatialAnchors.push({
        pos: [-stageW * 0.5 - 1.15, 0.65, cz],
        text: st.label,
        kind: isAct ? 'emerald' : 'cyan',
      });

      pushInst(
        opaque,
        0, 0.40, cz,
        stageW + 0.6, 0.08, 0.10,
        0.18, 0.75,
        isAct ? 0.16 : 0.30, isAct ? 0.88 : 0.38, isAct ? 0.65 : 0.52,
        isAct ? 0.85 : 0.08, 0, 1, si / Math.max(nS - 1, 1), 0
      );

      for (let li = 0; li < nL; li++) {
        const val = arr[li] || 0;
        const rel = clamp((val - minVal) / span, 0, 1);
        const h = 0.28 + Math.pow(rel, 0.85) * 4.8;
        const cx = -stageW * 0.5 + (li + 0.5) * cw;
        const mapped = mapWeightColor(rel, 1.0, 'magnitude');
        pushInst(
          opaque,
          cx, 0.42 + h * 0.5, cz,
          cw * 0.74, h, cd * 0.66,
          0.22, 0.52,
          mapped.rgb[0], mapped.rgb[1], mapped.rgb[2],
          mapped.emit * (isAct ? 0.95 : 0.58),
          isAct ? 4 : 0, 1.0, si / Math.max(nS - 1, 1), 0
        );
      }
    }

    const scanZ = -stageD * 0.5 + (activeIdx + 0.5) * cd;
    pushInst(trans, 0, 3.1, scanZ, stageW + 0.9, 5.8, 0.09, 0.1, 0.1, 0.20, 0.95, 0.85, 1.45, 0, 0.26, 0.5, 0);

    for (const lx of [-stageW * 0.48, 0, stageW * 0.48]) {
      curves.push({
        p0: [lx, 0.65, -stageD * 0.5],
        p1: [lx, 0.85, -stageD * 0.15],
        p2: [lx, 0.85, stageD * 0.15],
        p3: [lx, 0.65, stageD * 0.5],
        radius: 0.045,
        lane: 0,
        intensity: 0.95,
      });
    }
  }

  _uploadInstances(opaque, trans, curves, animateMorph = true) {
    const gl = this.gl;
    const nextGoal = new Float32Array(opaque);
    this.opaqueCount = nextGoal.length / 16;

    // Liquid-crystal GPU morphing when instance count matches previous state!
    if (animateMorph && this._morphGoal && this._morphGoal.length === nextGoal.length && nextGoal.length > 0) {
      this._morphPrev = new Float32Array(this._morphWork || this._morphGoal);
      this._morphGoal = nextGoal;
      this._morphWork = new Float32Array(nextGoal.length);
      this._morphT = 0.0;
    } else {
      this._morphPrev = null;
      this._morphGoal = nextGoal;
      this._morphWork = nextGoal;
      this._morphT = 1.0;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.opaqueInstBuf);
      gl.bufferData(gl.ARRAY_BUFFER, nextGoal, gl.DYNAMIC_DRAW);
    }

    this.transCount = trans.length / 16;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.transInstBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(trans), gl.DYNAMIC_DRAW);

    if (curves.length > 0) {
      const geo = createConduitGeometry(curves);
      this.conduitVertexCount = geo.vertexCount;
      gl.bindVertexArray(this.conduitVao);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.conduitPosBuf);
      gl.bufferData(gl.ARRAY_BUFFER, geo.positions, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.conduitNormBuf);
      gl.bufferData(gl.ARRAY_BUFFER, geo.normals, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.conduitMetaBuf);
      gl.bufferData(gl.ARRAY_BUFFER, geo.meta, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);
    } else {
      this.conduitVertexCount = 0;
    }

    if (this.callbacks.onStats) {
      this.callbacks.onStats({
        instances: this.opaqueCount + this.transCount,
        fps: this.fps,
      });
    }
  }

  _stepMorph(dt) {
    if (this._morphT >= 1.0 || !this._morphPrev || !this._morphGoal || !this._morphWork) return;
    this._morphT = Math.min(1.0, this._morphT + dt / this._morphDuration);
    const e = 1.0 - Math.pow(1.0 - this._morphT, 3.0); // cubic ease-out
    const prev = this._morphPrev;
    const goal = this._morphGoal;
    const work = this._morphWork;
    for (let i = 0; i < goal.length; i++) {
      work[i] = prev[i] + (goal[i] - prev[i]) * e;
    }
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.opaqueInstBuf);
    gl.bufferData(gl.ARRAY_BUFFER, work, gl.DYNAMIC_DRAW);
  }

  // -------------------------------------------------------------------------
  // Render Pipeline (Shadow -> Reflection -> HDR Scene -> Bloom -> SSAO+DoF+ACES)
  // -------------------------------------------------------------------------

  _startLoop() {
    const tick = (now) => {
      const dt = Math.min(0.05, (now - this._lastTime) * 0.001);
      this._lastTime = now;
      this._elapsed += dt;

      this._updateCamera(dt);
      this._updatePulse(dt);
      this._stepMorph(dt);
      this._renderFrame();

      this._frameCount++;
      if (now - this._fpsTimer >= 500) {
        this.fps = Math.round((this._frameCount * 1000) / (now - this._fpsTimer));
        this._frameCount = 0;
        this._fpsTimer = now;
        if (this.callbacks.onStats) {
          this.callbacks.onStats({
            instances: this.opaqueCount + this.transCount,
            fps: this.fps,
          });
        }
      }

      this._rafId = requestAnimationFrame(tick);
    };
    this._rafId = requestAnimationFrame(tick);
  }

  _updateCamera(dt) {
    if (this.autoOrbit && !this._isDragging) {
      this.cam.goalYaw += dt * 0.22;
    }
    const smooth = 1.0 - Math.exp(-9.5 * dt);
    this.cam.yaw = lerp(this.cam.yaw, this.cam.goalYaw, smooth);
    this.cam.pitch = lerp(this.cam.pitch, clamp(this.cam.goalPitch, 0.06, 1.48), smooth);
    this.cam.distance = lerp(this.cam.distance, clamp(this.cam.goalDistance, 4.5, 90), smooth);
    for (let i = 0; i < 3; i++) {
      this.cam.target[i] = lerp(this.cam.target[i], this.cam.goalTarget[i], smooth);
    }

    const cp = Math.cos(this.cam.pitch);
    const sp = Math.sin(this.cam.pitch);
    const cy = Math.cos(this.cam.yaw);
    const sy = Math.sin(this.cam.yaw);
    this.cameraPos[0] = this.cam.target[0] + this.cam.distance * cp * sy;
    this.cameraPos[1] = this.cam.target[1] + this.cam.distance * sp;
    this.cameraPos[2] = this.cam.target[2] + this.cam.distance * cp * cy;

    const aspect = this.canvas.width / Math.max(this.canvas.height, 1);
    const proj = mat4Perspective((42 * Math.PI) / 180, aspect, 0.2, 260);
    const view = mat4LookAt(this.cameraPos, this.cam.target, vec3(0, 1, 0));
    mat4Multiply(proj, view, this.viewProj);
    mat4Invert(this.viewProj, this.invViewProj);

    // Light-space orthographic matrix for key softbox shadow mapping
    const rig = STUDIO_RIGS[this.rigKey] || STUDIO_RIGS.obsidian;
    const ld = normalize3(rig.keyDir);
    const lightEye = vec3(ld[0] * 36, ld[1] * 36, ld[2] * 36);
    const lView = mat4LookAt(lightEye, vec3(0, 1.0, 0), vec3(0, 1, 0));
    const extent = Math.max(22, this.footprintSize[0] * 1.4, this.footprintSize[1] * 1.3);
    const lProj = mat4Ortho(-extent, extent, -extent, extent, 2.0, 95.0);
    mat4Multiply(lProj, lView, this.lightViewProj);
  }

  _updatePulse(dt) {
    if (this.pulseWave >= -0.2 && this.pulseWave <= 1.25) {
      const dir = this.pulseMode > 0.5 ? -1 : 1;
      this.pulseWave += dt * 0.72 * dir;
      if (this.pulseWave > 1.25 || this.pulseWave < -0.25) {
        this.pulseWave = -1.0;
        this.pulseSpeed = 0.35;
      }
    }
  }

  _setPbrUniforms(prog, rig, mirrorY = 1.0) {
    const gl = this.gl;
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uViewProj'), false, this.viewProj);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uLightViewProj'), false, this.lightViewProj);
    gl.uniform1f(gl.getUniformLocation(prog, 'uMirrorY'), mirrorY);
    gl.uniform3fv(gl.getUniformLocation(prog, 'uCameraPos'), this.cameraPos);
    gl.uniform3fv(gl.getUniformLocation(prog, 'uBgTop'), rig.bgTop);
    gl.uniform3fv(gl.getUniformLocation(prog, 'uBgHorizon'), rig.bgHorizon);
    gl.uniform3fv(gl.getUniformLocation(prog, 'uKeyDir'), rig.keyDir);
    gl.uniform3fv(gl.getUniformLocation(prog, 'uKeyColor'), rig.keyColor);
    gl.uniform3fv(gl.getUniformLocation(prog, 'uRimDir'), rig.rimDir);
    gl.uniform3fv(gl.getUniformLocation(prog, 'uRimColor'), rig.rimColor);
    gl.uniform3fv(gl.getUniformLocation(prog, 'uFillDir'), rig.fillDir);
    gl.uniform3fv(gl.getUniformLocation(prog, 'uFillColor'), rig.fillColor);
    gl.uniform3fv(gl.getUniformLocation(prog, 'uAmbient'), rig.ambient);
    gl.uniform1f(gl.getUniformLocation(prog, 'uTime'), this._elapsed);
    gl.uniform1f(gl.getUniformLocation(prog, 'uPulseWave'), this.pulseWave);
    gl.uniform1f(gl.getUniformLocation(prog, 'uPulseMode'), this.pulseMode);

    if (this.fboShadow && this.enableShadows && mirrorY > 0) {
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, this.fboShadow.depthTex);
      gl.uniform1i(gl.getUniformLocation(prog, 'uShadowMap'), 3);
      gl.uniform1f(gl.getUniformLocation(prog, 'uShadowEnabled'), 1.0);
    } else {
      gl.uniform1f(gl.getUniformLocation(prog, 'uShadowEnabled'), 0.0);
    }
  }

  _renderFrame() {
    const gl = this.gl;
    const rig = STUDIO_RIGS[this.rigKey] || STUDIO_RIGS.obsidian;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Pass 0: Real-Time PCF Shadow Map (1024x1024 depth pass)
    if (this.enableShadows && this.fboShadow && this.opaqueCount > 0) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboShadow.fb);
      gl.viewport(0, 0, this.fboShadow.width, this.fboShadow.height);
      gl.enable(gl.DEPTH_TEST);
      gl.depthMask(true);
      gl.clear(gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.FRONT);
      gl.useProgram(this.progShadow);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.progShadow, 'uLightViewProj'), false, this.lightViewProj);
      gl.bindVertexArray(this.opaqueVao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, this.boxVertexCount, this.opaqueCount);
      gl.cullFace(gl.BACK);
    }

    // Pass 0b: Real-Time Planar Floor Reflection (half-res Y-mirrored pass)
    if (this.enableReflections && this.fboReflect && this.opaqueCount > 0) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboReflect.fb);
      gl.viewport(0, 0, this.fboReflect.width, this.fboReflect.height);
      gl.clearColor(rig.bgHorizon[0] * 0.5, rig.bgHorizon[1] * 0.5, rig.bgHorizon[2] * 0.5, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.depthMask(true);
      gl.disable(gl.BLEND);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.FRONT); // mirrored Y flips winding
      gl.useProgram(this.progPbr);
      this._setPbrUniforms(this.progPbr, rig, -1.0);
      gl.bindVertexArray(this.opaqueVao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, this.boxVertexCount, this.opaqueCount);
      gl.cullFace(gl.BACK);
    }

    // Pass 1: Main HDR Scene + Readable Depth Buffer
    if (this.postReady) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboScene.fb);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    gl.viewport(0, 0, w, h);
    gl.clearColor(rig.bgTop[0], rig.bgTop[1], rig.bgTop[2], 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // 1a. Studio Dome Skybox
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.disable(gl.CULL_FACE);
    gl.useProgram(this.progSkybox);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.progSkybox, 'uInvViewProj'), false, this.invViewProj);
    gl.uniform3fv(gl.getUniformLocation(this.progSkybox, 'uCameraPos'), this.cameraPos);
    gl.uniform3fv(gl.getUniformLocation(this.progSkybox, 'uBgTop'), rig.bgTop);
    gl.uniform3fv(gl.getUniformLocation(this.progSkybox, 'uBgHorizon'), rig.bgHorizon);
    gl.uniform3fv(gl.getUniformLocation(this.progSkybox, 'uKeyDir'), rig.keyDir);
    gl.uniform3fv(gl.getUniformLocation(this.progSkybox, 'uRimDir'), rig.rimDir);
    gl.uniform3fv(gl.getUniformLocation(this.progSkybox, 'uKeyColor'), rig.keyColor);
    gl.uniform3fv(gl.getUniformLocation(this.progSkybox, 'uRimColor'), rig.rimColor);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // 1b. Anodized Optical Bench Floor (with Planar Reflection + Cast Shadows!)
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.useProgram(this.progFloor);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.progFloor, 'uViewProj'), false, this.viewProj);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.progFloor, 'uLightViewProj'), false, this.lightViewProj);
    gl.uniform1f(gl.getUniformLocation(this.progFloor, 'uFloorY'), 0.0);
    gl.uniform3fv(gl.getUniformLocation(this.progFloor, 'uCameraPos'), this.cameraPos);
    gl.uniform3fv(gl.getUniformLocation(this.progFloor, 'uFloorColor'), rig.floorColor);
    gl.uniform3fv(gl.getUniformLocation(this.progFloor, 'uGridColor'), rig.gridColor);
    gl.uniform3fv(gl.getUniformLocation(this.progFloor, 'uBgHorizon'), rig.bgHorizon);
    gl.uniform3fv(gl.getUniformLocation(this.progFloor, 'uKeyDir'), rig.keyDir);
    gl.uniform3fv(gl.getUniformLocation(this.progFloor, 'uKeyColor'), rig.keyColor);
    gl.uniform3fv(gl.getUniformLocation(this.progFloor, 'uRimDir'), rig.rimDir);
    gl.uniform3fv(gl.getUniformLocation(this.progFloor, 'uRimColor'), rig.rimColor);
    gl.uniform2fv(gl.getUniformLocation(this.progFloor, 'uFootprintSize'), this.footprintSize);
    gl.uniform3fv(gl.getUniformLocation(this.progFloor, 'uGlowCenter'), this.glowCenter);
    gl.uniform3fv(gl.getUniformLocation(this.progFloor, 'uGlowColor'), this.glowColor);
    gl.uniform1f(gl.getUniformLocation(this.progFloor, 'uTime'), this._elapsed);
    if (this.enableReflections && this.fboReflect) {
      gl.activeTexture(gl.TEXTURE4);
      gl.bindTexture(gl.TEXTURE_2D, this.fboReflect.tex);
      gl.uniform1i(gl.getUniformLocation(this.progFloor, 'uReflectTex'), 4);
      gl.uniform1f(gl.getUniformLocation(this.progFloor, 'uReflectStrength'), this.rigKey === 'cleanroom' ? 0.25 : 0.62);
    } else {
      gl.uniform1f(gl.getUniformLocation(this.progFloor, 'uReflectStrength'), 0.0);
    }
    if (this.enableShadows && this.fboShadow) {
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, this.fboShadow.depthTex);
      gl.uniform1i(gl.getUniformLocation(this.progFloor, 'uShadowMap'), 3);
      gl.uniform1f(gl.getUniformLocation(this.progFloor, 'uShadowEnabled'), 1.0);
    } else {
      gl.uniform1f(gl.getUniformLocation(this.progFloor, 'uShadowEnabled'), 0.0);
    }
    gl.bindVertexArray(this.floorVao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // 1c. Opaque PBR Instanced Crystals & Wafer Chassis
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.useProgram(this.progPbr);
    this._setPbrUniforms(this.progPbr, rig, 1.0);
    if (this.opaqueCount > 0) {
      gl.bindVertexArray(this.opaqueVao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, this.boxVertexCount, this.opaqueCount);
    }

    // 1d. Volumetric Optical Signal Conduits
    if (this.showConduits && this.conduitVertexCount > 0) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      gl.disable(gl.CULL_FACE);
      gl.useProgram(this.progConduit);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.progConduit, 'uViewProj'), false, this.viewProj);
      gl.uniform1f(gl.getUniformLocation(this.progConduit, 'uMirrorY'), 1.0);
      gl.uniform3fv(gl.getUniformLocation(this.progConduit, 'uCameraPos'), this.cameraPos);
      gl.uniform1f(gl.getUniformLocation(this.progConduit, 'uTime'), this._elapsed);
      gl.uniform1f(gl.getUniformLocation(this.progConduit, 'uPulseSpeed'), this.pulseSpeed);
      gl.uniform1f(gl.getUniformLocation(this.progConduit, 'uPulseMode'), this.pulseMode);
      gl.bindVertexArray(this.conduitVao);
      gl.drawArrays(gl.TRIANGLES, 0, this.conduitVertexCount);
    }

    // 1e. Translucent Optical Glass / Ghost Pre-Edit Reference Overlay / Head Dividers
    if (this.transCount > 0) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      gl.enable(gl.CULL_FACE);
      gl.useProgram(this.progPbr);
      this._setPbrUniforms(this.progPbr, rig, 1.0);
      gl.bindVertexArray(this.transVao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, this.boxVertexCount, this.transCount);
    }
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);

    if (!this.postReady) return;

    // Pass 2: HDR Bright-Pass Extraction
    gl.disable(gl.DEPTH_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboBright.fb);
    gl.viewport(0, 0, this.fboBright.width, this.fboBright.height);
    gl.useProgram(this.progBright);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.fboScene.tex);
    gl.uniform1i(gl.getUniformLocation(this.progBright, 'uScene'), 0);
    gl.uniform1f(gl.getUniformLocation(this.progBright, 'uThreshold'), this.rigKey === 'cleanroom' ? 1.35 : 0.85);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Pass 3-4: Dual-Tier Separable Gaussian Bloom
    this._runBlurPass(this.fboBright.tex, this.fboBlur1A, 1, 0);
    this._runBlurPass(this.fboBlur1A.tex, this.fboBlur1B, 0, 1);
    this._runBlurPass(this.fboBlur1B.tex, this.fboBlur2A, 1.4, 0);
    this._runBlurPass(this.fboBlur2A.tex, this.fboBlur2B, 0, 1.4);

    // Pass 5: Final SSAO + Bokeh DoF + ACES Filmic Composite
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.progComposite);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.fboScene.tex);
    gl.uniform1i(gl.getUniformLocation(this.progComposite, 'uScene'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.fboBlur1B.tex);
    gl.uniform1i(gl.getUniformLocation(this.progComposite, 'uBloom1'), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.fboBlur2B.tex);
    gl.uniform1i(gl.getUniformLocation(this.progComposite, 'uBloom2'), 2);
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, this.fboScene.depthTex);
    gl.uniform1i(gl.getUniformLocation(this.progComposite, 'uDepth'), 5);

    gl.uniform2f(gl.getUniformLocation(this.progComposite, 'uTexelSize'), 1.0 / w, 1.0 / h);
    gl.uniform1f(gl.getUniformLocation(this.progComposite, 'uExposure'), this.exposureOverride ?? rig.exposure);
    gl.uniform1f(gl.getUniformLocation(this.progComposite, 'uBloomStrength'), this.bloomOverride ?? rig.bloomStrength);
    gl.uniform1f(gl.getUniformLocation(this.progComposite, 'uSsaoStrength'), this.ssaoStrength);
    gl.uniform1f(gl.getUniformLocation(this.progComposite, 'uDofStrength'), this.dofStrength);
    gl.uniform1f(gl.getUniformLocation(this.progComposite, 'uFocusDist'), this.cam.distance);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Project floating callout & 3D spatial ruler labels
    const cw = this.canvas.clientWidth;
    const ch = this.canvas.clientHeight;
    if (this.callbacks.onProjection && this.selectedTensorName && this.viewMode === 'stack') {
      const hit = this.hitBoxes.find((hb) => hb.tensorName === this.selectedTensorName);
      if (hit) {
        const pt = projectPoint(
          [(hit.min[0] + hit.max[0]) * 0.5, hit.max[1] + 0.12, (hit.min[2] + hit.max[2]) * 0.5],
          this.viewProj,
          cw,
          ch
        );
        this.callbacks.onProjection(pt, hit);
      } else {
        this.callbacks.onProjection(null, null);
      }
    } else if (this.callbacks.onProjection) {
      this.callbacks.onProjection(null, null);
    }

    if (this.callbacks.onSpatialLabels) {
      if (!this.showSpatialRulers || !this.spatialAnchors.length) {
        this.callbacks.onSpatialLabels([]);
      } else {
        const projected = [];
        for (const a of this.spatialAnchors) {
          const p = projectPoint(a.pos, this.viewProj, cw, ch);
          if (p && p.x > 16 && p.x < cw - 16 && p.y > 46 && p.y < ch - 46) {
            projected.push({ x: p.x, y: p.y, text: a.text, kind: a.kind });
          }
        }
        this.callbacks.onSpatialLabels(projected);
      }
    }
  }

  _runBlurPass(srcTex, dstFbo, dx, dy) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, dstFbo.fb);
    gl.viewport(0, 0, dstFbo.width, dstFbo.height);
    gl.useProgram(this.progBlur);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(gl.getUniformLocation(this.progBlur, 'uTex'), 0);
    gl.uniform2f(gl.getUniformLocation(this.progBlur, 'uTexelStep'), dx / dstFbo.width, dy / dstFbo.height);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // -------------------------------------------------------------------------
  // Picking, Orbit/Pan/Zoom, and 3D Surgical Brush
  // -------------------------------------------------------------------------

  _pickAtClient(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const ndcX = ((clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
    const ndcY = 1 - ((clientY - rect.top) / Math.max(rect.height, 1)) * 2;
    const { origin, dir } = screenToRay(ndcX, ndcY, this.invViewProj);

    if (this.viewMode === 'microscope' && this.microscopeMeta) {
      const m = this.microscopeMeta;
      const pt = rayIntersectsPlaneY(origin, dir, m.baseY + 0.35);
      if (!pt) return null;
      const col = Math.floor((pt[0] + m.stageW * 0.5) / m.cellW);
      const row = Math.floor((pt[2] + m.stageD * 0.5) / m.cellD);
      if (row >= 0 && row < m.rows && col >= 0 && col < m.cols) {
        const idx = row * m.cols + col;
        const [r0, r1, c0, c1] = m.crop;
        const origRow = Math.floor(r0 + ((row + 0.5) * (r1 - r0)) / m.rows);
        const origCol = Math.floor(c0 + ((col + 0.5) * (c1 - c0)) / m.cols);
        return {
          kind: 'cell',
          gridRow: row,
          gridCol: col,
          origRow,
          origCol,
          value: m.values[idx] || 0,
          delta: m.dValues ? (m.dValues[idx] || 0) : 0,
          screenX: clientX - rect.left,
          screenY: clientY - rect.top,
        };
      }
      return null;
    }

    if (this.viewMode === 'stack') {
      let best = null;
      let bestT = Infinity;
      for (const hb of this.hitBoxes) {
        const t = rayIntersectsAABB(origin, dir, hb.min, hb.max);
        if (t !== null && t < bestT) {
          bestT = t;
          best = hb;
        }
      }
      if (best) {
        return {
          kind: 'tensor',
          ...best,
          screenX: clientX - rect.left,
          screenY: clientY - rect.top,
        };
      }
    }
    return null;
  }

  _bindEvents() {
    this._isDragging = false;
    let dragMode = 'orbit';
    let lastX = 0, lastY = 0;
    let downX = 0, downY = 0;

    this.canvas.addEventListener('pointerdown', (e) => {
      this.canvas.setPointerCapture(e.pointerId);
      downX = lastX = e.clientX;
      downY = lastY = e.clientY;

      if (this.viewMode === 'microscope' && (this.brushMode || e.shiftKey) && e.button === 0) {
        const hit = this._pickAtClient(e.clientX, e.clientY);
        if (hit && hit.kind === 'cell') {
          this._brushDragging = true;
          this._brushAnchor = { r: hit.gridRow, c: hit.gridCol };
          this.brushRect = { r0: hit.gridRow, r1: hit.gridRow, c0: hit.gridCol, c1: hit.gridCol };
          this.rebuildScene(false);
          this._emitBrushSelection();
          return;
        }
      }

      this._isDragging = true;
      dragMode = (e.button === 2 || e.altKey || e.ctrlKey) ? 'pan' : 'orbit';
    });

    this.canvas.addEventListener('pointermove', (e) => {
      if (this._brushDragging && this._brushAnchor) {
        const hit = this._pickAtClient(e.clientX, e.clientY);
        if (hit && hit.kind === 'cell') {
          this.brushRect = {
            r0: this._brushAnchor.r,
            r1: hit.gridRow,
            c0: this._brushAnchor.c,
            c1: hit.gridCol,
          };
          this.rebuildScene(false);
          this._emitBrushSelection();
        }
        return;
      }

      if (this._isDragging) {
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;

        if (dragMode === 'orbit') {
          this.cam.goalYaw -= dx * 0.0068;
          this.cam.goalPitch = clamp(this.cam.goalPitch + dy * 0.0058, 0.08, 1.48);
        } else {
          const panScale = this.cam.distance * 0.0014;
          const forward = normalize3(sub3(this.cam.target, this.cameraPos));
          const right = normalize3(cross3(forward, vec3(0, 1, 0)));
          const up = normalize3(cross3(right, forward));
          add3(this.cam.goalTarget, scale3(right, -dx * panScale), this.cam.goalTarget);
          add3(this.cam.goalTarget, scale3(up, dy * panScale), this.cam.goalTarget);
        }
        return;
      }

      const hit = this._pickAtClient(e.clientX, e.clientY);
      const prevKey = this.hoveredHit ? `${this.hoveredHit.tensorName}:${this.hoveredHit.gridRow}:${this.hoveredHit.gridCol}` : '';
      const nextKey = hit ? `${hit.tensorName}:${hit.gridRow}:${hit.gridCol}` : '';
      this.hoveredHit = hit;
      this.canvas.style.cursor = hit ? (this.brushMode && this.viewMode === 'microscope' ? 'crosshair' : 'pointer') : 'grab';
      if (prevKey !== nextKey && (this.viewMode === 'stack' || this.viewMode === 'microscope')) {
        this.rebuildScene(false);
      }
      if (this.callbacks.onHover) this.callbacks.onHover(hit);
    });

    this.canvas.addEventListener('pointerup', (e) => {
      if (this._brushDragging) {
        this._brushDragging = false;
        this._brushAnchor = null;
        return;
      }
      this._isDragging = false;
      const moved = Math.hypot(e.clientX - downX, e.clientY - downY);
      if (moved < 5 && e.button === 0) {
        const hit = this._pickAtClient(e.clientX, e.clientY);
        if (hit && hit.kind === 'tensor' && this.callbacks.onSelectTensor) {
          this.callbacks.onSelectTensor(hit.tensorName, false);
        }
      }
    });

    this.canvas.addEventListener('dblclick', (e) => {
      const hit = this._pickAtClient(e.clientX, e.clientY);
      if (hit && hit.kind === 'tensor' && this.callbacks.onSelectTensor) {
        this.callbacks.onSelectTensor(hit.tensorName, true);
      }
    });

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = Math.exp(e.deltaY * 0.0011);
      this.cam.goalDistance = clamp(this.cam.goalDistance * factor, 4.8, 86);
    }, { passive: false });

    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  _emitBrushSelection() {
    if (!this.brushRect || !this.microscopeMeta || !this.callbacks.onBrushRegion) return;
    const m = this.microscopeMeta;
    const [cr0, cr1, cc0, cc1] = m.crop;
    const gr0 = Math.min(this.brushRect.r0, this.brushRect.r1);
    const gr1 = Math.max(this.brushRect.r0, this.brushRect.r1);
    const gc0 = Math.min(this.brushRect.c0, this.brushRect.c1);
    const gc1 = Math.max(this.brushRect.c0, this.brushRect.c1);

    const rowStart = Math.floor(cr0 + (gr0 * (cr1 - cr0)) / m.rows);
    const rowEnd = Math.max(rowStart + 1, Math.ceil(cr0 + ((gr1 + 1) * (cr1 - cr0)) / m.rows));
    const colStart = Math.floor(cc0 + (gc0 * (cc1 - cc0)) / m.cols);
    const colEnd = Math.max(colStart + 1, Math.ceil(cc0 + ((gc1 + 1) * (cc1 - cc0)) / m.cols));

    this.callbacks.onBrushRegion({ row_start: rowStart, row_end: rowEnd, col_start: colStart, col_end: colEnd });
  }
}
