/**
 * StateSet 3D Weight Studio v2 — Interactive Application Controller.
 *
 * Connects:
 * - Binary Float32Array 128x128 Microscope streaming + Crop Zoom/Pan
 * - Wanda/Fisher Importance & Adapter Sign-Conflict 3D visualization
 * - Surgical Weight Editor (Scale, Magnitude/Wanda Prune, Clamp, SVD, Head Surgery,
 *   ROME Rank-1 w/ Drift Check, 3D Brush, TIES/DARE/SLERP Merge)
 * - Finetuning Workbench (Datasets & Traces, Workspace Validation, VRAM/Cost, Live PyTorch SSE)
 * - Interactive Model Playground (PyTorch Hooks / Remote vLLM / Weight Probe,
 *   Layer-by-Layer Logit Lens, Causal Activation Patching, A/B Diff & Reward Rubric)
 */

import {
  drawHistogramChart,
  drawLayerProfileChart,
  drawSvdChart,
  drawTrainingTelemetryChart,
} from './charts.js';
import { WeightStudioRenderer } from './renderer.js';

const API = '/api/weight-studio';

const state = {
  models: [],
  modelId: null,
  checkpointId: null,
  secondaryModelId: null,
  structure: null,
  selectedTensorName: null,
  tensorDetail: null,
  binaryGrid: null,
  activeCrop: null, // [r0, r1, c0, c1] or null
  trajectory: null,
  editHistory: null,
  activeOp: 'scale',
  brushRegion: { row_start: 0, row_end: 8, col_start: 0, col_end: 8 },
  ftCatalog: null,
  ftAlgorithm: 'gspo',
  ftSelectedTargets: new Set(['q_proj', 'k_proj', 'v_proj', 'o_proj']),
  ftDatasetPath: '',
  ftJobId: null,
  ftStream: null,
  ftPollTimer: null,
  ftHistory: [],
  causalPatches: [], // [{ layer, action }]
  circuitData: null,
  steeringPresets: [],
  activeSteeringVector: null, // { vector_id, name, recommended_layer, ... }
  steeringAlpha: 0.0,
  saeClamps: {}, // { feature_id: value }
  splitCompare: false,
  splitRatio: 0.5,
  splitTargetSpec: 'base', // 'base' | 'cp:<id>' | 'model:<id>'
  dispatchPollTimer: null,
  trajPlayTimer: null,
  lastInferResult: null,
};

const $ = (id) => document.getElementById(id);
let renderer = null;

async function apiGet(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function showToast(msg, duration = 2600) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => el.classList.add('hidden'), duration);
}

function formatParams(n) {
  if (!n) return '0';
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

async function init() {
  const canvas = $('gl-canvas');
  renderer = new WeightStudioRenderer(canvas, {
    onStats: ({ instances, fps }) => {
      $('hud-fps').textContent = `${fps} FPS`;
      $('hud-instances').textContent = `${instances.toLocaleString()} CRYSTALS`;
    },
    onSelectTensor: async (tensorName, openMicroscope) => {
      await selectTensor(tensorName, { focusCamera: !openMicroscope });
      if (openMicroscope) setViewMode('microscope');
    },
    onSelectAttentionHead: (headIdx) => {
      const sel = $('attn-head-select');
      if (sel) sel.value = String(headIdx);
      renderer.setData({ attentionHeadFilter: headIdx });
      renderer._fitCameraToViewMode();
      showToast(`Focused Attention Head H${headIdx} — click ✂ Ablate Head to perform live head surgery`);
    },
    onHover: (hit) => renderHoverTooltip(hit),
    onProjection: (pt, hit) => render3DCallout(pt, hit),
    onSpatialLabels: (labels) => renderSpatialLabels(labels),
    onBrushRegion: (region) => {
      state.brushRegion = region;
      if (state.activeOp === 'region_edit') renderOpParams();
      showToast(
        `3D Brush Patch: rows [${region.row_start}:${region.row_end}] × cols [${region.col_start}:${region.col_end}]`,
        1600
      );
    },
  });

  window.addEventListener('resize', () => {
    renderer.resize();
    redrawVisibleCharts();
  });

  bindTopbarControls();
  bindWorkbenchTabs();
  bindEditControls();
  bindFinetuneControls();
  bindInferenceControls();
  bindExportModal();
  bindSplitCurtain();
  bindKeyboardShortcuts();

  await Promise.all([
    loadFinetuneCatalog(),
    loadDatasets(),
    loadInferenceScenarios(),
    loadSteeringPresets(),
    loadModels(false),
  ]);
}

// ---------------------------------------------------------------------------
// Spatial 3D Labels (Pooled DOM elements for zero GC jitter)
// ---------------------------------------------------------------------------

let _labelPool = [];
function renderSpatialLabels(labels) {
  const layer = $('spatial-labels-layer');
  if (!layer) return;
  while (_labelPool.length < labels.length) {
    const el = document.createElement('div');
    el.className = 'spatial-tag';
    layer.appendChild(el);
    _labelPool.push(el);
  }
  for (let i = 0; i < _labelPool.length; i++) {
    const el = _labelPool[i];
    if (i < labels.length) {
      const l = labels[i];
      el.style.display = 'block';
      el.style.left = `${l.x.toFixed(1)}px`;
      el.style.top = `${l.y.toFixed(1)}px`;
      el.textContent = l.text;
      el.className = `spatial-tag ${l.kind || ''}`;
    } else {
      el.style.display = 'none';
    }
  }
}

// ---------------------------------------------------------------------------
// Model & Tensor Loading (with Binary Float32Array 128x128 Streaming!)
// ---------------------------------------------------------------------------

async function loadModels(refresh = false) {
  const data = await apiGet(`/models${refresh ? '?refresh=true' : ''}`);
  state.models = data.models || [];
  const select = $('model-select');
  select.innerHTML = '';

  const localGroup = document.createElement('optgroup');
  localGroup.label = 'Local Real Checkpoints (.safetensors)';
  const presetGroup = document.createElement('optgroup');
  presetGroup.label = 'StateSet Model Presets (Architecture Twins)';

  for (const m of state.models) {
    const opt = document.createElement('option');
    opt.value = m.model_id;
    const trajTag = m.checkpoints && m.checkpoints.length > 1 ? ` [${m.checkpoints.length} steps]` : '';
    const sizeTag = m.size_mb > 0 ? ` · ${m.size_mb}MB` : '';
    opt.textContent = `${m.display_name}${trajTag}${sizeTag}`;
    if (m.kind === 'preset_twin') presetGroup.appendChild(opt);
    else localGroup.appendChild(opt);
  }
  if (localGroup.children.length) select.appendChild(localGroup);
  if (presetGroup.children.length) select.appendChild(presetGroup);

  if (state.models.length > 0) {
    const preferred =
      state.modelId && state.models.some((m) => m.model_id === state.modelId)
        ? state.modelId
        : state.models[0].model_id;
    select.value = preferred;
    // Default secondary model for Sign Conflict / TIES merge
    const alt = state.models.find((m) => m.model_id !== preferred && m.kind === state.models[0].kind) || state.models[1] || state.models[0];
    state.secondaryModelId = alt?.model_id || preferred;
    await loadModelStructure(preferred, null);
  }
}

async function loadModelStructure(modelId, checkpointId = null) {
  state.modelId = modelId;
  state.checkpointId = checkpointId;
  state.trajectory = null;
  state.activeCrop = null;
  state.causalPatches = [];
  renderActivePatches();

  const qs = new URLSearchParams({ model_id: modelId });
  if (checkpointId) qs.set('checkpoint_id', checkpointId);
  const struct = await apiGet(`/models/structure?${qs.toString()}`);
  state.structure = struct;
  state.checkpointId = struct.checkpoint_id;
  state.editHistory = struct.edit_history;

  const m = struct.model;
  $('model-summary-pill').textContent = `${struct.summary.num_layers}L · ${struct.summary.num_tensors}T · ${formatParams(struct.summary.total_params)}`;
  $('model-meta-json').textContent = JSON.stringify(
    {
      model_id: m.model_id,
      base_model: m.base_model,
      architecture: m.architecture,
      kind: m.kind,
      lora_r: m.lora_r,
      lora_alpha: m.lora_alpha,
      lora_targets: m.lora_targets,
      checkpoints: (m.checkpoints || []).length,
      ...(m.metadata?.config || {}),
    },
    null,
    2
  );

  // Populate Causal Patching & Steering layer dropdowns
  const patchSel = $('patch-layer-select');
  const steerLayerSel = $('steer-layer-select');
  patchSel.innerHTML = '';
  steerLayerSel.innerHTML = '';
  for (const l of struct.layers || []) {
    const opt = document.createElement('option');
    opt.value = String(l.index);
    opt.textContent = l.name;
    patchSel.appendChild(opt);
    steerLayerSel.appendChild(opt.cloneNode(true));
  }
  if (struct.layers?.length) {
    const midIdx = Math.floor(struct.layers.length * 0.6);
    steerLayerSel.value = String(struct.layers[midIdx]?.index ?? 0);
  }

  initLoraTargetChipsForModel(struct);

  const cps = m.checkpoints || [];
  const cpBar = $('checkpoint-bar');
  if (cps.length > 1) {
    cpBar.classList.remove('hidden');
    const slider = $('checkpoint-slider');
    slider.min = '0';
    slider.max = String(cps.length - 1);
    const activeIdx = Math.max(0, cps.findIndex((c) => c.id === state.checkpointId));
    slider.value = String(activeIdx);
    $('checkpoint-step-label').textContent = cps[activeIdx]?.label || state.checkpointId;
    renderer.setActiveCheckpointIndex(activeIdx);
  } else {
    cpBar.classList.add('hidden');
  }

  renderLayerTree();
  renderEditHistory();
  populateSplitComparators();

  const allNames = [];
  for (const l of struct.layers) for (const t of l.tensors) allNames.push(t.name);
  for (const t of struct.global_tensors) allNames.push(t.name);

  if (!state.selectedTensorName || !allNames.includes(state.selectedTensorName)) {
    const preferred =
      struct.layers[0]?.tensors.find((t) => t.shape.length === 2 && t.shape[0] * t.shape[1] >= 4096 && (t.name.includes('in_proj_qkv') || t.name.includes('gate_proj') || t.name.includes('q_proj'))) ||
      struct.layers[0]?.tensors.find((t) => t.shape.length === 2 && t.shape[0] * t.shape[1] >= 1024) ||
      struct.layers[0]?.tensors[0] ||
      struct.global_tensors[0];
    state.selectedTensorName = preferred?.name || null;
  }

  renderer.setData({ structure: struct, selectedTensorName: state.selectedTensorName });
  renderer.setCameraPreset('reset');

  if (state.selectedTensorName) {
    await selectTensor(state.selectedTensorName, { focusCamera: false });
  }
  if (renderer.viewMode === 'trajectory') {
    await loadTrajectory();
  }
  await updateFinetuneEstimate();
  renderOpParams();
}

function getSecondaryQueryParams() {
  if (state.splitCompare && state.splitTargetSpec && state.splitTargetSpec !== 'base') {
    if (state.splitTargetSpec.startsWith('cp:')) {
      return {
        secondary_model_id: state.modelId,
        secondary_checkpoint_id: state.splitTargetSpec.slice(3),
      };
    }
    if (state.splitTargetSpec.startsWith('model:')) {
      return {
        secondary_model_id: state.splitTargetSpec.slice(6),
        secondary_checkpoint_id: null,
      };
    }
  }
  if (state.secondaryModelId && renderer.colorMode === 'conflict') {
    return { secondary_model_id: state.secondaryModelId, secondary_checkpoint_id: null };
  }
  return { secondary_model_id: null, secondary_checkpoint_id: null };
}

async function fetchBinaryGrid(tensorName, res) {
  const qs = new URLSearchParams({
    model_id: state.modelId,
    tensor_name: tensorName,
    resolution: String(res),
  });
  if (state.checkpointId) qs.set('checkpoint_id', state.checkpointId);
  const sec = getSecondaryQueryParams();
  if (sec.secondary_model_id) qs.set('secondary_model_id', sec.secondary_model_id);
  if (sec.secondary_checkpoint_id) qs.set('secondary_checkpoint_id', sec.secondary_checkpoint_id);
  if (state.activeCrop) {
    qs.set('r0', String(state.activeCrop[0]));
    qs.set('r1', String(state.activeCrop[1]));
    qs.set('c0', String(state.activeCrop[2]));
    qs.set('c1', String(state.activeCrop[3]));
  }

  const resp = await fetch(`${API}/models/tensor.bin?${qs.toString()}`);
  if (!resp.ok) return null;
  const meta = JSON.parse(resp.headers.get('X-Tensor-Meta') || '{}');
  const buf = await resp.arrayBuffer();
  const floats = new Float32Array(buf);
  const stride = (meta.rows || 1) * (meta.cols || 1);
  const unpacked = {
    rows: meta.rows,
    cols: meta.cols,
    crop: meta.crop,
    matrix_shape: meta.matrix_shape,
    abs_max: meta.abs_max,
    delta_abs_max: meta.delta_abs_max,
    secondary_abs_max: meta.secondary_abs_max || 0,
    conflict_rate: meta.conflict_rate || 0,
    values: null,
    delta: null,
    importance: null,
    conflict: null,
    secondary: null,
  };
  (meta.channels || []).forEach((ch, idx) => {
    unpacked[ch] = floats.subarray(idx * stride, (idx + 1) * stride);
  });
  return unpacked;
}

async function selectTensor(tensorName, { focusCamera = true } = {}) {
  if (!tensorName || !state.modelId) return;
  state.selectedTensorName = tensorName;

  document.querySelectorAll('.tensor-row').forEach((el) => {
    el.classList.toggle('selected', el.dataset.tensor === tensorName);
  });

  const res = Number($('microscope-res').value || 64);
  const sec = getSecondaryQueryParams();
  const useBinary = res >= 96 || renderer.colorMode === 'conflict' || Boolean(sec.secondary_model_id);

  const qs = new URLSearchParams({
    model_id: state.modelId,
    tensor_name: tensorName,
    resolution: String(Math.min(res, 64)),
    include_svd: 'true',
  });
  if (state.checkpointId) qs.set('checkpoint_id', state.checkpointId);
  if (sec.secondary_model_id) qs.set('secondary_model_id', sec.secondary_model_id);
  if (sec.secondary_checkpoint_id) qs.set('secondary_checkpoint_id', sec.secondary_checkpoint_id);
  if (state.activeCrop) {
    qs.set('r0', String(state.activeCrop[0]));
    qs.set('r1', String(state.activeCrop[1]));
    qs.set('c0', String(state.activeCrop[2]));
    qs.set('c1', String(state.activeCrop[3]));
  }

  const [detail, binGrid] = await Promise.all([
    apiGet(`/models/tensor?${qs.toString()}`),
    useBinary ? fetchBinaryGrid(tensorName, res) : Promise.resolve(null),
  ]);
  state.tensorDetail = detail;
  state.binaryGrid = binGrid;

  $('hud-selected-label').textContent =
    detail.layer !== null ? `L${String(detail.layer).padStart(2, '0')} · ${detail.label}` : detail.label;
  $('hud-delta-pill').classList.toggle('hidden', !detail.edited);
  $('selected-tensor-shape').textContent = `[${detail.shape.join(' × ')}]`;
  $('edit-target-preview').textContent = `${tensorName} [${detail.shape.join('×')}]`;
  $('btn-reset-crop').classList.toggle('hidden', !state.activeCrop);

  const s = detail.stats;
  $('st-frob').textContent = s.frobenius_norm.toFixed(3);
  $('st-rms').textContent = s.rms.toFixed(4);
  $('st-std').textContent = s.std.toFixed(4);
  $('st-sparsity').textContent = `${(s.sparsity * 100).toFixed(1)}%`;
  $('st-minmax').textContent = `${s.min.toFixed(2)} / ${s.max.toFixed(2)}`;
  $('st-kurt').textContent = s.kurtosis.toFixed(2);

  if (detail.svd) {
    const rank = detail.svd.rank || 1;
    const slider = $('spectral-rank-slider');
    slider.max = String(rank);
    const curVal = Math.min(Number(slider.value) || rank, rank);
    slider.value = String(curVal);
    $('spectral-rank-val').textContent = `k=${curVal}/${rank}`;
    $('svd-eff-rank').textContent = `eff_r = ${detail.svd.effective_rank}`;
    $('svd-norm').textContent = detail.svd.spectral_norm.toFixed(4);
    $('svd-cond').textContent = detail.svd.condition_number.toFixed(1);
    $('svd-mode').textContent = detail.svd.mode === 'lora_exact_qr' ? 'LoRA QR' : 'Dense SVD';
  }

  const outEl = $('outlier-list');
  outEl.innerHTML = '';
  for (const o of detail.grid.outliers || []) {
    const row = document.createElement('div');
    row.className = 'outlier-row';
    row.innerHTML = `<span>[${o.row}, ${o.col}]</span><span style="color:${o.value >= 0 ? '#fbbf24' : '#38bdf8'}">${o.value >= 0 ? '+' : ''}${o.value.toFixed(5)}</span>`;
    outEl.appendChild(row);
  }

  renderer.setData({ tensorDetail: detail, binaryGrid: binGrid, selectedTensorName: tensorName });
  if (focusCamera) renderer.focusOnTensor(tensorName);
  if (state.activeOp === 'head_edit' || state.activeOp === 'svd_truncate') renderOpParams();
  redrawVisibleCharts();
}

async function loadTrajectory() {
  if (!state.modelId) return;
  const traj = await apiGet(`/models/trajectory?model_id=${encodeURIComponent(state.modelId)}`);
  state.trajectory = traj;
  renderer.setData({ trajectory: traj });
}

function redrawVisibleCharts() {
  if (state.tensorDetail?.stats?.histogram) {
    drawHistogramChart(
      $('chart-histogram'),
      state.tensorDetail.stats.histogram,
      state.tensorDetail.edited ? state.tensorDetail.base_stats?.histogram : null
    );
  }
  if (state.structure?.layers) {
    drawLayerProfileChart($('chart-layers'), state.structure.layers, state.tensorDetail?.layer ?? null);
  }
  if (state.tensorDetail?.svd) {
    const cutoff = Number($('spectral-rank-slider').value || state.tensorDetail.svd.rank);
    drawSvdChart($('chart-svd'), state.tensorDetail.svd, cutoff);
  }
  drawTrainingTelemetryChart($('chart-training'), state.ftHistory);
}

// ---------------------------------------------------------------------------
// Architecture Tree Rendering
// ---------------------------------------------------------------------------

function renderLayerTree() {
  const container = $('layer-tree');
  const query = ($('tensor-search').value || '').trim().toLowerCase();
  container.innerHTML = '';
  if (!state.structure) return;

  const maxNorm = Math.max(1e-5, state.structure.summary?.max_layer_norm || 1);

  const appendGroup = (title, badge, tensors, normVal, expandedDefault) => {
    const filtered = query
      ? tensors.filter((t) => t.name.toLowerCase().includes(query) || t.label.toLowerCase().includes(query) || t.role.includes(query))
      : tensors;
    if (query && filtered.length === 0) return;

    const grp = document.createElement('div');
    grp.className = 'layer-group';

    const hdr = document.createElement('div');
    hdr.className = 'layer-header';
    const pct = Math.min(100, (normVal / maxNorm) * 100);
    hdr.innerHTML = `
      <div class="layer-name-wrap">
        <span>${title}</span>
        <span class="mono-muted">${badge}</span>
      </div>
      <div class="layer-spark" title="Layer Frobenius Norm: ${normVal.toFixed(2)}">
        <div class="layer-spark-fill" style="width:${pct}%"></div>
      </div>
    `;

    const body = document.createElement('div');
    body.className = `layer-tensors${expandedDefault || query ? '' : ' collapsed'}`;
    hdr.addEventListener('click', () => body.classList.toggle('collapsed'));

    for (const t of filtered) {
      const row = document.createElement('div');
      row.className = `tensor-row${t.name === state.selectedTensorName ? ' selected' : ''}`;
      row.dataset.tensor = t.name;
      row.innerHTML = `
        <div class="tensor-row-left">
          <span class="role-pip role-${t.role}"></span>
          <span class="tensor-label" title="${t.name}">${t.label}</span>
        </div>
        <div style="display:flex;align-items:center;gap:5px;">
          ${t.edited ? '<span class="edited-dot" title="Surgically Modified"></span>' : ''}
          <span class="mono-muted">${t.shape.join('×')}</span>
        </div>
      `;
      row.addEventListener('click', () => {
        state.activeCrop = null;
        selectTensor(t.name, { focusCamera: true });
      });
      row.addEventListener('dblclick', async () => {
        state.activeCrop = null;
        await selectTensor(t.name, { focusCamera: false });
        setViewMode('microscope');
      });
      body.appendChild(row);
    }

    grp.appendChild(hdr);
    grp.appendChild(body);
    container.appendChild(grp);
  };

  if (state.structure.global_tensors?.length) {
    appendGroup('Global / Embed', `${state.structure.global_tensors.length}T`, state.structure.global_tensors, maxNorm * 0.6, true);
  }
  state.structure.layers.forEach((layer, idx) => {
    const tag = layer.layer_type === 'hybrid_linear_attn' ? 'LIN' : 'ATTN';
    appendGroup(layer.name, `${tag}·${layer.tensor_count}T`, layer.tensors, layer.frobenius_norm, idx < 2);
  });
}

// ---------------------------------------------------------------------------
// Topbar, Camera, Optics & Mode Switching
// ---------------------------------------------------------------------------

function setViewMode(mode) {
  document.querySelectorAll('.mode-pill').forEach((btn) => {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
  });
  $('microscope-hud').classList.toggle('hidden', mode !== 'microscope');
  $('spectral-hud').classList.toggle('hidden', mode !== 'spectral');
  $('circuit-hud').classList.toggle('hidden', mode !== 'circuit');
  $('attention-hud')?.classList.toggle('hidden', mode !== 'attention');
  $('residual-hud')?.classList.toggle('hidden', mode !== 'residual');

  if (mode !== 'microscope' && state.splitCompare) {
    setSplitCompare(false);
  }

  if (mode === 'trajectory' && !state.trajectory) {
    loadTrajectory().then(() => renderer.setViewMode('trajectory'));
  } else if (mode === 'circuit') {
    if (!state.circuitData) {
      discoverCircuit({ switchMode: true });
    } else {
      renderer.setViewMode('circuit');
    }
  } else if (mode === 'attention') {
    if (!state.attentionData) {
      loadAttentionMap({ switchMode: true });
    } else {
      renderer.setViewMode('attention');
    }
  } else if (mode === 'residual') {
    if (!state.residualData) {
      loadResidualStream({ switchMode: true });
    } else {
      renderer.setViewMode('residual');
    }
  } else {
    renderer.setViewMode(mode);
  }
  if (mode === 'spectral') redrawVisibleCharts();
}

function updateLegendGradient(colorMode) {
  const bar = $('legend-gradient');
  const lo = $('legend-lo');
  const hi = $('legend-hi');
  if (colorMode === 'importance') {
    bar.style.background = 'linear-gradient(90deg, #0d1220 0%, #7c2d12 45%, #f59e0b 78%, #fef08a 100%)';
    lo.textContent = 'Prunable'; hi.textContent = 'Critical';
  } else if (colorMode === 'conflict') {
    bar.style.background = 'linear-gradient(90deg, #ec4899 0%, #141c2c 50%, #10b981 100%)';
    lo.textContent = 'Conflict'; hi.textContent = 'Aligned';
  } else if (colorMode === 'magnitude') {
    bar.style.background = 'linear-gradient(90deg, #131924 0%, #0ea5e9 38%, #f59e0b 75%, #fffbeb 100%)';
    lo.textContent = '0.0'; hi.textContent = '|Max|';
  } else if (colorMode === 'delta') {
    bar.style.background = 'linear-gradient(90deg, #f43f5e 0%, #1b2434 50%, #10b981 100%)';
    lo.textContent = '−ΔW'; hi.textContent = '+ΔW';
  } else if (colorMode === 'gradient') {
    bar.style.background = 'linear-gradient(90deg, #1e1456 0%, #a855f7 45%, #ef4444 75%, #fde047 100%)';
    lo.textContent = 'Low'; hi.textContent = 'Hot';
  } else if (colorMode === 'outlier') {
    bar.style.background = 'linear-gradient(90deg, #1e293b 0%, #334155 55%, #ec4899 80%, #ffffff 100%)';
    lo.textContent = '<2σ'; hi.textContent = '>5σ';
  } else {
    bar.style.background = 'linear-gradient(90deg, #1d4ed8 0%, #38bdf8 32%, #182030 50%, #d97706 68%, #fde68a 100%)';
    lo.textContent = '−Max'; hi.textContent = '+Max';
  }
}

function bindTopbarControls() {
  $('model-select').addEventListener('change', (e) => loadModelStructure(e.target.value, null));
  $('btn-refresh-models').addEventListener('click', () => loadModels(true));
  $('tensor-search').addEventListener('input', () => renderLayerTree());

  document.querySelectorAll('.mode-pill').forEach((btn) => {
    btn.addEventListener('click', () => setViewMode(btn.dataset.mode));
  });

  $('colormap-select').addEventListener('change', async (e) => {
    renderer.colorMode = e.target.value;
    updateLegendGradient(e.target.value);
    if (e.target.value === 'conflict' && state.selectedTensorName) {
      await selectTensor(state.selectedTensorName, { focusCamera: false });
    } else {
      renderer.rebuildScene(true);
    }
  });

  $('rig-select').addEventListener('change', (e) => renderer.setRig(e.target.value));
  $('bloom-slider').addEventListener('input', (e) => { renderer.bloomOverride = Number(e.target.value); });
  $('spread-slider').addEventListener('input', (e) => {
    renderer.layerSpread = Number(e.target.value);
    renderer.rebuildScene(false);
  });

  document.querySelectorAll('.cam-btn[data-cam]').forEach((btn) => {
    btn.addEventListener('click', () => renderer.setCameraPreset(btn.dataset.cam));
  });

  $('btn-orbit-toggle').addEventListener('click', (e) => {
    renderer.autoOrbit = !renderer.autoOrbit;
    e.currentTarget.classList.toggle('active', renderer.autoOrbit);
  });
  $('btn-conduits-toggle').addEventListener('click', (e) => {
    renderer.showConduits = !renderer.showConduits;
    e.currentTarget.classList.toggle('active', renderer.showConduits);
  });
  $('btn-reflect-toggle').addEventListener('click', (e) => {
    renderer.enableReflections = !renderer.enableReflections;
    e.currentTarget.classList.toggle('active', renderer.enableReflections);
  });
  $('btn-shadow-toggle').addEventListener('click', (e) => {
    renderer.enableShadows = !renderer.enableShadows;
    e.currentTarget.classList.toggle('active', renderer.enableShadows);
  });
  $('btn-ssao-toggle').addEventListener('click', (e) => {
    renderer.ssaoStrength = renderer.ssaoStrength > 0 ? 0.0 : 0.65;
    e.currentTarget.classList.toggle('active', renderer.ssaoStrength > 0);
  });
  $('btn-dof-toggle').addEventListener('click', (e) => {
    renderer.dofStrength = renderer.dofStrength > 0 ? 0.0 : 0.72;
    e.currentTarget.classList.toggle('active', renderer.dofStrength > 0);
  });
  $('btn-rulers-toggle').addEventListener('click', (e) => {
    renderer.showSpatialRulers = !renderer.showSpatialRulers;
    e.currentTarget.classList.toggle('active', renderer.showSpatialRulers);
  });

  $('btn-toggle-brush').addEventListener('click', (e) => {
    renderer.brushMode = !renderer.brushMode;
    e.currentTarget.classList.toggle('active', renderer.brushMode);
    if (renderer.brushMode) {
      state.activeOp = 'region_edit';
      document.querySelectorAll('.op-chip').forEach((c) => c.classList.toggle('active', c.dataset.op === 'region_edit'));
      renderOpParams();
      showToast('3D Surgical Brush active — drag across crystal terrain to paint a patch');
    }
  });

  $('btn-zoom-crop').addEventListener('click', async () => {
    const br = state.brushRegion;
    if (!br || !state.selectedTensorName) return;
    state.activeCrop = [br.row_start, br.row_end, br.col_start, br.col_end];
    renderer.brushRect = null;
    await selectTensor(state.selectedTensorName, { focusCamera: false });
    showToast(`Deep-zoomed to rows [${br.row_start}:${br.row_end}] × cols [${br.col_start}:${br.col_end}]`);
  });

  $('btn-reset-crop').addEventListener('click', async () => {
    state.activeCrop = null;
    if (state.selectedTensorName) await selectTensor(state.selectedTensorName, { focusCamera: false });
  });

  $('chk-ghost-overlay').addEventListener('change', (e) => {
    renderer.showGhostOverlay = e.target.checked;
    renderer.rebuildScene(false);
  });
  $('chk-head-dividers').addEventListener('change', (e) => {
    renderer.showHeadDividers = e.target.checked;
    renderer.rebuildScene(false);
  });

  $('microscope-res').addEventListener('change', () => {
    if (state.selectedTensorName) selectTensor(state.selectedTensorName, { focusCamera: false });
  });

  $('spectral-rank-slider').addEventListener('input', (e) => {
    const k = Number(e.target.value);
    const maxR = state.tensorDetail?.svd?.rank || k;
    $('spectral-rank-val').textContent = `k=${k}/${maxR}`;
    renderer.setSvdCutoff(k);
    if (state.tensorDetail?.svd) drawSvdChart($('chart-svd'), state.tensorDetail.svd, k);
  });

  $('btn-apply-svd-rank').addEventListener('click', async () => {
    const k = Number($('spectral-rank-slider').value || 8);
    if (!state.selectedTensorName) return;
    await executeSurgicalEdit('svd_truncate', state.selectedTensorName, { rank: k }, `SVD Rank-k=${k}`);
  });

  $('checkpoint-slider').addEventListener('input', async (e) => {
    const idx = Number(e.target.value);
    const cps = state.structure?.model?.checkpoints || [];
    const cp = cps[idx];
    if (!cp) return;
    $('checkpoint-step-label').textContent = cp.label;
    renderer.setActiveCheckpointIndex(idx);
    await loadModelStructure(state.modelId, cp.id);
  });

  $('btn-play-traj').addEventListener('click', (e) => {
    const btn = e.currentTarget;
    if (state.trajPlayTimer) {
      clearInterval(state.trajPlayTimer);
      state.trajPlayTimer = null;
      btn.textContent = '▶';
      return;
    }
    const cps = state.structure?.model?.checkpoints || [];
    if (cps.length < 2) return;
    btn.textContent = '⏸';
    state.trajPlayTimer = setInterval(async () => {
      const slider = $('checkpoint-slider');
      const next = (Number(slider.value) + 1) % cps.length;
      slider.value = String(next);
      $('checkpoint-step-label').textContent = cps[next].label;
      renderer.setActiveCheckpointIndex(next);
      if (renderer.viewMode !== 'trajectory') {
        await loadModelStructure(state.modelId, cps[next].id);
      }
    }, 1100);
  });

  $('btn-screenshot').addEventListener('click', () => {
    const url = renderer.captureScreenshotDataUrl();
    const a = document.createElement('a');
    a.href = url;
    a.download = `stateset-weight-studio-${renderer.viewMode}-${Date.now()}.png`;
    a.click();
    showToast('Captured 3D Viewport PNG');
  });

  $('btn-record-360').addEventListener('click', async () => {
    const btn = $('btn-record-360');
    if (btn.classList.contains('recording')) return;
    btn.classList.add('recording');
    showToast('Recording 60 FPS 360° Turntable WebM (5.5s)…', 5600);
    try {
      const blob = await renderer.recordTurntableWebM(5500);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `stateset-weight-studio-${renderer.viewMode}-360-${Date.now()}.webm`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      showToast('Saved 360° Turntable WebM video!');
    } catch (err) {
      showToast(`Video record unavailable: ${err.message}`, 3200);
    } finally {
      btn.classList.remove('recording');
    }
  });

  $('btn-zen-mode').addEventListener('click', toggleZenMode);
  $('btn-toggle-split').addEventListener('click', () => setSplitCompare(!state.splitCompare));
  $('btn-rediscover-circuit').addEventListener('click', () => discoverCircuit({ switchMode: true }));
  $('circuit-topk-slider').addEventListener('input', (e) => {
    $('circuit-summary-val').textContent = `Top-${e.target.value} Nodes`;
  });
  $('circuit-topk-slider').addEventListener('change', () => discoverCircuit({ switchMode: true }));

  $('btn-export-recipe').addEventListener('click', async () => {
    if (!state.modelId) return;
    try {
      const qs = new URLSearchParams({ model_id: state.modelId });
      if (state.checkpointId) qs.set('checkpoint_id', state.checkpointId);
      const recipe = await apiGet(`/edits/recipe?${qs.toString()}`);
      const blob = new Blob([JSON.stringify(recipe, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${state.modelId.replace(/[^a-z0-9_-]/gi, '_')}.studio-recipe.json`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
      showToast(`Exported edit recipe (${recipe.operations.length} ops)`);
    } catch (err) {
      showToast(`Recipe export failed: ${err.message}`, 3000);
    }
  });

  $('btn-import-recipe').addEventListener('click', () => $('recipe-file-input').click());
  $('recipe-file-input').addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file || !state.modelId) return;
    try {
      const text = await file.text();
      const recipe = JSON.parse(text);
      const res = await apiPost('/edits/recipe/apply', {
        model_id: state.modelId,
        checkpoint_id: state.checkpointId,
        recipe,
        reset_first: true,
      });
      await reloadAfterEdit(`Replayed recipe: ${res.applied_count} applied, ${res.skipped_count} skipped`);
    } catch (err) {
      showToast(`Recipe replay failed: ${err.message}`, 3400);
    } finally {
      e.target.value = '';
    }
  });

  $('chart-layers').addEventListener('click', (e) => {
    const layers = state.structure?.layers || [];
    if (!layers.length) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / Math.max(rect.width, 1);
    const idx = Math.max(0, Math.min(layers.length - 1, Math.floor(frac * layers.length)));
    const firstT = layers[idx]?.tensors?.[0];
    if (firstT) selectTensor(firstT.name, { focusCamera: true });
  });

  $('btn-clamp-outliers').addEventListener('click', async () => {
    if (!state.selectedTensorName) return;
    await executeSurgicalEdit('clamp', state.selectedTensorName, { mode: 'percentile', percentile: 99.0 }, 'Clamp P99 Outliers');
  });
}

function renderHoverTooltip(hit) {
  const tip = $('hover-tooltip');
  if (!hit) {
    tip.classList.add('hidden');
    return;
  }
  if (hit.kind === 'cell') {
    const sign = hit.value >= 0 ? '+' : '';
    const dStr = Math.abs(hit.delta) > 1e-7 ? ` · Δ=${hit.delta >= 0 ? '+' : ''}${hit.delta.toFixed(5)}` : '';
    tip.innerHTML = `<b>[${hit.origRow}, ${hit.origCol}]</b> · w = <span style="color:${hit.value >= 0 ? '#fbbf24' : '#38bdf8'}">${sign}${hit.value.toFixed(5)}</span>${dStr}`;
  } else if (hit.kind === 'attention_head') {
    const pk = hit.peak_pair ? ` · Peak "${hit.peak_pair.query_token}"→"${hit.peak_pair.key_token}" (${hit.peak_pair.weight})` : '';
    tip.innerHTML = `<b>${hit.label} · ${hit.role}</b> · H=${hit.entropy} · Sink=${Math.round((hit.sink_ratio || 0) * 100)}%${pk} <span style="color:#7dd3fc">(Click to zoom)</span>`;
  } else if (hit.kind === 'residual_station') {
    tip.innerHTML = `<b>${hit.name}</b> · Top Token: <span style="color:#a3e635">"${hit.edited_top_token}"</span> (${Math.round((hit.edited_top_prob || 0) * 100)}%) · Drift=${(hit.cosine_drift || 0).toFixed(4)} · Attn ${Math.round((hit.attn_ratio || 0.5) * 100)}% / MLP ${Math.round((hit.mlp_ratio || 0.5) * 100)}%`;
  } else {
    const lTag = hit.layer !== null && hit.layer !== undefined ? `L${String(hit.layer).padStart(2, '0')} · ` : '';
    tip.innerHTML = `<b>${lTag}${hit.label}</b> · [${(hit.shape || []).join('×')}] · ‖W‖_F=${Number(hit.frob || 0).toFixed(2)}`;
  }
  tip.style.left = `${Math.min(hit.screenX + 14, window.innerWidth - 260)}px`;
  tip.style.top = `${Math.max(12, hit.screenY - 34)}px`;
  tip.classList.remove('hidden');
}

function render3DCallout(pt, hit) {
  const el = $('callout-3d');
  if (!pt || !hit) {
    el.classList.add('hidden');
    return;
  }
  el.style.left = `${pt.x}px`;
  el.style.top = `${pt.y}px`;
  $('callout-role').textContent = `${(hit.role || 'tensor').toUpperCase()}${hit.layer !== null ? ` · LAYER ${hit.layer}` : ''}`;
  $('callout-name').textContent = hit.label;
  $('callout-meta').textContent = `[${(hit.shape || []).join(' × ')}] · ‖W‖_F ${Number(hit.frob || 0).toFixed(2)}`;
  el.classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// Workbench Tabs & Advanced Surgical Weight Editing
// ---------------------------------------------------------------------------

function bindWorkbenchTabs() {
  document.querySelectorAll('.wb-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      const key = tab.dataset.tab;
      document.querySelectorAll('.wb-tab').forEach((t) => t.classList.toggle('active', t === tab));
      document.querySelectorAll('.wb-pane').forEach((p) => p.classList.toggle('active', p.id === `tab-${key}`));
      if (key === 'finetune') renderer.setLoraTargets([...state.ftSelectedTargets]);
      else renderer.setLoraTargets([]);
      setTimeout(redrawVisibleCharts, 30);
    });
  });
}

function bindEditControls() {
  document.querySelectorAll('.op-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      state.activeOp = chip.dataset.op;
      document.querySelectorAll('.op-chip').forEach((c) => c.classList.toggle('active', c === chip));
      renderOpParams();
    });
  });

  $('btn-apply-edit').addEventListener('click', handleApplyEditClick);
  $('btn-undo').addEventListener('click', async () => {
    if (!state.modelId) return;
    await apiPost('/edits/undo', { model_id: state.modelId, checkpoint_id: state.checkpointId });
    await reloadAfterEdit('Undid last surgical edit');
  });
  $('btn-redo').addEventListener('click', async () => {
    if (!state.modelId) return;
    await apiPost('/edits/redo', { model_id: state.modelId, checkpoint_id: state.checkpointId });
    await reloadAfterEdit('Redid surgical edit');
  });
  $('btn-reset-edits').addEventListener('click', async () => {
    if (!state.modelId) return;
    await apiPost('/edits/reset', { model_id: state.modelId, checkpoint_id: state.checkpointId });
    await reloadAfterEdit('Reset all weight edits to base checkpoint');
  });

  renderOpParams();
}

function renderOpParams() {
  const c = $('op-params-container');
  const op = state.activeOp;

  if (op === 'scale') {
    c.innerHTML = `
      <label class="field">
        <span>Scale Factor α <b id="p-scale-val" class="mono-accent">1.25×</b></span>
        <input id="p-scale" type="range" min="0" max="2.5" step="0.05" value="1.25" />
      </label>
    `;
    $('p-scale').addEventListener('input', (e) => { $('p-scale-val').textContent = `${Number(e.target.value).toFixed(2)}×`; });
  } else if (op === 'prune') {
    c.innerHTML = `
      <label class="field">
        <span>Pruning Criterion</span>
        <select id="p-prune-mode" class="precision-select compact">
          <option value="wanda" selected>Wanda + Fisher Importance (Smart)</option>
          <option value="percentile">Magnitude Percentile |W|</option>
        </select>
      </label>
      <label class="field">
        <span>Target Sparsity <b id="p-prune-val" class="mono-accent">35%</b></span>
        <input id="p-prune" type="range" min="0.05" max="0.95" step="0.05" value="0.35" />
      </label>
    `;
    $('p-prune').addEventListener('input', (e) => { $('p-prune-val').textContent = `${Math.round(Number(e.target.value) * 100)}%`; });
  } else if (op === 'clamp') {
    c.innerHTML = `
      <label class="field">
        <span>Percentile Clamp Ceiling <b id="p-clamp-val" class="mono-accent">P98.0</b></span>
        <input id="p-clamp" type="range" min="80" max="99.9" step="0.5" value="98" />
      </label>
    `;
    $('p-clamp').addEventListener('input', (e) => { $('p-clamp-val').textContent = `P${Number(e.target.value).toFixed(1)}`; });
  } else if (op === 'svd_truncate') {
    const maxR = state.tensorDetail?.svd?.rank || 16;
    const defR = Math.max(1, Math.floor(maxR * 0.5));
    c.innerHTML = `
      <label class="field">
        <span>Keep Top-k Singular Modes <b id="p-svd-val" class="mono-accent">k = ${defR} / ${maxR}</b></span>
        <input id="p-svd" type="range" min="1" max="${maxR}" step="1" value="${defR}" />
      </label>
    `;
    $('p-svd').addEventListener('input', (e) => { $('p-svd-val').textContent = `k = ${e.target.value} / ${maxR}`; });
  } else if (op === 'head_edit') {
    const nh = state.tensorDetail?.head_info?.num_heads || 8;
    const headOpts = Array.from({ length: nh }, (_, i) => `<option value="${i}">Head H${i}</option>`).join('');
    c.innerHTML = `
      <div class="form-grid-2">
        <label class="field">
          <span>Attention Head (${nh} total)</span>
          <select id="p-head-idx" class="precision-select compact">${headOpts}</select>
        </label>
        <label class="field">
          <span>Head Action</span>
          <select id="p-head-action" class="precision-select compact">
            <option value="ablate">Ablate (Zero)</option>
            <option value="scale">Scale Head</option>
            <option value="prune">Prune Head</option>
            <option value="invert">Invert Sign</option>
          </select>
        </label>
      </div>
      <label class="field">
        <span>Scale / Sparsity Value</span>
        <input id="p-head-val" type="number" step="0.1" value="0.5" class="precision-input mono" />
      </label>
    `;
  } else if (op === 'rank1_edit') {
    c.innerHTML = `
      <label class="field">
        <span>Key Subject (k)</span>
        <input id="p-r1-subj" class="precision-input" value="Order #77701 refund window" />
      </label>
      <label class="field">
        <span>Target Association (v*)</span>
        <input id="p-r1-val" class="precision-input" value="Auto-approve within 30 days" />
      </label>
      <label class="field">
        <span>Update Strength <b id="p-r1-str-val" class="mono-accent">0.65</b></span>
        <input id="p-r1-str" type="range" min="0.1" max="1.5" step="0.05" value="0.65" />
      </label>
    `;
    $('p-r1-str').addEventListener('input', (e) => { $('p-r1-str-val').textContent = Number(e.target.value).toFixed(2); });
  } else if (op === 'noise') {
    c.innerHTML = `
      <div class="form-grid-2">
        <label class="field"><span>Gaussian σ</span><input id="p-sigma" type="number" step="0.002" min="0.0005" max="0.2" value="0.01" class="precision-input mono" /></label>
        <label class="field"><span>Seed</span><input id="p-seed" type="number" value="42" class="precision-input mono" /></label>
      </div>
    `;
  } else if (op === 'region_edit') {
    const br = state.brushRegion;
    c.innerHTML = `
      <div class="form-grid-2">
        <label class="field"><span>Row Start</span><input id="p-r0" type="number" value="${br.row_start}" class="precision-input mono" /></label>
        <label class="field"><span>Row End</span><input id="p-r1" type="number" value="${br.row_end}" class="precision-input mono" /></label>
        <label class="field"><span>Col Start</span><input id="p-c0" type="number" value="${br.col_start}" class="precision-input mono" /></label>
        <label class="field"><span>Col End</span><input id="p-c1" type="number" value="${br.col_end}" class="precision-input mono" /></label>
      </div>
      <div class="form-grid-2">
        <label class="field">
          <span>Brush Action</span>
          <select id="p-reg-action" class="precision-select compact">
            <option value="zero">Zero Out Patch</option>
            <option value="scale" selected>Scale Patch</option>
            <option value="invert">Invert Sign (−W)</option>
            <option value="clamp">Clamp Patch</option>
          </select>
        </label>
        <label class="field"><span>Value</span><input id="p-reg-val" type="number" step="0.1" value="0.25" class="precision-input mono" /></label>
      </div>
    `;
  } else if (op === 'quantize') {
    c.innerHTML = `
      <label class="field">
        <span>Quantization Precision Scheme</span>
        <select id="p-quant-scheme" class="precision-select compact">
          <option value="nf4" selected>NF4 (QLoRA NormalFloat4 · 4×)</option>
          <option value="int4_awq">INT4 AWQ (Activation-Aware · 4×)</option>
          <option value="fp8">FP8 (E4M3 Micro-Scale · 2×)</option>
          <option value="int8">INT8 (Per-Channel Sym · 2×)</option>
        </select>
      </label>
      <label class="inline-toggle mt-6">
        <input id="p-quant-smooth" type="checkbox" checked />
        <span>SmoothQuant Outlier Channel Protection</span>
      </label>
    `;
  } else if (op === 'blend') {
    const options = state.models
      .map((m) => `<option value="${m.model_id}"${m.model_id === state.secondaryModelId ? ' selected' : ''}>${m.display_name}</option>`)
      .join('');
    c.innerHTML = `
      <label class="field">
        <span>Secondary Adapter / Checkpoint</span>
        <select id="p-blend-model" class="precision-select">${options}</select>
      </label>
      <div class="form-grid-2">
        <label class="field">
          <span>Merge Method</span>
          <select id="p-blend-method" class="precision-select compact">
            <option value="ties" selected>TIES-Merging</option>
            <option value="dare">DARE Drop+Rescale</option>
            <option value="slerp">SLERP Spherical</option>
            <option value="linear">Linear Lerp</option>
          </select>
        </label>
        <label class="field">
          <span>Weight λ <b id="p-blend-val" class="mono-accent">0.50</b></span>
          <input id="p-blend-alpha" type="range" min="0" max="1" step="0.05" value="0.5" />
        </label>
      </div>
      <button id="btn-preview-conflict" class="mini-btn full">✦ Visualize 3D Sign-Conflict Map (A vs B)</button>
    `;
    $('p-blend-alpha').addEventListener('input', (e) => { $('p-blend-val').textContent = Number(e.target.value).toFixed(2); });
    $('p-blend-model').addEventListener('change', (e) => { state.secondaryModelId = e.target.value; });
    $('btn-preview-conflict').addEventListener('click', async () => {
      state.secondaryModelId = $('p-blend-model').value;
      $('colormap-select').value = 'conflict';
      renderer.colorMode = 'conflict';
      updateLegendGradient('conflict');
      setViewMode('microscope');
      if (state.selectedTensorName) await selectTensor(state.selectedTensorName, { focusCamera: false });
      const cr = state.binaryGrid?.conflict_rate ?? state.tensorDetail?.conflict_rate ?? 0;
      showToast(`3D Sign Conflict: ${(cr * 100).toFixed(1)}% opposing signs between Adapter A & B`);
    });
  }
}

async function handleApplyEditClick() {
  if (!state.modelId) return;
  const scope = $('edit-scope-select').value;
  let target = state.selectedTensorName;
  if (scope === 'layer') target = `layer:${state.tensorDetail?.layer ?? 0}`;
  else if (scope !== 'selected') target = scope;
  if (!target) {
    showToast('Select a tensor first');
    return;
  }

  const op = state.activeOp;
  let params = {};
  let label = '';
  if (op === 'scale') {
    const factor = Number($('p-scale').value);
    params = { factor };
    label = `Scale ×${factor.toFixed(2)}`;
  } else if (op === 'prune') {
    const sparsity = Number($('p-prune').value);
    const mode = $('p-prune-mode').value;
    params = { mode, sparsity };
    label = `${mode === 'wanda' ? 'Wanda' : 'Mag'} Prune ${Math.round(sparsity * 100)}%`;
  } else if (op === 'clamp') {
    const percentile = Number($('p-clamp').value);
    params = { mode: 'percentile', percentile };
    label = `Clamp P${percentile.toFixed(1)}`;
  } else if (op === 'svd_truncate') {
    const rank = Number($('p-svd').value);
    params = { rank };
    label = `SVD Rank-k=${rank}`;
  } else if (op === 'head_edit') {
    const head_index = Number($('p-head-idx').value);
    const action = $('p-head-action').value;
    const value = Number($('p-head-val').value);
    params = { head_index, action, value };
    target = state.selectedTensorName;
    label = `Head H${head_index} ${action}`;
  } else if (op === 'rank1_edit') {
    const subject = $('p-r1-subj').value;
    const target_fact = $('p-r1-val').value;
    const strength = Number($('p-r1-str').value);
    params = { subject, target_fact, strength };
    target = state.selectedTensorName;
    label = `ROME Rank-1 (${subject.slice(0, 18)})`;
  } else if (op === 'noise') {
    params = { sigma: Number($('p-sigma').value), seed: Number($('p-seed').value) };
    label = `Noise σ=${params.sigma}`;
  } else if (op === 'region_edit') {
    params = {
      row_start: Number($('p-r0').value),
      row_end: Number($('p-r1').value),
      col_start: Number($('p-c0').value),
      col_end: Number($('p-c1').value),
      action: $('p-reg-action').value,
      value: Number($('p-reg-val').value),
    };
    target = state.selectedTensorName;
    label = `Brush ${params.action} [${params.row_start}:${params.row_end}, ${params.col_start}:${params.col_end}]`;
  } else if (op === 'quantize') {
    const scheme = $('p-quant-scheme').value;
    const smooth_outliers = $('p-quant-smooth').checked;
    params = { scheme, smooth_outliers };
    label = `Quantize ${scheme.toUpperCase()}${smooth_outliers ? '+Smooth' : ''}`;
  } else if (op === 'blend') {
    const method = $('p-blend-method').value;
    params = {
      secondary_model_id: $('p-blend-model').value,
      alpha: Number($('p-blend-alpha').value),
      method,
    };
    label = `${method.toUpperCase()} Merge λ=${params.alpha.toFixed(2)}`;
  }

  await executeSurgicalEdit(op, target, params, label);
}

async function executeSurgicalEdit(op, target, params, label) {
  try {
    const res = await apiPost('/edits/apply', {
      model_id: state.modelId,
      checkpoint_id: state.checkpointId,
      op,
      target,
      params,
      label,
    });
    renderer.triggerPulse('backprop');
    const d = res.edit.diagnostics || {};
    const extra = d.sqnr_db !== undefined
      ? ` · SQNR=${d.sqnr_db} dB (${d.compression_ratio}×)`
      : d.protected_mean_cosine
        ? ` · Drift cos=${d.protected_mean_cosine} (${d.drift_passed ? 'PASS' : 'WARN'})`
        : d.conflict_rate !== undefined
          ? ` · Conflict ${(d.conflict_rate * 100).toFixed(1)}%`
          : '';
    await reloadAfterEdit(`Applied ${res.edit.label} (${res.edit.changed_weights.toLocaleString()} w${extra})`);
  } catch (err) {
    showToast(`Edit failed: ${err.message}`, 3200);
  }
}

async function reloadAfterEdit(toastMsg) {
  const qs = new URLSearchParams({ model_id: state.modelId });
  if (state.checkpointId) qs.set('checkpoint_id', state.checkpointId);
  const struct = await apiGet(`/models/structure?${qs.toString()}`);
  state.structure = struct;
  state.editHistory = struct.edit_history;
  renderLayerTree();
  renderEditHistory();
  populateSplitComparators();
  renderer.setData({ structure: struct });
  if (state.selectedTensorName) {
    await selectTensor(state.selectedTensorName, { focusCamera: false });
  }
  if (toastMsg) showToast(toastMsg);
}

function renderEditHistory() {
  const h = state.editHistory;
  const edits = h?.edits || [];
  const activeCount = edits.filter((e) => e.enabled).length;
  const badge = $('edit-count-badge');
  badge.textContent = `${activeCount} EDIT${activeCount === 1 ? '' : 'S'}`;
  badge.classList.toggle('active', activeCount > 0);
  $('btn-undo').disabled = !h?.can_undo;
  $('btn-redo').disabled = !h?.can_redo;
  $('stack-delta-summary').textContent = `${h?.modified_tensor_count || 0} modified`;

  const list = $('edit-stack-list');
  if (!edits.length) {
    list.innerHTML = '<div class="empty-hint">No surgical edits yet. Try Scale, Wanda Prune, Head Surgery, ROME Rank-1, or TIES/DARE Merge above.</div>';
    return;
  }
  list.innerHTML = '';
  for (const e of [...edits].reverse()) {
    const diag = e.diagnostics || {};
    const diagTag = diag.protected_mean_cosine
      ? ` · drift=${diag.protected_mean_cosine}`
      : diag.conflict_rate !== undefined
        ? ` · conflict=${(diag.conflict_rate * 100).toFixed(0)}%`
        : '';
    const item = document.createElement('div');
    item.className = `edit-item${e.enabled ? '' : ' disabled'}`;
    item.innerHTML = `
      <div>
        <div class="edit-item-title">${e.label}</div>
        <div class="edit-item-sub">${e.affected_tensors.length}T · ${e.changed_weights.toLocaleString()} w · Δ‖W‖=${e.frobenius_delta}${diagTag}</div>
      </div>
      <button class="mini-btn" title="Enable/Disable Edit">${e.enabled ? 'ON' : 'OFF'}</button>
    `;
    item.querySelector('button').addEventListener('click', async () => {
      await apiPost('/edits/toggle', {
        model_id: state.modelId,
        checkpoint_id: state.checkpointId,
        edit_id: e.edit_id,
      });
      await reloadAfterEdit(`Toggled ${e.label}`);
    });
    list.appendChild(item);
  }
}

// ---------------------------------------------------------------------------
// Finetuning Workbench, Curated Datasets & Live PyTorch Trainer
// ---------------------------------------------------------------------------

async function loadFinetuneCatalog() {
  const cat = await apiGet('/finetune/catalog');
  state.ftCatalog = cat;

  const algoGrid = $('algo-pills');
  algoGrid.innerHTML = '';
  for (const [key, info] of Object.entries(cat.algorithms)) {
    const btn = document.createElement('button');
    btn.className = `algo-pill${key === state.ftAlgorithm ? ' active' : ''}`;
    btn.textContent = info.name;
    btn.title = info.full_name;
    btn.addEventListener('click', () => {
      state.ftAlgorithm = key;
      document.querySelectorAll('.algo-pill').forEach((b) => b.classList.toggle('active', b === btn));
      $('algo-desc').textContent = `${info.full_name}: ${info.description}`;
      $('ft-clip').value = String(info.default_clip);
      $('ft-kl').value = String(info.default_kl);
      $('ft-group').value = String(info.default_group);
      updateFinetuneEstimate();
    });
    algoGrid.appendChild(btn);
  }

  const provSel = $('ft-provider');
  provSel.innerHTML = '';
  for (const [key, p] of Object.entries(cat.providers)) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = `${p.name} (${p.vram_gb}GB${p.hourly_usd > 0 ? ` · $${p.hourly_usd}/hr` : ''})`;
    provSel.appendChild(opt);
  }
  provSel.value = 'runpod_a40';
}

async function loadDatasets() {
  try {
    const res = await apiGet('/finetune/datasets');
    const sel = $('ft-dataset-select');
    sel.innerHTML = '';
    if (!res.datasets?.length) {
      sel.innerHTML = '<option value="improved/curated.jsonl">No local JSONL found (improved/curated.jsonl)</option>';
      return;
    }
    for (const ds of res.datasets) {
      const opt = document.createElement('option');
      opt.value = ds.path;
      opt.textContent = `${ds.path} (${ds.size_kb} KB)`;
      sel.appendChild(opt);
    }
    state.ftDatasetPath = res.datasets[0].path;
    await inspectSelectedDataset(state.ftDatasetPath);
  } catch {
    // Optional discovery
  }
}

async function inspectSelectedDataset(path) {
  if (!path) return;
  try {
    const info = await apiGet(`/finetune/datasets/inspect?path=${encodeURIComponent(path)}`);
    const rwStr = info.mean_reward !== null ? ` · avgR=${info.mean_reward}` : '';
    $('dataset-stats-pill').textContent = `${info.total_rows} rows${rwStr}`;
    const list = $('dataset-preview-list');
    list.innerHTML = (info.samples || [])
      .slice(0, 4)
      .map(
        (s) =>
          `<div class="trace-row"><span class="mono-accent">#${s.index}${s.reward !== null ? ` R=${s.reward}` : ''}</span> ${s.preview.replace(/</g, '&lt;')}</div>`
      )
      .join('');
  } catch {
    $('dataset-stats-pill').textContent = '—';
  }
}

function initLoraTargetChipsForModel(struct) {
  const detected = new Set();
  const candidates = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'in_proj_qkv', 'in_proj_z', 'out_proj',
    'gate_proj', 'up_proj', 'down_proj', 'c_attn', 'c_proj', 'c_fc',
  ];
  for (const l of struct.layers || []) {
    for (const t of l.tensors || []) {
      for (const c of candidates) {
        if (t.name.includes(c)) detected.add(c);
      }
    }
  }
  const list = detected.size > 0 ? [...detected] : ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'];
  state.ftSelectedTargets = new Set(list.slice(0, 4));

  const wrap = $('lora-target-chips');
  wrap.innerHTML = '';
  for (const mod of list) {
    const chip = document.createElement('button');
    chip.className = `target-chip${state.ftSelectedTargets.has(mod) ? ' active' : ''}`;
    chip.textContent = mod;
    chip.addEventListener('click', () => {
      if (state.ftSelectedTargets.has(mod)) state.ftSelectedTargets.delete(mod);
      else state.ftSelectedTargets.add(mod);
      chip.classList.toggle('active', state.ftSelectedTargets.has(mod));
      renderer.setLoraTargets([...state.ftSelectedTargets]);
      updateFinetuneEstimate();
    });
    wrap.appendChild(chip);
  }
}

function getCurrentFinetuneConfig() {
  return {
    algorithm: state.ftAlgorithm,
    learning_rate: Number($('ft-lr').value || 5e-6),
    max_steps: Number($('ft-steps').value || 32),
    clip_ratio: Number($('ft-clip').value || 0.2),
    kl_coef: Number($('ft-kl').value || 0.02),
    num_generations: Number($('ft-group').value || 4),
    batch_size: Number($('ft-batch').value || 4),
    grad_accum: Number($('ft-accum').value || 4),
    use_lora: $('ft-use-lora').checked,
    lora_r: Number($('ft-lora-r').value || 16),
    lora_alpha: Number($('ft-lora-alpha').value || 32),
    precision: $('ft-precision').value,
    provider: $('ft-provider').value,
    target_modules: [...state.ftSelectedTargets],
    dataset_path: state.ftDatasetPath || 'improved/curated.jsonl',
    gradient_checkpointing: true,
  };
}

async function updateFinetuneEstimate() {
  if (!state.modelId) return;
  try {
    const est = await apiPost('/finetune/estimate', {
      model_id: state.modelId,
      checkpoint_id: state.checkpointId,
      config: getCurrentFinetuneConfig(),
    });
    const v = est.vram_gb;
    $('vram-total-label').textContent = `${v.total.toFixed(1)} / ${v.device_capacity.toFixed(0)} GB (${v.utilization_pct}%) · ~$${est.compute.est_cost_usd.toFixed(2)}`;
    const cap = Math.max(v.total, v.device_capacity, 1);
    const segs = [
      { label: 'Base', val: v.base_weights, color: '#38bdf8' },
      { label: 'LoRA+Grad', val: v.adapters + v.gradients, color: '#34d399' },
      { label: 'AdamW', val: v.optimizer, color: '#fbbf24' },
      { label: 'Act+KV', val: v.activations + v.kv_cache, color: '#a78bfa' },
    ];
    $('vram-stacked-bar').innerHTML = segs
      .map((s) => `<div class="vram-seg" style="width:${(s.val / cap) * 100}%;background:${s.color}" title="${s.label}: ${s.val.toFixed(2)} GB"></div>`)
      .join('');
    $('vram-legend').innerHTML =
      segs.map((s) => `<span class="vram-item"><span class="vram-swatch" style="background:${s.color}"></span>${s.label} ${s.val.toFixed(1)}G</span>`).join('') +
      `<span class="vram-item" style="color:#fbbf24">Trainable ${est.trainable_pct.toFixed(2)}% (${formatParams(est.trainable_params)})</span>` +
      (est.workspace_validated ? '<span class="vram-item" style="color:#34d399">✓ Schema Valid</span>' : '');

    const warnBox = $('ft-warnings');
    if (est.warnings?.length) {
      warnBox.textContent = est.warnings.join(' ');
      warnBox.classList.remove('hidden');
    } else {
      warnBox.classList.add('hidden');
    }
    $('ft-cli-preview').textContent = est.cli_command;
  } catch {
    // Ignore transient estimation errors
  }
}

function bindFinetuneControls() {
  for (const id of ['ft-lr', 'ft-steps', 'ft-clip', 'ft-kl', 'ft-group', 'ft-batch', 'ft-accum', 'ft-use-lora', 'ft-lora-r', 'ft-lora-alpha', 'ft-precision', 'ft-provider']) {
    $(id).addEventListener('change', updateFinetuneEstimate);
  }
  $('ft-dataset-select').addEventListener('change', async (e) => {
    state.ftDatasetPath = e.target.value;
    await inspectSelectedDataset(state.ftDatasetPath);
    await updateFinetuneEstimate();
  });

  $('btn-copy-cli').addEventListener('click', () => {
    navigator.clipboard?.writeText($('ft-cli-preview').textContent || '');
    showToast('Copied StateSet CLI command');
  });

  $('btn-ft-start').addEventListener('click', startLiveFinetune);
  $('btn-ft-pause').addEventListener('click', async () => {
    if (!state.ftJobId) return;
    const paused = $('btn-ft-pause').dataset.paused === 'true';
    const res = await apiPost(`/finetune/jobs/${state.ftJobId}/${paused ? 'resume' : 'pause'}`, {});
    $('btn-ft-pause').dataset.paused = String(!paused);
    $('btn-ft-pause').textContent = paused ? '⏸' : '▶';
    $('ft-status-pill').textContent = res.status.toUpperCase();
  });
  $('btn-ft-stop').addEventListener('click', async () => {
    if (!state.ftJobId) return;
    await apiPost(`/finetune/jobs/${state.ftJobId}/stop`, {});
    finishFinetuneStream('STOPPED');
  });
}

async function startLiveFinetune() {
  if (!state.modelId) return;
  if (state.ftStream) {
    state.ftStream.close();
    state.ftStream = null;
  }
  state.ftHistory = [];
  const cfg = getCurrentFinetuneConfig();
  const job = await apiPost('/finetune/start', {
    model_id: state.modelId,
    checkpoint_id: state.checkpointId,
    config: cfg,
  });
  state.ftJobId = job.job_id;
  $('ft-status-pill').textContent = 'RUNNING';
  $('ft-status-pill').classList.add('running');
  $('btn-ft-start').disabled = true;
  $('btn-ft-pause').disabled = false;
  $('btn-ft-stop').disabled = false;
  showToast(`Started live PyTorch ${cfg.algorithm.toUpperCase()} micro-finetuning (${job.total_steps} steps)`);

  const es = new EventSource(`${API}/finetune/stream/${job.job_id}`);
  state.ftStream = es;

  let last3DRefresh = 0;
  es.onmessage = async (evt) => {
    const msg = JSON.parse(evt.data);
    if (msg.type === 'step' && msg.metric) {
      const m = msg.metric;
      state.ftHistory.push(m);
      $('ft-step-counter').textContent = `${m.step} / ${m.total_steps}`;
      drawTrainingTelemetryChart($('chart-training'), state.ftHistory);

      const heatMap = {};
      const grads = Object.entries(m.layer_grad_norms || {});
      const maxG = Math.max(1e-5, ...grads.map(([, v]) => v));
      for (const [lName, gVal] of grads) {
        const match = lName.match(/(\d+)/);
        if (match) heatMap[Number(match[1])] = gVal / maxG;
      }
      renderer.triggerPulse('backprop', heatMap);

      const now = performance.now();
      if (now - last3DRefresh > 1100 || m.step === m.total_steps) {
        last3DRefresh = now;
        await reloadAfterEdit(null);
      }
    } else if (msg.type === 'completed') {
      await reloadAfterEdit('Live PyTorch finetuning completed — 3D weights morphed!');
      finishFinetuneStream('COMPLETED');
    } else if (msg.type === 'failed') {
      showToast(`Finetune failed: ${msg.error}`, 3600);
      finishFinetuneStream('FAILED');
    }
  };

  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) return;
    es.close();
    if (!state.ftPollTimer) {
      state.ftPollTimer = setInterval(async () => {
        if (!state.ftJobId) return;
        const j = await apiGet(`/finetune/jobs/${state.ftJobId}`);
        state.ftHistory = j.history || [];
        $('ft-step-counter').textContent = `${j.current_step} / ${j.total_steps}`;
        drawTrainingTelemetryChart($('chart-training'), state.ftHistory);
        if (['completed', 'stopped', 'failed'].includes(j.status)) {
          await reloadAfterEdit('Live finetuning finished');
          finishFinetuneStream(j.status.toUpperCase());
        }
      }, 650);
    }
  };
}

function finishFinetuneStream(statusText) {
  if (state.ftStream) {
    state.ftStream.close();
    state.ftStream = null;
  }
  if (state.ftPollTimer) {
    clearInterval(state.ftPollTimer);
    state.ftPollTimer = null;
  }
  $('ft-status-pill').textContent = statusText;
  $('ft-status-pill').classList.remove('running');
  $('ft-status-pill').classList.toggle('active', statusText === 'COMPLETED');
  $('btn-ft-start').disabled = false;
  $('btn-ft-pause').disabled = true;
  $('btn-ft-pause').dataset.paused = 'false';
  $('btn-ft-pause').textContent = '⏸';
  $('btn-ft-stop').disabled = true;
}

// ---------------------------------------------------------------------------
// Interactive Model Playground, Causal Activation Patching & Logit Lens
// ---------------------------------------------------------------------------

async function loadInferenceScenarios() {
  const data = await apiGet('/inference/scenarios');
  const wrap = $('scenario-chips');
  wrap.innerHTML = '';
  for (const sc of data.scenarios || []) {
    const chip = document.createElement('button');
    chip.className = 'target-chip';
    chip.textContent = sc.title.split('·')[1]?.trim() || sc.title;
    chip.title = sc.prompt;
    chip.addEventListener('click', () => {
      $('inf-system').value = sc.system;
      $('inf-prompt').value = sc.prompt;
      runModelInference();
    });
    wrap.appendChild(chip);
  }
}

function renderActivePatches() {
  const wrap = $('active-patches-wrap');
  if (!state.causalPatches.length) {
    wrap.innerHTML = '<span class="empty-hint" style="padding:2px">No active causal patches (unpatched forward pass).</span>';
    return;
  }
  wrap.innerHTML = '';
  state.causalPatches.forEach((p, idx) => {
    const chip = document.createElement('button');
    chip.className = 'target-chip active';
    chip.style.borderColor = 'rgba(251, 113, 133, 0.55)';
    chip.style.color = '#fda4af';
    chip.textContent = `L${String(p.layer).padStart(2, '0')} · ${p.action} ✕`;
    chip.addEventListener('click', () => {
      state.causalPatches.splice(idx, 1);
      renderActivePatches();
    });
    wrap.appendChild(chip);
  });
}

function bindInferenceControls() {
  $('inf-temp').addEventListener('input', (e) => { $('inf-temp-val').textContent = Number(e.target.value).toFixed(2); });
  $('inf-topp').addEventListener('input', (e) => { $('inf-topp-val').textContent = Number(e.target.value).toFixed(2); });
  $('btn-run-infer').addEventListener('click', runModelInference);
  $('btn-run-circuit').addEventListener('click', () => discoverCircuit({ switchMode: true }));

  $('btn-add-patch').addEventListener('click', () => {
    const layer = Number($('patch-layer-select').value || 0);
    const action = $('patch-action-select').value;
    state.causalPatches = state.causalPatches.filter((p) => p.layer !== layer);
    state.causalPatches.push({ layer, action });
    renderActivePatches();
    showToast(`Added causal patch: Layer ${layer} → ${action}`);
  });
  $('btn-clear-patches').addEventListener('click', () => {
    state.causalPatches = [];
    renderActivePatches();
  });

  // Contrastive Activation Steering (CAA / RepE) controls
  $('steer-preset-select').addEventListener('change', (e) => {
    const p = state.steeringPresets.find((item) => item.id === e.target.value);
    if (p) {
      $('steer-pos-prompt').value = p.positive;
      $('steer-neg-prompt').value = p.negative;
    }
  });
  $('btn-extract-steer').addEventListener('click', extractSteeringVector);
  $('steer-alpha').addEventListener('input', (e) => {
    state.steeringAlpha = Number(e.target.value);
    $('steer-alpha-val').textContent = `${state.steeringAlpha >= 0 ? '+' : ''}${state.steeringAlpha.toFixed(2)}`;
    if (state.activeSteeringVector) {
      $('steer-status-pill').textContent = `α=${state.steeringAlpha >= 0 ? '+' : ''}${state.steeringAlpha.toFixed(2)}`;
    }
  });
  $('steer-alpha').addEventListener('change', () => {
    if (state.activeSteeringVector) runModelInference();
  });
  $('btn-bake-steer').addEventListener('click', bakeSteeringVector);

  renderActivePatches();
}

async function loadSteeringPresets() {
  try {
    const data = await apiGet('/steering/presets');
    state.steeringPresets = data.presets || [];
    const sel = $('steer-preset-select');
    sel.innerHTML = '';
    for (const p of state.steeringPresets) {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = p.name;
      sel.appendChild(opt);
    }
    if (state.steeringPresets.length) {
      $('steer-pos-prompt').value = state.steeringPresets[0].positive;
      $('steer-neg-prompt').value = state.steeringPresets[0].negative;
    }
  } catch {
    // Optional
  }
}

async function extractSteeringVector() {
  if (!state.modelId) return;
  const preset = state.steeringPresets.find((p) => p.id === $('steer-preset-select').value);
  const pos = ($('steer-pos-prompt').value || '').trim();
  const neg = ($('steer-neg-prompt').value || '').trim();
  if (!pos || !neg) {
    showToast('Positive and negative contrast prompts are required');
    return;
  }
  const layerVal = $('steer-layer-select').value;
  try {
    const vec = await apiPost('/steering/extract', {
      model_id: state.modelId,
      checkpoint_id: state.checkpointId,
      name: preset?.name || 'Custom Steering',
      positive_prompt: pos,
      negative_prompt: neg,
      layer_index: layerVal !== '' ? Number(layerVal) : null,
    });
    state.activeSteeringVector = vec;
    $('steer-layer-select').value = String(vec.recommended_layer);
    if (Math.abs(state.steeringAlpha) < 0.05) {
      state.steeringAlpha = 0.8;
      $('steer-alpha').value = '0.8';
      $('steer-alpha-val').textContent = '+0.80';
    }
    $('steer-status-pill').textContent = `L${String(vec.recommended_layer).padStart(2, '0')} · sep=${vec.peak_separation}`;
    $('btn-bake-steer').disabled = false;
    showToast(`Extracted CAA steering vector at Layer ${vec.recommended_layer} (sep=${vec.peak_separation})`);
    await runModelInference();
  } catch (err) {
    showToast(`Steering extraction failed: ${err.message}`, 3200);
  }
}

async function bakeSteeringVector() {
  if (!state.modelId || !state.activeSteeringVector) return;
  const strength = Math.abs(state.steeringAlpha) > 0.05 ? state.steeringAlpha : 0.75;
  const targetLayer = Number($('steer-layer-select').value || state.activeSteeringVector.recommended_layer);
  try {
    const res = await apiPost('/steering/bake', {
      model_id: state.modelId,
      checkpoint_id: state.checkpointId,
      vector_id: state.activeSteeringVector.vector_id,
      strength,
      target_layer: targetLayer,
    });
    renderer.triggerPulse('backprop');
    await reloadAfterEdit(`Baked CAA vector into Layer ${targetLayer} (${res.edit.changed_weights.toLocaleString()} w)`);
  } catch (err) {
    showToast(`Bake failed: ${err.message}`, 3200);
  }
}

async function discoverCircuit({ switchMode = true } = {}) {
  if (!state.modelId) return;
  const prompt = ($('inf-prompt').value || '').trim() || 'Customer says: My order #77701 arrived damaged. Can you refund me?';
  const topK = Number($('circuit-topk-slider')?.value || 10);
  try {
    const data = await apiPost('/inference/circuit', {
      model_id: state.modelId,
      checkpoint_id: state.checkpointId,
      prompt,
      system_prompt: $('inf-system')?.value || '',
      top_k: topK,
    });
    state.circuitData = data;
    $('circuit-summary-val').textContent = `Top-${data.summary.circuit_nodes} · ${data.summary.arc_count} Arcs`;
    $('circuit-sparsity-pill').textContent = `${(data.summary.circuit_sparsity * 100).toFixed(0)}% sparse`;

    const list = $('circuit-node-list');
    list.innerHTML = '';
    for (const nd of data.top_nodes || []) {
      const row = document.createElement('div');
      row.className = 'circuit-node-row';
      row.innerHTML = `
        <span style="color:${nd.kind === 'mlp' ? '#34d399' : '#fda4af'};font-weight:700">${nd.id}</span>
        <span style="color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${nd.role_tag}</span>
        <span style="color:#fbbf24;text-align:right">${(nd.importance * 100).toFixed(0)}%</span>
        <button class="mini-btn" title="Ablate Layer ${nd.layer} in Causal Patching">Ablate</button>
      `;
      row.querySelector('button').addEventListener('click', (ev) => {
        ev.stopPropagation();
        state.causalPatches = state.causalPatches.filter((p) => p.layer !== nd.layer);
        state.causalPatches.push({ layer: nd.layer, action: 'ablate' });
        renderActivePatches();
        runModelInference();
        showToast(`Ablated causal circuit node ${nd.id} (Layer ${nd.layer})`);
      });
      row.addEventListener('click', () => {
        if (nd.tensor_name) selectTensor(nd.tensor_name, { focusCamera: false });
      });
      list.appendChild(row);
    }

    renderer.setData({ circuitData: data });
    if (switchMode) {
      document.querySelectorAll('.mode-pill').forEach((btn) => {
        const active = btn.dataset.mode === 'circuit';
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-selected', String(active));
      });
      $('microscope-hud').classList.add('hidden');
      $('spectral-hud').classList.add('hidden');
      $('circuit-hud').classList.remove('hidden');
      renderer.setViewMode('circuit');
    }
    renderer.triggerPulse('forward');
    showToast(`Discovered ${data.summary.circuit_nodes}-node causal circuit (${data.summary.arc_count} 3D flow arcs)`);
  } catch (err) {
    showToast(`Circuit discovery failed: ${err.message}`, 3200);
  }
}

function getActiveSteeringPayload() {
  if (!state.activeSteeringVector || Math.abs(state.steeringAlpha) < 0.05) return null;
  return {
    vector_id: state.activeSteeringVector.vector_id,
    strength: state.steeringAlpha,
    layer: Number($('steer-layer-select').value || state.activeSteeringVector.recommended_layer),
  };
}

async function runModelInference() {
  if (!state.modelId) return;
  const prompt = ($('inf-prompt').value || '').trim();
  if (!prompt) return;

  const steeringPayload = getActiveSteeringPayload();
  const res = await apiPost('/inference/run', {
    model_id: state.modelId,
    checkpoint_id: state.checkpointId,
    prompt,
    system_prompt: $('inf-system').value || '',
    temperature: Number($('inf-temp').value || 0.7),
    top_p: Number($('inf-topp').value || 0.9),
    max_tokens: Number($('inf-maxtok').value || 48),
    compare_base: $('inf-compare-base').checked,
    causal_patches: state.causalPatches,
    steering: steeringPayload,
    endpoint_url: $('inf-endpoint-url').value || '',
    api_key: $('inf-api-key').value || '',
  });
  state.lastInferResult = res;

  const badge = $('inf-backend-badge');
  badge.textContent = (res.execution_backend || 'probe').replace('_', ' ').toUpperCase();
  badge.classList.toggle('active', res.execution_backend === 'pytorch_hooks' || res.execution_backend === 'remote_vllm');

  const actMap = {};
  for (const la of res.active.layer_activations || []) {
    actMap[la.layer] = la.intensity;
  }
  renderer.triggerPulse('forward', actMap);

  $('inf-latency').textContent = `${res.latency_ms} ms · H=${res.active.mean_entropy}`;

  const stream = $('inf-token-stream');
  stream.innerHTML = '';
  res.active.tokens.forEach((tok, idx) => {
    const span = document.createElement('span');
    span.className = `tok-chip${idx === 0 ? ' selected' : ''}`;
    const klGlow = Math.min(1, tok.kl_vs_base * 6);
    span.style.background = klGlow > 0.25
      ? `rgba(251, 191, 36, ${0.18 + klGlow * 0.35})`
      : `rgba(56, 189, 248, ${0.10 + tok.prob * 0.28})`;
    span.textContent = tok.token;
    span.title = `p=${(tok.prob * 100).toFixed(1)}% · H=${tok.entropy} · KL=${tok.kl_vs_base}`;
    span.addEventListener('click', () => {
      stream.querySelectorAll('.tok-chip').forEach((c) => c.classList.remove('selected'));
      span.classList.add('selected');
      renderTokenLogitLens(tok);
    });
    stream.appendChild(span);
  });

  if (res.active.tokens.length) renderTokenLogitLens(res.active.tokens[0]);

  const ab = $('inf-ab-compare');
  const hasIntervention = res.has_edits || state.causalPatches.length > 0 || Boolean(steeringPayload);
  if (hasIntervention && $('inf-compare-base').checked) {
    ab.classList.remove('hidden');
    $('inf-base-text').textContent = res.base.text;
    const rd = res.comparison.reward_delta;
    $('inf-shift-metrics').textContent =
      `cos=${res.comparison.hidden_cosine_similarity.toFixed(4)} · ΔReward=${rd >= 0 ? '+' : ''}${rd.toFixed(3)}`;
  } else {
    ab.classList.add('hidden');
  }

  const rw = res.active.rewards;
  $('reward-bars').innerHTML = Object.entries(rw)
    .map(([k, val]) => `
      <div class="bar-row">
        <span style="color:#94a3b8">${k.replace('_', ' ')}</span>
        <div class="bar-track"><div class="bar-fill reward" style="width:${Math.round(val * 100)}%"></div></div>
        <span style="text-align:right;color:#34d399">${val.toFixed(3)}</span>
      </div>
    `)
    .join('');

  // Render Layer-by-Layer Logit Lens Matrix!
  const lensWrap = $('layer-logit-lens');
  lensWrap.innerHTML = '';
  for (const row of res.active.logit_lens || []) {
    const top1 = row.top_tokens?.[0] || { token: '—', prob: 0 };
    const el = document.createElement('div');
    el.className = `lens-row${row.patched ? ' patched' : ''}`;
    el.title = `Click to focus 3D camera on ${row.name} (H=${row.entropy})`;
    el.innerHTML = `
      <span style="color:#94a3b8">${row.name}</span>
      <span style="color:#fde68a;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">"${top1.token}"</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.round(top1.prob * 100)}%"></div></div>
      <span style="text-align:right">${(top1.prob * 100).toFixed(0)}%</span>
    `;
    el.addEventListener('click', () => {
      const targetLayer = state.structure?.layers?.find((l) => l.index === row.layer);
      const firstT = targetLayer?.tensors?.[0];
      if (firstT) selectTensor(firstT.name, { focusCamera: true });
    });
    lensWrap.appendChild(el);
  }
}

function renderTokenLogitLens(tok) {
  $('inf-selected-tok').textContent = `"${tok.token.trim() || '␣'}" · p=${(tok.prob * 100).toFixed(1)}%`;
  $('logit-bars').innerHTML = (tok.top_candidates || [])
    .map((cand) => `
      <div class="bar-row">
        <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${cand.token}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.round(cand.prob * 100)}%"></div></div>
        <span style="text-align:right">${(cand.prob * 100).toFixed(1)}%</span>
      </div>
    `)
    .join('');
}

// ---------------------------------------------------------------------------
// Verified .safetensors Export & merge_adapter Effect Probe Modal
// ---------------------------------------------------------------------------

function bindExportModal() {
  const modal = $('export-modal');
  $('btn-export-modal').addEventListener('click', () => {
    $('export-result-box').classList.add('hidden');
    modal.classList.remove('hidden');
  });
  $('btn-close-export').addEventListener('click', () => modal.classList.add('hidden'));
  modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.add('hidden'); });

  const syncFormatUi = () => {
    const fmt = $('export-format-select')?.value || 'safetensors';
    $('export-lora-opts')?.classList.toggle('hidden', fmt !== 'lora');
    $('export-gguf-opts')?.classList.toggle('hidden', fmt !== 'gguf');
    $('export-hfcard-opts')?.classList.toggle('hidden', fmt !== 'hf_card');
    $('export-merge-probe-row')?.classList.toggle('hidden', fmt !== 'safetensors');
    const btn = $('btn-confirm-export');
    if (btn) {
      btn.textContent = fmt === 'lora'
        ? 'Extract & Verify LoRA Adapter'
        : fmt === 'gguf'
          ? 'Write GGUF v3 + Ollama Modelfile'
          : fmt === 'hf_card'
            ? 'Write HF Model Card & Recipe'
            : 'Write & Verify .safetensors';
    }
  };
  $('export-format-select')?.addEventListener('change', syncFormatUi);
  syncFormatUi();

  $('btn-run-eval-gate').addEventListener('click', runExportEvalGate);
  $('btn-auto-repair')?.addEventListener('click', async () => {
    if (!state.modelId) return;
    const btn = $('btn-auto-repair');
    btn.disabled = true;
    btn.textContent = 'Repairing…';
    try {
      const rewardFn = $('eval-reward-select')?.value || 'composite';
      const res = await apiPost('/edits/auto-repair', {
        model_id: state.modelId,
        checkpoint_id: state.checkpointId,
        reward_fn: rewardFn,
        max_steps: 4,
      });
      await reloadAfterEdit(
        `Auto-Repair: ${res.initial_verdict} → ${res.final_verdict} (${(res.actions_taken || []).length} repair steps)`
      );
      await runExportEvalGate();
    } catch (err) {
      showToast(`Auto-Repair error: ${err.message}`, 3200);
    } finally {
      btn.disabled = false;
      btn.textContent = '🔧 Auto-Repair';
    }
  });

  $('btn-confirm-export').addEventListener('click', async () => {
    if (!state.modelId) return;
    const name = ($('export-name-input').value || '').trim() || `edited-${Date.now()}`;
    const fmt = $('export-format-select')?.value || 'safetensors';
    const box = $('export-result-box');
    try {
      if (fmt === 'hf_card') {
        const repoId = ($('export-hf-repo')?.value || '').trim();
        const m = await apiPost('/export/hf-card', {
          model_id: state.modelId,
          checkpoint_id: state.checkpointId,
          export_name: name,
          repo_id: repoId,
          include_eval: true,
        });
        box.innerHTML = `
          <b>✓ Generated HF Model Card &amp; Reproducible Recipe</b><br/>
          Card: ${m.model_card_file}<br/>
          Recipe: ${m.recipe_file}<br/>
          Upload CLI: <code>${m.cli_upload_command}</code>
        `;
        box.classList.remove('hidden');
        showToast(`Generated Hugging Face Model Card for ${m.repo_id}!`);
      } else if (fmt === 'lora') {
        const rank = Number($('export-lora-rank')?.value || 16);
        const alpha = Number($('export-lora-alpha')?.value || 32);
        const m = await apiPost('/export/lora', {
          model_id: state.modelId,
          checkpoint_id: state.checkpointId,
          export_name: name,
          rank,
          alpha,
        });
        box.innerHTML = `
          <b>✓ Extracted PEFT LoRA Adapter (r=${m.rank}, α=${m.lora_alpha})</b><br/>
          File: ${m.export_file}<br/>
          SHA-256: ${m.sha256.slice(0, 24)}… · Verified: ${m.reload_verified ? 'YES' : 'NO'}<br/>
          Compression: ${m.compression_ratio}× · Mean Rel Error: ${(m.mean_relative_error * 100).toFixed(2)}%<br/>
          Targets: ${(m.target_modules || []).join(', ')}
        `;
        box.classList.remove('hidden');
        showToast(`Extracted Diff-as-LoRA adapter (${m.compression_ratio}× compression)!`);
        await loadModels(true);
      } else if (fmt === 'gguf') {
        const quant = $('export-gguf-quant')?.value || 'Q8_0';
        const m = await apiPost('/export/gguf', {
          model_id: state.modelId,
          checkpoint_id: state.checkpointId,
          export_name: name,
          quant_type: quant,
        });
        box.innerHTML = `
          <b>✓ Exported GGUF v3 + Ollama Package (${m.quant_type})</b><br/>
          GGUF: ${m.gguf_file} (${(m.file_size_bytes / 1024).toFixed(1)} KB)<br/>
          Header Verified: ${m.header_verified ? 'GGUF v3 OK' : 'WARN'} · SHA-256: ${m.sha256.slice(0, 24)}…<br/>
          Ollama: <code>${m.ollama_command}</code>
        `;
        box.classList.remove('hidden');
        showToast(`Exported GGUF v3 (${m.quant_type}) + Ollama Modelfile!`);
      } else {
        const runProbe = $('export-merge-probe').checked;
        const manifest = await apiPost('/edits/export', {
          model_id: state.modelId,
          checkpoint_id: state.checkpointId,
          export_name: name,
          run_merge_probe: runProbe,
        });
        const probeHtml = manifest.merge_probe
          ? `<br/><b>Merge Probe:</b> effect=${manifest.merge_probe.effect_verified ? 'VERIFIED' : 'NONE'} · Δ‖W‖=${manifest.merge_probe.total_delta_frobenius} · keys remapped=${manifest.merge_probe.composite_keys_remapped}`
          : '';
        box.innerHTML = `
          <b>✓ Exported &amp; Reload-Verified</b><br/>
          File: ${manifest.export_file}<br/>
          SHA-256: ${manifest.sha256.slice(0, 24)}…<br/>
          Tensors: ${manifest.tensor_count} (${manifest.modified_tensors.length} modified)${probeHtml}
        `;
        box.classList.remove('hidden');
        showToast('Verified .safetensors checkpoint exported!');
        await loadModels(true);
      }
    } catch (err) {
      showToast(`Export error: ${err.message}`, 3500);
    }
  });
}

async function runExportEvalGate() {
  if (!state.modelId) return;
  const btn = $('btn-run-eval-gate');
  btn.disabled = true;
  btn.textContent = 'Running regression eval suite…';
  try {
    const rewardFn = $('eval-reward-select')?.value || 'composite';
    const customJsonl = ($('eval-custom-jsonl')?.value || '').trim() || null;
    const rep = await apiPost('/eval/run', {
      model_id: state.modelId,
      checkpoint_id: state.checkpointId,
      dataset_path: state.ftDatasetPath || null,
      custom_jsonl: customJsonl,
      reward_fn: rewardFn,
      max_cases: 6,
      causal_patches: state.causalPatches,
      steering: getActiveSteeringPayload(),
    });
    const pill = $('eval-verdict-pill');
    pill.textContent = rep.verdict;
    pill.className = `verdict-pill ${rep.verdict}`;
    pill.classList.remove('hidden');
    const ps = rep.paired_stats || {};
    const ciStr = ps.ci_95_low !== undefined
      ? ` · 95% CI [${ps.ci_95_low >= 0 ? '+' : ''}${ps.ci_95_low.toFixed(3)}, ${ps.ci_95_high >= 0 ? '+' : ''}${ps.ci_95_high.toFixed(3)}] · W/T/L ${ps.wins}/${ps.ties}/${ps.losses}`
      : '';
    $('eval-summary-copy').textContent = `${rep.verdict_reason} (Base ${(rep.base_summary.pass_rate * 100).toFixed(0)}% → Edited ${(rep.edited_summary.pass_rate * 100).toFixed(0)}%${ciStr} · ${rep.latency_ms} ms)`;

    const list = $('eval-case-list');
    list.classList.remove('hidden');
    list.innerHTML = (rep.cases || [])
      .map((c) => {
        const col = c.status === 'PASS' || c.status === 'IMPROVED' ? '#34d399' : c.status === 'REGRESSION' ? '#fb7185' : '#fbbf24';
        const dStr = `${c.delta >= 0 ? '+' : ''}${c.delta.toFixed(3)}`;
        return `
          <div class="eval-case-row">
            <span style="color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${c.category}</span>
            <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${c.prompt_preview}</span>
            <span style="text-align:right;color:${c.delta >= 0 ? '#34d399' : '#fb7185'}">${dStr}</span>
            <span style="text-align:right;font-weight:700;color:${col}">${c.status}</span>
          </div>
        `;
      })
      .join('');
    showToast(`Batch Eval Gate (${rewardFn}): ${rep.verdict} (${rep.case_count} cases)`);
  } catch (err) {
    showToast(`Eval gate error: ${err.message}`, 3200);
  } finally {
    btn.disabled = false;
    btn.textContent = '⚡ Re-Run Regression Eval Gate';
  }
}

// ---------------------------------------------------------------------------
// Split-Screen 3D A/B Wipe Curtain (Cross-Checkpoint / Base), SAE & Dispatch
// ---------------------------------------------------------------------------

function populateSplitComparators() {
  const sel = $('split-compare-select');
  if (!sel) return;
  const prev = sel.value || 'base';
  sel.innerHTML = '<option value="base">vs: Unedited Base</option>';

  const cps = state.structure?.model?.checkpoints || [];
  for (const cp of cps) {
    if (cp.id === state.checkpointId) continue;
    const opt = document.createElement('option');
    opt.value = `cp:${cp.id}`;
    opt.textContent = `vs: ${cp.label}`;
    sel.appendChild(opt);
  }
  for (const m of state.models || []) {
    if (m.model_id === state.modelId || m.kind === 'preset_twin') continue;
    const opt = document.createElement('option');
    opt.value = `model:${m.model_id}`;
    opt.textContent = `vs: ${m.display_name}`;
    sel.appendChild(opt);
  }
  if ([...sel.options].some((o) => o.value === prev)) {
    sel.value = prev;
  } else {
    state.splitTargetSpec = 'base';
  }
  updateSplitBadgeLabels();
}

function updateSplitBadgeLabels() {
  const sel = $('split-compare-select');
  const leftEl = $('split-left-badge');
  const rightEl = $('split-right-badge');
  if (!leftEl || !rightEl) return;
  const label = sel?.selectedOptions?.[0]?.textContent?.replace(/^vs:\s*/i, '') || 'BASE (UNEDITED)';
  leftEl.textContent = label.toUpperCase();
  const activeCp = state.structure?.checkpoint_id && state.structure.checkpoint_id !== 'default'
    ? state.structure.checkpoint_id.toUpperCase()
    : 'ACTIVE';
  rightEl.textContent = state.tensorDetail?.edited ? `${activeCp} · EDITED` : `${activeCp} (LIVE)`;
}

async function setSplitCompare(enabled) {
  state.splitCompare = Boolean(enabled);
  $('btn-toggle-split').classList.toggle('active', state.splitCompare);
  $('split-curtain').classList.toggle('hidden', !state.splitCompare);
  updateSplitBadgeLabels();
  if (state.splitCompare && state.splitTargetSpec !== 'base' && state.selectedTensorName) {
    renderer.splitCompare = true;
    await selectTensor(state.selectedTensorName, { focusCamera: false });
  } else {
    renderer.setSplitCompare(state.splitCompare, state.splitRatio);
  }
  if (state.splitCompare) {
    showToast(`Split A|B 3D Wipe active (${$('split-left-badge')?.textContent || 'BASE'} vs LIVE)`);
  }
}

async function analyzeSaeFeatures() {
  if (!state.modelId) return;
  const prompt = ($('inf-prompt')?.value || '').trim() || 'Customer says: My order #77701 arrived damaged.';
  const clamps = Object.entries(state.saeClamps).map(([feature_id, value]) => ({ feature_id, value }));
  const layerVal = $('steer-layer-select')?.value;
  try {
    const res = await apiPost('/sae/analyze', {
      model_id: state.modelId,
      checkpoint_id: state.checkpointId,
      prompt,
      system_prompt: $('inf-system')?.value || '',
      layer_index: layerVal !== '' ? Number(layerVal) : null,
      top_k: 8,
      clamps,
    });
    $('sae-stats-pill').textContent = `L${String(res.layer).padStart(2, '0')} · L0=${res.l0_active}/${res.dictionary_size} · R²=${res.reconstruction_r2}`;
    const list = $('sae-feature-list');
    list.innerHTML = '';
    for (const f of res.features || []) {
      const row = document.createElement('div');
      row.className = 'circuit-node-row';
      row.style.gridTemplateColumns = '34px 1fr 42px auto';
      row.innerHTML = `
        <span style="color:${f.clamped ? '#fbbf24' : '#38bdf8'};font-weight:700">${f.feature_id}</span>
        <span style="color:#cbd5e1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${f.label} (${f.domain})">${f.label}</span>
        <span style="color:#34d399;text-align:right">${f.activation.toFixed(2)}</span>
        <span style="display:flex;gap:3px">
          <button class="mini-btn" data-act="zero" title="Suppress feature to 0.0">0×</button>
          <button class="mini-btn" data-act="amp" title="Amplify feature to 2.5">2.5×</button>
        </span>
      `;
      row.querySelector('[data-act="zero"]').addEventListener('click', () => {
        state.saeClamps[f.feature_id] = 0.0;
        analyzeSaeFeatures();
        renderer.triggerPulse('forward');
        showToast(`Clamped SAE ${f.feature_id} (${f.label}) → 0.0`);
      });
      row.querySelector('[data-act="amp"]').addEventListener('click', () => {
        state.saeClamps[f.feature_id] = 2.5;
        analyzeSaeFeatures();
        renderer.triggerPulse('forward');
        showToast(`Amplified SAE ${f.feature_id} (${f.label}) → 2.5×`);
      });
      list.appendChild(row);
    }
    renderer.triggerPulse('forward');
  } catch (err) {
    showToast(`SAE error: ${err.message}`, 3200);
  }
}

async function dispatchValidatedRun(mode) {
  if (!state.modelId) return;
  const pill = $('dispatch-status-pill');
  const box = $('dispatch-log-box');
  pill.textContent = 'RUNNING';
  box.classList.remove('hidden');
  box.textContent = `Dispatching ${mode} run…`;
  try {
    const rec = await apiPost('/finetune/dispatch', {
      model_id: state.modelId,
      checkpoint_id: state.checkpointId,
      mode,
      config: getCurrentFinetuneConfig(),
    });
    const renderRec = (r) => {
      pill.textContent = String(r.status || '').toUpperCase();
      box.textContent = (r.logs || []).join('\n');
      box.scrollTop = box.scrollHeight;
    };
    renderRec(rec);
    if (rec.status === 'completed') {
      showToast(`Plan validated (${rec.plan_sha256.slice(0, 10)}) & catalog refreshed`);
      return;
    }
    clearInterval(state.dispatchPollTimer);
    state.dispatchPollTimer = setInterval(async () => {
      try {
        const cur = await apiGet(`/finetune/dispatch/${rec.run_id}`);
        renderRec(cur);
        if (['completed', 'failed', 'stopped'].includes(cur.status)) {
          clearInterval(state.dispatchPollTimer);
          state.dispatchPollTimer = null;
          if (cur.status === 'completed') {
            showToast('Local CLI run completed — checkpoints hot-reloaded!');
            await loadModels(true);
          }
        }
      } catch {
        clearInterval(state.dispatchPollTimer);
        state.dispatchPollTimer = null;
      }
    }, 900);
  } catch (err) {
    pill.textContent = 'ERROR';
    box.textContent = String(err.message);
    showToast(`Dispatch failed: ${err.message}`, 3200);
  }
}

function bindSplitCurtain() {
  const divider = $('split-divider');
  const stage = document.querySelector('.viewport-stage');
  if (divider && stage) {
    let dragging = false;
    divider.addEventListener('pointerdown', (e) => {
      e.stopPropagation();
      dragging = true;
      divider.setPointerCapture(e.pointerId);
    });
    divider.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const rect = stage.getBoundingClientRect();
      const frac = Math.max(0.08, Math.min(0.92, (e.clientX - rect.left) / Math.max(rect.width, 1)));
      state.splitRatio = frac;
      divider.style.left = `${(frac * 100).toFixed(2)}%`;
      renderer.setSplitCompare(true, frac);
    });
    divider.addEventListener('pointerup', () => { dragging = false; });
  }

  $('split-compare-select')?.addEventListener('change', async (e) => {
    state.splitTargetSpec = e.target.value;
    updateSplitBadgeLabels();
    if (!state.splitCompare && state.splitTargetSpec !== 'base') {
      await setSplitCompare(true);
    } else if (state.splitCompare && state.selectedTensorName) {
      await selectTensor(state.selectedTensorName, { focusCamera: false });
    }
  });

  $('btn-analyze-sae')?.addEventListener('click', analyzeSaeFeatures);
  $('btn-clear-sae-clamps')?.addEventListener('click', () => {
    state.saeClamps = {};
    analyzeSaeFeatures();
    showToast('Cleared all SAE feature clamps');
  });

  $('btn-dispatch-dryrun')?.addEventListener('click', () => dispatchValidatedRun('dry_run'));
  $('btn-dispatch-local')?.addEventListener('click', () => dispatchValidatedRun('local_stub'));

  // Studio v5: GPU Photon Particles + 3D Tomographic MRI Slice Plane
  $('btn-particles-toggle')?.addEventListener('click', (e) => {
    renderer.showParticles = !renderer.showParticles;
    e.currentTarget.classList.toggle('active', renderer.showParticles);
  });
  $('btn-slice-toggle')?.addEventListener('click', (e) => {
    const active = renderer.sliceAxis === 0;
    renderer.sliceAxis = active ? Number($('slice-axis-select')?.value || 2) : 0;
    renderer.slicePos = Number($('slice-pos-slider')?.value || 2);
    e.currentTarget.classList.toggle('active', active);
    $('slice-hud')?.classList.toggle('hidden', !active);
    showToast(active ? '3D MRI Cutaway Plane active — scrub slider to slice interior weights' : '3D MRI Cutaway OFF', 2000);
  });
  $('slice-axis-select')?.addEventListener('change', (e) => {
    if (renderer.sliceAxis > 0) renderer.sliceAxis = Number(e.target.value);
  });
  $('slice-pos-slider')?.addEventListener('input', (e) => {
    renderer.slicePos = Number(e.target.value);
    const v = renderer.slicePos;
    $('slice-pos-val').textContent = `${v >= 0 ? '+' : ''}${v.toFixed(1)}`;
  });

  // Studio v5: Zero-Download Hugging Face HTTP-Range Modal
  const hfModal = $('hf-modal');
  $('btn-hf-modal')?.addEventListener('click', () => hfModal?.classList.remove('hidden'));
  $('btn-close-hf')?.addEventListener('click', () => hfModal?.classList.add('hidden'));
  hfModal?.addEventListener('click', (e) => { if (e.target === hfModal) hfModal.classList.add('hidden'); });
  $('btn-confirm-hf')?.addEventListener('click', async () => {
    const repo_id = ($('hf-repo-input')?.value || '').trim();
    if (!repo_id) { showToast('Enter a Hugging Face repo ID (org/model)'); return; }
    const btn = $('btn-confirm-hf');
    btn.disabled = true;
    btn.textContent = 'Streaming headers…';
    try {
      const res = await apiPost('/models/register-hf', {
        repo_id,
        revision: ($('hf-rev-input')?.value || 'main').trim(),
        token: ($('hf-token-input')?.value || '').trim(),
      });
      hfModal.classList.add('hidden');
      await loadModels(false);
      $('model-select').value = res.model.model_id;
      await loadModelStructure(res.model.model_id, null);
      showToast(`Streamed HF headers for ${repo_id} (${res.model.num_layers}L · ${res.model.num_tensors}T)`);
    } catch (err) {
      showToast(`HF Range error: ${err.message}`, 3600);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Stream Headers Over HTTP Range';
    }
  });

  // Studio v5: Auto-Sensitivity Profiler & Adaptive Rank/Sparsity Optimizer
  let lastAdaptiveRecipe = null;
  $('btn-run-sensitivity')?.addEventListener('click', async () => {
    if (!state.modelId) return;
    try {
      const rep = await apiPost('/models/sensitivity', {
        model_id: state.modelId,
        checkpoint_id: state.checkpointId,
        target_compression: 0.35,
      });
      lastAdaptiveRecipe = rep.recommended_recipe;
      $('sens-comp-pill').textContent = `−${rep.estimated_compression_pct}% params`;
      $('btn-apply-adaptive').disabled = false;

      const heat = {};
      const list = $('sensitivity-list');
      list.innerHTML = '';
      for (const l of rep.layers || []) {
        heat[l.layer] = l.sensitivity;
        const col = l.tier === 'CRITICAL' ? '#fb7185' : l.tier === 'BALANCED' ? '#fbbf24' : '#34d399';
        const row = document.createElement('div');
        row.className = 'circuit-node-row';
        row.innerHTML = `
          <span style="color:${col};font-weight:700">${l.name}</span>
          <span style="color:#94a3b8">${l.tier} · r=${l.recommended_rank}</span>
          <span style="color:#38bdf8;text-align:right">sp ${Math.round(l.recommended_sparsity * 100)}%</span>
          <span style="color:${col};text-align:right">${(l.sensitivity * 100).toFixed(0)}%</span>
        `;
        list.appendChild(row);
      }
      renderer.triggerPulse('forward', heat);
      showToast(`Profiled ${rep.layers.length} layers — adaptive recipe saves ~${rep.estimated_compression_pct}% params`);
    } catch (err) {
      showToast(`Sensitivity error: ${err.message}`, 3200);
    }
  });

  $('btn-apply-adaptive')?.addEventListener('click', async () => {
    if (!state.modelId || !lastAdaptiveRecipe) return;
    try {
      const res = await apiPost('/edits/recipe/apply', {
        model_id: state.modelId,
        checkpoint_id: state.checkpointId,
        recipe: lastAdaptiveRecipe,
        reset_first: false,
      });
      await reloadAfterEdit(`Applied adaptive sensitivity recipe (${res.applied_count} layer ops)`);
    } catch (err) {
      showToast(`Adaptive apply failed: ${err.message}`, 3200);
    }
  });

  // Studio v5: External Trainer Telemetry Sync (trainer_state.json / W&B / JSONL)
  (async () => {
    try {
      const res = await apiGet('/finetune/telemetry-sources');
      const sel = $('telemetry-source-select');
      if (!sel) return;
      sel.innerHTML = '';
      if (!res.sources?.length) {
        sel.innerHTML = '<option value="">No local trainer_state.json / W&B logs</option>';
      } else {
        for (const s of res.sources) {
          const opt = document.createElement('option');
          opt.value = s.path;
          opt.textContent = `${s.path} (${s.size_kb} KB)`;
          sel.appendChild(opt);
        }
      }
    } catch { /* optional */ }
  })();

  $('btn-sync-telemetry')?.addEventListener('click', async () => {
    const path = $('telemetry-source-select')?.value;
    if (!path) { showToast('No telemetry file selected'); return; }
    try {
      const res = await apiPost('/finetune/sync-telemetry', { path });
      state.ftHistory = res.history || [];
      drawTrainingTelemetryChart($('chart-training'), state.ftHistory);
      $('telemetry-sync-pill').textContent = `${res.points} pts`;
      showToast(`Synced ${res.points} telemetry points & refreshed checkpoints`);
    } catch (err) {
      showToast(`Telemetry sync failed: ${err.message}`, 3200);
    }
  });

  // Studio v5: Shareable Single-File Interactive HTML PR Report Export
  $('btn-export-html-report')?.addEventListener('click', async () => {
    if (!state.modelId) return;
    const btn = $('btn-export-html-report');
    btn.disabled = true;
    btn.textContent = 'Building HTML…';
    try {
      const resp = await fetch(`${API}/report/html`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: state.modelId,
          checkpoint_id: state.checkpointId,
          tensor_name: state.selectedTensorName,
          include_eval: true,
          include_sensitivity: true,
        }),
      });
      if (!resp.ok) throw new Error(resp.statusText);
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `stateset-weight-studio-report-${Date.now()}.html`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
      showToast('Exported standalone interactive 3D HTML PR report!');
    } catch (err) {
      showToast(`HTML Report error: ${err.message}`, 3200);
    } finally {
      btn.disabled = false;
      btn.textContent = '📄 HTML Report';
    }
  });

  // Studio v6/v7: Multi-Head Attention 3D + Direct Head Ablation & Residual 3D
  $('btn-refresh-attention')?.addEventListener('click', () => loadAttentionMap({ switchMode: false }));
  $('attn-layer-select')?.addEventListener('change', () => loadAttentionMap({ switchMode: false }));
  $('attn-head-select')?.addEventListener('change', (e) => {
    const hVal = Number(e.target.value);
    renderer.setData({ attentionHeadFilter: hVal });
    renderer._fitCameraToViewMode();
  });
  $('btn-ablate-attn-head')?.addEventListener('click', ablateSelectedAttentionHead);
  $('btn-refresh-residual')?.addEventListener('click', () => loadResidualStream({ switchMode: false }));

  // Studio v7: Workspace Session Persistence (Save / Load)
  $('btn-save-session')?.addEventListener('click', async () => {
    if (!state.modelId) return;
    const name = ($('session-name-input')?.value || '').trim() || `studio-${Date.now()}`;
    try {
      const res = await apiPost('/sessions/save', {
        session_name: name,
        model_id: state.modelId,
        checkpoint_id: state.checkpointId,
        metadata: {
          prompt: $('inf-prompt')?.value || '',
          sae_atlas: $('sae-dict-select')?.value || 'stateset_builtin',
          causal_patches: state.causalPatches,
        },
      });
      showToast(`Saved session '${res.session_name}' (${res.operation_count} ops)`);
    } catch (err) {
      showToast(`Save session error: ${err.message}`, 3200);
    }
  });
  $('btn-load-session')?.addEventListener('click', async () => {
    const name = ($('session-name-input')?.value || '').trim();
    if (!name) { showToast('Enter a saved session name to restore'); return; }
    try {
      const res = await apiPost('/sessions/load', { session_name: name });
      if (res.metadata?.prompt && $('inf-prompt')) $('inf-prompt').value = res.metadata.prompt;
      await reloadAfterEdit(`Restored session '${res.session_name}' (${res.replay.applied_count} ops replayed)`);
    } catch (err) {
      showToast(`Load session error: ${err.message}`, 3200);
    }
  });

  // Studio v6: Pretrained SAE Dictionary Atlas Loader
  (async () => {
    try {
      const res = await apiGet('/sae/dictionaries');
      const sel = $('sae-dict-select');
      if (!sel) return;
      sel.innerHTML = '';
      for (const d of res.dictionaries || []) {
        const opt = document.createElement('option');
        opt.value = d.id;
        opt.textContent = d.name;
        sel.appendChild(opt);
      }
    } catch { /* optional */ }
  })();

  $('btn-load-sae-dict')?.addEventListener('click', async () => {
    const sourceId = $('sae-dict-select')?.value || 'stateset_builtin';
    try {
      const res = await apiPost('/sae/load', { source_id: sourceId });
      showToast(`Loaded SAE Atlas: ${res.name} (${res.feature_count} features)`);
      await analyzeSaeFeatures();
    } catch (err) {
      showToast(`SAE load error: ${err.message}`, 3200);
    }
  });

  // Studio v6: Multi-Edit A/B Arena
  $('btn-run-arena')?.addEventListener('click', runArenaComparison);

  // Studio v6: Command Palette (Ctrl+K / ⌘K)
  bindCommandPalette();
}

async function ablateSelectedAttentionHead() {
  if (!state.modelId) return;
  const layerIdx = Number($('attn-layer-select')?.value || 0);
  const rawHead = Number($('attn-head-select')?.value ?? -1);
  const headIdx = rawHead >= 0 ? rawHead : 0;
  try {
    const layer = state.structure?.layers?.find((l) => l.index === layerIdx);
    const attnTensor = (layer?.tensors || []).find((t) => /attn|q_proj|o_proj|qkv/i.test(t.name)) || layer?.tensors?.[0];
    if (!attnTensor) {
      showToast(`No attention tensor found on Layer ${layerIdx}`);
      return;
    }
    await executeSurgicalEdit(
      'head_edit',
      attnTensor.name,
      { head_index: headIdx, num_heads: 8, action: 'ablate' },
      `Ablate Head L${String(layerIdx).padStart(2, '0')}.H${headIdx}`
    );
    await loadAttentionMap({ switchMode: false });
  } catch (err) {
    showToast(`Head ablation error: ${err.message}`, 3200);
  }
}

async function loadResidualStream({ switchMode = true } = {}) {
  if (!state.modelId) return;
  const prompt = ($('inf-prompt')?.value || '').trim() || 'Customer says: My order #77701 arrived damaged after 9 days. Can you refund me?';
  try {
    const data = await apiPost('/inference/residual', {
      model_id: state.modelId,
      checkpoint_id: state.checkpointId,
      prompt,
      system_prompt: $('inf-system')?.value || '',
      causal_patches: state.causalPatches,
      steering: getActiveSteeringPayload(),
    });
    state.residualData = data;
    $('residual-summary-val').textContent = `${data.layer_count}L · Peak Drift L${String(data.peak_drift_layer).padStart(2, '0')} (${(data.max_cosine_drift * 100).toFixed(2)}%)`;
    renderer.setData({ residualData: data });
    if (switchMode) renderer.setViewMode('residual');
    renderer.triggerPulse('forward');
    showToast(`Traced ${data.layer_count}-layer Base vs. Active Residual Stream (${data.latency_ms} ms)`);
  } catch (err) {
    showToast(`Residual 3D error: ${err.message}`, 3200);
  }
}

async function loadAttentionMap({ switchMode = true } = {}) {
  if (!state.modelId) return;
  const prompt = ($('inf-prompt')?.value || '').trim() || 'Customer says: My order #77701 arrived damaged after 9 days. Can you refund me?';
  const layerSel = $('attn-layer-select');
  if (layerSel && !layerSel.options.length && state.structure?.layers?.length) {
    layerSel.innerHTML = '';
    for (const l of state.structure.layers) {
      const opt = document.createElement('option');
      opt.value = String(l.index);
      opt.textContent = `L${String(l.index).padStart(2, '0')}`;
      layerSel.appendChild(opt);
    }
  }
  const layerIdx = Number(layerSel?.value || 0);
  try {
    const data = await apiPost('/inference/attention', {
      model_id: state.modelId,
      checkpoint_id: state.checkpointId,
      prompt,
      system_prompt: $('inf-system')?.value || '',
      layer_index: layerIdx,
      num_heads: 8,
      max_tokens: 10,
    });
    state.attentionData = data;
    const headVal = Number($('attn-head-select')?.value ?? -1);
    $('attn-summary-val').textContent = `L${String(data.layer).padStart(2, '0')} · ${data.num_heads}H · ${data.tokens.length}T`;
    renderer.setData({ attentionData: data, attentionHeadFilter: headVal });
    if (switchMode) renderer.setViewMode('attention');
    renderer.triggerPulse('forward');
    showToast(`Computed ${data.num_heads}-head Attention 3D map for Layer ${data.layer} (${data.latency_ms} ms)`);
  } catch (err) {
    showToast(`Attention 3D error: ${err.message}`, 3200);
  }
}

async function runArenaComparison() {
  if (!state.modelId) return;
  const btn = $('btn-run-arena');
  const prompt = ($('inf-prompt')?.value || '').trim() || 'Customer says: My order #77701 arrived damaged. Can you refund me?';
  const rewardFn = $('eval-reward-select')?.value || 'composite';
  btn.disabled = true;
  btn.textContent = 'Running Multi-Edit Arena…';
  try {
    const res = await apiPost('/arena/compare', {
      model_id: state.modelId,
      checkpoint_id: state.checkpointId,
      prompt,
      system_prompt: $('inf-system')?.value || '',
      reward_fn: rewardFn,
    });
    $('arena-winner-pill').textContent = `★ ${res.winner.slice(0, 18)}`;
    const list = $('arena-leaderboard');
    list.innerHTML = '';
    (res.candidates || []).forEach((c, idx) => {
      const row = document.createElement('div');
      row.className = `arena-row${idx === 0 ? ' winner' : ''}`;
      const d = Number(c.delta_vs_base || 0);
      row.innerHTML = `
        <span style="color:${idx === 0 ? '#6ee7b7' : '#e2e8f0'};overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${c.output_preview || ''}">${idx === 0 ? '★ ' : ''}${c.name}</span>
        <span style="color:#38bdf8;text-align:right">${Number(c.composite || 0).toFixed(3)}</span>
        <span style="color:${d >= 0 ? '#34d399' : '#fb7185'};text-align:right">${d >= 0 ? '+' : ''}${d.toFixed(3)}</span>
        <span style="color:#94a3b8;text-align:right">cos ${(c.hidden_cosine ?? 1).toFixed(2)}</span>
      `;
      list.appendChild(row);
    });
    showToast(`Arena complete: winner is ${res.winner} (${res.latency_ms} ms)`);
  } catch (err) {
    showToast(`Arena error: ${err.message}`, 3200);
  } finally {
    btn.disabled = false;
    btn.textContent = '⚔ Compare Base vs. NF4 vs. Wanda vs. SVD';
  }
}

function bindCommandPalette() {
  const modal = $('cmd-palette-modal');
  const input = $('cmd-palette-input');
  const list = $('cmd-palette-list');
  if (!modal || !input || !list) return;

  const commands = [
    { label: 'View: Architecture 3D', kbd: '1', run: () => setViewMode('stack') },
    { label: 'View: Microscope 128² Heightfield', kbd: '2', run: () => setViewMode('microscope') },
    { label: 'View: SVD Spectral Decomposition', kbd: '3', run: () => setViewMode('spectral') },
    { label: 'View: Trajectory 4D Evolution', kbd: '4', run: () => setViewMode('trajectory') },
    { label: 'View: Causal Circuit 3D Arcs', kbd: '5', run: () => setViewMode('circuit') },
    { label: 'View: Multi-Head Attention 3D', kbd: '6', run: () => setViewMode('attention') },
    { label: 'View: Residual Stream 3D & Logit Lens', kbd: '7', run: () => setViewMode('residual') },
    { label: 'Toggle 3D MRI Cutaway Plane (SLICE)', kbd: 'MRI', run: () => $('btn-slice-toggle')?.click() },
    { label: 'Toggle Cinema Zen Mode', kbd: 'H', run: () => toggleZenMode() },
    { label: 'Run Multi-Edit A/B Arena', kbd: 'ARENA', run: () => { document.querySelector('.wb-tab[data-tab="infer"]')?.click(); runArenaComparison(); } },
    { label: 'Analyze SAE Monosemantic Features', kbd: 'SAE', run: () => { document.querySelector('.wb-tab[data-tab="infer"]')?.click(); analyzeSaeFeatures(); } },
    { label: 'Profile Auto-Sensitivity (Wanda/Fisher)', kbd: 'SENS', run: () => { document.querySelector('.wb-tab[data-tab="analyze"]')?.click(); $('btn-run-sensitivity')?.click(); } },
    { label: 'Auto-Repair Destructive Edits to GO', kbd: 'FIX', run: () => { $('btn-export-modal')?.click(); $('btn-auto-repair')?.click(); } },
    { label: 'Export: Diff-as-LoRA PEFT Adapter', kbd: 'LORA', run: () => { $('btn-export-modal')?.click(); if ($('export-format-select')) { $('export-format-select').value = 'lora'; $('export-format-select').dispatchEvent(new Event('change')); } } },
    { label: 'Export: GGUF v3 + Ollama Modelfile', kbd: 'GGUF', run: () => { $('btn-export-modal')?.click(); if ($('export-format-select')) { $('export-format-select').value = 'gguf'; $('export-format-select').dispatchEvent(new Event('change')); } } },
    { label: 'Export: HF Model Card & Recipe', kbd: 'HF', run: () => { $('btn-export-modal')?.click(); if ($('export-format-select')) { $('export-format-select').value = 'hf_card'; $('export-format-select').dispatchEvent(new Event('change')); } } },
    { label: 'Export: Standalone HTML PR Report', kbd: 'HTML', run: () => $('btn-export-html-report')?.click() },
    { label: 'Lighting Rig: Deep Photonics', kbd: 'RIG', run: () => { $('rig-select').value = 'photonics'; renderer.setRig('photonics'); } },
    { label: 'Lighting Rig: Copper Forge', kbd: 'RIG', run: () => { $('rig-select').value = 'copper'; renderer.setRig('copper'); } },
    { label: 'Lighting Rig: Cleanroom White', kbd: 'RIG', run: () => { $('rig-select').value = 'cleanroom'; renderer.setRig('cleanroom'); } },
  ];

  let selIdx = 0;
  let filtered = commands;

  const renderCmds = () => {
    const q = (input.value || '').trim().toLowerCase();
    filtered = q ? commands.filter((c) => c.label.toLowerCase().includes(q) || c.kbd.toLowerCase().includes(q)) : commands;
    selIdx = Math.min(selIdx, Math.max(0, filtered.length - 1));
    list.innerHTML = '';
    filtered.forEach((cmd, idx) => {
      const btn = document.createElement('button');
      btn.className = `cmd-item${idx === selIdx ? ' selected' : ''}`;
      btn.innerHTML = `<span>${cmd.label}</span><span class="cmd-kbd">${cmd.kbd}</span>`;
      btn.addEventListener('click', () => {
        modal.classList.add('hidden');
        cmd.run();
      });
      list.appendChild(btn);
    });
  };

  const openPalette = () => {
    modal.classList.remove('hidden');
    input.value = '';
    selIdx = 0;
    renderCmds();
    setTimeout(() => input.focus(), 20);
  };

  $('btn-cmd-palette')?.addEventListener('click', openPalette);
  modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.add('hidden'); });
  input.addEventListener('input', () => { selIdx = 0; renderCmds(); });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      modal.classList.add('hidden');
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      selIdx = (selIdx + 1) % Math.max(1, filtered.length);
      renderCmds();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      selIdx = (selIdx - 1 + Math.max(1, filtered.length)) % Math.max(1, filtered.length);
      renderCmds();
    } else if (e.key === 'Enter' && filtered[selIdx]) {
      e.preventDefault();
      modal.classList.add('hidden');
      filtered[selIdx].run();
    }
  });

  window.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      if (modal.classList.contains('hidden')) openPalette();
      else modal.classList.add('hidden');
    }
  });
}

function toggleZenMode() {
  const shell = $('app');
  const active = shell.classList.toggle('zen-mode');
  $('btn-zen-mode').classList.toggle('active', active);
  setTimeout(() => {
    renderer.resize();
    redrawVisibleCharts();
  }, 40);
  showToast(active ? 'Cinema Zen Mode ON (press H to exit)' : 'Cinema Zen Mode OFF', 1800);
}

function bindKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    const tag = (e.target?.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
    if (e.key === '1') setViewMode('stack');
    else if (e.key === '2') setViewMode('microscope');
    else if (e.key === '3') setViewMode('spectral');
    else if (e.key === '4') setViewMode('trajectory');
    else if (e.key === '5') setViewMode('circuit');
    else if (e.key === '6') setViewMode('attention');
    else if (e.key === '7') setViewMode('residual');
    else if (e.key === 'h' || e.key === 'H') toggleZenMode();
    else if (e.key === 's' || e.key === 'S') {
      if (renderer.viewMode === 'microscope') setSplitCompare(!state.splitCompare);
    } else if (e.key === ' ') {
      e.preventDefault();
      renderer.autoOrbit = !renderer.autoOrbit;
      $('btn-orbit-toggle').classList.toggle('active', renderer.autoOrbit);
    }
  });
}

init().catch((err) => {
  console.error('Weight Studio initialization error:', err);
  showToast(`Init error: ${err.message}`, 5000);
});
