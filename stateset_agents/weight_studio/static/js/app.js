/**
 * StateSet 3D Weight Studio — Interactive Application Controller.
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
  structure: null,
  selectedTensorName: null,
  tensorDetail: null,
  trajectory: null,
  editHistory: null,
  activeOp: 'scale',
  brushRegion: { row_start: 0, row_end: 8, col_start: 0, col_end: 8 },
  ftCatalog: null,
  ftAlgorithm: 'gspo',
  ftSelectedTargets: new Set(['q_proj', 'k_proj', 'v_proj', 'o_proj']),
  ftJobId: null,
  ftWs: null,
  ftPollTimer: null,
  ftHistory: [],
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
      if (openMicroscope) {
        setViewMode('microscope');
      }
    },
    onHover: (hit) => renderHoverTooltip(hit),
    onProjection: (pt, hit) => render3DCallout(pt, hit),
    onBrushRegion: (region) => {
      state.brushRegion = region;
      if (state.activeOp === 'region_edit') {
        renderOpParams();
      }
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

  await Promise.all([loadFinetuneCatalog(), loadInferenceScenarios(), loadModels(false)]);
}

// ---------------------------------------------------------------------------
// Model & Tensor Loading
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
    const preferred = state.modelId && state.models.some((m) => m.model_id === state.modelId)
      ? state.modelId
      : state.models[0].model_id;
    select.value = preferred;
    await loadModelStructure(preferred, null);
  }
}

async function loadModelStructure(modelId, checkpointId = null) {
  state.modelId = modelId;
  state.checkpointId = checkpointId;
  state.trajectory = null;

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

  // Sync LoRA target chips to this model's actual modules
  initLoraTargetChipsForModel(struct);

  // Checkpoint timeline visibility
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

  // Select first interesting 2D tensor if current selection isn't in this model
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

async function selectTensor(tensorName, { focusCamera = true } = {}) {
  if (!tensorName || !state.modelId) return;
  state.selectedTensorName = tensorName;

  // Highlight in left tree
  document.querySelectorAll('.tensor-row').forEach((el) => {
    el.classList.toggle('selected', el.dataset.tensor === tensorName);
  });

  const res = Number($('microscope-res').value || 64);
  const qs = new URLSearchParams({
    model_id: state.modelId,
    tensor_name: tensorName,
    resolution: String(res),
    include_svd: 'true',
  });
  if (state.checkpointId) qs.set('checkpoint_id', state.checkpointId);

  const detail = await apiGet(`/models/tensor?${qs.toString()}`);
  state.tensorDetail = detail;

  // Update HUD & left telemetry dock
  $('hud-selected-label').textContent =
    detail.layer !== null ? `L${String(detail.layer).padStart(2, '0')} · ${detail.label}` : detail.label;
  $('hud-delta-pill').classList.toggle('hidden', !detail.edited);
  $('selected-tensor-shape').textContent = `[${detail.shape.join(' × ')}]`;
  $('edit-target-preview').textContent = `${tensorName} [${detail.shape.join('×')}]`;

  const s = detail.stats;
  $('st-frob').textContent = s.frobenius_norm.toFixed(3);
  $('st-rms').textContent = s.rms.toFixed(4);
  $('st-std').textContent = s.std.toFixed(4);
  $('st-sparsity').textContent = `${(s.sparsity * 100).toFixed(1)}%`;
  $('st-minmax').textContent = `${s.min.toFixed(2)} / ${s.max.toFixed(2)}`;
  $('st-kurt').textContent = s.kurtosis.toFixed(2);

  // Sync SVD controls
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

  // Outliers list
  const outEl = $('outlier-list');
  outEl.innerHTML = '';
  for (const o of detail.grid.outliers || []) {
    const row = document.createElement('div');
    row.className = 'outlier-row';
    row.innerHTML = `<span>[${o.row}, ${o.col}]</span><span style="color:${o.value >= 0 ? '#fbbf24' : '#38bdf8'}">${o.value >= 0 ? '+' : ''}${o.value.toFixed(5)}</span>`;
    outEl.appendChild(row);
  }

  renderer.setData({ tensorDetail: detail, selectedTensorName: tensorName });
  if (focusCamera) renderer.focusOnTensor(tensorName);
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
      row.addEventListener('click', () => selectTensor(t.name, { focusCamera: true }));
      row.addEventListener('dblclick', async () => {
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
// Topbar, Camera, Mode Switching & Overlays
// ---------------------------------------------------------------------------

function setViewMode(mode) {
  document.querySelectorAll('.mode-pill').forEach((btn) => {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
  });
  $('microscope-hud').classList.toggle('hidden', mode !== 'microscope');
  $('spectral-hud').classList.toggle('hidden', mode !== 'spectral');

  if (mode === 'trajectory' && !state.trajectory) {
    loadTrajectory().then(() => renderer.setViewMode('trajectory'));
  } else {
    renderer.setViewMode(mode);
  }
  if (mode === 'spectral') {
    redrawVisibleCharts();
  }
}

function updateLegendGradient(colorMode) {
  const bar = $('legend-gradient');
  const lo = $('legend-lo');
  const hi = $('legend-hi');
  if (colorMode === 'magnitude') {
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

  $('colormap-select').addEventListener('change', (e) => {
    renderer.colorMode = e.target.value;
    updateLegendGradient(e.target.value);
    renderer.rebuildScene();
  });

  $('rig-select').addEventListener('change', (e) => {
    renderer.setRig(e.target.value);
  });

  $('bloom-slider').addEventListener('input', (e) => {
    renderer.bloomOverride = Number(e.target.value);
  });

  $('spread-slider').addEventListener('input', (e) => {
    renderer.layerSpread = Number(e.target.value);
    renderer.rebuildScene();
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

  $('chk-ghost-overlay').addEventListener('change', (e) => {
    renderer.showGhostOverlay = e.target.checked;
    renderer.rebuildScene();
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
// Workbench Tabs & Surgical Weight Editing
// ---------------------------------------------------------------------------

function bindWorkbenchTabs() {
  document.querySelectorAll('.wb-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      const key = tab.dataset.tab;
      document.querySelectorAll('.wb-tab').forEach((t) => t.classList.toggle('active', t === tab));
      document.querySelectorAll('.wb-pane').forEach((p) => p.classList.toggle('active', p.id === `tab-${key}`));
      // Highlight LoRA targets in 3D when user enters Finetune tab, clear when leaving
      if (key === 'finetune') {
        renderer.setLoraTargets([...state.ftSelectedTargets]);
      } else {
        renderer.setLoraTargets([]);
      }
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
    $('p-scale').addEventListener('input', (e) => {
      $('p-scale-val').textContent = `${Number(e.target.value).toFixed(2)}×`;
    });
  } else if (op === 'prune') {
    c.innerHTML = `
      <label class="field">
        <span>Target Sparsity <b id="p-prune-val" class="mono-accent">35%</b></span>
        <input id="p-prune" type="range" min="0.05" max="0.95" step="0.05" value="0.35" />
      </label>
    `;
    $('p-prune').addEventListener('input', (e) => {
      $('p-prune-val').textContent = `${Math.round(Number(e.target.value) * 100)}%`;
    });
  } else if (op === 'clamp') {
    c.innerHTML = `
      <label class="field">
        <span>Percentile Clamp Ceiling <b id="p-clamp-val" class="mono-accent">P98.0</b></span>
        <input id="p-clamp" type="range" min="80" max="99.9" step="0.5" value="98" />
      </label>
    `;
    $('p-clamp').addEventListener('input', (e) => {
      $('p-clamp-val').textContent = `P${Number(e.target.value).toFixed(1)}`;
    });
  } else if (op === 'svd_truncate') {
    const maxR = state.tensorDetail?.svd?.rank || 16;
    const defR = Math.max(1, Math.floor(maxR * 0.5));
    c.innerHTML = `
      <label class="field">
        <span>Keep Top-k Singular Modes <b id="p-svd-val" class="mono-accent">k = ${defR} / ${maxR}</b></span>
        <input id="p-svd" type="range" min="1" max="${maxR}" step="1" value="${defR}" />
      </label>
    `;
    $('p-svd').addEventListener('input', (e) => {
      $('p-svd-val').textContent = `k = ${e.target.value} / ${maxR}`;
    });
  } else if (op === 'noise') {
    c.innerHTML = `
      <div class="form-grid-2">
        <label class="field">
          <span>Gaussian σ</span>
          <input id="p-sigma" type="number" step="0.002" min="0.0005" max="0.2" value="0.01" class="precision-input mono" />
        </label>
        <label class="field">
          <span>Seed</span>
          <input id="p-seed" type="number" value="42" class="precision-input mono" />
        </label>
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
        <label class="field">
          <span>Value</span>
          <input id="p-reg-val" type="number" step="0.1" value="0.25" class="precision-input mono" />
        </label>
      </div>
    `;
  } else if (op === 'blend') {
    const options = state.models
      .map((m) => `<option value="${m.model_id}">${m.display_name}</option>`)
      .join('');
    c.innerHTML = `
      <label class="field">
        <span>Secondary Checkpoint</span>
        <select id="p-blend-model" class="precision-select">${options}</select>
      </label>
      <label class="field">
        <span>Interpolation λ <b id="p-blend-val" class="mono-accent">0.50</b></span>
        <input id="p-blend-alpha" type="range" min="0" max="1" step="0.05" value="0.5" />
      </label>
    `;
    $('p-blend-alpha').addEventListener('input', (e) => {
      $('p-blend-val').textContent = Number(e.target.value).toFixed(2);
    });
  }
}

async function handleApplyEditClick() {
  if (!state.modelId) return;
  const scope = $('edit-scope-select').value;
  let target = state.selectedTensorName;
  if (scope === 'layer') {
    const lIdx = state.tensorDetail?.layer ?? 0;
    target = `layer:${lIdx}`;
  } else if (scope !== 'selected') {
    target = scope;
  }
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
    params = { mode: 'percentile', sparsity };
    label = `Prune ${Math.round(sparsity * 100)}%`;
  } else if (op === 'clamp') {
    const percentile = Number($('p-clamp').value);
    params = { mode: 'percentile', percentile };
    label = `Clamp P${percentile.toFixed(1)}`;
  } else if (op === 'svd_truncate') {
    const rank = Number($('p-svd').value);
    params = { rank };
    label = `SVD Rank-k=${rank}`;
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
  } else if (op === 'blend') {
    params = {
      secondary_model_id: $('p-blend-model').value,
      alpha: Number($('p-blend-alpha').value),
    };
    label = `Blend λ=${params.alpha.toFixed(2)}`;
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
    await reloadAfterEdit(`Applied ${res.edit.label} (${res.edit.changed_weights.toLocaleString()} weights Δ=${res.edit.frobenius_delta})`);
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
    list.innerHTML = '<div class="empty-hint">No surgical edits yet. Scale, prune, clamp, or paint a 3D brush patch above.</div>';
    return;
  }
  list.innerHTML = '';
  for (const e of [...edits].reverse()) {
    const item = document.createElement('div');
    item.className = `edit-item${e.enabled ? '' : ' disabled'}`;
    item.innerHTML = `
      <div>
        <div class="edit-item-title">${e.label}</div>
        <div class="edit-item-sub">${e.affected_tensors.length}T · ${e.changed_weights.toLocaleString()} w · Δ‖W‖=${e.frobenius_delta}</div>
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
// Finetuning Workbench & Live PyTorch Micro-Finetuning
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
      `<span class="vram-item" style="color:#fbbf24">Trainable ${est.trainable_pct.toFixed(2)}% (${formatParams(est.trainable_params)})</span>`;

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
  if (state.ftWs) {
    state.ftWs.close();
    state.ftWs = null;
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

  // Connect SSE EventSource for live step metrics & 3D backprop pulses (works natively over HTTP/1.1!)
  const es = new EventSource(`${API}/finetune/stream/${job.job_id}`);
  state.ftWs = es;

  let last3DRefresh = 0;
  es.onmessage = async (evt) => {
    const msg = JSON.parse(evt.data);
    if (msg.type === 'step' && msg.metric) {
      const m = msg.metric;
      state.ftHistory.push(m);
      $('ft-step-counter').textContent = `${m.step} / ${m.total_steps}`;
      drawTrainingTelemetryChart($('chart-training'), state.ftHistory);

      // Map per-layer gradient norms directly to glowing 3D backprop heat!
      const heatMap = {};
      const grads = Object.entries(m.layer_grad_norms || {});
      const maxG = Math.max(1e-5, ...grads.map(([, v]) => v));
      for (const [lName, gVal] of grads) {
        const match = lName.match(/(\d+)/);
        if (match) heatMap[Number(match[1])] = gVal / maxG;
      }
      renderer.triggerPulse('backprop', heatMap);

      // Refresh actual 3D crystal heights every 4 steps so user sees real weights morph!
      const now = performance.now();
      if (now - last3DRefresh > 1100 || m.step === m.total_steps) {
        last3DRefresh = now;
        await reloadAfterEdit(null);
      }
    } else if (msg.type === 'completed') {
      await reloadAfterEdit('Live PyTorch finetuning completed — 3D weights updated!');
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
  if (state.ftWs) {
    state.ftWs.close();
    state.ftWs = null;
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
// Interactive Model Inference & Logit Lens Playground
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

function bindInferenceControls() {
  $('inf-temp').addEventListener('input', (e) => {
    $('inf-temp-val').textContent = Number(e.target.value).toFixed(2);
  });
  $('inf-topp').addEventListener('input', (e) => {
    $('inf-topp-val').textContent = Number(e.target.value).toFixed(2);
  });
  $('btn-run-infer').addEventListener('click', runModelInference);
}

async function runModelInference() {
  if (!state.modelId) return;
  const prompt = ($('inf-prompt').value || '').trim();
  if (!prompt) return;

  const res = await apiPost('/inference/run', {
    model_id: state.modelId,
    checkpoint_id: state.checkpointId,
    prompt,
    system_prompt: $('inf-system').value || '',
    temperature: Number($('inf-temp').value || 0.7),
    top_p: Number($('inf-topp').value || 0.9),
    max_tokens: Number($('inf-maxtok').value || 48),
    compare_base: $('inf-compare-base').checked,
  });
  state.lastInferResult = res;

  // Fire luminous forward-pass activation wave through the 3D architecture!
  const actMap = {};
  for (const la of res.active.layer_activations || []) {
    actMap[la.layer] = la.intensity;
  }
  renderer.triggerPulse('forward', actMap);

  $('inf-latency').textContent = `${res.latency_ms} ms · H=${res.active.mean_entropy}`;

  // Interactive token probability chips
  const stream = $('inf-token-stream');
  stream.innerHTML = '';
  res.active.tokens.forEach((tok, idx) => {
    const span = document.createElement('span');
    span.className = `tok-chip${idx === 0 ? ' selected' : ''}`;
    // Color-code by confidence (high prob emerald/cyan, high KL shift amber/rose)
    const klGlow = Math.min(1, tok.kl_vs_base * 6);
    const bg = klGlow > 0.25
      ? `rgba(251, 191, 36, ${0.18 + klGlow * 0.35})`
      : `rgba(56, 189, 248, ${0.10 + tok.prob * 0.28})`;
    span.style.background = bg;
    span.textContent = tok.token;
    span.title = `p=${(tok.prob * 100).toFixed(1)}% · H=${tok.entropy} · KL=${tok.kl_vs_base}`;
    span.addEventListener('click', () => {
      stream.querySelectorAll('.tok-chip').forEach((c) => c.classList.remove('selected'));
      span.classList.add('selected');
      renderTokenLogitLens(tok);
    });
    stream.appendChild(span);
  });

  if (res.active.tokens.length) {
    renderTokenLogitLens(res.active.tokens[0]);
  }

  // Base vs Edited A/B diff section
  const ab = $('inf-ab-compare');
  if (res.has_edits && $('inf-compare-base').checked) {
    ab.classList.remove('hidden');
    $('inf-base-text').textContent = res.base.text;
    const rd = res.comparison.reward_delta;
    $('inf-shift-metrics').textContent =
      `cos=${res.comparison.hidden_cosine_similarity.toFixed(4)} · ΔReward=${rd >= 0 ? '+' : ''}${rd.toFixed(3)}`;
  } else {
    ab.classList.add('hidden');
  }

  // StateSet Multi-Axis Reward Rubric Bars
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
// Verified .safetensors Export Modal
// ---------------------------------------------------------------------------

function bindExportModal() {
  const modal = $('export-modal');
  $('btn-export-modal').addEventListener('click', () => {
    $('export-result-box').classList.add('hidden');
    modal.classList.remove('hidden');
  });
  $('btn-close-export').addEventListener('click', () => modal.classList.add('hidden'));
  modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.classList.add('hidden');
  });

  $('btn-confirm-export').addEventListener('click', async () => {
    if (!state.modelId) return;
    const name = ($('export-name-input').value || '').trim() || `edited-${Date.now()}`;
    try {
      const manifest = await apiPost('/edits/export', {
        model_id: state.modelId,
        checkpoint_id: state.checkpointId,
        export_name: name,
      });
      const box = $('export-result-box');
      box.innerHTML = `
        <b>✓ Exported &amp; Reload-Verified</b><br/>
        File: ${manifest.export_file}<br/>
        SHA-256: ${manifest.sha256.slice(0, 24)}…<br/>
        Tensors: ${manifest.tensor_count} (${manifest.modified_tensors.length} modified)
      `;
      box.classList.remove('hidden');
      showToast('Verified .safetensors checkpoint exported!');
      await loadModels(true);
    } catch (err) {
      showToast(`Export error: ${err.message}`, 3500);
    }
  });
}

init().catch((err) => {
  console.error('Weight Studio initialization error:', err);
  showToast(`Init error: ${err.message}`, 5000);
});
