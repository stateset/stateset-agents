/**
 * Modularized Command Palette (⌘K), Cinema Zen Mode & Keyboard Shortcuts for 3D Weight Studio.
 */

export function toggleZenMode({ $, renderer, redrawVisibleCharts, showToast }) {
  const shell = $('app');
  const active = shell.classList.toggle('zen-mode');
  $('btn-zen-mode')?.classList.toggle('active', active);
  setTimeout(() => {
    renderer.resize();
    redrawVisibleCharts();
  }, 40);
  showToast(active ? 'Cinema Zen Mode ON (press H to exit)' : 'Cinema Zen Mode OFF', 1800);
}

export function initCommandPalette({
  $,
  state,
  renderer,
  setViewMode,
  setSplitCompare,
  redrawVisibleCharts,
  runArenaComparison,
  analyzeSaeFeatures,
  showToast,
}) {
  const modal = $('cmd-palette-modal');
  const input = $('cmd-palette-input');
  const list = $('cmd-palette-list');
  if (!modal || !input || !list) return;

  const zen = () => toggleZenMode({ $, renderer, redrawVisibleCharts, showToast });

  const commands = [
    { label: 'View: Transformer Stack 3D', kbd: '1', run: () => setViewMode('stack') },
    { label: 'View: Tensor Microscope 128²', kbd: '2', run: () => setViewMode('microscope') },
    { label: 'View: SVD Spectral Canyon', kbd: '3', run: () => setViewMode('spectral') },
    { label: 'View: Training Trajectory 3D', kbd: '4', run: () => setViewMode('trajectory') },
    { label: 'View: Causal Circuit 3D', kbd: '5', run: () => setViewMode('circuit') },
    { label: 'View: Multi-Head Attention 3D', kbd: '6', run: () => setViewMode('attention') },
    { label: 'View: Residual Stream 3D & Logit Lens', kbd: '7', run: () => setViewMode('residual') },
    { label: 'Toggle 3D MRI Cutaway Plane (SLICE)', kbd: 'MRI', run: () => $('btn-slice-toggle')?.click() },
    { label: 'Toggle Cinema Zen Mode', kbd: 'H', run: zen },
    { label: 'Run Multi-Edit A/B Arena', kbd: 'ARENA', run: () => { document.querySelector('.wb-tab[data-tab="infer"]')?.click(); runArenaComparison(); } },
    { label: 'Autonomous Self-Heal (Logs → CAA Surgery)', kbd: 'HEAL', run: () => { document.querySelector('.wb-tab[data-tab="edit"]')?.click(); $('btn-run-auto-heal')?.click(); } },
    { label: 'Red-Team Safety Audit (10 Attack Probes)', kbd: 'RED', run: () => { document.querySelector('.wb-tab[data-tab="edit"]')?.click(); $('btn-run-redteam-audit')?.click(); } },
    { label: 'Red-Team Surgical Immunization', kbd: 'SHIELD', run: () => { document.querySelector('.wb-tab[data-tab="edit"]')?.click(); $('btn-run-redteam-immunize')?.click(); } },
    { label: 'Multi-Model Pareto Tournament & Merge', kbd: 'MERGE', run: () => { document.querySelector('.wb-tab[data-tab="edit"]')?.click(); $('btn-run-tournament')?.click(); } },
    { label: 'Canary Shadow-Rollout Promotion Gate', kbd: 'CANARY', run: () => { document.querySelector('.wb-tab[data-tab="edit"]')?.click(); $('btn-run-canary')?.click(); } },
    { label: 'Full Production Autopilot Release', kbd: 'AUTO', run: () => { document.querySelector('.wb-tab[data-tab="edit"]')?.click(); $('btn-run-autopilot')?.click(); } },
    { label: 'Audit Reward Calibration & Reward-Hacking', kbd: 'CAL', run: () => { document.querySelector('.wb-tab[data-tab="edit"]')?.click(); $('btn-audit-calibration')?.click(); } },
    { label: 'Heal Model from Live Production Traces', kbd: 'TRACE', run: () => { document.querySelector('.wb-tab[data-tab="edit"]')?.click(); $('btn-heal-live-traces')?.click(); } },
    { label: 'Run Live vLLM / Ollama Shadow A/B', kbd: 'LIVE', run: () => { document.querySelector('.wb-tab[data-tab="edit"]')?.click(); $('btn-run-live-ab')?.click(); } },
    { label: 'Connect AI Agent (Live HTTP / Stdio MCP)', kbd: 'MCP', run: () => $('btn-mcp-modal')?.click() },
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
    else if (e.key === 'h' || e.key === 'H') zen();
    else if (e.key === 's' || e.key === 'S') {
      if (renderer.viewMode === 'microscope') setSplitCompare(!state.splitCompare);
    } else if (e.key === ' ') {
      e.preventDefault();
      renderer.autoOrbit = !renderer.autoOrbit;
      $('btn-orbit-toggle')?.classList.toggle('active', renderer.autoOrbit);
    }
  });
}
