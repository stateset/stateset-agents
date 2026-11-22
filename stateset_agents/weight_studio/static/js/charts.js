/**
 * Crisp HiDPI 2D Canvas scientific telemetry plots for the Weight Studio.
 */

function setupCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(120, Math.round(rect.width || canvas.width || 280));
  const h = Math.max(60, Math.round(rect.height || canvas.height || 110));
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

export function drawHistogramChart(canvas, histogram, baseHistogram = null) {
  if (!canvas || !histogram || !histogram.counts || !histogram.counts.length) return;
  const { ctx, w, h } = setupCanvas(canvas);
  const counts = histogram.counts;
  const edges = histogram.edges || [];
  const maxC = Math.max(1, ...counts, ...(baseHistogram?.counts || [1]));

  const padL = 6, padR = 6, padT = 10, padB = 18;
  const pw = w - padL - padR;
  const ph = h - padT - padB;
  const bw = pw / counts.length;

  // Baseline axis
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT + ph);
  ctx.lineTo(padL + pw, padT + ph);
  ctx.stroke();

  // Zero line if edges cross zero
  if (edges.length >= 2 && edges[0] < 0 && edges[edges.length - 1] > 0) {
    const zFrac = (0 - edges[0]) / (edges[edges.length - 1] - edges[0]);
    const zx = padL + zFrac * pw;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(zx, padT);
    ctx.lineTo(zx, padT + ph);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Optional base ghost outline when edited
  if (baseHistogram && baseHistogram.counts && baseHistogram.counts.length === counts.length) {
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.42)';
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    for (let i = 0; i < counts.length; i++) {
      const x = padL + (i + 0.5) * bw;
      const y = padT + ph - (baseHistogram.counts[i] / maxC) * ph;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  for (let i = 0; i < counts.length; i++) {
    const bh = (counts[i] / maxC) * ph;
    const x = padL + i * bw;
    const y = padT + ph - bh;
    const midVal = edges.length > i + 1 ? (edges[i] + edges[i + 1]) * 0.5 : (i / counts.length - 0.5);
    const grad = ctx.createLinearGradient(0, y, 0, padT + ph);
    if (midVal < 0) {
      grad.addColorStop(0, 'rgba(56, 189, 248, 0.92)');
      grad.addColorStop(1, 'rgba(29, 78, 216, 0.35)');
    } else {
      grad.addColorStop(0, 'rgba(251, 191, 36, 0.94)');
      grad.addColorStop(1, 'rgba(217, 119, 6, 0.35)');
    }
    ctx.fillStyle = grad;
    ctx.fillRect(x + 0.5, y, Math.max(1, bw - 1), bh);
  }

  // Min / 0 / Max labels
  ctx.fillStyle = 'rgba(148, 163, 184, 0.72)';
  ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
  if (edges.length >= 2) {
    ctx.textAlign = 'left';
    ctx.fillText(edges[0].toFixed(3), padL, h - 4);
    ctx.textAlign = 'center';
    ctx.fillText('0.0', padL + pw * 0.5, h - 4);
    ctx.textAlign = 'right';
    ctx.fillText(edges[edges.length - 1].toFixed(3), padL + pw, h - 4);
  }
}

export function drawSvdChart(canvas, svd, cutoffRank = null) {
  if (!canvas || !svd || !svd.singular_values || !svd.singular_values.length) return;
  const { ctx, w, h } = setupCanvas(canvas);
  const sv = svd.singular_values;
  const cum = svd.cumulative_energy || [];
  const maxS = Math.max(1e-6, sv[0]);
  const k = cutoffRank ? Math.min(cutoffRank, sv.length) : sv.length;

  const padL = 8, padR = 8, padT = 14, padB = 18;
  const pw = w - padL - padR;
  const ph = h - padT - padB;
  const bw = pw / sv.length;

  // Grid lines
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.12)';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 2; g++) {
    const y = padT + (g / 2) * ph;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + pw, y);
    ctx.stroke();
  }

  // Bars
  for (let i = 0; i < sv.length; i++) {
    const kept = i < k;
    const bh = (sv[i] / maxS) * ph;
    const x = padL + i * bw;
    const y = padT + ph - bh;
    ctx.fillStyle = kept ? 'rgba(56, 189, 248, 0.85)' : 'rgba(100, 116, 139, 0.28)';
    ctx.fillRect(x + 1, y, Math.max(2, bw - 2), bh);
  }

  // Cumulative energy curve
  if (cum.length === sv.length) {
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < cum.length; i++) {
      const x = padL + (i + 0.5) * bw;
      const y = padT + ph - cum[i] * ph;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Rank cutoff vertical laser
  if (cutoffRank && cutoffRank < sv.length) {
    const cx = padL + cutoffRank * bw;
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 2]);
    ctx.beginPath();
    ctx.moveTo(cx, padT);
    ctx.lineTo(cx, padT + ph);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.fillStyle = 'rgba(148, 163, 184, 0.78)';
  ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
  ctx.textAlign = 'left';
  ctx.fillText(`σ₁=${sv[0].toFixed(3)}`, padL, 10);
  ctx.textAlign = 'right';
  const energyAtK = cum[Math.max(0, k - 1)] ?? 1.0;
  ctx.fillStyle = '#fbbf24';
  ctx.fillText(`E(k=${k})=${(energyAtK * 100).toFixed(1)}%`, padL + pw, 10);
  ctx.fillStyle = 'rgba(148, 163, 184, 0.65)';
  ctx.textAlign = 'left';
  ctx.fillText('1', padL, h - 4);
  ctx.textAlign = 'right';
  ctx.fillText(`r=${sv.length}`, padL + pw, h - 4);
}

export function drawLayerProfileChart(canvas, layers, selectedLayerIdx = null) {
  if (!canvas || !layers || !layers.length) return;
  const { ctx, w, h } = setupCanvas(canvas);
  const maxNorm = Math.max(1e-6, ...layers.map((l) => l.frobenius_norm));
  const padL = 8, padR = 8, padT = 12, padB = 18;
  const pw = w - padL - padR;
  const ph = h - padT - padB;
  const bw = pw / layers.length;

  for (let i = 0; i < layers.length; i++) {
    const l = layers[i];
    const bh = (l.frobenius_norm / maxNorm) * ph;
    const x = padL + i * bw;
    const y = padT + ph - bh;
    const isSel = l.index === selectedLayerIdx;
    if (l.has_edits) {
      ctx.fillStyle = isSel ? '#34d399' : 'rgba(16, 185, 129, 0.78)';
    } else if (l.layer_type === 'hybrid_linear_attn') {
      ctx.fillStyle = isSel ? '#38bdf8' : 'rgba(56, 189, 248, 0.65)';
    } else {
      ctx.fillStyle = isSel ? '#fbbf24' : 'rgba(245, 158, 11, 0.62)';
    }
    ctx.fillRect(x + 1, y, Math.max(2, bw - 2), bh);
    if (isSel) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 1, y, Math.max(2, bw - 2), bh);
    }
  }

  ctx.fillStyle = 'rgba(148, 163, 184, 0.72)';
  ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
  ctx.textAlign = 'left';
  ctx.fillText('L0', padL, h - 4);
  ctx.textAlign = 'right';
  ctx.fillText(`L${layers[layers.length - 1].index}`, padL + pw, h - 4);
}

export function drawTrainingTelemetryChart(canvas, history) {
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  const padL = 8, padR = 8, padT = 16, padB = 18;
  const pw = w - padL - padR;
  const ph = h - padT - padB;

  // Grid
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.12)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = padT + (i / 3) * ph;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + pw, y);
    ctx.stroke();
  }

  if (!history || history.length === 0) {
    ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
    ctx.font = '11px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Start Live Finetune to stream PyTorch Loss & Reward', w * 0.5, h * 0.54);
    return;
  }

  const losses = history.map((m) => m.loss);
  const rewards = history.map((m) => m.reward);
  const minL = Math.min(...losses);
  const maxL = Math.max(minL + 1e-4, ...losses);
  const minR = Math.min(0, ...rewards);
  const maxR = Math.max(1.0, ...rewards);

  const plotLine = (vals, lo, hi, color, fillAlpha = 0) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < vals.length; i++) {
      const x = padL + (vals.length === 1 ? 0.5 : i / (vals.length - 1)) * pw;
      const y = padT + ph - ((vals[i] - lo) / Math.max(hi - lo, 1e-5)) * ph;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    if (fillAlpha > 0 && vals.length > 1) {
      ctx.lineTo(padL + pw, padT + ph);
      ctx.lineTo(padL, padT + ph);
      ctx.closePath();
      ctx.fillStyle = color.replace('1)', `${fillAlpha})`);
      ctx.fill();
    }
  };

  plotLine(losses, minL * 0.9, maxL * 1.05, 'rgba(248, 113, 113, 1)', 0.10);
  plotLine(rewards, minR, maxR, 'rgba(52, 211, 153, 1)', 0.12);

  const last = history[history.length - 1];
  ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
  ctx.textAlign = 'left';
  ctx.fillStyle = '#f87171';
  ctx.fillText(`LOSS ${last.loss.toFixed(4)}`, padL, 11);
  ctx.textAlign = 'right';
  ctx.fillStyle = '#34d399';
  ctx.fillText(`REWARD ${last.reward.toFixed(3)}`, padL + pw, 11);
  ctx.fillStyle = 'rgba(148, 163, 184, 0.7)';
  ctx.textAlign = 'left';
  ctx.fillText(`Step ${last.step}/${last.total_steps}`, padL, h - 4);
  ctx.textAlign = 'right';
  ctx.fillText(`KL ${last.kl.toFixed(5)} · ‖g‖ ${last.grad_norm.toFixed(3)}`, padL + pw, h - 4);
}
