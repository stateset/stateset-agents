"""Standalone Single-File Interactive HTML PR Report Generator for StateSet 3D Weight Studio.

Produces a self-contained, offline-capable HTML document embedding:
- Interactive 3D isometric canvas orbit/heightfield visualization of the active tensor or layer profile
- Pre-Export Batch Eval Gate (GO / WARN / BLOCK) scorecard
- Auto-Sensitivity layer breakdown & SVD spectrum
- Active surgical edit lineage & copyable .studio-recipe.json
"""

from __future__ import annotations

import html
import json
import time
from typing import Any


def build_standalone_html_report(
    *,
    model_summary: dict[str, Any],
    structure_summary: dict[str, Any],
    layers: list[dict[str, Any]],
    edit_history: dict[str, Any],
    recipe: dict[str, Any],
    eval_report: dict[str, Any] | None = None,
    sensitivity_report: dict[str, Any] | None = None,
    tensor_detail: dict[str, Any] | None = None,
) -> str:
    """Render a zero-dependency single-file interactive HTML PR report."""
    model_name = html.escape(
        str(
            model_summary.get("display_name")
            or model_summary.get("model_id")
            or "Model"
        )
    )
    base_model = html.escape(str(model_summary.get("base_model") or "—"))
    arch = html.escape(str(model_summary.get("architecture") or "—"))
    verdict = (eval_report or {}).get("verdict", "UNRUN")
    verdict_class = verdict if verdict in {"GO", "WARN", "BLOCK"} else "WARN"
    generated_at = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    # Pack compact visual payload for the embedded interactive 3D heightfield canvas
    grid_vals: list[float] = []
    grid_rows = 0
    grid_cols = 0
    grid_label = "Layer Frobenius Profile"
    if tensor_detail and tensor_detail.get("grid"):
        g = tensor_detail["grid"]
        grid_rows = int(g.get("rows", 0))
        grid_cols = int(g.get("cols", 0))
        # Cap embedded grid to 32x32 for instant standalone HTML load
        raw_vals = g.get("values") or []
        if grid_rows > 32 or grid_cols > 32:
            step_r = max(1, grid_rows // 32)
            step_c = max(1, grid_cols // 32)
            sampled: list[float] = []
            out_r = 0
            for r in range(0, grid_rows, step_r):
                out_r += 1
                out_c = 0
                for c in range(0, grid_cols, step_c):
                    out_c += 1
                    sampled.append(float(raw_vals[r * grid_cols + c]))
            grid_rows, grid_cols, grid_vals = out_r, out_c, sampled
        else:
            grid_vals = [float(v) for v in raw_vals]
        grid_label = str(tensor_detail.get("tensor_name") or "Selected Tensor")

    embedded_data = {
        "grid": {
            "label": grid_label,
            "rows": grid_rows,
            "cols": grid_cols,
            "values": [round(v, 5) for v in grid_vals],
        },
        "layers": [
            {
                "index": layer.get("index"),
                "name": layer.get("name"),
                "frobenius_norm": layer.get("frobenius_norm", 0.0),
                "sparsity": layer.get("mean_sparsity", 0.0),
                "has_edits": bool(layer.get("has_edits")),
            }
            for layer in layers
        ],
        "recipe": recipe,
    }
    safe_json = json.dumps(embedded_data).replace("</", "<\\/")
    recipe_pretty = html.escape(json.dumps(recipe, indent=2))

    # Build Eval Gate HTML rows
    eval_rows_html = ""
    if eval_report and eval_report.get("cases"):
        for case in eval_report["cases"]:
            st = html.escape(str(case.get("status", "PASS")))
            delta = float(case.get("delta", 0.0))
            d_color = "#34d399" if delta >= 0 else "#fb7185"
            eval_rows_html += (
                f"<tr><td>{html.escape(str(case.get('category', '')))}</td>"
                f"<td>{html.escape(str(case.get('prompt_preview', '')))}</td>"
                f"<td>{float(case.get('base_composite', 0.0)):.3f}</td>"
                f"<td>{float(case.get('edited_composite', 0.0)):.3f}</td>"
                f"<td style='color:{d_color};font-weight:700'>{delta:+.3f}</td>"
                f"<td><span class='pill {st}'>{st}</span></td></tr>"
            )
    else:
        eval_rows_html = "<tr><td colspan='6' style='color:#64748b'>Batch Eval Gate was not run before report export.</td></tr>"

    # Build Edit Lineage rows
    edits = [e for e in (edit_history.get("edits") or []) if e.get("enabled", True)]
    if edits:
        edit_rows_html = "".join(
            f"<tr><td>{html.escape(str(e.get('label', '')))}</td>"
            f"<td><code>{html.escape(str(e.get('op', '')))}</code></td>"
            f"<td>{len(e.get('affected_tensors') or [])}</td>"
            f"<td>{int(e.get('changed_weights', 0)):,}</td>"
            f"<td>{e.get('frobenius_delta', 0)}</td></tr>"
            for e in edits
        )
    else:
        edit_rows_html = "<tr><td colspan='5' style='color:#64748b'>Base checkpoint (0 active surgical edits).</td></tr>"

    # Build optional Sensitivity summary
    sens_html = ""
    if sensitivity_report and sensitivity_report.get("layers"):
        sens_rows = "".join(
            f"<tr><td>{html.escape(str(r.get('name', '')))}</td>"
            f"<td>{float(r.get('sensitivity', 0.0)):.3f}</td>"
            f"<td>{float(r.get('effective_rank', 0.0)):.1f}</td>"
            f"<td>r={int(r.get('recommended_rank', 8))}</td>"
            f"<td>{int(float(r.get('recommended_sparsity', 0.0)) * 100)}%</td></tr>"
            for r in sensitivity_report["layers"][:16]
        )
        sens_html = f"""
        <section class="card">
          <h2>Auto-Sensitivity Profile &amp; Adaptive Allocation</h2>
          <p class="sub">Estimated parameter reduction: <b>{sensitivity_report.get('estimated_compression_pct', 0)}%</b> across {len(sensitivity_report['layers'])} profiled layers.</p>
          <table>
            <thead><tr><th>Layer</th><th>Sensitivity</th><th>Eff. Rank</th><th>Rec. LoRA Rank</th><th>Rec. Sparsity</th></tr></thead>
            <tbody>{sens_rows}</tbody>
          </table>
        </section>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>StateSet 3D Weight Studio Report — {model_name}</title>
  <style>
    :root {{ --bg:#060a12; --card:#0e1624; --border:rgba(148,163,184,0.18); --text:#eef2f8; --muted:#94a3b8; --cyan:#38bdf8; --amber:#fbbf24; --emerald:#34d399; --rose:#fb7185; }}
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ background:var(--bg); color:var(--text); font-family:Inter,-apple-system,BlinkMacSystemFont,sans-serif; padding:28px; line-height:1.5; }}
    .wrap {{ max-width:1120px; margin:0 auto; display:flex; flex-direction:column; gap:18px; }}
    header {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; padding:18px 22px; border-radius:12px; background:linear-gradient(135deg,#0d1729,#0a101d); border:1px solid rgba(56,189,248,0.32); }}
    h1 {{ font-size:20px; letter-spacing:0.04em; }}
    h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:0.09em; color:var(--cyan); margin-bottom:10px; }}
    .sub {{ font-size:12px; color:var(--muted); margin-bottom:10px; }}
    .grid-4 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; }}
    .metric {{ background:var(--card); border:1px solid var(--border); border-radius:9px; padding:12px 14px; }}
    .metric .k {{ font-size:10px; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); }}
    .metric .v {{ font-family:ui-monospace,monospace; font-size:18px; font-weight:700; margin-top:4px; }}
    .card {{ background:var(--card); border:1px solid var(--border); border-radius:11px; padding:16px 18px; }}
    .pill {{ display:inline-block; padding:3px 9px; border-radius:6px; font-family:ui-monospace,monospace; font-size:11px; font-weight:800; }}
    .pill.GO, .pill.PASS, .pill.IMPROVED {{ background:rgba(52,211,153,0.18); color:var(--emerald); border:1px solid rgba(52,211,153,0.5); }}
    .pill.WARN {{ background:rgba(251,191,36,0.18); color:var(--amber); border:1px solid rgba(251,191,36,0.5); }}
    .pill.BLOCK, .pill.FAIL, .pill.REGRESSION {{ background:rgba(251,113,133,0.2); color:var(--rose); border:1px solid rgba(251,113,133,0.5); }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th, td {{ text-align:left; padding:7px 9px; border-bottom:1px solid rgba(148,163,184,0.10); }}
    th {{ font-size:10px; text-transform:uppercase; color:var(--muted); letter-spacing:0.07em; }}
    canvas {{ width:100%; height:360px; display:block; border-radius:9px; background:radial-gradient(circle at 50% 35%, #111f36, #05080f); border:1px solid var(--border); cursor:grab; }}
    pre {{ background:#060a12; border:1px solid var(--border); border-radius:8px; padding:12px; overflow:auto; max-height:260px; font-size:11px; color:#cbd5e1; }}
    button {{ padding:6px 12px; border-radius:6px; border:1px solid rgba(56,189,248,0.5); background:#0ea5e9; color:#fff; font-weight:600; cursor:pointer; font-size:12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>STATESET 3D WEIGHT STUDIO · INTERACTIVE PR REPORT</h1>
        <div class="sub">{model_name} · Base: {base_model} · {arch} · Generated {generated_at}</div>
      </div>
      <span class="pill {verdict_class}">EVAL GATE: {verdict}</span>
    </header>

    <div class="grid-4">
      <div class="metric"><div class="k">Layers / Tensors</div><div class="v">{structure_summary.get('num_layers', 0)}L · {structure_summary.get('num_tensors', 0)}T</div></div>
      <div class="metric"><div class="k">Parameters</div><div class="v">{int(structure_summary.get('total_params', 0)):,}</div></div>
      <div class="metric"><div class="k">Active Surgical Edits</div><div class="v">{len(edits)}</div></div>
      <div class="metric"><div class="k">Eval Pass Rate</div><div class="v">{((eval_report or {}).get('edited_summary', {}).get('pass_rate', 0.0) * 100):.0f}%</div></div>
    </div>

    <section class="card">
      <h2>Interactive 3D Crystal Relief Preview (Drag to Orbit)</h2>
      <div class="sub">{html.escape(grid_label)} — offline interactive isometric 3D projection embedded directly in this HTML file.</div>
      <canvas id="c3d" width="1040" height="360"></canvas>
    </section>

    <section class="card">
      <h2>Pre-Export Batch Eval Gate Scorecard</h2>
      <div class="sub">{html.escape(str((eval_report or {}).get('verdict_reason', 'Run Batch Eval Gate in Weight Studio to populate regression metrics.')))}</div>
      <table>
        <thead><tr><th>Category</th><th>Case</th><th>Base</th><th>Edited</th><th>Delta</th><th>Status</th></tr></thead>
        <tbody>{eval_rows_html}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Surgical Edit Lineage</h2>
      <table>
        <thead><tr><th>Edit Label</th><th>Op</th><th>Tensors</th><th>Changed Weights</th><th>Δ‖W‖_F</th></tr></thead>
        <tbody>{edit_rows_html}</tbody>
      </table>
    </section>

    {sens_html}

    <section class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <h2 style="margin:0">Portable Edit Recipe (.studio-recipe.json)</h2>
        <button id="btn-copy">Copy Recipe JSON</button>
      </div>
      <pre id="recipe-pre">{recipe_pretty}</pre>
    </section>
  </div>

  <script>
    const DATA = {safe_json};
    document.getElementById('btn-copy').onclick = () => {{
      navigator.clipboard?.writeText(document.getElementById('recipe-pre').textContent || '');
      document.getElementById('btn-copy').textContent = 'Copied!';
    }};

    const canvas = document.getElementById('c3d');
    const ctx = canvas.getContext('2d');
    let yaw = 0.68, pitch = 0.56, dragging = false, lx = 0, ly = 0;

    canvas.addEventListener('pointerdown', (e) => {{ dragging = true; lx = e.clientX; ly = e.clientY; }});
    window.addEventListener('pointerup', () => {{ dragging = false; }});
    window.addEventListener('pointermove', (e) => {{
      if (!dragging) return;
      yaw += (e.clientX - lx) * 0.008;
      pitch = Math.max(0.2, Math.min(1.25, pitch + (e.clientY - ly) * 0.006));
      lx = e.clientX; ly = e.clientY;
      draw();
    }});

    function draw() {{
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      const g = DATA.grid;
      const bars = [];
      if (g.rows > 0 && g.cols > 0 && g.values.length) {{
        const absMax = Math.max(1e-6, ...g.values.map(Math.abs));
        for (let r = 0; r < g.rows; r++) {{
          for (let c = 0; c < g.cols; c++) {{
            const v = g.values[r * g.cols + c] || 0;
            bars.push({{
              x: (c / Math.max(g.cols - 1, 1) - 0.5) * 14,
              z: (r / Math.max(g.rows - 1, 1) - 0.5) * 14,
              h: 0.3 + (Math.abs(v) / absMax) * 5.2,
              pos: v >= 0,
              mag: Math.abs(v) / absMax,
            }});
          }}
        }}
      }} else {{
        const ls = DATA.layers || [];
        const maxN = Math.max(1e-5, ...ls.map((l) => l.frobenius_norm || 0));
        ls.forEach((l, i) => {{
          bars.push({{
            x: (i / Math.max(ls.length - 1, 1) - 0.5) * 14,
            z: 0,
            h: 0.4 + ((l.frobenius_norm || 0) / maxN) * 5.6,
            pos: !l.has_edits,
            mag: (l.frobenius_norm || 0) / maxN,
          }});
        }});
      }}

      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const proj = (x, y, z) => {{
        const rx = x * cy - z * sy;
        const rz = x * sy + z * cy;
        const ry = y * cp - rz * sp;
        const dz = y * sp + rz * cp + 24;
        const f = 430 / Math.max(dz, 2);
        return {{ sx: W * 0.5 + rx * f, sy: H * 0.62 - ry * f, depth: dz }};
      }};

      const items = bars.map((b) => ({{ ...b, p0: proj(b.x, 0, b.z), p1: proj(b.x, b.h, b.z) }}));
      items.sort((a, b) => b.p0.depth - a.p0.depth);
      const barW = Math.max(4, Math.min(18, 220 / Math.sqrt(Math.max(items.length, 1))));
      for (const it of items) {{
        const col = it.pos
          ? `rgba(${{Math.round(90 + it.mag * 165)}}, ${{Math.round(80 + it.mag * 115)}}, ${{Math.round(30 + it.mag * 25)}}, 0.92)`
          : `rgba(${{Math.round(25 + it.mag * 45)}}, ${{Math.round(115 + it.mag * 115)}}, ${{Math.round(180 + it.mag * 75)}}, 0.92)`;
        ctx.strokeStyle = col;
        ctx.lineWidth = barW;
        ctx.lineCap = 'butt';
        ctx.beginPath();
        ctx.moveTo(it.p0.sx, it.p0.sy);
        ctx.lineTo(it.p1.sx, it.p1.sy);
        ctx.stroke();
      }}
    }}
    draw();
  </script>
</body>
</html>
"""
