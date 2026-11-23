/**
 * Modularized v8+ Autopilot, Red-Team, Tournament, Canary & Live AI-Agent MCP Panel.
 *
 * Extracted out of app.js to keep the Weight Studio frontend cleanly modularized
 * with zero bundler dependencies.
 */

export function initAutopilotAndMcpPanel({
  $,
  API,
  state,
  renderer,
  apiPost,
  loadModelStructure,
  showToast,
}) {
  // 1. Adversarial Red-Team Safety Audit & Surgical Immunization
  const runRedTeam = async (immunize) => {
    if (!state.modelId) return;
    const pill = $('redteam-status-pill');
    const summary = $('redteam-result-summary');
    const btnAudit = $('btn-run-redteam-audit');
    const btnImm = $('btn-run-redteam-immunize');
    if (pill) pill.textContent = immunize ? 'IMMUNIZING…' : 'AUDITING…';
    if (btnAudit) btnAudit.disabled = true;
    if (btnImm) btnImm.disabled = true;
    try {
      const res = await apiPost('/redteam/run', {
        model_id: state.modelId,
        checkpoint_id: state.checkpointId,
        immunize,
        export_format: 'none',
      });
      const asrPct = (Number(res.attack_success_rate || 0) * 100).toFixed(0);
      const passPct = (Number(res.safety_pass_rate || 0) * 100).toFixed(0);
      if (pill) pill.textContent = `${res.verdict} · ASR ${asrPct}%`;
      if (summary) {
        if (res.immunization) {
          const imm = res.immunization;
          summary.textContent =
            `Immunized L${String(imm.target_layer).padStart(2, '0')} (α=${Number(imm.chosen_strength).toFixed(2)}) · ` +
            `ASR ${(imm.asr_before * 100).toFixed(0)}% → ${(imm.asr_after * 100).toFixed(0)}% · ` +
            `Benign Gate: ${imm.benign_eval_verdict}`;
        } else {
          summary.textContent =
            `Pass ${passPct}% (${res.after.total_probes - res.after.attacks_succeeded}/${res.after.total_probes} probes) · ` +
            `Mean safety ${Number(res.after.mean_safety_score).toFixed(3)}`;
        }
      }
      renderer.triggerPulse(immunize ? 'backprop' : 'forward');
      if (immunize) {
        await loadModelStructure(state.modelId, state.checkpointId);
      }
      showToast(
        immunize
          ? `🛡️ Immunized (${res.verdict}, ASR ${asrPct}%)`
          : `🔎 Red-Team Audit: ${res.verdict} (ASR ${asrPct}%)`,
        3200
      );
    } catch (err) {
      if (pill) pill.textContent = 'ERR';
      if (summary) summary.textContent = err.message;
      showToast(`Red-Team error: ${err.message}`, 4000);
    } finally {
      if (btnAudit) btnAudit.disabled = false;
      if (btnImm) btnImm.disabled = false;
    }
  };

  $('btn-run-redteam-audit')?.addEventListener('click', () => runRedTeam(false));
  $('btn-run-redteam-immunize')?.addEventListener('click', () => runRedTeam(true));

  // 2. Multi-Model Pareto Tournament & Evolutionary Surgical Merge Lab
  $('btn-run-tournament')?.addEventListener('click', async () => {
    const pill = $('tournament-status-pill');
    const summary = $('tournament-result-summary');
    const btn = $('btn-run-tournament');
    if (pill) pill.textContent = 'RUNNING…';
    if (btn) btn.disabled = true;
    try {
      const contenders = state.modelId ? [state.modelId] : [];
      if (state.secondaryModelId && state.secondaryModelId !== state.modelId) {
        contenders.push(state.secondaryModelId);
      }
      const res = await apiPost('/tournament/run', {
        model_ids: contenders.length ? contenders : null,
        reward_fn: $('heal-reward-select')?.value || 'customer_support',
        include_redteam: true,
        synthesize_merge: true,
        export_format: 'none',
      });
      const champ = res.champion || {};
      if (pill) pill.textContent = `FIT ${Number(champ.fitness || 0).toFixed(3)}`;
      if (summary) {
        summary.textContent =
          `Champion: ${champ.label || '?'} · Reward ${Number(champ.reward_score || 0).toFixed(3)} · ` +
          `Safety ${(Number(champ.safety_pass_rate || 0) * 100).toFixed(0)}% ` +
          `(${res.contender_count} contenders, ${res.merge_candidates_evaluated} merges)`;
      }
      renderer.triggerPulse(res.promoted_merge ? 'backprop' : 'forward');
      if (res.promoted_merge && state.modelId) {
        await loadModelStructure(state.modelId, state.checkpointId);
      }
      showToast(`🏆 Tournament Champion: ${champ.label || 'done'}`, 3200);
    } catch (err) {
      if (pill) pill.textContent = 'ERR';
      if (summary) summary.textContent = err.message;
      showToast(`Tournament error: ${err.message}`, 4000);
    } finally {
      if (btn) btn.disabled = false;
    }
  });

  // 3. Production Canary Shadow Gate & One-Call Full Autopilot
  const runCanaryOrAutopilot = async (autopilot) => {
    if (!state.modelId) return;
    const pill = $('canary-status-pill');
    const summary = $('canary-result-summary');
    const btnC = $('btn-run-canary');
    const btnA = $('btn-run-autopilot');
    if (pill) pill.textContent = autopilot ? 'AUTOPILOT…' : 'GATING…';
    if (btnC) btnC.disabled = true;
    if (btnA) btnA.disabled = true;
    try {
      const res = autopilot
        ? await apiPost('/autopilot/run', {
            model_id: state.modelId,
            checkpoint_id: state.checkpointId,
            reward_fn: $('heal-reward-select')?.value || 'customer_support',
            immunize_redteam: true,
            export_format: 'lora',
          })
        : await apiPost('/canary/gate', {
            model_id: state.modelId,
            checkpoint_id: state.checkpointId,
            stage: 'canary_10',
            reward_fn: $('heal-reward-select')?.value || 'composite',
          });
      if (pill) pill.textContent = `${res.decision} · ${res.recommended_traffic_pct}%`;
      const pr = res.paired_record || (res.canary_gate && res.canary_gate.paired_record) || {};
      if (summary) {
        summary.textContent =
          `${res.decision} (${res.rollout_id}) · W/T/L ${pr.wins || 0}/${pr.ties || 0}/${pr.losses || 0} · ` +
          `Δ=${Number(pr.mean_delta || 0) >= 0 ? '+' : ''}${Number(pr.mean_delta || 0).toFixed(4)}` +
          (res.exported ? ' · Verified LoRA + Dossier exported' : '');
      }
      renderer.triggerPulse(autopilot ? 'backprop' : 'forward');
      if (autopilot) {
        await loadModelStructure(state.modelId, state.checkpointId);
      }
      showToast(
        autopilot
          ? `🚀 Autopilot: ${res.decision} (traffic ${res.recommended_traffic_pct}%)`
          : `🧪 Canary Gate: ${res.decision} (traffic ${res.recommended_traffic_pct}%)`,
        3200
      );
    } catch (err) {
      if (pill) pill.textContent = 'ERR';
      if (summary) summary.textContent = err.message;
      showToast(`Canary/Autopilot error: ${err.message}`, 4000);
    } finally {
      if (btnC) btnC.disabled = false;
      if (btnA) btnA.disabled = false;
    }
  };

  $('btn-run-canary')?.addEventListener('click', () => runCanaryOrAutopilot(false));
  $('btn-run-autopilot')?.addEventListener('click', () => runCanaryOrAutopilot(true));

  // 4. Live AI-Agent MCP Connect Modal & Real-Time SSE Co-Pilot Stream
  const mcpModal = $('mcp-modal');
  const openMcpModal = () => {
    if (!mcpModal) return;
    const baseUrl = window.location.origin;
    const httpUrl = `${baseUrl}/api/weight-studio/mcp`;
    const httpPre = $('mcp-http-url');
    const cliPre = $('mcp-claude-cmd');
    const jsonPre = $('mcp-json-snippet');
    if (httpPre) httpPre.textContent = httpUrl;
    if (cliPre) cliPre.textContent = `claude mcp add stateset-agents -- stateset-agents mcp --workspace .`;
    if (jsonPre) {
      jsonPre.textContent = JSON.stringify(
        {
          mcpServers: {
            'stateset-weight-studio': {
              url: httpUrl,
              transport: 'streamable-http',
            },
            'stateset-agents': {
              command: 'stateset-agents',
              args: ['mcp', '--workspace', '.'],
            },
          },
        },
        null,
        2
      );
    }
    mcpModal.classList.remove('hidden');
  };

  $('btn-mcp-modal')?.addEventListener('click', openMcpModal);
  $('btn-close-mcp')?.addEventListener('click', () => mcpModal?.classList.add('hidden'));
  mcpModal?.addEventListener('click', (e) => {
    if (e.target === mcpModal) mcpModal.classList.add('hidden');
  });
  $('btn-copy-mcp-http')?.addEventListener('click', () => {
    const txt = $('mcp-http-url')?.textContent || '';
    navigator.clipboard?.writeText(txt).catch(() => {});
    showToast('Copied Live HTTP MCP endpoint URL', 1800);
  });
  $('btn-copy-mcp-json')?.addEventListener('click', () => {
    const txt = $('mcp-json-snippet')?.textContent || '';
    navigator.clipboard?.writeText(txt).catch(() => {});
    showToast('Copied MCP client JSON config', 1800);
  });

  const appendActivity = (ev) => {
    const log = $('mcp-activity-log');
    if (!log) return;
    const line = `[${(ev.timestamp || '').slice(11, 19)}] ${ev.tool || ev.type}: ${ev.summary || ''}`;
    const existing =
      log.textContent === 'Waiting for AI agent MCP calls…' ? [] : log.textContent.split('\n');
    existing.unshift(line);
    log.textContent = existing.slice(0, 12).join('\n');
  };

  try {
    const es = new EventSource(`${API}/events`);
    es.onmessage = async (msg) => {
      try {
        const ev = JSON.parse(msg.data);
        if (!ev || ev.type === 'heartbeat') return;
        appendActivity(ev);
        const dot = $('mcp-live-dot');
        if (dot) {
          dot.style.boxShadow = '0 0 10px #10b981';
          setTimeout(() => {
            dot.style.boxShadow = 'none';
          }, 1200);
        }
        if (ev.pulse) renderer.triggerPulse(ev.pulse);
        if (ev.summary) showToast(`🤖 MCP Agent: ${ev.summary}`, 2600);
        if (ev.mutated && (!ev.model_id || ev.model_id === state.modelId) && state.modelId) {
          await loadModelStructure(state.modelId, state.checkpointId);
        }
      } catch (_err) {
        // Ignore malformed SSE frames
      }
    };
  } catch (_err) {
    // EventSource unavailable in non-browser environment
  }
}
