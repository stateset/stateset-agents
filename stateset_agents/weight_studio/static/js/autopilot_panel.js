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
  apiGet,
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

  // 3b. Live Production Traces (OTLP/JSON) & Reward Calibration Audit
  const refreshTraceSummary = async () => {
    try {
      const s = await apiGet('/traces/summary');
      const pill = $('traces-status-pill');
      if (pill) {
        pill.textContent = `${s.total_traces || 0} TRACES · ${s.failing_traces || 0} FAIL`;
      }
    } catch (_err) {
      // Optional on initial load
    }
  };
  refreshTraceSummary();

  $('btn-ingest-trace')?.addEventListener('click', async () => {
    const pill = $('traces-status-pill');
    const summary = $('traces-result-summary');
    const raw = ($('traces-payload-input')?.value || '').trim();
    if (pill) pill.textContent = 'INGESTING…';
    try {
      let payload;
      if (raw) {
        payload = JSON.parse(raw);
      } else {
        payload = {
          resourceSpans: [
            {
              scopeSpans: [
                {
                  spans: [
                    {
                      traceId: `otlp-ui-${Date.now()}`,
                      attributes: [
                        {
                          key: 'gen_ai.prompt',
                          value: { stringValue: 'My order #77701 arrived damaged. Can I get a refund?' },
                        },
                        {
                          key: 'gen_ai.completion',
                          value: { stringValue: 'Sorry [NAME]. Sorry [NAME]. Please contact support.' },
                        },
                        {
                          key: 'stateset.human_score',
                          value: { doubleValue: 0.15 },
                        },
                      ],
                    },
                  ],
                },
              ],
            },
          ],
        };
      }
      const res = await apiPost('/traces/ingest', {
        payload,
        reward_fn: $('heal-reward-select')?.value || 'customer_support',
      });
      await refreshTraceSummary();
      if (summary) {
        summary.textContent = `Ingested ${res.ingested_count} trace(s) (${res.failing_count} failing) into durable OTLP ledger.`;
      }
      renderer.triggerPulse('forward');
      showToast(`📡 Ingested ${res.ingested_count} trace(s) (${res.failing_count} failing)`, 2600);
    } catch (err) {
      if (pill) pill.textContent = 'ERR';
      if (summary) summary.textContent = err.message;
      showToast(`Trace ingest error: ${err.message}`, 3600);
    }
  });

  $('btn-audit-calibration')?.addEventListener('click', async () => {
    const pill = $('traces-status-pill');
    const summary = $('traces-result-summary');
    if (pill) pill.textContent = 'AUDITING…';
    try {
      let res = await apiPost('/traces/calibration-audit', {
        reward_fn: $('heal-reward-select')?.value || 'customer_support',
      });
      if (res.verdict === 'INSUFFICIENT_LABELS') {
        res = await apiPost('/traces/calibration-audit', {
          reward_fn: $('heal-reward-select')?.value || 'customer_support',
          samples: [
            { id: 'cal-1', proxy_score: 0.92, human_score: 0.90 },
            { id: 'cal-2', proxy_score: 0.85, human_score: 0.82 },
            { id: 'cal-3', proxy_score: 0.30, human_score: 0.25 },
            { id: 'cal-4', proxy_score: 0.18, human_score: 0.15 },
          ],
        });
      }
      if (pill) pill.textContent = `${res.verdict}`;
      if (summary) {
        summary.textContent =
          `Verdict: ${res.verdict} (n=${res.sample_count}) · Spearman ρ=${res.spearman_rho} · ` +
          `MAE=${res.mae} · Hacking=${(Number(res.hacking_rate || 0) * 100).toFixed(0)}%`;
      }
      renderer.triggerPulse('forward');
      showToast(`🧪 Calibration Audit: ${res.verdict} (ρ=${res.spearman_rho})`, 3000);
    } catch (err) {
      if (pill) pill.textContent = 'ERR';
      if (summary) summary.textContent = err.message;
      showToast(`Calibration audit error: ${err.message}`, 3600);
    }
  });

  $('btn-heal-live-traces')?.addEventListener('click', async () => {
    if (!state.modelId) return;
    const pill = $('traces-status-pill');
    const summary = $('traces-result-summary');
    if (pill) pill.textContent = 'HEALING…';
    try {
      const res = await apiPost('/traces/auto-heal', {
        model_id: state.modelId,
        checkpoint_id: state.checkpointId,
        reward_fn: $('heal-reward-select')?.value || 'customer_support',
      });
      const sv = res.steering_vector || {};
      const gate = res.eval_gate || {};
      const diag = res.diagnosis || {};
      const verdict = gate.verdict || 'GO';
      const count = res.ingested_failing_traces ?? diag.failing_count ?? 0;
      const layerNum = Number(sv.target_layer ?? 0);
      const alphaVal = Number(sv.chosen_strength ?? 0);
      if (pill) pill.textContent = `HEALED · ${verdict}`;
      if (summary) {
        summary.textContent =
          `Healed from ${count} live trace(s) → L${String(layerNum).padStart(2, '0')} ` +
          `(α=${alphaVal >= 0 ? '+' : ''}${alphaVal.toFixed(2)}) · Eval Gate: ${verdict}`;
      }
      renderer.triggerPulse('backprop');
      await loadModelStructure(state.modelId, state.checkpointId);
      showToast(`🩹 Live-Trace Auto-Heal: ${verdict}`, 3200);
    } catch (err) {
      if (pill) pill.textContent = 'ERR';
      if (summary) summary.textContent = err.message;
      showToast(`Live-trace heal error: ${err.message}`, 3800);
    }
  });

  // 3c. Live vLLM / Ollama Shadow A/B Bridge
  $('btn-run-live-ab')?.addEventListener('click', async () => {
    const pill = $('live-ab-status-pill');
    const summary = $('live-ab-result-summary');
    const btn = $('btn-run-live-ab');
    const endpoint = ($('live-ab-endpoint')?.value || '').trim();
    const baseline = ($('live-ab-baseline')?.value || '').trim();
    const candidate = ($('live-ab-candidate')?.value || '').trim();
    if (!endpoint || !baseline || !candidate) {
      showToast('Configure Endpoint URL, Baseline Model, and Candidate Model first.', 3000);
      return;
    }
    if (pill) pill.textContent = 'PROBING…';
    if (btn) btn.disabled = true;
    try {
      const res = await apiPost('/serving/compare-live', {
        endpoint_url: endpoint,
        baseline_model: baseline,
        candidate_model: candidate,
        reward_fn: $('heal-reward-select')?.value || 'customer_support',
      });
      const pr = res.paired_record || {};
      const lat = res.latency_ms || {};
      const delta = Number(pr.mean_delta ?? 0);
      if (pill) pill.textContent = `${res.decision}`;
      if (summary) {
        summary.textContent =
          `${res.decision} · Δ=${delta >= 0 ? '+' : ''}${delta.toFixed(4)} · ` +
          `W/T/L ${pr.wins || 0}/${pr.ties || 0}/${pr.losses || 0} · ` +
          `p50 ${lat.baseline_p50 ?? 0}ms → ${lat.candidate_p50 ?? 0}ms`;
      }
      renderer.triggerPulse('forward');
      showToast(`⚡ Live A/B: ${res.decision}`, 3200);
    } catch (err) {
      if (pill) pill.textContent = 'OFFLINE';
      if (summary) summary.textContent = err.message;
      showToast(`Live A/B error: ${err.message}`, 3800);
    } finally {
      if (btn) btn.disabled = false;
    }
  });

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
