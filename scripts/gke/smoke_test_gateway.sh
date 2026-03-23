#!/usr/bin/env bash
set -euo pipefail

namespace="${NAMESPACE:-stateset-agents}"
service="${SERVICE:-stateset-agents-api}"
local_port="${LOCAL_PORT:-8000}"
model="${MODEL_ID:-moonshotai/Kimi-K2.5}"
api_key="${STATESET_API_KEY:-}"

usage() {
  cat <<EOF
Smoke test the StateSet Agents gateway (/v1/messages + OpenAI compatibility).

Env vars:
  NAMESPACE        (default: ${namespace})
  SERVICE          (default: ${service})
  LOCAL_PORT       (default: ${local_port})
  MODEL_ID         (default: ${model})
  STATESET_API_KEY (optional) Bearer token for auth-enabled gateways

Example:
  NAMESPACE=stateset-agents SERVICE=stateset-agents-api \\
    STATESET_API_KEY=... \\
    bash scripts/gke/smoke_test_gateway.sh
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if ! kubectl -n "${namespace}" get svc "${service}" >/dev/null 2>&1; then
  echo "service not found: ${namespace}/${service}" >&2
  echo "hint: deploy the gateway first (Helm or deployment/kubernetes/production-deployment.yaml)" >&2
  exit 2
fi

remote_port="$(kubectl -n "${namespace}" get svc "${service}" -o jsonpath='{.spec.ports[0].port}')"
if [ -z "${remote_port}" ]; then
  echo "could not determine service port for ${namespace}/${service}" >&2
  exit 2
fi

headers=(-H "Content-Type: application/json")
if [ -n "${api_key}" ]; then
  headers+=(-H "Authorization: Bearer ${api_key}")
fi

tmp_log="$(mktemp)"
cleanup() {
  if [ -n "${pf_pid:-}" ]; then
    kill "${pf_pid}" >/dev/null 2>&1 || true
  fi
  rm -f "${tmp_log}" || true
}
trap cleanup EXIT

echo "port-forward: svc/${service} ${local_port}:${remote_port} (namespace=${namespace})"
kubectl -n "${namespace}" port-forward "service/${service}" "${local_port}:${remote_port}" >"${tmp_log}" 2>&1 &
pf_pid="$!"

sleep 2

echo "GET /healthz"
curl -fsS "http://127.0.0.1:${local_port}/healthz" | cat
echo ""

echo "POST /v1/messages (anthropic-style)"
curl -fsS "http://127.0.0.1:${local_port}/v1/messages" \
  "${headers[@]}" \
  -d "{\"model\":\"${model}\",\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}" \
  | head -c 500
echo ""
echo ""

echo "POST /v1/messages (openai response_format)"
curl -fsS "http://127.0.0.1:${local_port}/v1/messages" \
  "${headers[@]}" \
  -d "{\"model\":\"${model}\",\"response_format\":\"openai\",\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}" \
  | head -c 500
echo ""
echo ""

echo "POST /v1/chat/completions (openai-compatible)"
curl -fsS "http://127.0.0.1:${local_port}/v1/chat/completions" \
  "${headers[@]}" \
  -d "{\"model\":\"${model}\",\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}" \
  | head -c 500
echo ""
echo ""

echo "POST /v1/messages (stream=true) [first events]"
curl -fsS -N "http://127.0.0.1:${local_port}/v1/messages" \
  "${headers[@]}" \
  -d "{\"model\":\"${model}\",\"stream\":true,\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}" \
  | head -n 20
echo ""

echo "ok"

