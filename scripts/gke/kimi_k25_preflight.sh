#!/usr/bin/env bash
set -euo pipefail

namespace="${1:-stateset-agents}"

context="$(kubectl config current-context 2>/dev/null || true)"
if [ -n "${context}" ]; then
  echo "kubectl context: ${context}"
fi

mode="unknown"
detect_mode_from_gcloud() {
  if ! command -v gcloud >/dev/null 2>&1; then
    return 1
  fi
  if [ -z "${context}" ]; then
    return 1
  fi
  if [[ "${context}" =~ ^gke_([^_]+)_([^_]+)_([^_]+)$ ]]; then
    local project="${BASH_REMATCH[1]}"
    local location="${BASH_REMATCH[2]}"
    local cluster="${BASH_REMATCH[3]}"
    local enabled=""
    if [[ "${location}" =~ -[a-z]$ ]]; then
      enabled="$(gcloud container clusters describe "${cluster}" --project="${project}" --zone="${location}" --format='value(autopilot.enabled)' 2>/dev/null || true)"
    else
      enabled="$(gcloud container clusters describe "${cluster}" --project="${project}" --region="${location}" --format='value(autopilot.enabled)' 2>/dev/null || true)"
    fi
    if echo "${enabled}" | grep -qi '^true$'; then
      echo "autopilot"
      return 0
    fi
    if echo "${enabled}" | grep -qi '^false$'; then
      echo "standard"
      return 0
    fi
  fi
  return 1
}

if mode_from_gcloud="$(detect_mode_from_gcloud)"; then
  mode="${mode_from_gcloud}"
else
  autopilot_values="$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.labels.cloud\.google\.com/gke-autopilot}{"\n"}{end}' 2>/dev/null || true)"
  provisioning_values="$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.labels.cloud\.google\.com/gke-provisioning}{"\n"}{end}' 2>/dev/null || true)"

  if echo "${autopilot_values}" | grep -qiE '^true$|^enabled$'; then
    mode="autopilot"
  elif echo "${provisioning_values}" | grep -qiE '^standard$'; then
    mode="standard"
  fi
fi

echo "cluster mode: ${mode}"
echo ""

echo "namespace: ${namespace}"
if kubectl get namespace "${namespace}" >/dev/null 2>&1; then
  echo "  ok: namespace exists"
else
  echo "  missing: namespace does not exist"
  echo "  create: kubectl create namespace ${namespace}"
fi
echo ""

gpu_nodes="$(kubectl get nodes -l cloud.google.com/gke-accelerator --no-headers 2>/dev/null | wc -l | tr -d ' ')"
if [ "${gpu_nodes}" -gt 0 ]; then
  echo "gpu nodes: ${gpu_nodes}"
  echo "gpu labels:"
  kubectl get nodes -l cloud.google.com/gke-accelerator \
    -o custom-columns=NAME:.metadata.name,ACCEL:.metadata.labels.cloud\\.google\\.com/gke-accelerator \
    --no-headers \
    | head -n 20 || true
else
  echo "gpu nodes: (none detected)"
  if [ "${mode}" = "standard" ]; then
    echo "  standard next step: create a GPU node pool (see scripts/gke/create_gpu_nodepool.sh)"
  elif [ "${mode}" = "autopilot" ]; then
    echo "  autopilot next step: request GPUs in Pod specs (nvidia.com/gpu + nodeSelector)"
  fi
fi
echo ""

plugin_pods="$(kubectl -n kube-system get pods -l k8s-app=nvidia-gpu-device-plugin --no-headers 2>/dev/null | wc -l | tr -d ' ')"
if [ "${plugin_pods}" -gt 0 ]; then
  echo "nvidia device plugin pods: ${plugin_pods}"
else
  echo "nvidia device plugin pods: (none running)"
  if [ "${gpu_nodes}" -gt 0 ]; then
    echo "  warning: GPU nodes exist but device plugin pods are not scheduled yet"
  fi
fi
echo ""

echo "secrets:"
for s in hf-secret stateset-agents-secrets wandb-secret; do
  if kubectl -n "${namespace}" get secret "${s}" >/dev/null 2>&1; then
    echo "  ok: ${s}"
  else
    echo "  missing: ${s}"
  fi
done
echo ""

echo "pvc:"
if kubectl -n "${namespace}" get pvc stateset-models-pvc >/dev/null 2>&1; then
  kubectl -n "${namespace}" get pvc stateset-models-pvc
else
  echo "  missing: stateset-models-pvc"
  echo "  create: kubectl apply -f deployment/kubernetes/stateset-models-pvc.yaml"
fi
echo ""

echo "workloads (namespace=${namespace}):"
kubectl -n "${namespace}" get deploy,svc 2>/dev/null || true
echo ""

cat <<'EOF'
next steps:
- If you are on Standard and have no GPU nodes, create a GPU node pool.
- Apply vLLM + API gateway manifests, or use Helm.
- Use scripts/render_kimi_k25_helm_values.py to render Helm values from training output.
EOF
