#!/usr/bin/env bash
set -euo pipefail

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

echo "node pools:"
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.labels.cloud\.google\.com/gke-nodepool}{"\n"}{end}' \
  | sed '/^$/d' | sort | uniq -c || true
echo ""

echo "accelerators (node labels):"
accel_values="$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.labels.cloud\.google\.com/gke-accelerator}{"\n"}{end}' 2>/dev/null || true)"
if [ -z "$(echo "${accel_values}" | sed '/^$/d')" ]; then
  echo "  (none detected)"
else
  echo "${accel_values}" | sed '/^$/d' | sort | uniq -c
fi
echo ""

echo "example node label view:"
kubectl get nodes -o custom-columns=NAME:.metadata.name,PROVISIONING:.metadata.labels.cloud\\.google\\.com/gke-provisioning,NODEPOOL:.metadata.labels.cloud\\.google\\.com/gke-nodepool,ACCEL:.metadata.labels.cloud\\.google\\.com/gke-accelerator --no-headers \
  | head -n 20 || true
echo ""

case "${mode}" in
  autopilot)
    cat <<'EOF'
next steps (autopilot):
- You do not create/manage node pools. Request GPUs in Pod specs:
  - resources.requests/limits: nvidia.com/gpu: "N"
  - nodeSelector: cloud.google.com/gke-accelerator: <gpu-type>
  - optionally: cloud.google.com/gke-gpu-driver-version: latest
- See: docs/KIMI_K25_GKE_AUTOPILOT.md
EOF
    ;;
  standard)
    cat <<'EOF'
next steps (standard):
- Create a GPU node pool (or enable Node Auto Provisioning for GPUs), then deploy vLLM + the gateway.
- See: docs/KIMI_K25_GKE_STANDARD.md
EOF
    ;;
  *)
    cat <<'EOF'
next steps:
- Could not confidently detect Autopilot vs Standard from node labels.
- Check the cluster in GCP console, or re-run after nodes are ready.
EOF
    ;;
esac
