# CoreWeave provider

StateSet uses CoreWeave in two distinct ways:

1. `train-remote --provider coreweave` submits a Kubernetes `Job` to an
   existing CoreWeave Kubernetes Service (CKS) cluster.
2. `inference-deploy --provider coreweave` creates a CoreWeave Dedicated
   Inference gateway and BYOW vLLM or Dynamo-vLLM deployment.

The two paths deliberately have separate interfaces. A LoRA adapter returned
by training is not necessarily a complete model directory accepted by BYOW
inference; merge or otherwise materialize complete weights before deployment.

## Installation

```bash
pip install "stateset-agents[coreweave]"
```

Training also requires `kubectl` and a kubeconfig authorized for the target
namespace. CoreWeave manages the NVIDIA GPU Operator on CKS; StateSet does not
install or modify it.

## Training configuration

Create an S3-compatible bucket and a Kubernetes Secret containing the AWS SDK
variables used to access it:

```bash
kubectl -n stateset create secret generic stateset-object-storage \
  --from-literal=AWS_ACCESS_KEY_ID=... \
  --from-literal=AWS_SECRET_ACCESS_KEY=...

export COREWEAVE_KUBE_CONTEXT=cks-usw04
export COREWEAVE_KUBE_NAMESPACE=stateset
export COREWEAVE_S3_BUCKET=my-training-artifacts
export COREWEAVE_S3_ENDPOINT_URL=https://<object-storage-endpoint>
export COREWEAVE_JOB_S3_ENDPOINT_URL=http://cwlota.com
export COREWEAVE_STORAGE_SECRET=stateset-object-storage
```

For gated Hugging Face models, create a second Secret with an `HF_TOKEN` key
and set `COREWEAVE_HF_SECRET` to its name. StateSet places only Secret
references in the Job manifest; credentials never enter `RemoteJobSpec`, job
labels, command arguments, or durable state.

```bash
stateset-agents train-remote \
  --provider coreweave \
  --dataset improved/curated.jsonl \
  --base-model Qwen/Qwen3.5-9B \
  --gpu H100 \
  --gpu-count 2 \
  --timeout 7200 \
  --output-dir outputs/coreweave
```

The executor uploads the dataset to a job-owned prefix, creates a bounded
Kubernetes Job, polls its durable name, streams logs, downloads the adapter,
and removes the object-store prefix after a verified fetch. Cancellation
deletes the Kubernetes Job and its job-owned objects. Kubernetes' active
deadline enforces `--timeout`.

`--max-cost` fails closed on CKS because cluster capacity is billed outside an
individual Kubernetes Job and no authoritative per-job price exists at submit
time.

## Dedicated Inference

`weights-uri` must point to a complete model directory in CoreWeave AI Object
Storage with the bucket policy required by Dedicated Inference.

```bash
export COREWEAVE_API_TOKEN=...

stateset-agents inference-deploy \
  --provider coreweave \
  --name support-production \
  --model-name support-model \
  --weights-uri s3://model-weights/support-model \
  --gpu gd-8xh100ib-i128 \
  --gpu-count 2 \
  --min-replicas 1 \
  --max-replicas 3 \
  --runtime dynamo-vllm \
  --zone US-WEST-04A
```

If `--gateway-id` is omitted, StateSet creates a CoreWeave-IAM-authenticated,
OpenAI body-routed gateway. A deployment failure rolls that gateway back. The
returned handle records ownership so `inference-delete` removes both the
deployment and an automatically created gateway.

```bash
stateset-agents inference-status --provider coreweave \
  --deployment-id <id> --model-name support-model
stateset-agents inference-delete --provider coreweave \
  --deployment-id <id> --model-name support-model \
  --gateway-id <gateway-id> --delete-gateway
```

The deploy command prints both IDs and whether StateSet created the gateway.
Persist that JSON. Because status/delete commands are independent processes,
pass `--gateway-id ... --delete-gateway` only when the printed handle has
`owns_gateway: true`. Shared gateways are never deleted implicitly.

Choose GPU instance types, zones, and runtime versions from CoreWeave's live
`/deployments/parameters` and `/gateways/parameters` endpoints. CoreWeave now
recommends Dynamo-vLLM; plain vLLM remains available for compatibility.

## Evidence status

The adapter, manifest generation, state recovery, secret references,
Kubernetes lifecycle, artifact cleanup, gateway rollback, deployment payload,
and deletion paths are unit-pinned. Live CKS training and Dedicated Inference
certification remain pending until credentials and retained provider evidence
are available. See [`PROOFS.md`](PROOFS.md).

Official references: [CKS GPU workloads](https://docs.coreweave.com/products/cks/tutorials/hello-world-nims-on-cks)
and [Dedicated Inference](https://docs.coreweave.com/products/inference/getting-started).
