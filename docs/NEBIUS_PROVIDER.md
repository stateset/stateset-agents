# Nebius provider

StateSet integrates with Nebius Serverless AI for bounded training jobs and
authenticated vLLM endpoints. It uses the official `nebius` CLI for control
plane operations and Nebius' S3-compatible Object Storage for datasets and
artifacts.

## Installation and authentication

```bash
pip install "stateset-agents[nebius]"
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
nebius profile create
```

For automation, use a Nebius service-account profile rather than browser
authentication. Configure the training substrate:

```bash
export NEBIUS_PROFILE=stateset
export NEBIUS_SUBNET_ID=vpcsubnet-...
export NEBIUS_S3_BUCKET=stateset-training
export NEBIUS_S3_ENDPOINT_URL=https://storage.<region>.nebius.cloud
export NEBIUS_PLATFORM=gpu-l40s-a
export NEBIUS_PRESET=1gpu-8vcpu-32gb
```

The preset's leading GPU count must exactly match `--gpu-count`; StateSet
rejects mismatches instead of silently allocating a different topology.

The caller needs ordinary AWS SDK credentials to upload and retrieve objects.
The remote job receives credentials through SecretStash selectors (called
MysteryBox in some CLI/API surfaces), not raw values:

```bash
export NEBIUS_S3_ACCESS_KEY_SECRET=mbsec-access-key
export NEBIUS_S3_SECRET_KEY_SECRET=mbsec-secret-key
# Optional for gated models:
export NEBIUS_HF_TOKEN_SECRET=mbsec-hugging-face
```

## Training

```bash
stateset-agents train-remote \
  --provider nebius \
  --dataset improved/curated.jsonl \
  --base-model Qwen/Qwen3.5-9B \
  --gpu gpu-l40s-a \
  --timeout 7200 \
  --output-dir outputs/nebius
```

Submission uploads the exact local dataset to a unique prefix, launches the
same packaged `run_sft_job` entrypoint used by other machine providers, and
persists a durable job handle. `remote-job` can reconnect after the original
process exits. Fetching downloads all returned artifacts and then removes the
job-owned object prefix. If submission fails, staged input is rolled back;
cancellation requests provider cancellation and removes job-owned objects.

Nebius' CLI timeout has a one-hour minimum, so shorter StateSet timeouts round
up to one hour and longer values round up to the next hour. `--max-cost` fails
closed because the documented job CLI does not expose an authoritative price
before allocation.

## Serverless vLLM endpoint

Endpoints mount a complete model directory read-only from Object Storage.
Create a SecretStash AWS-profile secret and an endpoint-token secret. Both are
required: StateSet refuses the provider-generated random-token path because
that credential cannot be recovered reliably from the durable handle.

```bash
export NEBIUS_S3_PROFILE_SECRET=mbsec-aws-profile
export NEBIUS_ENDPOINT_TOKEN_SECRET=mbsec-endpoint-token

stateset-agents inference-deploy \
  --provider nebius \
  --name support-production \
  --model-name support-model \
  --weights-uri s3://model-weights/support-model \
  --gpu gpu-h100-sxm
```

The provider-managed HTTPS endpoint is token-authenticated; StateSet does not
request an additional public VM IP. Delete it explicitly when it is no longer
needed:

```bash
stateset-agents inference-delete --provider nebius \
  --deployment-id <id> --model-name support-model
```

Nebius currently manages endpoint replica behavior, so the adapter accepts
only the truthful `min_replicas=1`, `max_replicas=1` representation instead
of pretending StateSet controls unsupported autoscaling knobs.

## Evidence status

CLI construction, durable reconnects, state mapping, SecretStash secret
injection, artifact transfer and cleanup, cancellation, endpoint creation,
status, deletion, and fail-closed constraints are unit-pinned. Live Serverless
AI training and endpoint certification remain pending. See
[`PROOFS.md`](PROOFS.md).

Official references: [Serverless AI jobs](https://docs.nebius.com/serverless/jobs/manage)
and [`nebius ai endpoint create`](https://docs.nebius.com/cli/reference/ai/endpoint/create).
