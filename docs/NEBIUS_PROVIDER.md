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

Submission records a unique job name locally, uploads the dataset to a
job-specific prefix, and requests an asynchronous create. The returned handle
contains the Nebius job ID. `train-remote` prints this ID before polling.
To reconnect from the same state directory and
Nebius profile after the submitting process exits, run

```bash
stateset-agents remote-job --provider nebius --job-id <job-id> --wait
```

If the create reply is lost, the error prints the same command with the
generated `stateset-sft-...` name. Run it to look up the existing job and
fetch its output. This lookup can be repeated while the job becomes visible.
After an interruption, the provisional name is also the filename stem of its
record under `${XDG_CACHE_HOME:-$HOME/.cache}/stateset-agents/nebius-jobs/`.
Recovery needs that record and works after the original dataset is removed.
For a confirmed rejection such as insufficient capacity, StateSet removes the
staged input. If lookup continues to find no job, confirm the outcome in the
submitting project before removing the retained prefix or submitting again.

Fetching downloads artifacts before removing the job-owned prefix. A
cancellation request keeps that prefix until Nebius reports `CANCELLED`. If
the job completes or fails during cancellation, its output remains available
for fetch.

StateSet passes the requested timeout in seconds. The CLI reference gives a
range of five minutes to 168 hours, which StateSet validates before upload.
The job management guide lists a one-hour minimum, so acceptance of shorter
jobs still needs live verification. Nebius checks the timeout asynchronously.
`--max-cost` requires an authoritative price that the job CLI does not provide.
[Job create reference](https://docs.nebius.com/cli/reference/ai/job/create)
and [job management guide](https://docs.nebius.com/serverless/jobs/manage).

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

The deployment handle includes the managed HTTPS URL when Nebius has assigned
one. If the URL is pending, inspect `status.public_endpoints` with

```bash
stateset-agents inference-status --provider nebius \
  --deployment-id <id> --model-name support-model
```

Use the configured endpoint token when connecting. Delete the endpoint when
you finish using it.

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
