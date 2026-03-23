# Cloud Deployment Guide

Production deployment guides for StateSet Agents on AWS, GCP, and Azure.

Note: as of `stateset-agents` v0.7.x, deployment is split into:
- `stateset/stateset-agents-api`: FastAPI gateway (Anthropic/OpenAI compatible)
- `stateset/stateset-agents-trainer`: training job image (bring your own CUDA base)

GPU inference should run in a dedicated model server (e.g. vLLM) deployed
alongside the gateway. The gateway proxies requests to that backend.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [AWS Deployment](#aws-deployment)
- [GCP Deployment](#gcp-deployment)
- [Azure Deployment](#azure-deployment)
- [Docker Configuration](#docker-configuration)
- [Environment Variables](#environment-variables)
- [Cost Optimization](#cost-optimization)

---

## Prerequisites

- Docker 20.10+
- Terraform 1.5+ or Pulumi 3.0+
- Cloud CLI tools (aws-cli, gcloud, az)
- Python 3.10+

---

## AWS Deployment

### ECS/Fargate Setup

```hcl
# terraform/aws/main.tf
provider "aws" {
  region = var.aws_region
}

resource "aws_ecs_cluster" "stateset" {
  name = "stateset-agents"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_task_definition" "api" {
  family                   = "stateset-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 2048
  memory                   = 4096
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "api"
    image = "${aws_ecr_repository.stateset.repository_url}:latest"

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    environment = [
      { name = "API_ENVIRONMENT", value = "production" },
      { name = "API_WORKERS", value = "4" }
    ]

    secrets = [
      { name = "API_JWT_SECRET", valueFrom = aws_secretsmanager_secret.jwt.arn }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.api.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "api"
      }
    }
  }])
}

resource "aws_ecs_service" "api" {
  name            = "stateset-api"
  cluster         = aws_ecs_cluster.stateset.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnets
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
}

# Auto-scaling
resource "aws_appautoscaling_target" "api" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.stateset.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "api_cpu" {
  name               = "cpu-auto-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.api.resource_id
  scalable_dimension = aws_appautoscaling_target.api.scalable_dimension
  service_namespace  = aws_appautoscaling_target.api.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}
```

### SageMaker Training

```python
# scripts/sagemaker_training.py
import sagemaker
from sagemaker.pytorch import PyTorch

def launch_training_job(
    role: str,
    instance_type: str = "ml.p3.2xlarge",
    instance_count: int = 1,
):
    estimator = PyTorch(
        entry_point="training/train.py",
        source_dir=".",
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        framework_version="2.0.0",
        py_version="py310",
        hyperparameters={
            "model_name": "meta-llama/Llama-2-7b-chat-hf",
            "num_episodes": 1000,
            "learning_rate": 1e-5,
            "algorithm": "gspo",
        },
        environment={
            "WANDB_API_KEY": "your-wandb-key",
        },
        metric_definitions=[
            {"Name": "train:loss", "Regex": "train_loss: ([0-9\\.]+)"},
            {"Name": "train:reward", "Regex": "train_reward: ([0-9\\.]+)"},
        ],
    )

    estimator.fit({"training": "s3://your-bucket/training-data/"})
    return estimator
```

### EC2 GPU Instance

```bash
#!/bin/bash
# scripts/ec2_setup.sh

# Install NVIDIA drivers
sudo apt-get update
sudo apt-get install -y nvidia-driver-535

# Install Docker with GPU support
curl -fsSL https://get.docker.com | sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Run vLLM (GPU inference, OpenAI-compatible)
docker run -d --gpus all --network host \
  vllm/vllm-openai:nightly \
  python3 -m vllm.entrypoints.openai.api_server \
    --model moonshotai/Kimi-K2.5 \
    --host 0.0.0.0 \
    --port 8001 \
    --trust-remote-code

# Run StateSet Agents gateway (Anthropic/OpenAI compatible)
docker run -d --network host \
  -e API_JWT_SECRET=$API_JWT_SECRET \
  -e API_ENVIRONMENT=production \
  -e INFERENCE_BACKEND=vllm \
  -e INFERENCE_BACKEND_URL=http://127.0.0.1:8001 \
  stateset/stateset-agents-api:latest
```

---

## GCP Deployment

### Cloud Run

```yaml
# cloudrun/service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: stateset-api
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
        - image: gcr.io/PROJECT_ID/stateset-agents-api:latest
          ports:
            - containerPort: 8000
          resources:
            limits:
              cpu: "2"
              memory: 4Gi
          env:
            - name: API_ENVIRONMENT
              value: production
            - name: API_JWT_SECRET
              valueFrom:
                secretKeyRef:
                  name: jwt-secret
                  key: latest
```

```bash
# Deploy to Cloud Run
gcloud run deploy stateset-api \
  --image gcr.io/$PROJECT_ID/stateset-agents-api:latest \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --set-secrets API_JWT_SECRET=jwt-secret:latest
```

### Vertex AI Training

```python
# scripts/vertex_training.py
from google.cloud import aiplatform

def launch_vertex_training(
    project: str,
    location: str = "us-central1",
    machine_type: str = "n1-standard-8",
    accelerator_type: str = "NVIDIA_TESLA_V100",
):
    aiplatform.init(project=project, location=location)

    job = aiplatform.CustomContainerTrainingJob(
        display_name="stateset-grpo-training",
        container_uri=f"gcr.io/{project}/stateset-agents-trainer:latest",
        command=["bash", "-lc", "python examples/finetune_kimi_k25_gspo.py --model moonshotai/Kimi-K2.5 --task customer_service --use-lora --export-merged --output-dir /models/kimi-k25"],
        model_serving_container_image_uri=f"gcr.io/{project}/stateset-agents-api:latest",
    )

    model = job.run(
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=1,
        args=[
            "--model_name=meta-llama/Llama-2-7b-chat-hf",
            "--algorithm=gspo",
            "--num_episodes=1000",
        ],
    )

    return model
```

### GKE with GPU

For large-model GPU inference, deploy a dedicated model server (vLLM) and point
the gateway at it.

Recommended manifests:
- vLLM: `deployment/kubernetes/kimi-k25-vllm.yaml`
- Gateway: `deployment/kubernetes/production-deployment.yaml`

Helm alternative:
- Chart: `deployment/helm/stateset-agents`
- GPU profiles: `deployment/helm/stateset-agents/values-a100.yaml`,
  `deployment/helm/stateset-agents/values-h100.yaml`,
  `deployment/helm/stateset-agents/values-b200.yaml`

---

## Azure Deployment

### Azure Container Instances

```bash
# Deploy to ACI
az container create \
  --resource-group stateset-rg \
  --name stateset-api \
  --image statesetacr.azurecr.io/stateset-agents-api:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables API_ENVIRONMENT=production \
  --secure-environment-variables API_JWT_SECRET=$JWT_SECRET \
  --dns-name-label stateset-api
```

### Azure ML Training

```python
# scripts/azure_ml_training.py
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, AmlCompute
from azure.identity import DefaultAzureCredential

def launch_azure_training(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
):
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id,
        resource_group,
        workspace_name,
    )

    # Create compute cluster
    gpu_cluster = AmlCompute(
        name="gpu-cluster",
        size="Standard_NC6s_v3",
        min_instances=0,
        max_instances=4,
    )
    ml_client.compute.begin_create_or_update(gpu_cluster)

    # Submit training job
    job = command(
        code=".",
        command="python -m training.train --algorithm gspo --num_episodes 1000",
        environment="stateset-training@latest",
        compute="gpu-cluster",
        display_name="stateset-grpo-training",
    )

    returned_job = ml_client.jobs.create_or_update(job)
    return returned_job
```

### AKS with GPU

```yaml
# k8s/aks-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stateset-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: stateset-api
  template:
    metadata:
      labels:
        app: stateset-api
    spec:
      nodeSelector:
        accelerator: nvidia
      tolerations:
        - key: "sku"
          operator: "Equal"
          value: "gpu"
          effect: "NoSchedule"
      containers:
        - name: vllm
          image: vllm/vllm-openai:nightly
          resources:
            limits:
              nvidia.com/gpu: 1
          ports:
            - containerPort: 8000
```

---

## Docker Configuration

### docker-compose.yml (Local Development)

Use the repo compose files in `deployment/docker/`:
- `deployment/docker/docker-compose.yml`: runtime gateway + optional deps (redis/postgres/prometheus)
- `deployment/docker/docker-compose.dev.yml`: minimal dev gateway + test container

Examples:

```bash
docker compose -f deployment/docker/docker-compose.yml up -d stateset-agents-api
docker compose -f deployment/docker/docker-compose.yml --profile monitoring up -d prometheus
```

### Production Dockerfile

The repo includes a production gateway Dockerfile at `deployment/docker/Dockerfile`.

Build the gateway image:

```bash
docker build -f deployment/docker/Dockerfile --target runtime -t stateset/stateset-agents-api:latest .
```

Run the gateway locally (proxying to a vLLM backend):

```bash
docker run --rm -p 8000:8000 \
  -e API_ENVIRONMENT=production \
  -e API_JWT_SECRET="$API_JWT_SECRET" \
  -e INFERENCE_BACKEND=vllm \
  -e INFERENCE_BACKEND_URL="http://localhost:8001" \
  -e INFERENCE_DEFAULT_MODEL="moonshotai/Kimi-K2.5" \
  stateset/stateset-agents-api:latest
```

Note: GPU model serving should run in a dedicated model-server container (for example, vLLM) and the gateway should point `INFERENCE_BACKEND_URL` at that service.

---

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `API_ENVIRONMENT` | Environment (development/staging/production) | development | Yes |
| `API_JWT_SECRET` | JWT signing secret (required in production) | - | Yes |
| `API_REQUIRE_AUTH` | Require auth for protected endpoints | true | No |
| `API_CORS_ORIGINS` | Allowed CORS origins (`*` only allowed in development) | - | No |
| `API_RATE_LIMIT_ENABLED` | Enable rate limiting | true | No |
| `WANDB_API_KEY` | Weights & Biases API key | - | No |
| `HUGGING_FACE_HUB_TOKEN` | Hugging Face token (used by training jobs / vLLM model downloads) | - | No |
| `REDIS_URL` | Redis connection URL | - | No |
| `INFERENCE_BACKEND` | Inference backend type (`vllm` or `stub`) | vllm | No |
| `INFERENCE_BACKEND_URL` | Base URL for the inference backend | http://localhost:8000 | No |
| `INFERENCE_DEFAULT_MODEL` | Default model id when request omits `model` | - | No |
| `INFERENCE_TIMEOUT_SECONDS` | Backend request timeout (seconds) | 120 | No |
| `INFERENCE_MAX_RETRIES` | Backend retry count | 2 | No |
| `INFERENCE_MODEL_MAP` | Optional model id remapping (`alias=id` pairs) | - | No |

---

## Cost Optimization

### AWS
- Use Spot Instances for training (70% savings)
- Enable S3 Intelligent-Tiering for model storage
- Use Reserved Capacity for production API servers
- Set up auto-scaling with appropriate cooldown periods

### GCP
- Use Preemptible VMs for training (80% savings)
- Enable Coldline storage for archived models
- Use Committed Use Discounts for production
- Configure Cloud Run min instances carefully

### Azure
- Use Spot VMs for training (90% savings)
- Enable Cool/Archive tiers for Blob Storage
- Use Reserved Instances for production AKS nodes
- Leverage Azure Hybrid Benefit if applicable

### General Tips
1. **Right-size instances** - Start small, scale based on metrics
2. **Use spot/preemptible** - For fault-tolerant training workloads
3. **Enable auto-scaling** - Scale down during off-peak hours
4. **Monitor costs** - Set up billing alerts at 50%, 80%, 100%
5. **Clean up resources** - Delete unused models, logs, and checkpoints

---

## Monitoring & Observability

### Prometheus Metrics

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'stateset-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
```

### CloudWatch (AWS)

```hcl
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "stateset-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
```

---

## Security Best Practices

1. **Secrets Management** - Never commit secrets; use cloud secret managers
2. **Network Isolation** - Deploy API in private subnets with ALB/NLB
3. **TLS Everywhere** - Enable HTTPS with managed certificates
4. **IAM Least Privilege** - Grant minimal required permissions
5. **Container Security** - Scan images, use non-root users
6. **API Security** - Enable rate limiting, authentication, input validation

---

## Quick Start Commands

```bash
# AWS
terraform init && terraform apply -var-file=production.tfvars

# GCP
gcloud run deploy stateset-api --source .

# Azure
az deployment group create --template-file azure/main.bicep

# Local
docker compose -f deployment/docker/docker-compose.yml up -d stateset-agents-api
```

---

For detailed API documentation, see [API_EXAMPLES.md](./API_EXAMPLES.md).
For performance benchmarks, see [BENCHMARKS.md](./BENCHMARKS.md).
