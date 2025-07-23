# GRPO Agent Framework - Deployment Guide

This directory contains comprehensive deployment configurations and scripts for the GRPO Agent Framework. The framework supports multiple deployment scenarios from local development to production cloud environments.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Deployment Options](#deployment-options)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployments](#cloud-deployments)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Prerequisites

- **Docker**: Version 20.0+ with Docker Compose
- **Kubernetes**: kubectl and Helm (for K8s deployments)
- **Cloud CLI**: AWS CLI or gcloud (for cloud deployments)
- **Terraform**: Version 1.0+ (for infrastructure as code)

### 1-Minute Docker Deployment

```bash
# Clone the repository
git clone https://github.com/your-org/grpo-agent-framework.git
cd grpo-agent-framework

# Deploy with Docker Compose
./deployment/scripts/deploy.sh docker production

# Check deployment health
./deployment/scripts/health_check.sh docker localhost 8001 8002
```

Your GRPO Agent Framework will be running at:
- **API**: http://localhost:8001
- **Training API**: http://localhost:8002
- **Grafana Dashboard**: http://localhost:3000 (admin/grpo_grafana)

## 🎯 Deployment Options

| Deployment Type | Use Case | Complexity | Scalability |
|----------------|----------|------------|-------------|
| **Docker** | Local development, testing | Low | Limited |
| **Kubernetes** | Production, multi-environment | Medium | High |
| **AWS EKS** | AWS-native production | High | Very High |
| **Google GKE** | GCP-native production | High | Very High |

## 🐳 Docker Deployment

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ GRPO Framework  │    │ PostgreSQL      │    │ Redis           │
│ (API + Training)│────│ (Conversations) │    │ (Cache/Session) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Prometheus      │    │ Grafana         │    │ Load Balancer   │
│ (Metrics)       │────│ (Dashboards)    │    │ (nginx)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Configuration

Edit `deployment/docker/docker-compose.yml` to customize:

```yaml
environment:
  - GRPO_WORKERS=4              # Number of worker processes
  - GRPO_ENABLE_MONITORING=true # Enable metrics collection
  - GRPO_CACHE_SIZE=1000       # Cache size for conversations
  - GRPO_LOG_LEVEL=INFO        # Logging level
```

### Commands

```bash
# Deploy
./deployment/scripts/deploy.sh docker production

# Check logs
docker-compose -f deployment/docker/docker-compose.yml logs -f grpo-framework

# Scale services
docker-compose -f deployment/docker/docker-compose.yml up -d --scale grpo-framework=3

# Stop services
docker-compose -f deployment/docker/docker-compose.yml down

# Cleanup
./deployment/scripts/deploy.sh docker production cleanup
```

## ☸️ Kubernetes Deployment

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                    │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Ingress         │    │ Load Balancer   │                │
│  │ (nginx/traefik) │────│ (External)      │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ GRPO Framework  │    │ HPA             │                │
│  │ (3 replicas)    │────│ (Auto-scaling)  │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ PostgreSQL      │    │ Redis           │                │
│  │ (StatefulSet)   │    │ (Deployment)    │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Persistent      │    │ ConfigMaps      │                │
│  │ Volumes         │    │ & Secrets       │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify connection to cluster
kubectl cluster-info
```

### Configuration

Update `deployment/kubernetes/configmap.yaml`:

```yaml
data:
  WORKERS: "8"                    # Increase for production
  TRAINING_BATCH_SIZE: "64"       # Larger batches for efficiency
  NUM_WORKERS: "16"               # More computational workers
  MAX_CONVERSATION_TURNS: "50"    # Longer conversations
```

Update secrets in `deployment/kubernetes/secret.yaml`:

```bash
# Base64 encode your API keys
echo -n "your-openai-api-key" | base64
echo -n "your-anthropic-api-key" | base64
```

### Commands

```bash
# Deploy to existing cluster
./deployment/scripts/deploy.sh kubernetes production

# Check deployment status
kubectl get pods -n grpo-framework -w

# Scale deployment
kubectl scale deployment grpo-framework --replicas=5 -n grpo-framework

# View logs
kubectl logs -f deployment/grpo-framework -n grpo-framework

# Port forward for local access
kubectl port-forward service/grpo-framework-service 8001:8001 -n grpo-framework

# Cleanup
./deployment/scripts/deploy.sh kubernetes production cleanup
```

## ☁️ Cloud Deployments

### AWS EKS

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                           AWS                                │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Application     │    │ Elastic Load    │                │
│  │ Load Balancer   │────│ Balancer        │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ EKS Cluster     │    │ Auto Scaling    │                │
│  │ (3 AZs)         │────│ Group           │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ RDS PostgreSQL  │    │ ElastiCache     │                │
│  │ (Multi-AZ)      │    │ Redis           │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ EBS Volumes     │    │ VPC & Security  │                │
│  │ (Encrypted)     │    │ Groups          │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

#### Prerequisites

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

# Configure AWS credentials
aws configure

# Install Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip && sudo mv terraform /usr/local/bin/
```

#### Configuration

Update `deployment/cloud/aws/terraform/main.tf`:

```hcl
variable "node_group_instance_types" {
  default = ["m5.xlarge", "m5.2xlarge"]  # Larger instances for production
}

variable "node_group_scaling_config" {
  default = {
    desired_size = 5    # More nodes for high availability
    max_size     = 20   # Higher max for auto-scaling
    min_size     = 3    # Minimum for availability
  }
}
```

#### Commands

```bash
# Deploy infrastructure and application
./deployment/scripts/deploy.sh aws production

# Check cluster status
aws eks describe-cluster --name grpo-framework

# Update kubeconfig
aws eks update-kubeconfig --name grpo-framework

# View resources
kubectl get all -n grpo-framework

# Scale cluster
aws eks update-nodegroup-config --cluster-name grpo-framework \
  --nodegroup-name grpo-framework-node-group \
  --scaling-config minSize=5,maxSize=30,desiredSize=10

# Cleanup
./deployment/scripts/deploy.sh aws production cleanup
```

### Google GKE

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          GCP                                 │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Global Load     │    │ Cloud Armor     │                │
│  │ Balancer        │────│ (WAF)           │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ GKE Cluster     │    │ Node Auto       │                │
│  │ (Multi-zone)    │────│ Scaling         │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Cloud SQL       │    │ Cloud Memorystore│                │
│  │ PostgreSQL      │    │ Redis           │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Persistent      │    │ Cloud Storage   │                │
│  │ Disks           │    │ (Models/Data)   │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

#### Prerequisites

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login
gcloud config set project YOUR-PROJECT-ID

# Install kubectl via gcloud
gcloud components install kubectl
```

#### Configuration

Update `deployment/cloud/gcp/terraform/main.tf`:

```hcl
variable "machine_type" {
  default = "e2-standard-8"  # Larger instances for production
}

variable "node_count" {
  default = 5                # More nodes for high availability
}
```

#### Commands

```bash
# Deploy infrastructure and application
./deployment/scripts/deploy.sh gcp production

# Check cluster status
gcloud container clusters describe grpo-framework --zone us-central1-a

# Get cluster credentials
gcloud container clusters get-credentials grpo-framework --zone us-central1-a

# View resources
kubectl get all -n grpo-framework

# Scale cluster
gcloud container clusters resize grpo-framework --size=10 --zone us-central1-a

# Cleanup
./deployment/scripts/deploy.sh gcp production cleanup
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `GRPO_LOG_LEVEL` | Logging level | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `GRPO_WORKERS` | Number of worker processes | `4` | `8`, `16` |
| `GRPO_ENABLE_MONITORING` | Enable metrics collection | `true` | `true`, `false` |
| `GRPO_CACHE_SIZE` | Cache size for conversations | `1000` | `5000`, `10000` |
| `OPENAI_API_KEY` | OpenAI API key | `""` | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `""` | `sk-ant-...` |
| `DATABASE_URL` | PostgreSQL connection string | `""` | `postgresql://user:pass@host:5432/db` |
| `REDIS_URL` | Redis connection string | `""` | `redis://host:6379` |

### Resource Requirements

#### Minimum Requirements (Development)

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

#### Recommended Requirements (Production)

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

#### High-Performance Requirements (Enterprise)

```yaml
resources:
  requests:
    memory: "8Gi"
    cpu: "4000m"
  limits:
    memory: "32Gi"
    cpu: "16000m"
```

## 📊 Monitoring

### Metrics Available

- **Request Metrics**: Total requests, request rate, response times
- **Conversation Metrics**: Active conversations, conversation length, user satisfaction
- **Training Metrics**: Training iterations, reward scores, model performance
- **System Metrics**: CPU usage, memory usage, disk I/O, network I/O
- **Business Metrics**: User engagement, feature usage, error rates

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (Docker) or via ingress (Kubernetes):

1. **Overview Dashboard**: High-level system metrics
2. **API Performance**: Request/response metrics
3. **Training Dashboard**: Model training progress
4. **Infrastructure**: System resource utilization
5. **Business Metrics**: User engagement and feature usage

### Alerting

Configure alerts in `deployment/docker/prometheus.yml`:

```yaml
groups:
  - name: grpo-framework
    rules:
      - alert: HighResponseTime
        expr: avg(grpo_response_time_seconds) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          
      - alert: HighErrorRate
        expr: rate(grpo_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Service Not Starting

```bash
# Check logs
docker-compose logs grpo-framework
# or
kubectl logs -f deployment/grpo-framework -n grpo-framework

# Common causes:
# - Missing API keys
# - Database connection issues
# - Insufficient resources
```

#### 2. High Memory Usage

```bash
# Check resource usage
docker stats
# or
kubectl top pods -n grpo-framework

# Solutions:
# - Increase memory limits
# - Reduce conversation cache size
# - Optimize batch sizes
```

#### 3. Database Connection Errors

```bash
# Check database status
docker exec grpo-postgres pg_isready
# or
kubectl exec -it postgres-pod -- pg_isready

# Solutions:
# - Verify connection strings
# - Check network connectivity
# - Ensure database is running
```

#### 4. Training Performance Issues

```bash
# Check training metrics
curl http://localhost:8002/metrics

# Solutions:
# - Increase computational workers
# - Optimize batch sizes
# - Use GPU instances
# - Enable distributed training
```

### Health Checks

```bash
# Run comprehensive health check
./deployment/scripts/health_check.sh docker localhost 8001 8002

# Check specific endpoints
curl http://localhost:8001/health
curl http://localhost:8001/ready
curl http://localhost:8002/metrics
```

### Performance Tuning

#### For High Throughput

```yaml
environment:
  - GRPO_WORKERS=16
  - GRPO_CACHE_SIZE=10000
  - TRAINING_BATCH_SIZE=128
  - NUM_WORKERS=32
```

#### For Low Latency

```yaml
environment:
  - GRPO_WORKERS=8
  - GRPO_CACHE_SIZE=5000
  - TRAINING_BATCH_SIZE=32
  - NUM_WORKERS=16
```

#### For Memory Efficiency

```yaml
environment:
  - GRPO_WORKERS=4
  - GRPO_CACHE_SIZE=1000
  - TRAINING_BATCH_SIZE=16
  - NUM_WORKERS=8
```

## 🔐 Security

### API Keys

Store API keys securely:

```bash
# Docker
echo "OPENAI_API_KEY=sk-your-key" >> .env

# Kubernetes
kubectl create secret generic grpo-secrets \
  --from-literal=OPENAI_API_KEY=sk-your-key \
  -n grpo-framework
```

### Network Security

- Use HTTPS in production
- Configure firewall rules
- Enable VPC/network policies
- Use private subnets for databases

### Access Control

- Implement API authentication
- Use RBAC for Kubernetes
- Enable audit logging
- Regular security updates

## 📈 Scaling

### Horizontal Scaling

```bash
# Docker
docker-compose up -d --scale grpo-framework=5

# Kubernetes
kubectl scale deployment grpo-framework --replicas=10 -n grpo-framework

# Auto-scaling (Kubernetes)
kubectl autoscale deployment grpo-framework --cpu-percent=70 --min=3 --max=20 -n grpo-framework
```

### Vertical Scaling

```yaml
# Increase resource limits
resources:
  limits:
    memory: "16Gi"
    cpu: "8000m"
```

### Database Scaling

```bash
# Read replicas
# AWS RDS
aws rds create-db-instance-read-replica

# Google Cloud SQL
gcloud sql instances create read-replica
```

## 📚 Additional Resources

- [GRPO Framework Documentation](../README.md)
- [API Reference](../api/README.md)
- [Training Guide](../training/README.md)
- [Monitoring Guide](../monitoring/README.md)
- [Security Best Practices](../security/README.md)

## 🆘 Support

For deployment issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs and metrics
3. Run health checks
4. Search existing issues on GitHub
5. Create a new issue with deployment logs

---

**Happy Deploying! 🚀**

The GRPO Agent Framework is designed to scale with your computational resources. Whether you're running on a laptop or a multi-region cloud deployment, the framework will adapt to maximize performance.