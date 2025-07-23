#!/bin/bash

# GRPO Agent Framework - Deployment Script
# This script automates the deployment of the GRPO Agent Framework

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_TYPE=${1:-"docker"}
ENVIRONMENT=${2:-"production"}
NAMESPACE="grpo-framework"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    case $DEPLOYMENT_TYPE in
        "docker")
            if ! command -v docker &> /dev/null; then
                log_error "Docker is not installed"
                exit 1
            fi
            if ! command -v docker-compose &> /dev/null; then
                log_error "Docker Compose is not installed"
                exit 1
            fi
            ;;
        "kubernetes")
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is not installed"
                exit 1
            fi
            if ! command -v helm &> /dev/null; then
                log_error "Helm is not installed"
                exit 1
            fi
            ;;
        "aws")
            if ! command -v aws &> /dev/null; then
                log_error "AWS CLI is not installed"
                exit 1
            fi
            if ! command -v terraform &> /dev/null; then
                log_error "Terraform is not installed"
                exit 1
            fi
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is not installed"
                exit 1
            fi
            ;;
        "gcp")
            if ! command -v gcloud &> /dev/null; then
                log_error "gcloud CLI is not installed"
                exit 1
            fi
            if ! command -v terraform &> /dev/null; then
                log_error "Terraform is not installed"
                exit 1
            fi
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is not installed"
                exit 1
            fi
            ;;
        *)
            log_error "Unsupported deployment type: $DEPLOYMENT_TYPE"
            echo "Supported types: docker, kubernetes, aws, gcp"
            exit 1
            ;;
    esac
    
    log_success "Prerequisites check passed"
}

build_docker_image() {
    log_info "Building Docker image..."
    
    cd "$(dirname "$0")/../.."
    
    docker build -t grpo-framework:latest -f deployment/docker/Dockerfile .
    
    log_success "Docker image built successfully"
}

deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "$(dirname "$0")/../docker"
    
    # Create necessary directories
    mkdir -p data logs models
    
    # Pull latest images
    docker-compose pull
    
    # Build and start services
    docker-compose up -d --build
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    if docker-compose ps | grep -q "Up"; then
        log_success "Docker deployment completed successfully"
        log_info "Services are running at:"
        log_info "- GRPO Framework API: http://localhost:8001"
        log_info "- Training API: http://localhost:8002"
        log_info "- Grafana Dashboard: http://localhost:3000 (admin/grpo_grafana)"
        log_info "- Prometheus: http://localhost:9090"
    else
        log_error "Some services failed to start"
        docker-compose logs
        exit 1
    fi
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    cd "$(dirname "$0")/../kubernetes"
    
    # Create namespace
    kubectl apply -f namespace.yaml
    
    # Apply configurations
    kubectl apply -f configmap.yaml
    kubectl apply -f secret.yaml
    kubectl apply -f pvc.yaml
    
    # Deploy services
    kubectl apply -f deployment.yaml
    kubectl apply -f service.yaml
    
    # Setup ingress
    kubectl apply -f ingress.yaml
    
    # Setup auto-scaling
    kubectl apply -f hpa.yaml
    
    # Wait for deployment
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/grpo-framework -n $NAMESPACE
    
    # Get service information
    log_success "Kubernetes deployment completed successfully"
    log_info "Getting service information..."
    kubectl get services -n $NAMESPACE
    kubectl get pods -n $NAMESPACE
}

deploy_aws() {
    log_info "Deploying to AWS..."
    
    cd "$(dirname "$0")/../cloud/aws/terraform"
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -out=tfplan
    
    # Apply infrastructure
    log_info "Creating AWS infrastructure..."
    terraform apply tfplan
    
    # Get cluster information
    CLUSTER_NAME=$(terraform output -raw cluster_name)
    
    # Configure kubectl
    log_info "Configuring kubectl for EKS..."
    aws eks update-kubeconfig --name $CLUSTER_NAME
    
    # Deploy application
    cd ../../kubernetes
    deploy_kubernetes
    
    log_success "AWS deployment completed successfully"
    log_info "EKS cluster: $CLUSTER_NAME"
}

deploy_gcp() {
    log_info "Deploying to GCP..."
    
    cd "$(dirname "$0")/../cloud/gcp/terraform"
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -out=tfplan
    
    # Apply infrastructure
    log_info "Creating GCP infrastructure..."
    terraform apply tfplan
    
    # Get cluster information
    CLUSTER_NAME=$(terraform output -raw cluster_name)
    CLUSTER_LOCATION=$(terraform output -raw cluster_location)
    
    # Configure kubectl
    log_info "Configuring kubectl for GKE..."
    gcloud container clusters get-credentials $CLUSTER_NAME --zone $CLUSTER_LOCATION
    
    # Deploy application
    cd ../../kubernetes
    deploy_kubernetes
    
    log_success "GCP deployment completed successfully"
    log_info "GKE cluster: $CLUSTER_NAME"
}

cleanup() {
    log_info "Cleaning up deployment..."
    
    case $DEPLOYMENT_TYPE in
        "docker")
            cd "$(dirname "$0")/../docker"
            docker-compose down -v
            docker image prune -f
            ;;
        "kubernetes")
            kubectl delete namespace $NAMESPACE --ignore-not-found=true
            ;;
        "aws")
            cd "$(dirname "$0")/../cloud/aws/terraform"
            terraform destroy -auto-approve
            ;;
        "gcp")
            cd "$(dirname "$0")/../cloud/gcp/terraform"
            terraform destroy -auto-approve
            ;;
    esac
    
    log_success "Cleanup completed"
}

show_help() {
    echo "GRPO Agent Framework Deployment Script"
    echo ""
    echo "Usage: $0 [DEPLOYMENT_TYPE] [ENVIRONMENT] [COMMAND]"
    echo ""
    echo "DEPLOYMENT_TYPE:"
    echo "  docker      Deploy using Docker Compose (default)"
    echo "  kubernetes  Deploy to Kubernetes cluster"
    echo "  aws         Deploy to AWS EKS"
    echo "  gcp         Deploy to Google GKE"
    echo ""
    echo "ENVIRONMENT:"
    echo "  production  Production environment (default)"
    echo "  staging     Staging environment"
    echo "  development Development environment"
    echo ""
    echo "COMMANDS:"
    echo "  deploy      Deploy the application (default)"
    echo "  cleanup     Clean up deployment"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 docker production"
    echo "  $0 kubernetes staging"
    echo "  $0 aws production cleanup"
    echo "  $0 gcp production"
}

# Main execution
main() {
    local command=${3:-"deploy"}
    
    case $command in
        "deploy")
            log_info "Starting GRPO Agent Framework deployment"
            log_info "Deployment type: $DEPLOYMENT_TYPE"
            log_info "Environment: $ENVIRONMENT"
            
            check_prerequisites
            
            case $DEPLOYMENT_TYPE in
                "docker")
                    build_docker_image
                    deploy_docker
                    ;;
                "kubernetes")
                    deploy_kubernetes
                    ;;
                "aws")
                    deploy_aws
                    ;;
                "gcp")
                    deploy_gcp
                    ;;
            esac
            
            log_success "Deployment completed successfully!"
            ;;
        "cleanup")
            cleanup
            ;;
        "help")
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"