#!/bin/bash

# GRPO Agent Framework - Health Check Script
# This script performs comprehensive health checks on the deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_TYPE=${1:-"docker"}
API_HOST=${2:-"localhost"}
API_PORT=${3:-"8001"}
TRAINING_PORT=${4:-"8002"}

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

check_http_endpoint() {
    local url=$1
    local expected_code=${2:-200}
    local timeout=${3:-10}
    
    local response_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $timeout "$url" || echo "000")
    
    if [ "$response_code" = "$expected_code" ]; then
        log_success "✓ $url (HTTP $response_code)"
        return 0
    else
        log_error "✗ $url (HTTP $response_code, expected $expected_code)"
        return 1
    fi
}

check_api_functionality() {
    local base_url="http://$API_HOST:$API_PORT"
    local health_check_passed=true
    
    log_info "Checking API functionality..."
    
    # Health check endpoint
    if ! check_http_endpoint "$base_url/health" 200; then
        health_check_passed=false
    fi
    
    # Readiness check endpoint
    if ! check_http_endpoint "$base_url/ready" 200; then
        health_check_passed=false
    fi
    
    # Metrics endpoint
    if ! check_http_endpoint "$base_url/metrics" 200; then
        health_check_passed=false
    fi
    
    # API documentation
    if ! check_http_endpoint "$base_url/docs" 200; then
        health_check_passed=false
    fi
    
    # Chat API endpoint (POST)
    local chat_response=$(curl -s -X POST "$base_url/api/chat" \
        -H "Content-Type: application/json" \
        -d '{"message": "Hello, this is a health check", "strategy": "default"}' \
        -w "%{http_code}" || echo "000")
    
    if [[ "$chat_response" == *"200" ]]; then
        log_success "✓ Chat API endpoint"
    else
        log_error "✗ Chat API endpoint (Response: $chat_response)"
        health_check_passed=false
    fi
    
    return $health_check_passed
}

check_training_api() {
    local base_url="http://$API_HOST:$TRAINING_PORT"
    local training_check_passed=true
    
    log_info "Checking Training API functionality..."
    
    # Training health check
    if ! check_http_endpoint "$base_url/health" 200; then
        training_check_passed=false
    fi
    
    # Training metrics
    if ! check_http_endpoint "$base_url/metrics" 200; then
        training_check_passed=false
    fi
    
    # Start training endpoint (POST)
    local train_response=$(curl -s -X POST "$base_url/api/train" \
        -H "Content-Type: application/json" \
        -d '{"prompts": ["Health check prompt"], "strategy": "computational", "num_iterations": 1}' \
        -w "%{http_code}" || echo "000")
    
    if [[ "$train_response" == *"200" ]]; then
        log_success "✓ Training API endpoint"
    else
        log_error "✗ Training API endpoint (Response: $train_response)"
        training_check_passed=false
    fi
    
    return $training_check_passed
}

check_docker_deployment() {
    log_info "Checking Docker deployment..."
    
    local docker_check_passed=true
    
    # Check if containers are running
    local containers=("grpo-framework" "grpo-redis" "grpo-postgres" "grpo-monitoring" "grpo-grafana")
    
    for container in "${containers[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "$container"; then
            log_success "✓ Container $container is running"
        else
            log_error "✗ Container $container is not running"
            docker_check_passed=false
        fi
    done
    
    # Check container health
    local unhealthy_containers=$(docker ps --filter "health=unhealthy" --format "{{.Names}}")
    if [ -n "$unhealthy_containers" ]; then
        log_error "✗ Unhealthy containers: $unhealthy_containers"
        docker_check_passed=false
    else
        log_success "✓ All containers are healthy"
    fi
    
    return $docker_check_passed
}

check_kubernetes_deployment() {
    log_info "Checking Kubernetes deployment..."
    
    local k8s_check_passed=true
    local namespace="grpo-framework"
    
    # Check namespace
    if kubectl get namespace "$namespace" &> /dev/null; then
        log_success "✓ Namespace $namespace exists"
    else
        log_error "✗ Namespace $namespace does not exist"
        k8s_check_passed=false
        return $k8s_check_passed
    fi
    
    # Check deployment
    if kubectl get deployment grpo-framework -n "$namespace" &> /dev/null; then
        local ready_replicas=$(kubectl get deployment grpo-framework -n "$namespace" -o jsonpath='{.status.readyReplicas}')
        local desired_replicas=$(kubectl get deployment grpo-framework -n "$namespace" -o jsonpath='{.spec.replicas}')
        
        if [ "$ready_replicas" = "$desired_replicas" ]; then
            log_success "✓ Deployment grpo-framework ($ready_replicas/$desired_replicas replicas ready)"
        else
            log_error "✗ Deployment grpo-framework ($ready_replicas/$desired_replicas replicas ready)"
            k8s_check_passed=false
        fi
    else
        log_error "✗ Deployment grpo-framework not found"
        k8s_check_passed=false
    fi
    
    # Check services
    if kubectl get service grpo-framework-service -n "$namespace" &> /dev/null; then
        log_success "✓ Service grpo-framework-service exists"
    else
        log_error "✗ Service grpo-framework-service not found"
        k8s_check_passed=false
    fi
    
    # Check pods
    local pod_count=$(kubectl get pods -n "$namespace" --field-selector=status.phase=Running | wc -l)
    if [ "$pod_count" -gt 1 ]; then
        log_success "✓ $((pod_count - 1)) pods running"
    else
        log_error "✗ No running pods found"
        k8s_check_passed=false
    fi
    
    return $k8s_check_passed
}

check_database_connectivity() {
    log_info "Checking database connectivity..."
    
    local db_check_passed=true
    
    # Check PostgreSQL connectivity
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        if docker exec grpo-postgres pg_isready -U grpo &> /dev/null; then
            log_success "✓ PostgreSQL is ready"
        else
            log_error "✗ PostgreSQL is not ready"
            db_check_passed=false
        fi
        
        # Check Redis connectivity
        if docker exec grpo-redis redis-cli ping | grep -q "PONG"; then
            log_success "✓ Redis is ready"
        else
            log_error "✗ Redis is not ready"
            db_check_passed=false
        fi
    else
        # For Kubernetes/Cloud deployments, check via API
        local db_status=$(curl -s "http://$API_HOST:$API_PORT/health" | grep -o '"database":"[^"]*"' | cut -d'"' -f4)
        if [ "$db_status" = "healthy" ]; then
            log_success "✓ Database connectivity through API"
        else
            log_error "✗ Database connectivity issues"
            db_check_passed=false
        fi
    fi
    
    return $db_check_passed
}

check_monitoring_stack() {
    log_info "Checking monitoring stack..."
    
    local monitoring_check_passed=true
    
    # Check Prometheus
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        if check_http_endpoint "http://localhost:9090/-/healthy" 200; then
            log_success "✓ Prometheus is healthy"
        else
            log_error "✗ Prometheus is not healthy"
            monitoring_check_passed=false
        fi
        
        # Check Grafana
        if check_http_endpoint "http://localhost:3000/api/health" 200; then
            log_success "✓ Grafana is healthy"
        else
            log_error "✗ Grafana is not healthy"
            monitoring_check_passed=false
        fi
    fi
    
    return $monitoring_check_passed
}

check_performance_metrics() {
    log_info "Checking performance metrics..."
    
    local metrics_url="http://$API_HOST:$API_PORT/metrics"
    local metrics_response=$(curl -s "$metrics_url" || echo "")
    
    if [ -n "$metrics_response" ]; then
        log_success "✓ Metrics endpoint is responding"
        
        # Check for specific metrics
        if echo "$metrics_response" | grep -q "grpo_requests_total"; then
            log_success "✓ Request metrics are available"
        else
            log_warning "⚠ Request metrics not found"
        fi
        
        if echo "$metrics_response" | grep -q "grpo_response_time"; then
            log_success "✓ Response time metrics are available"
        else
            log_warning "⚠ Response time metrics not found"
        fi
        
        if echo "$metrics_response" | grep -q "grpo_active_conversations"; then
            log_success "✓ Conversation metrics are available"
        else
            log_warning "⚠ Conversation metrics not found"
        fi
    else
        log_error "✗ Metrics endpoint is not responding"
        return 1
    fi
    
    return 0
}

run_load_test() {
    log_info "Running basic load test..."
    
    local base_url="http://$API_HOST:$API_PORT"
    local success_count=0
    local total_requests=10
    
    for i in $(seq 1 $total_requests); do
        local response=$(curl -s -X POST "$base_url/api/chat" \
            -H "Content-Type: application/json" \
            -d "{\"message\": \"Load test message $i\", \"strategy\": \"default\"}" \
            -w "%{http_code}" || echo "000")
        
        if [[ "$response" == *"200" ]]; then
            ((success_count++))
        fi
    done
    
    local success_rate=$((success_count * 100 / total_requests))
    
    if [ $success_rate -ge 90 ]; then
        log_success "✓ Load test passed ($success_count/$total_requests requests successful, $success_rate%)"
    else
        log_error "✗ Load test failed ($success_count/$total_requests requests successful, $success_rate%)"
        return 1
    fi
    
    return 0
}

show_help() {
    echo "GRPO Agent Framework Health Check Script"
    echo ""
    echo "Usage: $0 [DEPLOYMENT_TYPE] [API_HOST] [API_PORT] [TRAINING_PORT]"
    echo ""
    echo "DEPLOYMENT_TYPE:"
    echo "  docker      Check Docker Compose deployment (default)"
    echo "  kubernetes  Check Kubernetes deployment"
    echo "  aws         Check AWS EKS deployment"
    echo "  gcp         Check Google GKE deployment"
    echo ""
    echo "API_HOST:     API host address (default: localhost)"
    echo "API_PORT:     API port (default: 8001)"
    echo "TRAINING_PORT: Training API port (default: 8002)"
    echo ""
    echo "Examples:"
    echo "  $0 docker localhost 8001 8002"
    echo "  $0 kubernetes my-cluster.example.com 80 80"
    echo "  $0 aws my-aws-lb.amazonaws.com 443 443"
}

# Main execution
main() {
    if [ "$1" = "help" ]; then
        show_help
        exit 0
    fi
    
    log_info "Starting GRPO Agent Framework health check"
    log_info "Deployment type: $DEPLOYMENT_TYPE"
    log_info "API endpoint: http://$API_HOST:$API_PORT"
    log_info "Training endpoint: http://$API_HOST:$TRAINING_PORT"
    
    local overall_health=true
    
    # Deployment-specific checks
    case $DEPLOYMENT_TYPE in
        "docker")
            if ! check_docker_deployment; then
                overall_health=false
            fi
            ;;
        "kubernetes"|"aws"|"gcp")
            if ! check_kubernetes_deployment; then
                overall_health=false
            fi
            ;;
    esac
    
    # API functionality checks
    if ! check_api_functionality; then
        overall_health=false
    fi
    
    if ! check_training_api; then
        overall_health=false
    fi
    
    # Database connectivity
    if ! check_database_connectivity; then
        overall_health=false
    fi
    
    # Monitoring stack (for Docker deployments)
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        if ! check_monitoring_stack; then
            overall_health=false
        fi
    fi
    
    # Performance metrics
    if ! check_performance_metrics; then
        overall_health=false
    fi
    
    # Load test
    if ! run_load_test; then
        overall_health=false
    fi
    
    # Final result
    echo ""
    if [ "$overall_health" = true ]; then
        log_success "🎉 All health checks passed! GRPO Agent Framework is healthy."
        exit 0
    else
        log_error "❌ Some health checks failed. Please check the logs above."
        exit 1
    fi
}

# Run main function
main "$@"