# GRPO Agent Framework - GCP Infrastructure
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Variables
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "grpo-framework"
}

variable "node_count" {
  description = "Number of nodes in the GKE cluster"
  type        = number
  default     = 3
}

variable "machine_type" {
  description = "Machine type for GKE nodes"
  type        = string
  default     = "e2-standard-4"
}

# Enable required APIs
resource "google_project_service" "gke_api" {
  service = "container.googleapis.com"
}

resource "google_project_service" "compute_api" {
  service = "compute.googleapis.com"
}

resource "google_project_service" "sql_api" {
  service = "sqladmin.googleapis.com"
}

resource "google_project_service" "redis_api" {
  service = "redis.googleapis.com"
}

# VPC Network
resource "google_compute_network" "grpo_vpc" {
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "grpo_subnet" {
  name          = "${var.cluster_name}-subnet"
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.grpo_vpc.name

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/16"
  }
}

# Firewall Rules
resource "google_compute_firewall" "grpo_firewall" {
  name    = "${var.cluster_name}-firewall"
  network = google_compute_network.grpo_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8001", "8002"]
  }

  source_ranges = ["0.0.0.0/0"]
}

# Service Account for GKE
resource "google_service_account" "grpo_sa" {
  account_id   = "${var.cluster_name}-sa"
  display_name = "GRPO Framework Service Account"
}

resource "google_project_iam_binding" "grpo_sa_binding" {
  project = var.project_id
  role    = "roles/container.developer"

  members = [
    "serviceAccount:${google_service_account.grpo_sa.email}",
  ]
}

# GKE Cluster
resource "google_container_cluster" "grpo_cluster" {
  name     = var.cluster_name
  location = var.zone

  network    = google_compute_network.grpo_vpc.name
  subnetwork = google_compute_subnetwork.grpo_subnet.name

  # Remove default node pool
  remove_default_node_pool = true
  initial_node_count       = 1

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  depends_on = [
    google_project_service.gke_api,
    google_project_service.compute_api,
  ]
}

# GKE Node Pool
resource "google_container_node_pool" "grpo_node_pool" {
  name       = "${var.cluster_name}-node-pool"
  cluster    = google_container_cluster.grpo_cluster.name
  location   = var.zone
  node_count = var.node_count

  node_config {
    machine_type = var.machine_type
    disk_size_gb = 50
    disk_type    = "pd-standard"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    service_account = google_service_account.grpo_sa.email

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 10
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# Cloud SQL PostgreSQL Instance
resource "google_sql_database_instance" "grpo_postgres" {
  name             = "${var.cluster_name}-postgres"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier = "db-f1-micro"
    
    backup_configuration {
      enabled    = true
      start_time = "03:00"
    }

    ip_configuration {
      ipv4_enabled    = true
      private_network = google_compute_network.grpo_vpc.id
    }

    database_flags {
      name  = "max_connections"
      value = "200"
    }
  }

  depends_on = [google_project_service.sql_api]
}

resource "google_sql_database" "grpo_db" {
  name     = "grpo_framework"
  instance = google_sql_database_instance.grpo_postgres.name
}

resource "google_sql_user" "grpo_user" {
  name     = "grpo"
  instance = google_sql_database_instance.grpo_postgres.name
  password = "grpo_password_change_me"
}

# Cloud Memorystore Redis Instance
resource "google_redis_instance" "grpo_redis" {
  name           = "${var.cluster_name}-redis"
  memory_size_gb = 1
  region         = var.region

  authorized_network = google_compute_network.grpo_vpc.id

  depends_on = [google_project_service.redis_api]
}

# Cloud Storage Bucket for Models
resource "google_storage_bucket" "grpo_models" {
  name     = "${var.project_id}-${var.cluster_name}-models"
  location = var.region

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 30
    }
  }
}

# Cloud Storage Bucket for Data
resource "google_storage_bucket" "grpo_data" {
  name     = "${var.project_id}-${var.cluster_name}-data"
  location = var.region

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 90
    }
  }
}

# Outputs
output "cluster_name" {
  value = google_container_cluster.grpo_cluster.name
}

output "cluster_location" {
  value = google_container_cluster.grpo_cluster.location
}

output "cluster_endpoint" {
  value = google_container_cluster.grpo_cluster.endpoint
}

output "postgres_connection_name" {
  value = google_sql_database_instance.grpo_postgres.connection_name
}

output "redis_host" {
  value = google_redis_instance.grpo_redis.host
}

output "redis_port" {
  value = google_redis_instance.grpo_redis.port
}

output "models_bucket" {
  value = google_storage_bucket.grpo_models.name
}

output "data_bucket" {
  value = google_storage_bucket.grpo_data.name
}