# GRPO Agent Framework - AWS Infrastructure
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "grpo-framework"
}

variable "node_group_instance_types" {
  description = "EC2 instance types for EKS node group"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge"]
}

variable "node_group_scaling_config" {
  description = "Scaling configuration for EKS node group"
  type = object({
    desired_size = number
    max_size     = number
    min_size     = number
  })
  default = {
    desired_size = 3
    max_size     = 10
    min_size     = 1
  }
}

# Database credentials (do not set defaults for sensitive values)
variable "db_username" {
  description = "RDS PostgreSQL username"
  type        = string
  default     = "grpo"
}

variable "db_password" {
  description = "RDS PostgreSQL password"
  type        = string
  sensitive   = true
}

# VPC
resource "aws_vpc" "grpo_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.cluster_name}-vpc"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "grpo_igw" {
  vpc_id = aws_vpc.grpo_vpc.id

  tags = {
    Name        = "${var.cluster_name}-igw"
    Environment = var.environment
  }
}

# Subnets
resource "aws_subnet" "grpo_public_subnets" {
  count = 2

  vpc_id                  = aws_vpc.grpo_vpc.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name        = "${var.cluster_name}-public-subnet-${count.index + 1}"
    Environment = var.environment
    Type        = "public"
  }
}

resource "aws_subnet" "grpo_private_subnets" {
  count = 2

  vpc_id            = aws_vpc.grpo_vpc.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name        = "${var.cluster_name}-private-subnet-${count.index + 1}"
    Environment = var.environment
    Type        = "private"
  }
}

# Route Tables
resource "aws_route_table" "grpo_public_rt" {
  vpc_id = aws_vpc.grpo_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.grpo_igw.id
  }

  tags = {
    Name        = "${var.cluster_name}-public-rt"
    Environment = var.environment
  }
}

resource "aws_route_table_association" "grpo_public_rta" {
  count = length(aws_subnet.grpo_public_subnets)

  subnet_id      = aws_subnet.grpo_public_subnets[count.index].id
  route_table_id = aws_route_table.grpo_public_rt.id
}

# NAT Gateway
resource "aws_eip" "grpo_nat_eip" {
  domain = "vpc"

  tags = {
    Name        = "${var.cluster_name}-nat-eip"
    Environment = var.environment
  }
}

resource "aws_nat_gateway" "grpo_nat_gw" {
  allocation_id = aws_eip.grpo_nat_eip.id
  subnet_id     = aws_subnet.grpo_public_subnets[0].id

  tags = {
    Name        = "${var.cluster_name}-nat-gw"
    Environment = var.environment
  }
}

resource "aws_route_table" "grpo_private_rt" {
  vpc_id = aws_vpc.grpo_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.grpo_nat_gw.id
  }

  tags = {
    Name        = "${var.cluster_name}-private-rt"
    Environment = var.environment
  }
}

resource "aws_route_table_association" "grpo_private_rta" {
  count = length(aws_subnet.grpo_private_subnets)

  subnet_id      = aws_subnet.grpo_private_subnets[count.index].id
  route_table_id = aws_route_table.grpo_private_rt.id
}

# Security Groups
resource "aws_security_group" "grpo_cluster_sg" {
  name_prefix = "${var.cluster_name}-cluster-sg"
  vpc_id      = aws_vpc.grpo_vpc.id

  ingress {
    from_port = 443
    to_port   = 443
    protocol  = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port = 80
    to_port   = 80
    protocol  = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.cluster_name}-cluster-sg"
    Environment = var.environment
  }
}

# EKS Cluster
resource "aws_eks_cluster" "grpo_cluster" {
  name     = var.cluster_name
  role_arn = aws_iam_role.grpo_cluster_role.arn
  version  = "1.28"

  vpc_config {
    subnet_ids              = concat(aws_subnet.grpo_public_subnets[*].id, aws_subnet.grpo_private_subnets[*].id)
    endpoint_private_access = true
    endpoint_public_access  = true
    security_group_ids      = [aws_security_group.grpo_cluster_sg.id]
  }

  depends_on = [
    aws_iam_role_policy_attachment.grpo_cluster_policy,
    aws_iam_role_policy_attachment.grpo_cluster_vpc_policy,
  ]

  tags = {
    Name        = var.cluster_name
    Environment = var.environment
  }
}

# IAM Role for EKS Cluster
resource "aws_iam_role" "grpo_cluster_role" {
  name = "${var.cluster_name}-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "grpo_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.grpo_cluster_role.name
}

resource "aws_iam_role_policy_attachment" "grpo_cluster_vpc_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController"
  role       = aws_iam_role.grpo_cluster_role.name
}

# EKS Node Group
resource "aws_eks_node_group" "grpo_node_group" {
  cluster_name    = aws_eks_cluster.grpo_cluster.name
  node_group_name = "${var.cluster_name}-node-group"
  node_role_arn   = aws_iam_role.grpo_node_role.arn
  subnet_ids      = aws_subnet.grpo_private_subnets[*].id
  instance_types  = var.node_group_instance_types

  scaling_config {
    desired_size = var.node_group_scaling_config.desired_size
    max_size     = var.node_group_scaling_config.max_size
    min_size     = var.node_group_scaling_config.min_size
  }

  depends_on = [
    aws_iam_role_policy_attachment.grpo_node_policy,
    aws_iam_role_policy_attachment.grpo_node_cni_policy,
    aws_iam_role_policy_attachment.grpo_node_registry_policy,
  ]

  tags = {
    Name        = "${var.cluster_name}-node-group"
    Environment = var.environment
  }
}

# IAM Role for EKS Node Group
resource "aws_iam_role" "grpo_node_role" {
  name = "${var.cluster_name}-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "grpo_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.grpo_node_role.name
}

resource "aws_iam_role_policy_attachment" "grpo_node_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.grpo_node_role.name
}

resource "aws_iam_role_policy_attachment" "grpo_node_registry_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.grpo_node_role.name
}

# RDS PostgreSQL
resource "aws_db_subnet_group" "grpo_db_subnet_group" {
  name       = "${var.cluster_name}-db-subnet-group"
  subnet_ids = aws_subnet.grpo_private_subnets[*].id

  tags = {
    Name        = "${var.cluster_name}-db-subnet-group"
    Environment = var.environment
  }
}

resource "aws_security_group" "grpo_rds_sg" {
  name_prefix = "${var.cluster_name}-rds-sg"
  vpc_id      = aws_vpc.grpo_vpc.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.grpo_cluster_sg.id]
  }

  tags = {
    Name        = "${var.cluster_name}-rds-sg"
    Environment = var.environment
  }
}

resource "aws_db_instance" "grpo_postgres" {
  identifier             = "${var.cluster_name}-postgres"
  engine                 = "postgres"
  engine_version         = "15.4"
  instance_class         = "db.t3.micro"
  allocated_storage      = 20
  max_allocated_storage  = 100
  storage_type           = "gp2"
  storage_encrypted      = true
  
  db_name  = "grpo_framework"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.grpo_rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.grpo_db_subnet_group.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
  deletion_protection = false

  tags = {
    Name        = "${var.cluster_name}-postgres"
    Environment = var.environment
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "grpo_redis_subnet_group" {
  name       = "${var.cluster_name}-redis-subnet-group"
  subnet_ids = aws_subnet.grpo_private_subnets[*].id

  tags = {
    Name        = "${var.cluster_name}-redis-subnet-group"
    Environment = var.environment
  }
}

resource "aws_security_group" "grpo_redis_sg" {
  name_prefix = "${var.cluster_name}-redis-sg"
  vpc_id      = aws_vpc.grpo_vpc.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.grpo_cluster_sg.id]
  }

  tags = {
    Name        = "${var.cluster_name}-redis-sg"
    Environment = var.environment
  }
}

resource "aws_elasticache_cluster" "grpo_redis" {
  cluster_id           = "${var.cluster_name}-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.grpo_redis_subnet_group.name
  security_group_ids   = [aws_security_group.grpo_redis_sg.id]

  tags = {
    Name        = "${var.cluster_name}-redis"
    Environment = var.environment
  }
}

# Outputs
output "cluster_endpoint" {
  value = aws_eks_cluster.grpo_cluster.endpoint
}

output "cluster_security_group_id" {
  value = aws_eks_cluster.grpo_cluster.vpc_config[0].cluster_security_group_id
}

output "cluster_name" {
  value = aws_eks_cluster.grpo_cluster.name
}

output "rds_endpoint" {
  value = aws_db_instance.grpo_postgres.endpoint
}

output "redis_endpoint" {
  value = aws_elasticache_cluster.grpo_redis.cache_nodes[0].address
}
