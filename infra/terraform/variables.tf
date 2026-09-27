variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "eu-central-1"
}

variable "project" {
  description = "Short name used to prefix/tag all resources."
  type        = string
  default     = "coredrill"
}

variable "environment" {
  description = "Environment name, e.g. prod, staging."
  type        = string
  default     = "prod"
}

variable "cluster_version" {
  description = "Kubernetes version for the EKS control plane."
  type        = string
  default     = "1.31"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.42.0.0/16"
}

variable "node_instance_types" {
  description = "EC2 instance types for the CPU node group. LaBSE + coredrill needs ~1.5-2GB RSS; t3.large (2 vCPU / 8GB) gives headroom for 2-3 pods per node."
  type        = list(string)
  default     = ["t3.large"]
}

variable "node_min_size" {
  description = "Minimum nodes in the managed node group."
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum nodes in the managed node group."
  type        = number
  default     = 3
}

variable "node_desired_size" {
  description = "Starting node count."
  type        = number
  default     = 1
}

variable "node_capacity_type" {
  description = "ON_DEMAND or SPOT. SPOT is cheaper but nodes can be reclaimed with 2 minutes notice."
  type        = string
  default     = "ON_DEMAND"
}
