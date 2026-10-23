variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "eu-central-1"
}

variable "aws_profile" {
  description = "Named AWS CLI profile to use (see ~/.aws/credentials). Kept as its own variable so this stays a dedicated profile, never the shared default."
  type        = string
  default     = "rhizome-classifier"
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

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.42.0.0/16"
}

variable "instance_type" {
  description = "EC2 instance type for the coredrill-api box. LaBSE + coredrill needs ~1.5-2GB RSS; t3.medium (2 vCPU / 4GB) covers that with headroom."
  type        = string
  default     = "t3.medium"
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB (Docker image + LaBSE weights baked in are ~2-3GB)."
  type        = number
  default     = 30
}

variable "budget_notification_email" {
  description = "Email address for AWS Budgets cost alerts. Set it in terraform.tfvars (gitignored); see terraform.tfvars.example."
  type        = string
}

variable "monthly_budget_limit_usd" {
  description = "Monthly cost budget in USD for this project's resources (see infra/README.md cost estimate)."
  type        = number
  default     = 60
}
