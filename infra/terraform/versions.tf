terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state. Create the bucket + lock table once (see infra/README.md),
  # then uncomment this block and run `terraform init -migrate-state`.
  # backend "s3" {
  #   bucket         = "coredrill-terraform-state"
  #   key            = "eks/terraform.tfstate"
  #   region         = "eu-central-1"
  #   dynamodb_table = "coredrill-terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}

data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

locals {
  azs  = slice(data.aws_availability_zones.available.names, 0, 2)
  name = "${var.project}-${var.environment}"
}
