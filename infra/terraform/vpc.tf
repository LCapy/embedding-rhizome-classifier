# Minimal network for a single public instance: no NAT gateway (that was
# most of the EKS-era cost), no private subnets - the instance gets a
# public IP directly and is reached through the security group in ec2.tf.
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.13"

  name = local.name
  cidr = var.vpc_cidr

  azs            = [local.azs[0]]
  public_subnets = [cidrsubnet(var.vpc_cidr, 4, 0)]

  enable_dns_hostnames = true
}
