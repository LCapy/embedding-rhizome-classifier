# EKS cluster with a single CPU-only managed node group.
# Nodes live in the private subnets; the control plane endpoint is public
# (simplest for a first deploy - restrict via cluster_endpoint_public_access_cidrs
# once you know the IPs that need access, e.g. your CI runner and your own IP).
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.31"

  cluster_name    = local.name
  cluster_version = var.cluster_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  # Grants the identity that ran `terraform apply` cluster-admin via EKS
  # access entries, so you don't have to hand-edit aws-auth.
  enable_cluster_creator_admin_permissions = true

  eks_managed_node_groups = {
    cpu = {
      instance_types = var.node_instance_types
      capacity_type  = var.node_capacity_type
      ami_type       = "AL2023_x86_64_STANDARD"

      min_size     = var.node_min_size
      max_size     = var.node_max_size
      desired_size = var.node_desired_size

      labels = {
        workload = "coredrill-api"
      }
    }
  }
}
