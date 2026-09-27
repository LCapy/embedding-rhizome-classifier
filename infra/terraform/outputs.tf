output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "configure_kubectl" {
  description = "Run this to point kubectl at the new cluster."
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.coredrill_api.repository_url
}

output "s3_bucket_name" {
  value = aws_s3_bucket.coredrill_data.bucket
}

output "irsa_role_arn" {
  description = "Put this in infra/k8s/serviceaccount.yaml's eks.amazonaws.com/role-arn annotation."
  value       = module.coredrill_irsa.iam_role_arn
}
