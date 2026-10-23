output "ecr_repository_url" {
  value = aws_ecr_repository.coredrill_api.repository_url
}

output "s3_bucket_name" {
  value = aws_s3_bucket.coredrill_data.bucket
}

output "instance_id" {
  value = aws_instance.app.id
}

output "public_ip" {
  description = "Elastic IP - the API is reachable at http://<this>/health once the instance finishes booting."
  value       = aws_eip.app.public_ip
}

output "ssm_connect" {
  description = "Shell into the instance without SSH/a key pair."
  value       = "aws ssm start-session --target ${aws_instance.app.id} --profile ${var.aws_profile} --region ${var.aws_region}"
}

output "api_key_ssm_param" {
  description = "Where the bearer token lives. Read it with: aws ssm get-parameter --name <this> --with-decryption --profile <your aws_profile>"
  value       = aws_ssm_parameter.api_key.name
}

output "cost_dashboard_url" {
  description = "CloudWatch dashboard: instance CPU, network, status checks."
  value       = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${local.name}"
}

output "budget_name" {
  description = "AWS Budgets alert - view/edit at https://us-east-1.console.aws.amazon.com/billing/home#/budgets"
  value       = aws_budgets_budget.monthly.name
}
