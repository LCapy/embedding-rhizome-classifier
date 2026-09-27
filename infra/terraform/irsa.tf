# IAM role the coredrill-api pod assumes (via its Kubernetes ServiceAccount)
# to read coredrill_hierarchical.json from S3 - no static AWS keys in the pod.
data "aws_iam_policy_document" "coredrill_s3_read" {
  statement {
    sid       = "ReadCoredrillData"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.coredrill_data.arn,
      "${aws_s3_bucket.coredrill_data.arn}/*",
    ]
  }
}

resource "aws_iam_policy" "coredrill_s3_read" {
  name   = "${local.name}-s3-read"
  policy = data.aws_iam_policy_document.coredrill_s3_read.json
}

module "coredrill_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.48"

  role_name = "${local.name}-api"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["coredrill:coredrill-api"]
    }
  }

  role_policy_arns = {
    s3_read = aws_iam_policy.coredrill_s3_read.arn
  }
}
