# Single instance running the coredrill-api container directly via Docker.
# No SSH: administer through SSM Session Manager (IAM-authenticated, no
# open port 22, no key pair to lose).
data "aws_ssm_parameter" "al2023_ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

resource "random_password" "api_key" {
  length  = 48
  special = false
}

resource "aws_ssm_parameter" "api_key" {
  name  = "/${var.project}/${var.environment}/api-key"
  type  = "SecureString"
  value = random_password.api_key.result
}

resource "aws_security_group" "app" {
  name        = "${local.name}-app"
  description = "coredrill-api: inbound HTTP only, no SSH (use SSM)"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ec2" {
  name               = "${local.name}-ec2"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

data "aws_iam_policy_document" "ec2_extra" {
  statement {
    sid     = "ReadCoredrillData"
    actions = ["s3:GetObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.coredrill_data.arn,
      "${aws_s3_bucket.coredrill_data.arn}/*",
    ]
  }

  statement {
    sid       = "ReadApiKey"
    actions   = ["ssm:GetParameter"]
    resources = [aws_ssm_parameter.api_key.arn]
  }

  statement {
    sid       = "DecryptApiKey"
    actions   = ["kms:Decrypt"]
    resources = ["*"] # default aws/ssm key - SecureString params only
  }
}

resource "aws_iam_policy" "ec2_extra" {
  name   = "${local.name}-ec2-extra"
  policy = data.aws_iam_policy_document.ec2_extra.json
}

resource "aws_iam_role_policy_attachment" "ec2_extra" {
  role       = aws_iam_role.ec2.name
  policy_arn = aws_iam_policy.ec2_extra.arn
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${local.name}-ec2"
  role = aws_iam_role.ec2.name
}

resource "aws_instance" "app" {
  ami                    = data.aws_ssm_parameter.al2023_ami.value
  instance_type          = var.instance_type
  subnet_id              = module.vpc.public_subnets[0]
  vpc_security_group_ids = [aws_security_group.app.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2.name

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  # ECR repo must exist before the instance tries to pull from it.
  depends_on = [aws_ecr_repository.coredrill_api]

  user_data = templatefile("${path.module}/templates/user_data.sh.tpl", {
    region    = var.aws_region
    ecr_url   = aws_ecr_repository.coredrill_api.repository_url
    s3_uri    = "s3://${aws_s3_bucket.coredrill_data.bucket}/coredrill_hierarchical.json"
    ssm_param = aws_ssm_parameter.api_key.name
  })
  user_data_replace_on_change = true

  tags = {
    Name = local.name
  }
}

resource "aws_eip" "app" {
  domain = "vpc"
}

resource "aws_eip_association" "app" {
  instance_id   = aws_instance.app.id
  allocation_id = aws_eip.app.id
}
