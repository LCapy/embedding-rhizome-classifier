#!/bin/bash
# Runs once on first boot. Installs Docker, pulls the coredrill-api image
# from ECR, fetches the API key from SSM Parameter Store, and runs the
# container with --restart unless-stopped so it survives crashes/reboots.
set -euo pipefail

dnf install -y docker awscli
systemctl enable --now docker

REGION="${region}"
ECR_URL="${ecr_url}"
S3_URI="${s3_uri}"
SSM_PARAM="${ssm_param}"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$${ECR_URL%%/*}"

API_KEY=$(aws ssm get-parameter --name "$SSM_PARAM" --with-decryption \
  --region "$REGION" --query 'Parameter.Value' --output text)

docker pull "$ECR_URL:latest"
docker rm -f coredrill-api 2>/dev/null || true
docker run -d --name coredrill-api --restart unless-stopped \
  -p 80:7860 \
  -e PORT=7860 \
  -e MODEL_NAME=sentence-transformers/LaBSE \
  -e COREDRILL_S3_URI="$S3_URI" \
  -e API_KEY="$API_KEY" \
  "$ECR_URL:latest"
