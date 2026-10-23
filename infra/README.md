# Self-hosted deployment on EC2 (option, not applied)

**Status:** not applied. Hosting stays on the Hugging Face Space for now,
for budget reasons; this design is kept as a ready option.

It would move `coredrill-api` off Hugging Face and onto a single EC2
instance in your own AWS account. Terraform provisions everything: network,
the instance, ECR, S3, IAM, and monitoring. See the diagram in
[`docs/architecture.md`](../docs/architecture.md#2-deployment-topology-option-not-applied-self-hosted-on-ec2).

This started as an EKS design; dropped for cost - the EKS control plane
alone is a fixed ~$73/mo no matter how small the rest is. A single EC2
instance running the container directly gets the same outcome for roughly a
third of that.

This is real, billable AWS infrastructure. Nothing here has been applied -
review every file before you run `terraform apply`.

## What gets created

| Resource | Purpose |
|---|---|
| VPC, 1 public subnet, no NAT gateway | Network - the instance gets a public IP directly |
| EC2 instance, `t3.medium`, Amazon Linux 2023 | Runs the container via Docker (`--restart unless-stopped`) |
| Elastic IP | Stable public IP for the API |
| ECR repo `coredrill-api` | Holds the Docker image |
| S3 bucket `coredrill-data-<account-id>` | Holds `coredrill_hierarchical.json`, replaces the HF Dataset repo |
| SSM Parameter (`SecureString`) | Holds the generated API bearer token |
| IAM instance role | S3 read, ECR pull, SSM (Session Manager + the parameter above) - no static keys on the box |
| CloudWatch alarm | Auto-recovers the instance on a system status check failure |
| CloudWatch dashboard + AWS Budget + cost allocation tags | Cost and runtime monitoring - see below |

**Estimated cost** (eu-central-1, ballpark): `t3.medium` on-demand ~$30/mo +
30GB gp3 EBS ~$2.50/mo + Elastic IP (free while attached to a running
instance) + ECR/S3 storage (a few cents) + CloudWatch ≈ **$35-40/mo**.
Switching to a `t3.small` (2GB RAM - tighter but workable) or reserving/
Savings-Plan pricing the instance can push this lower; ask and I'll adjust
`instance_type` in [`terraform/variables.tf`](terraform/variables.tf).

## Prerequisites

- `terraform` >= 1.6 - already installed in this workspace at
  `C:\Users\lucas\bin\terraform.exe` (not on `PATH` by default on this
  machine; use the full path or add that dir to `PATH`)
- `aws` CLI v2, with the dedicated `rhizome-classifier` profile (see below) -
  never point this at a shared/default profile
- `docker`, to build and push the image
- No `kubectl` needed - this isn't Kubernetes

A dedicated IAM user (`rhizome-classifier`) and AWS CLI profile of the same
name already exist for this project - `terraform/variables.tf`'s
`aws_profile` variable defaults to it, so `terraform` commands below use it
automatically. Confirm it works before running anything:

```bash
aws sts get-caller-identity --profile rhizome-classifier
```

## 1. Provision AWS infra

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # set budget_notification_email; gitignored
terraform init
terraform plan -out=tfplan.out    # review every resource before applying
terraform apply tfplan.out
```

Note the outputs - `ecr_repository_url`, `s3_bucket_name`, `public_ip`,
`ssm_connect`.

## 2. Upload the coredrill data to S3

```bash
aws s3 cp path/to/coredrill_hierarchical.json \
  s3://$(terraform output -raw s3_bucket_name)/coredrill_hierarchical.json \
  --profile rhizome-classifier
```

## 3. Build and push the image to ECR

From the repo root:

```bash
ECR_URL=$(terraform -chdir=infra/terraform output -raw ecr_repository_url)
aws ecr get-login-password --region eu-central-1 --profile rhizome-classifier \
  | docker login --username AWS --password-stdin "${ECR_URL%/*}"

docker build -t "$ECR_URL:latest" .
docker push "$ECR_URL:latest"
```

## 4. First boot / redeploy

On first `apply`, the instance's `user_data` script (see
[`terraform/templates/user_data.sh.tpl`](terraform/templates/user_data.sh.tpl))
installs Docker, pulls `$ECR_URL:latest`, fetches the API key from SSM, and
starts the container automatically - nothing else to run.

If step 3 happened *after* the instance was already created (image didn't
exist yet on first boot, or you're shipping a new version), redeploy without
SSH, over SSM:

```bash
aws ssm send-command \
  --instance-ids "$(terraform -chdir=infra/terraform output -raw instance_id)" \
  --document-name "AWS-RunShellScript" \
  --parameters commands='["docker pull '"$ECR_URL"':latest","docker rm -f coredrill-api","docker run -d --name coredrill-api --restart unless-stopped -p 80:7860 -e PORT=7860 -e MODEL_NAME=sentence-transformers/LaBSE -e COREDRILL_S3_URI=s3://'"$(terraform -chdir=infra/terraform output -raw s3_bucket_name)"'/coredrill_hierarchical.json -e API_KEY=$(aws ssm get-parameter --name '"$(terraform -chdir=infra/terraform output -raw api_key_ssm_param)"' --with-decryption --query Parameter.Value --output text) '"$ECR_URL"':latest"]' \
  --profile rhizome-classifier --region eu-central-1
```

That one-liner is unwieldy on purpose - it's a stopgap. See "Next steps" for
proper CI/CD.

## 5. Verify

```bash
curl "http://$(terraform -chdir=infra/terraform output -raw public_ip)/health"
```

Need a shell on the box? No SSH key exists for it - use SSM instead:

```bash
$(terraform -chdir=infra/terraform output -raw ssm_connect)
```

## Monitoring: costs and runs

**Costs.** `aws_budgets_budget.monthly` emails `budget_notification_email`
at 50%/80% actual spend and 100% forecasted against
`monthly_budget_limit_usd` (default $60), filtered to this project's
`Project` tag so it doesn't pick up spend from anything else in the account.
Cost allocation tags take up to 24h after `apply` to activate on AWS's side
before Cost Explorer can filter by them. View/edit budgets at
[console.aws.amazon.com/billing/home#/budgets](https://us-east-1.console.aws.amazon.com/billing/home#/budgets).

**Runs.** The `coredrill-prod` CloudWatch dashboard
(`terraform output cost_dashboard_url`) shows instance CPU, network in/out,
and status checks - all from EC2's built-in metrics, no agent installed, no
extra cost. A `StatusCheckFailed_System` alarm auto-recovers the instance on
an AWS-side hardware fault; an app crash is instead caught by Docker's own
`--restart unless-stopped`. This does **not** cover request-level metrics
(request count, latency, error rate) - that needs instrumenting `api.py`
directly (e.g. `prometheus-fastapi-instrumentator` + a scrape target, or
just structured logging to CloudWatch Logs). Ask and I'll wire it in.

```bash
terraform -chdir=infra/terraform output cost_dashboard_url
```

## Next steps (not included yet)

- **TLS + custom domain**: point a domain's A record at the Elastic IP, then
  run [Caddy](https://caddyserver.com/) on the box in front of the container
  (`reverse_proxy localhost:7860` in a two-line Caddyfile) for automatic
  Let's Encrypt HTTPS - no load balancer needed at this scale.
- **CI/CD**: a GitHub Action to build, push to ECR, and redeploy over SSM on
  every push to `main`, replacing the manual steps 3-4 above - ask and I'll
  wire it up.
- **App-level metrics**: request count/latency/error rate from `api.py`
  itself, not just host CPU/network.
- **Remote Terraform state**: the `backend "s3"` block in
  [`terraform/versions.tf`](terraform/versions.tf) is commented out; create a
  state bucket and DynamoDB lock table first, then uncomment it so state isn't
  only on your laptop.
- **If you outgrow one box**: the EKS design this replaced is recoverable
  from git history if traffic ever justifies the ~$73/mo control-plane cost;
  ECS Fargate is a middle ground worth considering first (no fixed
  control-plane fee, still gets auto-restart and native CloudWatch logs).
