# Self-hosted deployment on EKS

Moves `coredrill-api` off Hugging Face and onto your own EKS cluster:
Terraform provisions the AWS side (VPC, EKS, ECR, S3, IAM), plain Kubernetes
manifests run the app. See the diagram in
[`docs/architecture.md`](../docs/architecture.md#2-deployment-topology-target-self-hosted-on-eks).

This is real, billable AWS infrastructure. Nothing here has been applied -
review every file before you run `terraform apply`.

## What gets created

| Resource | Purpose |
|---|---|
| VPC, 2 AZs, private + public subnets, 1 NAT gateway | Network for the cluster |
| EKS cluster, 1.31 | Control plane |
| Managed node group, `t3.large` x1-3, CPU only, on-demand | Runs the API pods |
| ECR repo `coredrill-api` | Holds the Docker image |
| S3 bucket `coredrill-data-<account-id>` | Holds `coredrill_hierarchical.json`, replaces the HF Dataset repo |
| IAM role (IRSA) | Lets the pod read the S3 bucket without static credentials |

**Estimated cost** (eu-central-1, on-demand, ballpark): EKS control plane
~$73/mo + 1x `t3.large` ~$60/mo + 1 NAT gateway ~$35/mo + data transfer ≈
**$170-200/mo** for a single always-on node. Scaling to 3 nodes for load adds
~$120/mo. Switching `node_capacity_type` to `SPOT` in
[`terraform/variables.tf`](terraform/variables.tf) cuts node cost by roughly
60-70%, at the cost of nodes being reclaimed with ~2 minutes notice.

## Prerequisites

- `terraform` >= 1.6 ([download](https://developer.hashicorp.com/terraform/install))
- `aws` CLI v2, configured with credentials that can create VPC/EKS/IAM/ECR/S3
  resources (`aws sts get-caller-identity` should succeed)
- `kubectl`
- `docker`, to build and push the image

None of these are installed in this workspace yet - install them locally or
in CI before running the commands below.

## 1. Provision AWS infra

```bash
cd infra/terraform
terraform init
terraform plan      # review every resource before applying
terraform apply
```

Note the outputs - you'll need `ecr_repository_url`, `s3_bucket_name`, and
`irsa_role_arn` in the next steps.

```bash
$(terraform output -raw configure_kubectl)   # aws eks update-kubeconfig ...
kubectl get nodes                            # confirm you can reach the cluster
```

## 2. Upload the coredrill data to S3

```bash
aws s3 cp path/to/coredrill_hierarchical.json \
  s3://$(terraform output -raw s3_bucket_name)/coredrill_hierarchical.json
```

## 3. Build and push the image to ECR

From the repo root:

```bash
ECR_URL=$(terraform -chdir=infra/terraform output -raw ecr_repository_url)
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin "${ECR_URL%/*}"

docker build -t "$ECR_URL:latest" .
docker push "$ECR_URL:latest"
```

## 4. Deploy to the cluster

Fill in the three placeholders before applying:

```bash
IRSA_ROLE_ARN=$(terraform -chdir=infra/terraform output -raw irsa_role_arn)
ECR_URL=$(terraform -chdir=infra/terraform output -raw ecr_repository_url)
S3_BUCKET=$(terraform -chdir=infra/terraform output -raw s3_bucket_name)

sed -i "s#REPLACE_WITH_TERRAFORM_OUTPUT_irsa_role_arn#${IRSA_ROLE_ARN}#" infra/k8s/serviceaccount.yaml
sed -i "s#REPLACE_WITH_ECR_REPOSITORY_URL:latest#${ECR_URL}:latest#" infra/k8s/deployment.yaml
sed -i "s#REPLACE_WITH_S3_BUCKET_NAME#${S3_BUCKET}#" infra/k8s/deployment.yaml
```

Create the API bearer token as a Secret (never commit it to a manifest):

```bash
kubectl create namespace coredrill
kubectl -n coredrill create secret generic coredrill-api-key \
  --from-literal=api-key="$(openssl rand -hex 32)"
```

Apply everything:

```bash
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/serviceaccount.yaml
kubectl apply -f infra/k8s/deployment.yaml
kubectl apply -f infra/k8s/service.yaml
```

The HPA needs `metrics-server`, which EKS doesn't install by default:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl apply -f infra/k8s/hpa.yaml
```

## 5. Verify

```bash
kubectl -n coredrill get pods -w                     # wait for Running + Ready
LB_HOST=$(kubectl -n coredrill get svc coredrill-api -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
curl "http://$LB_HOST/health"
```

## Next steps (not included yet)

- **TLS + custom domain**: install the [AWS Load Balancer
  Controller](https://kubernetes-sigs.github.io/aws-load-balancer-controller/),
  request an ACM certificate, and switch `infra/k8s/service.yaml` to an
  `Ingress` with `alb.ingress.kubernetes.io/certificate-arn`.
- **CI/CD**: a GitHub Action to build, push to ECR, and roll the Deployment on
  every push to `main` - ask and I'll wire it up.
- **Remote Terraform state**: the `backend "s3"` block in
  [`terraform/versions.tf`](terraform/versions.tf) is commented out; create a
  state bucket and DynamoDB lock table first, then uncomment it so state isn't
  only on your laptop.
- **Restrict the control-plane endpoint**: `cluster_endpoint_public_access` is
  open to the internet for simplicity; set
  `cluster_endpoint_public_access_cidrs` once you know which IPs need access.
