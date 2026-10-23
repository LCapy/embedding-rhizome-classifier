# Cost and runtime monitoring for this project's AWS spend and the running
# instance. The dashboard/alarm below use EC2's built-in metrics (no agent,
# no extra cost) - see infra/README.md for what's not covered (app-level
# request/latency metrics would need instrumenting api.py directly).

# ── Cost: budget + email alerts ──────────────────────────────────────────
resource "aws_budgets_budget" "monthly" {
  name         = "${local.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_limit_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = [format("user:Project$%s", var.project)]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_notification_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_notification_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_notification_email]
  }
}

# ── Cost: turn the Project/Environment tags into cost-allocation tags ──────
# Lets Cost Explorer break spend down by this project once AWS activates
# them (can take up to 24h after creation - that delay is on AWS's side,
# not something Terraform controls).
resource "aws_ce_cost_allocation_tag" "project" {
  tag_key = "Project"
  status  = "Active"
}

resource "aws_ce_cost_allocation_tag" "environment" {
  tag_key = "Environment"
  status  = "Active"
}

# ── Runs: auto-recover the instance if AWS detects a hardware/system fault ──
# Doesn't catch an application crash (docker --restart unless-stopped in
# user_data.sh.tpl handles that) - this is for the underlying host.
resource "aws_cloudwatch_metric_alarm" "status_check_failed" {
  alarm_name          = "${local.name}-status-check-failed"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "StatusCheckFailed_System"
  namespace           = "AWS/EC2"
  period              = 60
  statistic           = "Maximum"
  threshold           = 0
  dimensions = {
    InstanceId = aws_instance.app.id
  }
  alarm_actions = ["arn:aws:automate:${var.aws_region}:ec2:recover"]
}

# ── Runs: at-a-glance dashboard (built-in EC2 metrics, no agent needed) ────
resource "aws_cloudwatch_dashboard" "coredrill" {
  dashboard_name = local.name

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "CPU utilization"
          region = var.aws_region
          view   = "timeSeries"
          stat   = "Average"
          period = 300
          metrics = [
            ["AWS/EC2", "CPUUtilization", "InstanceId", aws_instance.app.id]
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Network in/out"
          region = var.aws_region
          view   = "timeSeries"
          stat   = "Sum"
          period = 300
          metrics = [
            ["AWS/EC2", "NetworkIn", "InstanceId", aws_instance.app.id],
            ["AWS/EC2", "NetworkOut", "InstanceId", aws_instance.app.id]
          ]
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "Status checks failed (system + instance)"
          region = var.aws_region
          view   = "timeSeries"
          stat   = "Maximum"
          period = 300
          metrics = [
            ["AWS/EC2", "StatusCheckFailed_System", "InstanceId", aws_instance.app.id],
            ["AWS/EC2", "StatusCheckFailed_Instance", "InstanceId", aws_instance.app.id]
          ]
        }
      }
    ]
  })
}
