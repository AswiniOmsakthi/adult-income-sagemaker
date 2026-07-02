# Adult Income Prediction — End-to-End MLOps Pipeline on AWS SageMaker

A fully automated, CI/CD-driven MLOps pipeline that trains a machine learning model to predict whether a person earns more than $50,000/year, then deploys it to a live prediction endpoint — all triggered from a single `git push`, with infrastructure managed by AWS CDK.

Built with **AWS SageMaker**, **AWS CDK (Python)**, **GitHub Actions**, **EventBridge**, and **Lambda**.

---

## What This Project Does (In Plain English)

Imagine you push code to GitHub. Without touching the AWS console, the following happens automatically:

1. **Infrastructure is built** — a SageMaker Studio workspace, storage bucket, security roles, and an event-listener are all created in AWS.
2. **A model is trained** — the pipeline takes census data, trains a machine learning model, checks its accuracy, and files it in a "model registry" waiting for sign-off.
3. **A human approves the model** — you review the model in AWS and click "Approve" (this is the one manual gate, on purpose).
4. **The model goes live automatically** — approval triggers an automatic deployment to a web endpoint you can send data to and get predictions back.

That's the whole loop: **push → train → approve → deploy**, with no manual AWS clicking except the deliberate approval step.

---

## Architecture at a Glance

```
                          git push origin master
                                    │
                                    ▼
                  ┌─────────────────────────────────────┐
                  │  PIPELINE 1 — INFRASTRUCTURE (CDK)   │
                  │  infra-pipeline.yml                  │
                  │                                       │
                  │  Runs: cdk deploy                    │
                  │  Creates / imports:                  │
                  │    • S3 Bucket                       │
                  │    • SageMaker Studio Domain         │
                  │    • IAM Roles                       │
                  │    • Lambda function                 │
                  │    • EventBridge rule                │
                  └──────────────────┬────────────────────┘
                                     │ auto-triggers (workflow_run)
                                     ▼
                  ┌─────────────────────────────────────┐
                  │  PIPELINE 2 — MACHINE LEARNING       │
                  │  sagemaker-pipeline.yml              │
                  │                                       │
                  │  Runs: run_pipeline.py               │
                  │  5 steps inside SageMaker:           │
                  │    1. Upload data to S3              │
                  │    2. Split train/test               │
                  │    3. Train model (Spot instance)    │
                  │    4. Evaluate + Quality Gate        │
                  │    5. Register model (Pending)       │
                  └──────────────────┬────────────────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │   HUMAN APPROVAL         │
                        │   (SageMaker Registry)   │
                        │   You click "Approve"    │
                        └───────────┬──────────────┘
                                    │ EventBridge detects approval
                                    ▼
                  ┌─────────────────────────────────────┐
                  │  PIPELINE 3 — DEPLOYMENT             │
                  │  deploy-pipeline.yml                 │
                  │                                       │
                  │  Triggered by: EventBridge → Lambda  │
                  │                → GitHub API          │
                  │  Runs: deploy_endpoint.py            │
                  │    • Deploys model to endpoint       │
                  │    • Tests predictions               │
                  └───────────────────────────────────────┘
                                     │
                                     ▼
                        Live endpoint returns predictions
                        (CSV or JSON, via Studio or code)
```

---

## Repository Structure

```
adult-income-sagemaker/
│
├── .github/workflows/              ← The 3 automation pipelines
│   ├── infra-pipeline.yml          ← Pipeline 1: builds AWS infra via CDK
│   ├── sagemaker-pipeline.yml      ← Pipeline 2: trains & registers model
│   └── deploy-pipeline.yml         ← Pipeline 3: deploys approved model
│
├── cdk/                            ← Infrastructure as Code (AWS CDK)
│   ├── app.py                      ← CDK entry point
│   ├── adult_income_stack.py       ← Defines ALL AWS resources
│   ├── save_outputs.py             ← Saves infra details to S3
│   ├── cdk.json                    ← CDK configuration
│   └── requirements.txt            ← CDK Python dependencies
│
├── infra/                          ← RETIRED boto3 scripts (kept for teaching)
│   ├── create_studio.py            ← "Before CDK" version
│   └── setup_eventbridge.py        ← "Before CDK" version
│
├── pipeline/                       ← The ML code
│   ├── preprocessing.py            ← Splits data into train/test
│   ├── train.py                    ← Trains model + inference handlers
│   └── evaluate.py                 ← Scores model, writes metrics
│
├── deploy/
│   └── deploy_endpoint.py          ← Deploys approved model to endpoint
│
├── data/
│   └── adult_final.csv             ← The dataset (committed to repo)
│
├── run_pipeline.py                 ← Defines the SageMaker ML pipeline
├── requirements.txt                ← Main Python dependencies
├── README.md                       ← This file
└── DOCUMENTATION.md                ← Detailed file-by-file walkthrough
```

---

## The Dataset

**Source:** UCI Machine Learning Repository — Adult (Census Income) dataset
**Size used:** 2,000 rows, balanced (1,000 earning >$50K, 1,000 earning ≤$50K)

Each row describes a person with these 7 features, and 1 target:

| Feature | Meaning | Example |
|---|---|---|
| `age` | Age in years | 39 |
| `education_num` | Years of education | 13 (Bachelor's) |
| `hours_per_week` | Weekly work hours | 40 |
| `capital_gain` | Investment gains | 2174 |
| `capital_loss` | Investment losses | 0 |
| `sex_male` | 1 = male, 0 = female | 1 |
| `is_married` | 1 = married, 0 = not | 0 |
| **`income_above_50k`** | **TARGET: 1 = >$50K, 0 = ≤$50K** | **1** |

---

## Prerequisites

Before you begin, you need:

- **An AWS account** with permissions for SageMaker, S3, IAM, Lambda, EventBridge, CloudFormation, Secrets Manager
- **A GitHub account** and a repository for this code
- **A GitHub Personal Access Token** (with `repo` and `workflow` scopes)
- **AWS CLI** access (or CloudShell in the browser)

---

## One-Time Setup

### Step 1 — Store your GitHub token in AWS Secrets Manager

```bash
aws secretsmanager create-secret \
  --name github-token \
  --secret-string "ghp_YOUR_TOKEN_HERE" \
  --region us-east-1
```

> Stored as plain text (not JSON). The Lambda reads it directly.

### Step 2 — Add GitHub repository secrets

Go to your repo → **Settings → Secrets and variables → Actions**, and add:

| Secret Name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Your IAM user's access key |
| `AWS_SECRET_ACCESS_KEY` | Your IAM user's secret key |
| `AWS_REGION` | `us-east-1` |
| `AWS_ROLE_ARN` | SageMaker execution role ARN |
| `AWS_ACCOUNT_ID` | Your 12-digit AWS account number |

### Step 3 — Update the GitHub repo name in the CDK stack

In `cdk/app.py`, set your repo:

```python
github_repo="YOUR_USERNAME/adult-income-sagemaker"
```

### Step 4 — Push

```bash
git add .
git commit -m "initial setup"
git push origin master
```

Pipeline 1 and Pipeline 2 run automatically.

---

## Approving a Model & Deploying

1. Wait for Pipeline 2 to finish (green check in GitHub Actions)
2. Open **SageMaker Studio → Models → AdultIncomeGroup**
3. Select the latest version → **Update status → Approved**
4. Deployment (Pipeline 3) triggers automatically within ~30 seconds

---

## Testing the Live Endpoint

### From SageMaker Studio Playground

```
Endpoint → Playground
Content type: application/json
Body: {"instances": [[39, 13, 40, 2174, 0, 1, 0]]}
Result: {"predictions": [1]}   ← 1 means income > $50K
```

### From the command line (CloudShell)

```bash
echo -n '{"instances": [[39, 13, 40, 2174, 0, 1, 0]]}' > input.json
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name adult-income-predictor-endpoint \
  --content-type application/json \
  --body fileb://input.json \
  --region us-east-1 \
  result.json
cat result.json
```

---

## Cost Summary

| Item | Cost |
|---|---|
| One full pipeline run (train + deploy) | ~$0.025 (~₹2) |
| Model training (Spot instance) | Up to 90% cheaper than normal |
| **Live endpoint running 24/7** | **~$82.80/month (~₹6,900)** |
| S3 storage | ~$0.01/month |

> **The endpoint is the big cost.** Always delete it after testing:
> ```bash
> aws sagemaker delete-endpoint \
>   --endpoint-name adult-income-predictor-endpoint \
>   --region us-east-1
> ```

---

## Why AWS CDK (Not Plain Scripts)?

This project uses **AWS CDK** for infrastructure instead of raw boto3 scripts because CDK provides:

- **State tracking** — knows what already exists, won't duplicate
- **Preview changes** — `cdk diff` shows what will change before it happens
- **Automatic rollback** — if deployment fails, it cleanly reverts
- **One-command teardown** — `cdk destroy` removes everything
- **Dependency ordering** — figures out what to create first, automatically

The old boto3 scripts are kept in `infra/` as a teaching reference showing the "before CDK" approach.

---

## Full Documentation

For a detailed, file-by-file explanation of how everything works — including the debugging issues we solved and why each design choice was made — see **DOCUMENTATION.md**.

---

## License

Dataset: UCI Machine Learning Repository (public domain, research/educational use).
Code: MIT.
