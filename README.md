# Adult Income Prediction — End-to-End MLOps Pipeline on AWS SageMaker

A fully automated, CI/CD-driven MLOps pipeline built on Amazon SageMaker, GitHub Actions, and EventBridge. Predicts whether an individual's income exceeds $50K/year using the [UCI Adult Income dataset](https://archive.ics.uci.edu/dataset/2/adult).

This project demonstrates a complete, cost-optimized MLOps workflow — from infrastructure provisioning to model training, evaluation, approval-gated registration, and automatic deployment — using only GitHub Actions and AWS, with zero manual console steps required after initial setup.

---

## Architecture Overview

```
git push origin master
        │
        ▼
┌─────────────────────────────────────┐
│  Pipeline 1 — Infra                  │
│  (infra-pipeline.yml)                │
│                                       │
│  • Creates SageMaker Studio domain   │
│  • Creates IAM execution role        │
│  • Creates S3 bucket                 │
│  • Creates EventBridge rule          │
│  • Creates Lambda auto-deploy trigger│
└───────────────┬───────────────────────┘
                │ workflow_run (auto)
                ▼
┌─────────────────────────────────────┐
│  Pipeline 2 — ML                     │
│  (sagemaker-pipeline.yml)            │
│                                       │
│  • Uploads dataset to S3 (via code)  │
│  • DataPreparation (train/test split)│
│  • ModelTraining (Spot instance)     │
│  • ModelEvaluation                   │
│  • QualityGate (accuracy ≥ 0.75)     │
│  • ModelRegistration (Pending)       │
└───────────────┬───────────────────────┘
                │
                ▼
      ┌───────────────────┐
      │  Manual Approval    │
      │  (SageMaker Model   │
      │   Registry)          │
      └─────────┬───────────┘
                │ EventBridge detects approval
                ▼
┌─────────────────────────────────────┐
│  Pipeline 3 — Deploy                 │
│  (deploy-pipeline.yml)               │
│                                       │
│  • Triggered automatically via       │
│    EventBridge → Lambda → GitHub API │
│  • Deploys to real-time endpoint     │
│  • Tests predictions (CSV + JSON)    │
└───────────────────────────────────────┘
```

---

## Repository Structure

```
adult-income-sagemaker/
│
├── .github/workflows/
│   ├── infra-pipeline.yml        # Pipeline 1 — creates AWS infrastructure
│   ├── sagemaker-pipeline.yml    # Pipeline 2 — trains and registers model
│   └── deploy-pipeline.yml       # Pipeline 3 — deploys to endpoint
│
├── infra/
│   ├── create_studio.py          # Creates SageMaker Studio domain, IAM role, S3 bucket
│   └── setup_eventbridge.py      # Creates EventBridge rule + Lambda auto-deploy trigger
│
├── pipeline/
│   ├── preprocessing.py          # Train/test split (SageMaker Processing step)
│   ├── train.py                  # Model training + inference handlers (model_fn,
│   │                              #   input_fn, predict_fn, output_fn)
│   └── evaluate.py               # Model evaluation, writes evaluation.json
│
├── deploy/
│   └── deploy_endpoint.py        # Deploys approved model to a real-time endpoint
│
├── data/
│   └── adult_final.csv           # Pre-processed Adult Income dataset (committed to repo)
│
├── run_pipeline.py               # Defines and triggers the SageMaker ML Pipeline
├── requirements.txt
└── .gitignore
```

---

## Dataset

**Source:** [UCI Machine Learning Repository — Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult)
**Mirror used:** `jbrownlee/Datasets` (GitHub)
**Size:** 2,000 rows, balanced (1,000 positive / 1,000 negative), sampled from the full 45,222-row cleaned dataset
**License:** Public domain / open for research and educational use

### Features (after preprocessing)

| Column | Description |
|---|---|
| `age` | Age in years |
| `education_num` | Years of education |
| `hours_per_week` | Hours worked per week |
| `capital_gain` | Capital gains |
| `capital_loss` | Capital losses |
| `sex_male` | 1 = Male, 0 = Female |
| `is_married` | 1 = Married, 0 = Not married |
| `income_above_50k` | **Target** — 1 = income >$50K, 0 = income ≤$50K |

The dataset is committed directly to the repository (`data/adult_final.csv`) and uploaded to S3 programmatically by `run_pipeline.py` — no manual upload step is ever required, making the entire pipeline reproducible from a clean clone.

---

## Prerequisites

- AWS account with permissions to create: SageMaker resources, IAM roles, S3 buckets, Lambda functions, EventBridge rules, Secrets Manager secrets
- GitHub repository with Actions enabled
- A GitHub Personal Access Token (scopes: `repo`, `workflow`) stored in AWS Secrets Manager as `github-token`

---

## Setup

### 1. Create an IAM user for GitHub Actions

Attach a policy covering: `sagemaker:*`, S3 bucket/object actions, IAM role creation/PassRole, EventBridge rule management, Lambda function management, and CloudWatch Logs.

### 2. Add GitHub repository secrets

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `AWS_ROLE_ARN` | SageMaker execution role ARN |

### 3. Store the GitHub token in AWS Secrets Manager

```bash
aws secretsmanager create-secret \
  --name github-token \
  --secret-string "ghp_xxxxxxxxxxxxxxxxxxxx" \
  --region us-east-1
```

> Stored as **plaintext**, not JSON — the Lambda function reads `secret["SecretString"]` directly.

### 4. Update the repo reference inside the Lambda code

In `infra/setup_eventbridge.py`, update:

```python
repo = "YOUR_GITHUB_USERNAME/adult-income-sagemaker"
```

### 5. Push to trigger the pipeline

```bash
git push origin master
```

Pipeline 1 and Pipeline 2 will run automatically in sequence.

---

## Approving a Model and Triggering Deployment

1. After Pipeline 2 completes, go to **SageMaker Studio → Models → Logged → AdultIncomeGroup**
2. Select the latest version → **Update Status → Approved**
3. EventBridge detects the approval event and invokes the Lambda function
4. The Lambda dispatches Pipeline 3 via the GitHub Actions API
5. Pipeline 3 deploys the model to `adult-income-predictor-endpoint`

> **One-time setup note:** GitHub's `workflow_dispatch` API returns a 404 until a workflow file has been indexed at least once. `deploy-pipeline.yml` includes a path-filtered `push` trigger (combined with an `if` condition that prevents it from actually executing on push) solely to register the workflow with GitHub's API immediately on first push — this avoids requiring a manual "Run workflow" click before the first automated trigger works.

---

## Testing the Deployed Endpoint

### Via AWS CLI / CloudShell

**CSV:**
```bash
echo -n "39,13,40,2174,0,1,0" > input.csv
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name adult-income-predictor-endpoint \
  --content-type text/csv \
  --body fileb://input.csv \
  --region us-east-1 \
  result.txt
cat result.txt
```

**JSON:**
```bash
echo -n '{"instances": [[39, 13, 40, 2174, 0, 1, 0]]}' > input.json
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name adult-income-predictor-endpoint \
  --content-type application/json \
  --body fileb://input.json \
  --region us-east-1 \
  result.json
cat result.json
# {"predictions": [1]}
```

### Via SageMaker Studio Playground

```
Endpoints → adult-income-predictor-endpoint → Playground
Content type: application/json
Body: {"instances": [[39, 13, 40, 2174, 0, 1, 0]]}
```

The endpoint's `output_fn` explicitly returns a `(body, content_type)` tuple, which sets the correct `application/json` response header — required for Studio's Playground panel to render the prediction inline rather than showing "Content type is not supported for display."

---

## Cost Reference

| Component | Approx. Cost |
|---|---|
| Full pipeline run (DataPrep + Train + Eval + Deploy setup) | ~$0.025 (~₹2) per run |
| Model training (`ml.m5.large`, Spot) | Up to ~90% cheaper than On-Demand |
| **Endpoint left running 24/7** | **~$82.80/month (~₹6,900/month)** |
| S3 storage (dataset + artifacts) | ~$0.01/month |

> **The dominant cost driver is a forgotten live endpoint, not pipeline execution.** Always delete the endpoint after testing if it isn't serving production traffic:
> ```bash
> aws sagemaker delete-endpoint --endpoint-name adult-income-predictor-endpoint --region us-east-1
> ```

---

## Key Design Decisions

- **Binary classification, not multi-class** — predicting `income_above_50k` (1/0) rather than any multi-category target keeps the problem well-posed for a small dataset.
- **Spot instances for training only** — SageMaker Processing jobs do not support Spot; only `TrainingStep` does, so DataPreparation and Evaluation run on On-Demand `ml.t3.medium`.
- **Quality gate before registration** — a `ConditionStep` checks evaluation accuracy (≥ 0.75) before allowing `RegisterModel` to run, preventing underperforming models from ever reaching the registry.
- **Manual approval gate before deployment** — models are never auto-deployed; a human must explicitly approve in SageMaker Model Registry, which then triggers deployment automatically via EventBridge.
- **Code-based S3 sync, not manual upload** — the dataset is committed to the repo and uploaded via `boto3` inside `run_pipeline.py`, making the entire system reproducible from a fresh clone with zero manual AWS console steps.
- **Explicit `model_fn` / `input_fn` / `predict_fn` / `output_fn`** — required for the SKLearn serving container to correctly handle real-time inference requests (the default container otherwise fails on raw CSV row shaping and JSON content-type negotiation).

---

## License

Dataset: UCI Machine Learning Repository (Adult / Census Income), public domain for research and educational use.
Code: MIT (or your preferred license).
