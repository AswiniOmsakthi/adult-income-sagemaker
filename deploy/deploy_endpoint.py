import boto3
import sagemaker
import os
import json
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.sklearn import SKLearn, SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile


def run_pipeline():

    # ─── Setup ──────────────────────────────────────────────
    region  = os.environ.get("AWS_REGION", "us-east-1")
    account = boto3.client("sts", region_name=region)\
                   .get_caller_identity()["Account"]
    bucket  = f"sagemaker-adult-income-pipeline-{account}"

    boto_session = boto3.Session(region_name=region)
    sm_client    = boto3.client("sagemaker", region_name=region)
    s3_client    = boto3.client("s3", region_name=region)

    # ─── Create Bucket if Not Exists ────────────────────────
    try:
        s3_client.create_bucket(Bucket=bucket)
        print(f"✅ Bucket created : {bucket}")
    except Exception:
        print(f"✅ Bucket exists  : {bucket}")

    # ─── ✅ UPLOAD DATA TO S3 VIA CODE (no manual upload) ───
    print("⏳ Uploading dataset from repo to S3...")
    local_data_path = "data/adult_final.csv"
    s3_client.upload_file(local_data_path, bucket, "data/adult_final.csv")
    print(f"✅ Data uploaded to s3://{bucket}/data/adult_final.csv")

    # ─── Read Infra Outputs from S3 ─────────────────────────
    try:
        response  = s3_client.get_object(
            Bucket=bucket,
            Key="infra/infra_outputs.json"
        )
        infra     = json.loads(response["Body"].read())
        role      = infra["role_arn"]
        domain_id = infra["domain_id"]
        bucket    = infra["bucket_name"]

        print(f"✅ Infra outputs loaded")
        print(f"✅ Domain ID : {domain_id}")
        print(f"✅ Role ARN  : {role}")
        print(f"✅ Bucket    : {bucket}")

    except Exception as e:
        print(f"⚠️ S3 read failed : {str(e)}")
        role      = os.environ.get("AWS_ROLE_ARN")
        domain_id = os.environ.get("DOMAIN_ID", "")
        print(f"✅ Fallback Role   : {role}")

<<<<<<< HEAD
=======
    # ─── Get Latest Approved Model ──────────────────────────
    try:
        model_packages = sm_client.list_model_packages(
            ModelPackageGroupName="IPLMatchPredictionGroup-New",
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1
        )

        if not model_packages["ModelPackageSummaryList"]:
            raise Exception(
                "No approved model found in IPLMatchPredictionGroup-New! "
                "Please approve model first."
            )

        model_package_arn = model_packages[
            "ModelPackageSummaryList"
        ][0]["ModelPackageArn"]

        model_version = model_packages[
            "ModelPackageSummaryList"
        ][0]["ModelPackageVersion"]

        package_detail = sm_client.describe_model_package(
            ModelPackageName=model_package_arn
        )
        model_data_url = package_detail["InferenceSpecification"]\
                          ["Containers"][0]["ModelDataUrl"]

        print(f"✅ Model Package ARN     : {model_package_arn}")
        print(f"✅ Model Package Version : {model_version}")
        print(f"✅ Model Data URL        : {model_data_url}")

    except Exception as e:
        print(f"❌ Model fetch error: {str(e)}")
        raise

    # ─── Create SageMaker Session ───────────────────────────
>>>>>>> 6c7866ae8e12f4a2c35d2e6a6afbde50769cf491
    session = sagemaker.Session(
        boto_session=boto_session,
        sagemaker_client=sm_client,
        default_bucket=bucket
    )

<<<<<<< HEAD
    # ─── Step 1: Data Preparation ───────────────────────────
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        sagemaker_session=session
=======
    # ─── Create Model WITH Inference Entry Point ────────────
    model_name    = f"ipl-match-predictor-{int(time.time())}"
    endpoint_name = "ipl-match-predictor-endpoint"
    config_name   = f"ipl-endpoint-config-{int(time.time())}"  # ✅ Unique

    try:
        model = SKLearnModel(
            model_data=model_data_url,
            role=role,
            entry_point="train.py",
            source_dir="pipeline",
            framework_version="1.2-1",
            sagemaker_session=session,
            name=model_name
        )
        print(f"✅ Model object created : {model_name}")

    except Exception as e:
        print(f"❌ Model creation error: {str(e)}")
        raise

    # ─── Delete Old Endpoint + Config if Exists ─────────────
    try:
        existing        = sm_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        endpoint_status = existing["EndpointStatus"]
        print(f"✅ Endpoint exists : {endpoint_name}")
        print(f"✅ Status          : {endpoint_status}")

        print(f"⏳ Deleting old endpoint...")
        sm_client.delete_endpoint(EndpointName=endpoint_name)

        while True:
            try:
                sm_client.describe_endpoint(EndpointName=endpoint_name)
                time.sleep(10)
            except sm_client.exceptions.ClientError:
                break
        print(f"✅ Old endpoint deleted")

    except sm_client.exceptions.ClientError:
        print(f"⏳ No existing endpoint found, creating fresh one")

    # ─── Delete Old Endpoint Configs (any leftover ones) ────
    try:
        configs = sm_client.list_endpoint_configs(
            NameContains="ipl-endpoint-config"
        )
        for cfg in configs.get("EndpointConfigs", []):
            try:
                sm_client.delete_endpoint_config(
                    EndpointConfigName=cfg["EndpointConfigName"]
                )
                print(f"✅ Deleted old config: {cfg['EndpointConfigName']}")
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ Config cleanup skipped: {str(e)}")

    # ─── Create Model in SageMaker ──────────────────────────
    model.create(instance_type="ml.m5.large")
    print(f"✅ Model created in SageMaker")

    # ─── Create New Endpoint Config ─────────────────────────
    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName"         : "AllTraffic",
                "ModelName"           : model_name,
                "InitialInstanceCount": 1,
                "InstanceType"        : "ml.m5.large",
                "InitialVariantWeight": 1
            }
        ]
    )
    print(f"✅ Endpoint config created : {config_name}")

    # ─── Create Endpoint ─────────────────────────────────────
    sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name
    )
    print(f"✅ Endpoint creation started!")

    # ─── Wait for Endpoint InService ────────────────────────
    print(f"⏳ Waiting for endpoint to be InService...")

    while True:
        status = sm_client.describe_endpoint(
            EndpointName=endpoint_name
        )["EndpointStatus"]

        print(f"   Status: {status}")

        if status == "InService":
            break
        elif status in ["Failed", "OutOfService"]:
            detail = sm_client.describe_endpoint(EndpointName=endpoint_name)
            raise Exception(f"Endpoint failed: {detail.get('FailureReason', status)}")

        time.sleep(30)

    print(f"✅ Endpoint is InService!")

    # ─── Test Endpoint ───────────────────────────────────────
    print(f"\n⏳ Testing endpoint...")

    test_cases = [
        ("1,1,0,6",   "Team won toss + bat + won by 6 wickets"),
        ("0,0,0,0",   "Team lost toss + field + no result"),
        ("1,0,25,0",  "Team won toss + field + won by 25 runs"),
    ]

    for test_input, description in test_cases:
        response   = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=test_input
        )
        prediction = response["Body"].read().decode("utf-8").strip()
        result     = "WIN ✅" if prediction == "1" else "LOSS ❌"

        print(f"   Input      : {test_input}")
        print(f"   Scenario   : {description}")
        print(f"   Prediction : {result}")
        print()

    # ─── Save Deploy Outputs to S3 ──────────────────────────
    deploy_outputs = {
        "endpoint_name"    : endpoint_name,
        "model_name"       : model_name,
        "model_package_arn": model_package_arn,
        "model_version"    : model_version,
        "group_name"       : "IPLMatchPredictionGroup-New",
        "domain_id"        : domain_id,
        "status"           : "InService",
        "region"           : region
    }

    s3_client.put_object(
        Bucket=bucket,
        Key="deploy/deploy_outputs.json",
        Body=json.dumps(deploy_outputs, indent=2)
>>>>>>> 6c7866ae8e12f4a2c35d2e6a6afbde50769cf491
    )

    data_prep_step = ProcessingStep(
        name="DataPreparation",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{bucket}/data/adult_final.csv",
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket}/processed/train"
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket}/processed/test"
            )
        ],
        code="pipeline/preprocessing.py"
    )

    # ─── Step 2: Model Training ─────────────────────────────
    estimator = SKLearn(
        entry_point="train.py",
        source_dir="pipeline",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version="1.2-1",
        use_spot_instances=True,
        max_run=3600,
        max_wait=7200,
        sagemaker_session=session,
        metric_definitions=[
            {"Name": "accuracy", "Regex": "accuracy: ([0-9\\.]+)"}
        ]
    )

    train_step = TrainingStep(
        name="ModelTraining",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=data_prep_step.properties\
                        .ProcessingOutputConfig\
                        .Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )

    # ─── Step 3: Model Evaluation ───────────────────────────
    evaluator = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        sagemaker_session=session
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )

    eval_step = ProcessingStep(
        name="ModelEvaluation",
        processor=evaluator,
        inputs=[
            ProcessingInput(
                source=train_step.properties\
                       .ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=data_prep_step.properties\
                       .ProcessingOutputConfig\
                       .Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/evaluation"
            )
        ],
        code="pipeline/evaluate.py",
        property_files=[evaluation_report]
    )

    # ─── Step 4: Model Registration ─────────────────────────
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"s3://{bucket}/evaluation/evaluation.json",
            content_type="application/json"
        )
    )

    register_step = RegisterModel(
        name="ModelRegistration",
        estimator=estimator,
        model_data=train_step.properties\
                  .ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="AdultIncomeGroup",
        approval_status="PendingManualApproval",
        model_metrics=model_metrics
    )

    # ─── Step 5: Quality Gate ───────────────────────────────
    accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=eval_step.name,
            property_file=evaluation_report,
            json_path="metrics.accuracy.value"
        ),
        right=0.75
    )

    condition_step = ConditionStep(
        name="QualityGate",
        conditions=[accuracy_condition],
        if_steps=[register_step],
        else_steps=[]
    )

    # ─── Build & Run Pipeline ───────────────────────────────
    pipeline = Pipeline(
        name="AdultIncomePredictionPipeline",
        steps=[
            data_prep_step,
            train_step,
            eval_step,
            condition_step
        ],
        sagemaker_session=session
    )

    pipeline.upsert(role_arn=role)
    execution = pipeline.start()

    print(f"✅ Pipeline Started!")
    print(f"✅ Execution ARN : {execution.arn}")
    print(f"✅ Monitor here  : SageMaker → Pipelines → AdultIncomePredictionPipeline")


if __name__ == "__main__":
<<<<<<< HEAD
    run_pipeline()
=======
    deploy_model()
>>>>>>> 6c7866ae8e12f4a2c35d2e6a6afbde50769cf491
