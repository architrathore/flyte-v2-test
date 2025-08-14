# backtest.py
#
# Multi-Model Backtesting Workflow
#
# This workflow supports training and evaluating multiple machine learning models:
# - GBM (Gradient Boosting Machine) models using LightGBM/XGBoost
# - DNN (Deep Neural Network) models using PyTorch/TensorFlow
# - Ensemble models combining GBM and DNN predictions
#
# The workflow follows the pattern from the fraud detection backtest workflow:
# /Users/architr/stripe/zoolander/src/python/flyte/fraud_detection/workflows/backtest/
#
# Run using
# ```
# source .venv/bin/activate
# uv run --prerelease allow backtest.py
# ```
#
# /// script
# requires-python = ">=3.11"
# dependencies = [
#    "flyte==0.2.0b23",
#    "pydantic>=2.0.0",
#    "pandas>=2.0.0",
#    "loguru>=0.7.3",
# ]
# ///

import asyncio
import logging
from enum import Enum
from typing import Any

import flyte
import flyte.report
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Task Environment
# ============================================================================

# Create task environment for backtest workflow
env = flyte.TaskEnvironment(
    name="backtest_workflow",
    resources=flyte.Resources(memory="2Gi", cpu="1"),
    # reusable=flyte.ReusePolicy(
    #     replicas=5,
    #     idle_ttl=300
    # ),
)


# ============================================================================
# HTML Report Helper Functions
# ============================================================================


def create_html_header(title: str, color: str = "#2c3e50") -> str:
    """Create a standardized HTML header with title and color."""
    return f"""
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
        <h2 style="color: {color}; border-bottom: 2px solid {color}; padding-bottom: 10px;">
            {title}
        </h2>
    """


def create_config_table(config_data: dict[str, Any]) -> str:
    """Create a standardized configuration table from a dictionary."""
    rows = []
    for key, value in config_data.items():
        if isinstance(value, str) and value.startswith("s3://"):
            # Format S3 paths as code
            formatted_value = "<code>" + value + "</code>"
        else:
            formatted_value = str(value)

        rows.append(f"""
        <tr>
            <td style="padding: 10px; border: 1px solid #95a5a6;"><strong>{key}</strong></td>
            <td style="padding: 10px; border: 1px solid #95a5a6;">{formatted_value}</td>
        </tr>
        """)

    return f"""
    <div style="background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h3 style="color: #34495e; margin-top: 0;">Configuration</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #bdc3c7;">
                <th style="padding: 10px; text-align: left; border: 1px solid #95a5a6;">Parameter</th>
                <th style="padding: 10px; text-align: left; border: 1px solid #95a5a6;">Value</th>
            </tr>
            {"".join(rows)}
        </table>
    </div>
    """


def create_success_section(message: str, details: list[str] | None = None) -> str:
    """Create a standardized success section with optional details."""
    details_html = ""
    if details:
        details_html = (
            "<ul>" + "".join([f"<li>{detail}</li>" for detail in details]) + "</ul>"
        )

    return f"""
    <div style="background: #e8f5e8; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h3 style="color: #27ae60; margin-top: 0;">✅ {message}</h3>
        {details_html}
    </div>
    """


def create_note_section(note: str) -> str:
    """Create a standardized note section."""
    return f"""
    <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <p style="margin: 0; color: #856404;">
            <strong>Note:</strong> {note}
        </p>
    </div>
    """


def create_metrics_grid(metrics: list[dict[str, Any]]) -> str:
    """Create a grid of metric cards."""
    cards = []
    for metric in metrics:
        cards.append(f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;">
            <h4 style="color: #495057; margin: 0 0 10px 0;">{metric["title"]}</h4>
            <p style="font-size: 24px; font-weight: bold; color: {metric["color"]}; margin: 0;">{metric["value"]}</p>
            <p style="color: #6c757d; margin: 5px 0 0 0;">{metric["subtitle"]}</p>
        </div>
        """)

    return f"""
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0;">
        {"".join(cards)}
    </div>
    """


def create_model_metrics_section(model_name: str, metrics: list[dict[str, Any]]) -> str:
    """Create a section for model-specific metrics."""
    metric_cards = []
    for metric in metrics:
        metric_cards.append(f"""
        <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
            <h5 style="color: #495057; margin: 0 0 10px 0;">{metric["title"]}</h5>
            <p style="font-size: 20px; font-weight: bold; color: {metric["color"]}; margin: 0;">{metric["value"]}</p>
        </div>
        """)

    return f"""
    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h4 style="color: #495057; margin: 0 0 15px 0;">{model_name}</h4>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            {"".join(metric_cards)}
        </div>
    </div>
    """


def generate_report_html(
    title: str,
    color: str,
    config_data: dict[str, Any],
    success_message: str,
    success_details: list[str] | None = None,
    note: str | None = None,
) -> str:
    """Generate a complete report HTML with all components."""
    html_parts = [
        create_html_header(title, color),
        create_config_table(config_data),
        create_success_section(success_message, success_details),
    ]

    if note:
        html_parts.append(create_note_section(note))

    html_parts.append("</div>")
    return "".join(html_parts)


# ============================================================================
# Data Types (using Pydantic v2 instead of dataclasses)
# ============================================================================


class ModelType(str, Enum):
    CARD_CASHING = "card_cashing"
    FRAUD_BIZ = "fraud_biz"
    CARD_TESTING = "card_testing"


class LayerType(str, Enum):
    FEATURES = "features"
    MODEL = "model"
    CALIBRATION = "calibration"


class TemporalBounds(BaseModel):
    start_date: str
    end_date: str


class DatasetConfig(BaseModel):
    model_type: ModelType
    temporal_bounds: TemporalBounds


class TrainConfig(BaseModel):
    label: str
    gbm_config: dict[str, Any] | None = None
    dnn_config: dict[str, Any] | None = None
    ensemble_config: dict[str, Any] | None = None

    def validate_model_configs(self):
        """Validate that at least one model config is provided and ensemble config is valid"""
        # At least one model config should be provided
        if not any([self.gbm_config, self.dnn_config]):
            raise ValueError(
                "At least one of gbm_config or dnn_config should be provided"
            )

        # If both models are provided, ensemble config should also be provided
        if self.gbm_config and self.dnn_config and not self.ensemble_config:
            raise ValueError(
                "Both gbm_config and dnn_config are provided, ensemble_config should also be provided"
            )

        # If only one model is provided, ensemble config should not be provided
        if (not self.gbm_config or not self.dnn_config) and self.ensemble_config:
            raise ValueError(
                "Only single base model config is provided, ensemble_config should not be provided"
            )

        # If ensemble config is provided, validate weights sum to 1
        if self.ensemble_config:
            weights = self.ensemble_config.get("model_weights", {})
            if abs(sum(weights.values()) - 1.0) > 1e-5:
                raise ValueError(
                    f"model_weights in ensemble_config should sum to 1, got {weights}"
                )


class EvalConfig(BaseModel):
    n_bootstraps: int = 5
    report_spec_name: str | None = None


class ExperimentConfig(BaseModel):
    dataset_config: DatasetConfig
    train_config: TrainConfig
    eval_config: EvalConfig


class PreparedData(BaseModel):
    train_path: str
    validation_path: str
    calibration_path: str
    test_path: str


class ExperimentArtifact(BaseModel):
    experiment_id: str
    prepared_data: PreparedData
    model_sha: str | None = None
    scores_path: str | None = None
    reports: dict[str, str] = Field(default_factory=dict)


# ============================================================================
# Experiment Registry
# ============================================================================


class ExperimentRegistry:
    def __init__(self):
        self._experiments: dict[str, ExperimentConfig] = {}

    def register(self, experiment_id: str, config: ExperimentConfig):
        self._experiments[experiment_id] = config

    def get(self, experiment_id: str) -> ExperimentConfig:
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found in registry")
        return self._experiments[experiment_id]

    def list_experiments(self) -> list[str]:
        return list(self._experiments.keys())


# Create global registry
registry = ExperimentRegistry()


# Register some example experiments
def register_example_experiments():
    """Register example experiments in the registry"""

    # Baseline card cashing experiment (GBM only)
    baseline_cc_config = ExperimentConfig(
        dataset_config=DatasetConfig(
            model_type=ModelType.CARD_CASHING,
            temporal_bounds=TemporalBounds(
                start_date="2024-01-01", end_date="2024-01-31"
            ),
        ),
        train_config=TrainConfig(label="is_fraud", gbm_config={"n_estimators": 100}),
        eval_config=EvalConfig(n_bootstraps=5),
    )
    registry.register("baseline_cc", baseline_cc_config)

    # Baseline fraud business experiment (GBM only)
    baseline_fb_config = ExperimentConfig(
        dataset_config=DatasetConfig(
            model_type=ModelType.FRAUD_BIZ,
            temporal_bounds=TemporalBounds(
                start_date="2024-01-01", end_date="2024-01-31"
            ),
        ),
        train_config=TrainConfig(
            label="is_fraud_biz", gbm_config={"n_estimators": 100}
        ),
        eval_config=EvalConfig(n_bootstraps=5),
    )
    registry.register("baseline_fb", baseline_fb_config)

    # Multi-model ensemble experiment (DNN + XGB + Ensemble)
    ensemble_experiment_config = ExperimentConfig(
        dataset_config=DatasetConfig(
            model_type=ModelType.CARD_CASHING,
            temporal_bounds=TemporalBounds(
                start_date="2024-01-01", end_date="2024-01-31"
            ),
        ),
        train_config=TrainConfig(
            label="is_fraud",
            gbm_config={
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "num_leaves": 31,
            },
            dnn_config={
                "hidden_layers": [256, 128, 64],
                "dropout_rate": 0.3,
                "learning_rate": 0.001,
                "batch_size": 1024,
                "epochs": 50,
            },
            ensemble_config={"model_weights": {"gbm": 0.6, "dnn": 0.4}},
        ),
        eval_config=EvalConfig(n_bootstraps=5),
    )
    registry.register("ensemble_experiment", ensemble_experiment_config)

    # DNN-only experiment
    dnn_experiment_config = ExperimentConfig(
        dataset_config=DatasetConfig(
            model_type=ModelType.CARD_CASHING,
            temporal_bounds=TemporalBounds(
                start_date="2024-01-01", end_date="2024-01-31"
            ),
        ),
        train_config=TrainConfig(
            label="is_fraud",
            dnn_config={
                "hidden_layers": [512, 256, 128],
                "dropout_rate": 0.4,
                "learning_rate": 0.0005,
                "batch_size": 512,
                "epochs": 100,
            },
        ),
        eval_config=EvalConfig(n_bootstraps=5),
    )
    registry.register("dnn_experiment", dnn_experiment_config)


# ============================================================================
# Core Workflow Tasks
# ============================================================================


@env.task(report=True)
async def parse_experiment_ids(experiment_ids_str: str) -> list[str]:
    """Parse space-separated experiment IDs into a list"""
    return experiment_ids_str.strip().split()


@env.task(report=True)
async def prepare_data(
    dataset_config: DatasetConfig, experiment_id: str, snapshot_date: str | None
) -> PreparedData:
    """
    Prepare training and test datasets for the experiment.

    This is a stub implementation that would normally:
    - Load raw data from various sources
    - Apply filters and transformations
    - Split into train/validation/calibration/test sets
    - Save to S3 paths
    """
    # Create HTML report for data preparation
    config_data = {
        "Model Type": dataset_config.model_type.value,
        "Start Date": dataset_config.temporal_bounds.start_date,
        "End Date": dataset_config.temporal_bounds.end_date,
        "Snapshot Date": snapshot_date or "Not specified",
    }

    success_details = [
        f"<strong>Training:</strong> {experiment_id}/train.parquet",
        f"<strong>Validation:</strong> {experiment_id}/validation.parquet",
        f"<strong>Calibration:</strong> {experiment_id}/calibration.parquet",
        f"<strong>Test:</strong> {experiment_id}/test.parquet",
    ]

    html = generate_report_html(
        title=f"📊 Data Preparation Report - {experiment_id}",
        color="#3498db",
        config_data=config_data,
        success_message="Data Preparation Complete",
        success_details=success_details,
        note="This is a stub implementation. In production, this would perform complex ETL operations including data filtering, feature engineering, and dataset splitting.",
    )

    flyte.report.replace(html)

    # Create a second tab for detailed statistics
    stats_tab = flyte.report.get_tab("Data Statistics")

    metrics = [
        {
            "title": "Training Set",
            "value": "10,000",
            "color": "#007bff",
            "subtitle": "samples",
        },
        {
            "title": "Test Set",
            "value": "2,500",
            "color": "#28a745",
            "subtitle": "samples",
        },
        {
            "title": "Features",
            "value": "150",
            "color": "#ffc107",
            "subtitle": "dimensions",
        },
        {
            "title": "Positive Rate",
            "value": "5.2%",
            "color": "#dc3545",
            "subtitle": "fraud cases",
        },
    ]

    stats_html = f"""
    <div style="font-family: Arial, sans-serif;">
        <h3 style="color: #2c3e50;">📈 Mock Dataset Statistics</h3>
        {create_metrics_grid(metrics)}
    </div>
    """

    stats_tab.log(stats_html)

    flyte.report.flush()

    # Return mock paths
    return PreparedData(
        train_path=f"s3://mock-bucket/{experiment_id}/train.parquet",
        validation_path=f"s3://mock-bucket/{experiment_id}/validation.parquet",
        calibration_path=f"s3://mock-bucket/{experiment_id}/calibration.parquet",
        test_path=f"s3://mock-bucket/{experiment_id}/test.parquet",
    )


@env.task(report=True)
async def train_gbm_model(
    prepared_data: PreparedData,
    gbm_config: dict[str, Any],
    label: str,
    experiment_id: str,
) -> str:
    """
    Train a GBM (Gradient Boosting Machine) model.

    This is a stub implementation that would normally:
    - Load training data
    - Train LightGBM/XGBoost model
    - Save model artifacts
    - Return model SHA
    """
    # Create HTML report for GBM training
    config_data = {
        "Target Label": label,
        "N Estimators": gbm_config.get("n_estimators", "Default"),
        "Learning Rate": gbm_config.get("learning_rate", "Default"),
        "Max Depth": gbm_config.get("max_depth", "Default"),
    }

    model_sha = f"gbm_sha_{experiment_id}_{hash(str(gbm_config))}"

    html = generate_report_html(
        title=f"🌳 GBM Model Training - {experiment_id}",
        color="#28a745",
        config_data=config_data,
        success_message="GBM Training Complete",
        success_details=[f"<strong>Model SHA:</strong> <code>{model_sha}</code>"],
    )

    flyte.report.replace(html)

    flyte.report.flush()

    # Return mock GBM model SHA
    return f"gbm_sha_{experiment_id}_{hash(str(gbm_config))}"


@env.task(report=True)
async def train_dnn_model(
    prepared_data: PreparedData,
    dnn_config: dict[str, Any],
    label: str,
    experiment_id: str,
) -> str:
    """
    Train a DNN (Deep Neural Network) model.

    This is a stub implementation that would normally:
    - Load training data
    - Train PyTorch/TensorFlow neural network
    - Save model artifacts
    - Return model SHA
    """
    # Create HTML report for DNN training
    config_data = {
        "Target Label": label,
        "Hidden Layers": dnn_config.get("hidden_layers", "Default"),
        "Dropout Rate": dnn_config.get("dropout_rate", "Default"),
        "Learning Rate": dnn_config.get("learning_rate", "Default"),
        "Batch Size": dnn_config.get("batch_size", "Default"),
        "Epochs": dnn_config.get("epochs", "Default"),
    }

    model_sha = f"dnn_sha_{experiment_id}_{hash(str(dnn_config))}"

    html = generate_report_html(
        title=f"🧠 DNN Model Training - {experiment_id}",
        color="#9b59b6",
        config_data=config_data,
        success_message="DNN Training Complete",
        success_details=[f"<strong>Model SHA:</strong> <code>{model_sha}</code>"],
    )

    flyte.report.replace(html)

    flyte.report.flush()

    # Return mock DNN model SHA
    return f"dnn_sha_{experiment_id}_{hash(str(dnn_config))}"


@env.task(report=True)
async def create_ensemble_model(
    gbm_sha: str, dnn_sha: str, ensemble_config: dict[str, Any], experiment_id: str
) -> str:
    """
    Create an ensemble model combining GBM and DNN predictions.

    This is a stub implementation that would normally:
    - Load trained GBM and DNN models
    - Create ensemble with specified weights
    - Save ensemble model artifacts
    - Return ensemble model SHA
    """
    # Create HTML report for ensemble creation
    config_data = {
        "GBM Model SHA": f"<code>{gbm_sha}</code>",
        "DNN Model SHA": f"<code>{dnn_sha}</code>",
        "Model Weights": ensemble_config.get("model_weights", {}),
    }

    ensemble_sha = f"ensemble_sha_{experiment_id}_{hash(str(ensemble_config))}"

    html = generate_report_html(
        title=f"🎯 Ensemble Model Creation - {experiment_id}",
        color="#f39c12",
        config_data=config_data,
        success_message="Ensemble Creation Complete",
        success_details=[
            f"<strong>Ensemble Model SHA:</strong> <code>{ensemble_sha}</code>"
        ],
    )

    flyte.report.replace(html)

    flyte.report.flush()

    # Return mock ensemble model SHA
    return f"ensemble_sha_{experiment_id}_{hash(str(ensemble_config))}"


@env.task(report=True)
async def train_model(
    prepared_data: PreparedData, train_config: TrainConfig, experiment_id: str
) -> str:
    """
    Train machine learning models on the prepared data.

    This function supports training:
    - GBM (Gradient Boosting Machine) models
    - DNN (Deep Neural Network) models
    - Ensemble models combining GBM and DNN

    This is a stub implementation that would normally:
    - Load training data
    - Train individual models (GBM, DNN)
    - Create ensemble if multiple models are specified
    - Save model artifacts
    - Return model SHA
    """
    # Validate model configurations
    train_config.validate_model_configs()

    # Train individual models
    gbm_sha = None
    dnn_sha = None

    if train_config.gbm_config:
        logger.info(f"Training GBM model for experiment {experiment_id}")
        gbm_sha = train_gbm_model(
            prepared_data=prepared_data,
            gbm_config=train_config.gbm_config,
            label=train_config.label,
            experiment_id=experiment_id,
        )

    if train_config.dnn_config:
        logger.info(f"Training DNN model for experiment {experiment_id}")
        dnn_sha = train_dnn_model(
            prepared_data=prepared_data,
            dnn_config=train_config.dnn_config,
            label=train_config.label,
            experiment_id=experiment_id,
        )

    # Create ensemble if both models are trained and ensemble config is provided
    if train_config.ensemble_config and gbm_sha and dnn_sha:
        logger.info(f"Creating ensemble model for experiment {experiment_id}")
        gbm_sha, dnn_sha = await asyncio.gather(gbm_sha, dnn_sha)
        final_model_sha = await create_ensemble_model(
            gbm_sha=gbm_sha,
            dnn_sha=dnn_sha,
            ensemble_config=train_config.ensemble_config,
            experiment_id=experiment_id,
        )
    elif gbm_sha:
        logger.info(f"Using GBM model for experiment {experiment_id}")
        final_model_sha = await gbm_sha
    elif dnn_sha:
        logger.info(f"Using DNN model for experiment {experiment_id}")
        final_model_sha = await dnn_sha
    else:
        raise ValueError("No models were successfully trained")

    logger.info(f"Final model SHA for experiment {experiment_id}: {final_model_sha}")
    # Return the final model SHA
    return final_model_sha


@env.task(report=True)
async def score_data(
    prepared_data: PreparedData, model_sha: str, experiment_id: str
) -> str:
    """
    Score test data using the trained model.

    This is a stub implementation that would normally:
    - Load the trained model
    - Load test data
    - Generate predictions
    - Save scores to S3
    """
    logger.info(f"Scoring data for experiment {experiment_id}, model {model_sha}")
    # Return mock scores path
    return f"s3://mock-bucket/{experiment_id}/scores.parquet"


@env.task(report=True)
async def generate_reports(
    prepared_data: PreparedData,
    scores_path: str,
    eval_config: EvalConfig,
    experiment_id: str,
) -> dict[str, str]:
    """
    Generate evaluation reports from scored data.

    This is a stub implementation that would normally:
    - Load scored data
    - Generate various evaluation metrics
    - Create visualizations
    - Save reports as notebooks/HTML
    """
    logger.info(
        f"Generating reports for experiment {experiment_id}, scores_path {scores_path}"
    )

    # Create HTML report for report generation
    config_data = {
        "Number of Bootstraps": eval_config.n_bootstraps,
        "Report Spec": eval_config.report_spec_name or "Default",
        "Scores Path": scores_path,
    }

    success_details = [
        f"<strong>Evaluation Report:</strong> <code>s3://mock-bucket/{experiment_id}/evaluation_report.html</code>",
        f"<strong>Calibration Report:</strong> <code>s3://mock-bucket/{experiment_id}/calibration_report.html</code>",
        f"<strong>Feature Importance:</strong> <code>s3://mock-bucket/{experiment_id}/feature_importance.html</code>",
    ]

    html = generate_report_html(
        title=f"📋 Report Generation - {experiment_id}",
        color="#f39c12",
        config_data=config_data,
        success_message="Reports Generated",
        success_details=success_details,
        note="This is a stub implementation. In production, this would generate actual evaluation metrics, ROC curves, precision-recall plots, and feature importance visualizations.",
    )

    flyte.report.replace(html)

    # Create a second tab for evaluation metrics
    metrics_tab = flyte.report.get_tab("Evaluation Metrics")

    metrics = [
        {
            "title": "Test AUC",
            "value": "0.867",
            "color": "#007bff",
            "subtitle": "ROC AUC",
        },
        {
            "title": "Precision",
            "value": "0.823",
            "color": "#28a745",
            "subtitle": "at 0.5 threshold",
        },
        {
            "title": "Recall",
            "value": "0.756",
            "color": "#ffc107",
            "subtitle": "at 0.5 threshold",
        },
        {
            "title": "F1 Score",
            "value": "0.788",
            "color": "#dc3545",
            "subtitle": "harmonic mean",
        },
    ]

    metrics_html = f"""
    <div style="font-family: Arial, sans-serif;">
        <h3 style="color: #2c3e50;">📊 Mock Evaluation Results</h3>
        {create_metrics_grid(metrics)}
    </div>
    """

    metrics_tab.log(metrics_html)

    flyte.report.flush()

    # Return mock report paths
    return {
        "evaluation_report": f"s3://mock-bucket/{experiment_id}/evaluation_report.html",
        "calibration_report": f"s3://mock-bucket/{experiment_id}/calibration_report.html",
        "feature_importance": f"s3://mock-bucket/{experiment_id}/feature_importance.html",
    }


@env.task(report=True, cache=None)
async def run_single_experiment(
    experiment_id: str,
    snapshot_date: str | None,
) -> ExperimentArtifact:
    """
    Run a single experiment end-to-end.

    This orchestrates the full pipeline for one experiment:
    1. Get experiment config from registry
    2. Prepare data
    3. Train model
    4. Score data
    5. Generate reports
    """
    # Register experiments in the task environment
    register_example_experiments()

    # Get experiment configuration from registry
    config = registry.get(experiment_id)

    # Prepare data
    logger.info(f"Preparing data for experiment {experiment_id}")
    prepared_data = await prepare_data(
        dataset_config=config.dataset_config,
        experiment_id=experiment_id,
        snapshot_date=snapshot_date,
    )

    # Train model
    logger.info(f"Training model for experiment {experiment_id}")
    model_sha = await train_model(
        prepared_data=prepared_data,
        train_config=config.train_config,
        experiment_id=experiment_id,
    )

    # Score data
    logger.info(f"Scoring data for experiment {experiment_id}")
    scores_path = await score_data(
        prepared_data=prepared_data, model_sha=model_sha, experiment_id=experiment_id
    )

    # Generate reports
    logger.info(f"Generating reports for experiment {experiment_id}")
    reports = await generate_reports(
        prepared_data=prepared_data,
        scores_path=scores_path,
        eval_config=config.eval_config,
        experiment_id=experiment_id,
    )

    logger.info(f"Experiment {experiment_id} completed")
    return ExperimentArtifact(
        experiment_id=experiment_id,
        prepared_data=prepared_data,
        model_sha=model_sha,
        scores_path=scores_path,
        reports=reports,
    )


@env.task(report=True, cache=None)
async def generate_summary_report(
    experiment_artifacts: list[ExperimentArtifact],
) -> str:
    """
    Generate a summary report comparing all experiments.

    This is a stub implementation that would normally:
    - Compare metrics across experiments
    - Create comparison visualizations
    - Generate summary statistics
    """
    logger.info("Generating summary report")
    logger.info(f"Number of experiments: {len(experiment_artifacts)}")

    for artifact in experiment_artifacts:
        logger.info(f"Experiment {artifact.experiment_id}: {artifact.reports}")

    return "s3://mock-bucket/summary_report.html"


# ============================================================================
# Main Workflow
# ============================================================================


@env.task(report=True, cache=None)
async def backtest(
    experiment_ids: str = "baseline_cc",
    # snapshot_date: str | None = None,
) -> dict[str, str]:
    logger.info(f"Starting backtest workflow with experiment_ids: {experiment_ids}")
    """
    Main backtest workflow that orchestrates the entire pipeline.
    
    This workflow:
    1. Parses experiment IDs
    2. Runs each experiment in parallel
    3. Runs legacy baselines if requested
    4. Generates summary report
    5. Returns all report URLs
    """
    # Parse experiment IDs
    experiment_id_list = await parse_experiment_ids(experiment_ids)

    snapshot_date = "2025-01-15"

    # Run experiments sequentially (since we're using async)
    experiment_artifacts = await asyncio.gather(
        *[
            run_single_experiment(experiment_id=exp_id, snapshot_date=snapshot_date)
            for exp_id in experiment_id_list
        ]
    )

    # Generate summary report
    summary_report = await generate_summary_report(experiment_artifacts)

    # Collect all report URLs
    all_reports = {"summary_report": summary_report}
    for artifact in experiment_artifacts:
        for report_name, report_url in artifact.reports.items():
            all_reports[f"{artifact.experiment_id}_{report_name}"] = report_url

    logger.info(
        f"Backtest workflow returning {len(all_reports)} reports: {list(all_reports.keys())}"
    )
    return all_reports


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Register example experiments
    register_example_experiments()

    # Initialize Flyte connection
    flyte.init_from_config("config.yaml")

    # Test with a simpler workflow first
    print("Testing simple workflow execution...")

    # Run the backtest workflow
    run = flyte.run(
        backtest,
        # experiment_ids="baseline_cc baseline_fb ensemble_experiment dnn_experiment",
        experiment_ids="baseline_cc",
        snapshot_date="2025-01-15",
    )

    # Print results
    print(f"Run name: {run.name}")
    print(f"Run URL: {run.url}")

    # Wait for completion and get results
    try:
        print("Waiting for workflow completion...")
        results = run.wait(run)
        print(f"Raw results type: {type(results)}")
        print(f"Raw results: {results}")

        if results is not None:
            print("Backtest results:")
            if isinstance(results, dict):
                for report_name, report_url in results.items():
                    print(f"  {report_name}: {report_url}")
            else:
                print(f"  Unexpected result format: {results}")
        else:
            print("Backtest completed but returned no results")
            print("This might be due to:")
            print("1. Task environment configuration issue")
            print("2. Workflow not properly returning results")
            print("3. Flyte framework issue with async task execution")
    except Exception as e:
        print(f"Backtest failed with error: {e}")
        print(f"Check the run URL for more details: {run.url}")

