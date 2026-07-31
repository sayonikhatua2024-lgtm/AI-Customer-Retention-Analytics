import json

import pytest

from src import config
from src.data_pipeline import clean_data, load_dashboard_data, load_raw_data, prepare_model_data
from src.model import load_model, predict_proba, validate_model_metadata


def test_model_loads_with_valid_metadata():
    model = load_model()
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_one_row_prediction_probability_bounds():
    model = load_model()
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    X, _ = prepare_model_data(clean_df)

    probabilities = predict_proba(model, X.head(1))

    assert len(probabilities) == 1
    assert 0 <= probabilities[0] <= 1


def test_model_feature_schema_matches_config():
    model = load_model()
    preprocessor = model.named_steps["preprocessor"]
    transformer_features = []
    for _, _, columns in preprocessor.transformers_:
        transformer_features.extend(columns)

    assert set(transformer_features) == set(config.MODEL_FEATURES)
    assert len(preprocessor.get_feature_names_out()) == 46


def test_model_metadata_validation_rejects_schema_mismatch(tmp_path):
    model = load_model()
    metadata = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    metadata["feature_schema"]["numeric_features"] = ["Unexpected Feature"]
    invalid_metadata_path = tmp_path / "model_metadata.json"
    invalid_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="feature schema"):
        validate_model_metadata(model, metadata_path=invalid_metadata_path)


def test_existing_dashboard_data_restores_verified_customer_ids():
    dashboard_df = load_dashboard_data()
    raw_df = load_raw_data()

    assert "CustomerID" in dashboard_df.columns
    assert dashboard_df["CustomerID"].is_unique
    assert dashboard_df["CustomerID"].tolist() == raw_df["CustomerID"].tolist()
