import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from tsfresh import extract_features
from datetime import datetime, timedelta
import sys
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from utils import load_config

models_dir = project_root / "data" / "models"
bst = xgb.Booster()
bst.load_model(str(models_dir / "xgb_model.json"))
# bst.set_param({'predictor': 'cpu_predictor'})


le = joblib.load(str(models_dir / "label_encoder.pkl"))

config = load_config("feature_config.yaml")
tsfresh_features = config["tsfresh_har_features"]
extraction_params = config["extraction_params"]

# --- Feature extraction for a single channel ---
def extract_features_for_channel(channel_data, sensor_name, modality, channel):
    """
    channel_data: array-like or pd.DataFrame with one column (the channel)
    Returns: dict of tsfresh features for this channel, with keys as in batch pipeline
    """
    if isinstance(channel_data, pd.DataFrame):
        values = channel_data.iloc[:, 0].values
    elif isinstance(channel_data, (np.ndarray, list, pd.Series)):
        values = np.asarray(channel_data)
    else:
        raise ValueError("Channel data must be array-like or DataFrame with one column.")
    dt_start = datetime.now()
    datetimes = [dt_start + timedelta(milliseconds=100*i) for i in range(len(values))]
    df = pd.DataFrame({
        "datetime": datetimes,
        "value": values
    })
    df["sensor"] = sensor_name
    df["modality"] = modality
    df["channel"] = channel
    df["window_id"] = 0
    df["timestamp"] = (df["datetime"] - df["datetime"].min()).dt.total_seconds()
    df["id"] = (
        df["window_id"].astype(str) + "@" +
        df["sensor"] + "@" +
        df["modality"] + "@" +
        df["channel"]
    )
    features = extract_features(
        df,
        column_id="id",
        column_sort="timestamp",
        column_value="value",
        default_fc_parameters=tsfresh_features,
        n_jobs=1,
        disable_progressbar=True
    )
    # Flatten to dict with correct column names
    row = {}
    for idx in features.index:
        parts = idx.split("@")
        sensor, modality, channel = parts[1], parts[2], parts[3]
        for feat in features.columns:
            colname = f"{sensor}@{modality}@{channel}@{feat}"
            row[colname] = features.loc[idx, feat]
    return row

# --- Main prediction function ---
def predict_realtime(
    m5_wrist_l_df,
    m5_wrist_r_df,
    m5_wheel_df,
    polar_chest_df,
    sensomative_bottom_df
):
    """
    Takes 5 DataFrames as input, one for each sensor modality, matching the structure in run_feature_extraction.py.
    Each DataFrame should have the same columns as in the batch pipeline (e.g., linear_acceleration_x, ...).
    Returns: predicted class label
    """
    # Build sensor_config as in run_feature_extraction.py
    sensor_config = [
        ("M5 Wrist L", m5_wrist_l_df, "linear_acceleration", ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]),
        ("M5 Wrist L", m5_wrist_l_df, "angular_velocity", ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]),
        ("M5 Wrist R", m5_wrist_r_df, "linear_acceleration", ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]),
        ("M5 Wrist R", m5_wrist_r_df, "angular_velocity", ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]),
        ("M5 Wheel", m5_wheel_df, "linear_acceleration", ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]),
        ("M5 Wheel", m5_wheel_df, "angular_velocity", ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]),
        ("Polar Chest", polar_chest_df, "linear_acceleration", ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]),
        ("Sensomative Bottom", sensomative_bottom_df, "pressure", [f"pressure_{i}" for i in range(12)]),
    ]
    feature_row = {}
    for sensor_name, df, modality, channels in sensor_config:
        for channel in channels:
            if channel not in df.columns:
                continue
            channel_data = df[channel]
            feats = extract_features_for_channel(channel_data, sensor_name, modality, channel)
            feature_row.update(feats)
    features_df = pd.DataFrame([feature_row])

    # Handle NaN/Inf values as in run_feature_extraction.py
    features_df = features_df.replace({
        np.nan: 11000,
        np.inf: 9000,
        -np.inf: -9000
    })

    # Align columns to the order expected by the model
    feature_names = bst.feature_names
    if feature_names is not None:
        features_df = features_df.reindex(columns=feature_names, fill_value=0)

    dtest = xgb.DMatrix(features_df)
    y_pred = bst.predict(dtest)
    label = le.inverse_transform(y_pred.astype(int))[0]
    return label

# Example usage (to be replaced with real sensor input in deployment):
if __name__ == "__main__":
    # Simulate 5 DataFrames with the correct columns and 50 rows each
    n = 50
    m5_wrist_l_df = pd.DataFrame({
        "linear_acceleration_x": 1*np.random.randn(n),
        "linear_acceleration_y": 1*np.random.randn(n),
        "linear_acceleration_z": np.random.randn(n),
        "angular_velocity_x": 20*np.random.randn(n),
        "angular_velocity_y": 20*np.random.randn(n),
        "angular_velocity_z": 20*np.random.randn(n),
    })
    m5_wrist_r_df = m5_wrist_l_df.copy()
    m5_wheel_df = pd.DataFrame({
        "linear_acceleration_x": 2*np.random.randn(n),
        "linear_acceleration_y": 2*np.random.randn(n),
        "linear_acceleration_z": 2*np.random.randn(n),
        "angular_velocity_x": 100*np.random.randn(n),
        "angular_velocity_y": 100*np.random.randn(n),
        "angular_velocity_z": 100*np.random.randn(n),
    })
    polar_chest_df = pd.DataFrame({
        "linear_acceleration_x": 1.1*np.random.randn(n),
        "linear_acceleration_y": 1.2*np.random.randn(n),
        "linear_acceleration_z": 1.3*np.random.randn(n),
    })
    sensomative_bottom_df = pd.DataFrame({
        f"pressure_{i}": 100*np.ones(n) for i in range(12)
    })
    pred = predict_realtime(
        m5_wrist_l_df,
        m5_wrist_r_df,
        m5_wheel_df,
        polar_chest_df,
        sensomative_bottom_df
    )
    print("Predicted ADL class:", pred)
