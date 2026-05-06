import pandas as pd
import numpy as np
import os

# Configuration
SEQUENCE_LENGTH = 16
DT = 1.0 / 15.0  # Time step between frames (data is at 15 Hz)
FUTURE_BBOX_COLUMNS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]
TIME_OF_DAY_CATEGORIES = ["", "daytime", "nighttime"]
WEATHER_CATEGORIES = ["", "clear", "cloudy", "rain", "snow"]
LOCATION_CATEGORIES = ["", "indoor", "plaza", "street"]


def _safe_bbox_array(bbox_values, expected_len):
    """
    Converts bbox values to shape (expected_len, 4), falling back to zeros if malformed.
    """
    try:
        arr = np.asarray(bbox_values)
    except (TypeError, ValueError):
        return np.zeros((expected_len, 4), dtype=np.float32)

    if arr.ndim == 1:
        if arr.size == 4 and expected_len == 1:
            return arr.reshape(1, 4).astype(np.float32, copy=False)
        try:
            stacked = [np.asarray(v, dtype=np.float32).reshape(4,) for v in arr.tolist()]
            arr = np.stack(stacked, axis=0)
        except (TypeError, ValueError):
            return np.zeros((expected_len, 4), dtype=np.float32)
    else:
        try:
            arr = arr.astype(np.float32, copy=False)
        except (TypeError, ValueError):
            return np.zeros((expected_len, 4), dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != 4:
        return np.zeros((expected_len, 4), dtype=np.float32)

    if arr.shape[0] < expected_len:
        pad = np.zeros((expected_len - arr.shape[0], 4), dtype=np.float32)
        arr = np.vstack([arr, pad])
    elif arr.shape[0] > expected_len:
        arr = arr[:expected_len]

    return arr.astype(np.float32, copy=False)


def _normalize_category(value, feature_name):
    """Normalize noisy category values to a stable canonical set."""
    if value is None:
        text = ""
    else:
        text = str(value).strip().lower()

    if text in {"n/a", "na", "none", "null", "nan"}:
        text = ""

    if feature_name == "weather" and text == "cloud":
        text = "cloudy"

    return text


def _one_hot(value, categories):
    vec = np.zeros(len(categories), dtype=np.float32)
    idx = categories.index(value) if value in categories else 0
    vec[idx] = 1.0
    return vec


def extract_context_features(row):
    """Extract one-hot encoded static context features for a row."""
    tod = _normalize_category(row.get("time_of_day", ""), "time_of_day")
    weather = _normalize_category(row.get("weather", ""), "weather")
    location = _normalize_category(row.get("location", ""), "location")
    return np.concatenate(
        [
            _one_hot(tod, TIME_OF_DAY_CATEGORIES),
            _one_hot(weather, WEATHER_CATEGORIES),
            _one_hot(location, LOCATION_CATEGORIES),
        ],
        axis=0,
    )

def extract_and_normalize_bboxes(bbox_history, frame_w, frame_h):
    """
    Extracts bbox centers/dimensions, calculates velocities and accelerations,
    and normalizes them by frame size.
    """
    if frame_w == 0 or frame_h == 0:
        return np.zeros((SEQUENCE_LENGTH, 8))

    bboxes = _safe_bbox_array(bbox_history, SEQUENCE_LENGTH)
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    # Normalize by frame dimensions
    norm_cx = cx / frame_w
    norm_cy = cy / frame_h
    norm_w = w / frame_w
    norm_h = h / frame_h

    norm_bbox_features = np.stack([norm_cx, norm_cy, norm_w, norm_h], axis=1)

    # Calculate velocities
    delta_pos = np.diff(norm_bbox_features[:, :2], axis=0, prepend=norm_bbox_features[0:1, :2])
    velocities = delta_pos / DT

    # Calculate accelerations
    delta_vel = np.diff(velocities, axis=0, prepend=velocities[0:1, :])
    accelerations = delta_vel / DT
    
    # Combine all features: [cx, cy, w, h, vx, vy, ax, ay]
    combined_features = np.hstack([norm_bbox_features, velocities, accelerations])

    return combined_features

def process_ego_motion(ego_speed_history, ego_yaw_history, ego_available):
    """
    Processes ego motion data, replacing it with zeros if not available.
    """
    if not (ego_available and ego_speed_history is not None and ego_yaw_history is not None):
        return np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)

    speed = np.asarray(ego_speed_history, dtype=np.float32).reshape(-1)
    yaw = np.asarray(ego_yaw_history, dtype=np.float32).reshape(-1)

    if speed.shape[0] < SEQUENCE_LENGTH:
        speed = np.pad(speed, (0, SEQUENCE_LENGTH - speed.shape[0]))
    if yaw.shape[0] < SEQUENCE_LENGTH:
        yaw = np.pad(yaw, (0, SEQUENCE_LENGTH - yaw.shape[0]))

    speed = speed[:SEQUENCE_LENGTH]
    yaw = yaw[:SEQUENCE_LENGTH]
    ego_motion = np.stack([speed, yaw], axis=1)
    ego_motion = np.nan_to_num(ego_motion, nan=0.0, posinf=0.0, neginf=0.0)
    return ego_motion.astype(np.float32, copy=False)

def create_sequences(df):
    """
    Processes an entire dataframe to create structured sequences for model input and output.
    """
    n = len(df)
    context_dim = len(TIME_OF_DAY_CATEGORIES) + len(WEATHER_CATEGORIES) + len(LOCATION_CATEGORIES)
    feature_dim = 8 + 2 + context_dim
    X = np.zeros((n, SEQUENCE_LENGTH, feature_dim), dtype=np.float32)
    y_traj = np.zeros((n, 8), dtype=np.float32)
    
    for i, row in enumerate(df.itertuples(index=False)):
        # Process pedestrian motion
        ped_motion_features = extract_and_normalize_bboxes(
            row.bbox_history, row.frame_w, row.frame_h
        )
        
        # Process ego vehicle motion
        ego_motion_features = process_ego_motion(
            row.ego_speed_history, row.ego_yaw_history, row.ego_available
        )

        # Static scene context encoded once and repeated across timesteps.
        context = extract_context_features(
            {
                "time_of_day": row.time_of_day,
                "weather": row.weather,
                "location": row.location,
            }
        )
        context_features = np.tile(context, (SEQUENCE_LENGTH, 1))
        
        # Combine features: dynamic sequence + repeated global context.
        full_sequence = np.hstack([ped_motion_features, ego_motion_features, context_features])
        X[i] = np.nan_to_num(full_sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # Trajectory Target: normalized center coordinates of 4 future bboxes
        future_bboxes = _safe_bbox_array(
            [getattr(row, col) for col in FUTURE_BBOX_COLUMNS],
            len(FUTURE_BBOX_COLUMNS),
        )
        frame_w, frame_h = float(row.frame_w), float(row.frame_h)
        if frame_w == 0 or frame_h == 0:
            continue
        x1, y1, x2, y2 = future_bboxes[:, 0], future_bboxes[:, 1], future_bboxes[:, 2], future_bboxes[:, 3]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        centers = np.stack([cx / frame_w, cy / frame_h], axis=1).reshape(-1)
        y_traj[i] = np.nan_to_num(centers, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # Intent Target
    y_intent = df["will_cross_2s"].to_numpy(dtype=np.float32)
    y_intent = np.nan_to_num(y_intent, nan=0.0, posinf=1.0, neginf=0.0)

    return X, y_intent, y_traj

def main(train_path='data/train.parquet', dev_path='data/dev.parquet', output_path='processed_data.npz'):
    """
    Main function to load data, process it, and save the output.
    """
    print("Starting data preparation pipeline...")
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from '{train_path}' and '{dev_path}'...")
    try:
        train_df = pd.read_parquet(train_path)
        dev_df = pd.read_parquet(dev_path)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure your parquet files are in the 'data/' directory.")
        return

    print("\nProcessing training data...")
    X_train, y_intent_train, y_traj_train = create_sequences(train_df)
    print("Training data shapes:")
    print(f"  Input features (X_train):         {X_train.shape}")
    print(f"  Intent labels (y_intent_train):   {y_intent_train.shape}")
    print(f"  Trajectory labels (y_traj_train): {y_traj_train.shape}")
    
    print("\nProcessing development data...")
    X_dev, y_intent_dev, y_traj_dev = create_sequences(dev_df)
    print("Development data shapes:")
    print(f"  Input features (X_dev):           {X_dev.shape}")
    print(f"  Intent labels (y_intent_dev):     {y_intent_dev.shape}")
    print(f"  Trajectory labels (y_traj_dev):   {y_traj_dev.shape}")

    print(f"\nSaving processed data to '{output_path}'...")
    np.savez_compressed(output_path, 
                        X_train=X_train, y_intent_train=y_intent_train, y_traj_train=y_traj_train,
                        X_dev=X_dev, y_intent_dev=y_intent_dev, y_traj_dev=y_traj_dev)
    print("Data preparation complete.")

if __name__ == '__main__':
    main()
