import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

TRAIN_PATH = "data/train.parquet"
DEV_PATH = "data/dev.parquet"
OUT_PATH = "traj_residual_models.pkl"
TIME_OF_DAY_CATEGORIES = ["", "daytime", "nighttime"]
WEATHER_CATEGORIES = ["", "clear", "cloudy", "rain", "snow"]
LOCATION_CATEGORIES = ["", "indoor", "plaza", "street"]


def _safe_bbox_array(bbox_values, expected_len=16):
    try:
        arr = np.asarray(bbox_values)
    except (TypeError, ValueError):
        return np.zeros((expected_len, 4), dtype=np.float32)
    if arr.ndim == 1:
        if arr.size == 4:
            arr = arr.reshape(1, 4).astype(np.float32, copy=False)
        else:
            try:
                arr = np.stack([np.asarray(v, dtype=np.float32).reshape(4,) for v in arr.tolist()], axis=0)
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
        arr = np.vstack([arr, np.zeros((expected_len - arr.shape[0], 4), dtype=np.float32)])
    elif arr.shape[0] > expected_len:
        arr = arr[:expected_len]
    return arr


def _normalize_category(value, feature_name):
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


def _context_features(row):
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


def _cv_future_norm_from_history(row):
    fw = float(row["frame_w"]) if row["frame_w"] else 1.0
    fh = float(row["frame_h"]) if row["frame_h"] else 1.0
    b = _safe_bbox_array(row["bbox_history"], 16)
    cx = (b[:, 0] + b[:, 2]) * 0.5 / fw
    cy = (b[:, 1] + b[:, 3]) * 0.5 / fh
    centers = np.stack([cx, cy], axis=1)
    tail = centers[-4:]
    if len(tail) >= 2:
        vel = np.mean(np.diff(tail, axis=0), axis=0)
    else:
        vel = np.zeros(2, dtype=np.float32)
    steps = [8, 15, 23, 30]
    out = []
    for s in steps:
        c = centers[-1] + vel * s
        out.extend([float(c[0]), float(c[1])])
    return np.asarray(out, dtype=np.float32)


def _history_features(row):
    fw = float(row["frame_w"]) if row["frame_w"] else 1.0
    fh = float(row["frame_h"]) if row["frame_h"] else 1.0
    b = _safe_bbox_array(row["bbox_history"], 16)
    cx = (b[:, 0] + b[:, 2]) * 0.5 / fw
    cy = (b[:, 1] + b[:, 3]) * 0.5 / fh
    w = (b[:, 2] - b[:, 0]) / fw
    h = (b[:, 3] - b[:, 1]) / fh
    vx = np.diff(cx, prepend=cx[0]) * 15.0
    vy = np.diff(cy, prepend=cy[0]) * 15.0
    ax = np.diff(vx, prepend=vx[0]) * 15.0
    ay = np.diff(vy, prepend=vy[0]) * 15.0
    ego_speed = np.asarray(row.get("ego_speed_history", np.zeros(16)), dtype=np.float32)
    ego_yaw = np.asarray(row.get("ego_yaw_history", np.zeros(16)), dtype=np.float32)
    if ego_speed.shape[0] < 16:
        ego_speed = np.pad(ego_speed, (0, 16 - ego_speed.shape[0]))
    if ego_yaw.shape[0] < 16:
        ego_yaw = np.pad(ego_yaw, (0, 16 - ego_yaw.shape[0]))
    ego_speed = ego_speed[:16]
    ego_yaw = ego_yaw[:16]

    feats = []
    for arr in [cx, cy, w, h, vx, vy, ax, ay, ego_speed, ego_yaw]:
        feats.extend([arr[-1], arr[-2], np.mean(arr[-4:]), np.mean(arr), np.std(arr), np.min(arr), np.max(arr)])
    feats.append(float(bool(row.get("ego_available", False))))
    feats.extend(_context_features(row).tolist())
    cv = _cv_future_norm_from_history(row)
    feats.extend(cv.tolist())
    return np.asarray(feats, dtype=np.float32)


def _traj_targets_norm(row):
    fw = float(row["frame_w"]) if row["frame_w"] else 1.0
    fh = float(row["frame_h"]) if row["frame_h"] else 1.0
    vals = []
    for col in ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]:
        bb = np.asarray(row[col], dtype=np.float32).reshape(4,)
        vals.extend([float((bb[0] + bb[2]) * 0.5 / fw), float((bb[1] + bb[3]) * 0.5 / fh)])
    return np.asarray(vals, dtype=np.float32)


def build(df):
    X = np.stack([_history_features(r) for _, r in df.iterrows()], axis=0)
    y = np.stack([_traj_targets_norm(r) for _, r in df.iterrows()], axis=0)
    cv = np.stack([_cv_future_norm_from_history(r) for _, r in df.iterrows()], axis=0)
    resid = y - cv
    return X, resid


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    dev_df = pd.read_parquet(DEV_PATH)
    X_train, y_res_train = build(train_df)
    X_dev, y_res_dev = build(dev_df)

    models = []
    for i in range(8):
        reg = XGBRegressor(
            n_estimators=900,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=123 + i,
            n_jobs=4,
        )
        reg.fit(X_train, y_res_train[:, i], eval_set=[(X_dev, y_res_dev[:, i])], verbose=False)
        models.append(reg)

    with open(OUT_PATH, "wb") as f:
        pickle.dump({"residual_models": models}, f)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
