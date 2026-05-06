import importlib
import os

import pandas as pd

import grade


def evaluate(sample_df):
    records = sample_df[grade.REQUEST_FIELDS].to_dict("records")
    flat = [grade._flatten(grade.predict(r), r["ped_id"]) for r in records]
    preds_df = pd.DataFrame(flat, columns=grade.OUT_COLS)
    return grade.score(preds_df, sample_df)


def main():
    df = pd.read_parquet(grade.DATA / "dev.parquet")
    df = df.sample(n=1500, random_state=42).reset_index(drop=True)

    temps = [0.93, 0.95]
    residual_patterns = [
        [0.60, 0.65, 0.70, 0.75],
        [0.60, 0.70, 0.75, 0.80],
        [0.65, 0.70, 0.75, 0.80],
        [0.70, 0.70, 0.70, 0.70],
        [0.68, 0.68, 0.72, 0.76],
    ]
    seq_patterns = [
        [0.10, 0.12, 0.15, 0.20],
        [0.10, 0.15, 0.18, 0.22],
        [0.12, 0.15, 0.18, 0.20],
        [0.15, 0.15, 0.15, 0.15],
        [0.12, 0.14, 0.16, 0.18],
    ]

    best = None
    for t in temps:
        for rb in residual_patterns:
            for sb in seq_patterns:
                                        os.environ["INTENT_TEMPERATURE"] = str(t)
                                        os.environ["TRAJ_RESIDUAL_BLEND_H"] = ",".join(f"{v:.2f}" for v in rb)
                                        os.environ["TRAJ_SEQ_BLEND_H"] = ",".join(f"{v:.2f}" for v in sb)
                                        os.environ["TRAJ_RESIDUAL_BLEND"] = str(sum(rb) / 4.0)
                                        os.environ["TRAJ_SEQ_BLEND"] = str(sum(sb) / 4.0)

                                        import predict

                                        importlib.reload(predict)
                                        importlib.reload(grade)

                                        s = evaluate(df)
                                        row = {
                                            "intent_temp": t,
                                            "residual_blend_h": rb,
                                            "seq_blend_h": sb,
                                            **s,
                                        }
                                        if best is None or row["score"] < best["score"]:
                                            best = row
                                            print(
                                                f"best score={row['score']:.4f} "
                                                f"temp={t:.2f} rb={rb} sb={sb} "
                                                f"(intent_term={row['intent_term']:.3f}, traj_term={row['traj_term']:.3f})"
                                            )

    print("\nBest config:")
    print(best)
    if best is not None:
        rb_txt = ",".join(f"{v:.2f}" for v in best["residual_blend_h"])
        sb_txt = ",".join(f"{v:.2f}" for v in best["seq_blend_h"])
        print("\nExport settings:")
        print(f"INTENT_TEMPERATURE={best['intent_temp']:.2f}")
        print(f"TRAJ_RESIDUAL_BLEND_H={rb_txt}")
        print(f"TRAJ_SEQ_BLEND_H={sb_txt}")


if __name__ == "__main__":
    main()
