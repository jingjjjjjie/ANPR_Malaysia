# log_train_meta.py
from ultralytics import YOLO
from time import perf_counter
from datetime import datetime, timedelta
from pathlib import Path
import json
import os

def _summarize_best_metrics(run_dir: str):
    """Return best-epoch metrics from results.csv (tolerant to column name variants)."""
    import pandas as pd
    csv_path = Path(run_dir) / "results.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    def pick(substr_list):
        for s in substr_list:
            cands = [c for c in df.columns if s in c]
            if cands:
                return cands[0]
        return None

    p_col   = pick(["metrics/precision", "precision"])
    r_col   = pick(["metrics/recall", "recall"])
    m50_col = pick(["metrics/mAP50", "mAP_0.5", "map50"])
    m95_col = pick(["metrics/mAP50-95", "mAP_0.5:0.95", "map"])
    ep_col  = "epoch" if "epoch" in df.columns else None

    if m95_col and df[m95_col].notna().any():
        i_best = df[m95_col].idxmax()
    elif m50_col and df[m50_col].notna().any():
        i_best = df[m50_col].idxmax()
    elif r_col and df[r_col].notna().any():
        i_best = df[r_col].idxmax()
    else:
        return None

    row = df.loc[i_best]
    return {
        "best_epoch": int(row[ep_col]) if ep_col else int(i_best),
        "precision": float(row[p_col]) if p_col else None,
        "recall": float(row[r_col]) if r_col else None,
        "mAP50": float(row[m50_col]) if m50_col else None,
        "mAP50-95": float(row[m95_col]) if m95_col else None,
        "results_csv": str(csv_path)
    }

def train_and_log(
    model_ckpt="",
    data_yaml="data/plates.yaml",
    description="",
    out_project="runs/plates",
    out_name="",
    **train_kwargs
):
    model = YOLO(model_ckpt)

    # ---- timing ----
    t0 = perf_counter()
    model.train(
        data=data_yaml,
        project=out_project,
        name=out_name,
        **train_kwargs
    )
    dt = perf_counter() - t0

    # ---- where to save ----
    run_dir = str(model.trainer.save_dir)  # e.g., runs/plates/<name>
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_meta.txt")

    # ---- metadata ----
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir,
        "model_ckpt": model_ckpt,
        "data_yaml": data_yaml,
        "description": description.strip(),
        "train_args": train_kwargs,
        "elapsed_seconds": round(dt, 3),
        "elapsed_hms": str(timedelta(seconds=int(dt))),
    }

    # ---- collect best metrics from results.csv ----
    best = _summarize_best_metrics(run_dir)

    # ---- write single text file ----
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# YOLO Training Metadata\n")
        f.write(f"Timestamp:     {meta['timestamp']}\n")
        f.write(f"Run dir:       {meta['run_dir']}\n")
        f.write(f"Model ckpt:    {meta['model_ckpt']}\n")
        f.write(f"Data yaml:     {meta['data_yaml']}\n")
        f.write(f"Elapsed:       {meta['elapsed_hms']} ({meta['elapsed_seconds']} s)\n")
        f.write(f"Description:   {meta['description']}\n")
        f.write("\n# Train arguments (JSON)\n")
        f.write(json.dumps(meta["train_args"], indent=2))
        f.write("\n")

        f.write("\n# Best validation metrics\n")
        if best:
            f.write(
                f"Best (epoch {best['best_epoch']}): "
                f"P={best['precision']:.3f}  R={best['recall']:.3f}  "
                f"mAP50={best['mAP50']:.3f}  mAP50-95={best['mAP50-95']:.3f}\n"
            )
            f.write(f"results.csv:   {best['results_csv']}\n")
        else:
            f.write("results.csv not found or columns missing; no metrics summarized.\n")

    print(f"Wrote: {log_path}")
    return log_path

if __name__ == "__main__":
    train_and_log(
        model_ckpt="v2_best.pt",
        data_yaml="../dataset/v3/data.yaml",
        # write your description of the training session here
        description= "",
        out_project="../runs/plates", # your project output location
        out_name="v3", # your project output name
        epochs=100,
        imgsz=1280,
        batch=2,
        device=0  # using cuda device set 'cpu' for CPU
    )
