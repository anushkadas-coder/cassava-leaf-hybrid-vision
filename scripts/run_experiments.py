#!/usr/bin/env python
"""Run every (variant x seed) in sweep configs, skipping runs already in the log (resumable), then summarise.

Examples
  python scripts/run_experiments.py --config configs/color_space.yaml
  python scripts/run_experiments.py --config configs/frequency.yaml --set train.epochs=10
  python scripts/run_experiments.py --summarize --baseline cnn_rgb --output-dir outputs
"""
import argparse
from pathlib import Path

from cassava.config import expand_sweep, load_config
from cassava.evaluation.report import read_records, summarize_records, to_markdown
from cassava.experiments import run_experiment
from cassava.utils import get_logger

log = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--only", nargs="*", help="run only these variant names")
    ap.add_argument("--summarize", action="store_true", help="only print/save the summary table from the log")
    ap.add_argument("--baseline", help="variant name used for delta_vs_baseline")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--log-file", default="experiment_log.jsonl")
    a = ap.parse_args()

    for cfg_path in a.config:
        cfg = load_config(cfg_path, a.set)
        log_path = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["log_file"]
        done = {(r["name"], r["seed"]) for r in read_records(log_path)}
        for run in expand_sweep(cfg):
            key = (run["experiment"]["name"], run["experiment"]["seed"])
            if a.only and key[0] not in a.only:
                continue
            if key in done:
                log.info("skip %s seed %s (already in log)", *key)
                continue
            run_experiment(run)

    recs = read_records(Path(a.output_dir) / a.log_file)
    df = summarize_records(recs, a.baseline)
    md = to_markdown(df)
    print(md)
    if not df.empty:
        df.to_csv(Path(a.output_dir) / "summary.csv", index=False)
        (Path(a.output_dir) / "summary.md").write_text(md + "\n")


if __name__ == "__main__":
    main()
