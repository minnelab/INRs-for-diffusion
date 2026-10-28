#!/usr/bin/env python
"""Train an INR on one subject and score what was held out.

    python train.py --dataset hcp --task shell-holdout --subject 107422

Every reconstruction experiment in the paper is a (dataset, task) pair; run
`--list` to see them. Results land in OUTPUTS/<dataset>_<task>_<subject>_<stamp>/
as metrics_summary.json plus per-shell CSVs.
"""
from __future__ import annotations

import argparse
import sys

from inrdiff.benchmark import DATASETS, TASKS, configure, run


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="hcp")
    ap.add_argument("--task", choices=sorted(TASKS), default="shell-holdout",
                    help="which part of the acquisition is withheld from training")
    ap.add_argument("--subject", default=None,
                    help="subject id (default: the dataset's reference subject)")
    ap.add_argument("--directions", type=int, default=None,
                    help="directions retained per training shell, for --task direction-budget "
                         "(the paper sweeps 10, 20, 45, 81 of 90)")
    ap.add_argument("--factor", type=int, default=2,
                    help="voxel decimation factor for --task superres")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the task's epoch budget")
    ap.add_argument("--config", default=None,
                    help="base config (default: configs/ball_fodf_hcp.yml)")
    ap.add_argument("--list", action="store_true", help="list datasets and tasks, then exit")
    args = ap.parse_args()

    if args.list:
        print("datasets:")
        for k, d in sorted(DATASETS.items()):
            print(f"  {k:8s} shells {d.shells}, reference subject {d.default_subject}, "
                  f"fODF ground truth: {'yes' if d.has_fodf_gt else 'no'}")
        print("tasks:")
        for k, t in sorted(TASKS.items()):
            print(f"  {k:17s} train {t.train_shells or '(all)'}, "
                  f"held out {t.held_out_shells or '(directions/voxels only)'}")
        return 0

    dataset, task = DATASETS[args.dataset], TASKS[args.task]
    if task.name == "direction-budget" and args.directions is None:
        ap.error("--task direction-budget needs --directions (e.g. --directions 10)")
    if dataset.name == "parkmr" and task.name.startswith(("shell", "single")):
        ap.error("ParkMR uses --task parkmr-holdout; its shells are 100/700/1400/2000")

    configure(dataset, task, directions=args.directions, epochs=args.epochs,
              superres_factor=args.factor if task.name == "superres" else None)
    run(subject=args.subject, config=args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
