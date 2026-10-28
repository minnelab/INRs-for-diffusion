#!/usr/bin/env python
"""Scorers and non-learned baselines.

    python evaluate.py upsample   --subject 107422 --factor 2
    python evaluate.py dti-extrap --subject 107422
    python evaluate.py frame      --pred <fodf.nii.gz> --subject 107422
    python evaluate.py fodf       --pred <fodf.nii.gz> --gt <wm_fod.nii.gz> \
                                  --label ours --out acc.json --signs=-1,1,1

`fodf` needs scilpy, which conflicts with this environment's VTK/numpy stack —
install it separately and run that subcommand there. The others run on CPU here.

Reconstruction metrics for a trained INR are written by train.py; this script is
for the comparisons that do not involve the network.
"""
from __future__ import annotations

import sys

SUBCOMMANDS = {
    "upsample":   ("classical tri-cubic super-resolution baseline", "main_upsample"),
    "dti-extrap": ("mono-exponential DTI shell extrapolation baseline", "main_dti_extrap"),
    "frame":      ("signed-axis-permutation search for an fODF export", "main_frame"),
    "fodf":       ("ACC / AFD against an MSMT-CSD reference (scilpy env)", "main_fodf"),
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("subcommands:")
        for name, (desc, _) in SUBCOMMANDS.items():
            print(f"  {name:11s} {desc}")
        print("\nPass --help after a subcommand for its own options.")
        return 0
    name = sys.argv[1]
    if name not in SUBCOMMANDS:
        print(f"unknown subcommand {name!r}; expected one of {', '.join(SUBCOMMANDS)}",
              file=sys.stderr)
        return 2
    import inrdiff.scoring as scoring
    sys.argv = [f"evaluate.py {name}"] + sys.argv[2:]   # the scorer parses the rest
    return getattr(scoring, SUBCOMMANDS[name][1])() or 0


if __name__ == "__main__":
    sys.exit(main())
