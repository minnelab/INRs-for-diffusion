#!/usr/bin/env python
"""Classify subjects from their trained INRs.

One script, two cohorts. The network is never retrained here: each subject's INR
is read from disk and a feature vector taken from it, either as the raw
per-voxel network output or as the decoded biophysical parameters, over the
whole brain or an atlas ROI, densely or at a fixed random voxel subset.

    python classify.py --dataset ppmi
    python classify.py --dataset hcp --target age --readout decoded

Scoring is balanced accuracy under two cross-validation schemes: repeated
stratified 5-fold, and leave-one-group-out over scanner and institution so no
site is shared between train and test. The reported p is a label-permutation
null on the leave-one-scanner-out statistic.

This is the code that produced the paper's classification numbers, flattened
into one file; names that collided between the per-cohort modules carry a
ppmi_ / hcp_ / hcpdec_ / feat_ / cv_ / drv_ prefix.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Callable
import argparse
import glob
import json
import numpy as np
import os
import re
import sys
import torch


import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# ==========================================================================
# ppmi: ppmi.py
# ==========================================================================

"""PPMI cohort + data layer: discover subjects/sessions, select the raw DWI series,
parse acquisition type & session date, derive masks, and load diagnosis labels.

This mirrors ``labels.py`` (the ParkMR cohort) but adapts to PPMI's layout, which
differs in three ways that the piping has to handle explicitly:

  1. **Nested, longitudinal.** ``~/PPMI_nifti/<PATNO>/`` holds *several visits*
     (sessions, one per scan date) and usually *two raw DWI series per visit*
     (e.g. series 2 and series 11 — both full 65-volume acquisitions). We define
     one INR per (subject, session) and pick a single canonical series per session.
  2. **Scanner-derived maps masquerade as DWI.** Alongside the raw 65-volume DWI,
     some visits ship ``*_FA`` / ``*_ADC`` / ``*_TRACEW`` series — these are
     reconstructed scalar maps, NOT the diffusion signal, and must be excluded
     from INR training. We keep only series whose gradient table has a b0 + a
     populated shell (>= MIN_DIRS non-b0 directions).
  3. **Single shell.** PPMI DTI is single-shell b=1000 (1 b0 + 64 directions),
     unlike MUDI's [100,700,1400,2000] or HCP's [1000,2000,3000]. ``SHELLS``
     reflects that; see ``subject_config`` for the training-config implications.

Acquisition is mostly **cardiac-gated** (see ``acq_type``): each volume's readout
is triggered to the cardiac cycle so brainstem/CSF pulsation does not corrupt the
diffusion signal in the deep structures relevant to PD. A minority are non-gated.
Gated/non-gated use the *same* b=1000 64-direction scheme — only the timing
differs — but acquisition type is recorded as a nuisance covariate to control for
(it tends to track site/scanner), the same discipline we applied to the ParkMR
session confound.

Diagnosis labels are NOT in the imaging tree or the IDA metadata XML; they come
from the IDA collection manifest ``Rodrigo_*.csv`` (Subject/Group/Sex/Age/Visit,
Group in {PD, Control}) shipped in the repo root, or a LONI
``Participant_Status.csv`` (PATNO/COHORT). ``ppmi_get_cohort`` maps either to PD(1) vs
Control/HC(0); ``covariates`` exposes Age/Sex for confound control.
"""

PPMI_ROOT = Path(os.environ.get("BINR_PPMI_ROOT", str(Path.home() / "PPMI_nifti")))

SHELLS = [1000]

B0_THRESH = 50.0

MIN_DIRS = 30

_NONGATED = re.compile(r"(non|un)[\W_]*gated", re.I)

_GATED = re.compile(r"gat|trigger|pulse", re.I)

_DERIVED = re.compile(r"(_|^)(FA|ADC|TRACEW|EXP|COLFA)\b", re.I)

_TS = re.compile(r"_(\d{14})_(\d+)\.nii\.gz$")

def acq_type(desc: str) -> str:
    """'nongated' | 'gated' | 'unknown' from the series description."""
    if _NONGATED.search(desc):
        return "nongated"
    if _GATED.search(desc):
        return "gated"
    return "unknown"

class Series:
    """One DWI acquisition: its nii/bval/bvec stem, date, series no., acq type."""

    __slots__ = ("subject", "stem", "desc", "date", "series_no")

    def __init__(self, subject: str, nii: Path, desc: str, date: str, series_no: int):
        self.subject = subject
        self.stem = nii.with_name(nii.name[:-7])   # strip ".nii.gz"
        self.desc = desc
        self.date = date
        self.series_no = series_no

    @property
    def signal(self) -> Path: return Path(str(self.stem) + ".nii.gz")
    @property
    def bval(self) -> Path: return Path(str(self.stem) + ".bval")
    @property
    def bvec(self) -> Path: return Path(str(self.stem) + ".bvec")
    @property
    def acq(self) -> str: return acq_type(self.desc)
    @property
    def key(self) -> str:
        """Stable per-session identifier used as the INR/output key."""
        return f"{self.subject}_{self.date}"

    def is_raw_dwi(self) -> bool:
        """True iff this is a real DWI series (b0 + >= MIN_DIRS directions), not a
        derived FA/ADC/TRACEW map and not a malformed/partial file."""
        if _DERIVED.search(self.desc):
            return False
        if not (self.bval.exists() and self.bvec.exists() and self.signal.exists()):
            return False
        try:
            bv = np.atleast_1d(np.loadtxt(self.bval))
        except Exception:
            return False
        if bv.ndim != 1 or bv.size < MIN_DIRS:
            return False
        return int((bv > B0_THRESH).sum()) >= MIN_DIRS and int((bv <= B0_THRESH).sum()) >= 1

    def shells(self) -> list[int]:
        bv = np.atleast_1d(np.loadtxt(self.bval))
        return sorted({int(round(b / 100.0) * 100) for b in bv if b > B0_THRESH})

def _parse(subject: str, nii: Path) -> Series | None:
    m = _TS.search(nii.name)
    if not m:
        return None
    ts, sn = m.groups()
    # desc = filename between "<PATNO>_" and "_<timestamp>_<series>"
    desc = nii.name[len(subject) + 1: m.start()]
    return Series(subject, nii, desc, ts[:8], int(sn))

def ppmi_list_subjects() -> list[str]:
    """Numeric PATNO dirs under PPMI_ROOT that contain >= 1 raw DWI session."""
    subs = []
    for d in sorted(PPMI_ROOT.iterdir()):
        if d.is_dir() and d.name.isdigit() and sessions(d.name):
            subs.append(d.name)
    return subs

def _all_series(subject: str) -> list[Series]:
    d = PPMI_ROOT / subject
    out = []
    for nii in d.glob("*.nii.gz"):
        s = _parse(subject, nii)
        if s is not None and s.is_raw_dwi():
            out.append(s)
    return out

GATED_ONLY = os.environ.get("BINR_PPMI_GATED_ONLY", "1") != "0"

def sessions(subject: str) -> dict[str, Series]:
    """date(YYYYMMDD) -> the single canonical raw-DWI series for that visit.

    Gated-only by default (GATED_ONLY); a visit with no gated series is dropped.
    Selection per session: most non-b0 directions, then lowest series number
    (deterministic)."""
    by_date: dict[str, list[Series]] = defaultdict(list)
    for s in _all_series(subject):
        if GATED_ONLY and s.acq != "gated":
            continue
        by_date[s.date].append(s)
    chosen = {}
    for date, ser in by_date.items():
        chosen[date] = sorted(
            ser,
            key=lambda s: (s.acq != "gated", -int(np.atleast_1d(np.loadtxt(s.bval)).size), s.series_no),
        )[0]
    return chosen

def baseline_series(subject: str) -> Series:
    """The earliest visit's canonical series (one INR per subject, leakage-safe)."""
    ss = sessions(subject)
    return ss[min(ss)]

def list_sessions(subjects: list[str] | None = None) -> list[Series]:
    """Flat list of canonical per-visit series across the cohort (longitudinal)."""
    subjects = subjects or ppmi_list_subjects()
    return [s for sub in subjects for s in sessions(sub).values()]

MASK_DIR = Path(__file__).resolve().parent / "OUTPUTS" / "classification" / "ppmi_masks"

MRTRIX_ENV = "nipype"

def mask_path(series: "Series") -> Path:
    return MASK_DIR / f"{series.key}.nii.gz"

def make_mask(series: "Series", force: bool = False) -> Path:
    """mrtrix3 dwi2mask on the raw DWI (cached). Mirrors make_masks.make_mask."""
    import subprocess
    out = mask_path(series)
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["conda", "run", "-n", MRTRIX_ENV, "dwi2mask", str(series.signal), str(out),
           "-fslgrad", str(series.bvec), str(series.bval), "-force"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out

def ppmi_load_mask(series: "Series") -> np.ndarray:
    """Cached mrtrix mask if present, else derive (mean DWI > 0)."""
    import nibabel as nib
    mp = mask_path(series)
    if mp.exists():
        return nib.load(str(mp)).get_fdata() > 0
    data = nib.load(str(series.signal)).get_fdata()
    mean_signal = data.mean(axis=-1) if data.ndim == 4 else data
    return mean_signal > mean_signal.max() * 0.04

COHORT_NAME = {1: "PD", 2: "HC", 3: "SWEDD", 4: "Prodromal"}

_GROUP_TO_CODE = {"PD": 1, "CONTROL": 2, "HC": 2, "HEALTHY CONTROL": 2,
                  "SWEDD": 3, "PRODROMAL": 4, "GENPD": 1, "GENUNAFF": 2}

ppmi_REPO = Path(os.environ.get("BINR_CLASSIFICATION_ROOT",
                                Path(__file__).resolve().parent))

def _label_csv() -> Path:
    env = os.environ.get("BINR_PPMI_LABELS")
    cands = [Path(env)] if env else []
    cands += sorted(ppmi_REPO.glob("Rodrigo_*.csv")) + sorted(ppmi_REPO.glob("*IDA*Metadata*.csv"))
    cands += [PPMI_ROOT / "Participant_Status.csv", Path.home() / "Participant_Status.csv"]
    for c in cands:
        if c and c.exists():
            return c
    raise FileNotFoundError(
        "PPMI labels not found. Expected the IDA collection manifest (e.g. "
        "Rodrigo_*.csv with Subject/Group columns) in the repo root, or a LONI "
        "Participant_Status.csv. Set BINR_PPMI_LABELS to override.")

def _load_label_df():
    """Normalize either schema to columns: subject, code, sex, age, visit, date."""
    import pandas as pd
    df = pd.read_csv(_label_csv(), dtype=str)
    cols = {c.upper(): c for c in df.columns}
    if "SUBJECT" in cols and "GROUP" in cols:          # IDA collection manifest
        out = pd.DataFrame({
            "subject": df[cols["SUBJECT"]].astype(str),
            "code": df[cols["GROUP"]].str.upper().str.strip().map(_GROUP_TO_CODE),
            "sex": df[cols["SEX"]].str.upper() if "SEX" in cols else None,
            "age": pd.to_numeric(df[cols["AGE"]], errors="coerce") if "AGE" in cols else None,
            "visit": df[cols["VISIT"]] if "VISIT" in cols else None,
            "modality": df[cols["MODALITY"]] if "MODALITY" in cols else "DTI",
        })
        out = out[out["modality"] == "DTI"]            # diffusion rows only
    elif "PATNO" in cols and "COHORT" in cols:         # LONI Participant_Status
        out = pd.DataFrame({
            "subject": df[cols["PATNO"]].apply(lambda v: str(int(float(v))) if v else v),
            "code": pd.to_numeric(df[cols["COHORT"]], errors="coerce"),
            "sex": df[cols["SEX"]] if "SEX" in cols else None,
            "age": None, "visit": None, "modality": "DTI",
        })
    else:
        raise KeyError(f"Unrecognized label schema (cols={list(df.columns)})")
    return out.dropna(subset=["code"]).astype({"code": int})

def cohort_map() -> dict[str, int]:
    """PATNO(str) -> cohort code (1=PD,2=HC/Control,3=SWEDD,4=Prodromal)."""
    df = _load_label_df()
    return {r.subject: int(r.code) for r in df.drop_duplicates("subject").itertuples()}

def covariates(subjects: list[str]) -> dict[str, np.ndarray]:
    """Per-subject Age (float) and Sex (1=M,0=F) for confound control."""
    df = _load_label_df().drop_duplicates("subject").set_index("subject")
    age, sex = [], []
    for s in subjects:
        r = df.loc[s] if s in df.index else None
        age.append(float(r["age"]) if r is not None and r["age"] == r["age"] else np.nan)
        sex.append(1.0 if (r is not None and str(r["sex"]).upper().startswith("M")) else 0.0)
    return {"age": np.array(age), "sex": np.array(sex)}

def ppmi_get_cohort(positive=(1,), negative=(2,), b1000_only=True) -> tuple[list[str], np.ndarray, list[str]]:
    """Subject-level cohort (baseline visit per subject) for PD(1) vs Control(0).

    Returns (subjects, labels, dates) over extracted subjects that carry a label.
    SWEDD/Prodromal excluded by default; pass positive/negative cohort codes to
    change the contrast. b1000_only (default) drops the handful of legacy b=600
    sessions so the cohort is a single homogeneous protocol (b=1000)."""
    cmap = cohort_map()
    subs, y, dates = [], [], []
    for s in ppmi_list_subjects():
        c = cmap.get(s)
        if c in positive:
            lab = 1
        elif c in negative:
            lab = 0
        else:
            continue
        bl = baseline_series(s)
        if b1000_only and bl.shells() != [1000]:
            continue
        subs.append(s)
        y.append(lab)
        dates.append(bl.date)
    return subs, np.array(y, int), dates

INR_DIR = ppmi_REPO / "OUTPUTS" / "classification" / os.environ.get("BINR_PPMI_INR_SUBDIR", "inrs_ppmi_bs256")

def inr_dir(tag: str | None = None) -> Path:
    """Trained-INR dir; a tag (e.g. 'ep30') routes to a sibling for epoch sweeps."""
    return INR_DIR if not tag else (INR_DIR.parent / f"{INR_DIR.name}_{tag}")

def subject_config(series: "Series", epochs: int, hidden_size: int = 256,
                   tag: str | None = None, **kw):
    """Build a per-session ball-stick INR training config for a PPMI series.

    Reuses train_subject_inrs.build_config (ball-stick l_max=2, random shared init)
    then patches in PPMI's data paths, the single b=1000 shell, and PPMI output dirs.

    NOTE (single-shell caveat): the ball-fODF model separates ball (D_iso) from
    stick (D_par) using shell contrast; with one shell those diffusivities are
    weakly identified even though the *signal* fit and the derived MD remain valid.
    Same recipe as HCP/ParkMR otherwise, so the weight vectors stay comparable.
    """
    from .train_subject_inrs import build_config
    cfg = build_config(series.key, epochs=epochs, ballstick=4, hidden_size=hidden_size, **kw)
    out = inr_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    cfg.update({
        "data": str(series.signal),
        "bval": str(series.bval),
        "bvec": str(series.bvec),
        "mask": str(mask_path(series)),
        "bval_train": SHELLS, "bval_val": "all",
        "b0_threshold": B0_THRESH,
        "save_state_dict_path": str(out / f"{series.key}.pt"),
        "checkpoint_dirpath": str(out / "_ckpt_tmp" / series.key),
    })
    return cfg

def ppmi_main():
    subs = ppmi_list_subjects()
    sess = list_sessions(subs)
    print(f"PPMI_ROOT = {PPMI_ROOT}")
    print(f"subjects (>=1 raw DWI session): {len(subs)}")
    print(f"canonical raw-DWI sessions:     {len(sess)}")

    spc = defaultdict(int)
    for s in subs:
        spc[len(sessions(s))] += 1
    print("sessions/subject:", {k: spc[k] for k in sorted(spc)})

    acq = defaultdict(int)
    for s in sess:
        acq[s.acq] += 1
    print("acquisition type (canonical series):", dict(acq))

    sh = defaultdict(int)
    for s in sess:
        sh[tuple(s.shells())] += 1
    print("non-b0 shells present:")
    for k, n in sorted(sh.items(), key=lambda x: -x[1]):
        print(f"  {n:4d} sessions  b={list(k)}")

    try:
        subj, y, _ = ppmi_get_cohort()
        cov = covariates(subj)
        print(f"\ncohort (PD vs Control, baseline visit): n={len(subj)} | "
              f"PD={int((y == 1).sum())} Control={int((y == 0).sum())}")
        print(f"  age mean={np.nanmean(cov['age']):.1f} (PD {np.nanmean(cov['age'][y==1]):.1f} / "
              f"HC {np.nanmean(cov['age'][y==0]):.1f}) | %male={100*cov['sex'].mean():.0f}")
        cmap = cohort_map()
        comp = defaultdict(int)
        for s in subs:
            comp[COHORT_NAME.get(cmap.get(s), "unlabeled")] += 1
        print("full extracted-cohort composition:", dict(comp))
    except FileNotFoundError as e:
        print(f"\n[labels] {e}")

# ==========================================================================
# hcp: hcp_labels.py
# ==========================================================================

"""HCP-YA subjects: 3-class age labels, gender (control task), stratified sampling,
and per-subject multi-shell dwi accessors (all shells, matching baseline_mdfa).

Data:
  - preprocessed dwi: HCP_preprocessed/subject_<id>/dwi_<v>.npy  (3D per volume)
  - gradient table + mask path: HCP_preprocessed/dataset.csv columns
        subject_id, volume_id, image_path, bvalue, bvec_x, bvec_y, bvec_z, brainmask
  - labels: HCP_YA_subjects_*.csv  (Subject, Gender, Age[bins])

Age is the HCP 5-yr bin (22-25 / 26-30 / 31-35 / 36+). We use the 3 well-populated
classes and DROP 36+ (only ~5 subjects -> untrainable/untestable).
"""

hcp_REPO = Path(__file__).resolve().parent

PREP_DIR = Path(os.environ.get("BINR_HCP_PREPROCESSED",
                               "/data/HCP_preprocessed"))

DATASET_CSV = PREP_DIR / "dataset.csv"

LABEL_CSV = next(iter(sorted(glob.glob(str(hcp_REPO / "HCP_YA_subjects_*.csv")))), None)

AGE_CLASSES = ["22-25", "26-30", "31-35"]

AGE_TO_IDX = {a: i for i, a in enumerate(AGE_CLASSES)}

def _labels_df():
    import pandas as pd
    df = pd.read_csv(LABEL_CSV, usecols=["Subject", "Gender", "Age"])
    df["Subject"] = df["Subject"].astype(str)
    return df

def _label_maps():
    df = _labels_df()
    age = dict(zip(df["Subject"], df["Age"]))
    sex = dict(zip(df["Subject"], df["Gender"]))
    return age, sex

def _preprocessed_ids() -> frozenset:
    ids = [os.path.basename(d).replace("subject_", "")
           for d in glob.glob(str(PREP_DIR / "subject_*"))]
    return frozenset(ids)

def hcp_list_subjects() -> list:
    """Subjects with BOTH preprocessed dwi and a usable (3-class) age label."""
    age, _ = _label_maps()
    pp = _preprocessed_ids()
    subs = [s for s in sorted(pp) if age.get(s) in AGE_TO_IDX]
    return subs

def age_class(subject: str) -> int:
    age, _ = _label_maps()
    return AGE_TO_IDX[age[subject]]

def gender(subject: str) -> int:
    _, sex = _label_maps()
    return 1 if sex[subject] == "M" else 0

def labels_array(subjects, task: str = "age") -> np.ndarray:
    fn = age_class if task == "age" else gender
    return np.array([fn(s) for s in subjects])

def stratified_sample(n: int, task: str = "age", seed: int = 0) -> list:
    """Stratified subject sample: ~equal per class (age bins or sex)."""
    subs = hcp_list_subjects()
    y = labels_array(subs, task)
    rng = np.random.default_rng(seed)
    classes = sorted(set(y.tolist()))
    per = n // len(classes)
    picked = []
    for c in classes:
        idx = np.where(y == c)[0]
        take = min(per, len(idx))
        picked.extend(np.array(subs)[rng.choice(idx, take, replace=False)].tolist())
    rng.shuffle(picked)
    return picked

def _volume_index():
    """subject_id -> (sorted volume_ids, bvals[N], bvecs[N,3], mask_path)."""
    import pandas as pd
    df = pd.read_csv(DATASET_CSV,
                     usecols=["subject_id", "volume_id", "bvalue",
                              "bvec_x", "bvec_y", "bvec_z", "brainmask"])
    df["subject_id"] = df["subject_id"].astype(str)
    out = {}
    for sid, g in df.groupby("subject_id"):
        g = g.sort_values("volume_id")
        out[sid] = (
            g["volume_id"].to_numpy(),
            g["bvalue"].to_numpy(dtype=np.float64),
            g[["bvec_x", "bvec_y", "bvec_z"]].to_numpy(dtype=np.float64),
            g["brainmask"].iloc[0],
        )
    return out

def subject_gtab(subject: str):
    """(bvals, bvecs) over ALL shells, in volume order."""
    _, bvals, bvecs, _ = _volume_index()[subject]
    return bvals, bvecs

def load_dwi_masked(subject: str):
    """Memory-light loader: (signal[n_vox, N], mask) over ALL shells.

    Loads one volume at a time and keeps only in-mask voxels, so peak memory is
    ~n_vox*N*4 bytes (~0.85 GB) instead of the full 4D (~4.2 GB). dipy fits a 2D
    [n_vox, N] array directly, yielding 1-D MD/FA over the masked voxels.
    """
    vol_ids, _, _, _ = _volume_index()[subject]
    sdir = PREP_DIR / f"subject_{subject}"
    mask = hcp_load_mask(subject)
    nvox = int(mask.sum())
    sig = np.empty((nvox, len(vol_ids)), dtype=np.float32)
    for i, v in enumerate(vol_ids):
        sig[:, i] = np.load(sdir / f"dwi_{v}.npy")[mask]
    return sig, mask

def hcp_load_mask(subject: str) -> np.ndarray:
    import nibabel as nib
    _, _, _, mpath = _volume_index()[subject]
    p = mpath if os.path.isabs(mpath) else str((PREP_DIR.parent / mpath.lstrip("./")).resolve())
    if not os.path.exists(p):
        p = str(PREP_DIR / f"subject_{subject}" / "nodif_brain_mask.nii.gz")
    return nib.load(p).get_fdata() > 0

# ==========================================================================
# shared INR config helpers (train_subject_inrs.py)
# ==========================================================================

TRAIN_BASE_CONFIG = Path(__file__).resolve().parent / "configs" / "ball_fodf_hcp.yml"

def apply_ballstick(cfg: dict, n_sticks: int = 4):
    """Switch to the ball-and-stick physics INR (latest frontier work, src/
    ball_stick.py). The network outputs per-voxel physical diffusion params
    (f_ball, D_iso, D_par, N stick directions); signal is the closed-form
    ball+stick forward. MD/FA derive ANALYTICALLY from the params -> faithful
    (MD ~PSNR 34), unlike SHORE-recon + DTI refit. New output head -> random
    shared init, trained head."""
    # Mirror the autoresearch/20260423-ep6 FRONTIER: ball-fODF l_max=2 (8 raw
    # params, "ties ball-stick on held-out") + per-voxel S0 scaling. out_size is
    # auto-set by main.py for ball-stick, so we don't hardcode it.
    cfg.update({
        "model_output_type": "ball_stick",
        "ball_stick_use_fodf": True,
        "ball_stick_fodf_l_max": 2,
        "ball_stick_n_sticks": int(n_sticks),   # ignored in fodf mode
        "out_size": 8,                          # n_raw_params_fodf(l_max=2) = 8
        "use_b0_output_scale": True,            # per-voxel S0 (required by ball-stick)
        "shore_output": True,                   # ball-stick path is gated on this
        "shore_aniso_weight": 0.0,
        "pretrained_weights_path": None,
        "output_lr_mult": 1.0,
    })
    return cfg

# ==========================================================================
# hcp INRs: train_hcp_inrs.py (model rebuild for loading saved weights)
# ==========================================================================

"""Train one ball-stick INR per HCP-YA subject (all shells), for the age weight-
classification experiment. Mirrors classification/train_subject_inrs.py but feeds
the HCP_preprocessed per-volume .npy data via a monkeypatched loader instead of a
4D NIfTI -- no NIfTI is written to disk.

Architecture matches the ParkMR compact ball-stick (width 256, 7 layers, ball-fODF
l_max=2, 8 raw params/voxel) so the SAME weight_bench reps apply. ALL shells used
(bval_train="all"), per the all-shell directive.

Usage:
    PYTHONPATH=$(pwd) conda run -n torch python -m classification.train_hcp_inrs \
        --cohort-json OUTPUTS/classification/hcp_age_sample_n150.json \
        --epochs 20 --hidden-size 256 [--limit 5]
"""

TRAIN_REPO_ROOT = Path(__file__).resolve().parent

SRC_DIR = TRAIN_REPO_ROOT / "src"

for p in (str(TRAIN_REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

HCP_INR_DIR = TRAIN_REPO_ROOT / "OUTPUTS" / "classification" / os.environ.get("BINR_HCP_INR_SUBDIR", "inrs_hcp_bs256")

MANIFEST = TRAIN_REPO_ROOT / "OUTPUTS" / "classification" / "manifest_hcp.csv"

def _hcp_load_data(cfg, return_header=False):
    import torch
    subject = cfg["hcp_subject"]
    sig, mask = load_dwi_masked(subject)          # sig [nvox, C]
    bvals, bvecs = subject_gtab(subject)
    # scatter masked signal back into a dense 4D volume the dataset expects
    dwi = np.zeros((*mask.shape, sig.shape[1]), dtype=np.float32)
    dwi[mask] = sig
    dwi = torch.from_numpy(dwi)
    mask_t = torch.from_numpy(mask.astype(np.float32))
    bvec_t = torch.tensor(bvecs, dtype=torch.float32)        # [C,3]
    bval_t = torch.tensor(bvals, dtype=torch.float32)        # [C]
    if return_header:
        return dwi, mask_t, bvec_t, bval_t, None
    return dwi, mask_t, bvec_t, bval_t

def _install_loader():
    """Patch load_data everywhere it is referenced (train + eval pool)."""
    from inrdiff import main, utils
    utils.load_data = _hcp_load_data
    main.load_data = _hcp_load_data
    try:
        import experiment
        experiment.load_data = _hcp_load_data
    except Exception:
        pass

def build_config_hcp(subject: str, epochs: int, hidden_size: int = 256, num_layers: int = 7):
    from inrdiff.main import load_config
    cfg = load_config(str(TRAIN_BASE_CONFIG))
    HCP_INR_DIR.mkdir(parents=True, exist_ok=True)
    cfg.update({
        "hcp_subject": subject,                 # read by the patched loader
        "data": "UNUSED", "bvec": "UNUSED", "bval": "UNUSED",
        "mask": "UNUSED",
        # SHORE/ball-stick representation (ball-stick gated on shore_output)
        "shore_output": True, "sh_output": False, "use_b0_prior": False,
        "b0_threshold": 10, "shore_b0_normalize": False,
        "shore_radial_order": 8, "shore_zeta": 700,
        "bval_train": "all", "bval_val": "all",     # ALL shells (b0 stripped by threshold)
        "bval_encoding": "linear", "normalize_0_1": True,
        "hidden_size": int(hidden_size), "num_layers": int(num_layers),
        "incode_harmonizer_hidden": [128, 64],
        "epochs": epochs, "train_seed": 123,
        "eval_compute_ssim": True, "eval_max_voxels": None,
        "val_skip": True, "steps_per_viz": int(1e12),
        "disable_checkpointing": True, "checkpoint_path": None,
        "checkpoint_dirpath": str(HCP_INR_DIR / "_ckpt_tmp" / subject),
        "save_state_dict_path": str(HCP_INR_DIR / f"{subject}.pt"),
    })
    apply_ballstick(cfg, n_sticks=4)            # ball-fODF l_max=2, out_size=8, trained head
    cfg["hidden_size"] = int(hidden_size)       # apply_ballstick doesn't touch width
    cfg["num_layers"] = int(num_layers)
    return cfg

def train_one(subject: str, epochs: int, hidden_size: int = 256, do_eval: bool = True) -> dict:
    import shutil
    from inrdiff.main import train_with_config
    cfg = build_config_hcp(subject, epochs, hidden_size)
    shutil.rmtree(cfg["checkpoint_dirpath"], ignore_errors=True)  # main.py makedirs w/o exist_ok
    t0 = time.time()
    pl_module, _, _ = train_with_config(cfg)
    dt = time.time() - t0
    recon = {}
    if do_eval:
        try:
            recon = _recon_psnr(pl_module, cfg, subject)
        except Exception as e:
            recon = {"psnr": float("nan"), "err": str(e)[:120]}
    n_params = sum(p.numel() for p in pl_module.parameters())
    return {"subject": subject, "age": int(age_class(subject)),
            "gender": int(gender(subject)),
            "psnr": recon.get("psnr") if recon else None, "n_params": n_params,
            "epochs": epochs, "hidden_size": hidden_size, "train_sec": round(dt, 1),
            **({"err": recon["err"]} if "err" in recon else {})}

def _recon_psnr(model, cfg, subject) -> dict:
    """Recon PSNR over all volumes via the locked eval pool (HCP loader patched in)."""
    import experiment
    import torch
    from inrdiff.acquisition_adaptation import aggregate_metrics, evaluate_all_volumes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    eval_cfg, pool = experiment._build_eval_pool(cfg, subject)
    df = evaluate_all_volumes(model, pool, eval_cfg, None, device=device)
    overall, _ = aggregate_metrics(df)
    return {"psnr": round(float(overall.get("psnr", float("nan"))), 3),
            "ssim": round(float(overall.get("ssim", float("nan"))), 4)}

def hcp_train_inrs():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort-json", default=str(TRAIN_REPO_ROOT / "OUTPUTS/classification/hcp_age_sample_n150.json"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="only first N subjects (smoke test)")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    _install_loader()
    subs = args.subjects or json.load(open(args.cohort_json))["subjects"]
    if args.limit:
        subs = subs[:args.limit]
    HCP_INR_DIR.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(MANIFEST).to_dict("records") if MANIFEST.exists() else []
    rows = [r for r in rows if r["subject"] not in set(subs)]
    for i, s in enumerate(subs, 1):
        print(f"\n===== [{i}/{len(subs)}] HCP INR {s} (age={age_class(s)}) "
              f"@ {args.epochs}ep w{args.hidden_size} =====", flush=True)
        r = train_one(s, args.epochs, args.hidden_size)
        rows.append(r)
        pd.DataFrame(rows).to_csv(MANIFEST, index=False)
        print(f"  saved -> {HCP_INR_DIR / (s + '.pt')}  PSNR={r.get('psnr')} ({r['train_sec']}s)"
              + (f"  ERR={r['err']}" if "err" in r else ""), flush=True)
    print(f"\nmanifest -> {MANIFEST}")

# ==========================================================================
# hcpdec: hcp_decode.py
# ==========================================================================

"""Decode-then-classify features from HCP ball-stick INR weights.

The certifiable weight readout (analog of ParkMR recon-MD): run each subject's
trained INR forward on its brain-mask coords, decode the ball-fODF raw outputs into
per-voxel physical params (f_ball, D_iso, D_par, MD, fODF anisotropy), and summarize
each distribution as quantiles. A fixed physics decoder -> the features are a
function of the weights but biologically meaningful (no free read-out of raw weights).

Cached per subject to OUTPUTS/classification/hcp_decode/<subject>_phys.npz.
"""

hcpdec_REPO = Path(__file__).resolve().parent

for p in (str(hcpdec_REPO), str(hcpdec_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

hcpdec_DECODE_DIR = (hcpdec_REPO / "OUTPUTS" / "classification" /
              ("hcp_decode" if HCP_INR_DIR.name == "inrs_hcp_bs256"
               else f"hcp_decode_{HCP_INR_DIR.name}"))

hcpdec_L_MAX = 2

QUANTILES = np.linspace(0.05, 0.95, 19)

def hcpdec__build_model(subject, hidden_size=256, device="cuda"):
    import torch
    from acquisition_adaptation import create_network
    from model import INRLightningModule
    cfg = build_config_hcp(subject, epochs=0, hidden_size=hidden_size)
    cfg["in_size"] = 2 * int(cfg.get("spatial_freq_num", 256))   # SpatialOnlyEncoder out
    network, pos_encoder = create_network(cfg)
    pl = INRLightningModule(network, train_loader=None, val_loaders_dict=None,
                            pos_encoder=pos_encoder, shore_output=True)
    sd = torch.load(str(HCP_INR_DIR / f"{subject}.pt"), map_location="cpu")
    pl.load_state_dict(sd, strict=False)
    pl.to(device).eval()
    return pl

def physics_params(subject, hidden_size=256, device="cuda") -> dict:
    """In-memory + disk cached. NOTE: callers must treat the returned dict as
    read-only (it is shared across calls via lru_cache)."""
    hcpdec_DECODE_DIR.mkdir(parents=True, exist_ok=True)
    cache = hcpdec_DECODE_DIR / f"{subject}_phys.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    import torch
    import ball_stick
    model = hcpdec__build_model(subject, hidden_size, device)
    mask = hcp_load_mask(subject)
    dim = mask.shape
    xyz = np.stack(np.where(mask), 1).astype(np.float32)
    coords = torch.tensor((2 * xyz / np.array(dim) - 1).astype(np.float32), device=device)
    with torch.no_grad():
        raw = torch.cat([model(coords[i:i + 200000]) for i in range(0, coords.shape[0], 200000)], 0)
    p = ball_stick.decode_params_fodf(raw, hcpdec_L_MAX)
    f_ball, D_iso, D_par, fodf = p["f_ball"], p["D_iso"], p["D_par"], p["fODF_lm"]
    md = f_ball * D_iso + (1 - f_ball) * D_par / 3.0
    aniso = (fodf[:, 1:] ** 2).sum(1).clamp_min(0).sqrt() / (fodf[:, 0].abs() + 1e-6)
    out = {"f_ball": f_ball.cpu().numpy(), "D_iso": D_iso.cpu().numpy(),
           "D_par": D_par.cpu().numpy(), "md": md.cpu().numpy(), "aniso": aniso.cpu().numpy()}
    np.savez(cache, **out)
    return out

def _subject_feat(subject, keys):
    """Cached per-subject quantile feature vector (so the permutation null doesn't
    recompute quantiles over ~750k voxels on every fold)."""
    pp = physics_params(subject)
    return np.concatenate([np.quantile(pp[k], QUANTILES) for k in keys])

def physics_features(subjects, keys=("f_ball", "D_iso", "D_par", "md", "aniso")) -> np.ndarray:
    return np.array([_subject_feat(s, tuple(keys)) for s in subjects])

def recon_dti_md(subject, hidden_size=256, device="cuda") -> np.ndarray:
    """RECONSTRUCT the dMRI signal from the INR, then fit the SAME all-shell DTI
    the image baseline uses -> DTI-MD. The apples-to-apples 'recon-MD from weights'
    (analog of ParkMR ballstick_md.recon_dti_md). S0 is taken from the INR's stored
    per-voxel _b0_volume_scale (part of the weights) -> pure from-weights surrogate.
    Cached per subject."""
    hcpdec_DECODE_DIR.mkdir(parents=True, exist_ok=True)
    cache = hcpdec_DECODE_DIR / f"{subject}_reconmd.npy"
    if cache.exists():
        return np.load(cache)
    import torch
    import ball_stick
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    bvals, bvecs = subject_gtab(subject)
    mask = hcp_load_mask(subject)
    dim = mask.shape
    b0m = bvals <= 10
    nb = ~b0m
    bv_nb = bvals[nb].astype(np.float32)
    bvec_nb = bvecs[nb].astype(np.float32)
    bvec_nb = bvec_nb / (np.linalg.norm(bvec_nb, axis=1, keepdims=True) + 1e-9)

    model = hcpdec__build_model(subject, hidden_size, device)
    sd = torch.load(str(HCP_INR_DIR / f"{subject}.pt"), map_location="cpu")
    S0 = sd["_b0_volume_scale"].squeeze().numpy()[mask].astype(np.float32)   # [N] from weights
    xyz = np.stack(np.where(mask), 1).astype(np.float32)
    coords = torch.tensor((2 * xyz / np.array(dim) - 1).astype(np.float32), device=device)
    with torch.no_grad():
        raw = torch.cat([model(coords[i:i + 200000]) for i in range(0, coords.shape[0], 200000)], 0)
    Y = torch.from_numpy(ball_stick.compute_sh_basis_at_bvecs(bvec_nb, hcpdec_L_MAX)).float().to(device)
    uniq, b_idx = np.unique(np.round(bv_nb), return_inverse=True)
    quad = ball_stick.make_legendre_quadrature(hcpdec_L_MAX)
    l_idx = ball_stick.sh_coef_to_l_index(hcpdec_L_MAX)
    bvals_t = torch.from_numpy(bv_nb).to(device); bvecs_t = torch.from_numpy(bvec_nb).to(device)
    b_idx_t = torch.from_numpy(b_idx).long().to(device)
    uniq_t = torch.from_numpy(uniq.astype(np.float32)).to(device)
    quad_d = {"t": torch.from_numpy(quad["t"]).to(device), "Pl_w": torch.from_numpy(quad["Pl_w"]).to(device)}
    l_idx_t = torch.from_numpy(l_idx).long().to(device)
    # chunk the signal reconstruction over VOXELS so peak GPU memory stays bounded
    # (signal_at_fodf on all ~740k voxels x 271 dirs at once OOMs; ~5GB/subject)
    E_chunks = []
    with torch.no_grad():
        for i in range(0, raw.shape[0], 80000):
            Ec = ball_stick.signal_at_fodf(
                raw[i:i + 80000], bvals_t, bvecs_t, hcpdec_L_MAX, Y_at_bvecs=Y,
                b_idx_per_vol=b_idx_t, unique_bvals=uniq_t, quad_data=quad_d, l_idx_per_coef=l_idx_t)
            E_chunks.append(Ec.cpu().numpy())
    E = np.concatenate(E_chunks, 0)
    dwi_nb = E * S0[:, None]
    sig = np.concatenate([S0[:, None], dwi_nb], 1)                  # prepend reconstructed B0
    allb = np.concatenate([[0.0], bv_nb])
    allv = np.concatenate([[[0, 0, 0]], bvec_nb], 0)
    fit = TensorModel(gradient_table(allb, bvecs=allv)).fit(sig[:, None, None, :])
    md = np.nan_to_num(np.asarray(fit.md).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    np.save(cache, md)
    return md

def _recon_md_feat(subject):
    return np.quantile(recon_dti_md(subject), QUANTILES)

def recon_md_features(subjects) -> np.ndarray:
    return np.array([_recon_md_feat(s) for s in subjects])

# ==========================================================================
# features: probe_features.py
# ==========================================================================

"""Sparse functional probing of a FIXED per-subject INR (research question in
DIRECT_WEIGHT_RQ.md).

Idea: the phenotype signal is functional (in the INR's input->output map), so we *evaluate*
the network -- but only at a SMALL set of K probe coordinates, never densely reconstructing all
~740k voxels. Per probe point we read the 8 ball-fODF outputs AND the 256-d penultimate
activation, then pool DISTRIBUTIONALLY over the K points (quantiles + mean + std -> describes a
distribution, not a count, so it is brain-size invariant). A simple linear classifier on these
low-dim features is sample-efficient at N~150 (no large learned extractor to fit).

Two registration-free probe sets:
  - mask  : K deterministically-sampled in-mask voxels (real brain tissue; closest to decode).
  - canon : K fixed random points in normalized [-1,1]^3, shared across subjects (no mask).

Raw extraction (build model + one forward at K_max points for BOTH probe sets) is cached per
subject; pooling for any K<=K_max is a prefix-pool on the cache -> the sweep needs ONE model
build per subject. Mirrors the DecodePhysics fit/predict interface (weight_bench_hcp).
"""

feat_REPO = os.environ.get("BINR_CLASSIFICATION_ROOT",
                           str(Path(__file__).resolve().parent))

if feat_REPO not in sys.path:
    sys.path.insert(0, feat_REPO)

K_MAX = 4096

PROBE_DIR = hcpdec_DECODE_DIR.parent / f"probe_cache_{INR_DIR.name}"

feat_QUANTS = np.array([.05, .25, .5, .75, .95])

PENULT_LAYER = 6

def _canonical_coords(k_max=K_MAX, seed=0):
    """Fixed random points in normalized [-1,1]^3, shared across all subjects."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(-1, 1, size=(k_max, 3))).astype(np.float32)

def _mask_coords(subject, k_max=K_MAX, seed=0):
    """Deterministic random sample of <=k_max in-mask voxels, normalized to [-1,1].
    Returned in random order so any prefix is itself a uniform subsample."""
    mask = hcp_load_mask(subject)
    dim = np.array(mask.shape, dtype=np.float32)
    xyz = np.stack(np.where(mask), 1).astype(np.float32)             # (n_vox,3)
    rng = np.random.default_rng(abs(hash(("mask", subject, seed))) % (2**32))
    idx = rng.permutation(xyz.shape[0])[:k_max]
    return (2 * xyz[idx] / dim - 1).astype(np.float32)

def _extract_at(model, coords, device="cuda"):
    """Forward the INR at coords (K,3); return (outputs (K,8), penult acts (K,256))."""
    acts = {}
    h = model.network.hidden_layers[PENULT_LAYER].register_forward_hook(
        lambda m, i, o: acts.__setitem__("a", o.detach()))
    try:
        c = torch.tensor(coords, device=device)
        with torch.no_grad():
            out = model(c)
        return out.float().cpu().numpy(), acts["a"].float().cpu().numpy()
    finally:
        h.remove()

def extract_raw(subject, k_max=K_MAX, seed=0, device="cuda"):
    """Build the INR ONCE, evaluate at k_max mask + canonical points, cache raw arrays."""
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    cache = PROBE_DIR / f"{subject}_k{k_max}_s{seed}.npz"
    if cache.exists():
        return np.load(cache)
    model = hcpdec__build_model(subject, device=device)
    m_out, m_act = _extract_at(model, _mask_coords(subject, k_max, seed), device)
    c_out, c_act = _extract_at(model, _canonical_coords(k_max, seed), device)
    np.savez(cache, mask_out=m_out, mask_act=m_act, canon_out=c_out, canon_act=c_act)
    return np.load(cache)

def feat__pool(arr):
    """Distributional pooling over the K points: per-column quantiles + mean + std."""
    q = np.quantile(arr, feat_QUANTS, axis=0)               # (5, F)
    return np.concatenate([q.reshape(-1), arr.mean(0), arr.std(0)])

def _decode_phys(raw_np):
    """Apply decode's exact raw->physics transform to the K sampled outputs (NO signal
    reconstruction): -> (K, 5) = [f_ball, D_iso, D_par, MD, aniso]. 'Sparse decode'."""
    import ball_stick
    raw = torch.as_tensor(np.ascontiguousarray(raw_np), dtype=torch.float32)
    p = ball_stick.decode_params_fodf(raw, hcpdec_L_MAX)
    f_ball, D_iso, D_par, fodf = p["f_ball"], p["D_iso"], p["D_par"], p["fODF_lm"]
    md = f_ball * D_iso + (1 - f_ball) * D_par / 3.0
    aniso = (fodf[:, 1:] ** 2).sum(1).clamp_min(0).sqrt() / (fodf[:, 0].abs() + 1e-6)
    return torch.stack([f_ball, D_iso, D_par, md, aniso], 1).cpu().numpy()

def probe_vector(subject, probe="mask", K=1024, feat="both", k_max=K_MAX, seed=0):
    """Pooled feature vector for one subject at budget K (prefix of the cached k_max)."""
    d = extract_raw(subject, k_max, seed)
    out = d[f"{probe}_out"][:K]; act = d[f"{probe}_act"][:K]
    if feat == "out":    parts = [feat__pool(out)]
    elif feat == "act":  parts = [feat__pool(act)]
    elif feat == "phys": parts = [feat__pool(_decode_phys(out))]            # sparse decode
    elif feat == "both": parts = [feat__pool(out), feat__pool(act)]
    else:                parts = [feat__pool(_decode_phys(out)), feat__pool(act)]  # phys+act
    return np.concatenate(parts)

class ProbeFeatures:
    """Sparse-probe readout; fit/predict over reps with a 'subject' key (mirrors DecodePhysics).
    StandardScaler -> PCA -> class-balanced logistic regression."""
    def __init__(self, probe="mask", K=1024, feat="both", pca=20, C=1.0, seed=0):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        self.probe, self.K, self.feat, self.pca, self.seed = probe, K, feat, pca, seed
        self.pipe = Pipeline([("sc", StandardScaler()), ("pca", PCA(pca)),
                              ("clf", LogisticRegression(C=C, max_iter=1000,
                                                         class_weight="balanced"))])

    def _feat(self, reps):
        return np.stack([probe_vector(r["subject"], self.probe, self.K, self.feat, seed=self.seed)
                         for r in reps])

    def fit(self, reps, y):
        self.pipe.set_params(pca__n_components=min(self.pca, len(reps) - 1))
        self.pipe.fit(self._feat(reps), y); return self

    def predict(self, reps):
        return self.pipe.predict(self._feat(reps))

# ==========================================================================
# cv: cv.py
# ==========================================================================

"""Leakage-safe evaluation harness shared by every arm.

Contract (program.md §2):
  * Leave-one-subject-out (primary) — the subject is the unit.
  * Every fold-dependent transform (standardization, PCA, selection) is fit on
    the TRAINING fold only. We enforce this by always wrapping the estimator in
    an sklearn Pipeline and refitting per fold.
  * Every reported score carries a label-permutation null (>=1000 shuffles, the
    ENTIRE CV re-run per shuffle) and an empirical p-value.

The estimator is supplied as a *factory* (zero-arg callable returning a fresh,
unfitted sklearn estimator/Pipeline) so each fold and each permutation gets a
clean model with no state bleed.
"""

EstimatorFactory = Callable[[], object]

def _splits(X, y, groups=None):
    """Yield (train_idx, test_idx) folds. LOO-subject by default; if `groups` is
    given (e.g. scan-session date), leave-one-GROUP-out so a held-out session is
    never seen in training (defeats session-batch leakage). Test folds may hold
    multiple subjects."""
    if groups is None:
        return list(LeaveOneOut().split(X))
    return list(LeaveOneGroupOut().split(X, y, groups))

def default_logreg() -> Pipeline:
    """Standardize-then-logistic-regression, class-balanced. The simple default."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])

def _loo_oof_scores(X: np.ndarray, y: np.ndarray, factory: EstimatorFactory,
                    splits=None):
    """Out-of-fold hard predictions and positive-class scores.

    Returns (y_pred, y_score) aligned with y. All fitting happens inside the
    fold on training rows only -> no leakage. `splits` defaults to LOO-subject;
    pass grouped splits for leave-one-session-out. Folds may hold >1 test row.
    """
    n = len(y)
    y_pred = np.empty(n, dtype=int)
    y_score = np.empty(n, dtype=float)
    for tr, te in (splits if splits is not None else _splits(X, y)):
        model = clone(factory()) if hasattr(factory(), "get_params") else factory()
        model.fit(X[tr], y[tr])
        for i in te:
            y_pred[i] = int(model.predict(X[i:i + 1])[0])
            y_score[i] = _positive_score(model, X[i:i + 1])
    return y_pred, y_score

def _positive_score(model, x_row: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x_row)[0, 1])
    if hasattr(model, "decision_function"):
        return float(np.ravel(model.decision_function(x_row))[0])
    return float(model.predict(x_row)[0])

class CVResult:
    bal_acc: float
    auc: float
    sensitivity: float
    specificity: float
    perm_p: float
    n_perm: int
    null_mean: float
    null_std: float
    y_pred: np.ndarray = field(repr=False)
    y_score: np.ndarray = field(repr=False)

    def as_row(self) -> dict:
        return {
            "bal_acc": round(self.bal_acc, 4),
            "auc": round(self.auc, 4),
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "perm_p": round(self.perm_p, 4),
            "n_perm": self.n_perm,
            "null_mean": round(self.null_mean, 4),
        }

def evaluate_features(
    X: np.ndarray,
    y: np.ndarray,
    factory: EstimatorFactory = default_logreg,
    n_perm: int = 1000,
    seed: int = 0,
    groups=None,
) -> CVResult:
    """Run CV + label-permutation null on a (n_subjects x d) feature matrix.

    Default CV is LOO-subject; pass `groups` (e.g. scan-session date) for
    leave-one-session-out. The permutation statistic is balanced accuracy; the
    entire CV loop is re-run on each shuffled label vector.
    p = (1 + #{null >= observed}) / (1 + n_perm).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim == 1:
        X = X[:, None]
    splits = _splits(X, y, groups)

    y_pred, y_score = _loo_oof_scores(X, y, factory, splits)
    bal_acc = balanced_accuracy_score(y, y_pred)
    auc = _safe_auc(y, y_score)
    sens, spec = _sens_spec(y, y_pred)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        yp = rng.permutation(y)
        pred, _ = _loo_oof_scores(X, yp, factory, splits)
        null[k] = balanced_accuracy_score(yp, pred)
    perm_p = (1 + int((null >= bal_acc).sum())) / (1 + n_perm)

    return CVResult(
        bal_acc=bal_acc, auc=auc, sensitivity=sens, specificity=spec,
        perm_p=perm_p, n_perm=n_perm, null_mean=float(null.mean()),
        null_std=float(null.std()), y_pred=y_pred, y_score=y_score,
    )

def evaluate_reduced(
    X: np.ndarray,
    y: np.ndarray,
    reducer_factory: EstimatorFactory,
    clf_factory: EstimatorFactory = default_logreg,
    n_perm: int = 1000,
    seed: int = 0,
    groups=None,
) -> CVResult:
    """LOO CV + permutation null with a y-INDEPENDENT dimensionality reducer.

    The reducer (e.g. StandardScaler -> PCA) is fit on each fold's TRAINING rows
    only and applied to all rows; because it never sees y, its per-fold output is
    cached and reused across all label permutations. Only the (cheap) classifier
    is refit per permutation. This makes high-dimensional weight arms (PCA on
    ~17M-dim weight deltas) tractable while staying leakage-safe.
    """
    X = np.asarray(X)  # preserve dtype (weight deltas are float32, ~2GB at full dim)
    y = np.asarray(y, dtype=int)
    n = len(y)

    # Precompute the y-independent reduced features for every fold.
    folds = []  # (train_idx, test_idx, X_reduced_all_rows)
    for tr, te in _splits(X, y, groups):
        red = reducer_factory()
        red.fit(X[tr])
        folds.append((tr, te, red.transform(X)))

    def run(yy: np.ndarray):
        pred = np.empty(n, dtype=int)
        score = np.empty(n, dtype=float)
        for tr, te, Xr in folds:
            clf = clf_factory()
            clf.fit(Xr[tr], yy[tr])
            for i in te:
                pred[i] = int(clf.predict(Xr[i:i + 1])[0])
                score[i] = _positive_score(clf, Xr[i:i + 1])
        return pred, score

    y_pred, y_score = run(y)
    bal_acc = balanced_accuracy_score(y, y_pred)
    auc = _safe_auc(y, y_score)
    sens, spec = _sens_spec(y, y_pred)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        yp = rng.permutation(y)
        pred, _ = run(yp)
        null[k] = balanced_accuracy_score(yp, pred)
    perm_p = (1 + int((null >= bal_acc).sum())) / (1 + n_perm)

    return CVResult(
        bal_acc=bal_acc, auc=auc, sensitivity=sens, specificity=spec,
        perm_p=perm_p, n_perm=n_perm, null_mean=float(null.mean()),
        null_std=float(null.std()), y_pred=y_pred, y_score=y_score,
    )

def evaluate_kernel_pca(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    clf_factory: EstimatorFactory = default_logreg,
    n_perm: int = 1000,
    seed: int = 0,
    groups=None,
) -> CVResult:
    """Raw-weights linear classifier via the linear-kernel (Gram) trick.

    For d-dim weights with d >> n, a linear model lives entirely in the n-sample
    span. We compute the n x n linear Gram K = X X^T ONCE, then per fold do
    kernel-PCA (linear kernel) to k components — mathematically identical to PCA
    on the raw weights, but O(n^2) instead of O(n*d) per fold. k = n-1 is the
    lossless raw-weights linear classifier (no feature engineering); smaller k
    regularizes. The Gram is y-independent so the permutation null is ~free.
    """
    from sklearn.decomposition import KernelPCA

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    n = len(y)
    K = X @ X.T  # (n, n) linear Gram, computed once

    folds = []  # (train_idx, test_idx, Z_all_rows)
    for tr, te in _splits(X, y, groups):
        kk = min(k, len(tr) - 1)
        kpca = KernelPCA(n_components=kk, kernel="precomputed")
        kpca.fit(K[np.ix_(tr, tr)])          # fit on train-only kernel block
        Z = kpca.transform(K[:, tr])         # project all rows onto train basis
        folds.append((tr, te, Z))

    def run(yy: np.ndarray):
        pred = np.empty(n, dtype=int)
        score = np.empty(n, dtype=float)
        for tr, te, Z in folds:
            clf = clf_factory()
            clf.fit(Z[tr], yy[tr])
            for i in te:
                pred[i] = int(clf.predict(Z[i:i + 1])[0])
                score[i] = _positive_score(clf, Z[i:i + 1])
        return pred, score

    y_pred, y_score = run(y)
    bal_acc = balanced_accuracy_score(y, y_pred)
    auc = _safe_auc(y, y_score)
    sens, spec = _sens_spec(y, y_pred)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for j in range(n_perm):
        yp = rng.permutation(y)
        pred, _ = run(yp)
        null[j] = balanced_accuracy_score(yp, pred)
    perm_p = (1 + int((null >= bal_acc).sum())) / (1 + n_perm)

    return CVResult(
        bal_acc=bal_acc, auc=auc, sensitivity=sens, specificity=spec,
        perm_p=perm_p, n_perm=n_perm, null_mean=float(null.mean()),
        null_std=float(null.std()), y_pred=y_pred, y_score=y_score,
    )

def _safe_auc(y: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, y_score))

def _sens_spec(y: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    pos = y == 1
    neg = y == 0
    sens = float((y_pred[pos] == 1).mean()) if pos.any() else float("nan")
    spec = float((y_pred[neg] == 0).mean()) if neg.any() else float("nan")
    return sens, spec

# ==========================================================================
# driver: ppmi_site_cv_all.py
# ==========================================================================

"""Leave-one-group-out confound control across EVERY PPMI readout.

`ppmi_site_cv.py` was written as a targeted check on the one positive result: it is
hardcoded to the SN caches (`ppmi_sn_full`, `ppmi_sn_sparse`) and tests five SN
readouts. That leaves the whole-brain readouts, the image baseline and the ROI
specificity table without a site control -- so a scanner effect could be
suppressing the whole-brain nulls, or manufacturing the SN-vs-elsewhere contrast,
and nothing would have caught it.

This runs leave-one-group-out for all of them, over several grouping variables:

  scanner      DeviceSerialNumber  (19 values, clean -- the primary acquisition confound)
  institution  InstitutionName     (28 values, messy spelling variants)
  model        ManufacturersModelName (6 values, coarse)
  (cardiac gating is constant in this cohort -- all gated -- so it is omitted)

For the primary grouping (scanner) each readout also gets a label-permutation null.
The alternative groupings are reported as 5-fold-vs-LOGO deltas only; a large delta
there is the signal to chase with a null.

Usage:
    PYTHONPATH=$(pwd) python -m classification.ppmi_site_cv_all [--n-perm 300]
"""

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

drv_REPO = os.environ.get("BINR_CLASSIFICATION_ROOT",
                          os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, drv_REPO)

drv_OUT = os.path.join(drv_REPO, "OUTPUTS", "classification")

CL = os.path.join(drv_OUT, "classification") if False else drv_OUT

def _sidecar(series) -> dict:
    try:
        return json.load(open(str(series.stem) + ".json"))
    except Exception:
        return {}

def groupings(subs, series):
    """subject-aligned group labels for each confound variable."""
    side = [_sidecar(s) for s in series]
    g = {
        "scanner": [str(d.get("DeviceSerialNumber", "?")) for d in side],
        "institution": [str(d.get("InstitutionName", "?")).strip().lower() for d in side],
        "model": [str(d.get("ManufacturersModelName", "?")) for d in side],
    }
    # Cardiac gating is NOT included: ppmi_get_cohort() is already the single-protocol
    # b=1000 gated cohort, so all 241 subjects are 'gated' and there is nothing to
    # control for. Verified: Counter(series.acq) == {'gated': 241}.
    g = {k: np.array(v) for k, v in g.items()}
    acq = np.array([str(getattr(s, "acq", "?")) for s in series])
    if len(set(acq)) > 1:
        g["acq_type"] = acq
    return g

def _pipe():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"))])

def oof_kfold(X, y, R=10):
    accs = []
    for seed in range(R):
        pred = np.empty(len(y), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            pred[te] = _pipe().fit(X[tr], y[tr]).predict(X[te])
        accs.append(balanced_accuracy_score(y, pred))
    return float(np.mean(accs))

def oof_logo(X, y, groups):
    pred = np.empty(len(y), int)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr][0]
            continue
        pred[te] = _pipe().fit(X[tr], y[tr]).predict(X[te])
    return float(balanced_accuracy_score(y, pred))

def perm_logo(X, y, groups, obs, n=300):
    rng = np.random.default_rng(0)
    ge = sum(1 for _ in range(n) if oof_logo(X, rng.permutation(y), groups) >= obs)
    return (1 + ge) / (1 + n)

def _npz(path, keys):
    z = np.load(path)
    return np.concatenate([z[k].ravel() for k in keys])

def build_blocks(subs, keys):
    """-> {readout_name: (X, subject_index_mask)} from the on-disk caches."""
    blocks = {}

    def add(name, folder, suffix, feat_keys):
        rows, ok = [], []
        for i, k in enumerate(keys):
            p = os.path.join(drv_OUT, folder, f"{k}{suffix}")
            if os.path.exists(p):
                try:
                    rows.append(_npz(p, feat_keys)); ok.append(i)
                except Exception:
                    pass
        if len(ok) > 30:
            blocks[name] = (np.asarray(rows, float), np.array(ok))

    # whole-brain, from the INR
    add("WB decode-phys",        "ppmi_readout", ".npz", ["decode_phys"])
    add("WB sparse out",         "ppmi_readout", ".npz", ["sparse_out"])
    add("WB sparse phys",        "ppmi_readout", ".npz", ["sparse_phys"])
    # SN, image / reconstructed / decoded
    add("SN-MD (image)",         "ppmi_sn_full", ".npz", ["img_md"])
    add("SN-FA (image)",         "ppmi_sn_full", ".npz", ["img_fa"])
    add("SN-MD+FA (image)",      "ppmi_sn_full", ".npz", ["img_md", "img_fa"])
    add("SN-MD (INR recon)",     "ppmi_sn_full", ".npz", ["rec_md"])
    add("SN-FA (INR recon)",     "ppmi_sn_full", ".npz", ["rec_fa"])
    add("SN free-water (decode)", "ppmi_sn_full", ".npz", ["dec_fw"])
    add("SN FW + image MD+FA",   "ppmi_sn_full", ".npz", ["dec_fw", "img_md", "img_fa"])
    # SN sparse sampling at the locked K
    add("SN sparse out (K256)",  "ppmi_sn_sparse", "_K256.npz", ["out"])
    add("SN sparse phys (K256)", "ppmi_sn_sparse", "_K256.npz", ["phys"])
    # ROI specificity table
    for roi in ("SN", "Putamen", "Caudate", "RedNucleus", "STN", "VTA_PBP", "RANDOM"):
        add(f"{roi} sparse out (K128)",  "ppmi_roi_sparse", "_K128.npz", [f"{roi}_out"])
        add(f"{roi} sparse phys (K128)", "ppmi_roi_sparse", "_K128.npz", [f"{roi}_phys"])
    return blocks

def drv_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=300)
    args = ap.parse_args()

    subs, y_all, _ = ppmi_get_cohort()
    series = [baseline_series(s) for s in subs]
    keys = [s.key for s in series]
    y_all = np.asarray(y_all)
    G = groupings(subs, series)
    print(f"cohort n={len(subs)}  PD={int((y_all==1).sum())}  HC={int((y_all==0).sum())}")
    for k, v in G.items():
        both = sum(1 for g in set(v) if len(set(y_all[v == g])) == 2)
        print(f"  grouping {k:12s} {len(set(v)):3d} values, {both} with both classes")

    blocks = build_blocks(subs, keys)
    print(f"\n{len(blocks)} readouts with usable caches\n")
    hdr = (f"{'readout':28s} {'n':>4s} {'5fold':>7s} " +
           " ".join(f"{g[:9]:>9s}" for g in G) + f" {'perm p':>7s}")
    print(hdr); print("-" * len(hdr))

    res = {"cohort_n": len(subs), "n_perm": args.n_perm,
           "groupings": {k: int(len(set(v))) for k, v in G.items()}, "rows": []}
    for name, (X, idx) in blocks.items():
        y = y_all[idx]
        kf = oof_kfold(X, y)
        logos, cells = {}, []
        for gname, gv in G.items():
            g = gv[idx]
            try:
                logos[gname] = oof_logo(X, y, g)
            except Exception:
                logos[gname] = float("nan")
            cells.append(f"{logos[gname]:>9.3f}")
        p = perm_logo(X, y, G["scanner"][idx], logos["scanner"], n=args.n_perm)
        print(f"{name:28s} {len(y):>4d} {kf:>7.3f} " + " ".join(cells) + f" {p:>7.3f}", flush=True)
        res["rows"].append(dict(readout=name, n=int(len(y)), kfold=kf,
                                logo={k: (None if np.isnan(v) else v) for k, v in logos.items()},
                                perm_p_scanner=p))

    path = os.path.join(drv_OUT, "ppmi_site_cv_all.json")
    json.dump(res, open(path, "w"), indent=2)
    print(f"\nsaved -> {path}\nPPMI_SITE_CV_ALL_DONE")




# ==========================================================================
# command line
# ==========================================================================

def cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("ppmi", "hcp"), required=True)
    ap.add_argument("--target", choices=("age", "sex"), default="age",
                    help="HCP only: which label to predict")
    ap.add_argument("--readout", choices=("decoded", "recon-md"), default="decoded",
                    help="HCP only: decoded biophysical parameters, or MD from the "
                         "INR-reconstructed signal")
    ap.add_argument("--n-perm", type=int, default=300,
                    help="label permutations for the null")
    ap.add_argument("--train-inrs", action="store_true",
                    help="HCP only: train the per-subject INRs first (slow)")
    args = ap.parse_args()

    if args.dataset == "ppmi":
        # leave-one-group-out over every PPMI readout, with a scanner-permutation
        # null -- this is the table in the paper's classification section.
        import sys as _s
        _s.argv = ["classify.py", "--n-perm", str(args.n_perm)]
        return drv_main() or 0

    if args.train_inrs:
        hcp_train_inrs()
    subjects = hcp_list_subjects()
    labels = cohort_map()
    subjects = [s for s in subjects if str(s) in labels]
    y = np.asarray([labels[str(s)] for s in subjects])
    X = (physics_features(subjects) if args.readout == "decoded"
         else recon_md_features(subjects))
    print(f"HCP {args.target}: {len(subjects)} subjects, {X.shape[1]} features, "
          f"{len(set(y.tolist()))} classes", flush=True)
    res = evaluate_features(X, y, n_perm=args.n_perm)
    print(res)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(cli())
