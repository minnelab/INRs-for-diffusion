# Analytic Ball-and-fODF Implicit Neural Representations for Diffusion MRI Reconstruction and Classification

Code for the paper by Sanna Persson, Simone Bendazzoli, Fabian Leander Sinzinger,
Christoffer Olsson and Rodrigo Moreno (KTH Royal Institute of Technology).

A coordinate network maps each voxel position to the parameters of a two-compartment
ball-and-fODF forward model. Because the angular integral is solved in closed form via
Funk–Hecke, the signal can be evaluated at **any** b-value and gradient direction without a
per-shell response function — so the trained network predicts unacquired shells, and its
decoded parameters double as features for downstream classification.

## Install

```bash
git clone https://github.com/minnelab/INRs-for-diffusion.git
cd INRs-for-diffusion
pip install -r requirements.txt
pip install -e .
```

Python ≥ 3.10. Pinned versions in `requirements.txt` are those used for the paper's results.
A CUDA GPU is needed for training (~15 min per subject at the paper's settings).

## Layout

```
train.py      train an INR on one subject and score what was held out
evaluate.py   scorers and non-learned baselines
classify.py   classify subjects from their trained INRs
inrdiff/      the library
  ball_stick.py   forward model: Funk–Hecke angular integration, Gauss–Legendre
                  quadrature, b=0 normalisation
  model.py        Lightning module and training loss
  benchmark.py    the (dataset, task) pipeline behind train.py
  networks/       INCODE backbone, harmoniser, positional encoders
  main.py, dataset.py, val_dataset.py, evaluation.py, dti_eval.py, fodf_frame.py,
  acquisition_adaptation.py, scoring.py
configs/      ball_fodf_hcp.yml — the architecture used throughout
```

## Training and reconstruction

Every reconstruction experiment is a (dataset, task) pair:

```bash
export BINR_HCP_ROOT=/path/to/HCP
python train.py --list                                     # datasets and tasks

python train.py --task shell-holdout    --subject 107422   # hold out b=2000
python train.py --task single-shell     --subject 107422   # train b=1000, predict 2000 + 3000
python train.py --task direction-budget --directions 10    # withhold directions, not shells
python train.py --task superres --factor 2                 # train on a 1/8 voxel lattice
python train.py --task dense                               # all shells and directions

python train.py --dataset parkmr --task parkmr-holdout --subject PD20230223Control1
```

Results land in `OUTPUTS/<dataset>_<task>_<subject>_<stamp>/` as `metrics_summary.json`
plus per-shell CSVs.

## Baselines and scorers

```bash
python evaluate.py upsample   --subject 107422 --factors 2 4  # classical tri-cubic
python evaluate.py dti-extrap --subject 107422                # mono-exponential DTI
python evaluate.py frame  --pred <fodf.nii.gz> --subject 107422
python evaluate.py fodf   --pred <fodf.nii.gz> --gt <wm_fod.nii.gz> \
                          --label ours --out acc.json --signs=-1,1,1
```

`fodf` needs [scilpy](https://github.com/scilus/scilpy), whose VTK/numpy pins conflict with
this environment — install it separately and run that subcommand there.

## Classification

The network is never retrained here; each subject's INR is read from disk and a feature
vector taken from it.

```bash
python classify.py --dataset ppmi                           # leave-one-scanner-out, all readouts
python classify.py --dataset hcp --target age --readout decoded
```

## Data

None of the datasets are redistributable here.

| Dataset | Source | Used for |
|---|---|---|
| HCP Young Adult | [humanconnectome.org](https://www.humanconnectome.org) | reconstruction, fODF, age/sex classification |
| ParkMR | in-house, 29 subjects | reconstruction, shell holdout |
| PPMI | [ppmi-info.org](https://www.ppmi-info.org) (registration required) | Parkinson's classification |

HCP is expected as `<root>/<subject>/Diffusion/{data.nii.gz,bvals,bvecs,nodif_brain_mask.nii.gz}`.
Paths are read from the environment, never hardcoded:

| variable | what it points at |
|---|---|
| `BINR_HCP_ROOT` | the HCP cohort root |
| `BINR_PARKMR_ROOT` | the ParkMR cohort root |
| `BINR_PPMI_LABELS` | PPMI label CSV (IDA manifest, or LONI `Participant_Status.csv`) |
| `BINR_CLASSIFICATION_ROOT` | where `OUTPUTS/classification/` caches and trained INRs live |
| `BINR_HCP_PREPROCESSED` | preprocessed HCP volumes, for the classification readouts |

Ground-truth fODFs come from MRtrix `dwi2response dhollander` + `dwi2fod msmt_csd`; MRtrix is
not a Python dependency and is only needed to regenerate them.
