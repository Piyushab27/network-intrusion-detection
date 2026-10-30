# Network Intrusion Detection ML Challenge

Classify network flows into **Benign**, **Availability**, **Reconnaissance** and **Spoofing**.
The metric is Macro F1. Final selected submission: **0.84614** public.

## Results

| Submission | Method | Public Macro F1 |
|---|---|---|
| `submission_majority_vote.csv` | Majority vote of the three files below | **0.84614** |
| `submission_pseudo_1round.csv` | LightGBM + one round of confident pseudo-labelling | 0.84592 |
| `submission_random_cv.csv` | Tuned LightGBM, random 5-fold CV | 0.84585 |
| `submission_top2_blend.csv` | Probability blend of the two models above | not uploaded |

Random 5-fold CV for these models is about 0.99 Macro F1. The gap to the public score comes from distribution shift (see below).

## Pipeline

All models are LightGBM (800 trees, balanced class weights) on the 82 provided features plus a few ratio features.
Validation is stratified random 5-fold CV (seed 42).

```bash
pip install -r requirements.txt
DATA=data   # put train.csv, test.csv, sample_submission.csv here

# 1. Tuned LightGBM -> submission_random_cv.csv
python src/train_intrusion_model.py --data-dir $DATA --cv random --out submission_random_cv.csv

# 2. One-round pseudo-labelling -> submission_pseudo_1round.csv
python src/pseudo_label.py --data-dir $DATA --rounds 1 --out submission_pseudo_1round.csv

# 3. Out-of-fold predictions for the blend (cache folder of your choice)
python src/feature_shift.py --data-dir $DATA --cache-dir cache
python src/evaluate_candidates.py --data-dir $DATA --cache-dir cache --dropped "[]" --configs original_800
python src/pseudo_recipe_cv.py --data-dir $DATA --cache-dir cache
python src/top2_blend.py --data-dir $DATA --cache-dir cache --out submission_top2_blend.csv

# 4. Final majority vote
python src/majority_vote.py submission_pseudo_1round.csv submission_random_cv.csv \
    submission_top2_blend.csv --out submission_majority_vote.csv
```

Class multipliers are tuned on out-of-fold predictions only. Pseudo-labels use only test rows the model predicts with at least 90% confidence, and come from the model's own predictions, never from hidden labels.

## Key findings

- **Severe class imbalance:** Availability is 94.5% of training rows, so the three rare classes decide Macro F1.
- **Train and test come from different traffic regimes:** a train-vs-test classifier separates them perfectly (AUC 1.0). 95% of training flows last under 1 ms, against 63% of test flows; flows of 1 s or more are 2.8% of train and 15% of test.
- **Reconnaissance is almost absent from longer flows in training:** only 7 of 735 Reconnaissance rows last 1 ms or more, so no model can learn that region reliably. This is the likely ceiling around 0.846.
- **Things that did not help** (random CV and/or public score): pruning shifted features, rebuilt rate features, importance weighting, duration-specific expert models, multi-round pseudo-labelling (drifted), large ensembles.

## Repository layout

```
src/          training, evaluation and analysis scripts
results/      CV summaries and the per-feature train/test shift audit
submissions/  final and candidate submissions
data/         competition CSVs (not committed)
```

## Data

Dataset by B. S. and Rohini Nagapadma, The National Institute of Engineering, Mysuru.
Licensed under CC BY 4.0. The data is not included in this repository.
