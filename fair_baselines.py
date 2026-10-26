"""Fair comparison for the semi-supervised SVM: same labeled subsets, with and without pseudo-labels.

For labeled fractions {0.2, 0.5, 1.0} and 5 subset seeds, trains
  * supervised LinearSVC on the labeled subset only,
  * self-training LinearSVC (same recipe as train_models.train_semi_supervised_svm).
Writes experiments/v2/fair_baselines.json (+ prints a table).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import train_models as tm
from common_splits import CACHE

data = pd.read_csv(tm.FEATURES)
data = data[data["label"].isin(tm.models_labels())].copy()
data = tm.restrict_to_common_audio(data)
data = tm.add_validation_split(data)
skip = {"filename", "dataset", "speaker_id", "label", "original_label", "split", "experiment_split"}
cols = [c for c in data.columns if c not in skip]
train, test = data[data.experiment_split == "train"], data[data.experiment_split == "test"]


def make():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LinearSVC(class_weight="balanced", C=0.1, dual="auto", max_iter=5000, random_state=42))


def fit_eval(labeled):
    bal = tm.oversample_training(labeled)
    m = make().fit(bal[cols], bal["label"])
    return m, f1_score(test["label"], m.predict(test[cols]), average="macro")


rows = []
for frac in (0.2, 0.5, 1.0):
    for seed in range(5):
        labeled = train.sample(frac=frac, random_state=seed) if frac < 1 else train
        _, sup = fit_eval(labeled)
        semi = sup
        if frac < 1:
            unlabeled = train.drop(index=labeled.index)
            for _ in range(5):
                bal = tm.oversample_training(labeled)
                m = make().fit(bal[cols], bal["label"])
                s = m.decision_function(unlabeled[cols])
                top = np.partition(s, -2, axis=1)[:, -2:]
                keep = (top[:, 1] - top[:, 0]) >= 0.8
                if not keep.any():
                    break
                ps = unlabeled.loc[keep].copy()
                ps["label"] = m.classes_[s[keep].argmax(1)]
                labeled = pd.concat([labeled, ps]); unlabeled = unlabeled.drop(index=ps.index)
            _, semi = fit_eval(labeled)
        rows.append({"labeled_fraction": frac, "seed": seed, "supervised_f1": sup, "self_training_f1": semi})
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
(CACHE.parent / "fair_baselines.json").write_text(df.to_json(orient="records"))
print(df.groupby("labeled_fraction")[["supervised_f1", "self_training_f1"]].agg(["mean", "std"]).round(3))
