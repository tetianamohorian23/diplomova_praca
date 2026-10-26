"""Strong classical baseline: openSMILE eGeMAPSv02 functionals + LR / RBF-SVM / RandomForest.
Hyper-parameters chosen on validation macro-F1; test scored once for the chosen setting."""
import json
import numpy as np
import opensmile
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common_splits import CACHE, LABELS, load_splits

V2 = CACHE.parent
data = load_splits()
waves = np.load(V2 / "cnn_ssl" / "waves_int16.npy", mmap_mode="r")
lens = np.load(V2 / "cnn_ssl" / "lengths.npy")
cache = V2 / "egemaps.npy"


def feats(idx):
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    return np.vstack([smile.process_signal(waves[i, : lens[i]].astype(np.float32) / 32767, 16000).values for i in idx])


if not cache.exists():
    chunks = np.array_split(np.arange(len(data)), 40)
    X = np.vstack(Parallel(n_jobs=4, verbose=5)(delayed(feats)(c) for c in chunks))
    np.save(cache, X)
X = np.load(cache)
sp, y = data.experiment_split.to_numpy(), data.label_id.to_numpy()
tr, va, te = (np.where(sp == s)[0] for s in ("train", "validation", "test"))
grid = {
    "logreg": [(f"C={c}", make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))) for c in (0.01, 0.1, 1, 10)],
    "svm_rbf": [(f"C={c}", make_pipeline(StandardScaler(), SVC(C=c, class_weight="balanced"))) for c in (0.3, 1, 3, 10)],
    "random_forest": [(f"n={n}", RandomForestClassifier(n_estimators=n, class_weight="balanced", n_jobs=4, random_state=0)) for n in (300, 600)],
}
res = {}
for name, cands in grid.items():
    best = max(((f1_score(y[va], m.fit(X[tr], y[tr]).predict(X[va]), average="macro"), tag, m) for tag, m in cands), key=lambda t: t[0])
    pred = best[2].predict(X[te])
    res[name] = {"chosen": best[1], "val_macro_f1": float(best[0]), "test_macro_f1": float(f1_score(y[te], pred, average="macro")),
                 "test_accuracy": float(accuracy_score(y[te], pred)),
                 "per_class_f1": dict(zip(LABELS, f1_score(y[te], pred, average=None).round(3).tolist()))}
    print(name, res[name], flush=True)
(V2 / "opensmile_baseline.json").write_text(json.dumps(res, indent=1))
