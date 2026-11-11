"""Step 4: ablations (feature groups, morphology on/off) + pipeline block diagram."""
import os, json, numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score
import dip_features as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); DATA = os.path.join(ROOT, "data")
df = pd.read_csv(os.path.join(DATA, "labels.csv"))
X = pd.read_csv(os.path.join(RES, "features.csv"))[D.FEATURE_NAMES].values
y = df.label_id.values; g = df.product_id.values
tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, y, g))
F = {n: i for i, n in enumerate(D.FEATURE_NAMES)}

groups = {"sharpness only": ["sharpness"],
          "global stats (sharp+noise+contrast)": ["sharpness", "noise_mad", "sp_frac", "contrast_std", "hist_spread"],
          "+ background (border/bg_std)": ["sharpness", "noise_mad", "sp_frac", "contrast_std", "hist_spread", "border_white", "bg_std"],
          "all 10 (+ region features)": D.FEATURE_NAMES}
rows = []
for name, feats in groups.items():
    idx = [F[f] for f in feats]
    rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X[tr][:, idx], y[tr])
    rows.append((name, len(feats), round(f1_score(y[te], rf.predict(X[te][:, idx]), average="macro"), 4)))

# morphology ablation: recompute region features with a raw Otsu mask (no open/close, no largest-CC)
def raw_mask(gr):
    _, m = cv2.threshold(gr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU); return m
Xr = X.copy()
for i, p in enumerate(df.path):
    gr = D.to_grey(cv2.imread(os.path.join(DATA, p))); m = raw_mask(gr)
    a, c, f = D.region_feats(m); Xr[i, F["bbox_area"]], Xr[i, F["centroid_off"]], Xr[i, F["fg_fill"]] = a, c, f
    Xr[i, F["bg_std"]] = D.bg_std(gr, m)
rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(Xr[tr], y[tr])
rows.append(("all 10, mask WITHOUT morphology/largest-CC", 10, round(f1_score(y[te], rf.predict(Xr[te]), average="macro"), 4)))
abl = pd.DataFrame(rows, columns=["feature set", "n_feat", "macro_F1"]); abl.to_csv(os.path.join(RES, "ablation.csv"), index=False)
print(abl)

# ---------------- block diagram
fig, ax = plt.subplots(figsize=(9, 2.6)); ax.axis("off")
boxes = [("Product image\n(256x256 BGR)", "#dfe8f5"), ("Pre-processing\ngrey, pad/resize", "#dfe8f5"),
         ("DIP measures\nLaplacian var, MAD noise,\nhistogram spread, border", "#fbe5c8"),
         ("Segmentation\nOtsu + open/close +\nlargest CC, bbox", "#fbe5c8"),
         ("Classifier\nrules | RandomForest\n(vs CNN on pixels)", "#dcefd8"),
         ("Output\nPASS / FAIL + defect\ntype, metrics, failure cases", "#efe0f3")]
x = 0.01; w = 0.155
for i, (t, c) in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((x, 0.2), w, 0.6, boxstyle="round,pad=0.01", fc=c, ec="k", lw=0.8))
    ax.text(x + w/2, 0.5, t, ha="center", va="center", fontsize=7.2)
    if i < len(boxes) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.005, 0.5), (x + w + 0.035, 0.5), arrowstyle="-|>", mutation_scale=10, color="k"))
    x += w + 0.04
ax.text(0.5, 0.05, "10-D feature vector  →  rule thresholds (baseline)  /  Random Forest (improved)   |   raw pixels  →  small CNN (learned alternative)",
        ha="center", fontsize=7, style="italic")
fig.savefig(os.path.join(RES, "block_diagram.png"), dpi=220, bbox_inches="tight"); plt.close(fig)
print("diagram ok")
