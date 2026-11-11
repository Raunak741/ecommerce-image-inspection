"""
Step 3: Baseline (hand-tuned rules on DIP features)  vs
        Improved-1 (Random Forest on the same 10 DIP features)  vs
        Improved-2 (small CNN on raw 96x96 pixels)

Split is BY PRODUCT (all 6 variants of a product go to the same split) so
the models never see a test product during training.
Outputs: results/metrics.json, results/*.png, results/features.csv
"""
import os, json, time, csv
import numpy as np, cv2, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                             classification_report)
import torch, torch.nn as nn
from dip_features import extract_all, FEATURE_NAMES, intermediates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
CLASSES = ["clean", "blur", "noise", "low_contrast", "bg_clutter", "off_center"]
torch.manual_seed(0); np.random.seed(0)

# ---------------------------------------------------------------- data
df = pd.read_csv(os.path.join(DATA, "labels.csv"))
imgs = [cv2.imread(os.path.join(DATA, p)) for p in df.path]
t0 = time.time()
X = np.stack([extract_all(im) for im in imgs])
feat_ms = (time.time() - t0) / len(imgs) * 1000
y = df.label_id.values; groups = df.product_id.values
pd.DataFrame(X, columns=FEATURE_NAMES).assign(label=df.label).to_csv(
    os.path.join(RES, "features.csv"), index=False)

tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(X, y, groups))
print(f"train {len(tr)}  test {len(te)}  feature time {feat_ms:.2f} ms/img")

# ---------------------------------------------------------------- baseline: rules
# thresholds set by eye from the TRAIN-split feature distributions (see per-class means)
F = {n: i for i, n in enumerate(FEATURE_NAMES)}
def rule_predict(x):
    if x[F["noise_mad"]] > 3.0:                               return 2   # noise
    if x[F["hist_spread"]] < 100:                             return 3   # low contrast
    if x[F["border_white"]] < 0.3:                            return 4   # cluttered bg
    if x[F["bbox_area"]] < 0.12 and x[F["centroid_off"]] > 0.25: return 5 # off-centre
    if x[F["sharpness"]] < 30:                                return 1   # blur
    return 0                                                           # clean

# print per-class feature means on train split (for the paper table)
means = pd.DataFrame(X[tr], columns=FEATURE_NAMES).groupby(df.label.values[tr]).mean().loc[CLASSES]
means.round(3).to_csv(os.path.join(RES, "feature_means_by_class.csv"))
print(means.round(2))

yp_rule = np.array([rule_predict(x) for x in X[te]])

# ---------------------------------------------------------------- improved 1: RF on DIP features
rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
t0 = time.time(); rf.fit(X[tr], y[tr]); rf_train_s = time.time() - t0
yp_rf = rf.predict(X[te])
imp = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
imp.to_csv(os.path.join(RES, "rf_feature_importance.csv"))

# ---------------------------------------------------------------- improved 2: small CNN on pixels
def to_tensor(ims):
    a = np.stack([cv2.resize(im, (96, 96), interpolation=cv2.INTER_AREA) for im in ims])
    a = a[..., ::-1].astype(np.float32) / 255.0                # BGR->RGB
    return torch.tensor(a.transpose(0, 3, 1, 2).copy())
Xt_tr, Xt_te = to_tensor([imgs[i] for i in tr]), to_tensor([imgs[i] for i in te])
yt_tr = torch.tensor(y[tr])

class SmallCNN(nn.Module):
    def __init__(s, n=6):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 64), blk(64, 64))   # 96->6x6
        s.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 6 * 6, 128), nn.ReLU(), nn.Linear(128, n))
    def forward(s, x): return s.h(s.f(x))

cnn = SmallCNN(); opt = torch.optim.Adam(cnn.parameters(), 1e-3); lossf = nn.CrossEntropyLoss()
EPOCHS, BS = 20, 64
t0 = time.time(); hist = []
for ep in range(EPOCHS):
    cnn.train(); perm = torch.randperm(len(Xt_tr)); tot = 0
    for i in range(0, len(perm), BS):
        idx = perm[i:i+BS]
        xb = Xt_tr[idx]
        if torch.rand(1).item() < 0.5: xb = xb.flip(3)                      # horizontal-flip augmentation
        opt.zero_grad(); l = lossf(cnn(xb), yt_tr[idx]); l.backward(); opt.step()
        tot += l.item() * len(idx)
    hist.append(tot / len(perm)); print(f"epoch {ep+1} loss {hist[-1]:.3f}")
cnn_train_s = time.time() - t0
cnn.eval()
with torch.no_grad():
    yp_cnn = cnn(Xt_te).argmax(1).numpy()
    cnn_train_acc = float((cnn(Xt_tr).argmax(1).numpy() == y[tr]).mean())
print("CNN train acc", cnn_train_acc)

# ---------------------------------------------------------------- metrics
def summarise(name, yp):
    return {"model": name,
            "accuracy": round(float(accuracy_score(y[te], yp)), 4),
            "macro_f1": round(float(f1_score(y[te], yp, average="macro")), 4),
            "per_class_f1": dict(zip(CLASSES, np.round(f1_score(y[te], yp, average=None), 3).tolist())),
            # binary PASS/FAIL view: clean vs any defect
            "defect_recall": round(float(((yp != 0) & (y[te] != 0)).sum() / (y[te] != 0).sum()), 4),
            "false_reject_rate": round(float(((yp != 0) & (y[te] == 0)).sum() / (y[te] == 0).sum()), 4)}
results = [summarise("Rule baseline (DIP features)", yp_rule),
           summarise("Random Forest (DIP features)", yp_rf),
           summarise("Small CNN (raw pixels)", yp_cnn)]
meta = {"n_train": int(len(tr)), "n_test": int(len(te)), "feature_ms_per_img": round(feat_ms, 2),
        "rf_train_s": round(rf_train_s, 2), "cnn_train_s": round(cnn_train_s, 1), "cnn_epochs": EPOCHS, "cnn_train_acc": round(cnn_train_acc, 4),
        "rf_importance": imp.round(4).to_dict()}
json.dump({"results": results, "meta": meta}, open(os.path.join(RES, "metrics.json"), "w"), indent=2)
for r in results: print(r["model"], "acc", r["accuracy"], "macroF1", r["macro_f1"])
print(classification_report(y[te], yp_rf, target_names=CLASSES))

# ---------------------------------------------------------------- figures
def plot_cm(yp, title, fn):
    cm = confusion_matrix(y[te], yp)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues"); ax.set_title(title, fontsize=10)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(CLASSES, fontsize=8)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout(); fig.savefig(os.path.join(RES, fn), dpi=200); plt.close(fig)
plot_cm(yp_rule, "Rule baseline", "cm_rule.png")
plot_cm(yp_rf, "Random Forest", "cm_rf.png")
plot_cm(yp_cnn, "Small CNN", "cm_cnn.png")

# feature importance bar
fig, ax = plt.subplots(figsize=(4.2, 2.6)); imp[::-1].plot.barh(ax=ax, color="#3b6ea5")
ax.set_xlabel("RF importance"); fig.tight_layout(); fig.savefig(os.path.join(RES, "rf_importance.png"), dpi=200); plt.close(fig)

# intermediate DIP stages for one product (all 6 variants)
pid = df.product_id.iloc[te[0]]
fig, axes = plt.subplots(6, 6, figsize=(9, 9))
cols = ["input", "grey", "Laplacian", "Canny edges", "FG mask (Otsu+morph)", "bbox"]
for r, c in enumerate(CLASSES):
    im = cv2.imread(os.path.join(DATA, "images", c, f"{pid}.jpg")); it = intermediates(im)
    panels = [im[..., ::-1], it["grey"], it["laplacian"], it["edges"], it["mask"], it["bbox"][..., ::-1]]
    for k, p in enumerate(panels):
        axes[r, k].imshow(p, cmap=None if p.ndim == 3 else "gray", vmin=None if p.ndim == 3 else 0, vmax=None if p.ndim == 3 else 255); axes[r, k].axis("off")
        if r == 0: axes[r, k].set_title(cols[k], fontsize=8)
    axes[r, 0].text(-0.1, 0.5, c, transform=axes[r, 0].transAxes, rotation=90, va="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RES, "intermediate_stages.png"), dpi=170); plt.close(fig)

# histograms: clean vs low-contrast vs noise
fig, ax = plt.subplots(figsize=(4.2, 2.4))
for c, col in [("clean", "k"), ("low_contrast", "tab:orange"), ("noise", "tab:green")]:
    it = intermediates(cv2.imread(os.path.join(DATA, "images", c, f"{pid}.jpg")))
    ax.plot(it["hist"] / it["hist"].sum(), col, label=c, lw=1)
ax.set_yscale("log"); ax.set_xlabel("grey level"); ax.set_ylabel("p (log)"); ax.legend(fontsize=7); fig.tight_layout()
fig.savefig(os.path.join(RES, "histograms.png"), dpi=200); plt.close(fig)

# feature distributions per class (box plots for 4 key features)
fig, axes = plt.subplots(1, 4, figsize=(9, 2.3))
for ax, f in zip(axes, ["sharpness", "noise_mad", "hist_spread", "centroid_off"]):
    ax.boxplot([X[y == i, F[f]] for i in range(6)], showfliers=False)
    ax.set_xticks(range(1, 7)); ax.set_xticklabels(CLASSES, rotation=60, fontsize=6); ax.set_title(f, fontsize=8)
    if f == "sharpness": ax.set_yscale("log")
fig.tight_layout(); fig.savefig(os.path.join(RES, "feature_boxplots.png"), dpi=200); plt.close(fig)

# failure cases: RF mistakes
wrong = np.where(yp_rf != y[te])[0][:6]
if len(wrong):
    fig, axes = plt.subplots(1, len(wrong), figsize=(1.6 * len(wrong), 2.0))
    axes = np.atleast_1d(axes)
    for ax, w in zip(axes, wrong):
        ax.imshow(imgs[te[w]][..., ::-1]); ax.axis("off")
        ax.set_title(f"T:{CLASSES[y[te][w]]}\nP:{CLASSES[yp_rf[w]]}", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "failure_cases_rf.png"), dpi=200); plt.close(fig)
wrong = np.where(yp_rule != y[te])[0]
sel = wrong[np.linspace(0, len(wrong) - 1, min(6, len(wrong))).astype(int)] if len(wrong) else []
if len(sel):
    fig, axes = plt.subplots(1, len(sel), figsize=(1.6 * len(sel), 2.0)); axes = np.atleast_1d(axes)
    for ax, w in zip(axes, sel):
        ax.imshow(imgs[te[w]][..., ::-1]); ax.axis("off")
        ax.set_title(f"T:{CLASSES[y[te][w]]}\nP:{CLASSES[yp_rule[w]]}", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "failure_cases_rule.png"), dpi=200); plt.close(fig)

# sample dataset strip
fig, axes = plt.subplots(2, 6, figsize=(9, 3.2))
for r, p in enumerate(df.product_id.unique()[:2]):
    for k, c in enumerate(CLASSES):
        axes[r, k].imshow(cv2.imread(os.path.join(DATA, "images", c, f"{p}.jpg"))[..., ::-1]); axes[r, k].axis("off")
        if r == 0: axes[r, k].set_title(c, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RES, "dataset_samples.png"), dpi=170); plt.close(fig)
torch.save(cnn.state_dict(), os.path.join(RES, "small_cnn.pt"))
print("done")
