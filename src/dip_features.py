"""
Step 2: Classical DIP feature extraction.

Every function takes a BGR image and returns a scalar quality measure.
`extract_all()` returns the 10-D feature vector used by the rule-based
baseline and the Random-Forest classifier. `intermediates()` returns the
intermediate images (grey, Laplacian, edge map, background mask, bbox) for
the report figures.
"""
import numpy as np, cv2

def to_grey(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# ---- 1. Sharpness: variance of Laplacian (second-derivative energy) ----
def sharpness(g):
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

# ---- 2. Noise: MAD of high-pass residual (image - median filtered) ----
def noise_level(g):
    res = g.astype(np.float32) - cv2.medianBlur(g, 3).astype(np.float32)
    return float(1.4826 * np.median(np.abs(res - np.median(res))))

# ---- 3. Salt & pepper fraction: pixels at 0 or 255 that differ from median ----
def sp_fraction(g):
    med = cv2.medianBlur(g, 3)
    hit = ((g == 0) | (g == 255)) & (np.abs(g.astype(int) - med.astype(int)) > 100)
    return float(hit.mean())

# ---- 4/5. Contrast: std-dev of grey + 1-99 percentile histogram spread ----
def contrast_std(g):
    return float(g.std())

def hist_spread(g):
    h = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
    cdf = np.cumsum(h) / h.sum()
    lo = np.searchsorted(cdf, 0.01); hi = np.searchsorted(cdf, 0.99)
    return float(hi - lo)

# ---- 6. Foreground mask via Otsu + morphology; border-white fraction ----
def fg_mask(g):
    # product is darker than white studio bg -> invert threshold
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    # keep largest connected component
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.where(lab == big, 255, 0).astype(np.uint8)
    return m

def border_white_frac(g, t=235):
    b = np.concatenate([g[:8].ravel(), g[-8:].ravel(), g[:, :8].ravel(), g[:, -8:].ravel()])
    return float((b > t).mean())

# ---- 7. Background uniformity: std of grey outside the foreground mask ----
def bg_std(g, m):
    bg = g[m == 0]
    return float(bg.std()) if bg.size > 100 else 0.0

# ---- 8/9/10. Region features from bounding box of the foreground ----
def region_feats(m):
    h, w = m.shape
    ys, xs = np.where(m > 0)
    if len(xs) == 0: return 0.0, 0.5, 0.0
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    area_frac = (x1 - x0 + 1) * (y1 - y0 + 1) / (h * w)
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    centroid_off = float(np.hypot(cx - 0.5, cy - 0.5))          # 0 = centred
    fill = float((m > 0).mean())
    return float(area_frac), centroid_off, fill

FEATURE_NAMES = ["sharpness", "noise_mad", "sp_frac", "contrast_std", "hist_spread",
                 "border_white", "bg_std", "bbox_area", "centroid_off", "fg_fill"]

def extract_all(im):
    g = to_grey(im)
    m = fg_mask(g)
    a, c, f = region_feats(m)
    return np.array([sharpness(g), noise_level(g), sp_fraction(g), contrast_std(g),
                     hist_spread(g), border_white_frac(g), bg_std(g, m), a, c, f], np.float32)

def intermediates(im):
    g = to_grey(im)
    lap = cv2.convertScaleAbs(cv2.Laplacian(g, cv2.CV_64F))
    edges = cv2.Canny(g, 50, 150)
    m = fg_mask(g)
    box = im.copy()
    ys, xs = np.where(m > 0)
    if len(xs):
        cv2.rectangle(box, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 0, 255), 2)
    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
    return {"grey": g, "laplacian": lap, "edges": edges, "mask": m, "bbox": box, "hist": hist}
