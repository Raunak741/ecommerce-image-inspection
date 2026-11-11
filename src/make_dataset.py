"""
Step 1: Build the labelled dataset.

Takes clean e-commerce product images (Fashion Product Images - Small, HF mirror)
and synthetically injects 5 marketplace-style defects with known labels:

    0 clean          - as is
    1 blur           - Gaussian blur (out of focus)
    2 noise          - additive Gaussian + salt&pepper noise (low-light sensor)
    3 low_contrast   - histogram compression (washed-out)
    4 bg_clutter     - non-uniform / textured background (fails white-bg rule)
    5 off_center     - product shifted & shrunk (fails framing rule)

Output: data/images/<class>/<id>.jpg  +  data/labels.csv
"""
import io, os, sys, csv, random
import numpy as np, cv2
import pyarrow.parquet as pq
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data")
CLASSES = ["clean", "blur", "noise", "low_contrast", "bg_clutter", "off_center"]
N_BASE = 300          # base clean images
SIZE = 256            # working resolution
random.seed(0); np.random.seed(0)

def load_base_images(n):
    cache = os.path.join(ROOT, "data_raw", "shard0.parquet")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if not os.path.exists(cache):
        u = "https://huggingface.co/api/datasets/ashraq/fashion-product-images-small/parquet/default/train/0.parquet"
        print("downloading", u)
        urllib.request.urlretrieve(u, cache)
    t = pq.read_table(cache, columns=["id", "image", "masterCategory", "articleType"]).to_pandas()
    t = t.sample(frac=1, random_state=0).reset_index(drop=True)
    imgs = []
    for _, r in t.iterrows():
        b = r["image"]["bytes"]
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if im is None: continue
        # keep only images whose border is near-white (true studio shots)
        border = np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]])
        if border.mean() < 235: continue
        im = pad_to_square(im)
        im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        imgs.append((int(r["id"]), im))
        if len(imgs) >= n: break
    return imgs

def pad_to_square(im):
    h, w = im.shape[:2]; s = max(h, w)
    out = np.full((s, s, 3), 255, np.uint8)
    y, x = (s - h)//2, (s - w)//2
    out[y:y+h, x:x+w] = im
    return out

# ---------------- defect generators ----------------
def blur(im):
    k = random.choice([7, 9, 11, 13])
    return cv2.GaussianBlur(im, (k, k), 0)

def noise(im):
    sigma = random.uniform(15, 35)
    out = im.astype(np.float32) + np.random.normal(0, sigma, im.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    p = random.uniform(0.005, 0.02)                      # salt & pepper
    m = np.random.rand(*im.shape[:2])
    out[m < p/2] = 0; out[m > 1 - p/2] = 255
    return out

def low_contrast(im):
    a = random.uniform(0.25, 0.45)                        # contrast gain
    b = random.uniform(100, 140)                          # grey offset
    return np.clip(a * im.astype(np.float32) + b, 0, 255).astype(np.uint8)

def bg_clutter(im):
    """Replace white background with a textured / gradient background."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(g, 240, 255, cv2.THRESH_BINARY_INV)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    kind = random.choice(["noise_tex", "gradient", "stripes"])
    h, w = g.shape
    if kind == "noise_tex":
        bg = cv2.GaussianBlur(np.random.randint(60, 200, (h, w, 3)).astype(np.uint8), (0, 0), 3)
    elif kind == "gradient":
        c1 = np.random.randint(40, 220, 3); c2 = np.random.randint(40, 220, 3)
        t = np.linspace(0, 1, w)[None, :, None]
        bg = (c1 * (1 - t) + c2 * t).astype(np.uint8).repeat(h, 0)
    else:
        bg = np.zeros((h, w, 3), np.uint8)
        c1 = np.random.randint(60, 220, 3); c2 = np.random.randint(60, 220, 3)
        for x in range(0, w, 32): bg[:, x:x+16] = c1; bg[:, x+16:x+32] = c2
    mask = (fg > 0)[..., None]
    return np.where(mask, im, bg).astype(np.uint8)

def off_center(im):
    s = random.uniform(0.35, 0.55)
    small = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    out = np.full_like(im, 255)
    corner = random.choice([(0, 0), (0, SIZE - w), (SIZE - h, 0), (SIZE - h, SIZE - w)])
    out[corner[0]:corner[0]+h, corner[1]:corner[1]+w] = small
    return out

GEN = {"clean": lambda x: x, "blur": blur, "noise": noise,
       "low_contrast": low_contrast, "bg_clutter": bg_clutter, "off_center": off_center}

if __name__ == "__main__":
    base = load_base_images(N_BASE)
    print("base images:", len(base))
    rows = []
    for c in CLASSES: os.makedirs(os.path.join(OUT, "images", c), exist_ok=True)
    for pid, im in base:
        for ci, c in enumerate(CLASSES):
            out = GEN[c](im.copy())
            fn = os.path.join("images", c, f"{pid}.jpg")
            cv2.imwrite(os.path.join(OUT, fn), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append((fn, c, ci, pid))
    with open(os.path.join(OUT, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "label", "label_id", "product_id"]); w.writerows(rows)
    print("wrote", len(rows), "images ->", OUT)
