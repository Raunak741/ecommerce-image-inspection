# HSR-AIML-056 — Automated E-commerce Image Quality & Compliance Inspector

DIP course project. Flags seller product photos that are **blurred, noisy, low-contrast,
on a cluttered background, or off-centre**, and names the defect.

## Pipeline

```
product image → grey/resize → DIP measures (Laplacian var, MAD noise, histogram spread, border white)
             → Otsu + morphology + largest-CC mask → bbox / centroid / fill
             → 10-D feature vector → rules (baseline) | Random Forest (improved)
             (vs. small CNN on raw pixels as learned alternative)
```

## Results (540 test images, 90/class, product-disjoint split)

| Model                       | Acc   | Macro-F1 | Defect recall | False-reject |
|-----------------------------|-------|----------|---------------|--------------|
| Rule baseline (DIP feats)   | 0.913 | 0.915    | 0.991         | 0.111        |
| **Random Forest (DIP feats)** | **0.983** | **0.983** | **0.993** | **0.033**  |
| Small CNN (raw pixels)      | 0.926 | 0.921    | 0.960         | 0.033        |

Feature extraction: 2.4 ms/image on CPU.

## Reproduce

```bash
pip install opencv-python scikit-learn pandas pyarrow matplotlib torch
python src/make_dataset.py          # downloads 300 studio product images, injects 5 defects → data/ (1800 imgs)
python src/train_eval.py            # features, rules, RF, CNN, all figures → results/
python src/ablation_and_diagram.py  # feature-group ablation + block diagram
cd tex && pdflatex paper.tex && pdflatex paper.tex
```

Seed is fixed (0); every number in the paper regenerates exactly.

## Files

- `src/make_dataset.py` – dataset generator (defect injection with labels)
- `src/dip_features.py` – the 10 DIP measures + intermediate images
- `src/train_eval.py` – baseline / RF / CNN training, metrics, figures
- `src/ablation_and_diagram.py` – ablations, block diagram
- `results/` – metrics.json, features.csv, ablation.csv, all PNG figures, train_log.txt
- `data/` – labels.csv + images/<class>/<product_id>.jpg
- `tex/paper.pdf` – 4-page IEEE conference paper

Data source: Fashion Product Images (Small) — Kaggle, HF mirror `ashraq/fashion-product-images-small`.
