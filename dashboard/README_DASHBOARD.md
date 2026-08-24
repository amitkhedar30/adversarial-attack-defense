# Dashboard setup

## 1. File layout

Drop `dashboard/` into the root of your existing repo, alongside your
`models/`, `utils/`, and `attacks/` folders:

```
adversarial-attack-defense/
├── models/
├── utils/
├── attacks/
├── dashboard/
│   ├── app.py
│   ├── requirements.txt
│   └── assets/
│       ├── checkpoints/
│       │   ├── resnet18_standard_cifar10.pt
│       │   ├── resnet18_robust_pgd_cifar10.pt
│       │   ├── resnet18_trades_cifar10.pt
│       │   └── resnet18_gaussian_cifar10.pt
│       ├── demo_images.pt
│       ├── cw_gallery.pt
│       ├── empirical_benchmarks.csv
│       ├── acr_summary.csv
│       ├── sanity_test_scaled.csv
│       ├── transferability_matrix.csv
│       ├── certified_accuracy_curve.png
│       └── loss_landscape_comparison.png
└── scripts/
    ├── export_demo_images.py
    └── generate_cw_gallery.py
```

`app.py` imports `models.classifier` and `attacks.whitebox` directly, so
it needs to live one level below the repo root (`dashboard/app.py`, not
buried deeper) -- it walks up one directory to find those modules.

## 2. Two one-time Kaggle steps

Both scripts are meant to be pasted into your Kaggle notebook and run
once, in an active session where the repo is already cloned and your
checkpoints already exist in `/kaggle/working/`.

1. **`scripts/export_demo_images.py`** -- exports 40 test images as
   `demo_images.pt`. The deployed dashboard has no access to
   `/kaggle/input`, so it needs its own small bundled copy instead of
   pulling from the full CIFAR-10 test set at runtime.

2. **`scripts/generate_cw_gallery.py`** -- run *after* step 1. Precomputes
   binary-search C&W adversarial examples for the first 15 of those 40
   images and saves them as `cw_gallery.pt`. This is why C&W isn't a
   live slider in the app: 9-step binary search is ~1000-1800
   optimization steps per image, fine on a Kaggle GPU, far too slow on
   the free CPU-only tiers most dashboards deploy to.

Download both outputs from your Kaggle session and copy them into
`dashboard/assets/`, alongside the CSVs and PNGs you already have from
the main notebook (`empirical_benchmarks.csv`, `acr_summary.csv`,
`sanity_test_scaled.csv`, `transferability_matrix.csv`,
`certified_accuracy_curve.png`, `loss_landscape_comparison.png`) and the
four checkpoint `.pt` files under `dashboard/assets/checkpoints/`.

## 3. Run it locally first

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Every tab degrades gracefully with a yellow warning if a specific asset
is missing, so you can wire this up incrementally and check each tab as
its files land, rather than needing everything in place before you see
anything render.

## 4. Deploying

**Checkpoint size**: four ResNet-18 checkpoints run roughly 170MB total.
A plain `git push` to a normal GitHub repo will likely need Git LFS for
files that large. Hugging Face Spaces handles large files natively and
is generally the smoother path for this size of model asset; Streamlit
Community Cloud works too but expect to set up Git LFS first.

**Compute**: everything in this app is either a static file load or a
FGSM/PGD forward-backward pass on a single image -- fine on CPU. Nothing
here requires the deploy target to have a GPU.
