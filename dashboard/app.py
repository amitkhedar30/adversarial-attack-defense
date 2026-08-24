"""
Adversarial Robustness Dashboard
=================================
Interactive dashboard for the empirical + certified adversarial
robustness project (Madry et al. 2018 PGD-AT / TRADES + Cohen et al.
2019 randomized smoothing).

DEPLOYMENT NOTES
-----------------
- This file expects to sit inside the project repo, at <repo>/dashboard/
  app.py, so it can import `models.classifier` and `attacks.whitebox`
  directly -- the same modules used in the Kaggle training notebook,
  rather than reimplementing (and risking drift from) that code.
- All static assets (checkpoints, precomputed C&W gallery, CSVs, demo
  images) are expected under dashboard/assets/. See README_DASHBOARD.md
  for the two one-time Kaggle scripts that generate the assets this app
  cannot compute live (demo image subset + C&W gallery).
- FGSM and PGD run LIVE in this app -- both are fast on CPU (FGSM is a
  single gradient step, PGD-10 is ten). C&W does NOT run live: with
  9-step binary search it's ~1000-1800 optimization steps per image,
  which is fine on a Kaggle GPU but far too slow on the free CPU-only
  tiers Streamlit Community Cloud and HF Spaces use. C&W results are
  precomputed once (scripts/generate_cw_gallery.py) and just displayed
  here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Import the SAME model/attack code used for training, instead of
# reimplementing it here.
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from models.classifier import get_cifar10_resnet18  # noqa: E402
from attacks.whitebox import fgsm_attack, pgd_attack  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

MODEL_FILES = {
    "Standard (Baseline)": "resnet18_standard_cifar10.pt",
    "PGD Adversarial Training": "resnet18_robust_pgd_cifar10.pt",
    "TRADES": "resnet18_trades_cifar10.pt",
    "Gaussian (sigma=0.25)": "resnet18_gaussian_cifar10.pt",
}

st.set_page_config(page_title="Adversarial Robustness Dashboard", layout="wide")


# ------------------------------------------------------------------
# Cached loaders -- everything here reads static files, nothing trains
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_model(filename: str):
    path = ASSETS / "checkpoints" / filename
    if not path.exists():
        return None
    model = get_cifar10_resnet18(num_classes=10)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_data(show_spinner="Loading demo images...")
def load_demo_images():
    path = ASSETS / "demo_images.pt"
    if not path.exists():
        return None, None
    bundle = torch.load(path, map_location="cpu")
    return bundle["images"], bundle["labels"]


@st.cache_data(show_spinner="Loading C&W gallery...")
def load_cw_gallery():
    path = ASSETS / "cw_gallery.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


@st.cache_data
def load_csv(name: str):
    path = ASSETS / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def missing_asset_notice(name: str):
    st.warning(
        f"`{name}` not found under `dashboard/assets/`. Run the "
        f"corresponding export script in your Kaggle session and copy "
        f"the output here -- see README_DASHBOARD.md."
    )


# ------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------

def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """CHW tensor in [0,1] -> HWC numpy array for st.image."""
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def predict_probs(model, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = model(x.unsqueeze(0).to(DEVICE))
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def confidence_bar_chart(probs_a, label_a, probs_b, label_b, true_class_idx):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(CIFAR10_CLASSES))
    width = 0.35
    ax.bar(x - width / 2, probs_a, width, label=label_a, color="#d9534f")
    ax.bar(x + width / 2, probs_b, width, label=label_b, color="#428bca")
    ax.axvline(true_class_idx, color="black", linestyle="--", linewidth=1, label="True class")
    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# ==================================================================
# UI
# ==================================================================

st.title("Adversarial Robustness Dashboard")
st.caption(
    "Empirical (PGD-AT, TRADES) and certified (randomized smoothing) "
    "robustness on CIFAR-10 -- reproducing Madry et al. 2018 and "
    "Cohen et al. 2019."
)

tab_attack, tab_leaderboard, tab_certified, tab_analytics = st.tabs(
    ["Attack Demo", "Benchmark Leaderboard", "Certified Robustness", "Under the Hood"]
)

# ------------------------------------------------------------------
# TAB 1 -- Interactive Attack Demo
# ------------------------------------------------------------------
with tab_attack:
    st.subheader("See a model get fooled -- and a robust model resist it")

    images, labels = load_demo_images()
    if images is None:
        missing_asset_notice("demo_images.pt")
    else:
        col_controls, col_display = st.columns([1, 2])

        with col_controls:
            if "demo_idx" not in st.session_state:
                st.session_state.demo_idx = 0

            if st.button("Random image"):
                st.session_state.demo_idx = int(np.random.randint(0, len(images)))

            st.session_state.demo_idx = st.selectbox(
                "Or pick an image index",
                options=list(range(len(images))),
                index=st.session_state.demo_idx,
                format_func=lambda i: f"#{i} -- {CIFAR10_CLASSES[labels[i]]}",
            )

            robust_choice = st.selectbox(
                "Compare Standard model against",
                ["PGD Adversarial Training", "TRADES"],
            )

            attack_choice = st.radio("Attack", ["FGSM", "PGD", "C&W (precomputed)"])

            epsilon, alpha, iters = None, None, None
            if attack_choice in ("FGSM", "PGD"):
                epsilon_255 = st.slider("Epsilon (in /255 units)", 1, 16, 8)
                epsilon = epsilon_255 / 255
                if attack_choice == "PGD":
                    iters = st.slider("PGD iterations", 5, 20, 10)
                    alpha = epsilon / 4

        idx = st.session_state.demo_idx
        clean_img = images[idx]
        true_label = int(labels[idx])

        model_std = load_model(MODEL_FILES["Standard (Baseline)"])
        model_rob = load_model(MODEL_FILES[robust_choice])

        if model_std is None or model_rob is None:
            missing_asset_notice("checkpoints/*.pt")
        else:
            gallery_note = None
            if attack_choice == "FGSM":
                adv_img = fgsm_attack(
                    model_std, clean_img.unsqueeze(0).to(DEVICE),
                    torch.tensor([true_label]).to(DEVICE), epsilon=epsilon,
                ).squeeze(0).cpu()
            elif attack_choice == "PGD":
                adv_img = pgd_attack(
                    model_std, clean_img.unsqueeze(0).to(DEVICE),
                    torch.tensor([true_label]).to(DEVICE),
                    epsilon=epsilon, alpha=alpha, iters=iters,
                ).squeeze(0).cpu()
            else:
                gallery = load_cw_gallery()
                if gallery is None or idx not in gallery:
                    gallery_note = (
                        "This image isn't in the precomputed C&W gallery "
                        "(only the first N_GALLERY_IMAGES demo images are "
                        "precomputed). Showing the clean image instead -- "
                        "pick a lower-numbered image or extend the gallery "
                        "with scripts/generate_cw_gallery.py."
                    )
                    adv_img = clean_img.clone()
                else:
                    adv_img = gallery[idx]["adv_image"]
                    gallery_note = (
                        f"Precomputed via 9-step binary-search C&W "
                        f"(L2 norm: {gallery[idx]['l2_norm']:.4f}). C&W "
                        f"auto-tunes its own strength via binary search, "
                        f"so there's no epsilon slider for it."
                    )

            with col_display:
                if gallery_note:
                    st.info(gallery_note)

                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(tensor_to_image(clean_img), caption="Original", use_container_width=True)
                with img_col2:
                    st.image(tensor_to_image(adv_img), caption="Adversarial", use_container_width=True)

                probs_std = predict_probs(model_std, adv_img)
                probs_rob = predict_probs(model_rob, adv_img)

                pred_std = CIFAR10_CLASSES[int(probs_std.argmax())]
                pred_rob = CIFAR10_CLASSES[int(probs_rob.argmax())]
                true_name = CIFAR10_CLASSES[true_label]

                m1, m2 = st.columns(2)
                m1.metric(
                    "Standard model prediction", pred_std,
                    delta=("fooled" if pred_std != true_name else "correct"),
                    delta_color=("inverse" if pred_std != true_name else "normal"),
                )
                m2.metric(
                    f"{robust_choice} prediction", pred_rob,
                    delta=("fooled" if pred_rob != true_name else "correct"),
                    delta_color=("inverse" if pred_rob != true_name else "normal"),
                )

                st.pyplot(confidence_bar_chart(
                    probs_std, "Standard", probs_rob, robust_choice, true_label
                ))

# ------------------------------------------------------------------
# TAB 2 -- Benchmark Leaderboard
# ------------------------------------------------------------------
with tab_leaderboard:
    st.subheader("Empirical benchmark results")
    df = load_csv("empirical_benchmarks.csv")
    if df is None:
        missing_asset_notice("empirical_benchmarks.csv")
    else:
        st.dataframe(
            df.style.format(
                {
                    "clean_acc": "{:.2f}%",
                    "fgsm_acc": "{:.2f}%",
                    "pgd10_acc": "{:.2f}%",
                    "cw_l2_acc": "{:.2f}%",
                    "cw_l2_avg_l2_norm": "{:.4f}",
                },
                na_rep="--",
            ),
            use_container_width=True,
        )
        st.caption(
            "Baseline collapses under PGD; PGD-AT and TRADES trade some "
            "clean accuracy for robustness, with TRADES landing slightly "
            "ahead on PGD-10 at a comparable clean-accuracy cost."
        )

# ------------------------------------------------------------------
# TAB 3 -- Certified Robustness
# ------------------------------------------------------------------
with tab_certified:
    st.subheader("Certified robustness via randomized smoothing")
    st.caption("Cohen et al. 2019 -- Gaussian-augmented model, sigma=0.25")

    curve_path = ASSETS / "certified_accuracy_curve.png"
    if curve_path.exists():
        st.image(str(curve_path), use_container_width=True)
    else:
        missing_asset_notice("certified_accuracy_curve.png")

    acr_df = load_csv("acr_summary.csv")
    sanity_df = load_csv("sanity_test_scaled.csv")

    m1, m2, m3 = st.columns(3)
    if acr_df is not None:
        row = acr_df.iloc[0]
        m1.metric("Average Certified Radius (ACR)", f"{row['acr']:.4f}")
        m2.metric(
            "Correctly certified",
            f"{int(row['n_correct'])}/{int(row['n_images'])}",
            f"{100 * row['n_correct'] / row['n_images']:.1f}%",
        )
    else:
        missing_asset_notice("acr_summary.csv")

    if sanity_df is not None:
        held = int(sanity_df["bound_held"].sum())
        total = len(sanity_df)
        m3.metric("Certified bound held (sanity test)", f"{held}/{total}", f"{100 * held / total:.1f}%")
    else:
        missing_asset_notice("sanity_test_scaled.csv")

    st.caption(
        "The sanity test scales real C&W adversarial perturbations to "
        "just inside each image's certified radius and confirms the "
        "smoothed classifier's prediction never flips -- an empirical "
        "check on top of the theoretical guarantee."
    )

# ------------------------------------------------------------------
# TAB 4 -- Under the Hood
# ------------------------------------------------------------------
with tab_analytics:
    st.subheader("Why the defenses work")

    st.markdown("**Loss landscape: standard vs. robust model**")
    landscape_path = ASSETS / "loss_landscape_comparison.png"
    if landscape_path.exists():
        st.image(str(landscape_path), use_container_width=True)
        st.caption(
            "Adversarial training flattens the loss surface around the "
            "input -- the standard model's steep cliff becomes a "
            "shallow plateau, meaning small perturbations move the loss "
            "much less."
        )
    else:
        missing_asset_notice("loss_landscape_comparison.png")

    st.markdown("---")
    st.markdown("**Attack transferability across architectures**")
    transfer_df = load_csv("transferability_matrix.csv")
    if transfer_df is not None:
        transfer_df = transfer_df.set_index(transfer_df.columns[0])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(transfer_df.values, cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xticks(range(len(transfer_df.columns)))
        ax.set_xticklabels(transfer_df.columns, rotation=20, ha="right")
        ax.set_yticks(range(len(transfer_df.index)))
        ax.set_yticklabels(transfer_df.index)
        for i in range(transfer_df.shape[0]):
            for j in range(transfer_df.shape[1]):
                ax.text(j, i, f"{transfer_df.values[i, j]:.1f}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, label="Target accuracy under attack (%)")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Diagonal cells are white-box attacks (source = target); "
            "off-diagonal cells are black-box transfer. Higher accuracy "
            "off-diagonal means the attack transferred poorly."
        )
    else:
        missing_asset_notice("transferability_matrix.csv")
