import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import shutil
import glob

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartSwipe",
    page_icon="✦",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# Dark editorial aesthetic. Syne for headings, DM Sans for body.
# Acid green (#b5ff4d) for Keep, warm red (#ff4d4d) for Delete.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0ede8;
}
.stApp { background-color: #0d0d0d; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #f0ede8;
    line-height: 1;
}
.subtitle {
    font-size: 0.9rem;
    color: #555;
    font-weight: 300;
    margin-top: 0.3rem;
    font-style: italic;
}
.phase-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #b5ff4d;
    margin-bottom: 0.4rem;
}
.stat-box {
    background: #141414;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 1rem 0.5rem;
    text-align: center;
}
.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #f0ede8;
}
.stat-label {
    font-size: 0.65rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.2rem;
}
.info-text {
    font-size: 0.82rem;
    color: #555;
    text-align: center;
    margin-top: 0.5rem;
}
.filename-text {
    text-align: center;
    color: #444;
    font-size: 0.72rem;
    margin-top: 0.3rem;
    font-family: 'DM Sans', monospace;
}
.prob-label-keep {
    color: #b5ff4d;
    font-size: 0.78rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
}
.prob-label-delete {
    color: #ff4d4d;
    font-size: 0.78rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    text-align: right;
}

/* ── Button styling ── */
/* Wrap buttons in divs with specific classes to color them independently */
.keep-btn button {
    background-color: #0d2a12 !important;
    color: #b5ff4d !important;
    border: 1px solid #b5ff4d !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    font-size: 1rem !important;
    border-radius: 6px !important;
    width: 100% !important;
    padding: 0.9rem !important;
    transition: opacity 0.15s !important;
}
.keep-btn button:hover { opacity: 0.8 !important; }

.delete-btn button {
    background-color: #2a0d0d !important;
    color: #ff4d4d !important;
    border: 1px solid #ff4d4d !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    font-size: 1rem !important;
    border-radius: 6px !important;
    width: 100% !important;
    padding: 0.9rem !important;
    transition: opacity 0.15s !important;
}
.delete-btn button:hover { opacity: 0.8 !important; }

/* Progress bar color */
.stProgress > div > div > div > div { background-color: #b5ff4d !important; }

/* Text input */
.stTextInput input {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    color: #f0ede8 !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 6px !important;
}

/* Horizontal rule */
hr { border: none; border-top: 1px solid #1a1a1a; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE LOADING — @st.cache_resource means this runs ONCE and stays in
# memory. Every rerun (every button click) skips this entirely.
# Architecture is identical to your notebook: MobileNetV2, classifier replaced
# with Identity so the output is the raw 1280d feature vector.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_backbone():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = MobileNet_V2_Weights.DEFAULT
    backbone = models.mobilenet_v2(weights=weights).to(device)
    backbone.classifier = nn.Identity()   # cuts off classification head → (batch, 1280)
    backbone.eval()
    return backbone, device


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE TRANSFORM — must match your notebook exactly.
# MobileNetV2 expects 224×224, ImageNet normalisation.
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING EXTRACTION
# Scans the folder, loads every image, runs it through the frozen backbone.
# Returns:  embeddings → np.array (N, 1280)
#           paths      → list of N absolute file paths
# ─────────────────────────────────────────────────────────────────────────────
def extract_embeddings(folder_path, backbone, device):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.webp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder_path, ext)))
    paths = sorted(set(paths))   # deduplicate, deterministic order

    if not paths:
        return None, None

    embeddings = []
    valid_paths = []
    bar = st.progress(0, text="Extracting features...")

    for i, path in enumerate(paths):
        try:
            img = Image.open(path).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = backbone(tensor)               # (1, 1280)
            embeddings.append(feat.cpu().numpy())
            valid_paths.append(path)
        except Exception:
            pass   # skip corrupt / unreadable files
        bar.progress((i + 1) / len(paths),
                     text=f"Extracting features...  {i+1} / {len(paths)}")

    bar.empty()
    if not embeddings:
        return None, None

    return np.vstack(embeddings), valid_paths   # (N, 1280), list[N]


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE LEARNING — GET NEXT INDEX
# Called after every swipe. If both classes exist, fits the LogReg on the
# labeled set, then returns the unlabeled index whose predicted probability
# is closest to 0.5 (maximum uncertainty).
# If only one class exists, returns the next index in order (safe fallback).
# ─────────────────────────────────────────────────────────────────────────────
def get_next_index():
    labeled_data    = st.session_state.labeled_data
    unlabeled       = st.session_state.unlabeled_indices

    if len(unlabeled) == 0:
        return None

    labels_so_far = list(labeled_data.values())
    has_both_classes = (0 in labels_so_far) and (1 in labels_so_far)

    if not has_both_classes:
        return int(unlabeled[0])   # fallback: sequential until model is usable

    # ── Fit LogReg on everything labeled so far ──────────────────────────────
    X_labeled = st.session_state.embeddings[list(labeled_data.keys())]
    y_labeled = np.array(list(labeled_data.values()))
    st.session_state.model.fit(X_labeled, y_labeled)

    # ── Uncertainty sampling on remaining pool ───────────────────────────────
    X_unlabeled     = st.session_state.embeddings[unlabeled]
    probs           = st.session_state.model.predict_proba(X_unlabeled)[:, 1]
    uncertainty     = np.abs(probs - 0.5)            # 0 = maximally uncertain
    best_local      = np.argmin(uncertainty)

    return int(unlabeled[best_local])


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# Every key is guarded with `if key not in st.session_state` so it only runs
# on the very first load. Every subsequent rerun (button click) skips this
# and the stored values are preserved.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "phase":             "setup",    # 'setup' | 'swiping' | 'done'
    "embeddings":        None,       # np.array (N, 1280)
    "paths":             None,       # list of N file paths
    "labeled_data":      {},         # {index: 0 or 1}
    "unlabeled_indices": None,       # np.array of remaining indices
    "model":             LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "current_idx":       None,       # index of image currently displayed
    "folder_path":       "",
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">SmartSwipe ✦</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered gallery cleanup · MobileNetV2 + Active Learning</div>',
    unsafe_allow_html=True
)
st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — SETUP
# User enters a folder path. On button click: extract embeddings, store in
# session_state, transition to 'swiping'.
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "setup":

    st.markdown('<div class="phase-label">Step 01 — Load Gallery</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Point SmartSwipe at a folder of images. Features are extracted once — "
        "every swipe after that is instant."
    )
    st.markdown("")

    folder_path = st.text_input(
        "Folder path",
        value=st.session_state.folder_path,
        placeholder="/path/to/your/photos"
    )

    if st.button("⚡  Load & Precompute", type="primary", use_container_width=True):
        if not folder_path or not os.path.isdir(folder_path):
            st.error("Path not found. Please enter a valid directory.")
        else:
            backbone, device = load_backbone()
            with st.spinner("Loading backbone..."):
                embeddings, paths = extract_embeddings(folder_path, backbone, device)

            if embeddings is None or len(paths) < 5:
                st.error("Found fewer than 5 readable images. Point to a larger folder.")
            else:
                st.session_state.embeddings        = embeddings
                st.session_state.paths             = paths
                st.session_state.folder_path       = folder_path
                st.session_state.unlabeled_indices = np.arange(len(paths))
                st.session_state.labeled_data      = {}
                st.session_state.model             = LogisticRegression(
                                                        max_iter=1000, C=1.0, random_state=42)
                st.session_state.current_idx       = int(np.arange(len(paths))[0])
                st.session_state.phase             = "swiping"
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — SWIPING
# Shows one image at a time. Keep / Delete buttons trigger:
#   1. Record label in labeled_data
#   2. Remove index from unlabeled_indices
#   3. Call get_next_index() → fits model if ready, returns most uncertain idx
#   4. Set current_idx to that result, rerun
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "swiping":

    n_total     = len(st.session_state.paths)
    n_labeled   = len(st.session_state.labeled_data)
    n_remaining = len(st.session_state.unlabeled_indices)
    current_idx = st.session_state.current_idx

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.markdown('<div class="phase-label">Step 02 — Swipe to Label</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-number">{n_total}</div>'
            f'<div class="stat-label">Total</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-number">{n_labeled}</div>'
            f'<div class="stat-label">Labeled</div></div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-number">{n_remaining}</div>'
            f'<div class="stat-label">Remaining</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Image display ─────────────────────────────────────────────────────────
    if current_idx is not None:
        img_path = st.session_state.paths[current_idx]
        st.image(img_path, use_container_width=True)
        st.markdown(
            f'<div class="filename-text">{os.path.basename(img_path)}</div>',
            unsafe_allow_html=True
        )

        # ── Confidence bar (only visible once both classes have been labeled) ──
        labels_so_far = list(st.session_state.labeled_data.values())
        has_both      = (0 in labels_so_far) and (1 in labels_so_far)

        if has_both:
            prob      = float(st.session_state.model.predict_proba(
                            st.session_state.embeddings[[current_idx]])[0, 1])
            keep_pct  = int(round(prob * 100))
            del_pct   = 100 - keep_pct

            lc, rc = st.columns(2)
            with lc:
                st.markdown(
                    f'<div class="prob-label-keep">KEEP {keep_pct}%</div>',
                    unsafe_allow_html=True
                )
            with rc:
                st.markdown(
                    f'<div class="prob-label-delete">DELETE {del_pct}%</div>',
                    unsafe_allow_html=True
                )
            st.progress(prob, text="")   # green bar = P(Keep)
        else:
            st.markdown(
                '<div class="info-text">'
                'Model activates after you label at least one Keep and one Delete'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ── Keep / Delete buttons ─────────────────────────────────────────────
        btn_left, btn_right = st.columns(2)

        with btn_left:
            st.markdown('<div class="delete-btn">', unsafe_allow_html=True)
            if st.button("✕  Delete", key="btn_delete", use_container_width=True):
                st.session_state.labeled_data[current_idx] = 0
                st.session_state.unlabeled_indices = np.setdiff1d(
                    st.session_state.unlabeled_indices, [current_idx])
                next_idx = get_next_index()
                if next_idx is None:
                    st.session_state.phase = "done"
                else:
                    st.session_state.current_idx = next_idx
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with btn_right:
            st.markdown('<div class="keep-btn">', unsafe_allow_html=True)
            if st.button("✓  Keep", key="btn_keep", use_container_width=True):
                st.session_state.labeled_data[current_idx] = 1
                st.session_state.unlabeled_indices = np.setdiff1d(
                    st.session_state.unlabeled_indices, [current_idx])
                next_idx = get_next_index()
                if next_idx is None:
                    st.session_state.phase = "done"
                else:
                    st.session_state.current_idx = next_idx
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Early finish (available once the model has enough labels to be useful) ─
    if n_labeled >= 10 and has_both:
        if st.button("→  Finish & Auto-classify Remaining", use_container_width=True):
            st.session_state.phase = "done"
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — DONE / PROCESS
# Auto-classifies any images the user didn't swipe through (using the trained
# LogReg). Shows a preview of flagged images. "Process Gallery" moves them
# into <folder>/_smartswipe_review/ using shutil.move.
# Nothing is permanently deleted — user reviews the folder manually.
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "done":

    st.markdown('<div class="phase-label">Step 03 — Review & Process</div>',
                unsafe_allow_html=True)

    labeled_data      = st.session_state.labeled_data
    unlabeled_indices = st.session_state.unlabeled_indices
    labels_so_far     = list(labeled_data.values())
    has_model         = (0 in labels_so_far) and (1 in labels_so_far)

    # Separate user-labeled keeps and deletes
    user_deletes = [idx for idx, lbl in labeled_data.items() if lbl == 0]
    user_keeps   = [idx for idx, lbl in labeled_data.items() if lbl == 1]
    auto_deletes = []
    auto_keeps   = []

    # Auto-classify images the user never swiped
    if has_model and len(unlabeled_indices) > 0:
        X_remaining = st.session_state.embeddings[unlabeled_indices]
        preds       = st.session_state.model.predict(X_remaining)
        for i, idx in enumerate(unlabeled_indices):
            if preds[i] == 0:
                auto_deletes.append(int(idx))
            else:
                auto_keeps.append(int(idx))

    all_deletes = user_deletes + auto_deletes
    all_keeps   = user_keeps   + auto_keeps

    # ── Summary stats ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-number" style="color:#b5ff4d;">{len(all_keeps)}</div>'
            f'<div class="stat-label">Keep</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-number" style="color:#ff4d4d;">{len(all_deletes)}</div>'
            f'<div class="stat-label">To Delete</div></div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-number" style="color:#888;">{len(auto_deletes)}</div>'
            f'<div class="stat-label">Auto-classified</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Preview of flagged images (first 10) ──────────────────────────────────
    if all_deletes:
        st.markdown("**Flagged for deletion** (preview of first 10):")
        preview = all_deletes[:10]
        cols = st.columns(min(5, len(preview)))
        for i, idx in enumerate(preview):
            with cols[i % 5]:
                st.image(st.session_state.paths[idx], use_container_width=True)

    st.markdown("---")

    # ── Process Gallery ───────────────────────────────────────────────────────
    review_folder = os.path.join(st.session_state.folder_path, "_smartswipe_review")
    st.markdown(f"Flagged images will be **moved** (not deleted) to:")
    st.code(review_folder)
    st.markdown(
        '<div class="info-text">You can review and permanently delete from there.</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    if st.button("⚡  Process Gallery", type="primary", use_container_width=True):
        os.makedirs(review_folder, exist_ok=True)
        moved, failed = 0, 0
        for idx in all_deletes:
            src = st.session_state.paths[idx]
            dst = os.path.join(review_folder, os.path.basename(src))
            try:
                shutil.move(src, dst)
                moved += 1
            except Exception as e:
                st.warning(f"Could not move {os.path.basename(src)}: {e}")
                failed += 1
        if failed == 0:
            st.success(f"✓ Done. {moved} images moved to review folder.")
        else:
            st.warning(f"Moved {moved} images. {failed} failed — check warnings above.")

    st.markdown("---")

    if st.button("↺  Start Over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
