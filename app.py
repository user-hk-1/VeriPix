"""
VeriPix: Universal Deepfake & AI Image Forensics
Visual Architecture: HydroShield-AI Replica (Glassmorphic Obsidian Dark)
Backbone: openai/clip-vit-base-patch16 (Frozen Foundation Vision)
Developer: Hitesh Kumar | NIELIT AIDA B8
"""

import os
import io
import glob
import time
import base64
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ExifTags
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

# -----------------------------------------------------------------------------
# 1. Page Configuration & Paths
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VeriPix // AI Image Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(WORKSPACE_DIR, "univfd_bundle.pt")

# -----------------------------------------------------------------------------
# 2. Glassmorphic & Luxury Obsidian CSS (HydroShield Replica + Anurati Typography)
# -----------------------------------------------------------------------------
def inject_custom_css():
    doodle_file = os.path.join(WORKSPACE_DIR, "doodle_bg.png") if os.path.exists(os.path.join(WORKSPACE_DIR, "doodle_bg.png")) else None
    bg_file = os.path.join(WORKSPACE_DIR, "bg.jpg") if os.path.exists(os.path.join(WORKSPACE_DIR, "bg.jpg")) else (os.path.join(WORKSPACE_DIR, "bg.png") if os.path.exists(os.path.join(WORKSPACE_DIR, "bg.png")) else None)

    # Load Anurati font as base64 (from NeuroShield)
    anurati_path = os.path.join(WORKSPACE_DIR, "assets", "fonts", "Anurati-Regular.otf")
    anurati_css = ""
    if os.path.exists(anurati_path):
        with open(anurati_path, "rb") as f:
            anurati_b64 = base64.b64encode(f.read()).decode()
        anurati_css = f"""
        @font-face {{
            font-family: 'Anurati';
            src: url('data:font/otf;base64,{anurati_b64}') format('opentype');
            font-weight: normal;
            font-style: normal;
            font-display: swap;
        }}
        """

    if doodle_file:
        with open(doodle_file, "rb") as f:
            encoded_doodle = base64.b64encode(f.read()).decode()
        bg_css = f"""
        .stApp {{
            background-color: #000000 !important;
            background-image: url('data:image/png;base64,{encoded_doodle}') !important;
            background-repeat: repeat !important;
            background-size: 650px 362.6px !important;
            animation: scrollPattern 30s linear infinite !important;
        }}

        @keyframes scrollPattern {{
            0% {{
                background-position: 0 0;
            }}
            100% {{
                background-position: 650px 362.6px;
            }}
        }}

        .block-container {{
            position: relative !important;
            z-index: 1 !important;
        }}
        """
    elif bg_file:
        with open(bg_file, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
        bg_css = f"""
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """
    else:
        bg_css = """
        .stApp {
            background: #000000 !important;
            background-color: #000000 !important;
        }
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

        {anurati_css}

        {bg_css}

        /* Anurati Sci-Fi Heading (from NeuroShield) */
        .anurati-brand-title {{
            font-family: 'Anurati', sans-serif !important;
            font-size: 5.5rem !important;
            font-weight: 800 !important;
            letter-spacing: 8px !important;
            text-transform: uppercase !important;
            color: #FFFFFF !important;
            text-align: center !important;
            line-height: 1.15 !important;
            text-shadow: 0 0 25px rgba(255, 255, 255, 0.45), 0 0 50px rgba(255, 255, 255, 0.20) !important;
            margin-top: 0 !important;
            margin-bottom: 2rem !important;
        }}

        * {{
            font-family: 'Inter', sans-serif;
            color: #F8FAFC;
        }}

        .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            max-width: 1150px !important;
        }}

        header {{ display: none !important; }}
        [data-testid="stDecoration"] {{ display: none !important; }}

        /* Text shadows for readability */
        h1, h2, h3, h4, p, span, b, strong, label {{
            text-shadow: 0 1px 4px rgba(0, 0, 0, 0.6) !important;
        }}

        /* ===================================== */
        /* HYDROSHIELD BLACK GLASS PILL BUTTONS  */
        /* ===================================== */
        div.stButton > button {{
            background: rgba(0, 0, 0, 0.65) !important;
            backdrop-filter: blur(25px) !important;
            -webkit-backdrop-filter: blur(25px) !important;
            border: 1.5px solid rgba(255, 255, 255, 0.35) !important;
            border-radius: 50px !important;
            color: #FFFFFF !important;
            height: 52px !important;
            width: 100% !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.5px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35) !important;
            transition: all 0.25s ease-in-out !important;
        }}

        div.stButton > button:hover {{
            background: rgba(0, 0, 0, 0.85) !important;
            border: 1.5px solid #00BFFF !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 32px rgba(0, 191, 255, 0.35) !important;
            color: #00BFFF !important;
        }}

        /* Active Selected Button */
        .active-pill div.stButton > button {{
            background: rgba(0, 191, 255, 0.22) !important;
            border: 1.5px solid #00BFFF !important;
            color: #00BFFF !important;
            box-shadow: 0 0 25px rgba(0, 191, 255, 0.4) !important;
        }}

        /* Clean Glass Containers */
        .glass-box {{
            background: rgba(6, 12, 26, 0.75) !important;
            backdrop-filter: blur(50px) saturate(200%) !important;
            -webkit-backdrop-filter: blur(50px) saturate(200%) !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
            border-radius: 22px !important;
            padding: 1.4rem !important;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.7) !important;
        }}

        /* Glassmorphic Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 12px !important;
            background: transparent !important;
            padding: 0 !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: rgba(4, 10, 24, 0.65) !important;
            backdrop-filter: blur(40px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(40px) saturate(180%) !important;
            border-radius: 30px !important;
            color: #B0C4DE !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            padding: 10px 22px !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3) !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(0, 191, 255, 0.25) !important;
            color: #00BFFF !important;
            border: 1.5px solid #00BFFF !important;
            box-shadow: 0 10px 25px rgba(0, 191, 255, 0.3) !important;
        }}
        .stTabs [data-baseweb="tab-border"] {{
            display: none !important;
        }}

        /* File Uploader Rounding */
        [data-testid="stFileUploader"] section {{
            background: rgba(6, 12, 26, 0.6) !important;
            border: 1.5px dashed rgba(255, 255, 255, 0.25) !important;
            border-radius: 20px !important;
            backdrop-filter: blur(25px) !important;
        }}
        [data-testid="stFileUploader"] section:hover {{
            border: 1.5px dashed #00BFFF !important;
        }}

        /* Rounded expanders */
        [data-testid="stExpander"] details {{
            background: rgba(6, 12, 26, 0.75) !important;
            border-radius: 20px !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
            backdrop-filter: blur(60px) !important;
            -webkit-backdrop-filter: blur(60px) !important;
            box-shadow: 0 15px 40px rgba(0,0,0,0.5) !important;
        }}
        [data-testid="stExpander"] summary {{
            color: #00BFFF !important;
            font-weight: 700 !important;
            padding: 0.9rem 1.4rem !important;
        }}

        /* Verdict Glow Boxes */
        .verdict-safe {{
            background: rgba(0, 255, 102, 0.12) !important;
            border: 1.5px solid #00FF66 !important;
            border-radius: 18px !important;
            padding: 1.2rem !important;
            text-align: center !important;
            box-shadow: 0 0 35px rgba(0, 255, 102, 0.25) !important;
        }}
        .verdict-hazard {{
            background: rgba(255, 23, 68, 0.14) !important;
            border: 1.5px solid #FF1744 !important;
            border-radius: 18px !important;
            padding: 1.2rem !important;
            text-align: center !important;
            box-shadow: 0 0 35px rgba(255, 23, 68, 0.25) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

inject_custom_css()

# -----------------------------------------------------------------------------
# 3. Model Architecture & Bundle Loader
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "univfd_bundle.pt")
WORKSPACE_DIR = os.path.dirname(__file__)

class ForensicProbe(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128):
        super(ForensicProbe, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_forensic_pipeline():
    if not os.path.exists(BUNDLE_PATH):
        return None, None, None, None
    bundle = torch.load(BUNDLE_PATH, map_location=DEVICE, weights_only=False)
    probe = ForensicProbe(in_dim=512, hidden_dim=128)
    probe.load_state_dict(bundle["probe_state_dict"])
    probe.to(DEVICE)
    probe.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    clip_model.to(DEVICE)
    clip_model.eval()
    return processor, clip_model, probe, bundle

processor, clip_model, probe, bundle = load_forensic_pipeline()

def analyze_single_image(pil_img, processor, clip_model, probe):
    rgb = pil_img.convert("RGB")
    inputs = processor(images=rgb, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        pooled_768 = vision_outputs[1]
        pooled_512 = clip_model.visual_projection(pooled_768)
        norm_feats = pooled_512 / pooled_512.norm(dim=-1, keepdim=True)
        logits = probe(norm_feats)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        prob_real = float(probs[0])
        prob_fake = float(probs[1])

        # Patch Attention Heatmap (14x14 ViT Tokens)
        patch_tokens = vision_outputs[0][:, 1:, :]  # (1, 196, 768)
        patch_512 = clip_model.visual_projection(patch_tokens)  # (1, 196, 512)
        patch_norm = patch_512 / patch_512.norm(dim=-1, keepdim=True)
        b, num_patches, d = patch_norm.shape
        flat_patches = patch_norm.reshape(-1, d)
        probe_eval = ForensicProbe(in_dim=d, hidden_dim=128).to(DEVICE)
        probe_eval.load_state_dict(probe.state_dict())
        probe_eval.eval()
        for m in probe_eval.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = False
                m.running_mean = probe.net[1].running_mean
                m.running_var = probe.net[1].running_var
        patch_logits = probe_eval(flat_patches)
        patch_probs = torch.softmax(patch_logits, dim=1)[:, 1].cpu().numpy()
        heatmap_14x14 = patch_probs.reshape(14, 14)

    return prob_real, prob_fake, norm_feats.cpu().numpy()[0], heatmap_14x14

def get_exif_metadata(pil_img):
    exif_data = {}
    try:
        raw_exif = pil_img.getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(value, (str, int, float)):
                    exif_data[tag_name] = value
    except Exception:
        pass
    return exif_data

# -----------------------------------------------------------------------------
# 4. Session State Management
# -----------------------------------------------------------------------------
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = None
if "landing_mode" not in st.session_state:
    st.session_state.landing_mode = None  # None, 'single', or 'grouping'

# -----------------------------------------------------------------------------
# 5. Application Router (Landing vs Triage View)
# -----------------------------------------------------------------------------
if st.session_state.current_image is None and st.session_state.landing_mode != "grouping":
    # =========================================================================
    # STATE A: HYDROSHIELD REPLICA LANDING HERO
    # =========================================================================

    # Vertical Centering Spacer
    st.markdown("<div style='height: 14vh;'></div>", unsafe_allow_html=True)

    # Grand Centered Title in Anurati Futuristic Font (White with Soft Glow)
    st.markdown("""
        <h1 class="anurati-brand-title">VERIPIX</h1>
    """, unsafe_allow_html=True)

    # Action Buttons (Left: Single Image | Right: Grouping)
    col_l, col_btn1, col_btn2, col_r = st.columns([2, 1.6, 1.6, 2])

    with col_btn1:
        pill_class = "active-pill" if st.session_state.landing_mode == "single" else ""
        st.markdown(f'<div class="{pill_class}">', unsafe_allow_html=True)
        if st.button("🔍 Single Image", use_container_width=True):
            st.session_state.landing_mode = "single"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_btn2:
        pill_class = "active-pill" if st.session_state.landing_mode == "grouping" else ""
        st.markdown(f'<div class="{pill_class}">', unsafe_allow_html=True)
        if st.button("📁 Grouping", use_container_width=True):
            st.session_state.landing_mode = "grouping"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Expanded Single Image Uploader Container
    if st.session_state.landing_mode == "single":
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        c_l, c_mid, c_r = st.columns([1.5, 3, 1.5])
        with c_mid:
            st.markdown("""
                <div class="glass-box" style="text-align: center; padding: 1.5rem;">
                    <div style="font-weight: 700; font-size: 1rem; color: #00BFFF; margin-bottom: 12px; letter-spacing: 0.5px;">
                        📥 UPLOAD TARGET IMAGE FOR FORENSIC SCAN
                    </div>
            """, unsafe_allow_html=True)

            file_upload = st.file_uploader(
                "Upload Frame",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
                key="landing_single_uploader"
            )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                if st.button("📸 Real Camera Sample", use_container_width=True):
                    real_samples = sorted(glob.glob(os.path.join(WORKSPACE_DIR, "demo_samples", "real", "*"))) or sorted(glob.glob(os.path.join(WORKSPACE_DIR, "test", "0_real", "*")))
                    if real_samples:
                        st.session_state.current_image = Image.open(real_samples[0])
                        st.session_state.current_image_name = os.path.basename(real_samples[0])
                        st.rerun()
            with col_s2:
                if st.button("🤖 AI Synthetic Sample", use_container_width=True):
                    fake_samples = sorted(glob.glob(os.path.join(WORKSPACE_DIR, "demo_samples", "ai", "*"))) or sorted(glob.glob(os.path.join(WORKSPACE_DIR, "test", "fake", "*")))
                    if fake_samples:
                        st.session_state.current_image = Image.open(fake_samples[0])
                        st.session_state.current_image_name = os.path.basename(fake_samples[0])
                        st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

            if file_upload:
                st.session_state.current_image = Image.open(file_upload)
                st.session_state.current_image_name = file_upload.name
                st.rerun()

elif st.session_state.landing_mode == "grouping" and st.session_state.current_image is None:
    # =========================================================================
    # STATE B (GROUPING): BATCH ANALYSIS VIEW
    # =========================================================================

    # Title & Navigation
    col_nav1, col_nav2 = st.columns([1, 4])
    with col_nav1:
        if st.button("⬅️ Return to Home", use_container_width=True):
            st.session_state.landing_mode = None
            st.rerun()
    with col_nav2:
        st.markdown("""
            <h2 style="margin: 0; font-size: 2rem; font-weight: 800; color: #FFFFFF;">
                📁 Batch Image Forensics & Automatic Triage
            </h2>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload images to batch process:",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    col_btn_run, col_btn_scan = st.columns(2)
    with col_btn_scan:
        if st.button("⚡ Test on 10 Mixed Benchmark Samples", use_container_width=True):
            r_samples = (sorted(glob.glob(os.path.join(WORKSPACE_DIR, "demo_samples", "real", "*"))) or sorted(glob.glob(os.path.join(WORKSPACE_DIR, "test", "0_real", "*"))))[:5]
            f_samples = (sorted(glob.glob(os.path.join(WORKSPACE_DIR, "demo_samples", "ai", "*"))) or sorted(glob.glob(os.path.join(WORKSPACE_DIR, "test", "fake", "*"))))[:5]
            all_samples = r_samples + f_samples
            if all_samples and probe is not None:
                progress_bar = st.progress(0)
                results_real = []
                results_fake = []
                for idx, path in enumerate(all_samples):
                    img = Image.open(path)
                    p_real, p_fake, _, _ = analyze_single_image(img, processor, clip_model, probe)
                    rec = {
                        "filename": os.path.basename(path),
                        "confidence": max(p_real, p_fake) * 100.0,
                        "p_real": p_real,
                        "p_fake": p_fake,
                        "image": img
                    }
                    if p_fake > p_real:
                        results_fake.append(rec)
                    else:
                        results_real.append(rec)
                    progress_bar.progress((idx + 1) / len(all_samples))

                st.session_state.batch_results = (results_real, results_fake)

    if batch_files and probe is not None:
        with col_btn_run:
            if st.button("🚀 Execute Batch Grouping", use_container_width=True):
                progress_bar = st.progress(0)
                results_real = []
                results_fake = []
                for idx, bf in enumerate(batch_files):
                    img = Image.open(bf)
                    p_real, p_fake, _, _ = analyze_single_image(img, processor, clip_model, probe)
                    rec = {
                        "filename": bf.name,
                        "confidence": max(p_real, p_fake) * 100.0,
                        "p_real": p_real,
                        "p_fake": p_fake,
                        "image": img
                    }
                    if p_fake > p_real:
                        results_fake.append(rec)
                    else:
                        results_real.append(rec)
                    progress_bar.progress((idx + 1) / len(batch_files))

                st.session_state.batch_results = (results_real, results_fake)

    if "batch_results" in st.session_state:
        res_real, res_fake = st.session_state.batch_results
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.success(f"Triage Complete: {len(res_real)} Authentic Optical Photos | {len(res_fake)} Generative AI Images.")

        col_b_real, col_b_fake = st.columns(2, gap="medium")
        with col_b_real:
            st.markdown(f"""
                <div style="background: rgba(0, 255, 102, 0.12); border-left: 4px solid #00FF66; padding: 12px 18px; border-radius: 14px; margin-bottom: 14px;">
                    <span style="font-weight: 800; color: #00FF66; font-size: 1.05rem;">✅ Authentic Camera Group ({len(res_real)})</span>
                </div>
            """, unsafe_allow_html=True)
            for item in res_real:
                st.markdown(f"""
                    <div class="glass-box" style="padding: 12px 16px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 0.88rem; font-weight: 600;">{item['filename']}</span>
                            <span style="background: rgba(0,255,102,0.2); color:#00FF66; padding: 3px 10px; border-radius: 20px; font-weight: 700; font-size: 0.8rem;">
                                {item['confidence']:.1f}% Real
                            </span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.image(item["image"], use_container_width=True)

        with col_b_fake:
            st.markdown(f"""
                <div style="background: rgba(255, 23, 68, 0.12); border-left: 4px solid #FF1744; padding: 12px 18px; border-radius: 14px; margin-bottom: 14px;">
                    <span style="font-weight: 800; color: #FF1744; font-size: 1.05rem;">🚨 Generative AI Group ({len(res_fake)})</span>
                </div>
            """, unsafe_allow_html=True)
            for item in res_fake:
                st.markdown(f"""
                    <div class="glass-box" style="padding: 12px 16px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 0.88rem; font-weight: 600;">{item['filename']}</span>
                            <span style="background: rgba(255,23,68,0.2); color:#FF1744; padding: 3px 10px; border-radius: 20px; font-weight: 700; font-size: 0.8rem;">
                                {item['confidence']:.1f}% AI
                            </span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.image(item["image"], use_container_width=True)

else:
    # =========================================================================
    # STATE C: FORENSIC TRIAGE & DIAGNOSTIC VIEW (Single Image Inspected)
    # =========================================================================

    # Navigation Bar
    col_nav, _ = st.columns([1.5, 4])
    with col_nav:
        if st.button("⬅️ Inspect New Image", use_container_width=True):
            st.session_state.current_image = None
            st.session_state.current_image_name = None
            st.rerun()

    if probe is None:
        st.error("⚠️ Forensic pipeline model bundle ('univfd_bundle.pt') missing.")
        st.stop()

    target_img = st.session_state.current_image
    img_name = st.session_state.current_image_name or "Uploaded Target"

    # Execute Inference
    t0 = time.time()
    p_real, p_fake, emb, heatmap_14x14 = analyze_single_image(target_img, processor, clip_model, probe)
    t_elapsed = time.time() - t0

    is_ai = p_fake >= p_real
    confidence = max(p_real, p_fake) * 100.0

    col_vision, col_triage = st.columns([1, 1.2], gap="large")

    with col_vision:
        # Glass Header Card
        st.markdown(f"""
            <div class="glass-box" style="padding: 0.9rem 1.2rem; text-align: center; margin-bottom: 1rem;">
                <h3 style="color:#00BFFF; margin:0; font-size:1.25rem; font-weight:800; letter-spacing:0.5px;">
                    🔬 Forensic Image Analysis
                </h3>
            </div>
        """, unsafe_allow_html=True)

        # Image Frame
        st.image(target_img, use_container_width=True)

        # Information Sub-card
        st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 8px 14px; border-radius: 12px; margin-top: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                <span style="font-size: 0.82rem; color: #B0C4DE;">
                    <b>{img_name}</b> | Resolution: {target_img.size[0]} × {target_img.size[1]} px
                </span>
            </div>
        """, unsafe_allow_html=True)

    with col_triage:
        # Symmetrical Glass Header Card (Replacing removed tabs)
        st.markdown("""
            <div class="glass-box" style="padding: 0.9rem 1.2rem; text-align: center; margin-bottom: 1rem;">
                <h3 style="color:#00BFFF; margin:0; font-size:1.25rem; font-weight:800; letter-spacing:0.5px;">
                    📊 ViT Patch Attention Analysis
                </h3>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="font-weight: 700; font-size: 0.92rem; color: #FFFFFF; margin-bottom: 8px;">
                Microscopic Patch Attention Heatmap (14×14 Vision Transformer Grid)
            </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 3), facecolor="#000000")
        ax.set_facecolor("#000000")
        im = ax.imshow(heatmap_14x14, cmap="magma", interpolation="bicubic", vmin=0, vmax=1)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.18, aspect=32)
        cbar.set_label('Synthetic Likelihood Intensity', color='#8C93A8', fontsize=8)
        cbar.ax.tick_params(labelsize=7, colors='#8C93A8')
        st.pyplot(fig)
        plt.close(fig)

        # Probabilities Bar
        st.markdown(f"""
            <div class="glass-box" style="padding: 1rem; margin-top: 0.8rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.85rem;">
                    <span style="color: #00FF66; font-weight: 700;">Authentic Optical: {p_real*100:.1f}%</span>
                    <span style="color: #FF1744; font-weight: 700;">Synthetic AI: {p_fake*100:.1f}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); height: 10px; border-radius: 5px; overflow: hidden; display: flex;">
                    <div style="background: #00FF66; width: {p_real*100}%;"></div>
                    <div style="background: #FF1744; width: {p_fake*100}%;"></div>
                </div>
                <div style="color: #B0C4DE; font-size: 0.78rem; margin-top: 8px; text-align: right;">
                    Inference Latency: <b>{t_elapsed*1000:.0f} ms</b> on CPU
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Glowing Status & Authenticity Verdict Card (Shifted below probabilities bar)
        if is_ai:
            st.markdown(f"""
                <div class="verdict-hazard" style="margin-top: 1rem;">
                    <div style="display: inline-block; background: rgba(0,0,0,0.5); border: 1.5px solid #FF1744; border-radius: 30px; padding: 0.4rem 1rem; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.1rem;">🚨</span>
                        <span style="color: #FF1744; font-size: 1rem; font-weight: 800; margin-left: 0.4rem;">
                            SYNTHETIC DIFFUSION DETECTED
                        </span>
                    </div>
                    <div style="color: #E8F0FE; font-size: 0.92rem; font-weight: 600; margin-top: 4px;">
                        Generative AI Latent Signatures Identified (Confidence: <b>{confidence:.1f}%</b>)
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="verdict-safe" style="margin-top: 1rem;">
                    <div style="display: inline-block; background: rgba(0,0,0,0.5); border: 1.5px solid #00FF66; border-radius: 30px; padding: 0.4rem 1rem; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.1rem;">✅</span>
                        <span style="color: #00FF66; font-size: 1rem; font-weight: 800; margin-left: 0.4rem;">
                            AUTHENTIC CAMERA PHOTO
                        </span>
                    </div>
                    <div style="color: #E8F0FE; font-size: 0.92rem; font-weight: 600; margin-top: 4px;">
                        Optical Sensor Noise Signatures Verified (Confidence: <b>{confidence:.1f}%</b>)
                    </div>
                </div>
            """, unsafe_allow_html=True)


