"""
Synthetic Media Detection
WALL-E / Retro-Futuristic Industrial Sci-Fi Theme with Cosmic Dust & CRT Animations
"""

import sys
import io
import json
from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import torch

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import (
    BEST_MODEL_PATH, SAMPLES_DIR, REPORTS_DIR, DEVICE,
    DEFAULT_MODEL_NAME, IMG_SIZE
)
from src.models.detector import build_detector
from src.engine.authenticator import MediaAuthenticator
from src.utils.report_gen import generate_forensic_pdf
from test_robustness import (
    apply_jpeg_compression, apply_gaussian_blur,
    apply_resize_down_up
)

# ── EXIF Metadata Extraction ──────────────────────────────────────────────────
def extract_exif(img: Image.Image, filename: str = "") -> dict:
    """Return a clean dict of useful EXIF / image metadata."""
    info = {}
    try:
        raw = img._getexif() or {}
        from PIL.ExifTags import TAGS
        friendly = {TAGS.get(k, k): v for k, v in raw.items() if isinstance(v, (str, int, float, bytes))}
        for key in ["Make", "Model", "Software", "DateTime", "DateTimeOriginal",
                    "ExifImageWidth", "ExifImageHeight", "Flash", "ISO",
                    "FocalLength", "ExposureTime", "FNumber", "GPSInfo"]:
            if key in friendly:
                val = friendly[key]
                if isinstance(val, bytes):
                    val = val.decode(errors="ignore")
                info[key] = str(val)
    except Exception:
        pass

    info["FileFormat"] = img.format or Path(filename).suffix.upper().strip(".")
    info["Mode"] = img.mode
    info["Resolution"] = f"{img.width} × {img.height} px"
    info["MegaPixels"] = f"{img.width * img.height / 1_000_000:.2f} MP"

    # AI-generated images almost never have camera make/model EXIF
    has_camera = any(k in info for k in ["Make", "Model", "DateTimeOriginal"])
    info["_camera_exif_present"] = has_camera
    return info


# Page Setup
st.set_page_config(
    page_title="Synthetic Media Detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Pure Black Interactive Background + Clean Dark UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding: 1.45rem 2.2rem 3rem !important; max-width: 1540px; position: relative; z-index: 2;}

    /* Pure Black Background — transparent so Vanta.js shows through */
    [data-testid="stAppViewContainer"], .stApp, .main, [data-testid="stHeader"] {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* Glassmorphic Card */
    .walle-card {
        background: linear-gradient(135deg, rgba(19, 30, 63, 0.76), rgba(7, 11, 27, 0.68)) !important;
        border: 1px solid rgba(142, 174, 255, 0.16) !important;
        border-radius: 18px !important;
        padding: 1.4rem !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28), inset 0 1px rgba(255,255,255,.04) !important;
        margin-bottom: 1.2rem !important;
        transition: border-color 0.25s ease;
    }
    .walle-card:hover {
        border-color: rgba(119, 153, 255, 0.48) !important;
    }

    /* Title */
    .walle-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #FFFFFF;
        margin: 0 0 4px 0;
    }
    .walle-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        color: #9aa8c7;
        margin-bottom: 1.8rem;
    }

    /* Verdict Banners */
    .verdict-box {
        border-radius: 12px;
        padding: 1.3rem;
        text-align: center;
        backdrop-filter: blur(14px);
        margin: 1.2rem 0;
    }
    .verdict-synth {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #FCA5A5;
    }
    .verdict-auth {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.5);
        color: #86EFAC;
    }
    .verdict-susp {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.5);
        color: #FDE68A;
    }

    /* Metric Cards */
    .walle-metric {
        background: rgba(10, 17, 40, 0.7);
        border: 1px solid rgba(142, 174, 255, 0.14);
        border-radius: 16px;
        padding: 17px;
        text-align: left;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .walle-metric:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.2);
    }
    .walle-metric-val {
        font-family: 'Inter', sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
        color: #f7f8ff;
        margin-top: 4px;
    }
    .walle-metric-lbl {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #9aa8c7;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(125deg, #6577ff, #9a55f5) !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 11px !important;
        box-shadow: 0 8px 25px rgba(89, 93, 255, 0.26) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 28px rgba(89, 93, 255, 0.4) !important;
    }
    .aura-shell {display:flex; gap:18px; align-items:stretch; margin-bottom:22px;}
    .aura-brand {min-width:250px; padding:21px; border-radius:18px; background:rgba(7,13,33,.76); border:1px solid rgba(150,180,255,.15); backdrop-filter:blur(16px);}
    .aura-mark {width:38px; height:38px; display:inline-grid; place-items:center; border-radius:12px; color:#c8d5ff; font-size:22px; background:linear-gradient(135deg,#6e78ff,#31d8cc); box-shadow:0 0 28px rgba(93,116,255,.55);}
    .aura-brand h3 {display:inline-block; vertical-align:top; margin:3px 0 0 10px; color:#fff; font-size:15px; letter-spacing:.7px;}
    .aura-brand p {font:11px 'JetBrains Mono', monospace; color:#8d9aba; margin:5px 0 22px;}
    .aura-nav {padding:11px 12px; margin:4px -5px; border-radius:10px; color:#aeb9d4; font:13px Inter,sans-serif;}
    .aura-nav.active {color:#fff; background:linear-gradient(90deg,rgba(89,111,255,.58),rgba(155,77,242,.46)); border:1px solid rgba(161,164,255,.38);}
    .aura-status {margin-top:25px; border-top:1px solid rgba(255,255,255,.08); padding-top:18px; color:#96a4c4; font:11px 'JetBrains Mono',monospace; line-height:2.1;}
    .online-dot {color:#39e1a6; font-size:17px; vertical-align:-1px;}
    .aura-main {flex:1; min-width:0;}
    .topline {display:flex; align-items:center; justify-content:space-between; margin:2px 0 23px;}
    .eyebrow {font:11px 'JetBrains Mono',monospace; letter-spacing:1.4px; color:#8998ff; text-transform:uppercase;}
    .system-pill {color:#a8b7d6; font:11px 'JetBrains Mono',monospace; padding:8px 11px; border:1px solid rgba(119,153,255,.22); border-radius:20px; background:rgba(11,19,44,.45);}
    .dash-kpi {font:11px 'JetBrains Mono', monospace; color:#92a1c1; text-transform:uppercase; letter-spacing:.55px;}
    .dash-icon {font-size:21px; float:right; color:#a3aaff; opacity:.9;}
    .section-heading {font-size:15px; font-weight:700; color:#eef1ff; margin:0 0 5px;}
    [data-testid="stFileUploader"] {background:rgba(3,8,24,.33); border-radius:12px; padding:8px; border:1px dashed rgba(141,165,255,.28);}
    [data-testid="stExpander"] {background:rgba(7,13,32,.45); border:1px solid rgba(136,160,255,.12); border-radius:12px;}
    @media(max-width:900px){.aura-brand{display:none}.block-container{padding:1rem!important}.aura-shell{display:block}}

    /* ── Animated Confidence Gauge ── */
    .gauge-wrap { text-align:center; padding: 10px 0 4px; }
    .gauge-label { font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#8fa0c2; text-transform:uppercase; letter-spacing:.6px; margin-bottom:6px; }
    .gauge-ring { position:relative; display:inline-block; width:130px; height:130px; }
    .gauge-ring svg { transform:rotate(-90deg); }
    .gauge-track { fill:none; stroke:rgba(255,255,255,.07); stroke-width:10; }
    .gauge-fill  { fill:none; stroke-width:10; stroke-linecap:round; transition: stroke-dasharray 1.1s cubic-bezier(.4,0,.2,1), stroke 0.5s ease; }
    .gauge-center { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); text-align:center; }
    .gauge-pct  { font-family:'Inter',sans-serif; font-size:1.35rem; font-weight:800; }
    .gauge-word { font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#9aa8c7; letter-spacing:.5px; }

    /* ── Batch table rows ── */
    .batch-row { display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:10px; margin:5px 0; background:rgba(10,17,42,.55); border:1px solid rgba(130,158,255,.1); }
    .batch-fname { flex:1; font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#c5cfea; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .batch-badge-synth { background:rgba(239,68,68,.18); color:#fca5a5; border:1px solid rgba(239,68,68,.4); border-radius:6px; padding:2px 9px; font-size:0.72rem; font-weight:700; }
    .batch-badge-auth  { background:rgba(34,197,94,.14); color:#86efac; border:1px solid rgba(34,197,94,.36); border-radius:6px; padding:2px 9px; font-size:0.72rem; font-weight:700; }
    .batch-badge-susp  { background:rgba(245,158,11,.14); color:#fde68a; border:1px solid rgba(245,158,11,.36); border-radius:6px; padding:2px 9px; font-size:0.72rem; font-weight:700; }
    .batch-pct  { font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#8fa0c2; min-width:50px; text-align:right; }
</style>
""", unsafe_allow_html=True)

# Vanta.js FOG — restrained blue ambient field behind the dashboard
components.html("""
<script>
(function() {
    const p = window.parent;
    const doc = p.document;

    // Don't inject twice
    if (doc.getElementById('vanta-root')) return;

    // 1. Set body background deep blue
    doc.body.style.background = '#061431';
    doc.body.style.margin = '0';

    // 2. Make Streamlit layers transparent via injected style
    const styleTag = doc.createElement('style');
    styleTag.id = 'vanta-override-style';
    styleTag.textContent = `
        body, html { background: #061431 !important; }
        [data-testid="stAppViewContainer"],
        .stApp, .main,
        [data-testid="stBottom"],
        [data-testid="stDecoration"] {
            background: transparent !important;
            background-color: transparent !important;
        }
    `;
    doc.head.appendChild(styleTag);

    // 3. Create Vanta target div behind everything
    const vantaDiv = doc.createElement('div');
    vantaDiv.id = 'vanta-root';
    vantaDiv.style.cssText = `
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        z-index: 0;
    `;
    doc.body.prepend(vantaDiv);

    // 4. Load Three.js → then Vanta FOG
    function loadScript(src, cb) {
        const s = doc.createElement('script');
        s.src = src;
        s.onload = cb;
        doc.head.appendChild(s);
    }

    loadScript(
        'https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js',
        function() {
            loadScript(
                'https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.fog.min.js',
                function() {
                    p.VANTA.FOG({
                        el: vantaDiv,
                        mouseControls: true,
                        touchControls: true,
                        gyroControls: false,
                        minHeight: 200,
                        minWidth: 200,
                        scale: 1.00,
                        scaleMobile: 1.00,
                        highlightColor: 0x2d5db3,
                        midtoneColor: 0x173a83,
                        lowlightColor: 0x0a2154,
                        baseColor: 0x061431,
                        blurFactor: 0.62,
                        speed: 0.85,
                        zoom: 0.9
                    });
                }
            );
        }
    );
})();
</script>
""", height=0)


@st.cache_resource
def load_authenticator(model_name: str = DEFAULT_MODEL_NAME):
    model_path = str(BEST_MODEL_PATH) if BEST_MODEL_PATH.exists() else None
    model = build_detector(backbone_name=model_name, pretrained=True, checkpoint_path=model_path, device=DEVICE)
    return MediaAuthenticator(model)


# Dashboard shell and utility navigation
st.markdown("""
<div class="aura-shell">
  <aside class="aura-brand">
    <span class="aura-mark">◈</span><h3>MEDIA DETECTOR</h3>
    <p>AI-GENERATED IMAGE ANALYSIS</p>
    <div class="aura-nav active">⌂ &nbsp; Dashboard</div>
    <div class="aura-status">MODEL &nbsp;EFFICIENTNET-B4</div>
  </aside>
  <main class="aura-main">
    <div class="topline"><span class="eyebrow">AI IMAGE ANALYSIS</span></div>
    <h1 class="walle-title">Synthetic Media<br>Detection</h1>
    <p class="walle-sub">Analyze images to identify authentic photography and AI-generated content.</p>
  </main>
</div>
""", unsafe_allow_html=True)

# Live dashboard indicators
kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="medium")
with kpi1:
    st.markdown('<div class="walle-metric"><span class="dash-icon">◉</span><div class="dash-kpi">Benchmark Accuracy</div><div class="walle-metric-val">87.44%</div><div class="walle-sub" style="margin:4px 0 0">GenImage Val (AUC: 0.9434)</div></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown('<div class="walle-metric"><span class="dash-icon">⌁</span><div class="dash-kpi">Benchmark Dataset</div><div class="walle-metric-val">27,750</div><div class="walle-sub" style="margin:4px 0 0">published image pairs</div></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown('<div class="walle-metric"><span class="dash-icon">◈</span><div class="dash-kpi">External Test (RAISE)</div><div class="walle-metric-val" style="color:#55e6b2">1,000</div><div class="walle-sub" style="margin:4px 0 0">authentic camera captures</div></div>', unsafe_allow_html=True)
with kpi4:
    st.markdown('<div class="walle-metric"><span class="dash-icon">△</span><div class="dash-kpi">External Test (Synth)</div><div class="walle-metric-val" style="color:#ff8498">1,000</div><div class="walle-sub" style="margin:4px 0 0">9 diffusion generators</div></div>', unsafe_allow_html=True)

st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-heading">Start a new investigation</p><p class="walle-sub" style="margin-bottom:13px">Bring an image into the evidence workspace to generate a multi-signal verdict.</p>', unsafe_allow_html=True)

# Evidence workspace
col_left, col_right = st.columns([1.05, 0.95], gap="large")

with col_left:
    st.markdown('<div class="walle-card">', unsafe_allow_html=True)
    st.markdown("##### ◈  Evidence source")

    input_mode = st.radio("Source:", ["Upload Image", "Batch Upload", "Sample Gallery"], horizontal=True, label_visibility="collapsed")

    uploaded_image = None
    file_name = "image.jpg"
    batch_files = []

    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Drop image here (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            file_name = uploaded_file.name

    elif input_mode == "Batch Upload":
        raw_files = st.file_uploader("Upload multiple images (up to 20)", type=["jpg", "jpeg", "png", "webp"],
                                     accept_multiple_files=True, label_visibility="collapsed")
        batch_files = raw_files or []
        if batch_files:
            st.caption(f"**{len(batch_files)} image(s) loaded** — click Analyze All to run batch detection.")
        # Also set single image as the first file for preview
        if batch_files:
            uploaded_image = Image.open(batch_files[0]).convert("RGB")
            file_name = batch_files[0].name

    else:
        sample_files = sorted([f for f in SAMPLES_DIR.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]) if SAMPLES_DIR.exists() else []
        if sample_files:
            sample_names = [f.name for f in sample_files]
            chosen_sample = st.selectbox("Select a sample image:", sample_names)
            sample_path = SAMPLES_DIR / chosen_sample
            uploaded_image = Image.open(sample_path).convert("RGB")
            file_name = chosen_sample

    with st.expander("⚙️ Detection Settings", expanded=False):
        sensitivity = st.slider("Detection Threshold", 0.20, 0.80, 0.45, 0.05,
                                help="Threshold for classifying an image as synthetic AI.")
        st.caption(f"Device: **{DEVICE.upper()}**")

    if uploaded_image is not None:
        st.write("")
        if input_mode == "Batch Upload" and len(batch_files) > 1:
            btn_analyze = st.button("Analyze All →", type="primary", use_container_width=True)
        else:
            btn_analyze = st.button("Analyze evidence  →", type="primary", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="walle-card">', unsafe_allow_html=True)
    st.markdown("##### ◌  Evidence preview")
    if uploaded_image is not None:
        st.image(uploaded_image, caption=f"{file_name} ({uploaded_image.width} × {uploaded_image.height} px)", use_container_width=True)
    else:
        st.info("Upload an image or select a curated sample to begin.")
    st.markdown('</div>', unsafe_allow_html=True)


# ── Batch Analysis Results ────────────────────────────────────────────────────
if input_mode == "Batch Upload" and batch_files and len(batch_files) > 1 and 'btn_analyze' in locals() and btn_analyze:
    authenticator = load_authenticator()
    thresh = sensitivity if 'sensitivity' in locals() else 0.45
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📋 Batch Analysis Results")
    progress_bar = st.progress(0)
    rows_html = ""
    synth_count = auth_count = susp_count = 0

    for i, bf in enumerate(batch_files[:20]):  # cap at 20
        img_b = Image.open(bf).convert("RGB")
        res_b = authenticator.authenticate(img_b, threshold=thresh, run_gradcam=False)
        vc = res_b["verdict_code"]
        sp = res_b["synthetic_probability"]
        badge_cls = "batch-badge-synth" if vc == "SYNTHETIC" else ("batch-badge-auth" if vc == "AUTHENTIC" else "batch-badge-susp")
        badge_txt = "SYNTHETIC" if vc == "SYNTHETIC" else ("AUTHENTIC" if vc == "AUTHENTIC" else "SUSPECT")
        if vc == "SYNTHETIC": synth_count += 1
        elif vc == "AUTHENTIC": auth_count += 1
        else: susp_count += 1
        rows_html += f'<div class="batch-row"><span class="batch-fname">{bf.name}</span><span class="{badge_cls}">{badge_txt}</span><span class="batch-pct">{sp:.1f}%</span></div>'
        progress_bar.progress((i + 1) / min(len(batch_files), 20))

    # Summary pills
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f'<div class="walle-metric"><div class="dash-kpi">Synthetic</div><div class="walle-metric-val" style="color:#fca5a5">{synth_count}</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="walle-metric"><div class="dash-kpi">Authentic</div><div class="walle-metric-val" style="color:#86efac">{auth_count}</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="walle-metric"><div class="dash-kpi">Suspect</div><div class="walle-metric-val" style="color:#fde68a">{susp_count}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(rows_html, unsafe_allow_html=True)


# Results Section
if uploaded_image is not None and ('btn_analyze' in locals() and btn_analyze or 'last_analysis' in st.session_state):
    if 'btn_analyze' in locals() and btn_analyze:
        with st.spinner("Analyzing image..."):
            authenticator = load_authenticator()
            thresh = sensitivity if 'sensitivity' in locals() else 0.45
            results = authenticator.authenticate(uploaded_image, threshold=thresh)
            st.session_state["last_analysis"] = results
            st.session_state["last_img"] = uploaded_image
            st.session_state["last_name"] = file_name
    else:
        results = st.session_state["last_analysis"]
        uploaded_image = st.session_state["last_img"]
        file_name = st.session_state["last_name"]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Detection Results")

    verdict = results["verdict"]
    synth_prob = results["synthetic_probability"]
    confidence = results["confidence"]
    verdict_code = results["verdict_code"]

    v_class = "verdict-synth" if verdict_code == "SYNTHETIC" else ("verdict-auth" if verdict_code == "AUTHENTIC" else "verdict-susp")
    v_icon = "⚠️" if verdict_code == "SYNTHETIC" else ("✅" if verdict_code == "AUTHENTIC" else "🔍")

    st.markdown(f"""
    <div class="verdict-box {v_class}">
        <h2 style="margin:0; font-size:1.45rem; font-weight:800;">{v_icon} {verdict.upper()}</h2>
        <p style="margin:6px 0 0 0; font-size:0.95rem; font-family:'Space Mono',monospace;">{results['verdict_summary']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Animated Confidence Gauge + Metric Cards ──────────────────────────────
    g_col, m_col = st.columns([0.35, 0.65], gap="medium")

    with g_col:
        circ = 2 * 3.14159 * 54   # circumference for r=54
        filled = circ * (synth_prob / 100)
        gauge_color = "#EF4444" if synth_prob >= 60 else ("#FACC15" if synth_prob >= 40 else "#22C55E")
        gauge_word = "SYNTHETIC" if synth_prob >= 60 else ("SUSPECT" if synth_prob >= 40 else "AUTHENTIC")
        st.markdown(f"""
        <div class="gauge-wrap">
          <div class="gauge-label">Synthetic Probability</div>
          <div class="gauge-ring">
            <svg viewBox="0 0 120 120" width="130" height="130">
              <circle class="gauge-track" cx="60" cy="60" r="54"/>
              <circle class="gauge-fill" cx="60" cy="60" r="54"
                stroke="{gauge_color}"
                stroke-dasharray="{filled:.1f} {circ:.1f}"
                stroke-dashoffset="0"/>
            </svg>
            <div class="gauge-center">
              <div class="gauge-pct" style="color:{gauge_color}">{synth_prob:.0f}%</div>
              <div class="gauge-word">{gauge_word}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col:
        k1, k2 = st.columns(2)
        with k1:
            st.markdown(f"""
            <div class="walle-metric">
                <div class="walle-metric-lbl">Confidence</div>
                <div class="walle-metric-val" style="color:#38BDF8;">{confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="walle-metric">
                <div class="walle-metric-lbl">Uncertainty</div>
                <div class="walle-metric-val" style="color:#A8A29E;">±{results['uncertainty_margin']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        k3, k4 = st.columns(2)
        with k3:
            st.markdown(f"""
            <div class="walle-metric">
                <div class="walle-metric-lbl">Authenticity Index</div>
                <div class="walle-metric-val" style="color:#FACC15;">{100.0 - synth_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            # Quick EXIF camera signal pill
            exif_data = extract_exif(uploaded_image, file_name)
            has_cam = exif_data.get("_camera_exif_present", False)
            exif_color = "#86efac" if has_cam else "#fca5a5"
            exif_label = "Present ✓" if has_cam else "Absent ✗"
            st.markdown(f"""
            <div class="walle-metric">
                <div class="walle-metric-lbl">Camera EXIF</div>
                <div class="walle-metric-val" style="color:{exif_color};font-size:1.1rem">{exif_label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    t_cam, t_forensics, t_exif, t_stress, t_report = st.tabs([
        "🔥 Grad-CAM Heatmap",
        "🔬 Signal Breakdown",
        "🗂️ EXIF Metadata",
        "🧪 Perturbation Test",
        "📄 Export PDF"
    ])


    with t_cam:
        st.markdown("#### Grad-CAM Visual Heatmap")
        st.caption("Shows which regions of the image the model focused on when making its decision.")
        
        if results["gradcam"]["overlay_rgb"] is not None:
            c_orig, c_cam = st.columns(2)
            with c_orig:
                st.image(uploaded_image, caption="Original Image", use_container_width=True)
            with c_cam:
                st.image(results["gradcam"]["overlay_rgb"], caption="Attention Heatmap", use_container_width=True)
        else:
            st.info("Heatmap ready.")

    with t_forensics:
        st.markdown("#### Forensic Signal Analysis")
        st.caption("Scores across deep learning, spectral, and noise channels.")
        
        signals = results["signals"]
        c_left, c_right = st.columns([1.1, 0.9])
        
        with c_left:
            for name, key in [
                ("Deep Learning", "deep_learning"),
                ("2D FFT Spectral", "fft_spectral"),
                ("Error Level Analysis", "ela_compression"),
                ("Sensor Noise", "sensor_noise")
            ]:
                score = signals[key]["score"]
                st.write(f"**{name}:** `{score:.1f}%` ({signals[key]['severity']})")
                st.progress(int(np.clip(score, 0, 100)))

        with c_right:
            radar_cats = ['Deep Learning', '2D FFT', 'ELA', 'Sensor Noise']
            radar_vals = [signals['deep_learning']['score'], signals['fft_spectral']['score'], signals['ela_compression']['score'], signals['sensor_noise']['score']]
            fig = go.Figure(data=go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_cats + [radar_cats[0]],
                fill='toself',
                fillcolor='rgba(234, 179, 8, 0.2)',
                line=dict(color='#EAB308', width=2)
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=30, r=30, t=10, b=10),
                height=230
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔍 View FFT Spectrum & ELA Map", expanded=False):
            f1, f2 = st.columns(2)
            with f1:
                st.image(signals["fft_spectral"]["visualization"], caption="2D Fourier Spectrum", use_container_width=True)
            with f2:
                st.image(signals["ela_compression"]["visualization"], caption="Error Level Analysis (ELA 90%)", use_container_width=True)

    with t_exif:
        st.markdown("#### 🗂️ Image Metadata & EXIF Forensics")
        st.caption("AI-generated images typically lack camera EXIF data. Missing manufacturer/lens data is a soft signal of synthetic origin.")

        exif_show = {k: v for k, v in exif_data.items() if not k.startswith("_")}
        has_camera_exif = exif_data.get("_camera_exif_present", False)

        # Forensic interpretation banner
        if has_camera_exif:
            st.success("✅ **Camera EXIF detected** — image contains manufacturer, lens, or timestamp data consistent with a real camera capture.")
        else:
            st.warning("⚠️ **No camera EXIF found** — image lacks Make/Model/DateTimeOriginal fields. This is common in AI-generated images, screenshots, or heavily processed files.")

        # Metadata table
        col_meta, col_sig = st.columns([1.1, 0.9])
        with col_meta:
            st.markdown("**Extracted Metadata Fields:**")
            if exif_show:
                for field, val in exif_show.items():
                    icon = "📷" if field in ["Make","Model","FocalLength","FNumber","ExposureTime","Flash","ISO"] \
                           else ("📅" if "Date" in field else ("📐" if field in ["Resolution","MegaPixels","ExifImageWidth","ExifImageHeight"] else "🔖"))
                    st.markdown(f"`{icon} {field}` &nbsp; **{val}**")
            else:
                st.info("No EXIF fields found in this image.")

        with col_sig:
            st.markdown("**EXIF Signal Summary:**")
            signals_exif = {
                "Camera Manufacturer (Make)": "Make" in exif_show,
                "Camera Model": "Model" in exif_show,
                "Original Capture Date": "DateTimeOriginal" in exif_show,
                "Focal Length / Aperture": any(k in exif_show for k in ["FocalLength","FNumber"]),
                "ISO / Exposure Time": any(k in exif_show for k in ["ISO","ExposureTime"]),
                "Software Tag": "Software" in exif_show,
            }
            for sig_name, present in signals_exif.items():
                dot = "🟢" if present else "🔴"
                st.markdown(f"{dot} {sig_name}")

            st.markdown("<br>", unsafe_allow_html=True)
            exif_score = sum(signals_exif.values())
            st.markdown(f"**EXIF Authenticity Score: `{exif_score}/6`**")
            st.progress(exif_score / 6)
            st.caption("Higher score = more metadata consistent with a real camera.")

    with t_stress:
        st.markdown("#### Perturbation Stress Test")
        st.caption("Simulate real-world compression and blur to test model robustness.")
        
        attack_col, result_col = st.columns(2)
        with attack_col:
            attack_type = st.selectbox("Attack Type:", ["JPEG Compression", "Gaussian Blur", "Downsampling & Resizing"])
            
            if attack_type == "JPEG Compression":
                jq = st.slider("JPEG Quality", 10, 95, 45)
                pert_img = apply_jpeg_compression(uploaded_image, jq)
            elif attack_type == "Gaussian Blur":
                br = st.slider("Blur Radius", 0.5, 5.0, 2.0)
                pert_img = apply_gaussian_blur(uploaded_image, br)
            else:
                sf = st.slider("Scale Factor", 0.2, 0.9, 0.5)
                pert_img = apply_resize_down_up(uploaded_image, sf)
                
            st.image(pert_img, caption=f"Perturbed Image ({attack_type})", use_container_width=True)

        with result_col:
            st.markdown("##### Result on Perturbed Image")
            auth = load_authenticator()
            p_res = auth.authenticate(pert_img, run_gradcam=True)
            
            st.metric("Synthetic Probability", f"{p_res['synthetic_probability']:.1f}%", 
                      delta=f"{p_res['synthetic_probability'] - synth_prob:.1f}%")
            st.write(f"**Verdict:** `{p_res['verdict']}`")
            if p_res["gradcam"]["overlay_rgb"] is not None:
                st.image(p_res["gradcam"]["overlay_rgb"], caption="Heatmap on Perturbed Image", use_container_width=True)

    with t_report:
        st.markdown("#### Export PDF Report")
        st.caption("Download a PDF report with SHA-256 hash and evidence breakdown.")
        
        pdf_path = REPORTS_DIR / "Forensic_Report.pdf"
        if st.button("📥 Generate PDF Report", type="secondary"):
            with st.spinner("Generating report..."):
                generate_forensic_pdf(uploaded_image, results, pdf_path, filename=file_name)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=f.read(),
                        file_name="Synthetic_Media_Report.pdf",
                        mime="application/pdf"
                    )
            st.success("Report ready!")

# Benchmark Section
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("📊 Academic Benchmark Suite (GenImage, Synthbuster/RAISE & Robustness)", expanded=False):
    st.markdown("#### 1. Training Dynamics & Loss Convergence")
    th_file = REPORTS_DIR / "training_history.png"
    if th_file.exists():
        st.image(str(th_file), caption="Training & Validation Loss / Accuracy over Epochs (EfficientNet-B4 on RTX 4050 GPU)", use_container_width=True)
        
    st.markdown("#### 2. Internal Test Benchmark (GenImage — 3,250 Images)")
    b1, b2 = st.columns(2)
    with b1:
        cm_file = REPORTS_DIR / "confusion_matrix.png"
        if cm_file.exists():
            st.image(str(cm_file), caption="Internal Test Confusion Matrix (GenImage)", use_container_width=True)
    with b2:
        roc_file = REPORTS_DIR / "roc_curve.png"
        if roc_file.exists():
            st.image(str(roc_file), caption="Internal Test ROC-AUC Curve (GenImage)", use_container_width=True)

    st.markdown("#### 3. External Generalization Benchmark (Synthbuster + RAISE — 2,000 Images)")
    e1, e2 = st.columns(2)
    with e1:
        ext_cm = REPORTS_DIR / "external_synthbuster_raise_confusion_matrix.png"
        if ext_cm.exists():
            st.image(str(ext_cm), caption="External Generalization Confusion Matrix (RAISE vs Synthbuster)", use_container_width=True)
    with e2:
        ext_roc = REPORTS_DIR / "external_synthbuster_raise_roc_curve.png"
        if ext_roc.exists():
            st.image(str(ext_roc), caption="External Generalization ROC-AUC Curve", use_container_width=True)

    st.markdown("#### 4. Perturbation & Robustness Degradation Benchmarks")
    rob_file = REPORTS_DIR / "robustness_benchmarks.png"
    if rob_file.exists():
        st.image(str(rob_file), caption="Model Accuracy under Compression, Blur, Downsampling, and Additive Noise", use_container_width=True)
