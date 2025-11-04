import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
import cv2
from typing import Dict, Union
import math
import io 

# =============================================================================
# 1. MODEL CONFIGURATION (The Brains of the App)
# =============================================================================

# BASE VALUES (Generated from the 5000-image random sample)
# This dictionary is the calibrated "ruler" for the model.
POPULATION_STATS = {
    'brightness': (0.3344, 0.7716),
    'contrast': (0.1896, 0.3642),
    'saturation': (0.088, 0.5489),
    'red': (0.3294, 0.7412),
    'green': (0.274, 0.672),
    'blue': (0.2312, 0.627),
    'colorfulness': (0.0538, 0.2505),
    'sharpness': (0.002, 0.0515),
    'blur': (0.951, 0.998),
    'edges_density': (0.1118, 0.3804),
    'thirds': (0.0463, 0.4079),
    'simplicity': (0.7088, 0.9203),
    'warm_ratio': (0.0245, 0.7634),
}

# WEIGHT RUBRIC (The paper's findings, with corrections)
W = {
    "Openness":         {"sharpness": +0.9, "contrast": +0.7, "simplicity": +0.6, "is_gray": +0.5, "blue": +0.3,
                         "saturation": +0.4, "warm_ratio": -0.2, "colorfulness": -0.3, "smile_level": -0.4},
    "Conscientiousness":{"brightness": +0.6, "simplicity": +0.3, "sharpness": -0.4, "smile_level": +0.7,
                         "blur": +0.2, "group_photo": -0.3},
    "Extraversion":     {"colorfulness": +0.8, "saturation": +0.7, "brightness": +0.5, "warm_ratio": +0.4,
                         "red": +0.3, "group_photo": +0.6, "smile_level": +0.3, "simplicity": -0.2},
    "Agreeableness":    {"warm_ratio": +0.7, "brightness": +0.5, "saturation": +0.3, "red": +0.4,
                         "green": +0.3, "smile_level": +0.8, "blur": +0.4, "sharpness": -0.5, "simplicity": -0.2},
    "Neuroticism":      {"is_gray": +0.6, "simplicity": +0.4, "blur": +0.3,
                         "sharpness": -0.4, "colorfulness": -0.6, "saturation": -0.5, "brightness": -0.4, "smile_level": -0.7},
}

# =============================================================================
# 2. FEATURE EXTRACTION & SCORING FUNCTIONS
# =============================================================================

# --- Feature Extraction Utils ---
def load_image(uploaded_file, max_dim: int = 512) -> Image.Image:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale_factor = max_dim / max(w, h)
            img = img.resize((int(w * scale_factor), int(h * scale_factor)), Image.Resampling.BICUBIC)
        return img
    except Exception:
        return None

def to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32) / 255.0

def is_grayscale(img_np: np.ndarray, thr: float = 0.01) -> bool:
    channel_std_dev = np.std(img_np, axis=-1)
    return float(np.mean(channel_std_dev)) < thr

def colorfulness_hs(img_np: np.ndarray) -> float:
    R, G, B = img_np[..., 0], img_np[..., 1], img_np[..., 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

def laplacian_variance(img_np_gray: np.ndarray) -> float:
    gray_uint8 = (img_np_gray * 255).astype(np.uint8)
    return cv2.Laplacian(gray_uint8, cv2.CV_64F).var()

def thirds_alignment(edge_mag: np.ndarray) -> float:
    H, W = edge_mag.shape
    if H == 0 or W == 0: return 0.5
    yy, xx = np.mgrid[0:H, 0:W]
    weights = edge_mag + 1e-8
    center_of_mass_x = float(np.sum(xx * weights)) / np.sum(weights)
    center_of_mass_y = float(np.sum(yy * weights)) / np.sum(weights)
    thirds_points = [(W/3, H/3), (2*W/3, H/3), (W/3, 2*H/3), (2*W/3, 2*H/3)]
    min_dist = min(math.hypot(center_of_mass_x - x, center_of_mass_y - y) for x, y in thirds_points)
    diag = math.hypot(W/6, H/6)
    score = 1.0 - (min_dist / (diag + 1e-8))
    return float(max(0.0, min(1.0, score)))

def simplicity(img_np: np.ndarray, bins: int = 8) -> float:
    quantized_img = (img_np * bins).astype(int)
    quantized_img[quantized_img == bins] = bins - 1
    num_unique_colors = np.unique(quantized_img.reshape(-1, 3), axis=0).shape[0]
    return float(1.0 - num_unique_colors / (bins**3))

def warm_ratio(hsv_img: np.ndarray) -> float:
    h, s, v = hsv_img[..., 0], hsv_img[..., 1], hsv_img[..., 2]
    is_warm = ((h < 0.167) | (h > 0.833)) & (s > 0.2) & (v > 0.2)
    return float(np.mean(is_warm))

def extract_features(uploaded_file) -> Dict[str, Union[str, float]]:
    img = load_image(uploaded_file)
    if img is None: return None
    npimg = to_np(img)
    gray_img = np.mean(npimg, axis=-1)
    hsv_img = rgb_to_hsv(npimg)
    sharpness = laplacian_variance(gray_img)
    # For edge detection, we need to convert to 8-bit gray
    gray_uint8 = (gray_img * 255).astype(np.uint8)
    grad_x = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(grad_x**2 + grad_y**2) / 255.0
    
    return {
        "image": uploaded_file.name, "is_gray": 1.0 if is_grayscale(npimg) else 0.0,
        "brightness": float(np.mean(hsv_img[..., 2])), "contrast": float(np.std(hsv_img[..., 2])),
        "saturation": float(np.mean(hsv_img[..., 1])), "red": float(np.mean(npimg[..., 0])),
        "green": float(np.mean(npimg[..., 1])), "blue": float(np.mean(npimg[..., 2])),
        "colorfulness": colorfulness_hs(npimg), "sharpness": sharpness,
        "blur": 1.0 / (1.0 + sharpness) if sharpness > 0 else 1.0,
        "edges_density": float(np.mean(edge_mag)), "thirds": thirds_alignment(edge_mag),
        "simplicity": simplicity(npimg), "warm_ratio": warm_ratio(hsv_img),
    }

# --- Normalization and Scoring ---
def absolute_normalize(col: pd.Series) -> pd.Series:
    feature_name = col.name
    if feature_name not in POPULATION_STATS:
        return col
    p05, p95 = POPULATION_STATS[feature_name]
    if (p95 - p05) < 1e-6:
        return pd.Series([0.5] * len(col), index=col.index)
    scaled_col = (col - p05) / (p95 - p05)
    return scaled_col.clip(0, 1)

def score_trait(trait: str, norm_df: pd.DataFrame) -> pd.Series:
    score = pd.Series(np.zeros(len(norm_df)), index=norm_df.index)
    total_weight = 0
    for feature, weight in W[trait].items():
        if feature in norm_df.columns:
            score += weight * norm_df[feature].values
            total_weight += abs(weight)
    if total_weight > 0:
        final_score = 50 * (score / total_weight) + 50
    else:
        final_score = pd.Series([50.0] * len(norm_df), index=norm_df.index)
    return final_score.clip(0, 100).round(1)

# --- Visualization ---
def plot_radar_chart(scores_row: pd.Series) -> Image.Image:
    """
    Creates a radar chart and returns it as a PIL Image object.
    """
    traits = list(W.keys())
    vals = scores_row[traits].values.tolist()
    N = len(vals)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals += vals[:1]
    angles += angles[:1]
    
    # Create the plot with a higher resolution
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150, subplot_kw=dict(polar=True))  # Increased dpi
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits, size=12)  # Increased font size
    ax.set_rlabel_position(0)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], color="grey", size=10)  # Increased font size
    ax.set_ylim(0, 100)
    ax.plot(angles, vals, linewidth=2, linestyle='solid', color='#1f77b4')
    ax.fill(angles, vals, alpha=0.1, color='#1f77b4')
    
    # Save the plot to an in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # Close the figure to free up memory
    buf.seek(0)  # Rewind the buffer to the beginning
    
    # Load the image from the buffer and return it
    chart_image = Image.open(buf)
    return chart_image
# =============================================================================
# 3. STREAMLIT WEB APP INTERFACE
# =============================================================================

st.set_page_config(layout="wide")

st.title("🖼️ Profile Picture Personality Profiler")

st.markdown("""
This web app analyzes a profile picture to estimate a Big Five personality profile. It's based on the research paper **"Analyzing Personality through Social Media Profile Picture Choice"** (ICWSM 2016).

**How it works:**
1.  You upload an image.
2.  The app extracts visual features (colors, composition, aesthetics).
3.  These features are scored against a model calibrated on a large dataset of real-world selfies.
4.  The result is a personality profile based on the visual stereotypes discovered in the research.

---
""")

st.warning("""
**Disclaimer:** we used the findings from the paper to predict the behaviour. This is not the same as a psychological assessment.
""", icon="⚠️")

uploaded_file = st.file_uploader(
    "Choose your profile picture...", 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Image")
        image = Image.open(uploaded_file)
        # --- CHANGE 1: Set a fixed pixel width for the image ---
        # The 'width' parameter directly controls the display size in pixels.
        st.image(image, caption="Uploaded Profile Picture", width=400) # You can adjust this value

    with col2:
        st.subheader("Personality Profile Analysis")
        with st.spinner("Analyzing visual features..."):
            # --- Run the full analysis pipeline ---
            features_dict = extract_features(uploaded_file)
            
            if features_dict:
                df = pd.DataFrame([features_dict])
                df["group_photo"] = 0.0 
                df["smile_level"] = 0.0
                
                use_cols = list(W["Openness"].keys()) + list(W["Conscientiousness"].keys()) + list(W["Extraversion"].keys()) + list(W["Agreeableness"].keys()) + list(W["Neuroticism"].keys())
                use_cols = sorted(list(set(use_cols)))
                
                norm_df = df[use_cols].apply(absolute_normalize)
                if 'smile_level' in norm_df.columns:
                    norm_df['smile_level'] = norm_df['smile_level'] / 2.0

                scores = pd.DataFrame({"image": df["image"]})
                for t in W.keys():
                    scores[t] = score_trait(t, norm_df)

                # --- CHANGE 2: Remove 'use_container_width=True' from the chart ---
                # This allows the chart to render at the constant size defined by 'figsize' in the plot_radar_chart function.
                fig = plot_radar_chart(scores.iloc[0])
                chart_as_image = plot_radar_chart(scores.iloc[0])
                # We display it using st.image with a fixed width, just like the profile picture
                st.image(chart_as_image, width=450)

                st.markdown("**Personality Scores (0-100):**")
                display_scores = scores.drop(columns=['image']).T
                display_scores.columns = ["Score"]
                st.dataframe(display_scores)

                with st.expander("See Extracted Visual Features"):
                    st.dataframe(df.drop(columns=['image']).T.rename(columns={0: "Value"}).round(3))
            else:
                st.error("Could not process the uploaded image. Please try another one.")