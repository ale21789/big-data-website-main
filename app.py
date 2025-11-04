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
# 1. MODEL CONFIGURATION & SURVEY DATA
# =============================================================================

# BASE VALUES (Generated from the 5000-image random sample)
POPULATION_STATS = {
    'brightness': (0.3344, 0.7716), 'contrast': (0.1896, 0.3642),
    'saturation': (0.088, 0.5489), 'red': (0.3294, 0.7412),
    'green': (0.274, 0.672), 'blue': (0.2312, 0.627),
    'colorfulness': (0.0538, 0.2505), 'sharpness': (0.002, 0.0515),
    'blur': (0.951, 0.998), 'edges_density': (0.1118, 0.3804),
    'thirds': (0.0463, 0.4079), 'simplicity': (0.7088, 0.9203),
    'warm_ratio': (0.0245, 0.7634),
}

# WEIGHT RUBRIC (The paper's findings, with corrections)
W = {
    "Openness":         {"sharpness": +0.9, "contrast": +0.7, "simplicity": +0.6, "is_gray": +0.5, "blue": +0.3, "saturation": +0.4, "warm_ratio": -0.2, "colorfulness": -0.3, "smile_level": -0.4},
    "Conscientiousness":{"brightness": +0.6, "simplicity": +0.3, "sharpness": -0.4, "smile_level": +0.7, "blur": +0.2, "group_photo": -0.3},
    "Extraversion":     {"colorfulness": +0.8, "saturation": +0.7, "brightness": +0.5, "warm_ratio": +0.4, "red": +0.3, "group_photo": +0.6, "smile_level": +0.3, "simplicity": -0.2},
    "Agreeableness":    {"warm_ratio": +0.7, "brightness": +0.5, "saturation": +0.3, "red": +0.4, "green": +0.3, "smile_level": +0.8, "blur": +0.4, "sharpness": -0.5, "simplicity": -0.2},
    "Neuroticism":      {"is_gray": +0.6, "simplicity": +0.4, "blur": +0.3, "sharpness": -0.4, "colorfulness": -0.6, "saturation": -0.5, "brightness": -0.4, "smile_level": -0.7},
}

# IPIP-50 SURVEY QUESTIONS
# Format: (Question Text, Trait, Keying [+1 for normal, -1 for reversed])
IPIP_QUESTIONS = [
    ("I am the life of the party.", "E", 1), ("I feel little concern for others.", "A", -1), ("I am always prepared.", "C", 1), ("I get stressed out easily.", "N", 1), ("I have a rich vocabulary.", "O", 1),
    ("I don't talk a lot.", "E", -1), ("I am interested in people.", "A", 1), ("I leave my belongings around.", "C", -1), ("I am relaxed most of the time.", "N", -1), ("I have difficulty understanding abstract ideas.", "O", -1),
    ("I feel comfortable around people.", "E", 1), ("I insult people.", "A", -1), ("I pay attention to details.", "C", 1), ("I worry about things.", "N", 1), ("I have a vivid imagination.", "O", 1),
    ("I keep in the background.", "E", -1), ("I sympathize with others' feelings.", "A", 1), ("I make a mess of things.", "C", -1), ("I seldom feel blue.", "N", -1), ("I am not interested in abstract ideas.", "O", -1),
    ("I start conversations.", "E", 1), ("I am not interested in other people's problems.", "A", -1), ("I get chores done right away.", "C", 1), ("I am easily disturbed.", "N", 1), ("I have excellent ideas.", "O", 1),
    ("I have little to say.", "E", -1), ("I have a soft heart.", "A", 1), ("I often forget to put things back in their proper place.", "C", -1), ("I get upset easily.", "N", 1), ("I do not have a good imagination.", "O", -1),
    ("I talk to a lot of different people at parties.", "E", 1), ("I am not really interested in others.", "A", -1), ("I like order.", "C", 1), ("I change my mood a lot.", "N", 1), ("I am quick to understand things.", "O", 1),
    ("I don't like to draw attention to myself.", "E", -1), ("I take time out for others.", "A", 1), ("I shirk my duties.", "C", -1), ("I have frequent mood swings.", "N", 1), ("I use difficult words.", "O", 1),
    ("I don't mind being the center of attention.", "E", 1), ("I feel others' emotions.", "A", 1), ("I follow a schedule.", "C", 1), ("I get irritated easily.", "N", 1), ("I spend time reflecting on things.", "O", 1),
    ("I am quiet around strangers.", "E", -1), ("I make people feel at ease.", "A", 1), ("I am exacting in my work.", "C", 1), ("I often feel blue.", "N", 1), ("I am full of ideas.", "O", 1)
]
TRAIT_MAP = {"E": "Extraversion", "A": "Agreeableness", "C": "Conscientiousness", "N": "Neuroticism", "O": "Openness"}

# =============================================================================
# 2. FEATURE EXTRACTION & SCORING FUNCTIONS
# =============================================================================

# --- Feature Extraction Utils ---
@st.cache_data
def extract_features(uploaded_file_bytes) -> Dict[str, Union[str, float]]:
    img = load_image(io.BytesIO(uploaded_file_bytes))
    if img is None: return None
    npimg = to_np(img)
    gray_img = np.mean(npimg, axis=-1)
    hsv_img = rgb_to_hsv(npimg)
    sharpness = laplacian_variance(gray_img)
    gray_uint8 = (gray_img * 255).astype(np.uint8)
    grad_x = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(grad_x**2 + grad_y**2) / 255.0
    return {
        "is_gray": 1.0 if is_grayscale(npimg) else 0.0, "brightness": float(np.mean(hsv_img[..., 2])),
        "contrast": float(np.std(hsv_img[..., 2])), "saturation": float(np.mean(hsv_img[..., 1])),
        "red": float(np.mean(npimg[..., 0])), "green": float(np.mean(npimg[..., 1])),
        "blue": float(np.mean(npimg[..., 2])), "colorfulness": colorfulness_hs(npimg),
        "sharpness": sharpness, "blur": 1.0 / (1.0 + sharpness) if sharpness > 0 else 1.0,
        "edges_density": float(np.mean(edge_mag)), "thirds": thirds_alignment(edge_mag),
        "simplicity": simplicity(npimg), "warm_ratio": warm_ratio(hsv_img),
    }

# (Helper functions for extract_features are placed here to keep the code clean)
def load_image(uploaded_file, max_dim: int = 512) -> Image.Image:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale_factor = max_dim / max(w, h)
            img = img.resize((int(w * scale_factor), int(h * scale_factor)), Image.Resampling.BICUBIC)
        return img
    except Exception: return None
def to_np(img: Image.Image) -> np.ndarray: return np.asarray(img).astype(np.float32) / 255.0
def is_grayscale(img_np: np.ndarray, thr: float = 0.01) -> bool: return float(np.mean(np.std(img_np, axis=-1))) < thr
def colorfulness_hs(img_np: np.ndarray) -> float:
    R, G, B = img_np[..., 0], img_np[..., 1], img_np[..., 2]
    rg, yb = np.abs(R - G), np.abs(0.5 * (R + G) - B)
    return float(np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))
def laplacian_variance(img_np_gray: np.ndarray) -> float: return cv2.Laplacian((img_np_gray * 255).astype(np.uint8), cv2.CV_64F).var()
def thirds_alignment(edge_mag: np.ndarray) -> float:
    H, W = edge_mag.shape
    if H == 0 or W == 0: return 0.5
    yy, xx = np.mgrid[0:H, 0:W]
    weights = edge_mag + 1e-8
    cx, cy = float(np.sum(xx * weights)) / np.sum(weights), float(np.sum(yy * weights)) / np.sum(weights)
    pts = [(W/3, H/3), (2*W/3, H/3), (W/3, 2*H/3), (2*W/3, 2*H/3)]
    min_dist = min(math.hypot(cx - x, cy - y) for x, y in pts)
    diag = math.hypot(W/6, H/6)
    return float(max(0.0, min(1.0, 1.0 - (min_dist / (diag + 1e-8)))))
def simplicity(img_np: np.ndarray, bins: int = 8) -> float:
    q_img = (img_np * bins).astype(int)
    q_img[q_img == bins] = bins - 1
    return float(1.0 - np.unique(q_img.reshape(-1, 3), axis=0).shape[0] / (bins**3))
def warm_ratio(hsv_img: np.ndarray) -> float:
    h, s, v = hsv_img[..., 0], hsv_img[..., 1], hsv_img[..., 2]
    return float(np.mean(((h < 0.167) | (h > 0.833)) & (s > 0.2) & (v > 0.2)))

# --- Normalization and Scoring ---
def absolute_normalize(col: pd.Series) -> pd.Series:
    feature_name = col.name
    if feature_name not in POPULATION_STATS: return col
    p05, p95 = POPULATION_STATS[feature_name]
    if (p95 - p05) < 1e-6: return pd.Series([0.5] * len(col), index=col.index)
    return ((col - p05) / (p95 - p05)).clip(0, 1)

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
def plot_radar_chart(scores_row: pd.Series, title: str) -> Image.Image:
    traits = list(W.keys())
    vals = scores_row[traits].values.tolist()
    N = len(vals)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals += vals[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100, subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits, size=11)
    ax.set_rlabel_position(0)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], color="grey", size=8)
    ax.set_ylim(0, 100)
    ax.plot(angles, vals, linewidth=2, linestyle='solid', color='#1f77b4')
    ax.fill(angles, vals, alpha=0.1, color='#1f77b4')
    ax.set_title(title, size=14, y=1.1)
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# =============================================================================
# 3. STREAMLIT PAGE RENDERERS
# =============================================================================

def render_image_analyzer():
    st.header("Upload a Profile Picture")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("Your Image")
        _ , img_col, _ = st.columns([1, 2, 1])
        with img_col:
            st.image(uploaded_file, caption="Uploaded Profile Picture", width=300)
        
        st.subheader("Personality Profile Analysis")
        with st.spinner("Analyzing visual features..."):
            # --- FIX #1: Pass the raw bytes to the cached function ---
            features_dict = extract_features(uploaded_file.getvalue())
            
            if features_dict:
                # Add the image name back in for the DataFrame
                features_dict['image'] = uploaded_file.name
                df = pd.DataFrame([features_dict])
                
                df["group_photo"], df["smile_level"] = 0.0, 0.0
                use_cols = sorted(list(set(k for v in W.values() for k in v.keys())))
                
                norm_df = df[use_cols].apply(absolute_normalize)
                if 'smile_level' in norm_df.columns: 
                    norm_df['smile_level'] /= 2.0
                
                # --- FIX #2: Correctly initialize the DataFrame for a single row ---
                scores = pd.DataFrame({"image": [uploaded_file.name]}, index=[0])
                
                for t in W.keys(): 
                    scores[t] = score_trait(t, norm_df)
                
                tab1, tab2, tab3 = st.tabs(["📊 Personality Profile", "🔢 Score Details", "⚙️ Visual Features"])
                with tab1:
                    chart_as_image = plot_radar_chart(scores.iloc[0], "Predicted Profile (from Image)")
                    st.image(chart_as_image, width=400)
                with tab2:
                    display_scores = scores.drop(columns=['image']).T.rename(columns={0: "Score"})
                    st.dataframe(display_scores, use_container_width=True)
                with tab3:
                    st.dataframe(df.drop(columns=['image']).T.rename(columns={0: "Value"}).round(3), use_container_width=True)
            else:
                st.error("Could not process the uploaded image. Please try another one.")
def render_personality_test():
    st.header("Take the 50-Item IPIP Personality Survey")
    st.info("Answer these 50 questions to get a baseline personality profile. Your answers are not stored.")

    # --- Initialize or retrieve the image scores from session state ---
    if 'image_scores' not in st.session_state:
        st.session_state.image_scores = None

    # Initialize session state for the survey
    if 'survey_page' not in st.session_state:
        st.session_state.survey_page = 0
        st.session_state.answers = {}
        st.session_state.survey_complete = False

    if st.session_state.survey_complete:
        # --- STAGE 2: Display Results & Comparison ---
        st.success("Survey Complete! Here are your results.")
        
        # Calculate survey scores
        scores = {"E": 0, "A": 0, "C": 0, "N": 0, "O": 0}
        for i, (_, trait, keying) in enumerate(IPIP_QUESTIONS):
            answer = st.session_state.answers.get(f"q_{i}", 3)
            score = answer if keying == 1 else 6 - answer
            scores[trait] += score
        
        final_scores = {}
        for trait_code, raw_score in scores.items():
            normalized_score = (raw_score - 10) / 40 * 100
            final_scores[TRAIT_MAP[trait_code]] = round(normalized_score, 1)
        
        survey_scores_df = pd.DataFrame([final_scores])
        
        # --- NEW: Comparison View ---
        st.subheader("Comparison: Survey vs. Image Analysis")

        if st.session_state.image_scores is None:
            st.warning("You haven't analyzed an image yet. Go to the 'Analyze a Profile Picture' tab first to see a comparison.")
            
            # Display only the survey results
            chart = plot_radar_chart(survey_scores_df.iloc[0], "Your Survey Results")
            st.image(chart, width=400)
            st.dataframe(survey_scores_df.T.rename(columns={0: "Score (0-100)"}), use_container_width=True)

        else:
            # Create side-by-side charts
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Your Survey Results")
                survey_chart = plot_radar_chart(survey_scores_df.iloc[0], "Survey Profile")
                st.image(survey_chart)
            with col2:
                st.markdown("##### Image Analysis Results")
                image_chart = plot_radar_chart(st.session_state.image_scores.iloc[0], "Image Profile")
                st.image(image_chart)

            # Create a comparison dataframe
            comparison_df = survey_scores_df.T.rename(columns={0: "Survey Score"})
            comparison_df["Image Score"] = st.session_state.image_scores.drop(columns=['image']).T
            st.dataframe(comparison_df, use_container_width=True)

        if st.button("Take Survey Again"):
            st.session_state.survey_page = 0
            st.session_state.answers = {}
            st.session_state.survey_complete = False
            st.rerun()
    else:
        # --- STAGE 1: Display Questions ---
        QUESTIONS_PER_PAGE = 10
        start_idx = st.session_state.survey_page * QUESTIONS_PER_PAGE
        end_idx = start_idx + QUESTIONS_PER_PAGE
        
        st.progress((start_idx / len(IPIP_QUESTIONS)), text=f"Page {st.session_state.survey_page + 1} of {len(IPIP_QUESTIONS) // QUESTIONS_PER_PAGE}")

        options = ["Very Inaccurate", "Moderately Inaccurate", "Neither Accurate Nor Inaccurate", "Moderately Accurate", "Very Accurate"]
        
        for i in range(start_idx, end_idx):
            q_text, _, _ = IPIP_QUESTIONS[i]
            # --- CHANGE 1: Set horizontal=False for mobile-friendly vertical layout ---
            answer = st.radio(
                f"**{i+1}.** {q_text}", options, index=2, horizontal=False, 
                key=f"q_{i}", help="Select the option that best describes you."
            )
            st.session_state.answers[f"q_{i}"] = options.index(answer) + 1
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        if st.session_state.survey_page > 0:
            if col1.button("⬅️ Previous"):
                st.session_state.survey_page -= 1
                st.rerun()
        
        if st.session_state.survey_page < (len(IPIP_QUESTIONS) // QUESTIONS_PER_PAGE - 1):
            if col3.button("Next ➡️"):
                st.session_state.survey_page += 1
                st.rerun()
        else:
            if col3.button("✅ Submit & See Results", type="primary"):
                st.session_state.survey_complete = True
                st.rerun()

# =============================================================================
# 4. MAIN APP LAYOUT
# =============================================================================

st.set_page_config(layout="wide")
st.title("Profile Picture Personality Profiler")

st.warning("""
**Disclaimer:** we used the findings from the paper to predict the behaviour. This is not the same as a psychological assessment.
""", icon="⚠️")

# Main app navigation
tab1, tab2 = st.tabs(["🖼️ Analyze a Profile Picture", "📝 Take the Personality Survey"])

with tab1:
    render_image_analyzer()

with tab2:
    render_personality_test()