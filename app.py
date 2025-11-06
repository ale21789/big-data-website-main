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
from deepface import DeepFace

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

# WEIGHT RUBRIC (Hybrid model with facial emotions and user-provided gender)
W = {
    "Openness": {
        "is_gray": +0.5, "sharpness": +0.3, "contrast": +0.2, "blue": +0.3, "anger": +0.4, "sadness": +0.2,
        "saturation": -0.2, "colorfulness": -0.3, "smiling": -0.9, "joy": -0.9, "multiple_faces": -1.0,
        "is_not_face": +0.6
    },
    "Conscientiousness": {
        "one_face": +1.0, "smiling": +1.9, "joy": +1.8, "brightness": +0.3,
        "is_not_face": -1.2, "multiple_faces": -0.4, "anger": -0.8, "sadness": -0.5
    },
    "Extraversion": {
        "multiple_faces": +0.6, "smiling": +0.5, "joy": +0.6, "colorfulness": +0.8, "saturation": +0.7,
        "brightness": +0.5, "warm_ratio": +0.4, "red": +0.3,
        "is_not_face": -1.1, "gender_female": +0.2 # User-provided gender
    },
    "Agreeableness": {
        "smiling": +1.5, "joy": +1.4, "warm_ratio": +0.7, "brightness": +0.5, "saturation": +0.3,
        "red": +0.4, "green": +0.3, "blur": +0.4,
        "sharpness": -0.5, "anger": -0.6, "is_not_face": -0.7, "gender_female": +0.3 # User-provided gender
    },
    "Neuroticism": {
        "is_not_face": +0.7, "anger": +0.6, "sadness": +0.3, "is_gray": +0.6, "simplicity": +0.4,
        "blur": +0.3,
        "smiling": -1.0, "joy": -1.1, "colorfulness": -0.6, "saturation": -0.5, "brightness": -0.4
    },
}

# IPIP-50 SURVEY QUESTIONS
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

# --- Helper Functions ---
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

# --- Main Feature Extraction Functions ---
@st.cache_data
def extract_facial_features(image_bytes):
    """
    Extracts facial emotions and face count locally using DeepFace.
    It IGNORES the model's age and gender predictions.
    """
    try:
        img_arr = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        analysis = DeepFace.analyze(img_arr, actions=['emotion'], enforce_detection=False, silent=True)
        
        if not analysis or isinstance(analysis, list) and len(analysis) == 0:
             return {
                 "num_faces": 0, "is_not_face": 1.0, "one_face": 0.0, "multiple_faces": 0.0,
                 "smiling": 0.0, "joy": 0.0, "sadness": 0.0, "anger": 0.0
             }
             
        res = analysis[0]
        emotions = res['emotion']
        
        return {
            "num_faces": 1, "is_not_face": 0.0, "one_face": 1.0, "multiple_faces": 0.0,
            "smiling": emotions['happy'] / 100.0, "joy": emotions['happy'] / 100.0,
            "sadness": emotions['sad'] / 100.0, "anger": emotions['angry'] / 100.0,
        }
    except Exception as e:
        return {
            "num_faces": 0, "is_not_face": 1.0, "one_face": 0.0, "multiple_faces": 0.0,
            "smiling": 0.0, "joy": 0.0, "sadness": 0.0, "anger": 0.0
        }

@st.cache_data
def extract_all_features(uploaded_file_bytes) -> Dict[str, Union[str, float]]:
    # 1. Extract aesthetic features
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
    
    aesthetic_features = {
        "is_gray": 1.0 if is_grayscale(npimg) else 0.0, "brightness": float(np.mean(hsv_img[..., 2])),
        "contrast": float(np.std(hsv_img[..., 2])), "saturation": float(np.mean(hsv_img[..., 1])),
        "red": float(np.mean(npimg[..., 0])), "green": float(np.mean(npimg[..., 1])),
        "blue": float(np.mean(npimg[..., 2])), "colorfulness": colorfulness_hs(npimg),
        "sharpness": sharpness, "blur": 1.0 / (1.0 + sharpness) if sharpness > 0 else 1.0,
        "edges_density": float(np.mean(edge_mag)), "thirds": thirds_alignment(edge_mag),
        "simplicity": simplicity(npimg), "warm_ratio": warm_ratio(hsv_img),
    }

    # 2. Extract facial features
    facial_features = extract_facial_features(uploaded_file_bytes)

    # 3. Combine both dictionaries
    return {**aesthetic_features, **facial_features}

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
def plot_radar_chart(scores1: pd.Series, label1: str, scores2: pd.Series = None, label2: str = None) -> Image.Image:
    traits = list(W.keys())
    N = len(traits)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100, subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits, size=11)
    ax.set_rlabel_position(0)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], color="grey", size=8)
    ax.set_ylim(0, 100)
    vals1 = scores1[traits].values.tolist()
    vals1 += vals1[:1]
    ax.plot(angles, vals1, linewidth=2, linestyle='solid', color='#1f77b4', label=label1)
    ax.fill(angles, vals1, alpha=0.1, color='#1f77b4')
    if scores2 is not None:
        vals2 = scores2[traits].values.tolist()
        vals2 += vals2[:1]
        ax.plot(angles, vals2, linewidth=2, linestyle='solid', color='#d62728', label=label2)
        ax.fill(angles, vals2, alpha=0.1, color='#d62728')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# =============================================================================
# 3. STREAMLIT PAGE RENDERERS
# =============================================================================

def render_image_analyzer():
    st.header("Upload a Profile Picture & Enter Your Info")
    
    # --- NEW: User Input for Demographics ---
    # We create two columns to place the inputs side-by-side.
    gender = st.radio("Your Gender", ["Female", "Male"], horizontal=True)

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("Your Image")
        _ , img_col, _ = st.columns([1, 2, 1])
        with img_col:
            st.image(uploaded_file, caption="Uploaded Profile Picture", width=300)
        
        st.subheader("Personality Profile Analysis")
        with st.spinner("Analyzing visual features..."):
            features_dict = extract_features(uploaded_file.getvalue())
            
            if features_dict:
                features_dict['image'] = uploaded_file.name
                df = pd.DataFrame([features_dict], index=[0])
                
                # --- NEW: Add user-provided gender to the dataframe ---
                # We create a binary feature for gender, as this is how models work best.
                df['gender_female'] = 1.0 if gender == "Female" else 0.0
                
                # Get all unique feature names from the weight dictionary
                all_feature_keys = set()
                for trait_weights in W.values():
                    all_feature_keys.update(trait_weights.keys())
                
                # Ensure all required columns exist in the dataframe, filling with 0 if not
                for key in all_feature_keys:
                    if key not in df.columns:
                        df[key] = 0.0
                
                norm_df = df[list(all_feature_keys)].apply(absolute_normalize)
                
                scores = pd.DataFrame({"image": [uploaded_file.name]}, index=[0])
                for t in W.keys(): 
                    scores[t] = score_trait(t, norm_df)
                
                st.session_state.image_scores = scores.copy()
                
                tab1, tab2, tab3 = st.tabs(["📊 Personality Profile", "🔢 Score Details", "⚙️ Visual Features"])
                with tab1:
                    chart_as_image = plot_radar_chart(scores1=scores.iloc[0], label1="Image AI Prediction")
                    st.image(chart_as_image, width=400)
                with tab2:
                    display_scores = scores.drop(columns=['image']).T.rename(columns={0: "Score"})
                    st.dataframe(display_scores, use_container_width=True)
                with tab3:
                    # Display all features, including the new gender feature
                    display_features = df.drop(columns=['image']).T.rename(columns={0: "Value"}).round(3)
                    st.dataframe(display_features, use_container_width=True)
            else:
                st.error("Could not process the uploaded image. Please try another one.")
def render_personality_test():
    st.header("Take the 50-Item IPIP Personality Survey")
    
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
        
        # NEW, COMBINED RESULTS BLOCK
        st.subheader("Comparison: Survey vs. Image Analysis")

        # Generate the combined radar chart
        comparison_chart = plot_radar_chart(
            scores1=survey_scores_df.iloc[0], 
            label1="Survey Results",
            scores2=st.session_state.image_scores.iloc[0] if st.session_state.image_scores is not None else None,
            label2="Paper Prediction"
        )
        st.image(comparison_chart, width=600)

        # Display the comparison table
        if st.session_state.image_scores is not None:
            comparison_df = survey_scores_df.T.rename(columns={0: "Survey Score"})
            comparison_df["Image Score"] = st.session_state.image_scores.drop(columns=['image']).T
            st.dataframe(comparison_df, use_container_width=True)
        else:
            st.warning("You haven't analyzed an image yet. The chart above only shows your survey results.")
            st.dataframe(survey_scores_df.T.rename(columns={0: "Score (0-100)"}), use_container_width=True)
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
        
        # This loop now ONLY displays the questions.
        for i in range(start_idx, end_idx):
            q_text, _, _ = IPIP_QUESTIONS[i]
            answer = st.radio(
                f"**{i+1}.** {q_text}", options, index=2, horizontal=False, 
                key=f"q_{i}", help="Select the option that best describes you."
            )
            st.session_state.answers[f"q_{i}"] = options.index(answer) + 1
        
        # --- THE FIX: The navigation block is now OUTSIDE and AFTER the for loop ---
        st.markdown("---") 
        col1, col2 = st.columns(2)

        with col1:
            if st.session_state.survey_page > 0:
                if st.button("Previous", use_container_width=True, key="prev_button"):
                    st.session_state.survey_page -= 1
                    st.rerun()

        with col2:
            if st.session_state.survey_page < (len(IPIP_QUESTIONS) // QUESTIONS_PER_PAGE - 1):
                if st.button("Next", use_container_width=True, key="next_button"):
                    st.session_state.survey_page += 1
                    st.rerun()
            else:
                if st.button("Submit", type="primary", use_container_width=True, key="submit_button"):
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