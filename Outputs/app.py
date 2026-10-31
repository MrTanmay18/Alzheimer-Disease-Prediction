from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# TensorFlow is needed only for the MRI section.
try:
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input
except Exception:
    tf = None
    preprocess_input = None

# ----------------------------- Configuration -----------------------------
APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "Outputs"

# Update these paths if your filenames differ.
CLINICAL_MODEL_CANDIDATES = [
    # Your saved clinical model (filename variants supported)
    OUTPUT_DIR / "HistGradientBoosting.joblib",
    OUTPUT_DIR / "histgradient boosting.joblib",
    OUTPUT_DIR / "histgradient_boosting.joblib",
    OUTPUT_DIR / "HistGradientBoosting_Model_Only.joblib",
    OUTPUT_DIR / "HistGradientBoosting_Alzheimers.joblib",
    OUTPUT_DIR / "best_model_HistGradientBoosting.joblib",
]
MRI_MODEL_PATH = OUTPUT_DIR / "resnet50_model.keras"

MRI_CLASSES = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented",
]
MRI_SIZE = (224, 224)

CONTINUOUS = [
    "Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality",
    "SleepQuality", "SystolicBP", "DiastolicBP", "CholesterolTotal",
    "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides",
    "MMSE", "FunctionalAssessment", "ADL",
]
BINARY = [
    "Gender", "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease",
    "Diabetes", "Depression", "HeadInjury", "Hypertension",
    "MemoryComplaints", "BehavioralProblems", "Confusion", "Disorientation",
    "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness",
]
SCALE_COLS = CONTINUOUS + ["PulsePressure", "CholesterolRatio"]
ETHNICITY_COLUMNS = [
    "Ethnicity_Caucasian", "Ethnicity_AfricanAmerican",
    "Ethnicity_Asian", "Ethnicity_Other",
]
SYMPTOMS = [
    "MemoryComplaints", "BehavioralProblems", "Confusion", "Disorientation",
    "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness",
]
COMORBID = [
    "CardiovascularDisease", "Diabetes", "Depression", "HeadInjury", "Hypertension",
]

# ----------------------------- Page styling -----------------------------
st.set_page_config(
    page_title="Alzheimer's Multimodal Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background: linear-gradient(135deg, #f7f9fc 0%, #eef4ff 100%); }
  .hero { padding: 1.5rem 1.8rem; border-radius: 18px;
          background: linear-gradient(110deg, #102a43, #245b85);
          color: white; margin-bottom: 1.2rem; }
  .hero h1 { color: white; margin: 0; font-size: 2.15rem; }
  .hero p { color: #e5f0fb; margin: .45rem 0 0 0; font-size: 1.02rem; }
  .metric-card { background: white; border: 1px solid #dbe6f2;
                 border-radius: 14px; padding: 1rem 1.2rem; }
  .section-note { color: #53657a; font-size: .95rem; }
  div[data-testid="stForm"] { background: white; padding: 1rem 1.2rem;
      border-radius: 14px; border: 1px solid #dbe6f2; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🧠 Alzheimer’s Multimodal Screening Dashboard</h1>
  <p>Clinical risk prediction + MRI image classification in one research prototype</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## About this project")
    st.write(
        "This academic prototype combines structured clinical information "
        "with an MRI image classifier to explore two complementary prediction pathways."
    )
    st.markdown("---")
    st.markdown("### Models")
    st.write("• Clinical: saved classical ML model (`.pkl` / `.joblib`)")
    st.write("• MRI: ResNet50 transfer-learning model (`.keras`)")
    st.markdown("---")
    st.caption("Research/educational use only. Not a medical diagnosis.")

tab_clinical, tab_mri, tab_project = st.tabs(
    ["🩺 Clinical Assessment", "🧲 MRI Classification", "📘 Project Details"]
)

# ----------------------------- Helpers -----------------------------
@st.cache_resource(show_spinner=False)
def load_clinical_artifact(path_str):
    return joblib.load(path_str)

@st.cache_resource(show_spinner=False)
def load_mri_model(path_str):
    if tf is None:
        raise RuntimeError("TensorFlow is unavailable. Install tensorflow to use MRI classification.")
    # compile=False avoids loading training-only optimizer state.
    return tf.keras.models.load_model(path_str, compile=False)

def build_clinical_row(values):
    """Create one row in the same feature layout used in AD_ML_complete.ipynb."""
    row = dict(values)
    row["PulsePressure"] = row["SystolicBP"] - row["DiastolicBP"]
    row["CholesterolRatio"] = row["CholesterolTotal"] / max(row["CholesterolHDL"], 1e-6)
    row["SymptomCount"] = sum(int(row[c]) for c in SYMPTOMS)
    row["ComorbidityCount"] = sum(int(row[c]) for c in COMORBID)

    ethnicity_map = {
        "Caucasian": "Ethnicity_Caucasian",
        "African American": "Ethnicity_AfricanAmerican",
        "Asian": "Ethnicity_Asian",
        "Other": "Ethnicity_Other",
    }
    for col in ETHNICITY_COLUMNS:
        row[col] = 0
    row[ethnicity_map[values["Ethnicity"]]] = 1

    # The notebook removes EDA-only AgeGroup/BMICategory and does not use Diagnosis.
    row.pop("Ethnicity", None)
    row.pop("EducationLevel", None)  # model pipeline in notebook excludes it from feature groups? see below
    return pd.DataFrame([row])

def prepare_clinical_features(raw_row, artifact):
    """Align user-entered values with the saved model's expected feature columns."""
    model = artifact.get("model") if isinstance(artifact, dict) and "model" in artifact else artifact
    feature_columns = None
    scaler = None
    scale_columns = SCALE_COLS

    if isinstance(artifact, dict):
        feature_columns = artifact.get("feature_columns")
        scaler = artifact.get("scaler")
        scale_columns = artifact.get("scale_columns", SCALE_COLS)

    if feature_columns is None and hasattr(model, "feature_names_in_"):
        feature_columns = list(model.feature_names_in_)

    # Build features including EducationLevel encoded as the original ordinal numeric
    values = raw_row.copy()
    values["PulsePressure"] = values["SystolicBP"] - values["DiastolicBP"]
    values["CholesterolRatio"] = values["CholesterolTotal"] / max(values["CholesterolHDL"], 1e-6)
    values["SymptomCount"] = sum(int(values[c]) for c in SYMPTOMS)
    values["ComorbidityCount"] = sum(int(values[c]) for c in COMORB)
    ethnicity_map = {
        "Caucasian": "Ethnicity_Caucasian",
        "African American": "Ethnicity_AfricanAmerican",
        "Asian": "Ethnicity_Asian",
        "Other": "Ethnicity_Other",
    }
    for col in ETHNICITY_COLUMNS:
        values[col] = 0
    values[ethnicity_map[values.pop("Ethnicity")]] = 1
    # notebook retains EducationLevel as ordinal code
    X = pd.DataFrame([values])

    if feature_columns:
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X.reindex(columns=feature_columns, fill_value=0)
    else:
        # For a bare model, the feature count/order must match training.
        expected = getattr(model, "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            raise ValueError(
                f"The loaded model expects {expected} features, but the app built {X.shape[1]}. "
                "Use the complete model package containing feature_columns, or update the feature list."
            )

    if scaler is not None:
        cols = [c for c in scale_columns if c in X.columns]
        X.loc[:, cols] = scaler.transform(X[cols])
    return model, X

def predict_clinical(model, X):
    pred = int(np.asarray(model.predict(X)).reshape(-1)[0])
    labels = {0: "No Alzheimer's", 1: "Alzheimer's"}
    probability = None
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X))[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            probability = float(probs[classes.index(1)])
    return labels.get(pred, str(pred)), probability

def preprocess_mri_image(image):
    image = image.convert("RGB").resize(MRI_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ----------------------------- Clinical tab -----------------------------
with tab_clinical:
    st.subheader("Clinical Assessment")
    st.markdown(
        '<p class="section-note">Enter the clinical values. The app derives pulse pressure, '
        'cholesterol ratio, symptom count and comorbidity count automatically.</p>',
        unsafe_allow_html=True,
    )

    with st.form("clinical_form"):
        st.markdown("#### Demographics")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (years)", 60, 90, 70)
        gender = c2.selectbox("Gender", ["Male", "Female"])
        ethnicity = c3.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
        education = c1.selectbox("Education level", ["None", "High School", "Bachelor's", "Higher"])
        bmi = c2.number_input("BMI", 15.0, 40.0, 25.0, 0.1)

        st.markdown("#### Lifestyle")
        l1, l2, l3 = st.columns(3)
        alcohol = l1.number_input("Alcohol consumption", 0.0, 20.0, 5.0, 0.1)
        activity = l2.number_input("Physical activity", 0.0, 10.0, 5.0, 0.1)
        diet = l3.number_input("Diet quality", 0.0, 10.0, 5.0, 0.1)
        sleep = l1.number_input("Sleep quality", 4.0, 10.0, 7.0, 0.1)

        st.markdown("#### Vitals & clinical assessments")
        v1, v2, v3 = st.columns(3)
        systolic = v1.number_input("Systolic BP", 90, 180, 120)
        diastolic = v2.number_input("Diastolic BP", 60, 120, 80)
        total_chol = v3.number_input("Total cholesterol", 150.0, 300.0, 200.0, 0.1)
        ldl = v1.number_input("LDL cholesterol", 50.0, 200.0, 100.0, 0.1)
        hdl = v2.number_input("HDL cholesterol", 20.0, 100.0, 50.0, 0.1)
        triglycerides = v3.number_input("Triglycerides", 50.0, 400.0, 150.0, 0.1)
        mmse = v1.number_input("MMSE score", 0.0, 30.0, 25.0, 0.1)
        functional = v2.number_input("Functional assessment", 0.0, 10.0, 7.0, 0.1)
        adl = v3.number_input("ADL score", 0.0, 10.0, 7.0, 0.1)

        st.markdown("#### Medical history & symptoms")
        flag_labels = {
            "Smoking": "Smoking", "FamilyHistoryAlzheimers": "Family history of Alzheimer’s",
            "CardiovascularDisease": "Cardiovascular disease", "Diabetes": "Diabetes",
            "Depression": "Depression", "HeadInjury": "Head injury",
            "Hypertension": "Hypertension", "MemoryComplaints": "Memory complaints",
            "BehavioralProblems": "Behavioral problems", "Confusion": "Confusion",
            "Disorientation": "Disorientation", "PersonalityChanges": "Personality changes",
            "DifficultyCompletingTasks": "Difficulty completing tasks", "Forgetfulness": "Forgetfulness",
        }
        flags = {}
        cols = st.columns(4)
        for idx, (key, label) in enumerate(flag_labels.items()):
            flags[key] = int(cols[idx % 4].selectbox(label, ["No", "Yes"], key=f"flag_{key}") == "Yes")
        submitted = st.form_submit_button("🔎 Assess clinical profile", use_container_width=True)

    if submitted:
        gender_code = 0 if gender == "Male" else 1
        education_code = {"None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3}[education]
        raw = {
            "Age": age, "Gender": gender_code, "Ethnicity": ethnicity,
            "EducationLevel": education_code, "BMI": bmi,
            "AlcoholConsumption": alcohol, "PhysicalActivity": activity,
            "DietQuality": diet, "SleepQuality": sleep, "SystolicBP": systolic,
            "DiastolicBP": diastolic, "CholesterolTotal": total_chol,
            "CholesterolLDL": ldl, "CholesterolHDL": hdl,
            "CholesterolTriglycerides": triglycerides, "MMSE": mmse,
            "FunctionalAssessment": functional, "ADL": adl,
            **flags,
        }
        artifact_path = next((p for p in CLINICAL_MODEL_CANDIDATES if p.exists()), None)
        if artifact_path is None:
            st.error("Clinical model file not found. Place your joblib/pickle model in the Outputs folder.")
            st.caption("Checked: " + ", ".join(p.name for p in CLINICAL_MODEL_CANDIDATES))
        else:
            try:
                artifact = load_clinical_artifact(str(artifact_path))
                model, X = prepare_clinical_features(raw, artifact)
                label, prob = predict_clinical(model, X)
                st.markdown("### Clinical model result")
                if label == "Alzheimer's":
                    st.warning("Model output: Alzheimer’s class")
                else:
                    st.success("Model output: No Alzheimer’s class")
                if prob is not None:
                    st.metric("Estimated model probability for Alzheimer’s class", f"{prob:.1%}")
                st.caption(f"Artifact loaded: {artifact_path.name}")
                st.info("This is a model prediction from entered data, not a clinical diagnosis.")
            except Exception as exc:
                st.error(f"Could not run clinical prediction: {exc}")
                st.caption("For reliable feature alignment, use the complete joblib package saved with model, scaler and feature_columns.")

# ----------------------------- MRI tab -----------------------------
with tab_mri:
    st.subheader("MRI Image Classification")
    st.markdown(
        '<p class="section-note">Upload one MRI image. ResNet50 expects a 224 × 224 RGB image '
        'with its ImageNet preprocessing.</p>',
        unsafe_allow_html=True,
    )
    uploaded_image = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png", "bmp"], key="mri_upload")
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        left, right = st.columns([1, 1.2])
        with left:
            st.image(image, caption="Uploaded MRI image", use_container_width=True)
        with right:
            if st.button("🧠 Classify MRI", type="primary", use_container_width=True):
                if not MRI_MODEL_PATH.exists():
                    st.error(f"MRI model not found: {MRI_MODEL_PATH}")
                elif tf is None:
                    st.error("TensorFlow is not installed in this environment.")
                else:
                    try:
                        with st.spinner("Loading ResNet50 and analyzing image..."):
                            model = load_mri_model(str(MRI_MODEL_PATH))
                            batch = preprocess_mri_image(image)
                            probs = np.asarray(model.predict(batch, verbose=0))[0]
                        idx = int(np.argmax(probs))
                        st.markdown("### MRI model result")
                        st.success(f"Predicted class: **{MRI_CLASSES[idx]}**")
                        st.metric("Model confidence", f"{float(probs[idx]):.1%}")
                        prob_df = pd.DataFrame({"MRI class": MRI_CLASSES, "Probability": probs})
                        st.bar_chart(prob_df.set_index("MRI class"))
                        st.caption("Class order follows the alphabetical folder order from flow_from_directory.")
                    except Exception as exc:
                        st.error(f"MRI prediction failed: {exc}")

# ----------------------------- Project details -----------------------------
with tab_project:
    st.subheader("Project Overview")
    st.markdown("""
    **Objective:** Build a research prototype that explores Alzheimer’s-related prediction
    from two modalities: structured clinical data and brain MRI images.

    **Clinical pathway**
    - User enters demographics, lifestyle, vitals, medical history and assessment scores.
    - Derived features include pulse pressure, cholesterol ratio, symptom count and comorbidity count.
    - A saved classical machine-learning model generates a class prediction.

    **MRI pathway**
    - User uploads a brain MRI image.
    - The image is resized to 224 × 224 and processed using ResNet50 preprocessing.
    - The transfer-learning model predicts one of four MRI image classes.

    **MRI classes:** Non Demented, Very Mild Demented, Mild Demented, Moderate Demented.

    **Important limitation:** These pathways are shown separately in this prototype. They are not
    fused into a single multimodal prediction unless a dedicated fusion model is trained and
    integrated. Model outputs should not be used to diagnose, treat or make clinical decisions.
    """)
    st.markdown("---")
    st.markdown("**Project stack:** Python • Streamlit • Pandas • Scikit-learn • Joblib • TensorFlow/Keras • ResNet50")
