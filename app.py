
import os

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from tensorflow import keras


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AgroGuard AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = [
    "Armadilles",
    "Bear",
    "Birds",
    "Cow",
    "Crocodile",
    "Deer",
    "Elephant",
    "Goat",
    "Horse",
    "Jaguar",
    "Monkey",
    "Rabbit",
    "Skunk",
    "Tiger",
    "Wild Boar",
]

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Models",
    "efficientnetb0_best.keras",
)

IMAGE_SIZE = (224, 224)

# Basic confidence threshold.
# This is a warning rule, not a guaranteed unknown-animal detector.
UNKNOWN_THRESHOLD = 0.60


# =========================================================
# PAGE STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #101820;
        color: #F4F7F5;
    }

    .hero {
        padding: 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #173D35, #24594A);
        margin-bottom: 1.5rem;
        border: 1px solid #376B59;
    }

    .hero h1 {
        color: #FFFFFF;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }

    .hero p {
        color: #DCEBE4;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .result-card {
        padding: 1.2rem;
        border-radius: 14px;
        background-color: #1B2B2B;
        border: 1px solid #36564B;
        margin: 0.8rem 0;
    }

    div.stButton > button[kind="primary"] {
        background-color: #287A57;
        border: 1px solid #3E9A70;
        color: white;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #32966B;
        border-color: #56B987;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    """Load the trained EfficientNetB0 model once."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Check that the Models folder contains "
            "efficientnetb0_best.keras."
        )

    model = keras.models.load_model(MODEL_PATH)

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            "The model output count does not match "
            "the configured class count."
        )

    return model


# =========================================================
# IMAGE PREDICTION
# =========================================================

def predict_animal(model, image):
    """Return the top three predictions and their scores."""

    resized_image = image.convert("RGB").resize(IMAGE_SIZE)

    image_array = np.asarray(
        resized_image,
        dtype=np.float32,
    )

    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(
        image_array,
        verbose=0,
    )[0]

    if (
        len(probabilities) != len(CLASS_NAMES)
        or not np.all(np.isfinite(probabilities))
    ):
        raise ValueError("The model returned invalid predictions.")

    top_indices = np.argsort(probabilities)[-3:][::-1]

    best_index = int(top_indices[0])
    predicted_class = CLASS_NAMES[best_index]
    confidence = float(probabilities[best_index])

    return (
        top_indices,
        predicted_class,
        confidence,
        probabilities,
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🐾 AgroGuard AI</h1>
        <p>
            AI-powered animal image classification for
            agricultural and wildlife applications.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Upload an animal image to view its predicted class, "
    "confidence score, and alternative predictions."
)


# =========================================================
# SIDEBAR — SIMPLIFIED
# =========================================================

with st.sidebar:
    st.header("About AgroGuard AI")

    st.write(
        "AgroGuard AI uses a deep-learning model to classify "
        "animal images and highlight uncertain predictions."
    )

    st.divider()

    st.caption(
        "The model supports a limited set of animal classes. "
        "Animals outside those classes may be misclassified."
    )


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload an animal image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear JPG, JPEG, or PNG image.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

    except (UnidentifiedImageError, OSError):
        st.error(
            "Unable to open this image. "
            "Please upload a valid JPG, JPEG, or PNG file."
        )
        st.stop()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Uploaded image")

        st.image(
            image,
            caption="Image selected for classification",
            width="stretch",
        )

        st.caption(
            f"Image dimensions: {image.width} × {image.height}"
        )

    with right:
        st.subheader("Animal prediction")

        if st.button(
            "Identify Animal",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner("Analyzing image..."):
                    model = load_model()

                    (
                        top_indices,
                        predicted_class,
                        confidence,
                        probabilities,
                    ) = predict_animal(model, image)

                # -----------------------------------------
                # UNKNOWN / UNCERTAIN RESULT
                # -----------------------------------------

                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True,
                )

                if confidence < UNKNOWN_THRESHOLD:
                    st.warning(
                        "Unknown / Uncertain Animal: the model "
                        "cannot confidently identify this image "
                        "as one of its supported classes."
                    )

                    st.markdown("### 🐾 Unknown / Uncertain Animal")

                    st.caption(
                        f"Closest model match: {predicted_class} "
                        f"({confidence:.2%} confidence)"
                    )

                else:
                    st.success("Prediction completed")

                    st.markdown(
                        f"### 🐾 {predicted_class}"
                    )

                    st.caption(
                        "This is the model's best-supported "
                        "class prediction, not a guarantee "
                        "that the identification is correct."
                    )

                st.metric(
                    "Model confidence",
                    f"{confidence:.2%}",
                )

                st.markdown("</div>", unsafe_allow_html=True)

                # -----------------------------------------
                # TOP THREE PREDICTIONS
                # -----------------------------------------

                st.subheader("Top 3 model predictions")

                for index in top_indices:
                    index = int(index)
                    score = float(probabilities[index])
                    label = CLASS_NAMES[index]

                    st.write(f"**{label}** — {score:.2%}")

                    st.progress(
                        min(max(score, 0.0), 1.0)
                    )

                # -----------------------------------------
                # LIMITATION NOTICE
                # -----------------------------------------

                st.info(
                    "This model recognizes only limited animal "
                    "classes. An animal outside those classes "
                    "may still receive a high confidence score "
                    "for an incorrect class. The unknown/"
                    "uncertain warning is based on a confidence "
                    "threshold and cannot reliably detect every "
                    "unrecognized animal."
                )

            except FileNotFoundError as error:
                st.error(str(error))

            except Exception as error:
                st.error(
                    "Prediction failed. Check the model file "
                    "and application logs."
                )
                st.exception(error)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AgroGuard AI | "
    "Animal Image Classification | Deep Learning Project"
)
