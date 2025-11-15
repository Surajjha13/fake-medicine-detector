# # # app.py
# # import streamlit as st
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.applications.vgg16 import preprocess_input
# # from tensorflow.keras.preprocessing import image
# # import numpy as np
# # from PIL import Image

# # # Load the trained model
# # model = load_model("best_model.keras")

# # # Mapping class indices
# # class_indices = {0: 'Real', 1: 'Fake'}  # same as train_gen.class_indices

# # st.title("💊 Fake vs Real Medicine Classifier")
# # st.write("Upload an image of a medicine and the model will predict if it's real or fake.")

# # # File uploader
# # uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
# # if uploaded_file is not None:
# #     # Open image
# #     img = Image.open(uploaded_file).convert('RGB')
# #     st.image(img, caption='Uploaded Image', use_column_width=True)
    
# #     # Preprocess image
# #     img = img.resize((224,224))
# #     img_array = image.img_to_array(img)
# #     img_array = np.expand_dims(img_array, axis=0)
# #     img_array = preprocess_input(img_array)
    
# #     # Prediction
# #     pred_prob = model.predict(img_array)[0][0]
# #     if pred_prob > 0.5:
# #         pred_class = 1
# #         confidence = pred_prob * 100
# #     else:
# #         pred_class = 0
# #         confidence = (1 - pred_prob) * 100
# #     st.write(f"**Prediction:** {class_indices[pred_class]}")
# #     st.write(f"**Confidence:** {confidence:.2f}%")


# # hybrid_multiapi_app_nocache.py
# import os
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input
# import pytesseract
# import requests

# # -------------------------
# # Configuration / constants
# # -------------------------
# MODEL_PATH = "best_model.keras"
# TESSERACT_CMD = os.getenv("TESSERACT_CMD")
# if TESSERACT_CMD:
#     pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# MIN_WORD_LEN = 3
# WEIGHT_API_BOOST = 12.0
# WEIGHT_API_PENALTY = 0.7

# # API endpoints
# OPENFDA_BASE = "https://api.fda.gov/drug/label.json"
# RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST/drugs.json"
# HEALTHOS_BASE = "https://www.healthos.co/api/v1/search/medicines/"
# HEALTHOS_KEY = os.getenv("HEALTHOS_API_KEY", None)

# # -------------------------
# # Model loading
# # -------------------------
# @st.cache_resource(show_spinner=False)
# def load_model(path=MODEL_PATH):
#     return tf.keras.models.load_model(path)

# model = load_model(MODEL_PATH)
# class_indices = {0: "Real", 1: "Fake"}

# # -------------------------
# # Image prep + OCR
# # -------------------------
# def prepare_image(img: Image.Image):
#     img = img.resize((224, 224))
#     arr = image.img_to_array(img)
#     arr = np.expand_dims(arr, 0)
#     return preprocess_input(arr)

# def extract_text(img: Image.Image):
#     try:
#         raw = pytesseract.image_to_string(img)
#         return raw.strip()
#     except Exception as e:
#         st.warning(f"OCR error: {e}")
#         return ""

# # -------------------------
# # API query helpers
# # -------------------------
# def query_openfda(name):
#     try:
#         params = {"search": f'openfda.brand_name:"{name}"', "limit": 2}
#         r = requests.get(OPENFDA_BASE, params=params, timeout=6)
#         if r.status_code == 200:
#             results = []
#             for item in r.json().get("results", []):
#                 of = item.get("openfda", {})
#                 results.extend(of.get("brand_name", []) + of.get("generic_name", []))
#             return [r.lower() for r in set(results)]
#     except:
#         return []
#     return []

# def query_rxnorm(name):
#     try:
#         url = f"{RXNORM_BASE}?name={requests.utils.quote(name)}"
#         r = requests.get(url, timeout=6)
#         if r.status_code == 200:
#             groups = r.json().get("drugGroup", {}).get("conceptGroup", [])
#             found = []
#             for g in groups:
#                 for concept in g.get("conceptProperties", []):
#                     found.append(concept.get("name"))
#             return [f.lower() for f in set(found)]
#     except:
#         return []
#     return []

# def query_healthos(name):
#     if not HEALTHOS_KEY:
#         return []
#     try:
#         headers = {"Authorization": f"Bearer {HEALTHOS_KEY}"}
#         url = HEALTHOS_BASE + requests.utils.quote(name)
#         r = requests.get(url, headers=headers, timeout=6)
#         if r.status_code == 200:
#             hits = []
#             for item in r.json().get("data", []):
#                 nm = item.get("name") or item.get("brand_name")
#                 if nm:
#                     hits.append(nm.lower())
#             return list(set(hits))
#     except:
#         return []
#     return []

# # -------------------------
# # Multi-API exact check
# # -------------------------
# def check_drug_multisource(word):
#     q = word.strip().lower()
#     if len(q) < MIN_WORD_LEN:
#         return False, "too_short"

#     sources = []
#     if HEALTHOS_KEY:
#         hits = query_healthos(q)
#         if hits:
#             sources.append(("HealthOS", hits))
#     hits = query_openfda(q)
#     if hits:
#         sources.append(("OpenFDA", hits))
#     hits = query_rxnorm(q)
#     if hits:
#         sources.append(("RxNorm", hits))

#     verified = False
#     details = []
#     for src, hits in sources:
#         for candidate in hits:
#             details.append((src, candidate))
#             if q == candidate:
#                 verified = True
#                 break
#         if verified:
#             break

#     return verified, str(details)

# # -------------------------
# # Hybrid predictor
# # -------------------------
# def hybrid_predict(img: Image.Image):
#     ocr_text = extract_text(img)
#     words = [w.lower().strip(".,()[]:;") for w in ocr_text.split() if len(w) >= MIN_WORD_LEN]

#     candidates = [ocr_text] + words
#     api_verified = False
#     api_detail = None
#     for cand in candidates:
#         verified, raw = check_drug_multisource(cand)
#         if verified:
#             api_verified = True
#             api_detail = (cand, raw)
#             break

#     # CNN visual check
#     proc = prepare_image(img)
#     pred_prob = model.predict(proc, verbose=0)[0][0]
#     if pred_prob > 0.5:
#         pred_class = 1
#         vis_conf = pred_prob * 100
#     else:
#         pred_class = 0
#         vis_conf = (1 - pred_prob) * 100

#     # Final logic
#     if api_verified and pred_class == 0:
#         final_label = "✅ Real (API Verified)"
#         final_conf = min(vis_conf + WEIGHT_API_BOOST, 99.9)
#     elif api_verified and pred_class == 1:
#         final_label = "⚠️ Looks Fake, but API says exists"
#         final_conf = vis_conf * 0.8
#     elif (not api_verified) and pred_class == 0:
#         final_label = "⚠️ Real-looking but not found in APIs"
#         final_conf = vis_conf * WEIGHT_API_PENALTY
#     else:
#         final_label = "❌ Fake or Unknown"
#         final_conf = vis_conf

#     return {
#         "ocr_text": ocr_text,
#         "api_verified": api_verified,
#         "api_detail": api_detail,
#         "visual_prob_fake": pred_prob,
#         "visual_class": class_indices[pred_class],
#         "final_label": final_label,
#         "final_confidence": final_conf
#     }

# # -------------------------
# # Streamlit App
# # -------------------------
# def main():
#     st.set_page_config(page_title="Hybrid Medicine Verifier", layout="wide")
#     st.title("💊 Real-Time Hybrid Medicine Verifier (OCR + Vision + Multi-API)")

#     col1, col2 = st.columns([1, 2])
#     with col1:
#         st.header("Input")
#         uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
#         cam = st.camera_input("Or use webcam (click to capture)")

#         with st.expander("Advanced Settings"):
#             global WEIGHT_API_BOOST, WEIGHT_API_PENALTY
#             WEIGHT_API_BOOST = st.slider("API Boost (+%)", 0, 30, int(WEIGHT_API_BOOST))
#             WEIGHT_API_PENALTY = st.slider("API Penalty Multiplier (%)", 50, 100, int(WEIGHT_API_PENALTY*100)) / 100.0

#     with col2:
#         st.header("Result")
#         img = None
#         if cam:
#             img = Image.open(cam).convert("RGB")
#         elif uploaded:
#             img = Image.open(uploaded).convert("RGB")

#         if img is not None:
#             st.image(img, caption="Input Image", use_column_width=True)
#             with st.spinner("Analyzing with OCR, APIs, and CNN model..."):
#                 res = hybrid_predict(img)

#             st.subheader("OCR Text")
#             st.code(res["ocr_text"] or "No readable text")

#             st.subheader("API Verification")
#             st.write(f"Verified via API: {'✅' if res['api_verified'] else '❌'}")
#             if res["api_detail"]:
#                 st.write("Matched:", res["api_detail"][0])
#                 st.write("Details:", res["api_detail"][1])

#             st.subheader("Visual Model (CNN)")
#             st.write(f"Predicted class: **{res['visual_class']}**")
#             st.write(f"Fake probability: **{res['visual_prob_fake']*100:.2f}%**")

#             st.markdown("---")
#             st.subheader("Final Decision")
#             st.markdown(f"### {res['final_label']}")
#             st.progress(min(int(res["final_confidence"]), 100))
#             st.write(f"Overall Confidence: **{res['final_confidence']:.2f}%**")

#             st.info("⚠️ This app is for educational/research purposes only. Always verify medicines via official sources.")

# if __name__ == "__main__":
#     main()


# hybrid_multiapi_app_nocache.py
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import pytesseract
import requests
import re

# -------------------------
# Config / constants
# -------------------------
MODEL_PATH = "best_model.keras"
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

MIN_WORD_LEN = 3
WEIGHT_API_BOOST = 12.0
WEIGHT_API_PENALTY = 0.7

OPENFDA_BASE = "https://api.fda.gov/drug/label.json"
RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST/drugs.json"
HEALTHOS_BASE = "https://www.healthos.co/api/v1/search/medicines/"
HEALTHOS_KEY = os.getenv("HEALTHOS_API_KEY", None)

# -------------------------
# Model loading
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model_cached(path=MODEL_PATH):
    return tf.keras.models.load_model(path)

model = load_model_cached(MODEL_PATH)
class_indices = {0: "Real", 1: "Fake"}

# -------------------------
# Image prep + OCR
# -------------------------
def prepare_image(img: Image.Image):
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    return preprocess_input(arr)

def extract_text(img: Image.Image):
    try:
        raw = pytesseract.image_to_string(img)
        return raw.strip()
    except Exception as e:
        st.warning(f"OCR error: {e}")
        return ""

def clean_drug_name(raw):
    # keep only alphabetic words, drop units/symbols
    words = re.findall(r"[A-Za-z]+", raw)
    return " ".join(words[:2]).lower().strip()

# -------------------------
# API query helpers
# -------------------------
def query_openfda(name):
    try:
        params = {"search": f'openfda.brand_name:"{name}"', "limit": 2}
        r = requests.get(OPENFDA_BASE, params=params, timeout=6)
        if r.ok:
            results = []
            for item in r.json().get("results", []):
                of = item.get("openfda", {})
                results.extend(of.get("brand_name", []) + of.get("generic_name", []))
            return [r.lower() for r in set(results)]
    except:
        pass
    return []

def query_rxnorm(name):
    try:
        url = f"{RXNORM_BASE}?name={requests.utils.quote(name)}"
        r = requests.get(url, timeout=6)
        if r.ok:
            groups = r.json().get("drugGroup", {}).get("conceptGroup", [])
            found = []
            for g in groups:
                for c in g.get("conceptProperties", []):
                    found.append(c.get("name"))
            return [f.lower() for f in set(found)]
    except:
        pass
    return []

def query_healthos(name):
    if not HEALTHOS_KEY:
        return []
    try:
        headers = {"Authorization": f"Bearer {HEALTHOS_KEY}"}
        url = HEALTHOS_BASE + requests.utils.quote(name)
        r = requests.get(url, headers=headers, timeout=6)
        if r.ok:
            hits = []
            for item in r.json().get("data", []):
                nm = item.get("name") or item.get("brand_name")
                if nm:
                    hits.append(nm.lower())
            return list(set(hits))
    except:
        pass
    return []

# -------------------------
# Multi-API check
# -------------------------
def check_drug_multisource(name):
    q = clean_drug_name(name)
    if len(q) < MIN_WORD_LEN:
        return False, "too_short"

    sources = []
    if HEALTHOS_KEY:
        hits = query_healthos(q)
        if hits:
            sources.append(("HealthOS", hits))

    hits = query_openfda(q)
    if hits:
        sources.append(("OpenFDA", hits))

    hits = query_rxnorm(q)
    if hits:
        sources.append(("RxNorm", hits))

    verified = False
    details = []
    for src, hits in sources:
        for candidate in hits:
            details.append((src, candidate))
            if q in candidate:
                verified = True
                break
        if verified:
            break

    return verified, str(details or "no_hits")

# -------------------------
# Hybrid predictor
# -------------------------
def hybrid_predict(img: Image.Image):
    ocr_text = extract_text(img)
    words = [clean_drug_name(w) for w in ocr_text.split() if len(w) >= MIN_WORD_LEN]
    candidates = [clean_drug_name(ocr_text)] + words

    api_verified = False
    api_detail = None
    for cand in candidates:
        if not cand:
            continue
        verified, raw = check_drug_multisource(cand)
        if verified:
            api_verified = True
            api_detail = (cand, raw)
            break

    # CNN check
    proc = prepare_image(img)
    pred_prob = model.predict(proc, verbose=0)[0][0]
    pred_class = 1 if pred_prob > 0.5 else 0
    vis_conf = (pred_prob if pred_class == 1 else 1 - pred_prob) * 100

    # Merge logic
    if api_verified and pred_class == 0:
        final_label = "✅ Real (API Verified)"
        final_conf = min(vis_conf + WEIGHT_API_BOOST, 99.9)
    elif api_verified and pred_class == 1:
        final_label = "⚠️ Looks Fake, but API says exists"
        final_conf = vis_conf * 0.8
    elif not api_verified and pred_class == 0:
        final_label = "⚠️ Real-looking but not found in APIs"
        final_conf = vis_conf * WEIGHT_API_PENALTY
    else:
        final_label = "❌ Fake or Unknown"
        final_conf = vis_conf

    return {
        "ocr_text": ocr_text,
        "api_verified": api_verified,
        "api_detail": api_detail,
        "visual_prob_fake": pred_prob,
        "visual_class": class_indices[pred_class],
        "final_label": final_label,
        "final_confidence": final_conf,
    }

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="Hybrid Medicine Verifier", layout="wide")
    st.title("💊 Real-Time Hybrid Medicine Verifier (OCR + Vision + Multi-API)")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Input")
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        cam = st.camera_input("Or use webcam (click to capture)")

        with st.expander("Advanced Settings"):
            global WEIGHT_API_BOOST, WEIGHT_API_PENALTY
            WEIGHT_API_BOOST = st.slider("API Boost (+%)", 0, 30, int(WEIGHT_API_BOOST))
            WEIGHT_API_PENALTY = (
                st.slider("API Penalty Multiplier (%)", 50, 100, int(WEIGHT_API_PENALTY * 100))
                / 100.0
            )

    with col2:
        st.header("Result")
        img = None
        if cam:
            img = Image.open(cam).convert("RGB")
        elif uploaded:
            img = Image.open(uploaded).convert("RGB")

        if img is not None:
            st.image(img, caption="Input Image", use_column_width=True)
            with st.spinner("Analyzing with OCR, APIs, and CNN model..."):
                res = hybrid_predict(img)

            st.subheader("OCR Text")
            st.code(res["ocr_text"] or "No readable text")

            st.subheader("API Verification")
            st.write(f"Verified via API: {'✅' if res['api_verified'] else '❌'}")
            if res["api_detail"]:
                st.write("Matched:", res["api_detail"][0])
                st.write("Details:", res["api_detail"][1])

            st.subheader("Visual Model (CNN)")
            st.write(f"Predicted class: **{res['visual_class']}**")
            st.write(f"Fake probability: **{res['visual_prob_fake'] * 100:.2f}%**")

            st.markdown("---")
            st.subheader("Final Decision")
            st.markdown(f"### {res['final_label']}")
            st.progress(min(int(res["final_confidence"]), 100))
            st.write(f"Overall Confidence: **{res['final_confidence']:.2f}%**")

            #st.info("⚠️ Educational use only. Always verify medicines via official sources.")

if __name__ == "__main__":
    main()
