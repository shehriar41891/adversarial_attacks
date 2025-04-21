import streamlit as st
import requests

st.set_page_config(page_title="FGSM Attack Demo", layout="centered")

st.title("🔐 Adversarial Attack using FGSM")
st.write("Upload an image and see how a small perturbation can fool a deep learning model.")

uploaded_file = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])
epsilon = st.slider("⚠️ Epsilon (perturbation strength)", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
label = st.number_input("🔢 Ground Truth Label (ImageNet index, 0-999)", min_value=0, max_value=999, step=1)

if st.button("🚀 Run Attack"):
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        files = {"image": (uploaded_file.name, image_bytes, uploaded_file.type)}  # <-- MUST be "image"

        payload = {
            "label": str(label),        # <-- Must be string because it's multipart/form
            "epsilon": str(epsilon)
        }

        try:
            res = requests.post("http://localhost:8000/fgsm-attack/", files=files, data=payload)

            if res.status_code == 200:
                result = res.json()
                col1, col2 = st.columns(2)

                with col1:
                    st.image(result["original_image"], caption=f"Original: {result['original_label']} ({result['confidence_original']:.2f}%)")

                with col2:
                    st.image(result["adversarial_image"], caption=f"Adversarial: {result['adversarial_label']} ({result['confidence_adversarial']:.2f}%)")

                if result["success"]:
                    st.error("⚠️ Attack Successful: Prediction Changed!")
                else:
                    st.success("✅ Attack Failed: Prediction Remained Same.")
            else:
                st.error(f"Error from API: {res.status_code}\n{res.text}")

        except Exception as e:
            st.error(f"API request failed: {str(e)}")
    else:
        st.warning("Please upload an image to proceed.")
