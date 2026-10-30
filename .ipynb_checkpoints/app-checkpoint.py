import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import glob

# ======================
# CONFIGURATION
# ======================
MODEL_PATH = "runs/detect/train9/weights/best.pt"  # change this to your model path
EXAMPLES_DIR = "examples"  # folder with example images

# ======================
# LOAD MODEL
# ======================
st.set_page_config(page_title="🧠 Brain Tumor Detection", layout="wide")
st.title("🧠 Brain Tumor Detection System")
st.markdown("Upload an MRI scan to detect **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor**.")

model = YOLO(MODEL_PATH)

# ======================
# FILE UPLOAD SECTION
# ======================
uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("⏳ Analyzing image...")

    # Run detection
    results = model.predict(source=image, save=True, conf=0.25)
    result = results[0]

    # Show detection image
    import os
    result_img_path = os.path.join(result.save_dir, os.path.basename(result.path))
    st.image(result_img_path, caption="Detection Result", use_column_width=True)

    # Extract class name
    names = model.names
    detected_classes = [names[int(cls)] for cls in result.boxes.cls]

    if detected_classes:
        tumor_type = detected_classes[0].capitalize()
        st.success(f"✅ Detected Tumor Type: **{tumor_type}**")

        # Provide medical guidance
        st.subheader("🩺 Suggested Medical Guidance")
        if "Glioma" in tumor_type:
            st.write("""
            - Gliomas are tumors that start in the glial cells of the brain.
            - Recommended next steps:
              - MRI with contrast for grading
              - Consultation with a neuro-oncologist
              - Possible surgical removal followed by radiation or chemotherapy
            """)
        elif "Meningioma" in tumor_type:
            st.write("""
            - Meningiomas are usually benign and slow-growing.
            - Recommended next steps:
              - MRI follow-up every 6–12 months
              - Surgery only if it causes pressure symptoms
              - Stereotactic radiosurgery (Gamma Knife) for small, deep tumors
            """)
        elif "Pituitary" in tumor_type:
            st.write("""
            - Pituitary tumors can affect hormone levels.
            - Recommended next steps:
              - Endocrinology consult for hormone testing
              - MRI of the sella turcica
              - Medical therapy or transsphenoidal surgery if needed
            """)
        else:
            st.write("""
            - No tumor detected 🎉
            - Recommended next steps:
              - Maintain regular checkups if symptoms persist
              - Healthy lifestyle and hydration
            """)
    else:
        st.warning("No tumor detected in this image.")

# ======================
# 4×2 GRID OF EXAMPLES
# ======================
st.markdown("---")
st.header("📸 Example Tumor Types")

example_images = {
    "No Tumor": glob.glob(os.path.join(EXAMPLES_DIR, "notumor*.jpg")),
    "Glioma": glob.glob(os.path.join(EXAMPLES_DIR, "glioma*.jpg")),
    "Meningioma": glob.glob(os.path.join(EXAMPLES_DIR, "meningioma*.jpg")),
    "Pituitary": glob.glob(os.path.join(EXAMPLES_DIR, "pituitary*.jpg"))
}

cols = st.columns(4)
for i, (tumor, paths) in enumerate(example_images.items()):
    with cols[i]:
        st.subheader(tumor)
        for img_path in paths[:2]:  # show up to 2 images each
            st.image(img_path, use_column_width=True)
