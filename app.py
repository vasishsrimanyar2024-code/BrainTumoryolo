import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import glob

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "brain_tumor_yolo_best.pt")
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")

st.set_page_config(
    page_title="🧠 Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Detection System")

st.markdown(
    "Upload an MRI scan to detect **Glioma**, **Meningioma**, "
    "**Pituitary Tumor**, or **No Tumor**."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"❌ Model file not found: {MODEL_PATH}. "
        "Train it first by running `python train_model.py`."
    )
    st.stop()


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

st.header("📤 Upload MRI Scan")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("🖼️ Uploaded MRI")

    st.image(
        image,
        caption="Uploaded MRI Scan",
        width="stretch"
    )

    st.write("⏳ Analyzing image...")

    try:

        results = model.predict(
            source=image,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        st.subheader("🔍 Detection Result")

        plotted_image = result.plot()

        st.image(
            plotted_image,
            caption="YOLO Detection Result",
            width="stretch"
        )

        names = model.names
        detections = []

        if result.boxes is not None and len(result.boxes) > 0:

            for cls, conf in zip(
                result.boxes.cls,
                result.boxes.conf
            ):

                class_id = int(cls)
                confidence = float(conf)
                class_name = names[class_id]

                detections.append(
                    {
                        "class": class_name,
                        "confidence": confidence
                    }
                )

        if detections:

            st.subheader("🧠 Prediction")

            best_detection = max(
                detections,
                key=lambda x: x["confidence"]
            )

            tumor_type = best_detection["class"]
            confidence = best_detection["confidence"]

            st.success(
                f"✅ Detected: **{tumor_type}**\n\n"
                f"Confidence: **{confidence:.2%}**"
            )

            st.subheader("📊 All Detections")

            for detection in detections:

                st.write(
                    f"**{detection['class']}** — "
                    f"{detection['confidence']:.2%}"
                )

            normalized_type = tumor_type.lower()

            st.subheader("ℹ️ General Information")

            if "glioma" in normalized_type:

                st.write(
                    """
                    **Glioma** is a type of tumor that originates from
                    glial cells in the brain or spinal cord.

                    Further evaluation may involve MRI with contrast,
                    specialist consultation, and additional testing.
                    """
                )

            elif "meningioma" in normalized_type:

                st.write(
                    """
                    **Meningioma** develops from the meninges, the
                    membranes surrounding the brain and spinal cord.

                    Management depends on factors such as size,
                    location, growth, and symptoms.
                    """
                )

            elif "pituitary" in normalized_type:

                st.write(
                    """
                    **Pituitary tumors** develop in or around the
                    pituitary gland and may affect hormone production.

                    Medical evaluation may include hormone testing,
                    dedicated MRI imaging, and specialist consultation.
                    """
                )

            elif (
                "no tumor" in normalized_type
                or "notumor" in normalized_type
            ):

                st.write(
                    """
                    The model did not identify a tumor in the submitted
                    image.

                    This model result should not be considered a medical
                    diagnosis. If symptoms or clinical concerns persist,
                    consult a qualified medical professional.
                    """
                )

            else:

                st.write(
                    """
                    The model detected a class that is not specifically
                    handled by the information section.

                    Please consult a qualified medical professional for
                    clinical interpretation.
                    """
                )

        else:

            st.warning(
                "⚠️ No detectable tumor region was identified by the model."
            )

            st.info(
                "This is an AI model result and should not be considered "
                "a medical diagnosis."
            )

    except Exception as e:

        st.error(
            "❌ An error occurred while analyzing the image."
        )

        st.exception(e)


st.markdown("---")

st.header("📸 Example Tumor Types")

example_images = {
    "No Tumor": glob.glob(
        os.path.join(EXAMPLES_DIR, "notumor*.jpg")
    ),

    "Glioma": glob.glob(
        os.path.join(EXAMPLES_DIR, "glioma*.jpg")
    ),

    "Meningioma": glob.glob(
        os.path.join(EXAMPLES_DIR, "meningioma*.jpg")
    ),

    "Pituitary": glob.glob(
        os.path.join(EXAMPLES_DIR, "pituitary*.jpg")
    )
}

cols = st.columns(4)

for i, (tumor, paths) in enumerate(example_images.items()):

    with cols[i]:

        st.subheader(tumor)

        if not paths:
            st.caption("No example images found.")

        for img_path in paths[:2]:

            st.image(
                img_path,
                width="stretch"
            )