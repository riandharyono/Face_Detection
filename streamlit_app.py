
import streamlit as st
import cv2
from PIL import Image
import numpy as np

FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier('haarcascade_eye.xml')
SMILE_CASCADE = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect_faces(image):
    new_img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
        smile = SMILE_CASCADE.detectMultiScale(roi_gray, 2, 4)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    return img, faces

def main():
    """Face Detection App"""
    st.set_page_config(
        page_title="Face Detection App",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Face Detection App")
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #f5f5f5;
            padding-top: 1rem;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        .st-bq {
            padding: 0.75rem 1.25rem;
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
            background-color: #ffffff;
            border-radius: 0.25rem;
            box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Home":
        st.subheader("Face Detection")
        st.write("Upload an image and detect faces, eyes, and smiles.")

        col1, col2 = st.columns(2)
        with col1:
            image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
            if image_file is not None:
                image = Image.open(image_file)
                st.image(image, use_column_width=True)

        with col2:
            if st.button("Process"):
                if image_file is not None:
                    result_img, result_face = detect_faces(image)
                    st.success("Found {} faces".format(len(result_face)))
                    st.image(result_img, channels="BGR", use_column_width=True)
                else:
                    st.warning("Please upload an image.")

    elif choice == "About":
        st.subheader("About Face Detection App")
        st.markdown("Built with Streamlit and OpenCV for Data Scientist Project")
        st.text("Hello, I'm Rian Dwi Haryono. now, I'm learning Machine Learning by making a Face Detection Application project")
        

if __name__ == "__main__":
    main()
