import streamlit as st
from img_classification import teachable_machine_classification
st.title("Image Classification with Google's Teachable Machine")
st.header("Pneumonia Smart Detector")
st.text("Upload a chest X-ray image for image classification as pneumonia or normal")



uploaded_file = st.file_uploader("Choose a chest X-ray image", type="jpeg")
if uploaded_file is not None:
        image = image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = teachable_machine_classification(image, "keras_model.h5")
        if label == 0:
            st.write("The chest X-ray is normal")
        else:
            st.write("The chest X-ray has pneumonia")

