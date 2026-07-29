import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

IMG_SIZE = 100

st.title("Dogs vs. Cats Classifier")
st.caption("CNN trained on 24,946 grayscale images. Project 6.")

@st.cache_resource
def get_model():
    return load_model('dogs_cats_cnn.keras')

model = get_model()

uploaded = st.file_uploader("Upload a photo", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, width=300)

    # ---- YOUR FOUR PREPROCESSING STEPS GO HERE ----
   
    # 1. read the file from disk, already in grayscale
    arr = img.convert('L')

    # 2. resize to match training images
    arr = arr.resize((100, 100))

    # 3. scale pixels the same way you scaled X
    arr = np.array(arr) / 255.0

    # 4. reshape to a batch of one: (1, 100, 100, 1)
    arr = arr.reshape(1, 100, 100, 1)
   
  
    prob = float(model.predict(arr).flatten()[0])
    label = "Cat" if prob > 0.5 else "Dog"

    st.metric("Model output", f"{prob:.3f}")
    st.write(f"**Model says:** {label}")
    st.info("This number is the model's confidence, not its correctness.")
