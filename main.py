import io
import streamlit as st
from transformers import pipeline
from PIL import Image


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


st.title('Распознай китайский текст с изображения!')
img = load_image()

result = st.button('Распознать изображение')
if result:
    # Замените модель на китайскую модель OCR
    captioner = pipeline("image-to-text", model="microsoft/layoutlmv3-base-chinese")
    text = captioner(img)
    st.write('**Результаты распознавания:**')
    st.write(text[0]["generated_text"])
