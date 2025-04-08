import io
import streamlit as st
from paddleocr import PaddleOCR  # Импортируем PaddleOCR
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
    # Инициализация PaddleOCR с китайским языковым пакетом
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 'ch' для китайского языка
    result_ocr = ocr.ocr(img, cls=True)  # cls=True для улучшения распознавания углов текста

    st.write('**Результаты распознавания:**')
    
    # Выводим распознанный текст
    for line in result_ocr[0]:
        st.write(line[1])  # line[1] — это текст, распознанный на изображении
