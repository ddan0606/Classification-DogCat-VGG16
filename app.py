import streamlit as st
import tensorflow as tf

from utils import import_and_predict
from PIL import Image

st.title("PHÂN LOẠI HÌNH ẢNH VỚI VGG-16")
st.header("PHÂN LOẠI CHÓ MÈO")
st.text("")
file = st.file_uploader('Vui lòng tải lên ảnh liên quan đến chó hoặc mèo', type = ['jpeg', 'jpg', 'png'])
model = tf.keras.models.load_model(r'C:\Users\nguye\OneDrive - Ho Chi Minh City University of Foreign Languages and Information Technology - HUFLIT\Documents\Computer Vision\vgg16.hdf5')

if file is None:
    st.text("")
else:
    image = Image.open(file)
    st.image(image, use_column_width = True)
    predictions = import_and_predict(image, model)
