import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    if predictions[0][0] >= 0.5:
        st.header("Dự đoán là Chó")
    else:
        st.header("Dự đoán là Mèo")
