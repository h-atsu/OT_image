import streamlit as st
import ot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from utils import transform, resize_img

st.set_page_config(
    page_title="Optimal transport image color transfer",
)

st.title('Optimal transport image color transfer')

col1, col2 = st.columns(2)

with col1:
    st.header('Source file')
    source_file = st.file_uploader('', key='source')
    if source_file is not None:
        source_img = resize_img(Image.open(source_file).convert('RGB'))
        st.image(source_img)


with col2:
    st.header('Target file')
    target_file = st.file_uploader('', key='target')
    if target_file is not None:
        target_img = resize_img(Image.open(target_file).convert('RGB'))
        st.image(target_img)


if source_file is not None and target_file is not None:
    st.header('After conversion')
    ret_img = transform(source_img, target_img)
    st.image(ret_img)
    output = io.BytesIO()
    ret_img.save(output, format='JPEG')
    image_jpg = output.getvalue()
    st.download_button(
        label="Download",
        data=image_jpg,
        file_name='result_img.jpg',
    )
