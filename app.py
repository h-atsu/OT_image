from matplotlib.pyplot import imshow
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils import transform

st.title('Optimal transport image color transfer')

uploaded_files = st.file_uploader(
    'Choose a image file [(input, reference) order]', accept_multiple_files=True)

if len(uploaded_files) == 2:
    image_in, image_ref = Image.open(
        uploaded_files[0]), Image.open(uploaded_files[1])
    imarray_in = np.array(image_in)
    imarray_ref = np.array(image_ref)

    col1, col2 = st.columns(2)

    with col1:
        st.header("input image")
        st.image(image_in, use_column_width=True)

    with col2:
        st.header("reference image")
        st.image(image_ref, use_column_width=True)

    st.header("color transfered image")
    final_img = transform(imarray_in, imarray_ref)
    print(final_img)
    st.image(final_img, use_column_width=True)
