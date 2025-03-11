import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
import os
from utils import load_image, tensor_to_image
from style_transfer import dual_style_transfer, rot_style_transfer


def main():
    st.title("Neural Style Transfer Demo")

    # Sidebar for mode selection
    st.sidebar.title("Settings")
    mode = st.sidebar.radio(
        "Select Mode",
        ["Dual Style Transfer", "Rotated Style Features"]
    )

    # Upload content image
    st.sidebar.header("Upload Images")
    content_file = st.sidebar.file_uploader(
        "Choose a Content Image", type=["png", "jpg", "jpeg"])

    # Upload style image(s)
    style_file = st.sidebar.file_uploader(
        "Choose a Style Image", type=["png", "jpg", "jpeg"])

    # Additional parameters
    st.sidebar.header("Parameters")
    alpha = st.sidebar.slider("Content Weight (Alpha)", 0.0, 2.0, 1.0)

    # Process button
    process = st.sidebar.button("Generate Styled Image")

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.header("Content Image")
        if content_file is not None:
            content_img = Image.open(content_file).convert('RGB')
            st.image(content_img, use_container_width=True)

    with col2:
        st.header("Style Image")
        if style_file is not None:
            style_img = Image.open(style_file).convert('RGB')
            st.image(style_img, use_container_width=True)

    if process and content_file is not None and style_file is not None:
        # Results section
        st.header("Results")
        with st.spinner("Generating styled image..."):
            try:
                content_img = load_image(content_file)
                style_img = load_image(style_file)

                if mode == "Dual Style Transfer":
                    # Call your dual style transfer function
                    output_tensor = dual_style_transfer(
                        content_img, style_img, content_weight=alpha)

                    # Convert output tensor to image and display
                    output_img = tensor_to_image(output_tensor)
                    st.image(
                        output_img, caption="Dual Style Transfer Result", use_container_width=True)

                else:  # Rotated Style Features mode
                    # Call your rotated style features function
                    output_tensor1, output_tensor2 = rot_style_transfer(
                        content_img, style_img, content_weight=alpha)

                    # Convert output tensors to images and display
                    output_img1 = tensor_to_image(output_tensor1)
                    output_img2 = tensor_to_image(output_tensor2)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(
                            output_img1, caption="Rotated Style Result 1", use_container_width=True)
                    with col2:
                        st.image(
                            output_img2, caption="Rotated Style Result 2", use_container_width=True)

            except Exception as e:
                st.error(f"Error generating styled image: {str(e)}")
                st.error("Please check your input images and try again")


if __name__ == "__main__":
    main()
