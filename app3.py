import streamlit as st
import os
from PIL import Image
import torch
import random
import cv2
import numpy as np
from options import test_options
from model import create_model
from dataloader import data_loader
from torchvision import transforms
from util.task import center_mask, four_mask, random_regular_mask, random_irregular_mask
# from util.task import random_freefrom_mask
# Define the directories
IMAGE_DIR = r'D:\dev\canet\interface\image'
MASK_DIR = r'D:\dev\canet\interface\mask'
RESULT_DIR = r"D:\dev\canet\interface\result"

# Create directories if they do not exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# Define mask types
MASK_TYPES = {
    "Center Mask": center_mask,
    "Four Mask": four_mask,
    "Random Regular Mask": random_regular_mask,
    "Random Irregular Mask": random_irregular_mask
   
}

st.title("Image Inpainting App")

# Upload multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
num_files_uploaded = len(uploaded_files)

mask_type = st.selectbox("Select Mask Type", list(MASK_TYPES.keys()))

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save the uploaded image in the 'image' directory
        img_path = os.path.join(IMAGE_DIR, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image
        st.image(img_path, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)
        
        # Generate and save the selected type of mask in the 'mask' directory
        if mask_type == 'center_mask':
            mask = center_mask(uploaded_file)

            mask_path = os.path.join(MASK_DIR, f'{mask_type.replace(" ", "_").lower()}_{uploaded_file.name}')
            mask.save(mask_path)
        
        # Display the mask
        st.image(mask_path, caption=f'{mask_type} Mask for {uploaded_file.name}', use_column_width=True)


    # Inpaint button
    if st.button('Inpaint'):
        # Here you can add your inpainting logic

        # get testing options
        opt = test_options.TestOptions().parse()
        
        opt.ntest = num_files_uploaded
        opt.results_dir = RESULT_DIR
        opt.how_many = num_files_uploaded
        opt.img_file = IMAGE_DIR
        opt.mask_file = MASK_DIR
        
        dataset = data_loader.dataloader(opt)
        dataset_size = len(dataset) * opt.batchSize
        print('testing images = %d' % dataset_size)
        
        # create a model
        model = create_model(opt)
        model.eval()

        # Perform inpainting
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()
            print(opt.ntest)

        # Display inpainted images
        for i in range(num_files_uploaded):
            img_name = uploaded_files[i].name.split('.')[0]  # Extract image name without extension
            
            # Create HTML code snippet for displaying images
            html_code = f"""
                <h3>Image {i+1}</h3>
                <table border="1" style="table-layout: fixed;">
                  <tr>
                    <td halign="center" style="word-wrap: break-word;" valign="top">
                      <p>
                        <a href="{IMAGE_DIR}/{img_name}_m.png">
                          <img src="{RESULT_DIR}\{img_name}_m.png" style="width:256px">
                        </a><br>
                        <p>img_m</p>
                      </p>
                    </td>
                    <td halign="center" style="word-wrap: break-word;" valign="top">
                      <p>
                        <a href="{IMAGE_DIR}/{img_name}_truth.png">
                          <img src="{RESULT_DIR}\{img_name}_truth.png" style="width:256px">
                        </a><br>
                        <p>img_truth</p>
                      </p>
                    </td>
                    <td halign="center" style="word-wrap: break-word;" valign="top">
                      <p>
                        <a href="{IMAGE_DIR}/{img_name}_out.png">
                          <img src="{RESULT_DIR}\{img_name}_out.png" style="width:256px">
                        </a><br>
                        <p>img_out</p>
                      </p>
                    </td>
                  </tr>
                </table>
            """
            # Display the HTML code snippet
            st.markdown(html_code, unsafe_allow_html=True)

        # Display success message
        st.write("Images have been inpainted!")
