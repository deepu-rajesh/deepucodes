import streamlit as st
import os
from PIL import Image
from options import test_options
from model import create_model
from dataloader import data_loader

# Define the directories
IMAGE_DIR = r'D:\dev\canet\interface\image'
MASK_DIR = r'D:\dev\canet\interface\mask'
RESULT_DIR = r"D:\dev\canet\interface\result"

# Create directories if they do not exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

st.title("Image Inpainting App")

# Upload multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
num_files_uploaded = len(uploaded_files)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save the uploaded image in the 'image' directory
        img_path = os.path.join(IMAGE_DIR, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image
        st.image(img_path, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)
        
        # Generate and save a binary mask of size 256x256 in the 'mask' directory
        mask = Image.new('1', (256, 256), 0)  # Create a blank mask (all black)
        mask_center = Image.new('1', (128, 128), 1)  # Create a white square (binary mask center)
        mask.paste(mask_center, (64, 64))  # Paste the white square in the center of the mask

        mask_path = os.path.join(MASK_DIR, f'binary_mask_{uploaded_file.name}')
        mask.save(mask_path)
        
        # Display the mask (optional)
        st.image(mask_path, caption=f'Binary Mask for {uploaded_file.name}', use_column_width=True)

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

        # Display success message
        st.write("Images have been inpainted!")

        # Display the inpainted results in a structured layout
        for uploaded_file in uploaded_files:
            img_name = os.path.splitext(uploaded_file.name)[0]
            mask_path = os.path.join(RESULT_DIR, f"{img_name}_mask.png")
            out_path = os.path.join(RESULT_DIR, f"{img_name}_out.png")
            truth_path = os.path.join(RESULT_DIR, f"{img_name}_truth.png")

            st.write(f"Results for {img_name}:")
            cols = st.columns(3)
            
            cols[0].image(mask_path, caption='Mask', use_column_width=True)
            cols[1].image(out_path, caption='Output', use_column_width=True)
            cols[2].image(truth_path, caption='Ground Truth', use_column_width=True)
