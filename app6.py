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

# Header
header_image = "https://socs.mgu.ac.in/wp-content/uploads/2017/10/cropped-school-of-computer-sciences-logo.png"
header_image2 = "https://dca.cusat.ac.in/img/logo.png"
#st.image(header_image, use_column_width=True)
# st.image(header_image, width=150)
# st.image(header_image2, width=150)
# st.image([header_image, header_image2], width=250)
# Display header images side by side
col1, col2 = st.columns(2)
with col1:
    st.image(header_image, width=250)
with col2:
    st.image(header_image2, width=250)
st.markdown("# Context Aware Image Inpainting - A Transformer Based Approach")

# Upload multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
num_files_uploaded = len(uploaded_files)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save the uploaded image in the 'image' directory
        img_path = os.path.join(IMAGE_DIR, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Generate and save a binary mask of size 256x256 in the 'mask' directory
        mask = Image.new('1', (256, 256), 0)  # Create a blank mask (all black)
        mask_center = Image.new('1', (128, 128), 1)  # Create a white square (binary mask center)
        mask.paste(mask_center, (64, 64))  # Paste the white square in the center of the mask

        mask_path = os.path.join(MASK_DIR, f'binary_mask_{uploaded_file.name}')
        mask.save(mask_path)
        
        # Display the input image and its mask in a structured layout
        st.write(f"Input and Mask for {uploaded_file.name}:")
        cols = st.columns(2)
        cols[0].image(img_path, caption='Uploaded Image', use_column_width=True)
        cols[1].image(mask_path, caption='Binary Mask', use_column_width=True)
        
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

            # Add download button for each inpainted image
            st.download_button(
                label="Download Inpainted Image",
                data=open(out_path, 'rb').read(),
                file_name=f"{img_name}_out.png"
            )

# Footer
st.markdown("---")
st.markdown("<center>Copyright © 2024. All rights reserved.</center>", unsafe_allow_html=True)
st.text("  PROJECT DONE IN COLLABORATION WITH DEPARTMENT OF COMPUTER APPLICATIONS, CUSAT")
