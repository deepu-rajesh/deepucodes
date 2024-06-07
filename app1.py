import streamlit as st
import os
from PIL import Image, ImageOps
from options import test_options
from model import create_model
from dataloader import data_loader
# Define the directories
IMAGE_DIR = r'D:\dev\canet\interface\image'
MASK_DIR = r'D:\dev\canet\interface\mask'

opt = test_options.TestOptions().parse()

# Create directories if they do not exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

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
        opt.results_dir = r"D:\dev\canet\interface\result"
        opt.how_many = num_files_uploaded
        opt.img_file = r"D:\dev\canet\interface\image"
        opt.mask_file = r"D:\dev\canet\interface\mask"
        RESULT_DIR = r"D:\dev\canet\interface\result"
        dataset = data_loader.dataloader(opt)
        dataset_size = len(dataset) * opt.batchSize
        print('testing images = %d' % dataset_size)
        # create a model
        model = create_model(opt)
        model.eval()
        # create a visualizer
        #visualizer = visualizer.Visualizer(opt)

        for i, data in enumerate(dataset):
            #start = time.time()

            model.set_input(data)
            model.test()
            print(opt.ntest)
            #####################################################
            ####################################################
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
            ###########################################
            ##########################################
        st.write("Inpainting process goes here...")
            # For example, display a message (you can replace this with actual inpainting code)
        st.write("Images have been inpainted!")
