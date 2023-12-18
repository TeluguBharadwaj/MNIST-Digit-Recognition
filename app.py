# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:11:21 2023

@author: telug
"""
import subprocess

# Install streamlit_drawable_canvas
subprocess.run(["pip", "install", "streamlit_drawable_canvas"])

from streamlit_drawable_canvas import st_canvas
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import cv2
import torchvision

# Load the pre-trained model
Network = torch.load('model_torch_MNIST_bh.chk')

# Specify canvas parameters in the application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 9)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

def process_and_predict_image(canvas_data):
    input_numpy_array = np.array(canvas_data)
    input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
    input_image.save('user_input.png')
    input_image_gs = input_image.convert('L')
    input_image_gs_np = np.asarray(input_image_gs.getdata()).reshape(200, 200)

    # Assuming the correct size is 101250, reshape it to (225, 450)
    input_image_gs_np = input_image_gs_np.reshape(200, 200)

    input_image_gs.save('temp_for_cv2.jpg')
    image = cv2.imread('temp_for_cv2.jpg', 0)
    height, width = image.shape
    x, y, w, h = cv2.boundingRect(image)
    ROI = image[y:y+h, x:x+w]
    mask = np.zeros([ROI.shape[0]+10, ROI.shape[1]+10])
    x = mask.shape[0] // 2 - ROI.shape[0] // 2
    y = mask.shape[1] // 2 - ROI.shape[1] // 2
    mask[y:y+h, x:x+w] = ROI
    output_image = Image.fromarray(mask)
    compressed_output_image = output_image.resize((22, 22), Image.BILINEAR)
    convert_tensor = torchvision.transforms.ToTensor()
    tensor_image = convert_tensor(compressed_output_image)
    tensor_image = tensor_image / 255.
    tensor_image = torch.nn.functional.pad(tensor_image, (3, 3, 3, 3), "constant", 0)
    convert_tensor = torchvision.transforms.Normalize((0.1307), (0.3081))
    tensor_image = convert_tensor(tensor_image)

    im = Image.fromarray(tensor_image.detach().cpu().numpy().reshape(28, 28), mode='L')
    im.save("processed_tensor.png")
    plt.imsave('processed_tensor.png', tensor_image.detach().cpu().numpy().reshape(28, 28), cmap='gray')

    device = 'cpu'
    with torch.no_grad():
        output0 = Network(torch.unsqueeze(tensor_image, dim=0).to(device=device))
        certainty, output = torch.max(output0[0], 0)
        certainty = certainty.clone().cpu().item()
        output = output.clone().cpu().item()
        certainty1, output1 = torch.topk(output0[0], 3)
        certainty1 = certainty1.clone().cpu()
        output1 = output1.clone().cpu()

    return certainty, output, certainty1, output1

def main():
    st.write('# MNIST Digit Recognition')
    st.write('## Using a CNN `PyTorch` model')

    activity = ['Little Description', 'Prediction', 'About']
    choice = st.sidebar.selectbox('Choose an Activity', activity)

    if choice == 'Little Description':
        st.subheader("ORIGINAL DATA SOURCE")
        st.text("MNIST is a subset of a larger set available from NIST")
        st.markdown("http://yann.lecun.com/exdb/mnist/")
        st.text("The MNIST database of handwritten digits has a training set of 60,000")
        st.text("The MNIST database of handwritten digits has a test set of 10,000")

    if choice == 'Prediction':
        st.subheader("Digit recognition")

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color='#FFFFFF',
            background_color='#000000',
            update_streamlit=realtime_update,
            height=200,
            width=200,
            drawing_mode='freedraw',
            key="canvas",
        )

        # Do something interesting with the image data and paths
        if canvas_result.image_data is not None:
            certainty, output, certainty1, output1 = process_and_predict_image(canvas_result.image_data)

            # Display prediction results or any other information as needed
            st.write('### Prediction')
            st.write('### ' + str(output))
            st.write('### Certainty')
            st.write(str(certainty1[0].item()*100) + '%')
            st.write('### Top 3 candidates')
            st.write(str(output1))
            st.write('### Certainties')
            st.write(str(certainty1*100))

    if choice == 'About':
        st.subheader("Digit Recognition App made with Streamlit by Bharadwaj")
        st.info("Email: telugu.bharadwaj@gmail.com")

if __name__ == '__main__':
    main()
