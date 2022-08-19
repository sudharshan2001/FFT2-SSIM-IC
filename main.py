import streamlit as st
import numpy as np
from PIL import Image
# import time
import torch
from torchmetrics import StructuralSimilarityIndexMeasure

# '''
# I used the torch's FFT and SSIM because it's faster compared to others
# '''

def Bytes(Bytes) :
    # To Print the difference in size between original and compressed

    kilobytes = Bytes / 1024
    return kilobytes

# Load image
def load_image(image_file):
	img = Image.open(image_file)
	return img

st.subheader("Upload the Image")

filename = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

# last_time = time.time()

if filename is not None:

    A=np.asarray(load_image(filename))

    # Slice the dimension to extract R,G and B
    Red = A[:,:,0]
    Green = A[:,:,1]
    Blue  = A[:,:,2]

    info_list = [] # To store the info regarding fourier coefficient and size

    for i in np.arange(0.005,0.04,0.010): 

        # I chose 0.005 to 0.04 with 0.010 interval for fourier coefficients to 
        #     be kept after evaluating many images

        final_list = [] # To store inverse-FFT image

        # Looping through each dimension and FFT-ing it and exploring the coefficients
        for color in (Red, Green, Blue) :  
            
            fft_ = torch.fft.fft2(torch.Tensor(color))

            sorted_fft = np.sort(np.abs(fft_.reshape(-1))) 

            keep=torch.from_numpy(np.asarray(i)) # Choosing the coefficient and trying it out

            # Select the threshold
            thresh = sorted_fft[int(torch.floor((1-keep)*len(sorted_fft)))-1]
            filter_ = torch.abs(fft_)>thresh         
            abs_fft = fft_ * filter_   
            
            # Inverse FFT-ing 
            inverse_fft = torch.fft.ifft2(abs_fft).real

            # Storing IFFT of each stack
            final_list.append(inverse_fft.cpu().detach().numpy())

        # Stacking all RGB dimensions to create the final image
        modified = np.dstack((final_list[0],final_list[1],final_list[2])).astype('uint8')

        score = []
        for j in range(3):

            # Looping through original and Compressed image and finding SSIM

            dim_orig = A[:,:,j] #original image
            modified_orig = modified[:,:,j] # Compressed image
            score.append(StructuralSimilarityIndexMeasure(dim_orig, modified_orig)) # SSIM
        
        # Storing the Compressed image
        new_path = './compressed_images/'+filename.name.split('.')[0]+'_'+str(i)[:5]+'.jpg'

        Image.fromarray(modified).save(new_path)
        image_file = Image.open(new_path)
    
        info_list.append((i,sum(score)/3,Bytes(len(image_file.fp.read())),Bytes(filename.size)))

    # Choosing the best parameter for coefficient
    temp_list = [i for i in info_list if i[2]<i[3]]

    if temp_list is not None:
        new_list = temp_list[int(len(temp_list)/3)]
    else:
        new_list = [info_list[0]]

    percentage = (new_list[0]/info_list[-1][0])*100

    st.write('Percentage:' ,percentage)
    st.write('Size Before Compression (KB): ' ,new_list[3])
    st.write('Size After Compression (KB): ' ,new_list[2])

    for color in (Red, Green, Blue) :  
            
        fft_ = np.fft.fft2(color)
        sorted_fft = np.sort(np.abs(fft_.reshape(-1)))

        keep=new_list[0]

        thresh = sorted_fft[(int(np.floor((1-keep)*len(sorted_fft))))-1]

        filter_ = np.abs(fft_)>thresh         
        abs_fft = fft_ * filter_   

        inverse_fft = np.fft.ifft2(abs_fft).real
        final_list.append(inverse_fft)

    modified = np.dstack((final_list[0],final_list[1],final_list[2])).astype('uint8')
    
    col1, col2 = st.columns(2)
        
    with col1:
        st.header("Compressed Version"+str(np.asarray(Image.fromarray(modified)).shape))
        st.image(Image.fromarray(modified), width=250)

    with col2:
        st.header("Original Version"+str(np.asarray(load_image(filename)).shape))
        st.image(load_image(filename), width=250)

    # print("Time Taken: ", time.time() - last_time, " S")