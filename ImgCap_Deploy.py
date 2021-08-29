# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 23:27:49 2021

@author: Abhishek
"""

import streamlit as st
import warnings
warnings.filterwarnings("ignore")
# EDA Pkgs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('darkgrid')

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import models
import pickle
import cv2


#encoding_test = pickle.load( open( "encoding_test_dict.p", "rb" ))
#print(encoding_test['2271755053_e1b1ec8442.jpg'])

#pip install keras==2.6.0
import keras
print(keras. __version__)

#import os
#os.chdir(r'F:\Machine Learning\AI ML\Case Studies\DL\ImageCaption\Deployment')
#print(os.getcwd())



#To Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)


STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """



def main():
    
    model = load_model('Image_Caption_Model1.h5')
    encoding_test = pickle.load( open( "encoding_test_dict.p", "rb" ))
    wordtoix = pickle.load( open( "wordtoix.p", "rb" ) )
    ixtoword = pickle.load( open( "ixtoword.p", "rb" ) )
    max_len = 38
    
    def greedySearch(photo):
        in_text = 'startseq'
        for i in range(max_len):
            sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
            sequence = pad_sequences([sequence], maxlen=max_len)
            yhat = model.predict([photo,sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = ixtoword[yhat]
            in_text += ' ' + word
            if word == 'endseq':
                break

        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final
    
    def get_pic(pic_name,pic):
        image = encoding_test[pic_name].reshape((1,2048))
        x=plt.imread(pic)
        plt.imshow(x)
        plt.show()
        st.title("Selected Image :")
        st.image(pic , caption='Image')
        print("Greedy Search:",greedySearch(image))
        st.title("The Generated Caption is :")
        st.write(greedySearch(image))
        
    
    
    image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
    if image_file is not None:
        pic_name=image_file.name        
        pic = image_file
        #st.write(type(image_file))
        
        file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
        st.write(file_details)  
        get_pic(pic_name , pic)
    

    
    
    
    st.sidebar.header("About App")
    st.sidebar.info("Image captioning model that generate a sentence description for an image. \n Our model will take an image as input and generate an English sentence as output, describing the contents of the image.")
    st.sidebar.text("Built with Streamlit")
    st.sidebar.subheader("Scatter-plot setup")
    
    if st.button("Thanks"):
        st.balloons()



# To RUn : F:\Machine Learning\AI ML\Case Studies\DL\ImageCaption\Deployment>streamlit run ImgCap_Deploy.py
# Go to directory in cmd and do : streamlit run ImgCap_Deploy.py




if __name__ == '__main__':
    main()

