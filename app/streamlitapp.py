# dependencies
import streamlit as st
import os
import imageio

import tensorflow as tf 
from utils import load_data,num_to_char
from modelutil import load_model
import numpy as np


# set the layout
st.set_page_config(layout='wide')

# sidebar
with st.sidebar:
    st.image('https://cdn.dribbble.com/users/508588/screenshots/16476655/tv_m302_08.jpg')
    st.title('A Lip Reading App')
    st.info('This application is developed with reference to the LipNet deep learning model')

st.title('Mind Your Lips 🙊')

# generating the list of options or videos
options=os.listdir(os.path.join('..','data','s1'))
selected_video = st.selectbox('Choose Video',options)

# generate 2 cols
col1,col2=st.columns(2)

if options:
    with col1:
        st.info('The video below displays mp4 format converted video')
        file_path=os.path.join('..','data','s1',selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y') # changing the file format to be able to get rendered by the streamlit app
        
        #rendering the video
        video = open('test_video.mp4','rb')
        video_bytes=video.read()
        st.video(video_bytes)

    with col2:
        st.info('The data gettting which the ML Model sees👀')
        video,annotations=load_data(tf.convert_to_tensor(file_path))

        fv = (video.numpy().astype(np.uint8) * 210).squeeze()
        imageio.mimsave('./animation.gif', fv, duration=100) 
        st.image('animation.gif',width=400)

        st.info('Output of the ML Model as tokens🪙')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video,axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)
        
        st.info('Token Decoded into Words🖥️')
        converted_prediction=tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)


