import streamlit as st
from PIL import Image
import keras 



st.title("Diagnose COVID-19 based on X-ray images by AI")
link=' [based on our model](https://pesquisa.bvsalud.org/global-literature-on-novel-coronavirus-2019-ncov/resource/pt/covidwho-1458945?lang=en#main_container)'
st.markdown(link,unsafe_allow_html=True)


from BackEndFunctions import CNNClassifier #This is the name of our file/function
uploaded_file = st.file_uploader("Please upload image", type=['png','jpeg']) #Add file uploader to take the user's input. limited to .png files only
if uploaded_file is not None:
        image = Image.open(uploaded_file)                           #Assign the uploded image into a variable dubbed image     
        st.image(image, caption='Image uploading, please wait', use_column_width=True)     # let user read some word while png is displayed
        st.write("")                
        st.write(".... Working on it")                                                            # Let user know we are working          
        label = CNNClassifier(image, 'model.h5')                                               #Call the function and then display the test result based on outcome of predcition 
        if label == 0:      
            st.write("Negative")
        else:
            st.write("Positive")
            
