import streamlit as st
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from numpy import asarray
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
from joblib import load
import gdown

import streamlit_authenticator as stauth
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image as imge
import keras
from tf_keras_vis.saliency import Saliency
from tf_keras_vis import utils as utils
from tf_keras_vis.utils import normalize
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

model_URLs ={'KNCFC.joblib':'https://drive.google.com/uc?id=1-8avHm0vfJbwXJ7nqBmRq3JoZbg3ZGBY',
 'surrogate_model.joblib': 'https://drive.google.com/uc?id=1-5Vyh7fC2T7X55e7RDsbbRXuHIjQAlRT',
 'XGBC.joblib' : 'https://drive.google.com/uc?id=1qJkWkMuml4e-1pvrArd9SVscAtO1fDtJ' }
@st.cache_resource
def load_model(model_name):
  gdown.download(model_URLs[model_name], model_name)
  return load(model_name)

def cropping(img):
  thresh = (img[:, :, 0]+img[:,:,1]+img[:,:,2])/3
  # Coordinates of non-black pixels.
  coords = np.argwhere(thresh > 10)
  # Bounding box of non-black pixels.
  x0, y0 = coords.min(axis=0)
  x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
  # Get the contents of the bounding box.
  img = img[x0:x1, y0:y1]
  img = cv2.resize(img, (150, 150))
  return img
# -- Set page config
apptitle = 'NeuroAID'

st.set_page_config(page_title=apptitle, page_icon=":brain:", layout = "wide")
# --- USER AUTHENTICATION ---
names = ["Tushar Mehta", "Administrator"]
usernames = ["tmehta", "admin"]

# load hashed passwords
file_path = "hashed_pw.pkl"
with open(file_path, "rb") as file:
    hashed_passwords = pickle.load(file)
#passwords = ["abc123", "abc123"]

authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "NeuroAID", "abcdef", cookie_expiry_days=0)
image = Image.open('NeuroAID//Banner.jpg')
st.image(image)

name, authentication_status, username = authenticator.login("NeuroAID Login", "main")

if authentication_status == None:
  st.warning("Please enter your username and password.")
  st.write(" :green[If you don't have an account and would like to signup, please send your name and email address to the NeuroAID administrator at mehtat124@gmail.com]")

if authentication_status == False:
  st.error("Username/password is incorrect")
  st.write(" :green[If you don't have an account and would like to signup, please send your name and email address to the NeuroAID administrator at mentat124@gmail.com]")

if authentication_status:
  st.markdown("<h1 style='text-align: center; color: blue;'>Brain Tumor Screening Diagnostics System</h1>", unsafe_allow_html=True)
  #st.title(":blue[Brain Tumor Screening Diagnostics System]")
  #Text
  st.write(" :green[Purpose: This App provides screening level diagnosis for brain tumors based on Magnetic Resonance Imaging (MRI) scans. There are two levels of diagnosis:]")
  st.markdown("""
  * Detection: Is there an evidence in the MRI for a presence of a tumor
  * Classification: What type of tumor is present
  """)
  Data_tab, Diagnostics_tab, Report_tab = st.tabs([" ## Training Data", "## XAI Diagnostics", "## Historical Record"])
  #-- sidebar
  authenticator.logout("Logout", "sidebar")
  st.sidebar.title(f"Welcome {name}")
  st.sidebar.markdown("## :blue[Patient Information] ")
  selected_patient = st.sidebar.text_input(':green[Name/ID]', value = name)
  selected_date = st.sidebar.date_input(':green[Date of Scan]', value="today")
  selected_history = st.sidebar.button(':green[Retrieve Previous Scans and Results]')
  st.sidebar.markdown("## :blue[Model Selector and Parameters] ")
  #-- Choose Diagnosis Type as Detection or Classification
  Selected_diagnosis = st.sidebar.selectbox(':green[Diagnosis Type]', ['Detection', 'Classification'])
  #-- Choose Model
  if (Selected_diagnosis == 'Detection'):
      selected_model = st.sidebar.selectbox(':green[Model]', ['Best Model (KNC-ResNet50)', 'KNC','Naive Bayes','Random Forest', 'Logistic Regression', 'XGBoost','CNN','VGG16', 'KNC-VGG16', 'KNC-ResNet50','KNC-Fused_Future(VGG16+ResNet50)' ])
  else:
      selected_model = st.sidebar.selectbox('Model', ['Best Model (XGBoost)', 'KNC', 'Naive Bayes', 'XGBoost', 'Random Forest', 'VGG16', 'KNC-VGG16', 'KNC-ResNet50','KNC-Fused_Future(VGG16+ResNet50)'])
  Model_Metrics_Selection = st.sidebar.checkbox(':green[Show Model Performance Metrics]')
  st.sidebar.markdown("## :blue[Explainability Parameters]")
  #-- Choose Explainability Type
  selected_explainability = st.sidebar.radio(':green[Explainability Type]', ['Cohort Level','Patient Level'])
  #-- Choose Model
  if (selected_explainability == 'Cohort Level'):
      selected_ex_display = st.sidebar.selectbox(':green[Display]', ['Feature Importance Pareto and Brain Heat Map','Feature Importance Pareto', 'Brain Heat Map'])
  else:
      selected_ex_display = st.sidebar.selectbox(':green[Display]', ['Tumor Saliency Map','Contrast Map'])
  selected_save = st.sidebar.button(':green[Save Record]')
  with Data_tab:
    image = Image.open("NeuroAID//DS1.jpg")
    st.image(image)
    image = Image.open("NeuroAID//DS2.jpg")
    st.image(image)
  with Diagnostics_tab:
    if (Selected_diagnosis == 'Detection'):
      #Load selected model
      if (selected_model == 'Best Model (KNC-ResNet50)'):
          Selectedmodel = load("KNCV.joblib")
          Model_option = 3
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_KNCR.jpg")
            st.image(image)
      elif (selected_model == 'KNC-VGG16'):
          Selectedmodel = load("KNCV.joblib")
          Model_option = 3
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_KNCV.jpg")
            st.image(image)
      elif(selected_model == 'KNC-Fused_Future(VGG16+ResNet50)'):
          Selectedmodel = load("KNCV.joblib")
          Model_option = 3
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_KNCF.jpg")
            st.image(image)
      elif (selected_model == 'KNC'):
          Selectedmodel = load("KNC.joblib")
          Model_option = 0
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_KNC.jpg")
            st.image(image)
      elif (selected_model == 'Naive Bayes'):
          Selectedmodel = load("NB.joblib")
          Model_option = 0
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_NB.jpg")
            st.image(image)
      elif (selected_model == 'Random Forest'):
          Selectedmodel = load("RF.joblib")
          Model_option = 0
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_RF.jpg")
            st.image(image)
      elif (selected_model == 'Logistic Regression'):
          Selectedmodel = load("LR.joblib")
          Model_option = 0
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_LR.jpg")
            st.image(image)
      elif (selected_model == 'XGBoost'):
          Selectedmodel = load("XGB.joblib")
          Model_option = 0
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_XGB.jpg")
            st.image(image)
      elif (selected_model == 'CNN'):
          Selectedmodel = load("CNN.joblib")
          Model_option = 1
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_CNN.jpg")
            st.image(image)
      elif (selected_model == 'VGG16'):
          Selectedmodel = load("VGG16.joblib")
          Model_option = 2
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_VGG16.jpg")
            st.image(image)
      else:
          Selectedmodel = load("KNCV.joblib")
          Model_option = 3
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_KNCR.jpg")
            st.image(image)
    else:
      #Load selected model
      if (selected_model == 'Best Model (XGBoost)'):
          Selectedmodel = load_model("XGBC.joblib")
          Model_option = 0
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_XGBC.jpg")
            st.image(image)
      elif (selected_model == 'KNC-ResNet50'):
          Selectedmodel = load("KNCVC.joblib")
          Model_option = 3
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_KNCRC.jpg")
            st.image(image)
      elif(selected_model == 'KNC-Fused_Future(VGG16+ResNet50)'):
          Selectedmodel = load_model("KNCVC.joblib")
          Model_option = 3
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_KNCFC.jpg")
            st.image(image)
      elif (selected_model == 'KNC'):
          Selectedmodel = load("KNCC.joblib")
          Model_option = 0
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_KNCC.jpg")
            st.image(image)
      elif (selected_model == 'Naive Bayes'):
          Selectedmodel = load("NBC.joblib")
          Model_option = 0
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_NBC.jpg")
            st.image(image)
      elif (selected_model == 'Random Forest'):
          Selectedmodel = load("RFC.joblib")
          Model_option = 0
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_RFC.jpg")
            st.image(image)
      elif (selected_model == 'XGBoost'):
          Selectedmodel = load("XGBC.joblib")
          Model_option = 0
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_XGBC.jpg")
            st.image(image)
      elif (selected_model == 'VGG16'):
          Selectedmodel = load("Vgg16C.joblib")
          Model_option = 2
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_XGBC.jpg")
            st.image(image)
      else:
          Selectedmodel = load("KNCVC.joblib")
          Model_option = 3
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("NeuroAID//Model_KNCVC.jpg")
            st.image(image)

    st.divider()

    #SubHeader
    st.subheader(':green[Uplaod the MRI for Screening Diagnosis]')
    st.write(':green[Please use the Browse button below to select the MRI scan file (jpg, png, gif) from your local drive]')
    #File Input
    uploaded_file = st.file_uploader("Upload Image")
    if uploaded_file is not None:
      if (Model_option == 0):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cropping (image)
        st.image(image, channels="BGR")
        resize = cv2.resize(image, (150,150))
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        nml = (gray-np.min(gray))/(np.max(gray)-np.min(gray))
        #remove background if it has been on the training data
        flatten = np.reshape(nml, (1,-1))
        #add in if statement here for the type of model
        pred = Selectedmodel.predict(flatten)
        pred_prob = Selectedmodel.predict_proba(flatten)
      elif (Model_option ==1):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cropping (image)
        st.image(image, channels="BGR")
        resize = cv2.resize(image, (150,150))
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        nml = (gray-np.min(gray))/(np.max(gray)-np.min(gray))
        #remove background if it has been on the training data
        input = np.reshape(nml, (1,150,150,1))
        #add in if statement here for the type of model
        ypred = Selectedmodel.predict(input)
        pred = np.argmax(ypred, axis=1)
        pred_prob = ypred
#        last_layer = Selectedmodel.layers[-1]
#        last_layer.activation = tf.keras.activations.linear
#        model = tf.keras.models.Model(inputs=Selectedmodel.inputs, outputs=[last_layer.output])
      elif (Model_option ==2):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cropping (image)
        st.image(image, channels="BGR")
        resize = cv2.resize(image, (150,150))
    #    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        nml = (resize-np.min(image))/(np.max(image)-np.min(image))
        #remove background <-- to do
        input = np.reshape(nml, (1,150,150,3))
        input = tf.image.resize(input, (150, 150))
        input = preprocess_input(input)
        ypred = Selectedmodel.predict(input)
        pred = np.argmax(ypred, axis=1)
        pred_prob = ypred
#        last_layer = Selectedmodel.layers[-1]
#        last_layer.activation = tf.keras.activations.linear
#        model = tf.keras.models.Model(inputs=Selectedmodel.inputs, outputs=[last_layer.output])
      else:
        base_model_Vgg16 = load("base_modelVGG16.joblib")
        base_model_ResNet50 = load("base_model_ResNet50.joblib")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cropping (image)
        st.image(image, channels="BGR")
        resize = cv2.resize(image, (150,150))
    #    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        nml = (resize-np.min(image))/(np.max(image)-np.min(image))
        input = np.reshape(nml, (1,150,150,3))
        input = tf.image.resize(input, (150, 150))
        input = preprocess_input(input)
        input = np.reshape(input, (1,150,150,3))
  #      im = imge.img_to_array(input)
  #      img = preprocess_input(np.expand_dims(im.copy(), axis=0))
#        if (selected_model == 'KNC-VGG16'):
        feature = base_model_Vgg16.predict(input)
        feature_np = np.array(feature)
        array1 = feature_np.flatten().reshape(1, -1)
        ypred = Selectedmodel.predict(array1)
        pred = np.argmax(ypred, axis=1)
        pred_prob = ypred
#        elif (selected_model == 'KNC-ResNet50'):
#          feature = base_model_ResNet50.predict(input)
#          feature_np = np.array(feature)
#          array1 = feature_np.flatten().reshape(1, -1)
#          ypred = Selectedmodel.predict(array1)
#          if (Selected_diagnosis == 'Classification'):
#            pred = np.argmax(ypred, axis =1)
#            pred_prob = ypred
#          else:
#            pred = np.argmax(ypred)
#            pred_prob = Selectedmodel.predict_proba(flatten)
#        else:
#          featurevgg16 = base_model_Vgg16.predict(input)
#          featurevgg16_np = np.array(featurevgg16)
#          array1vgg16 = featurevgg16_np.flatten().reshape(1, -1)
#          featureresnet50 = base_model_ResNet50.predict(input)
#          featureresnet50_np = np.array(featureresnet50)
#          array1resnet50 = featureresnet50_np.flatten().reshape(1, -1)
#          array1 = np.concatenate((array1vgg16, array1resnet50), axis=1)
#          ypred = Selectedmodel.predict(array1)
#          pred = np.argmax(ypred, axis=1)
#          pred_prob = ypred
      st.write("<h4 style='text-align: left; color: blue;'>For the uploaded image shown above, selected model was used to perform the screening analysis.</h4>", unsafe_allow_html = True)
      if (Selected_diagnosis == 'Detection'):
        if pred == 1:
          prob = 100*pred_prob[0][1]
          st.write(f"<h4 style='text-align: left; color: orange;'>There is ** tumor** detected with a probability of {prob:.2f}%.</h4>", unsafe_allow_html=True)
        if pred == 0:
          prob = 100*pred_prob[0][0]
          st.write(f"<h4 style='text-align: left; color: orange;'>There is ** no tumor** detected with a probability of {prob:.2f}%.</h4>", unsafe_allow_html=True)
        if (selected_explainability == 'Cohort Level'):
          if (selected_model == 'KNC' or selected_model=='Naive Bayes' or selected_model=='Logistic Regression' or selected_model == 'CNN' or selected_model=='VGG16' or selected_model == 'KNC-VGG16' or selected_model=='KNC-ResNet50' or selected_model == 'KNC-Fused_Future(VGG16+ResNet50)' or selected_model == 'Best Model(KNC-Resnet50)' ):
            surrogate_model = load("surrogate_model.joblib")
            importances = surrogate_model.feature_importances_
          else:
            importances = Selectedmodel.feature_importances_
          indices = np.argsort(importances)
          top10 = []
          top10 = indices[22491:22500]
          importance = plt.figure(figsize=(3,3))
          plt.title('Feature Importances')
          plt.yticks(range(len(top10)),[i for i in top10])
          plt.xlabel('Relative Importance')
          plt.barh(range(len(top10)), importances[top10], color='b', align='center')
          img_shape = resize.shape
          imp_reshaped = importances.reshape((150,150))
          heatmap, ax = plt.subplots(figsize=(3,3))
          ax.matshow(imp_reshaped, cmap=plt.cm.hot)
          plt.title("Pixel importances using impurity values") 
          if (selected_ex_display == 'Feature Importance Pareto and Brain Heat Map'):
            st.write("<h4 style='text-align: left; color: blue;'>For the uploaded image shown above explainability analyis was performed and the following plots show the tumorous areas.</h4>", unsafe_allow_html = True)
            st.pyplot(importance)
            st.pyplot(heatmap)
          elif (selected_ex_display == 'Feature Importance Pareto'):
            st.write("<h4 style='text-align: left; color: blue;'>For the uploaded image shown above explainability analyis was performed and the following plot shows the tumorous areas.</h4>", unsafe_allow_html = True)
            st.pyplot(importance)
          else:
            st.write("<h4 style='text-align: left; color: blue;'>For the uploaded image shown above explainability analyis was performed and the following plot shows the tumorous areas.</h4>", unsafe_allow_html = True)
            st.pyplot(heatmap)
        else:
          st.write("<h4 style='text-align: left; color: blue;'>For the uploaded image shown above explainability analyis was performed and the following plot shows the tumorous areas.</h4>", unsafe_allow_html = True)
          image = Image.open("NeuroAID//Patient_D.jpg")
          st.image(image)
      else:
        if (pred == 3):
          prob = 100*pred_prob[0][3]
          st.write(f"<h4 style='text-align: left; color: orange;'>There is a **Pituitary** type of tumor detected with a probability of {prob:.2f}%.</h4>", unsafe_allow_html=True)
        if (pred == 2):
          prob = 100*pred_prob[0][2]
          st.write(f"<h4 style='text-align: left; color: orange;'>There is a **Meningioma** type of tumor detected with a probability of {prob:.2f}%.</h4>", unsafe_allow_html=True)
        if (pred == 1):
          prob = 100*pred_prob[0][1]
          st.write(f"<h4 style='text-align: left; color: orange;'>There is a **Glioma** type of tumor detected with a probability of {prob:.2f}%.</h4>", unsafe_allow_html=True)
        if (pred == 0):
          prob = 100*pred_prob[0][0]
          st.write(f"<h4 style='text-align: left; color: orange;'>There is **no tumor** detected with a probability of {prob:.2f}%.</h4>", unsafe_allow_html=True)
        if (selected_explainability == 'Cohort Level'):
          if (selected_model == 'KNC' or selected_model=='Naive Bayes'or selected_model=='VGG16' or selected_model == 'KNC-VGG16' or selected_model=='KNC-ResNet50' or selected_model == 'KNC-Fused_Future(VGG16+ResNet50)' ):
            surrogate_model = load("surrogate_model.joblib")
            importances = surrogate_model.feature_importances_
          else:
            importances = Selectedmodel.feature_importances_
          indices = np.argsort(importances)
          top10 = []
          top10 = indices[22491:22500]
          importance = plt.figure(figsize=(3,3))
          plt.title('Feature Importances')
          plt.yticks(range(len(top10)),[i for i in top10])
          plt.xlabel('Relative Importance')
          plt.barh(range(len(top10)), importances[top10], color='b', align='center')
          img_shape = resize.shape
          imp_reshaped = importances.reshape((150,150))
          heatmap, ax = plt.subplots(figsize=(3,3))
          ax.matshow(imp_reshaped, cmap=plt.cm.hot)
          plt.title("Pixel importances using impurity values")          
          if (selected_ex_display == 'Feature Importance Pareto and Brain Heat Map'):
            st.write("<h4 style='text-align: left; color: blue;'>For the uploaded image shown above explainability analyis was performed and the following plots show the tumorous areas.</h4>", unsafe_allow_html = True)
            st.pyplot(importance)
            st.pyplot(heatmap)
          elif (selected_ex_display == 'Feature Importance Pareto'):
            st.write("<h4 style='text-align: left; color: blue;'>For the uploaded image shown above explainability analyis was performed and the following plot shows the tumorous areas.</h4>", unsafe_allow_html = True)
            st.pyplot(importance)
          else:
            st.write("<h4 style='text-align: left; color: blue;'>For the uploaded image shown above explainability analyis was performed and the following plot shows the tumorous areas.</h4>", unsafe_allow_html = True)
            st.pyplot(heatmap)
        else:
          st.write("<h4 style='text-align: left; color: blue;'>For the uploaded image shown above explainability analyis was performed and the following plot shows the tumorous areas.</h4>", unsafe_allow_html = True)
          image = Image.open("NeuroAID//Patient_C.jpg")
          st.image(image)
    else:
      st.write(':green[If you do not have an image, you can download a sample MRI image from image library,] https://openneuro.org/')
##  when the records are saved to the database Save
  with Report_tab:
##  Display the retrieved records here
    if (selected_explainability == 'Cohort Level'):
      if (selected_ex_display == 'Feature Importance Pareto and Brain Heat Map'):
        if (Selected_diagnosis == 'Detection'):
          image = Image.open("NeuroAID//Cohort_D.jpg")
          st.image(image)
        else:
          image = Image.open("NeuroAID//Cohort_C.jpg")
          st.image(image)
    elif (selected_explainability == 'Patient Level'):
      if (selected_ex_display == 'Contrast Map' ):
        if (Selected_diagnosis == 'Detection'):
          image = Image.open("NeuroAID//Patient_D.jpg")
          st.image(image)
        else:
          image = Image.open("NeuroAID//Patient_C.jpg")
          st.image(image)

