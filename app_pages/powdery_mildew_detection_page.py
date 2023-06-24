import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import base64
from datetime import datetime

# The design of this dashboard page is inspired by the Malaria Detection page From the Code Institute Malaria Detector sample project.
# https://malaria-predictor.onrender.com/

# the plot_predictions_probabilities function was taken from Code Institute Malaria Detector Sample Project
# https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/src/machine_learning/predictive_analysis.py

def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Plot prediction probability results
    """

    prob_per_class = pd.DataFrame(
        data=[0, 0],
        index={'powdery_mildew': 0, 'healthy': 1}.keys(),
        columns=['Probability']
    )
    prob_per_class.loc[pred_class] = pred_proba
    for x in prob_per_class.index.to_list():
        if x not in pred_class:
            prob_per_class.loc[x] = 1 - pred_proba
    prob_per_class = prob_per_class.round(3)
    prob_per_class['Diagnostic'] = prob_per_class.index

    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y=prob_per_class['Probability'],
        range_y=[0, 1],
        width=600, height=300, template='seaborn')
    st.plotly_chart(fig)

# the resize_input_image function was taken from Code Institute Malaria Detector Sample Project
# https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/src/machine_learning/predictive_analysis.py

def resize_input_image(img, version):
    """
    Reshape image to model image size
    """
    # image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")
    image_shape = image_shape = joblib.load(filename=f"outputs/{version}/image_shape.pkl")
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.ANTIALIAS)
    my_image = np.expand_dims(img_resized, axis=0)/255

    return my_image

# the load_model_and_predict function was taken from Code Institute Malaria Detector Sample Project
# https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/src/machine_learning/predictive_analysis.py

def load_model_and_predict(my_image, version):
    """
    Load and perform ML prediction over live images
    """

    model = load_model(f"outputs/{version}/cherry_mildew_detector_model.h5")

    pred_proba = model.predict(my_image)[0, 0]

    target_map = {v: k for k, v in {'powdery_mildew': 0, 'healthy': 1}.items()}
    pred_class = target_map[pred_proba > 0.5]
    if pred_class == target_map[0]:
        pred_proba = 1 - pred_proba

    st.write(
        f"The predictive analysis diagnosis for the leaf sample: **{pred_class.lower()}**")

    return pred_proba, pred_class

# the download_dataframe_as_csv function was taken from Code Institute Malaria Detector Sample Project
# https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/src/data_management.py
def download_dataframe_as_csv(df):
    """
    Converts a DataFrame to a CSV file and generates a download link for it.
    """
    datetime_now = datetime.now().strftime("%d%b%Y_%Hh%Mmin%Ss")
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = (
        f'<a href="data:file/csv;base64,{b64}" download="Report {datetime_now}.csv" '
        f'target="_blank">Download Report</a>'
    )
    return href

def powdery_mildew_detection_body():
    """
    Displays the mildew detection page of the Cherry Leaves Powdery Mildew Detector.
    """
    st.info('* The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.')
    st.write('* You can download a set of images containing pictures of both healthy leaves and leaves infected with powdery mildew by clicking [here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).')
    st.write('---')

    # the idea to use the 'st.file_uploader' function was taken from streamlit documentation
    # https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader
    uploaded_files = st.file_uploader("Upload leaf samples. You may select more than one.", accept_multiple_files=True, type=['jpg'])

    # The following if statement was adapted from Code Institute's Malaria Detection Sample Project
    # https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/app_pages/page_malaria_detector.py
    if uploaded_files is not None:
        data_list = []
        for uploaded_file in uploaded_files:

            img_pil = (Image.open(uploaded_file))
            st.info(f"Cherry Leaf Image: **{uploaded_file.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v1'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            data_dict = {"Name":uploaded_file.name, 'Result': pred_class }
            data_list.append(data_dict)
        
        df_report = pd.DataFrame(data_list)
                
        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)