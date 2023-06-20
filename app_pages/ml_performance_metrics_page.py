import streamlit as st
import matplotlib.pyplot as plt
import joblib
import pandas as pd

def ml_perfomance_metrics_body():
    st.title('Cherry Leaves Powdery Mildew Detector')
    version = 'v1'
    st.header('Train, Validation and Test Set: Label Frequencies')
    label_distribution = plt.imread(f'outputs/{version}/labels_distribution_plot.png')
    st.image(label_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    st.header('Model History')
    # https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/app_pages/page_ml_performance.py
    col1, col2 = st.columns(2)
    with col1:
        model_accuracy = plt.imread(f'outputs/{version}/model_training_acc.png')
        st.image(model_accuracy, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f'outputs/{version}/model_training_losses.png')
        st.image(model_loss, caption='Model Training Losses')
    st.write('---')
    st.header('Generalised Performance on Test Set')
    test_set_evaluation = joblib.load(f'outputs/{version}/model_evaluation.pkl')
    st.dataframe(pd.DataFrame(test_set_evaluation, index=['Loss', 'Accuracy']))
    