import streamlit as st

# The design of this dashboard page is inspired by the Quick Project Summary page From the Code Institute Malaria Detector sample project.
# https://malaria-predictor.onrender.com/

def project_summary_page_body():
    """
    Displays the project summary page of the Cherry Leaves Powdery Mildew Detector.
    """
    st.subheader('Quick Project Summary')

    st.info(
        '###### General Information \n'
        '* Cherry leaf powdery mildew, also known as cherry powdery mildew or cherry leaf blight, is a fungal disease that affects various species of cherry trees, including sweet cherries (Prunus avium) and sour cherries (Prunus cerasus). \n'
        '* On leaves, powdery mildew appears as patches of white, powdery or felt-like fungal growth. Severely affected leaves and shoots are often puckered or distorted.\n'
        '* Season long disease control is critical to minimize overall disease pressure in the orchard and consequently to protect developing fruit from accumulating spores on their surfaces.\n'
        '###### Project Dataset \n'
        '* The available dataset contains 4208 images of healthy cherry tree leaves and cherry tree leaves infected with powdery mildew.'
    )

    st.warning(
        '* For additional information, please visit and **read** the [Project README file](https://malaria-predictor.onrender.com/)'
    )

    st.success(
        '###### Business Requirements \n'
        '* The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew. \n'
        '* The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.\n')