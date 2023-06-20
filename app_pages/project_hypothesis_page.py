import streamlit as st

def project_hypothesis_body():
    st.title('Cherry Leaves Powdery Mildew Detector')
    st.header('Project Hypothesis and Validation')

    st.success(
        '* We suspect the leaves infected with powdery mildew have white patches of color somewhere on the leaf. That is the main difference between them and healthy leaves.\n'
        '\n'
        '* An Image Montage shows the leaves infected with powdery mildew usually have white speckles, a deformed shape and a yellower shade of green. The Average Image and Variability Image studies revealed subtle hints the powdery mildew infected leaves have pale speckles on them. The Difference between Averages study did not reveal any clear pattern to differentiate between the two labels.'
    )