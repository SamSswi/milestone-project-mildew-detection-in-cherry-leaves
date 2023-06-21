import streamlit as st

def project_hypothesis_body():
    st.subheader('Project Hypotheses and Validations')

    st.success(
        f'* Hypothesis 1: There are visible patterns in the cherry leaf images that can be used to distinguish between healthy and powdery mildew affected leaves.\n'
        f'   * Validation: The Average Image and Variability Image revealed subtle hints the powdery mildew-affected leaves have pale speckles on them The Difference between Averages study did not reveal any clear pattern to differentiate between the two labels. An Image Montage shows that typically the cherry leaves that have powdery mildew, have superficial white growths and in some cases, their shape is deformed.'
    )

    st.success(
        f'* Hypothesis 2: Neural Networks can effectively map the relationships between the features extracted from cherry leaf images and the corresponding labels (healthy or powdery mildew).\n'
        f'   * Validation: A Convolutional Neural Network was used to map the relationships between the features extracted from cherry leaf images and the corresponding labels (healthy or powdery mildew). The usage of a Convolutional Neural Network was justified by its main advantages as efficient image processing, high accuracy rates, robustness to noise, and automated feature extraction. The trained Neural Network has an accuracy of 99.88% on the unseen data from the test set.\n'
    )

    st.success(
        f"* Hypothesis 3: Reducing the image shape to smaller dimensions (e.g., 100x100 or 50x50) will not significantly impact the model's performance in terms of accuracy.\n"
        f'   * Validation: The model was trained on images resized to 100x100px, and the accuracy of its performance on new unknown data was 99.88%.\n'
    )

    st.success(
        f'* Hypothesis 4: The model performance meets the project requirement of 97% accuracy, as agreed upon with the client.\n'
        f'   * Validation: After the evaluation of the model on a separate test dataset, the accuracy of the model was 99.88%. These metrics satisfy the requirement of 97% accuracy.\n'
    )