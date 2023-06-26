import streamlit as st
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import random


# The design of this dashboard page is inspired by the Cells Visualiser
# page From the Code Institute Malaria Detector sample project.
# https://malaria-predictor.onrender.com/
def leaves_visualizer_body():
    """
    Displays the leaves visualizer page in the Streamlit app.
    """

    st.subheader('Cherry Leaves Visualizer')
    st.info(
        'The client is interested in conducting a study to visually '
        'differentiate a cherry leaf that is healthy from one that '
        'contains powdery mildew.')

    # The usage of the 'checkbox' elements was inspired from the streamlit
    # documentation
    # https://docs.streamlit.io/library/api-reference/widgets/st.checkbox
    difference_avr_var = st.checkbox(
        'Difference between the average and variability image')
    version = 'v1'
    if difference_avr_var:
        avg_var_healthy = plt.imread(
            f'outputs/{version}/avg_and_var_image_healthy.png')
        avg_var_powdery_mildew = plt.imread(
            f'outputs/{version}/avg_and_var_image_powdery_mildew.png')
        st.warning(
            'We notice the average and the image variability images reveal '
            'subtle patterns where we could intuitively differentiate healthy '
            'cherry leaves from cherry leaves with powdery mildew. '
            'The healthy '
            'leaves exhibit a hightened intensity of green color while the '
            'leaves infected with powdery mildew displays a multitude of pale '
            'speckles on the leaf surface. The pattern can be observed when '
            'comparing the image variability images of the two labels.')
        st.image(
            avg_var_healthy,
            caption='Average and Variability Image - Healthy Leaves')
        st.image(
            avg_var_powdery_mildew,
            caption='Average and Variability Image - Powdery Mildew '
                    'Infected Leaves'
            )
        st.write('---')

    difference_avr_labels = st.checkbox(
        'Difference between the average image of healthy leaves and '
        'powdery mildew affected leaves')
    if difference_avr_labels:
        difference_labels = plt.imread(
            f'outputs/{version}/difference_image.png')
        st.warning(
            "We notice this study didn't show patterns where we could "
            "intuitively differentiate between the average images of "
            "the two labels.")
        st.image(
            difference_labels,
            caption='Difference between the average images')
        st.write('---')

    image_montage = st.checkbox('Image Montage')
    if image_montage:
        st.write(
            "* To create of refresh the montage click on the "
            "'Create Montage' button.")
        dir_path = 'inputs/cherry_leaves/cherry-leaves/validation'
        label_option = st.selectbox(
            'Please select a label',
            os.listdir(f'{dir_path}'))
        if st.button('Create Montage'):
            image_montage_function(
                dir_path=dir_path,
                label=label_option,
                nrows=5, ncols=3,
                figsize=(10, 10))
            st.write('---')
        else:
            st.write('')


# The following function taken was adapted from Code Institue Malaria Detector
# Walkthrough Sample Project
# https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/jupyter_notebooks/02%20-%20DataVisualization.ipynb
def image_montage_function(dir_path, label, nrows, ncols, figsize=(15, 10)):
    """
    Displays a montage of randomly sampled images from a specific label.
    """
    sns.set_style("white")
    label_list = os.listdir(f'{dir_path}')
    # check if the requested montage size is greater than the subset size
    if label in label_list:
        image_list = os.listdir(f'{dir_path}/{label}')
        length_image_list = len(image_list)
        requested_size = nrows*ncols
        if requested_size < length_image_list:
            image_sample = random.sample(image_list, nrows * ncols)
        elif requested_size == 0:
            print('Either ncols or nrows is 0(zero)')
            return
        else:
            print(
                'The sample size you requested is larger than the amount of '
                'images available in the subset')
            print(f'There are {length_image_list} images available')
            print(f'You requested {requested_size} images for your montage')
            print('Decrease the nrows or ncols to create a montage')
            return

        # create a list of cartesian products of nrows and ncols for
        # axes indices
        plot_idx = []
        for row in range(0, nrows):
            for col in range(0, ncols):
                plot_idx.append((row, col))

        # create the figure to display the image montage
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for x in range(0, requested_size):
            img = imread(f'{dir_path}/{label}/{image_sample[x]}')
            axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
            axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
            axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])

        plt.tight_layout()
        st.pyplot(fig)

    else:
        print("The label you selected doesn't exist.")
        print(f"Please select one of the following options: {label_list}")
