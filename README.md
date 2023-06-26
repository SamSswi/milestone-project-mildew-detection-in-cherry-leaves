# Cherry Leaf Mildew Detector

[Cherry Leaf Mildew Detector Live Application](https://cherry-leaves-mildew-detector-1936274102f0.herokuapp.com/)

This machine learning project was undertaken for my fifth project with the Code Institute, which was part of the Predictive Analytics module. The task was set to build an ML pipeline which would predict if a cherry leaf is healthy or contains powdery mildew. The machine learning task was utilising a deep learning neural neural network for prediction. 

## CRISP-DM workflow (Cross Industry Standard Process for Data Mining)
The project uses CRISP-DM workflow
* Business Understanding
    * How will the client benefit from using the model?
    * Is there sufficient data available to satisfy the Business Requirements?
    * What does success look like? How will it be measured?
    * What tooling or technologies will be required (Dashboard or API)?
* Data Understanding
    * Access and load the data
* Data Preparation
    * Remove non-image files
    * Split the data into Train, Validation and Test Sets
* Modelling
    * What modelling task is required
    * Perform Image Augmentation on the Train set Images
    * Rescale the Validation and Test set Images 
    * Fit data to trainset using default parameters
    * Iterate the following process until the model meets the accuracy level asked by the client:
        * Evaluate performance of hyperparameters
        * Adjust the hyperparameter choices/values
* Evaluation
    * Does the model meet the Business Requirements?
    * Explain results via metrics and visualizations
    * Deploy the model


## Dataset Content
* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). I then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset contains 4208 images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.


## Business Requirements
The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees, located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

* 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
* 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.


## Hypothesis and how to validate?
* Hypothesis 1: There are visible patterns in the cherry leaf images that can be used to distinguish between healthy and powdery mildew affected leaves.
    * Validation: The Average Image and Variability Image revealed subtle hints the powdery mildew-affected leaves have pale speckles on them The Difference between Averages study did not reveal any clear pattern to differentiate between the two labels. An Image Montage shows that typically the cherry leaves that have powdery mildew, have superficial white growths and in some cases, their shape is deformed.

* Hypothesis 2: Neural Networks can effectively map the relationships between the features extracted from cherry leaf images and the corresponding labels (healthy or powdery mildew).
    * Validation: A Convolutional Neural Network was used to map the relationships between the features extracted from cherry leaf images and the corresponding labels (healthy or powdery mildew). The usage of a Convolutional Neural Network was justified by its main advantages as efficient image processing, high accuracy rates, robustness to noise, and automated feature extraction. The trained Neural Network has an accuracy of 99.88% on the unseen data from the test set.
* Hypothesis 3: Reducing the image shape to smaller dimensions (e.g., 100x100 or 50x50) will not significantly impact the model's performance in terms of accuracy.
    * The model was trained on images resized to 100x100px, and the accuracy of its performance on new unknown data was 99.88%.
* Hypothesis 4: The model performance meets the project requirement of 97% accuracy, as agreed upon with the client.
    * After the evaluation of the model on a separate test dataset, the accuracy of the model was 99.88%. These metrics satisfy the requirement of 97% accuracy.


## The rationale to map the business requirements to the Data Visualisations and ML tasks
* List your business requirements and a rationale to map them to the Data Visualisations and ML tasks.

* 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
    * As a Data Practitioner I can clean and prepare the collected data then split it into training, validation, and test sets so that further analysis and model training can be done on it.
    * As a client I want to display the 'mean' and 'standard deviation' images for healthy and powdery mildew-containing cherry leaves so that I can intuitively differentiate cherry leaves
    * As a Client I want to display the difference between an average cherry leaf that is healthy and a cherry leaf that contains powdery mildew so that I can visually differentiate cherry leaves
    * As a Client I want to display an image montage for healthy and powdery mildew-containing cherry leaves so that I can visually differentiate cherry leaves
* 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.
    * As a client I want to build an ML model and generate reports
    * As a client I want to predict if a given cherry leaf is healthy or contains powdery mildew.


## ML Business Case
* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.

1. What are the business requirements?
    * The client is interested to visually differentiate between healthy cherry leaves and cherry leaves that have powdery mildew.
    *  The client is interested in predicting the health status of cherry leaves based on image data.

2. Is there any business requirement that can be answered with conventional data analysis?
    * A part of the business objective such as visually differentiating between healthy leaves and mildew containing ones can be potentially answered with conventional data analysis.
    * Predicting the health status of cherry leaves based on image data, however would require more advanced techniques such as machine learning or deep learning algorithms.
3. Does the client need a dashboard or an API endpoint?
    * The client needs a dashboard.
4. What does the client consider as a successful project outcome?
    * A study showing how to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
    * Also, the capability to predict if a cherry leaf is healthy or contains powdery mildew.
5. Can you break down the project into Epics and User Stories?
    * Information gathering and data collection.
    * Data visualization, cleaning, and preparation.
    * Model training, optimization and validation.
    * Dashboard planning, designing, and development.
    * Dashboard deployment and release.
6. Ethical or Privacy concerns?
    * The client provided the data under an NDA (non-disclosure agreement), therefore the data should only be shared with professionals that are officially involved in the project.
7. Does the data suggest a particular model?
    * The data suggests a binary classification model since it aims to differentiate between healthy and powdery mildew-affected leaves.
8. What are the model's inputs and intended outputs?
    * The inputs for the model would be cherry leaf images, and the expected output would be a prediction of the health status of each leaf (healthy or powdery mildew-affected).
9. What are the criteria for the performance goal of the predictions?
    * The client requested a degree of 97% accuracy.
10. How will the client benefit?
    * The benefits to the customer from this project include improved quality control, the ability to identify and remove diseased cherry leaves, and the delivery of higher-quality products to the market.


## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items, that your dashboard library supports.
* Finally, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project, you were confident you would use a given plot to display an insight, but later, you chose another plot type).


* Page 1: Project Summary
    * General Inforamtion - block of information
    * Project Dataset description - block of information
    * Link to the README file - block of information
    * Business Requirements list - block of information

* Page 2: Leaves Visualizer
    * The business requirement this page answers - block of information
    * Checkbox1 - Difference between average and variability image
        * Short explanation of the data visualization result - information block
        * Relevant image - image
    * Checkbox2 - Differences between the average image of healthy leaves and leaves that have powdery mildew
        * Short explanation of the data visualization result - information block
        * Relevant image - image
    * Checkbox3 - Image montage
        * Instruction for the user to click the "Create Montage" button
        * Selectbox - where the user can choose one label to create a montage of
        * Create Montage button
            * The image montage of the selected label
* Page 3: Cherry Leaves Powdery Mildew Detector
    * The business requirement this answers - block of information
    * Link to the download images of cherry leaves - block of information
    * horizontal line
    * File uploader where the user can upload cherry leaf images in order to get a diagnosis on them
        * Image Name - information block
        * the leaf sample itself - image
        * diagnosis on the leaf sample - information block
        * the diagnosis probability - barplot
        * Table with the analysis report on all uploaded images - dataframe
        * Download report link
* Page 4: Project Hypotheses
    * A hypothesis and its validation - information block
* Page 5: ML Performance Metrics
    * Train, Validation and Test Set: Label Frequencies plot - image
    * Model History: Accuracy and Loss Line Graph  - image
    * Model Accuracy and Loss table - pkl

## Testing
I installed and ran Flake8 from the terminal for testing the python files in the "app_pages" folder. The issues primarily consisted of missing whitespace, trainling white space and lines that exceeded the recommended length. I successfully solved all the issues identified by Flake8.

## Unfixed Bugs
* You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

* **False Positive Results:**
    * While the model performs well on the images provided by the client, it's important to note that when applied to images outside the dataset, particularly those with artificial light reflecting off the leaf surface, the model occasionally yields false positives due to the challenges posed by such lightning conditions. None of the images supplied by the client exhibit this specific issue, though. So it is not a critical concern for this application. 

* **Dependency on User-Provided Images**
    * While the model predicted well on lower resolution images, as well as images with a certain amount of noise or slight blur, it is important to know the model accuracy still depends on the quality and suitability of the images provided by the user. If the uploaded images are of poor quality, contain excessive noise, or do not clearly depict the cherry leaf, the diagnostic results might be affected.

* **Model Size and Deployment**
    * Convolutional Neural Networks can be computationally intensive, particularly when processing large images. Heroku has limited CPU resources on lower tier plans, which can impact the model prediction speed.

* **Convolutional Neural Network**
    * Limited Robustness to Adversarial Attacks - The model can be mislead into making incorrect predictions if subtle modifications are deliberately made to the input images, like for example imperceptible noise or pattern. By their nature the CNNs are susceptible to such attacks.

## Deployment
### Heroku

The App live link is: [Cherry Leaver Mildew Detector](https://cherry-leaves-mildew-detector-1936274102f0.herokuapp.com/) 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.

* The project was deployed to Heroku using the following steps.
1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select the  repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access the App.


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries used in the project and provide an example(s) of how you used these libraries.

* Python Standard Library - 
* streamlit
    * used for building the dashboard
    * example:
        st.info(' The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.')
* pandas
    * used for data manipulation, data analysis, and data preprocessing.
    * example: 
        * creating a dataframe from a list of pd.Series objects
        df_freq = pd.DataFrame(freq_list)
* numpy
    * used its wide range of methods for performing array operations
    * example
        * checking whether the labels exist in the set of unique labels
        if (label_1 not in np.unique(y)) or (label_2 not in np.unique(y)):
            print(f"Either label {label_1} or label {label_2}, are not in {np.unique(y)} ")
            return
* seaborn
    * used to set the style of the plot and create a barplot with the barplot() function.
    * example:
        * sns.set_style("whitegrid")
        * sns.barplot(data=df_freq, x='Set', y='Frequency')
* matplotlib
    * used for creating data visualizations in various formats
    * example:
        * fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)<br>
        axes[0].set_title('Average Image')<br>
        axes[0].imshow(avg_image)<br>
        axes[1].set_title('Image Variability')<br>
        axes[1].imshow(image_var)<br>
        plt.show()<br>

* joblib
    * used for storing and loading data
    * example:
        * joblib.dump(value=image_shape, filename=f"{file_path}/image_shape.pkl")
* keras
    * used for building and training the neural network
    * example: 
        * model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=input_shape, activation='relu',))
* PIL (Python Imaging Library)
    * used to perform various image-related operations in the code, such as opening, and resizing images
    * example:
        * img_pil = (Image.open(uploaded_file))
* plotly
    * used for data visualizations
    * example:
        *fig = px.bar(prob_per_class, x='Diagnostic', y=prob_per_class['Probability'], range_y=[0, 1], width=600, height=300, template='seaborn')
* tensorflow
    * used in the background for computations and training
    * example"
        * from tensorflow.keras.callbacks import EarlyStopping<br>
        early_stop = EarlyStopping(monitor='val_loss', patience=7)


## Credits 

* Dashboard
    * [Code Institute Malaria Detector sample project](https://malaria-predictor.onrender.com/) - dashboard design
* Leaves Visualizer Page:
    * [Streamlit Documentation](https://docs.streamlit.io/library/api-reference/widgets/st.checkbox) - st.checkbox function
    * [Code Institue Malaria Detector Walkthrough Sample Project - Data Visualisation Notebook](https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/jupyter_notebooks/02%20-%20DataVisualization.ipynb) - image montage function
* ML Performance Page
    * [Code Institue Malaria Detector Walkthrough Sample Project - ML Performance Metrics dashboard page](https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/app_pages/page_ml_performance.py) - st.columns function usage
* Multipage
    * [Code Institute's Malaria Detection Sample Project - multipage.py](https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/app_pages/multipage.py)
* Powdery Mildew Detection Page
    * [Code Institute Malaria Detector Sample Project - predictive_analysis.py](https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/src/machine_learning/predictive_analysis.py) - plot_predictions_probabilities function
    * [Code Institute Malaria Detector Sample Project - predictive_analysis.py](https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/src/machine_learning/predictive_analysis.py) - resize_input_image function
    * [Code Institute Malaria Detector Sample Project - predictive_analysis.py](https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/src/machine_learning/predictive_analysis.py) - load_model_and_predict function
    * [Code Institute Malaria Detector Sample Project - data_management.py](https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/src/machine_learning/data_management.py) - download_dataframe_as_csv function
    * [Streamlit Documentation](https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader) - st.file_uploader function usage
    * [Code Institute Malaria Detector Sample Project - page_malaria_detector.py](https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/app_pages/page_malaria_detector.py) - powdery_mildew_detection_body > if statement
* Data Collection Notebook
    * [Code Institue Malaria Detector Walkthrough Sample Project - Data Collection Notebook](https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/jupyter_notebooks/01%20-%20DataCollection.ipynb) - import, clean and split the dataset
* Data Visualization Notebook
    * [Code Institue Malaria Detector Walkthrough Sample Project - Data Visualization Notebook](https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/jupyter_notebooks/01%20-%20DataCollection.ipynb) - set input and output directories, calculate average image size, save image shape, difference between the average images calculation, image montage creation.
    * [Code Institute Data Analytics Packages Lesson, TensorFlow Unit 10: Image Classification](https://learn.codeinstitute.net/courses/course-v1:code_institute+CI_DA_ML+2021_Q4/courseware/1f851533cd6a4dcd8a280fd9f37ef4e2/b6cf6ce506324501bcf6aa0f31e0c20c/) - label frequency DataFrame and Barplot, Average Image and Image Variability calculation.
* Modelling and Evaluation Notebook
    * [Code Institue Malaria Detector Walkthrough Sample Project - Data Visualization Notebook](https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/jupyter_notebooks/01%20-%20DataCollection.ipynb) - set input and output directories.
    * [Code Institue Malaria Detector Walkthrough Sample Project - Modelling and Evaluating Notebook](https://github.com/SamSswi/WalkthroughProject01DataAnalytics/blob/main/jupyter_notebooks/03%20-%20Modelling%20and%20Evaluating.ipynb) - load files from the output folder, image augmentation, augmented images plotting, early stopping, model evaluation, prediction on unknown data.
    * [Code Institute Data Analytics Packages Lesson, TensorFlow Unit 10: Image Classification](https://learn.codeinstitute.net/courses/course-v1:code_institute+CI_DA_ML+2021_Q4/courseware/1f851533cd6a4dcd8a280fd9f37ef4e2/b6cf6ce506324501bcf6aa0f31e0c20c/) - model creation, network structure visualizing, model training, model saving, model learning curve. 
    * [Numpy Documentation](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html) - np.argmax function usage.
    * [medium.com](https://medium.com/onfido-tech/adversarial-attacks-and-defences-for-convolutional-neural-networks-66915ece52e7) - Unfixed Bugs - CNN shortcomings.