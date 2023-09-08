import streamlit as st
import pandas as pd
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import os
from Prediction.batch import BatchPrediction
from src.logger import logging
from src.components.data_transformation import DataTransformationConfig
from src.config.configuration import *
from src.pipeline.training_pipeline import Train
from werkzeug.utils import secure_filename
import base64

feature_engineering_file_path = FEATURE_ENG_OBJ_PATH
transformer_file_path = PREPROCESSING_OBJ_PATH
model_file_path = MODEL_FILE_PATH

UPLOAD_FOLDER = 'batch_prediction/Uploaded_CSV_FILE'
#predicted_file_path = 'batch_prediction/Predicted_CSV_FILE/predicted_results.csv'


# Set the title of the Streamlit app
st.title("Adult Census Income Prediction")

ALLOWED_EXTENSIONS = {'csv'}

# Streamlit sidebar
st.sidebar.header("Navigation")
selected_page = st.sidebar.radio("Select a Page", ["Home", "Predict", "Batch Prediction", "Train"])

if selected_page == "Home":
    st.header("Home Page")
    st.write("Welcome to the Adult Census Income Prediction App!")
    st.write("Use the sidebar to navigate to different pages.")

elif selected_page == "Predict":
    st.header("Single Data Point Prediction")
    
    # Create input form
    st.subheader("Enter Data for Prediction")
    Age = st.number_input("Age", min_value=0.0, format="%.2f")
    Education_num = st.selectbox("Education_num",['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'])
    Capital_gain = st.number_input("Capital_gain", min_value=0.0, format="%.2f")
    Hours_per_week = st.number_input("Hours_per_week", min_value=0.0, format="%.2f")
    Marital_status = st.selectbox("Marital_status",['Married-civ-spouse','Never-married','Divorced','Separated','Widowed','Married-spouse-absent','Married-AF-spouse'])
    Relationship = st.selectbox("Relationship", ['Husband','Not-in-family','Own-child','Unmarried','Wife','Other-relative'])    
    Race = st.selectbox("Race", ['White','Black','Asian-Pac-Islander','Amer-Indian-Eskimo','Other'])
    Sex = st.selectbox("Sex", ['Male','Female'])
    Native_country = st.selectbox("Native_country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "Puerto-Rico", 
                                "El-Salvador", "India", "Cuba", "England", "Jamaica", "South", "China", "Italy", 
                                "Dominican-Republic", "Vietnam", "Guatemala", "Japan", "Poland", "Columbia", "Taiwan", 
                                "Haiti", "Iran", "Portugal", "Nicaragua", "Peru", "Greece", "France", "Ecuador", 
                                "Ireland", "Hong", "Cambodia", "Trinadad&Tobago", "Laos", "Thailand", "Yugoslavia", 
                                "Outlying-US(Guam-USVI-etc)", "Hungary", "Honduras", "Scotland", "Holand-Netherlands"])
    New_occupation = st.selectbox("New_occupation", ['Professional_Managerial','Skilled_Technical','Sales_Administrative','Service_Care','Unclassified Occupations'])
    
    # Make a prediction
    if st.button("Predict"):
        data = CustomData(
            Age=Age,
            Education_num=Education_num,
            Capital_gain=Capital_gain,
            Hours_per_week=Hours_per_week,
            Marital_status=Marital_status,
            Relationship=Relationship,
            Race=Race,
            Sex=Sex,
            Native_country=Native_country,
            New_occupation=New_occupation
        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)
        result = int(pred[0])

        if result == 0:
            st.success("Predicted Income: <=50k")
        else:
            st.success("Predicted Income: >50k")


elif selected_page == "Batch Prediction":
    st.header("Batch Prediction")

    # Create a file uploader
    st.subheader("Upload a CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(uploaded_file.name))
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        logging.info("CSV received and Uploaded")

        # Perform batch prediction using the uploaded file
        batch = BatchPrediction(file_path, model_file_path, transformer_file_path, feature_engineering_file_path)
        batch.start_batch_prediction()

        # Define the function to create a download link
        def get_binary_file_downloader_html(file_path):
            with open(file_path, 'rb') as file:
                contents = file.read()
            encoded_file = base64.b64encode(contents).decode()
            return f'<a href="data:file/csv;base64,{encoded_file}" download="predicted_results.csv">Download Predicted File</a>'


        # Define a variable to store the path to the predicted file after batch prediction
        predicted_file_path = 'batch_prediction/Prediction_CSV/prediction.csv'


        # After batch prediction and displaying the success message, add a download button for the predicted file
        output = "Batch Prediction Done"
        st.success(output)

        # Add a download button for the predicted file
        if st.button("Download Predicted File"):
            st.markdown(get_binary_file_downloader_html(predicted_file_path), unsafe_allow_html=True)

# ... The rest of your code ...
elif selected_page == "Train":
    st.header("Model Training")

    if st.button("Train Model"):
        try:
            pipeline = Train()
            pipeline.main()
            st.success("Training complete")
        except Exception as e:
            logging.error(f"{e}")
            st.error(f"Error during training: {str(e)}")

# Run the Streamlit app
if __name__ == '__main__':
    st.sidebar.title("My Streamlit App")
    st.title("Main Content")