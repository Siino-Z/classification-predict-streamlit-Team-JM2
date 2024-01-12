import os
import joblib
from sklearn import feature_selection
from sklearn.datasets import make_classification
from sklearn.feature_selection import f_classif
from sklearn.linear_model import Ridge, RidgeClassifier, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np

# Suppress Streamlit warnings globally
st.set_page_config()

# CSS styling for a blue background and black gradient in the sidebar
st.markdown(
    """
    <style>
      .stApp{
            background: linear-gradient(to right, #2980b9, #6dd5fa);
            color: white;
        }
      .stApp header {
            background: linear-gradient(to right, #2980b9, #6dd5fa);
            color: white;
        }
        body {
            background: linear-gradient(to bottom, #87CEFA, #4682B4);
            font-family: 'Arial', sans-serif;
        }
        .sidebar-content {
            background: linear-gradient(to right, #2980b9, #6dd5fa);
            color: Blue;
        }
        .header {
            background-color: #2980b9;
            padding: 1rem;
            color: black;
            text-align: center;
            font-size: 3.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .subheader {
            color: #333;
            font-size: 1.5rem;
            font-weight: bold;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .footer {
            margin-top: 2rem;
            text-align: center;
            color: #555;
            font-size: 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to load the sentiment analysis model
@st.cache_data
def load_sentiment_model():
    return joblib.load(open("Trained_models/ridge_model.pkl", 'rb'))

# Function to load the Twitter sentiment analysis model
@st.cache_data
def load_twitter_sentiment_model():
    return joblib.load(open("Trained_models/logistic_model.pkl", 'rb'))

# Function to load SVM model
@st.cache_data
def load_svm_model():
    return joblib.load(open("Trained_models/svm_model.pkl", 'rb'))

# Function to load the feature selector model
@st.cache_data
def load_feature_selector():
    return joblib.load(open("Data/selector_model.pkl", 'rb'))


# Load the sentiment analysis model only once during app initialization
sentiment_model = load_sentiment_model()

# Load the Twitter sentiment analysis model only once during app initialization
twitter_sentiment_model = load_twitter_sentiment_model()

# Load SVM model only once during app initialization
svm_model = load_svm_model()

# Load the feature selector model only once during app initialization
feature_selector = load_feature_selector()

# List of employees and their job descriptions
employees = [
    {"name": "Asmaa Hassan", "job": "Data Scientist"},
    {"name": "Kaya Dumasi", "job": "Machine Learning Engineer"},
    {"name": "Sinothabo Zwane", "job": "Software Developer"},
    {"name": "Lebohang Lenaka", "job": "Natural Language Processing Expert"},
    {"name": "Nonkanyiso Mabaso", "job": "Data Analyst"},
]

# Sidebar navigation
page = st.sidebar.selectbox("Select a page", ["Home", "Twitter Sentiment", "Model Comparison", "Feedback"])

# Home page
if page == "Home":
    st.title('Sentiment Classification App')
    st.write("Welcome to the home page!")

    # Introduction to the team
    st.header('About Us')
    st.write("""
    Our team is dedicated to delivering high-quality sentiment analysis solutions. 
    We combine expertise in machine learning, natural language processing, and software development 
    to provide innovative and effective solutions to our clients.
    """)
    
    # List of employees
    st.subheader('Meet the team:')
    for employee in employees:
        st.write(f"- **{employee['name']}**: {employee['job']}")

    # Introduction to the company
    st.header('About Our Company')
    st.write("""
    Wakanda Tech is a leading provider of sentiment analysis services. 
    We leverage state-of-the-art machine learning models to analyze and understand sentiments 
    expressed in text data. Our goal is to empower businesses with valuable insights 
    that drive decision-making and enhance customer experiences.
    """)


# Twitter Sentiment page
elif page == "Twitter Sentiment":
    st.title('Climate Change Sentiment Classifier')
    st.subheader('Analyze Twitter Sentiments')

    # Load the trained models
    ridge_model = joblib.load("Trained_models/ridge_model.pkl")
    logistic_model = joblib.load("Trained_models/logistic_model.pkl")
    svm_model = joblib.load("Trained_models/svm_model.pkl")

    # Model selection
    model_options = ["Ridge", "Logistic", "SVM"]
    selected_model = st.selectbox("Choose Model", model_options)

    # Load your raw data
    raw = pd.read_csv("Data/train.csv")

    # Creating a text box for user input
    tweet_text = st.text_area("Enter Tweet", "Type Here")

    # Load the vectorizer
    news_vectorizer = open("Data/vectorizer.pkl", "rb")
    tweet_cv = joblib.load(news_vectorizer)


    if st.button("Classify"):
        # Ensure the input data is not empty
        if not tweet_text:
            st.warning('Please enter a tweet for sentiment prediction.')
        else:
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text])

            # Apply feature selection
            selected_features = feature_selector.transform(vect_text)

            # Load the selected model based on user choice
            if selected_model == "Ridge":
                # Make predictions using the trained model
                prediction = ridge_model.predict(selected_features)

            elif selected_model == "Logistic":
                # Make predictions using the trained model
                prediction = logistic_model.predict(selected_features)

            elif selected_model == "SVM":
                # Make predictions using the trained model
                prediction = svm_model.predict(selected_features)

            # Map numeric predictions to labels
            label_mapping = {1: 'Pro Climate Change', -1: 'Anti Climate Change', 0: 'Neutral', 2: 'News'}
            predicted_label = label_mapping.get(prediction[0], 'Unknown')

            # Display the prediction
            st.success("Text Categorized as: {}".format(predicted_label))


# Model Comparison page
elif page == "Model Comparison":
    st.title('Model Comparison')

    model_names = ['Logistic Regression', 'SVM', 'Ridge Model']
    accuracy = [0.7320, 0.7408, 0.7396]
    precision_positive = [0.82, 0.80, 0.81]
    precision_negative = [0.68, 0.73, 0.74]
    recall_positive = [0.79, 0.83, 0.81]
    recall_negative = [0.53, 0.41, 0.50]
    f1_score_positive = [0.80, 0.82, 0.81]
    f1_score_negative = [0.59, 0.54, 0.60]

    data = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracy,
        'Precision (Positive)': precision_positive,
        'Precision (Negative)': precision_negative,
        'Recall (Positive)': recall_positive,
        'Recall (Negative)': recall_negative,
        'F1 Score (Positive)': f1_score_positive,
        'F1 Score (Negative)': f1_score_negative
    })

    # Sidebar for user interaction
    st.sidebar.title('Model Comparison Settings')

    # Select metric
    selected_metric = st.sidebar.selectbox('Select Metric', ['Accuracy'])

    # Create an interactive plot based on the selected metric
    fig = px.bar(data, x='Model', y=selected_metric, text=selected_metric,
                 title=f'{selected_metric} Comparison',
                 labels={'Model': 'Models', selected_metric: selected_metric})
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')

    # Display the plot
    st.plotly_chart(fig)

    # Show the raw data table (optional)
    st.write("Raw Data:")
    st.write(data)

    # Display model information based on the selected model
    selected_model = st.sidebar.selectbox("Select Model for Information", model_names)

    # Function to display model information for each model
    def display_ridge_model_information():
        st.header("Ridge Model Information")
        st.write("73.96% Accuracy")
        st.write("Strengths: Provides a balance between precision and recall for class 1.")
        st.write("Weaknesses: Struggles with negative and neutral sentiments.")

    def display_svm_model_information():
        st.header("SVM Model Information")
        st.write("Accuracy: 74.08%")
        st.write("Strengths: Best overall performance. Balanced precision, recall, and F1-score for all sentiments.")
        st.write("Weaknesses: Slightly lower performance for negative sentiment.")

    def display_logistic_regression_information():
        st.header("Logistic Regression Information")
        st.write("Accuracy: 73.20%")
        st.write("Strengths: Achieves a relatively high precision for class 1, indicating a good ability to correctly classify positive instances.")
        st.write("Weaknesses: Shows lower recall for class -1, suggesting a challenge in identifying negative instances..")

    # Call the appropriate function based on the selected model
    if selected_model == 'Ridge Model':
        display_ridge_model_information()
    elif selected_model == 'SVM':
        display_svm_model_information()
    elif selected_model == 'Logistic Regression':
        display_logistic_regression_information()

# Feedback page
elif page == "Feedback":
    # Feedback section
    st.header('Feedback')

    # FAQ section
    st.subheader('FAQ')

    # Define your FAQ data
    faq_data = {
        "What is this app about?": "This app is for Climate Sentiment Analysis.",
        "How to use the app?": "Navigate through the pages to explore the dataset, compare models, and view visualizations.",
        "How accurate are the sentiment analysis models?": "The accuracy of the models is shown on the 'Model Comparison' page.",
        "Can I trust the sentiment predictions?": "While the models are trained on the available data, predictions may not be perfect. Use your discretion.",
        "Are there any specific keywords that affect sentiment?": "The models consider a variety of words, and the 'Visual Representation' page shows word frequencies.",
        "Is the dataset balanced for different sentiments?": "The dataset is balanced using SMOTE to address class imbalances.",
        "How often is the dataset updated?": "The dataset used in this app is static, and updates depend on your source of data.",
    }

    # Display buttons for each FAQ
    faq_option = st.radio("Select FAQ:", list(faq_data.keys()))

    # Check if the button is clicked
    if st.button("Show Answer", key='show_answer_button'):
        # Get and display the selected FAQ response
        response = faq_data[faq_option]
        st.text("Answer: " + response)

    # Create a form for feedback submission
    with st.form(key='feedback_form'):
        # Add a text area for feedback input
        user_feedback = st.text_area('Share your feedback here:')

        # Add a submit button with a custom style
        submit_button = st.form_submit_button('Submit Feedback')

    # Apply custom styles using CSS
    st.markdown(
        """
        <style>
            div[data-baseweb="button"] button {
                background-color: blue;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Process the feedback when the form is submitted
    if submit_button:
        # Process the feedback (you can add your logic here)
        st.success('Feedback submitted successfully!')

# Footer with custom CSS
    st.markdown(
    """
    <style>
    body {
        background-color: black; /* Set the background color of the entire app to black */
    }

    .css-1ax1fjk {
        background-color: black; /* Set the background color of the sidebar to black */
    }

    .footer {
        background: linear-gradient(to right, #f0f0f0, #c0c0c0); /* Light light grey to dark light grey gradient */
        color: black;
        text-align: center;
        padding: 7px;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding-left: 170px; /* Add left padding to move text towards the right */
    }
    </style>
    <div class="footer">
        <p>Built with ❤️ WAKANDA TECH TEAM © 2023 Streamlit Sentiment Analysis</p>
        <p>Check out the code on <a href="https://github.com/Siino-Z/classification-predict-streamlit-Team-JM2.git" style="color: black;">GitHub</a></p>
    </div>
    """,
    unsafe_allow_html=True
    )
