# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd
from scipy.sparse import load_npz
from app_resources.text_preprocessing import preprocess_text, tokenize_data, lemmatize_data

# Text preprocessing and vectorization dependencies
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Model and evaluation dependencies
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

# Visualization dependencies
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Vectorizer
news_vectorizer = open("Data/vectorizer.pkl", "rb")
vectorizer = joblib.load(news_vectorizer)

# Load raw data
raw = pd.read_csv("Data/train.csv")

# Load final data after cleaning
train_final = pd.read_csv("Data/train_final.csv")
test_final = pd.read_csv("Data/test_final.csv")

# Load the balanced training data after preprocessing
train_final_balanced = load_npz("Data/train_final_balanced.npz")
y_train_final_balanced = pd.read_csv("Data/y_train_final_balanced.csv")

# Load pre-trained models
ridge_model = joblib.load(open(os.path.join("Trained_models/ridge_model.pkl"), "rb"))
sgdc_model = joblib.load(open(os.path.join("Trained_models/sgdc_model.pkl"), "rb"))
svm_model = joblib.load(open(os.path.join("Trained_models/svm_model.pkl"), "rb"))


# Function to preprocess the text
def preprocess_user_input(tweet_text):
    cleaned_text = preprocess_text(tweet_text)
    tokenized_text = tokenize_data(cleaned_text)
    lemmatized_text = lemmatize_data(tokenized_text)
    return " ".join(lemmatized_text)


# Function to display home page
def display_home():
    st.info("Welcome to the Climate Change Twitter Sentiment Classifier App!")
    st.markdown("This app allows you to analyze sentiments of tweets related to climate change using machine learning models.")
    st.subheader("Basic Statistics and Information")
    st.write(f"Total Tweets: {len(raw)}")


# Function to display tweet sentiment analysis page
def display_tweet_sentiment_analysis():
    st.info("Tweet Sentiment Analysis")
    st.subheader("Enter a Tweet for Sentiment Analysis")

    # Creating a text box for user input
    tweet_text = st.text_area("Enter Tweet", "Type Here")

    if st.button("Classify"):
        # Preprocess user input
        processed_text = preprocess_user_input(tweet_text)

        # Vectorize the processed text
        vect_text = vectorizer.transform([processed_text])

        # Load your models and make predictions
        ridge_prediction = ridge_model.predict(vect_text)
        sgdc_prediction = sgdc_model.predict(vect_text)
        svm_prediction = svm_model.predict(vect_text)

        # Display predictions
        st.success("Ridge Classifier Prediction: {}".format(ridge_prediction[0]))
        st.success("SDG Classifier Prediction: {}".format(sgdc_prediction[0]))
        st.success("Support Vector Machine Prediction: {}".format(svm_prediction[0]))


# Function to display model comparison page
def display_model_comparison():
    st.info("Model Comparison")
    st.subheader("Comparing Trained Models")

    # Display model comparison metrics (precision, recall, f1_score)
    st.write("Add your model comparison visualizations here!")

    # Customize the layout and style of the page
    st.markdown(
        """
        <style>
            .model-comparison {
                padding: 20px;
                background-color: #f0f0f0;
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sample code for adding a bar chart (replace with your actual model comparison visuals)
    comparison_data = {
        "Model": ["Ridge Classifier", "SDG Classifier", "Support Vector Machine"],
        "Precision": [0.75, 0.74, 0.75],
        "Recall": [0.74, 0.72, 0.74],
        "F1 Score": [0.74, 0.73, 0.74],
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.markdown("### Model Comparison Metrics")
    st.bar_chart(comparison_df.set_index("Model"))


# Function to display word cloud and insights page
def display_word_cloud_and_insights():
    st.info("Word Cloud and Insights")
    st.subheader("Generating Word Cloud and Providing Insights")

    # Display word cloud and insights
    st.write("Add your word cloud visualizations and insights here!")

    # Customize the layout and style of the page
    st.markdown(
        """
        <style>
            .word-cloud {
                padding: 20px;
                background-color: #e6f7ff;
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sample code for adding a word cloud (replace with your actual word cloud visuals)
    st.markdown("### Word Cloud of Tweets")
    wordcloud_data = "Your word cloud data here"  # Replace with your actual data
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_data)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Tweets')
    st.pyplot(plt)


# Function to display user feedback and conclusion page
def display_user_feedback_and_conclusion():
    st.info("User Feedback and Conclusion")
    st.subheader("Provide Feedback and Summarize Key Findings")

    # Display user feedback form
    st.write("Add your user feedback form here!")

    # Customize the layout and style of the feedback form
    st.markdown(
        """
        <style>
            .feedback-form {
                padding: 20px;
                background-color: #ffd699;
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sample code for a basic feedback form (replace with your actual feedback form)
    st.markdown("### User Feedback Form")
    feedback_rating = st.slider("Rate your experience with the app (1-5)", 1, 5)
    feedback_comments = st.text_area("Any additional comments or suggestions?", "")

    if st.button("Submit Feedback"):
        # Process and store feedback (replace with your actual feedback handling logic)
        st.success("Feedback submitted successfully!")

    # Display key findings and conclusion
    st.write("Add your summary of key findings and conclusion here!")

    # Building out the footer
    st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background-color: midnightblue;
                padding: 10px 0;
                color: white;
                text-align: center;
            }
        </style>
        <div class="footer">
            <p>Climate Change Sentiment Classifier App</p>
            <p>© 2023 Your Company. All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# The main function where we will build the actual app
def main():
    """Climate Change Twitter Sentiment Classifier App"""

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Climate Change Sentiment Classifier")
    st.subheader("Analyzing Twitter Sentiments")

    # Setting the color scheme to midnight blue
    st.set_page_config(
        page_title="Climate Change Sentiment Classifier",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Adding a background image
    st.markdown(
        """
        <style>
            .reportview-container {
                background: url("app_resources/earth.gif") center center no-repeat;
                background-size: cover;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Home", "Tweet Sentiment Analysis", "Model Comparison", "Word Cloud and Insights", "User Feedback and Conclusion"]
    selection = st.sidebar.selectbox("Navigate to", options)

    # Building out the different pages based on selection
    if selection == "Home":
        display_home()
    elif selection == "Tweet Sentiment Analysis":
        display_tweet_sentiment_analysis()
    elif selection == "Model Comparison":
        display_model_comparison()
    elif selection == "Word Cloud and Insights":
        display_word_cloud_and_insights()
    elif selection == "User Feedback and Conclusion":
        display_user_feedback_and_conclusion()


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
