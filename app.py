import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
train_cleaned = pd.read_csv('Data/train.csv')

# Initial setup
st.set_page_config(page_title="Climate Sentiment Analysis", page_icon=":earth_africa:")

# CSS styling
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .header {
            background-color: #4285f4;
            padding: 1rem;
            color: white;
            text-align: center;
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .footer {
            margin-top: 2rem;
            text-align: center;
            color: #555;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page 1: Home
def home():
    st.markdown("<div class='header'>Climate Sentiment Analysis</div>", unsafe_allow_html=True)
    st.write("Welcome to our Climate Sentiment Analysis Web App!")
    st.write("Explore the sentiment analysis findings and compare different models.")

    # Display basic information about the dataset
    st.subheader("Dataset Overview:")
    st.write(f"Number of Rows: {train_cleaned.shape[0]}")
    st.write(f"Number of Columns: {train_cleaned.shape[1]}")

    # Display a sample of the dataset
    st.subheader("Sample Data:")
    st.write(train_cleaned.head())

# Page 2: Model Comparison
def model_comparison():
    st.markdown("<div class='header'>Model Comparison</div>", unsafe_allow_html=True)

    # Load the data
    train_cleaned = pd.read_csv('Data/train.csv')

    # Split the data into features (X) and target (y)
    X = train_cleaned['message']
    y = train_cleaned['sentiment']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=42)

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=8000, stop_words='english')
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Apply SMOTE to balance classes
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vectorized, y_train)

    # Model training and evaluation
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Ridge Classifier": RidgeClassifier(alpha=1.0),
    }

    st.write("Comparison of different sentiment analysis models:")
    for model_name, model in models.items():
        st.subheader(model_name)
        
        # Train the model
        model.fit(X_train_balanced, y_train_balanced)
        
        # Make predictions
        predictions = model.predict(X_test_vectorized)

        # Evaluate the model
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Accuracy: {accuracy:.4f}")

        # Display classification report
        st.write("Classification Report:")
        st.text(classification_report(y_test, predictions))

# Page 3: Visual Representation of Model Data
def visual_representation():
    st.markdown("<div class='header'>Visual Representation of Model Data</div>", unsafe_allow_html=True)

    # Load the data
    train_cleaned = pd.read_csv('Data/train.csv')

    # Split the data into features (X) and target (y)
    X = train_cleaned['message']
    y = train_cleaned['sentiment']

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=8000, stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)

    # Apply SMOTE to balance classes
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_vectorized, y)

    # Model training
    model = SVC()  # Use your chosen model

    model.fit(X_balanced, y_balanced)
    predictions = model.predict(X_vectorized)

    # Pie chart for sentiment distribution
    st.subheader("Sentiment Distribution:")
    labels = ['Anti', 'Neutral', 'Pro', 'News']
    sizes = pd.Series(predictions).value_counts().sort_index()
    colors = ['red', 'gray', 'blue', 'green']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)
    plt.close(fig)

    # Bar plot for word popularity
    tokens = ' '.join(train_cleaned['message']).split()

    # Create a pandas DataFrame to count word frequencies
    word_counts = pd.Series(tokens).value_counts()

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    word_counts[:20].plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Top 20 Word Frequencies')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    plt.close(fig)

# ...

selected_page = st.sidebar.selectbox("Select Page", ["Home", "Model Comparison", "Visual Representation"])

# Display selected page
if selected_page == "Home":
    home()
elif selected_page == "Model Comparison":
    model_comparison()
elif selected_page == "Visual Representation":
    visual_representation()

# Footer
st.markdown("<div class='footer'>© 2023 Climate Sentiment Analysis</div>", unsafe_allow_html=True)
