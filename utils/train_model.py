import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
train_df = pd.read_csv('data/train.csv')

# Define the labels for abusive categories
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Prepare the features and labels
X = train_df['comment_text']

# Train a separate classifier for each abusive category
models = {}

for label in label_cols:
    y = train_df[label]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline for training the model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('clf', LinearSVC())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Save the model for the current category
    models[label] = pipeline
    joblib.dump(pipeline, f'model/{label}_model.pkl')

print("✅ All models trained and saved.")
