import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib # For saving the final model

# --- 1. NLTK Setup (Run once) ---
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))


# --- 2. Core Functions ---

def load_data(file_path):
    """Loads raw text from the specified file path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

def preprocess_text(text):
    """Performs cleaning, tokenization, stopword removal, and lemmatization."""
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Removing punctuation and non-alphanumeric characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenization
    tokens = nltk.word_tokenize(text)
    
    # 4. Stopword removal & Lemmatization
    processed_tokens = [
        lemmatizer.lemmatize(w) for w in tokens 
        if w not in STOP_WORDS and len(w) > 1
    ]
    
    return " ".join(processed_tokens)

def train_and_evaluate_model(X, y):
    """Splits data, trains classifier, and prints evaluation metrics."""
    # Split data (Simulation for demonstration)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 1. Feature Extraction: TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 2. Model Training: Multinomial Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # 3. Prediction and Evaluation
    y_pred = model.predict(X_test_vec)
    
    print("\n--- Model Performance Report ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Return the trained components for potential saving
    return model, vectorizer

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    
    # --- Data Simulation ---
    # NOTE: Since actual labels are not provided, we simulate a small
    # labeled dataset using the processed text for demonstration.
    try:
        raw_data = load_data('data/sample_text.txt')
    except FileNotFoundError:
        print("Error: 'data/sample_text.txt' not found. Please place the file in the 'data' directory.")
        exit()

    processed_single_text = preprocess_text(raw_data)
    
    # Simulate a dataset of 10 documents and 2 classes (0 and 1)
    # In a real scenario, you would load a labeled dataset (e.g., a CSV file).
    SIMULATED_DOCUMENTS = [processed_single_text] * 5 + \
                          ["another processed document example"] * 5
    SIMULATED_LABELS = [0] * 5 + [1] * 5
    
    # Display processing result
    print("--- Sample of Processed Data ---")
    print(processed_single_text[:200] + "...")
    print(f"\nTotal simulated documents for training: {len(SIMULATED_DOCUMENTS)}")

    # Train the model
    final_model, final_vectorizer = train_and_evaluate_model(
        SIMULATED_DOCUMENTS, 
        SIMULATED_LABELS
    )

    # --- 4. Model Persistence (Saving the final model) ---
    joblib.dump(final_model, 'models/final_classifier.pkl')
    joblib.dump(final_vectorizer, 'models/tfidf_vectorizer.pkl')
    print("\nModel and Vectorizer successfully saved to the 'models/' directory.")
