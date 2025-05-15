import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  # Switched to SVM model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE  # Handling data imbalance

# 1. Fixed dataset with automatic length correction
data = {
    'text': [
        "I am so happy today!", "I love it!", "This made my day!", "Absolutely thrilled!",
        "Best experience ever!", "Feeling awesome!", "So grateful!", "I'm excited!",
        "Smiling all day", "Great vibes!",

        "This is terrible news", "What a horrible experience", "Awful!", "Disgusting service",
        "I'm feeling sad and alone", "I’m depressed", "Nothing feels right", "Crying again",
        "Rude and unkind", "So frustrating!", "I hate this", "So disappointed", "I feel sick",
        "It ruined my mood",

        "I'm scared", "I’m worried", "This is terrifying", "Panicking a bit",
        "I feel uneasy", "Nervous energy", "Cautious about it", "A bit jumpy",

        "It's okay, I guess", "So peaceful and calm", "Not bad", "No big deal",
        "This is surprising", "That was shocking", "Unexpected result", "Really odd experience",
        "Meh.", "Just okay", "Fine, I suppose", "Could be better", "Doesn’t matter",
        "Whatever", "Nothing special"
    ],
    'emotion': [
        "positive", "positive", "positive", "positive",
        "positive", "positive", "positive", "positive",
        "positive", "positive",

        "negative", "negative", "negative", "negative",
        "negative", "negative", "negative", "negative",
        "negative", "negative", "negative", "negative",
        "negative",

        "negative", "negative", "negative", "negative",
        "negative", "negative", "negative", "negative",

        "neutral", "neutral", "neutral", "neutral",
        "neutral", "neutral", "neutral", "neutral",
        "neutral", "neutral", "neutral", "neutral",
        "neutral", "neutral"
    ]
}

# Debugging: Check current list lengths
print("Text list length:", len(data['text']))
print("Emotion list length:", len(data['emotion']))

# Ensure lists match in length
min_length = min(len(data['text']), len(data['emotion']))
data['text'] = data['text'][:min_length]
data['emotion'] = data['emotion'][:min_length]

# 2. Load into DataFrame
df = pd.DataFrame(data)

# 3. Preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[{}]'.format(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text.strip()

df['clean_text'] = df['text'].apply(preprocess)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['emotion'], test_size=0.3, stratify=df['emotion'], random_state=42
)

# 5. TF-IDF Vectorization (Updated parameters)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # Capturing word pairs
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Handle Class Imbalance using SMOTE (ONLY if dataset is larger)
if len(set(y_train)) > 1:  # Ensures multiple classes exist before applying SMOTE
    smote = SMOTE(random_state=42)
    X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)

# 7. Train model (Switched to SVM)
model = SVC(kernel='linear', C=1.0)  # SVM with linear kernel
model.fit(X_train_vec, y_train)

# 8. Predict
y_pred = model.predict(X_test_vec)

# 9. Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 10. Confusion Matrix
labels = sorted(df['emotion'].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

# 11. Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
