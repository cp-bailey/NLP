import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD

# =================================================================
# Class_Ex1:
# Use the following dataframe as the sample data.
# Find the conditional probability of Char given the Occurrence.
# P(C|O) = P(C intersection O) / P(O)
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')
df = pd.DataFrame(
    {'Char': ['f', 'b', 'f', 'b', 'f', 'b', 'f', 'f'], 'Occurance': ['o1', 'o1', 'o2', 'o3', 'o2', 'o2', 'o1', 'o3'],
     'C': np.random.randn(8), 'D': np.random.randn(8)})

# Calculate counts of unique combinations of 'Char' and 'Occurrence'
counts = df.groupby(['Char', 'Occurance']).size()

# Calculate total occurrences of each 'Occurrence'
occurrence_counts = df['Occurance'].value_counts()

# Calculate conditional probabilities
conditional_probs = counts / occurrence_counts

print("Conditional Probability of Char given Occurrence:")
print(conditional_probs)

print(20 * '-' + 'End Q1' + 20 * '-')

# =================================================================
# Class_Ex2:
# Use the following dataframe as the sample data.
# Find the conditional probability occurrence of the word given a sentiment.
# P(W|S) = P(W intersection S) / P(S)
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

df1 = pd.DataFrame({'Word': ['Good', 'Bad', 'Awesome', 'Beautiful', 'Terrible', 'Horrible'],
                    'Occurrence': ['One', 'Two', 'One', 'Three', 'One', 'Two'],
                    'sentiment': ['P', 'N', 'P', 'P', 'N', 'N'], })

# Calculate counts of unique combinations of word and sentiment
counts = df1.groupby(['Word', 'sentiment']).size()

# Calculate total occurrences of each sentiment
sentiment_counts = df1['sentiment'].value_counts()

# Calculate conditional probabilities
conditional_probs = counts / sentiment_counts

print("Conditional Probability of word given sentiment:")
print(conditional_probs)

print(20 * '-' + 'End Q2' + 20 * '-')
# =================================================================
# Class_Ex3:
# Read the data.csv file.
# Answer the following question
# 1- In this dataset we have a lot of responses in text and each response has a label.
# 2- Our goal is to correctly model the texts into its label.
# Hint: you need to read the text responses and perform preprocessing on it.
# such as normalization, legitimation, cleaning, stopwords removal and POS tagging.
# then use any methods you learned in the lecture to convert each response into meaningful numbers.
# 3- Apply Naive bayes and look at appropriate evaluation metric.
# 4- Explain your results very carefully.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')

# Load the CSV file into a DataFrame
file_path = 'data.csv'
data_df = pd.read_csv(file_path, encoding='latin1')


# Define the preprocess_text function
def preprocess_text(text_data):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    preprocessed_text = []
    for text in text_data:
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]
        tokens = [token for token in tokens if token not in string.punctuation and not re.match(r'^\W+$', token)]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [token for token in tokens if token not in stop_words]
        pos_tags = pos_tag(tokens)
        preprocessed_text.append(pos_tags)
    return preprocessed_text

# Preprocess the text data in the DataFrame
preprocessed_data = preprocess_text(data_df['text'])

# Print the first three strings of preprocessed text
for i in range(2):
    print("Preprocessed text", i+1, ":", preprocessed_data[i])

# Split data into training and testing sets:
# X contains features, y contains target variable
X = data_df.drop(columns=['label'])
y = data_df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_df['text'], data_df['label'], test_size=0.2, random_state=42)

print(y_train.unique())

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer()

# Fit TF-IDF vectorizer to training data and transform it
X_train_tfidf = tfidf.fit_transform(X_train)

# Initialize and train Multinomial Naive Bayes classifier
clf = MultinomialNB().fit(X_train_tfidf, y_train)

# Transform test data using fitted TF-IDF vectorizer
X_test_tfidf = tfidf.transform(X_test)

# Make predictions
predicted = clf.predict(X_test_tfidf)

# Calculate F1 score
f1 = f1_score(y_test, predicted, average='weighted')

# Print the F1 score
print("F1 Score (TF-IDF):", f1)

# Compare results to count vectorized data:
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
# Initialize and train Multinomial Naive Bayes classifier
clf = MultinomialNB().fit(X_train_counts, y_train)
# Make predictions
X_test_count = count_vect.transform(X_test)
predicted = clf.predict(X_test_count)
# Calculate F1 score
f1c = f1_score(y_test, predicted, average='weighted')
print("F1 Score (count):", f1c)

"""
Explanation: After preprocessing the text and transforming the data using the TF-IDF vectorizer, 
I was able to generate a Naive Bayes classifier model with an F1 score of 82%, which I deem to be acceptable.
However, using the count vectorizer after preprocessing produced a slightly higher F1 score of 83%.
"""
print(20 * '-' + 'End Q3' + 20 * '-')
# =================================================================
# Class_Ex4:
# Use Naive bayes classifier for this problem,
# Write a text classification pipeline to classify movie reviews as either positive or negative.
# Find a good set of parameters using grid search. hint: grid search on n gram
# Evaluate the performance on a held out test set.
# hint1: use nltk movie reviews dataset
# from nltk.corpus import movie_reviews

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

# Load the movie reviews dataset
reviews = [(movie_reviews.raw(fileid), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
X, y = zip(*reviews)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the text classification pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

# Define the parameters for grid search
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],  # unigrams, bigrams, and trigrams
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3)
}

# Perform grid search to find the best parameters
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Print the best parameters found by grid search
print("Best parameters:")
print(grid_search.best_params_)

# Evaluate the performance of the best model on the held-out test set
y_pred = grid_search.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================
# Class_Ex5:
# Calculate accuracy percentage between two lists
# calculate a confusion matrix
# Write your own code - No packages
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')


def calculate_accuracy(true_labels, predicted_labels):
    """
    Calculate the accuracy percentage between true and predicted labels.

    Parameters:
        true_labels (list): List of true labels.
        predicted_labels (list): List of predicted labels.

    Returns:
        accuracy (float): Accuracy percentage.
    """
    correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    total_predictions = len(true_labels)
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def calculate_confusion_matrix(true_labels, predicted_labels):
    """
    Calculate the confusion matrix between true and predicted labels.

    Parameters:
        true_labels (list): List of true labels.
        predicted_labels (list): List of predicted labels.

    Returns:
        confusion_matrix (dict): Confusion matrix as a dictionary.
    """
    confusion_matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for true, pred in zip(true_labels, predicted_labels):
        if true == 1 and pred == 1:
            confusion_matrix['TP'] += 1
        elif true == 0 and pred == 0:
            confusion_matrix['TN'] += 1
        elif true == 0 and pred == 1:
            confusion_matrix['FP'] += 1
        elif true == 1 and pred == 0:
            confusion_matrix['FN'] += 1
    return confusion_matrix


# Example usage:
true_labels = [1, 0, 1, 0, 1, 0, 1, 0]
predicted_labels = [1, 1, 1, 0, 1, 1, 0, 0]

accuracy = calculate_accuracy(true_labels, predicted_labels)
print("Accuracy:", accuracy)

confusion_matrix = calculate_confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:", confusion_matrix)

print(20 * '-' + 'End Q5' + 20 * '-')
# =================================================================
# Class_Ex6:
# Read the data.csv file.
# Answer the following question
# 1- In this dataset we have a lot of responses in text and each response has a label.
# 2- Our goal is to correctly model the texts into its label.
# Hint: you need to read the text responses and perform preprocessing on it.
# such as normalization, legitimation, cleaning, stopwords removal and POS tagging.
# then use any methods you learned in the lecture to convert each response into meaningful numbers.
# 3- Apply Logistic Regression  and look at appropriate evaluation metric.
# 4- Apply LSA method and compare results.
# 5- Explain your results very carefully.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')

# Load the CSV file into a DataFrame
file_path = 'data.csv'
data_df = pd.read_csv(file_path, encoding='latin1')


# Define the preprocess_text function
def preprocess_text(text_data):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    preprocessed_text = []
    for text in text_data:
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]
        tokens = [token for token in tokens if token not in string.punctuation and not re.match(r'^\W+$', token)]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [token for token in tokens if token not in stop_words]
        pos_tags = pos_tag(tokens)
        preprocessed_text.append(pos_tags)
    return preprocessed_text

# Preprocess the text data in the DataFrame
preprocessed_data = preprocess_text(data_df['text'])

# Print the first three strings of preprocessed text
for i in range(2):
    print("Preprocessed text", i+1, ":", preprocessed_data[i])

# Split data into training and testing sets:
# X contains features, y contains target variable
X = data_df.drop(columns=['label'])
y = data_df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_df['text'], data_df['label'], test_size=0.2, random_state=42)

print(y_train.unique())

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer()

# Fit TF-IDF vectorizer to training data and transform it
X_train_tfidf = tfidf.fit_transform(X_train)

# Initialize and train logistic regression classifier
clf = LogisticRegression().fit(X_train_tfidf, y_train)

# Transform test data using fitted TF-IDF vectorizer
X_test_tfidf = tfidf.transform(X_test)

# Make predictions
predicted = clf.predict(X_test_tfidf)

# Calculate F1 score
f1 = f1_score(y_test, predicted, average='weighted')

# Print the F1 score
print("F1 Score (logistic):", f1)

# Compare results to LSA:
# Initialize LSA
lsa = TruncatedSVD(n_components=100, random_state=42)

# Fit LSA to the TF-IDF transformed training data and transform it
X_train_lsa = lsa.fit_transform(X_train_tfidf)

# Transform the TF-IDF transformed testing data using the fitted LSA model
X_test_lsa = lsa.transform(X_test_tfidf)

# Initialize and train logistic regression classifier on LSA-transformed data
clf_lsa = LogisticRegression().fit(X_train_lsa, y_train)

# Make predictions on LSA-transformed testing data
predicted_lsa = clf_lsa.predict(X_test_lsa)

# Calculate F1 score for LSA-based model
f1_lsa = f1_score(y_test, predicted_lsa, average='weighted')

# Print the F1 score for the LSA-based model
print("F1 Score (LSA):", f1_lsa)

"""
Explanation: After preprocessing the text and transforming the data using the TF-IDF vectorizer, 
I generated a logistic regression classifier model with an F1 score of 87%, which I deem to be acceptable.
In comparison, using the additional LSA transformation produced a lower F1 score of 82%.
"""

print(20 * '-' + 'End Q6' + 20 * '-')

# =================================================================
# Class_Ex7:
# Use logistic regression classifier for this problem,
# Write a text classification pipeline to classify movie reviews as either positive or negative.
# Find a good set of parameters using grid search. hint: grid search on n-gram
# Evaluate the performance on a held out test set.
# hint1: use nltk movie reviews dataset
# from nltk.corpus import movie_reviews
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

# Load the movie reviews dataset
reviews = [(movie_reviews.raw(fileid), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
X, y = zip(*reviews)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the text classification pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression())
])

# Define the parameters for grid search
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],  # unigrams, bigrams, and trigrams
    'tfidf__use_idf': (True, False),
    'clf__max_iter': [100, 200, 300]  # Max number of iterations for logistic regression
}

# Perform grid search to find the best parameters
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Print the best parameters found by grid search
print("Best parameters:")
print(grid_search.best_params_)

# Evaluate the performance of the best model on the held-out test set
y_pred = grid_search.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(20 * '-' + 'End Q7' + 20 * '-')
