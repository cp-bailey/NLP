# =================================================================
# Class_Ex1:
# Lets consider the 2 following sentences
# Sentence 1: I  am excited about the perceptron network.
# Sentence 2: we will not test the classifier with real data.
# Design your bag of words set and create your input set.
# Choose your BOW words that suits perceptron network.
# Design your classes that Sent 1 has positive sentiment and sent 2 has a negative sentiment.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

from sklearn.feature_extraction.text import CountVectorizer

# Sentences
sentences = [
    "I am excited about the perceptron network.",
    "We will not test the classifier with real data."
]

# Bag of words set relevant to perceptron network and sentiment
bow_set = [
    'perceptron', 'network', 'excited', 'test', 'classifier',
    'real', 'data', 'positive', 'negative', 'sentiment', 'not'
]

# Initialize CountVectorizer with custom vocabulary
vectorizer = CountVectorizer(vocabulary=bow_set)

# Transform the sentences into input vectors
X = vectorizer.fit_transform(sentences)

# Classes
# Class 1: Positive sentiment (Sentence 1)
# Class 0: Negative sentiment (Sentence 2)
y = [1, 0]

# Print the input vectors and classes
print("Input vectors:")
print(X.toarray())
print("\nClasses:", y)


print(20 * '-' + 'End Q1' + 20 * '-')

# =================================================================
# Class_Ex2:
# Use the same data in Example 1 but instead of hard-lim use log sigmoid as transfer function.
# modify your code inorder to classify negative and positive sentences correctly.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Sentences
sentences = [
    "I am excited about the perceptron network.",
    "We will not test the classifier with real data."
]

# Bag of words set relevant to perceptron network and sentiment
bow_set = [
    'perceptron', 'network', 'excited', 'test', 'classifier',
    'real', 'data', 'positive', 'negative', 'sentiment', 'not'
]

# Initialize CountVectorizer with custom vocabulary
vectorizer = CountVectorizer(vocabulary=bow_set)

# Transform the sentences into input vectors
X = vectorizer.fit_transform(sentences)

# Classes
# Class 1: Positive sentiment (Sentence 1)
# Class 0: Negative sentiment (Sentence 2)
y = np.array([1, 0])

# Train logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X, y)

# Predict sentiment for new sentences
new_sentences = [
    "The perceptron model performed admirably.",
    "The classifier failed to generalize well."
]
X_new = vectorizer.transform(new_sentences)
predictions = model.predict(X_new)

# Print predictions
for sentence, prediction in zip(new_sentences, predictions):
    if prediction == 1:
        print(f'Sentence: "{sentence}" Predicted Sentiment: Positive')
    else:
        print(f'Sentence: "{sentence}" Predicted Sentiment: Negative')


print(20 * '-' + 'End Q2' + 20 * '-')

# =================================================================
# Class_Ex2_1:

# For preprocessing, the text data is vectorized into feature vectors using a bag-of-words approach.
# Each sentence is converted into a vector where each element represents the frequency of a word from the vocabulary.
# This allows the textual data to be fed into the perceptron model.

# The training data consists of sample text sentences and corresponding sentiment labels (positive or negative).
# The text is vectorized and used to train the Perceptron model to associate words with positive/negative sentiment.

# For making predictions, new text input is vectorized using the same vocabulary. Then the Perceptron model makes a
# binary prediction on whether the new text has positive or negative sentiment.
# The output is based on whether the dot product of the input vector with the trained weight vectors is positive
# or negative.

# This provides a simple perceptron model for binary sentiment classification on textual data. The vectorization
# allows text to be converted into numerical features that the perceptron model can process. Overall,
# it demonstrates how a perceptron can be used for an NLP text classification task.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2_1' + 20 * '-')

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx in range(n_samples):
                linear_output = np.dot(X[idx], self.weights) + self.bias
                y_pred = np.sign(linear_output)
                update = self.lr * (y[idx] - y_pred)
                self.weights += update * X[idx]
                self.bias += update

                # Check if predictions match the ground truth
                if y[idx] != y_pred:
                    print("Misclassified sample at index:", idx)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = np.sign(linear_output)
        return y_pred

# Sample training data
X_train = [
    "I loved this movie, it was so much fun!",
    "The food at this restaurant is not good. Don't go there!",
    "The new iPhone looks amazing, can't wait to get my hands on it."
]
y_train = np.array([1, -1, 1])

# Initialize CountVectorizer with custom vocabulary
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train).toarray()

# Initialize and train the perceptron
perceptron = Perceptron()
perceptron.fit(X_train_bow, y_train)

# Sample test data
X_test = [
    "I hated this movie, it was boring.",
    "The service at this restaurant is excellent!"
]

# Preprocess the test data
X_test_bow = vectorizer.transform(X_test).toarray()

# Predictions
predictions = perceptron.predict(X_test_bow)
print(predictions)

print(20 * '-' + 'End Q2_1' + 20 * '-')

# =================================================================
# Class_Ex3:
# The following function is given
# F(x) = x1^2 + 2 x1 x2 + 2 x2^2 +x1
# use the steepest decent algorithm to find the minimum of the function.
# Plot the function in 3d and then plot the counter plot with the all the steps.
# use small value as a learning rate.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function and its gradient
def F(x):
    return x[0]**2 + 2*x[0]*x[1] + 2*x[1]**2 + x[0]

def gradient_F(x):
    return np.array([2*x[0] + 2*x[1] + 1, 2*x[0] + 4*x[1]])

# Steepest descent algorithm
def steepest_descent(initial_x, learning_rate, iterations):
    x = initial_x
    steps = [x]
    for _ in range(iterations):
        grad = gradient_F(x)
        x = x - learning_rate * grad
        steps.append(x)
    return np.array(steps)

# Define the range for plotting
x1_range = np.linspace(-2, 2, 100)
x2_range = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = F([X1, X2])

# Plotting the function in 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('F(x)')
ax.set_title('Function in 3D')

# Run steepest descent algorithm
initial_x = np.array([1.5, 1.5])
learning_rate = 0.01
iterations = 100
steps = steepest_descent(initial_x, learning_rate, iterations)

# Plotting the contour plot with all steps
plt.figure(figsize=(8, 6))
plt.contour(X1, X2, Z, levels=20)
plt.plot(steps[:, 0], steps[:, 1], marker='o', color='r')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contour Plot with Steps of Steepest Descent')
plt.grid(True)
plt.show()

print(20 * '-' + 'End Q3' + 20 * '-')

# =================================================================
# Class_Ex4:
# Use the following corpus of data
# sent1 : 'This is a sentence one, and I want to all data here.',
# sent2 :  'Natural language processing has nice tools for text mining and text classification.
#           I need to work hard and try a lot of exercises.',
# sent3 :  'Ohhhhhh what',
# sent4 :  'I am not sure what I am doing here.',
# sent5 :  'Neural Network is a power method. It is a very flexible architecture'

# Train ADALINE network to find  a relationship between POS (just verbs and nouns) and the length of the sentences.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Data
sentences = [
    'This is a sentence one, and I want to all data here.',
    'Natural language processing has nice tools for text mining and text classification. I need to work hard and try a lot of exercises.',
    'Ohhhhhh what',
    'I am not sure what I am doing here.',
    'Neural Network is a power method. It is a very flexible architecture'
]

# Preprocessing
tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
pos_tagged_sentences = [pos_tag(tokens) for tokens in tokenized_sentences]
sentence_lengths = [len(tokens) for tokens in tokenized_sentences]

# Feature extraction
X = [[sum(1 for _, tag in sent if tag.startswith('V') or tag.startswith('N')), length] for sent, length in zip(pos_tagged_sentences, sentence_lengths)]
y = sentence_lengths

# Training ADALINE
adaline = LinearRegression()
adaline.fit(X, y)

# Evaluation
predictions = adaline.predict(X)
mse = mean_squared_error(y, predictions)
print("Mean Squared Error:", mse)

# Prediction (example)
new_sentence = "I love programming"
new_tokens = word_tokenize(new_sentence)
new_pos_tags = pos_tag(new_tokens)
new_length = len(new_tokens)
new_X = [[sum(1 for _, tag in new_pos_tags if tag.startswith('V') or tag.startswith('N')), new_length]]
predicted_length = adaline.predict(new_X)
print("Predicted length of the sentence:", predicted_length[0])

print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================
# Class_Ex5:
# Read the data_set.csv file. This dataset is about the EmailSpam.
# Use a two layer network and to classify each email
# You are not allowed to use any NN packages.
# You can use previous NLP packages to read the data process it (NLTK, spaCY)
# Show the classification report and mse of training and testing.
# Try to improve your F1 score. Explain which methods you used.
# Hint. Clean the dataset, use all the preprocessing techniques that you learned.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')

import csv
import re
import random
import numpy as np
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Read the dataset
def read_dataset(filename):
    texts = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            texts.append(row['text'])
            labels.append(int(row['label_num']))
    return texts, labels


# Preprocess text data
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


# Read and preprocess dataset
texts, labels = read_dataset('data_set.csv')
processed_texts = [preprocess_text(text) for text in texts]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(processed_texts, labels, test_size=0.2, random_state=42)

# Build vocabulary
word_to_index = {}
index_to_word = {}
for text in X_train:
    for word in text:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
            index_to_word[len(word_to_index) - 1] = word


# Convert text data to vectors
def text_to_vector(text, word_to_index):
    vector = np.zeros(len(word_to_index))
    for word in text:
        if word in word_to_index:
            vector[word_to_index[word]] += 1
    return vector


X_train_vectors = np.array([text_to_vector(text, word_to_index) for text in X_train])
X_test_vectors = np.array([text_to_vector(text, word_to_index) for text in X_test])


# Define two-layer neural network
class TwoLayerNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.probs = self.softmax(self.z2)
        return self.probs

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        delta3 = self.probs
        delta3[range(m), y] -= 1
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Update parameters
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs=20, learning_rate=0.01): # increasing number of epochs increases F1 scores
        for epoch in range(epochs):
            probs = self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


# Train the model
input_size = len(word_to_index)
hidden_size = 10
output_size = 2  # Two classes: spam or not spam
model = TwoLayerNN(input_size, hidden_size, output_size)
model.train(X_train_vectors, y_train)

# Evaluate the model
train_predictions = model.predict(X_train_vectors)
test_predictions = model.predict(X_test_vectors)

# Classification report
train_report = classification_report(y_train, train_predictions)
test_report = classification_report(y_test, test_predictions)

# Mean Squared Error
train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print("Training Classification Report:")
print(train_report)
print("Training MSE:", train_mse)

print("\nTesting Classification Report:")
print(test_report)
print("Testing MSE:", test_mse)

# To improve my F1 score, I increased the number of epochs from 3 to 8 to 10, to 20.
# Each increase resulted in better accuracy scores.

print(20 * '-' + 'End Q5' + 20 * '-')
# =================================================================
# Class_Ex6:

# Follow the below instruction for writing the auto encoder code.

#The code implements a basic autoencoder model to learn word vector representations (word2vec style embeddings).
# It takes sentences of words as input and maps each word to an index in a vocabulary dictionary.

#The model has an encoder portion which converts word indexes into a low dimensional embedding via a learned weight
# matrix W1. This embedding is fed through another weight matrix W2 to a hidden layer.

#The decoder portion maps the hidden representation back to the original word index space via weight matrix W3.

#The model is trained to reconstruct the original word indexes from the hidden embedding by minimizing the
# reconstruction loss using backpropagation.

#After training, the weight matrix W1 contains the word embeddings that map words in the vocabulary to dense
# vector representations. These learned embeddings encode semantic meaning and can be used as features for
# downstream NLP tasks.


# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')





print(20 * '-' + 'End Q6' + 20 * '-')

# =================================================================
# Class_Ex7:
#
# The objective of this exercise to show the inner workings of Word2Vec in python using numpy.
# Do not be using any other libraries for that.
# We are not looking at efficient implementation, the purpose here is to understand the mechanism
# behind it. You can find the official paper here. https://arxiv.org/pdf/1301.3781.pdf
# The main component of your code should be the followings:
# Set your hyper-parameters
# Data Preparation (Read text file)
# Generate training data (indexing to an integer and the onehot encoding )
# Forward and backward steps of the autoencoder network
# Calculate the error
# look at error at by varying hidden dimensions and window size
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')



print(20 * '-' + 'End Q7' + 20 * '-')