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
