import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Load the dataset
df = pd.read_csv("data_set.csv")

# Assuming "text" is the column containing the text data
texts = df['text'].tolist()

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

# Apply text preprocessing
processed_texts = [preprocess_text(text) for text in texts]

# Tokenize the preprocessed text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_texts)

# Convert text to sequences of word indexes
sequences = tokenizer.texts_to_sequences(processed_texts)

# Pad sequences to a fixed length (adjust maxlen as needed)
maxlen = 100  # Example maximum sequence length
padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')


class Autoencoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Encoder layers
        self.encoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.encoder_hidden = tf.keras.layers.Dense(hidden_dim, activation='relu')

        # Decoder layers
        self.decoder_hidden = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self.decoder_embedding = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        # Encoder
        encoded = self.encoder_embedding(inputs)
        encoded = self.encoder_hidden(encoded)

        # Decoder
        decoded = self.decoder_hidden(encoded)
        decoded = self.decoder_embedding(decoded)

        return decoded


# Example usage:
# Define hyperparameters
vocab_size = 10000  # Example vocabulary size
embedding_dim = 30  # Dimension of word embeddings
hidden_dim = 50  # Dimension of the hidden layer

# Create an instance of the autoencoder model
model = Autoencoder(vocab_size, embedding_dim, hidden_dim)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Define callbacks for printing metrics
class PrintMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}/{self.params['epochs']}, Loss: {logs['loss']}")

# Train the model with your dataset
# Use padded_sequences as X_train
model.fit(padded_sequences, padded_sequences, epochs=10, batch_size=32, callbacks=[PrintMetricsCallback()])
