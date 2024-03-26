import numpy as np


class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, learning_rate):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        # Initialize weights
        self.W1 = np.random.randn(embedding_dim, vocab_size)
        self.W2 = np.random.randn(vocab_size, embedding_dim)

    def forward(self, input_word):
        self.input_word = input_word
        self.hidden_layer = np.dot(self.W1, input_word)
        self.output_layer = np.dot(self.W2, self.hidden_layer)
        return self.output_layer

    def backward(self, target_word):
        error = np.mean(np.square(self.output_layer - target_word))
        grad_output = 2 * (self.output_layer - target_word) / target_word.shape[0]
        grad_W2 = np.outer(grad_output, self.hidden_layer)
        grad_hidden = np.dot(self.W2.T, grad_output)
        grad_W1 = np.outer(grad_hidden, self.input_word)

        # Update weights
        self.W1 -= self.learning_rate * grad_W1
        self.W2 -= self.learning_rate * grad_W2

        return error

    def train(self, input_words, target_words, epochs):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(input_words)):
                input_word = input_words[i]
                target_word = target_words[i]

                # Forward pass
                self.forward(input_word)

                # Backward pass and weight update
                error = self.backward(target_word)
                total_error += error

            avg_error = total_error / len(input_words)
            print(f"Epoch {epoch + 1}/{epochs}, Average Error: {avg_error}")

    def word_embeddings(self):
        return self.W1.T  # Return the transpose of W1 as word embeddings


# Example usage
def main():
    # Sample text data
    text = "natural language processing and machine learning are fun and exciting"

    # Preprocessing
    words = text.lower().split()
    vocab = sorted(set(words))
    word_to_index = {w: i for i, w in enumerate(vocab)}
    index_to_word = {i: w for i, w in enumerate(vocab)}

    # Generate training data
    window_size = 2
    training_data = []
    for i in range(len(words)):
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                training_data.append((word_to_index[words[i]], word_to_index[words[j]]))

    # Convert training data to numpy arrays
    input_words = np.zeros((len(training_data), len(vocab)))
    target_words = np.zeros((len(training_data), len(vocab)))
    for i, (input_index, target_index) in enumerate(training_data):
        input_words[i, input_index] = 1
        target_words[i, target_index] = 1

    # Hyperparameters
    vocab_size = len(vocab)
    embedding_dim = 20
    learning_rate = 0.01
    epochs = 100

    # Initialize and train Word2Vec model
    model = Word2Vec(vocab_size, embedding_dim, learning_rate)
    model.train(input_words, target_words, epochs)

    # Get word embeddings
    word_embeddings = model.word_embeddings()

    # Print word embeddings
    for i, embedding in enumerate(word_embeddings):
        print(f"Word: {index_to_word[i]}, Embedding: {embedding}")


if __name__ == "__main__":
    main()
