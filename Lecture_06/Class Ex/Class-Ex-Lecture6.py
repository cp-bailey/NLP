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

print(20 * '-' + 'End Q1' + 20 * '-')

# =================================================================
# Class_Ex2:
# Use the same data in Example 1 but instead of hard-lim use log sigmoid as transfer function.
# modify your code inorder to classify negative and positive sentences correctly.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')





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

# Train ADALINE network to find  a relationship between POS (just verbs and nouns) and the the length of the sentences.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================
# Class_Ex5:
# Read the dataset.csv file. This dataset is about the EmailSpam.
# Use a two layer network and to classify each email
# You are not allowed to use any NN packages.
# You can use previous NLP packages to read the data process it (NLTK, spaCY)
# Show the classification report and mse of training and testing.
# Try to improve your F1 score. Explain which methods you used.
# Hint. Clean the dataset, use all the preprocessing techniques that you learned.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')





print(20 * '-' + 'End Q5' + 20 * '-')
# =================================================================
