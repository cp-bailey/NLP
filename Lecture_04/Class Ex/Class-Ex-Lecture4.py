import re
import ssl
from urllib import request
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from collections import Counter
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

# =================================================================
# Class_Ex1:
# Write a function that checks a string contains only a certain set of characters
# (all chars lower and upper case with all digits).
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

def contains_only_allowed_chars(input_string):
    # Define the regular expression pattern
    pattern = r'^[a-zA-Z0-9]+$'
    # Use the re.match() function to check if the input string matches the pattern
    if re.match(pattern, input_string):
        return True
    else:
        return False

# Test the function
input_string = input("Enter a string: ")
if contains_only_allowed_chars(input_string):
    print("The string contains only allowed characters.")
else:
    print("The string contains characters other than lowercase and uppercase letters and digits.")

print(20 * '-' + 'End Q1' + 20 * '-')

# =================================================================
# Class_Ex2:
# Write a function that matches a string in which a followed by zero or more b's.
# Sample String 'ac', 'abc', abbc'
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

# Sample String
sample_strings = ['ac', 'abc', 'abbc']

pattern1 = re.compile(r'[a]b*')

for string in sample_strings:
    matches = pattern1.finditer(string)
    print("Matching for string:", string)
    for match in matches:
        print(match)

print(20 * '-' + 'End Q2' + 20 * '-')
# =================================================================
# Class_Ex3:
# Write Python script to find numbers between 1 and 3 in a given string.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')

# Sample String
sample_strings = ['2300991']

pattern2 = re.compile(r'[1-3]')

for string in sample_strings:
    matches = pattern2.finditer(string)
    print("Numbers between 1 and 3 in the given string:", string)
    for match in matches:
        print(match)

print(20 * '-' + 'End Q3' + 20 * '-')
# =================================================================
# Class_Ex4:
# Write a Python script to find the a position of the substrings within a string.
# text = 'Python exercises, JAVA exercises, C exercises'
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

text = 'Python exercises, JAVA exercises, C exercises'

# Convert the text to lowercase
text_lower = text.lower()

pattern3 = re.compile(r'a')

# Find all matches and their positions in the lowercase string
matches = pattern3.finditer(text_lower)

# Print the positions of 'a' in the string
print("Positions of 'a' in the lowercase string:", text)
for match in matches:
    print("Position:", match.start())

print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================
# Class_Ex5:
# Write a Python script to find if two strings from a list starting with letter 'C'.
# words = ["Cython CHP", "Java JavaScript", "PERL S+"]
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')

words = ["Cython CHP", "Java JavaScript", "PERL S+"]

pattern4 = re.compile(r'^C', re.IGNORECASE)

# Find if two strings start with 'C'
found = False
for i in range(len(words)):
    for j in range(i + 1, len(words)):
        if pattern4.match(words[i]) and pattern4.match(words[j]):
            found = True
            break
    if found:
        break

# Print the result
if found:
    print("Two strings starting with 'C' are found.")
else:
    print("No two strings starting with 'C' are found.")

print(20 * '-' + 'End Q5' + 20 * '-')

# =================================================================
# Class_Ex6:
# Write a Python script to remove everything except chars and digits from a string.
# USe sub method
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')

string = "I can't wait to try this! @!*"

# Define a regular expression pattern to match non-alphanumeric characters
pattern5 = r'[^a-zA-Z0-9 ]'

# Use the sub method to replace non-alphanumeric characters with an empty string
clean_string = re.sub(pattern5, '', string)

# Print the cleaned string
print("Original string:", string)
print("Cleaned string:", clean_string)

print(20 * '-' + 'End Q6' + 20 * '-')
# =================================================================
# Class_Ex7:
# Scrape the following website
# https://en.wikipedia.org/wiki/Natural_language_processing
# Find the tag which related to the text. Extract all the textual data.
# Tokenize the cleaned text file.
# print the len of the corpus and print couple of the sentences.
# Calculate the words frequencies.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

ssl._create_default_https_context = ssl._create_unverified_context # Disable SSL certificate verification

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set of English stop words
stop_words = set(stopwords.words('english'))

def scrape_webpage(url):
    try:
        # Fetch HTML content from the URL
        html = request.urlopen(url).read().decode('utf8')

        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Find the relevant tag containing the main textual content
        # Here, we'll assume <p> tags contain the main text, adjust as needed
        main_content_tag = soup.find('div', class_='mw-parser-output')
        text = ''
        if main_content_tag:
            # Extract text content from the main tag
            paragraphs = main_content_tag.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])

        # Remove HTML tags and extra whitespace
        cleaned_text = ' '.join(text.split())

        # Define string of punctuation characters
        punctuation_chars = '''()-[]{};:'"\,<>/@#$%^&*_~'''

        # Remove punctuation
        cleaned_text = ''.join(char for char in cleaned_text if char not in punctuation_chars)

        # Tokenize the cleaned text
        tokens = word_tokenize(cleaned_text)

        # Remove stop words
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

        # Join filtered tokens back into a single string
        cleaned_text = ' '.join(filtered_tokens)

        return cleaned_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Call the function with the URL of the webpage
url = 'https://en.wikipedia.org/wiki/Natural_language_processing'
raw_text = scrape_webpage(url)

tokens = word_tokenize(raw_text)
sent_tokens = sent_tokenize(raw_text)

# Printing the length of the corpus
print("Length of the corpus:", len(tokens))

# Printing a couple of sentences
print("First three sentences:")
for i in range(3):  # Print the first 5 sentences
    print(sent_tokens[i])  # Print individual sentences

# Calculate word frequencies
word_freq = Counter(tokens)

# Print the most common words and their frequencies (excluding punctuation)
print("Most common words and their frequencies:")
for word, freq in word_freq.most_common(10):  # Print the top 10 most common words
    # Check if the word consists of alphabetic characters
    if word.isalpha():
        print(f"{word}: {freq}")

print(20 * '-' + 'End Q7' + 20 * '-')
# =================================================================
# Class_Ex8:
# Grab any text from Wikipedia and create a string of 3 sentences.
# Use that string and calculate the ngram of 1 from nltk package.
# Use BOW method and compare the most 3 common words.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q8' + 20 * '-')

# Initialize an empty string to store the first three sentences
first_three_sentences = ""

# Concatenate the first three sentences into a single string
for i in range(3):
    first_three_sentences += sent_tokens[i] + " "

# Print the resulting string
print("First three sentences as a single string:")
print(first_three_sentences)

# Tokenize the string into words
words = word_tokenize(first_three_sentences)

# Calculate unigrams (n-grams of size 1)
unigrams = list(ngrams(words, 1))

# Count the occurrences of each unigram
unigram_freq = Counter(unigrams)

# Print the most common unigrams
print("Three most common unigrams:")
for unigram, freq in unigram_freq.most_common(10):
    if unigram[0].isalpha():  # Check if the first element of the tuple is alphabetic
        print(unigram[0], ":", freq)

# Using BOW method
# Remove punctuation and convert to lowercase
words = [word.lower() for word in words if word.isalnum()]

# Calculate word frequencies
word_freq = Counter(words)

# Print the most common words
print("Three most common words:")
for word, freq in word_freq.most_common(10):
    print(word, ":", freq)

print(20 * '-' + 'End Q8' + 20 * '-')
# =================================================================
# Class_Ex9:
# Write a python script that accepts any string and do the following.
# 1- Tokenize the text
# 2- Doe word extraction and clean a text. Use regular expression to clean a text.
# 3- Generate BOW
# 4- Vectorized all the tokens.
# 5- The only package you can use is numpy and re.
# all sentences = ["sentence1", "sentence2", "sentence3",...]
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q9' + 20 * '-')

def tokenize_text(text):
    """Tokenize the text into words."""
    return text.split()

def clean_text(tokens):
    """Clean the tokens using regular expressions."""
    cleaned_tokens = []
    pattern = re.compile(r'\b[A-Za-z0-9]+\b')  # Pattern to match alphabetic words
    for token in tokens:
        if re.match(pattern, token):
            cleaned_tokens.append(token.lower())  # Convert to lowercase and keep alphabetic words
    return cleaned_tokens

def generate_bow(tokens):
    """Generate a bag of words (BOW) from the tokens."""
    bow = {}
    for token in tokens:
        bow[token] = bow.get(token, 0) + 1  # Count the frequency of each word
    return bow

def vectorize_tokens(tokens, bow):
    """Vectorize the tokens using the bag of words."""
    vector = np.zeros(len(bow))
    for i, word in enumerate(bow):
        if word in tokens:
            vector[i] = tokens.count(word)  # Assign the frequency of each word to the vector
    return vector

# Example sentences
sentences = ["This is sentence 1.", "Sentence number 2 is here!", "And here comes sentence 3."]

# Tokenize each sentence
tokenized_sentences = [tokenize_text(sentence) for sentence in sentences]

# Clean each tokenized sentence
cleaned_sentences = [clean_text(tokens) for tokens in tokenized_sentences]

# Generate BOW for all sentences combined
all_tokens = [token for sentence_tokens in cleaned_sentences for token in sentence_tokens]
bow = generate_bow(all_tokens)

# Vectorize each cleaned sentence
vectorized_sentences = [vectorize_tokens(tokens, bow) for tokens in cleaned_sentences]

# Print the results
print("Tokenized Sentences:", tokenized_sentences)
print("Cleaned Sentences:", cleaned_sentences)
print("Bag of Words:", bow)
print("Vectorized Sentences:", vectorized_sentences)

print(20 * '-' + 'End Q9' + 20 * '-')
# =================================================================
# Class_Ex10:
# Grab any text (almost a paragraph) from Wikipedia and call it text
# Preprocessing the text data (Normalize, remove special char, ...)
# Find total number of unique words
# Create an index for each word.
# Count number of the words.
# Define a function to calculate Term Frequency
# Define a function calculate Inverse Document Frequency
# Combining the TF-IDF functions
# Apply the TF-IDF Model to our text
# you are allowed to use just numpy and nltk tokenizer
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q10' + 20 * '-')

# Step 1: Preprocess the text data
def preprocess_text(text):
    # Normalize the text (convert to lowercase)
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenize the text using NLTK tokenizer
    tokens = word_tokenize(text)
    return tokens

# Step 2: Find total number of unique words and create an index for each word
def create_word_index(tokens):
    # Get unique words
    unique_words = sorted(set(tokens))
    # Create an index for each word
    word_index = {word: index for index, word in enumerate(unique_words)}
    return word_index

# Step 3: Count the number of occurrences of each word
def count_words(tokens, word_index):
    word_counts = np.zeros(len(word_index))
    for token in tokens:
        if token in word_index:
            word_counts[word_index[token]] += 1
    return word_counts

# Step 4: Define a function to calculate Term Frequency (TF)
def calculate_tf(word_counts):
    total_words = np.sum(word_counts)
    tf = word_counts / total_words
    return tf

# Step 5: Define a function to calculate Inverse Document Frequency (IDF)
def calculate_idf(word_counts, num_documents):
    idf = np.log(num_documents / (1 + word_counts))
    return idf

# Step 6: Combine the TF-IDF functions
def calculate_tfidf(tf, idf):
    return tf * idf

# Step 7: Apply the TF-IDF model to our text
def apply_tfidf(text, word_index, idf):
    # Preprocess the text
    tokens = preprocess_text(text)
    # Count the words
    word_counts = count_words(tokens, word_index)
    # Calculate TF
    tf = calculate_tf(word_counts)
    # Calculate TF-IDF
    tfidf = calculate_tfidf(tf, idf)
    return tfidf

# Example text from: https://en.wikipedia.org/wiki/Polar_bear
text = "The polar bear is a large bear native to the Arctic and nearby areas. \
Closely related to the brown bear, the polar bear is the largest extant species of bear and land carnivore, \
with adult males weighing 300 to 800 kg (700 to 1,800 lb). \
It has white or yellowish fur with black skin and a thick layer of fat. \
Polar bears live both on land and on sea ice, and usually live solitarily. \
They mainly prey on seals, especially ringed seals. \
Male bears guard females during the breeding season and defend them from rivals. \
Mothers give birth to cubs in maternity dens during the winter. \
The International Union for Conservation of Nature considers polar bears a vulnerable species. \
Their biggest threat is climate change as global warming has led to a decline in sea ice in the Arctic. \
They have been hunted for their coats, meat and other items. \
They have been kept in captivity and have played important roles in culture."

# Preprocess the text
tokens = preprocess_text(text)

# Step 2: Create word index
word_index = create_word_index(tokens)

# Define the number of documents (assuming there's only one document)
num_documents = 1

# Step 3: Count the number of occurrences of each word
word_counts = count_words(tokens, word_index)

# Step 5: Calculate IDF
idf = calculate_idf(word_counts, num_documents)

# Step 7: Apply the TF-IDF model to our text
tfidf = apply_tfidf(text, word_index, idf)

# Count the number of unique words
unique_words = set(tokens)
total_unique_words = len(unique_words)

# Count number of total words
total_words = len(tokens)

print("Total number of unique words:", total_unique_words)
print("Total number of words:", total_words)

print("TF-IDF for each word:")
for word, tfidf_score in zip(word_index.keys(), tfidf):
    print(f"{word}: {tfidf_score:.4f}")

print(20 * '-' + 'End Q10' + 20 * '-')
# =================================================================
# Class_Ex11:
# Grab arbitrary paragraph from any website.
# Creat  a list of stopwords manually.  Example :  stopwords = ['and', 'for', 'in', 'little', 'of', 'the', 'to']
# Create a list of ignore char Example: ' :,",! '
# Write a LSA class with the following functions.
# Parse function which tokenize the words lower cases them and count them. Use dictionary; keys are the tokens and value is count.
# Clac function that calculate SVD.
# TFIDF function
# Print function which print out the TFIDF matrix, first 3 columns of the U matrix and first 3 rows of the Vt matrix
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q11' + 20 * '-')

stopwords = ['and', 'for', 'in', 'of', 'the', 'to', 'a', 'is']

ignore_chars = ' :,",! '
ignore_list = list(ignore_chars)

class LSA:
    def __init__(self, text, stopwords=[], ignore_chars=[]):
        self.text = text
        self.stopwords = stopwords
        self.ignore_chars = ignore_chars
        self.tokens = self.parse()
        self.tfidf_matrix = None
        self.U = None
        self.Vt = None

    def parse(self):
        # Tokenize the text, lowercasing, removing stopwords, and ignoring specified characters
        tokens = re.findall(r'\b\w+\b', self.text.lower())
        tokens = [token for token in tokens if token not in self.stopwords]
        tokens = [token for token in tokens if token not in self.ignore_chars]
        # Count the tokens
        token_count = {}
        for token in tokens:
            if token in token_count:
                token_count[token] += 1
            else:
                token_count[token] = 1
        return token_count

    def tfidf(self):
        # Convert token count dictionary into a list of documents (in this case, a single document)
        documents = [' '.join([token] * count) for token, count in self.tokens.items()]
        # Calculate TF-IDF matrix
        vectorizer = TfidfVectorizer()
        self.tfidf_matrix = vectorizer.fit_transform(documents)

    def calc_svd(self):
        # Perform Singular Value Decomposition (SVD)
        svd = TruncatedSVD(n_components=3)
        self.U = svd.fit_transform(self.tfidf_matrix)
        self.Vt = svd.components_

    def print_results(self):
        print("TF-IDF Matrix:")
        print(self.tfidf_matrix.toarray())
        print("\nFirst 3 columns of U matrix:")
        print(self.U[:, :3])
        print("\nFirst 3 rows of Vt matrix:")
        print(self.Vt[:3, :])


# Example text from: https://en.wikipedia.org/wiki/Polar_bear
text = "The polar bear is a large bear native to the Arctic and nearby areas. \
Closely related to the brown bear, the polar bear is the largest extant species of bear and land carnivore, \
with adult males weighing 300 to 800 kg (700 to 1,800 lb). \
It has white or yellowish fur with black skin and a thick layer of fat. \
Polar bears live both on land and on sea ice, and usually live solitarily. \
They mainly prey on seals, especially ringed seals. \
Male bears guard females during the breeding season and defend them from rivals. \
Mothers give birth to cubs in maternity dens during the winter. \
The International Union for Conservation of Nature considers polar bears a vulnerable species. \
Their biggest threat is climate change as global warming has led to a decline in sea ice in the Arctic. \
They have been hunted for their coats, meat and other items. \
They have been kept in captivity and have played important roles in culture."

# Initialize LSA object
lsa = LSA(text, stopwords=stopwords, ignore_chars=ignore_list)

# Calculate TF-IDF matrix
lsa.tfidf()

# Perform SVD
lsa.calc_svd()

# Print results
lsa.print_results()

print(20 * '-' + 'End Q11' + 20 * '-')
# =================================================================
# Class_Ex12:
# Use the following doc
#  = ["An intern at OpenAI", "Developer at OpenAI", "A ML intern", "A ML engineer" ]
# Calculate the binary BOW.
# Use LSA method and distinguish two different topic from the document. Sent 1,2 is about OpenAI and sent3, 4 is about ML.
# Use pandas to show the values of dataframe and lsa components. Show there is two distinct topic.
# Use numpy take the absolute value of the lsa matrix sort them and use some threshold and see what words are the most important.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q12' + 20 * '-')

doc = ["An intern at OpenAI", "Developer at OpenAI", "A ML intern", "A ML engineer"]

# Step 1: Calculate the binary Bag-of-Words (BOW)
vectorizer = CountVectorizer(binary=True)
binary_bow = vectorizer.fit_transform(doc)
vocab = vectorizer.get_feature_names_out()

# Step 2: Apply LSA
vectorizer_tfidf = TfidfVectorizer()
X = vectorizer_tfidf.fit_transform(doc)

lsa = TruncatedSVD(n_components=2, n_iter=100)
lsa.fit(X)

# Step 3: Display DataFrame and LSA components
terms = vectorizer_tfidf.get_feature_names_out()
concept_names = [f"Concept {i+1}" for i in range(lsa.components_.shape[0])]
lsa_components_df = pd.DataFrame(lsa.components_, columns=terms, index=concept_names)

print("Binary BOW:")
print(binary_bow.toarray())

print("\nLSA Components:")
print(lsa_components_df.to_string())

# Step 4: Identify important words using a threshold
threshold = 0.4
important_words = {}

for i, comp in enumerate(lsa.components_):
    important_terms = np.where(np.abs(comp) > threshold)[0]
    important_words[f"Concept {i+1}"] = [terms[idx] for idx in important_terms]

print("\nImportant Words:")
print(important_words)

print(20 * '-' + 'End Q12' + 20 * '-')
# =================================================================

