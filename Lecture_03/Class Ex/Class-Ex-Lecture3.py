# =================================================================
# Class_Ex1:
# Import spacy abd from the language class import english.
# Create a doc object
# Process a text : This is a simple example to initiate spacy
# Print out the document text from the doc object.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

from spacy.lang.en import English
nlp = English()

doc = nlp('Hello world!')
print(doc.text)

print(20 * '-' + 'End Q1' + 20 * '-')
# =================================================================
# Class_Ex2:
# Solve Ex1 but this time use German Language.
# Grab a sentence from german text from any website.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

import requests
from bs4 import BeautifulSoup
from spacy.lang.de import German

# Initialize the German language model
nlp = German()

# Add the sentencizer component to the pipeline
if not nlp.has_pipe("sentencizer"):
    sentencizer = nlp.add_pipe("sentencizer")

# URL of the German Wikipedia page
url = "https://de.wikipedia.org/wiki/Wikipedia:Hauptseite"

# Fetch the content of the web page
response = requests.get(url)
html_content = response.content

# Parse the HTML content
soup = BeautifulSoup(html_content, "html.parser")

# Extract the text from the HTML
text = soup.get_text()

# Process the text using the German language model
doc = nlp(text)

# Tokenize the text into sentences
sentences = [sent.text.strip() for sent in doc.sents]

# Print the first sentence
print("A Sentence from the webpage:", sentences[20])

print(20 * '-' + 'End Q2' + 20 * '-')
# =================================================================
# Class_Ex3:
# Tokenize a sentence using sapaCy.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')

import spacy
nlp = spacy.load("en_core_web_sm")

sentence = "My doggo keeps barking at nothing!"

# Process the text
doc = nlp(sentence)

# Iterate over each token in the document
for token in doc:
    # Print the tokenized word
    print(token.text)

print(20 * '-' + 'End Q3' + 20 * '-')
# =================================================================
# Class_Ex4:
# Use the following sentence as a sample text. and Answer the following questions.
# "In 2020, more than 15% of people in World got sick from a pandemic ( www.google.com ). Now it is less than 1% are. Reference ( www.yahoo.com )"
# 1- Check if there is a token resemble a number.
# 2- Find a percentage in the text.
# 3- How many url is in the text.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

given_text = "In 2020, more than 15% of people in World got sick from a pandemic ( www.google.com ). Now it is less than 1% are. Reference ( www.yahoo.com )"

# Process the text
doc = nlp(given_text)

# Initialize an empty list to store digit tokens and percentage tokens
number_tokens = []
percentage_tokens = []

# Initialize a variable to store digits preceding the percentage sign
current_digits = ""

# Iterate over each token in the document
for token in doc:
    # Check if the token is a digit
    if token.text.isdigit():
        number_tokens.append(token.text)
        current_digits += token.text


    # Check if the token represents a percentage
    elif token.text == "%" and current_digits:
        percentage_tokens.append(current_digits + "%")
        current_digits = ""  # Reset current_digits

    # If the token is not a digit or percentage sign, reset current_digits
    else:
        current_digits = ""

# Print tokens resembling numbers
print("Digit tokens in the text:", number_tokens)

# Print percentages found in the text
print("Percentage tokens in the text:", percentage_tokens)

print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================
# Class_Ex5:
# Load small web english model into spaCy.
# USe the following text as a sample text. Answer the following questions
# "It is shown that: Google was not the first search engine in U.S. tec company. The value of google is 100 billion dollar"
# 1- Get the token text, part-of-speech tag and dependency label.
# 2- Print them in a tabular format.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')

import pandas as pd
nlp = spacy.load("en_core_web_sm")

given_text = "It is shown that: Google was not the first search engine in U.S. tec company. The value of google is 100 billion dollar"

doc = nlp(given_text)

# Initialize lists to store token information
tokens = []
part_of_speech = []
pos_tag = []
dependency = []

# Iterate over each token in the processed document
for token in doc:
    # Append token information to lists
    tokens.append(token.text)
    part_of_speech.append(token.pos_)
    pos_tag.append(token.tag_)
    dependency.append(token.dep_)

# Create a DataFrame
df = pd.DataFrame({
    "Token": tokens,
    "PartOfSpeech": part_of_speech,
    "POSTag": pos_tag,
    "Dependency": dependency
})

# Display the DataFrame
print(df)

print(20 * '-' + 'End Q5' + 20 * '-')

# =================================================================
# Class_Ex6:
# Use Ex 5 sample text and find all the entities in the text.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')

given_text = "It is shown that: Google was not the first search engine in U.S. tec company. The value of google is 100 billion dollar"

doc = nlp(given_text)

# Print the recognized entities
for entity in doc.ents:
    print(entity.text, entity.label_)

print(20 * '-' + 'End Q6' + 20 * '-')
# =================================================================
# Class_Ex7:
# Use SpaCy and find adjectives plus one or 2 nouns.
# Use the following Sample text.
# Features of the iphone applications include a beautiful design, smart search, automatic labels and optional voice responses.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Sample text
sample_text = "Features of the iPhone applications include a beautiful design, smart search, automatic labels and optional voice responses."

# Process the text
doc = nlp(sample_text)

# Initialize a list to store adjective-noun pairs
adj_noun_pairs = []

# Iterate over each token in the processed document
for i, token in enumerate(doc):
    # If the token is an adjective
    if token.pos_ == "ADJ":
        # Initialize a list to store associated nouns
        nouns = []
        # Iterate over the next two tokens to find nouns
        for j in range(i + 1, min(i + 3, len(doc))):
            # If the token is a noun, add it to the list of nouns
            if doc[j].pos_ == "NOUN":
                nouns.append(doc[j].text)
        # If there are one or two nouns associated with the adjective
        if nouns:
            # Append the adjective-noun pair to the list
            adj_noun_pairs.append((token.text, nouns))

# Print the adjective-noun pairs
for pair in adj_noun_pairs:
    print(f"Adjective: {pair[0]}, Nouns: {', '.join(pair[1])}")

print(20 * '-' + 'End Q7' + 20 * '-')
# =================================================================
# Class_Ex8:
# Use spacy lookup table and find the hash id for a cat
# Text : I have a cat.
# Next use the id and find the strings.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q8' + 20 * '-')

# Define the text
text = "I have a cat."

# Process the text
doc = nlp(text)

# Get the hash ID for the word "cat"
cat_hash = nlp.vocab.strings["cat"]
print("Hash ID for 'cat':", cat_hash)

# Retrieve the string associated with the hash ID
cat_string = nlp.vocab.strings[cat_hash]
print("String associated with the hash ID:", cat_string)

print(20 * '-' + 'End Q8' + 20 * '-')
# =================================================================
# Class_Ex9:
# Create a Doc object for the following sentence
# Spacy is a nice toolkit.
# Use the methods like text, token,... on the Doc and check the functionality.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q9' + 20 * '-')

text = "Spacy is a nice toolkit."

doc = nlp(text)

for token in doc:
    # Token text
    token_text = token.text
    # Dependency label
    dependency_label = token.dep_
    # Head text
    head_text = token.head.text
    # Head POS
    head_pos = token.head.pos_
    # Children tokens
    children_tokens = [child.text for child in token.children]

    # Print the information
    print(f"Token Text: {token_text}")
    print(f"Dependency Label: {dependency_label}")
    print(f"Head Text: {head_text}")
    print(f"Head POS: {head_pos}")
    print(f"Children: {children_tokens}")
    print("=" * 50)

print(20 * '-' + 'End Q9' + 20 * '-')
# =================================================================
# Class_Ex10:
# Use spacy and process the following text.
# Newyork looks like a nice city.
# Find which token is proper noun and which one is a verb.
#

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q10' + 20 * '-')

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define the text
text = "New York looks like a nice city."

# Process the text
doc = nlp(text)

# Iterate through the tokens
for token in doc:
    # Check if the token's POS is either a noun (NN) or a verb (VERB)
    if token.pos_ in ["PROPN", "VERB"]:
        print("Token:", token.text, "POS:", token.pos_)

print(20 * '-' + 'End Q10' + 20 * '-')
# =================================================================
# Class_Ex11:
# Read the list of countries in a json format.
# Use the following text as  sample text.
# Czech Republic may help Slovakia protect its airspace
# Use statistical method and rule based method to find the countries.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q11' + 20 * '-')

import json

# Load the list of countries from the JSON file
with open("countries.json", "r") as file:
    countries = json.load(file)

# Sample text
sample_text = "Czech Republic may help Slovakia protect its airspace"

# Process the text
doc = nlp(sample_text)

# Initialize a list to store detected countries
detected_countries = []

# Iterate over the named entities in the document
for ent in doc.ents:
    # Check if the entity text is in the set of countries
    if ent.text in countries:
        detected_countries.append(ent.text)

# Print the detected countries
print("Detected countries:", detected_countries)

print(20 * '-' + 'End Q11' + 20 * '-')
# =================================================================
# Class_Ex12:
# Use spacy attributions and answer the following questions.
# Define the getter function that takes a token and returns its reversed text.
# Add the Token property extension "reversed" with the getter function
# Process the text and print the results.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q12' + 20 * '-')

# Define the getter function
def get_reversed_text(token):
    return token.text[::-1]

# Add the Token property extension "reversed" with the getter function
spacy.tokens.Token.set_extension("reversed", getter=get_reversed_text)

# Process the text
text = "Hello world! This is a test."
doc = nlp(text)

# Print the reversed text for each token in the document
for token in doc:
    print("Token:", token.text, "Reversed:", token._.reversed)


print(20 * '-' + 'End Q12' + 20 * '-')
# =================================================================
# Class_Ex13:
# Read the tweets json file.
# Process the texts and print the entities
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q13' + 20 * '-')

# Load the list of countries from the JSON file
with open("tweets.json", "r") as file:
    tweets = json.load(file)

# Process each tweet and print the recognized entities
for tweet in tweets:
    doc = nlp(tweet)
    print("Full tweet:", tweet)
    for entity in doc.ents:
        print("Entity text:", entity.text, "Entity label:", entity.label_)
    print("-" * 50)

print(20 * '-' + 'End Q13' + 20 * '-')
# =================================================================
# Class_Ex14:
# Use just spacy tokenization. for the following text
# "Burger King is an American fast food restaurant chain"
# make sure other pipes are disabled and not used.
# Disable parser and tagger and process the text. Print the tokens
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q14' + 20 * '-')

# Load the English language model with parser and tagger disabled
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])

# Text to be tokenized
text = "Burger King is an American fast food restaurant chain"

# Tokenize the text using spaCy's tokenizer
tokens = nlp.tokenizer(text)

# Print the tokens
print("Tokens:")
for token in tokens:
    print(token.text)

print(20 * '-' + 'End Q14' + 20 * '-')

# =================================================================
