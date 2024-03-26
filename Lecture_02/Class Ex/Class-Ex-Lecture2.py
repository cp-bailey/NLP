# Import resources
import nltk
import matplotlib.pyplot as plt

# =================================================================
# Class_Ex1:
# Use NLTK Book fnd which the related Sense and Sensibility.
# Produce a dispersion plot of the four main protagonists in Sense and Sensibility:
# Elinor, Marianne, Edward, and Willoughby. What can you observe about the different
# roles played by the males and females in this novel? Can you identify the couples?
# Explain the result of plot in a couple of sentences.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

# Find Sense & Sensibility file ID
print(nltk.corpus.gutenberg.fileids())

# Load Sense and Sensibility text
s_and_s = nltk.corpus.gutenberg.words('austen-sense.txt')

# Create a Text object
text = nltk.Text(s_and_s)

# Define the main protagonists
protagonists = ['Elinor', 'Marianne', 'Edward', 'Willoughby']

# Generate a dispersion plot for the protagonists
dispersion_plot = text.dispersion_plot(protagonists)
plt.show()

'''
 The males are more prominent throughout the story.
 It is difficult to identify the couples from the dispersion plot alone.
'''

print(20 * '-' + 'End Q1' + 20 * '-')

# =================================================================
# Class_Ex2:
# What is the difference between the following two lines of code? Explain in details why?
# Make up and example base don your explanation.
# Which one will give a larger value? Will this be the case for other texts?
# 1- sorted(set(w.lower() for w in text1))
# 2- sorted(w.lower() for w in set(text1))
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

# create test text1
text1 = "Hello, this is some text."

# perform two different ways to process text1
print(sorted(set(w.lower() for w in text1)))
print(sorted(w.lower() for w in set(text1)))

'''
In Python, a set is a mutable, unordered collection of unique (non-duplicate) elements.
Therefore, line 1 produces a sorted list from the set of letters in text1 after the
have been lowercased.
Line 2 produces a sorted list that includes duplicates of any capitalized letters 
from text1 because the lowercase action occurs upon the set of text1.
'''


print(20 * '-' + 'End Q2' + 20 * '-')
# =================================================================
# Class_Ex3:
# Find all the four-letter words in the Chat Corpus (text5).
# With the help of a frequency distribution (FreqDist), show these words in decreasing order of frequency.
#
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')

# import text5
from nltk.book import text5

# Get all words from text5 and convert them to lowercase
words = [word.lower() for word in text5]

# Filter four-letter words
four_letter_words = [word for word in words if len(word) == 4]

# Create frequency distribution
fdist = nltk.FreqDist(four_letter_words)

# Print frequency distribution in decreasing order
for word, freq in fdist.most_common():
    print(word, freq)

print(20 * '-' + 'End Q3' + 20 * '-')
# =================================================================
# Class_Ex4:
# Write expressions for finding all words in text6 that meet the conditions listed below.
# The result should be in the form of a list of words: ['word1', 'word2', ...].
# a. Ending in ise
# b. Containing the letter z
# c. Containing the sequence of letters pt
# d. Having all lowercase letters except for an initial capital (i.e., titlecase)
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

# import text6
from nltk.book import text6

# Get list of words in text 6 that end in ise
list_a = [word for word in text6 if word.endswith('ise')]
print(list_a)

# Get list of words in text 6 that contain z
list_b = [word for word in text6 if 'z' in word]
print(list_b)

# Get list of words in text 6 that contains sequence pt
list_c = [word for word in text6 if 'pt' in word]
print(list_c)

# Get list of words in text 6 that an initial capital letter
list_d = [word for word in text6 if word.istitle()]
print(list_d)

print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================
# Class_Ex5:
#  Read in the texts of the State of the Union addresses, using the state_union corpus reader.
#  Count occurrences of men, women, and people in each document.
#  What has happened to the usage of these words over time?
# Since there would be a lot of document use every couple of years.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')

from nltk.corpus import state_union

# Get the list of State of the Union addresses
speech_years = state_union.fileids()

# Initialize dictionaries to store word counts for each year
men_counts = {}
women_counts = {}
people_counts = {}

# Iterate through each speech
for speech_year in speech_years:
    # Tokenize the speech text
    words = state_union.words(speech_year)
    # Count occurrences of the words
    men_counts[speech_year] = words.count('men')
    women_counts[speech_year] = words.count('women')
    people_counts[speech_year] = words.count('people')

# Extract years from file IDs
years = [int(year[:4]) for year in speech_years]

# Plot the usage of these words over time
plt.plot(list(years[::2]), list(men_counts.values())[::2], label='Men')
plt.plot(list(years[::2]), list(women_counts.values())[::2], label='Women')
plt.plot(list(years[::2]), list(people_counts.values())[::2], label='People')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.title('Usage of Men, Women, and People over Time')
plt.legend()
plt.show()

'''
The usage of the word "men" has declined and the usage of the word
"people" has increased."
'''

print(20 * '-' + 'End Q5' + 20 * '-')

# =================================================================
# Class_Ex6:
# The CMU Pronouncing Dictionary contains multiple pronunciations for certain words.
# How many distinct words does it contain? What fraction of words in this dictionary have more than one possible pronunciation?
#
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')

# Load the CMU Pronouncing Dictionary
prondict = nltk.corpus.cmudict.dict()

# Count distinct words and words with multiple pronunciations
total_words = len(prondict)
words_with_multiple_pronunciations = sum(len(pronunciations) > 1 for pronunciations in prondict.values())

# Calculate the fraction of words with multiple pronunciations
fraction = words_with_multiple_pronunciations / total_words

print("Total distinct words:", total_words)
print("Words with multiple pronunciations:", words_with_multiple_pronunciations)
print("Fraction of words with multiple pronunciations:", fraction)

print(20 * '-' + 'End Q6' + 20 * '-')
# =================================================================
# Class_Ex7:
# What percentage of noun synsets have no hyponyms?
# You can get all noun synsets using wn.all_synsets('n')
#
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

# import wordnet
from nltk.corpus import wordnet as wn

# all noun synsets
all_noun_synsets = list(wn.all_synsets('n'))

# nouns synsets with no hyponym
noun_synsets_no_hyponyms = [synset for synset in all_noun_synsets if not synset.hyponyms()]

# fraction of noun synsets with no hyponyms
fraction = len(noun_synsets_no_hyponyms) / len(all_noun_synsets)

print("Total number of noun synsets:", len(all_noun_synsets))
print("Total number of noun synsets with no hyponym:", len(noun_synsets_no_hyponyms))
print("Fraction of noun synsets with no hyponym:", fraction)

print(20 * '-' + 'End Q7' + 20 * '-')
# =================================================================
# Class_Ex8:
# Write a program to find all words that occur at least three times in the Brown Corpus.
# USe at least 2 different method.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q8' + 20 * '-')

from nltk.corpus import brown
from collections import defaultdict

# Initialize a dictionary to store word frequencies
word_freq = defaultdict(int)

# Initialize list of brown words
brown_words = brown.words()

# Count the occurrences of each word
word_counts = {}
for word in brown_words:
    word_counts[word] = word_counts.get(word, 0) + 1


# Find words that occur at least three times
words_at_least_three1 = [word for word, count in word_counts.items() if count >= 3]

print("Method 1:")
print("Words that occur at least three times:", words_at_least_three1[:10])

# Calculate the frequency distribution of words
word_freq = nltk.FreqDist(brown_words)

# Find words that occur at least three times
words_at_least_three2 = [word for word, freq in word_freq.items() if freq >= 3]

print("Method 2:")
print("Words that occur at least three times:", words_at_least_three2[:10])

print(20 * '-' + 'End Q8' + 20 * '-')
# =================================================================
# Class_Ex9:
# Write a function that finds the 50 most frequently occurring words of a text that are not stopwords.
# Test it on Brown corpus (humor), Gutenberg (whitman-leaves.txt).
# Did you find any strange word in the list? If yes investigate the cause?
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q9' + 20 * '-')

from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
import string


def most_common_non_stopwords(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Get English stopwords
    english_stopwords = set(stopwords.words('english'))

    # Remove stopwords, punctuation, dashes, apostrophes etc.
    non_stopwords = [word.lower() for word in tokens if word.lower() not in english_stopwords
                     and word.lower() not in string.punctuation
                     and '--' not in word
                     and "'" not in word
                     and "``" not in word]

    # Calculate frequency distribution of non-stopwords
    freq_dist = nltk.FreqDist(non_stopwords)

    # Get the 50 most frequent non-stopwords
    return freq_dist.most_common(50)


# Test on Brown corpus (humor)
brown_humor_words = ' '.join(brown.words(categories='humor'))
result_brown = most_common_non_stopwords(brown_humor_words)
print("50 most frequently occurring non-stopwords in Brown (humor) corpus:")
print(result_brown)

# Test on Gutenberg (whitman-leaves.txt)
whitman_leaves_words = nltk.corpus.gutenberg.raw('whitman-leaves.txt')
result_whitman = most_common_non_stopwords(whitman_leaves_words)
print("\n50 most frequently occurring non-stopwords in Gutenberg (whitman-leaves.txt):")
print(result_whitman)

'''
Some punctuation and other odd tokens needed to be removed.
'''

print(20 * '-' + 'End Q9' + 20 * '-')
# =================================================================
# Class_Ex10:
# Write a program to create a table of word frequencies by genre, like the one given in 1 for modals.
# Choose your own words and try to find words whose presence (or absence) is typical of a genre. Discuss your findings.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q10' + 20 * '-')

# Define the words to analyze
interesting_words = ['love', 'hate', 'war', 'peace', 'crime', 'freedom', 'justice', 'money', 'power', 'news']

# Create a table to store word frequencies by genre
genre_word_freq = {}

# Iterate over each genre in the Brown Corpus
for genre in nltk.corpus.brown.categories():
    # Get the words from the genre
    genre_words = nltk.corpus.brown.words(categories=genre)
    # Calculate the frequency distribution of interesting words in this genre
    freq_dist = nltk.FreqDist(w.lower() for w in genre_words if w.lower() in interesting_words)
    # Store the frequency distribution for this genre
    genre_word_freq[genre] = freq_dist

# Print the table
print("{:12}".format(""), end="")
for word in interesting_words:
    print("{:10}".format(word), end="")
print()

for genre, freq_dist in genre_word_freq.items():
    print("{:12}".format(genre), end="")
    for word in interesting_words:
        print("{:10}".format(freq_dist[word]), end="")
    print()

print(20 * '-' + 'End Q10' + 20 * '-')
# =================================================================
# Class_Ex11:
#  Write a utility function that takes a URL as its argument, and returns the contents of the URL,
#  with all HTML markup removed. Use from urllib import request and
#  then request.urlopen('http://nltk.org/').read().decode('utf8') to access the contents of the URL.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q11' + 20 * '-')

from urllib import request
from bs4 import BeautifulSoup


def get_text_from_url(url):
    try:
        # Fetch HTML content from the URL
        html = request.urlopen(url).read().decode('utf8')

        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Extract text content from the parsed HTML
        text = soup.get_text()

        # Remove extra whitespace and return the cleaned text
        return ' '.join(text.split())
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Example usage:
url = 'http://nltk.org/'
text_content = get_text_from_url(url)
if text_content:
    print(text_content[:100])  # Print the first 100 characters of the text content

print(20 * '-' + 'End Q11' + 20 * '-')
# =================================================================
# Class_Ex12:
# Read in some text from a corpus, tokenize it, and print the list of all
# wh-word types that occur. (wh-words in English are used in questions,
# relative clauses and exclamations: who, which, what, and so on.)
# Print them in order. Are any words duplicated in this list,
# because of the presence of case distinctions or punctuation?
# Note Use: Gutenberg('bryant-stories.txt')
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q12' + 20 * '-')

# Load corpus
text = nltk.corpus.gutenberg.raw('bryant-stories.txt')

# Tokenize
tokens = word_tokenize(text)

# Get wh-words
wh_words = [word.lower() for word in tokens if '_' not in word and word.lower().startswith('wh')]
print(sorted(set(wh_words)))

print(20 * '-' + 'End Q12' + 20 * '-')
# =================================================================
# Class_Ex13:
# Write code to access a  webpage and extract some text from it.
# For example, access a weather site and extract  a feels like temprature..
# Note use the following site https://darksky.net/forecast/40.7127,-74.0059/us12/en
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q13' + 20 * '-')

'''
Above link is unsafe so I used a different weather link: https://www.accuweather.com/en/us/washington/20006/weather-forecast/327659
'''

def get_text_from_url(url):
    try:
        # Fetch HTML content from the URL
        html = request.urlopen(url).read().decode('utf8')

        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Extract text content from the parsed HTML
        text = soup.get_text()

        # Remove extra whitespace and return the cleaned text
        return ' '.join(text.split())
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Usage:
url = 'https://www.accuweather.com/en/us/washington/20006/weather-forecast/327659'
text_content = get_text_from_url(url)
if text_content:
    print(text_content[:100])  # Print the first 100 characters of the text content

print(20 * '-' + 'End Q13' + 20 * '-')
# =================================================================
# Class_Ex14:
# Use the brown tagged sentences corpus news.
# make a test and train sentences and then  use bi-gram tagger to train it.
# Then evaluate the trained model.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q14' + 20 * '-')

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
bigram_tagger = nltk.BigramTagger(brown_tagged_sents)
print(bigram_tagger.accuracy(brown_tagged_sents))

size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
bigram_tagger = nltk.BigramTagger(train_sents)
print(bigram_tagger.accuracy(test_sents))

print(20 * '-' + 'End Q14' + 20 * '-')

# =================================================================
# Class_Ex15:
# Use sorted() and set() to get a sorted list of tags used in the Brown corpus, removing duplicates.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q15' + 20 * '-')

from nltk.corpus import brown

brown_tags = brown.tagged_words()

# Extracting unique tags and sorting them
unique_tags = [t for _, t in brown_tags]
print(sorted(set(unique_tags)))

print(20 * '-' + 'End Q15' + 20 * '-')

# =================================================================
# Class_Ex16:
# Write programs to process the Brown Corpus and find answers to the following questions:
# 1- Which nouns are more common in their plural form, rather than their singular form? (Only consider regular plurals, formed with the -s suffix.)
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q16' + 20 * '-')

import nltk
from nltk.corpus import brown
from collections import defaultdict

# Tokenize and tag the Brown Corpus
brown_tagged_words = brown.tagged_words()

# Initialize dictionaries to store counts of singular and plural nouns
singular_counts = defaultdict(int)
plural_counts = defaultdict(int)

# Count occurrences of singular and plural nouns
for word, tag in brown_tagged_words:
    if tag == 'NNS':  # Plural noun
        plural_counts[word.lower()] += 1
    elif tag == 'NN':  # Singular noun
        singular_counts[word.lower()] += 1

# Find nouns where the plural form is more common than the singular form
more_common_in_plural = []
for noun in singular_counts.keys():
    if plural_counts[noun + 's'] > singular_counts[noun]:
        more_common_in_plural.append(noun)

print("Nouns more common in plural form:", more_common_in_plural)

print(20 * '-' + 'End Q16' + 20 * '-')
