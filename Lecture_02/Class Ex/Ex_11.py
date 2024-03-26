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