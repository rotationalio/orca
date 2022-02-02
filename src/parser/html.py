from src.reader.reader import Reader
from bs4 import BeautifulSoup

TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']

class HTMLParser():
    """
    HTMLParser parses text content from HTML files using the provided reader.
    """
    def __init__(self, reader):
        if not isinstance(reader, Reader):
            raise ValueError("HTMLParser requires a Reader object.")
        self.reader = reader

    def parse(self):
        """
        Return a generator that parses text content from the reader.
        """
        for html in self.reader.read():
            try:
                if html is None:
                    return
                yield self._parse_text(html)
            except Exception as e:
                print("Error parsing HTML: " + str(e))

    def _parse_text(self, html):
        """
        Parse the text from the provided HTML.
        TODO: We can be smarter about which text content to return (e.g., for news
        articles, the most information-rich content is near the top of the article).
        """
        soup = BeautifulSoup(html, 'html.parser')
        text = ""
        for tag in soup.find_all(TAGS):
            text += tag.text + " "
        return text.strip()