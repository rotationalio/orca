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
        Parse the next HTML file from the reader and return the entire text.
        """
        try:
            html = self.reader.read()
        except Exception as e:
            raise ValueError("Error retrieving file from reader: " + str(e))
        if html is None:
            return None
        return self._parse_text(html)

    def _parse_text(self, html):
        """
        Parse the text from the provided HTML.
        """
        soup = BeautifulSoup(html, 'html.parser')
        text = ""
        for tag in soup.find_all(TAGS):
            text += tag.text + " "
        return text.strip()