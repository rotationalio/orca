from src.reader.reader import Reader
from src.parser.html import HTMLParser

import pytest

class InvalidReader(Reader):
    def __init__(self):
        self.source = None

    def read(self):
        raise IOError("read error")

class StringReader(Reader):
    def __init__(self, strings):
        self.strings = strings

    def read(self):
        for s in self.strings:
            yield s

class TestHTMLParser():
    """
    Tests for the HTMLParser class.
    """

    @pytest.mark.parametrize(
        "reader",
        [
            Reader(None),
            InvalidReader(),
        ]
    )
    def test_parse_invalid(self, reader):
        """
        Test that the parser handles underlying reader errors.
        """
        parser = HTMLParser(reader)
        with pytest.raises(ValueError):
            list(parser.parse())

    @pytest.mark.parametrize(
        "html, expected",
        [
            ([], [""]),
            (["<h1>Heading Text</h1>"], ["Heading Text"]),
            ([
                "<h2>Heading</h2><p>Paragraph</p>",
                "<p>Paragraph </p><li>List</li>"
              ],
              ["Heading Paragraph", "Paragraph List"]),
            (["<h3>Heading</h3><a>Link</a><p>Paragraph</p>"], ["Heading Paragraph"]),
        ]
    )
    def test_parse(self, html, expected):
        """
        Test that the parser parses the text from the provided HTML.
        """
        parser = HTMLParser(StringReader(html))
        texts = list(parser.parse())
        assert texts == expected