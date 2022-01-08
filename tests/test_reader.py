from src.reader.reader import Reader

import pytest

class ListReader():
    def __init__(self, data):
        self.data = data

    def read(self):
        if len(self.data) == 0:
            return None
        return self.data.pop(0)

class TestReader():
    """
    Tests for the Reader class.
    """

    @pytest.mark.parametrize(
        "reader",
        [
            Reader(None),
            Reader(ListReader(None)),
        ],
    )
    def test_read_invalid(self, reader):
        """
        Test that the Reader catches underlying read() errors.
        """
        with pytest.raises(ValueError):
            reader.read()

    @pytest.mark.parametrize(
        "reader, expected",
        [
            (Reader(ListReader(["Hello"])), "Hello"),
            (Reader(ListReader(["Hello", "World"])), "HelloWorld"),
        ],
    )
    def test_read(self, reader, expected):
        """
        Test that the Reader correctly reads data from the source.
        """
        data = ""
        while True:
            elem = reader.read()
            if elem is None:
                break
            data += elem
        assert data == expected