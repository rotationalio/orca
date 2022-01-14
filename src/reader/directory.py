from src.reader.reader import Reader

import os

class DirectoryReader(Reader):
    """
    DirectoryReader implements reading files from a directory into a stream.
    """
    def __init__(self, directory, encoding="ISO-8859-1"):
        self.directory = directory
        self.encoding = encoding

    def read(self):
        """
        Read files from the directory into a stream.
        """
        for f in os.listdir(self.directory):
            with open(os.path.join(self.directory, f), 'r', encoding=self.encoding) as f:
                yield f.read()