class Reader():
    """
    Reader is a generic wrapper for reading data from a source that supports a read() method.
    """
    def __init__(self, source):
        self.source = source

    def read(self):
        """
        Read data from the source.
        """
        try:
            data = self.source.read()
        except Exception as e:
            raise ValueError("Error reading from source: " + str(e))
        return data