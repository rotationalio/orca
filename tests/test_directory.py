from src.reader.directory import DirectoryReader

import os
import pytest

class TestDirectoryReader():
    """
    Tests for the DirectoryReader class.
    """

    @pytest.mark.parametrize(
        "files",
        [
            ({}),
            ({
                "foo.txt": "foo",
                "bar.txt": "bar",
            }),
        ],
    )
    def test_read(self, tmpdir, files):
        """
        Test that the DirectoryReader can stream files from a directory.
        """
        for name, data in files.items():
            with open(os.path.join(tmpdir, name), 'w') as f:
                f.write(data)

        reader = DirectoryReader(tmpdir)
        assert list(reader.read()) == list(files.values())