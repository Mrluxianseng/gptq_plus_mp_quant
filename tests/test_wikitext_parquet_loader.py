import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from utils import data_utils


class WikiTextParquetLoaderTest(unittest.TestCase):
    def test_reads_local_parquet_split(self):
        with tempfile.TemporaryDirectory() as directory:
            original_cwd = os.getcwd()
            try:
                os.chdir(directory)
                parquet = (
                    Path(directory)
                    / "datasets"
                    / "wikitext"
                    / "wikitext-2-raw-v1"
                    / "test-00000-of-00001.parquet"
                )
                parquet.parent.mkdir(parents=True)
                parquet.touch()
                load_dataset = Mock(return_value={"text": ["raw WikiText text"]})
                with patch.object(data_utils, "load_dataset", load_dataset):
                    result = data_utils._get_wikitext2("test")
            finally:
                os.chdir(original_cwd)

        self.assertEqual(result, ["raw WikiText text"])
        load_dataset.assert_called_once_with(
            "parquet",
            data_files={"test": [str(parquet.resolve())]},
            split="test",
        )

    def test_keeps_legacy_builder_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            original_cwd = os.getcwd()
            try:
                os.chdir(directory)
                load_dataset = Mock(return_value={"text": ["legacy WikiText text"]})
                with patch.object(data_utils, "load_dataset", load_dataset):
                    result = data_utils._get_wikitext2("validation")
            finally:
                os.chdir(original_cwd)

        self.assertEqual(result, ["legacy WikiText text"])
        load_dataset.assert_called_once_with(
            "./datasets/wikitext",
            "wikitext-2-raw-v1",
            split="validation",
            trust_remote_code=True,
        )


if __name__ == "__main__":
    unittest.main()
