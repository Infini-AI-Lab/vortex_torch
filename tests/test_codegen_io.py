from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from vortex_torch._codegen_io import write_generated_module


class CodegenIOTest(unittest.TestCase):
    def test_generated_modules_are_namespaced_and_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            common = {
                "cache_dir": directory,
                "flow_name": "same/flow",
            }

            indexer = write_generated_module(
                **common, namespace="indexer", source="VALUE = 1\n"
            )
            cache = write_generated_module(
                **common, namespace="cache", source="VALUE = 1\n"
            )
            variant = write_generated_module(
                **common, namespace="indexer", source="VALUE = 2\n"
            )

            self.assertEqual(len({indexer, cache, variant}), 3)
            self.assertEqual(Path(indexer).read_text(), "VALUE = 1\n")
            self.assertEqual(Path(cache).read_text(), "VALUE = 1\n")
            self.assertEqual(Path(variant).read_text(), "VALUE = 2\n")

    def test_concurrent_writers_only_publish_complete_files(self):
        source = "VALUE = 'complete'\n" * 4096

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write_once(_):
                return write_generated_module(
                    cache_dir=directory,
                    flow_name="shared",
                    namespace="indexer",
                    source=source,
                )

            with ThreadPoolExecutor(max_workers=16) as executor:
                paths = list(executor.map(write_once, range(64)))

            self.assertEqual(len(set(paths)), 1)
            self.assertEqual(Path(paths[0]).read_text(), source)
            self.assertFalse(list(root.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
