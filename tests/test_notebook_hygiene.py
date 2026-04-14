import json
import tempfile
import unittest
from pathlib import Path

from notebook_hygiene import format_issues, inspect_repo_notebooks


class NotebookHygieneTests(unittest.TestCase):
    def test_inspect_repo_notebooks_flags_outputs_and_local_paths(self):
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 7,
                    "metadata": {},
                    "outputs": [{"output_type": "stream", "name": "stdout", "text": ["hello\n"]}],
                    "source": ["DATA_PATH = '/Users/testuser/OneDrive/secure.csv'\n"],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dirty.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")

            issues = inspect_repo_notebooks(Path(temp_dir))

        kinds = {issue.kind for issue in issues}
        self.assertIn("outputs", kinds)
        self.assertIn("execution_count", kinds)
        self.assertIn("local_path", kinds)
        self.assertIn("dirty.ipynb", format_issues(issues))

    def test_inspect_repo_notebooks_accepts_clean_notebook(self):
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["print('clean')\n"],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clean.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")

            issues = inspect_repo_notebooks(Path(temp_dir))

        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
