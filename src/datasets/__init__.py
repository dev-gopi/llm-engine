"""Project data pipeline package.

This module intentionally makes ``datasets`` a regular package.  The project
also supports the third-party Hugging Face distribution with the same import
name; a regular package ensures the repository's configured ``src`` path takes
precedence for imports such as ``datasets.loader``.
"""
