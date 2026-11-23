"""Operational shortcut for the package's GPU setup command."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from invoice_ocr.cli import entrypoint

if __name__ == '__main__':
    entrypoint(['gpu', 'setup', *sys.argv[1:]])
