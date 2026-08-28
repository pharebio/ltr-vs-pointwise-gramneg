"""Put the repo root on sys.path so the flat modules import cleanly under pytest."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
