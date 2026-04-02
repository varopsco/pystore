#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Pytest configuration for local source import resolution.
"""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
repo_root_str = str(REPO_ROOT)
sys.path = [p for p in sys.path if p != repo_root_str]
sys.path.insert(0, repo_root_str)
