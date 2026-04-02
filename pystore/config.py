#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# PyStore: Flat-file datastore for timeseries data
# https://github.com/ranaroussi/pystore
#
# Copyright 2018-2020 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyStore configuration module."""

import os as os
from typing import Optional, TYPE_CHECKING

from .utils import Path

if TYPE_CHECKING:
    from dask.distributed import Client

DEFAULT_PATH: str = os.environ.get("PYSTORE_PATH", str(Path.home() / "pystore"))
DEFAULT_PARTITION_SIZE: float = 99e+6  # ~99MB
PARTITION_SIZE: float = 99e+6  # ~99MB

# dask distributed
_SCHEDULER: Optional[str] = None
_CLIENT: Optional["Client"] = None
