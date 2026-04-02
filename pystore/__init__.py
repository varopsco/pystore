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

"""PyStore - Fast data store for Pandas timeseries data.

This module provides a simple datastore for Pandas dataframes, designed
with storing timeseries data in mind. It's built on top of Pandas, Numpy,
Dask, and Parquet (via Fastparquet).

Example:
    >>> import pystore
    >>> pystore.set_path("~/pystore")
    >>> store = pystore.store('mydatastore')
    >>> collection = store.collection('NASDAQ')
    >>> collection.write('AAPL', df, metadata={'source': 'Quandl'})
"""

# temp fix for fastparquet 0.3.2 and numba 0.45.1
try:
    import numba as _
except ImportError:
    pass

from .store import Store, store
from .utils import (
    read_csv, set_path, get_path,
    set_client, get_client,
    set_partition_size, get_partition_size,
    list_stores, delete_store, delete_stores)

__version__: str = "0.1.22"
__author__: str = "Ran Aroussi"

__all__ = ["Store", "store", "read_csv", "get_path", "set_path",
           "set_client", "get_client",
           "set_partition_size", "get_partition_size",
           "list_stores", "delete_store", "delete_stores"]
