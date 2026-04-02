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

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
import json
import shutil
import pandas as pd
import numpy as np
from dask import dataframe as dd
from dask.distributed import Client


try:
    from pathlib import Path
    Path().expanduser()
except (ImportError, AttributeError):
    from pathlib2 import Path  # type: ignore

from . import config


# Configure logger for pystore - use NullHandler by default (best practice for libraries)
# Users can call configure_logging() to set up default logging behavior
logger = logging.getLogger('pystore')
logger.addHandler(logging.NullHandler())

METADATA_FILENAME = "metadata.json"
LEGACY_METADATA_FILENAME = "pystore_metadata.json"


def configure_logging(level: int = logging.INFO, format_string: Optional[str] = None) -> None:
    """Configure logging for pystore.
    
    This function can be called at application startup to configure logging
    instead of configuring at module import time.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string for log messages
                     (default: '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Remove any existing handlers (except NullHandler)
    for handler in logger.handlers[:]:
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
    
    # Add StreamHandler with the specified level and format
    handler = logging.StreamHandler()
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


def read_csv(urlpath: str, *args: Any, **kwargs: Any) -> dd.DataFrame:
    """Read CSV file into a Dask DataFrame.

    Args:
        urlpath: Path or URL to the CSV file
        *args: Additional positional arguments passed to dd.read_csv
        **kwargs: Additional keyword arguments passed to dd.read_csv
            Can include 'index_col' and 'index_name' for index handling

    Returns:
        Dask DataFrame with optional index configuration
    """
    def rename_dask_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
        df.index.name = name
        return df

    index_col: Optional[Union[str, List[str]]] = None
    index_name: Optional[str] = None

    if "index" in kwargs:
        del kwargs["index"]
    if "index_col" in kwargs:
        index_col = kwargs["index_col"]
        if isinstance(index_col, list):
            index_col = index_col[0]
        del kwargs["index_col"]
    if "index_name" in kwargs:
        index_name = kwargs["index_name"]
        del kwargs["index_name"]

    df = dd.read_csv(urlpath, *args, **kwargs)

    if index_col is not None:
        df = df.set_index(index_col)

    if index_name is not None:
        df = df.map_partitions(rename_dask_index, index_name)

    return df


def datetime_to_int64(df: Union[pd.DataFrame, dd.DataFrame]) -> Union[pd.DataFrame, dd.DataFrame]:
    """Convert datetime index to epoch int.

    Allows for cross language/platform portability.

    Args:
        df: DataFrame with datetime index

    Returns:
        DataFrame with datetime index converted to int64
    """
    # Check if the index is a DatetimeIndex with nanoseconds
    if isinstance(df.index, pd.DatetimeIndex):
        nanoseconds = df.index.nanosecond
        # Use .any() method on numpy array instead of built-in any()
        if hasattr(nanoseconds, 'any'):
            if nanoseconds.any():
                df.index = df.index.astype(np.int64)
        elif any(nanoseconds):  # Fallback for non-numpy arrays
            df.index = df.index.astype(np.int64)

    return df


def subdirs(d: Union[str, Path]) -> List[str]:
    """Get list of subdirectory names.

    Use this to construct paths for future storage support.

    Args:
        d: Directory path

    Returns:
        List of subdirectory names (excluding _snapshots)
    """
    return [o.parts[-1] for o in Path(d).iterdir()
            if o.is_dir() and o.parts[-1] != "_snapshots"]


def path_exists(path: Union[str, Path]) -> bool:
    """Check if path exists.

    Use this to construct paths for future storage support.

    Args:
        path: Path to check

    Returns:
        True if path exists, False otherwise
    """
    return Path(path).exists()


def read_metadata(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read metadata from JSON file.

    Use this to construct paths for future storage support.

    Args:
        path: Directory path containing metadata.json

    Returns:
        Dictionary with metadata, or None if file doesn't exist
    """
    dest = make_path(path, METADATA_FILENAME)
    if path_exists(dest):
        with dest.open() as f:
            metadata = json.load(f)
            logger.debug(f"Read metadata from {dest}")
            return metadata

    legacy_dest = make_path(path, LEGACY_METADATA_FILENAME)
    if path_exists(legacy_dest):
        with legacy_dest.open() as f:
            metadata = json.load(f)
            logger.debug(f"Read metadata from legacy path {legacy_dest}")

        # Migrate legacy metadata file to current filename for future reads
        with dest.open("w") as f:
            json.dump(metadata, f, ensure_ascii=False)
            logger.debug(f"Migrated metadata to {dest}")

        return metadata
    return None


def write_metadata(path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
    """Write metadata to JSON file.

    Use this to construct paths for future storage support.

    Args:
        path: Directory path to write metadata.json
        metadata: Dictionary with metadata to write
    """
    if metadata is None:
        metadata = {}
    
    now = datetime.now()
    metadata["_updated"] = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    meta_file = make_path(path, METADATA_FILENAME)
    with meta_file.open("w") as f:
        json.dump(metadata, f, ensure_ascii=False)
        logger.debug(f"Wrote metadata to {meta_file}")


def make_path(*args: Union[str, Path]) -> Path:
    """Construct path from components.

    Use this to construct paths for future storage support.

    Args:
        *args: Path components

    Returns:
        Path object constructed from components
    """
    # return Path(os.pathjoin(*args))
    return Path(*args)


def get_path(*args: Union[str, Path]) -> Path:
    """Get path relative to pystore root.

    Use this to construct paths for future storage support.

    Args:
        *args: Path components relative to pystore root

    Returns:
        Path object relative to pystore root
    """
    # return Path(os.path.join(config.DEFAULT_PATH, *args))
    return Path(config.DEFAULT_PATH, *args)


def set_path(path: Optional[str]) -> Path:
    """Set the pystore root path.

    Args:
        path: Path to use as pystore root. If None, uses default path.

    Returns:
        The current pystore root path

    Raises:
        ValueError: If path contains non-file:// URL scheme
    """
    path_str: str
    if path is None:
        path_str = str(get_path())
    else:
        path_str = path.rstrip("/").rstrip("\\").rstrip(" ")
        if "://" in path_str and "file://" not in path_str:
            raise ValueError(
                "PyStore currently only works with local file system")
        path_str = os.path.abspath(os.path.expanduser(path_str))

    config.DEFAULT_PATH = path_str
    result_path = get_path()

    # if path does not exist - create it
    if not path_exists(result_path):
        os.makedirs(result_path)

    return result_path


def list_stores() -> List[str]:
    """List all available stores.

    Returns:
        List of store names
    """
    if not path_exists(get_path()):
        os.makedirs(get_path())
    return subdirs(get_path())


def delete_store(store: str) -> bool:
    """Delete a store.

    Args:
        store: Name of the store to delete

    Returns:
        True on success
    """
    shutil.rmtree(get_path(store))
    return True


def delete_stores() -> bool:
    """Delete all stores.

    Returns:
        True on success
    """
    shutil.rmtree(get_path())
    return True


def set_client(scheduler: Optional[str] = None) -> Optional[Client]:
    """Set the Dask distributed client scheduler.

    Args:
        scheduler: Scheduler address or None for local

    Returns:
        The Dask Client instance, or None if no scheduler set
    """
    if scheduler != config._SCHEDULER and config._CLIENT is not None:
        try:
            config._CLIENT.shutdown()
            config._CLIENT = None
        except Exception:
            pass

    config._SCHEDULER = scheduler
    if scheduler is not None:
        config._CLIENT = Client(scheduler)

    return config._CLIENT


def get_client() -> Optional[Client]:
    """Get the current Dask distributed client.

    Returns:
        The Dask Client instance, or None if not set
    """
    return config._CLIENT


def set_partition_size(size: Optional[float] = None) -> float:
    """Set the default partition size.

    Args:
        size: Partition size in bytes. If None, uses default.

    Returns:
        The current partition size
    """
    if size is None:
        size = config.DEFAULT_PARTITION_SIZE * 1
    config.PARTITION_SIZE = size
    return config.PARTITION_SIZE


def get_partition_size() -> float:
    """Get the current partition size.

    Returns:
        The current partition size in bytes
    """
    return config.PARTITION_SIZE
