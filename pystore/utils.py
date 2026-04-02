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
import re
import logging
from datetime import datetime
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
    from pathlib2 import Path

from . import config


# Configure logger for pystore - use NullHandler by default (best practice for libraries)
# Users can call configure_logging() to set up default logging behavior
logger = logging.getLogger('pystore')
logger.addHandler(logging.NullHandler())


class PathSecurityError(Exception):
    """Exception raised when a path security violation is detected."""
    pass


def validate_path_component(component, allow_path_separators=False):
    """Validate a path component for security.

    This function sanitizes individual path components (like collection names,
    item names, snapshot names) to prevent path traversal attacks.

    Parameters
    ----------
    component : str or Path
        The path component to validate
    allow_path_separators : bool, optional (default=False)
        If True, allow path separators in the component (for multi-part paths)

    Returns
    -------
    str : The validated and sanitized component

    Raises
    ------
    PathSecurityError : If the component contains malicious patterns
    TypeError : If component is not a string or Path
    """
    if component is None:
        raise PathSecurityError("Path component cannot be None")

    # Convert Path to string
    if isinstance(component, Path):
        component = str(component)
    elif not isinstance(component, (str, bytes)):
        raise TypeError(f"Path component must be str or Path, got {type(component)}")

    if isinstance(component, bytes):
        component = component.decode('utf-8', errors='strict')

    # Check for null bytes (potential security bypass)
    if '\x00' in component or '\0' in component:
        raise PathSecurityError("Path component contains null bytes")

    # Check for absolute path indicators FIRST (before path separator check)
    # This gives more specific error messages
    if component.startswith('/') or (len(component) > 1 and component[1] == ':'):
        raise PathSecurityError(
            f"Path component '{component}' appears to be an absolute path"
        )

    # Check for path traversal sequences
    if '..' in component:
        raise PathSecurityError(
            f"Path component '{component}' contains path traversal sequence '..'"
        )

    # Check for path separators unless explicitly allowed
    if not allow_path_separators:
        # Check for both forward and backward slashes
        if '/' in component or '\\' in component:
            raise PathSecurityError(
                f"Path component '{component}' contains path separators. "
                "Use make_path() to construct multi-part paths safely."
            )

    # Check for shell expansion characters that could be dangerous
    dangerous_chars = ['$', '`', '!', ';', '&', '|', '<', '>', '(', ')']
    for char in dangerous_chars:
        if char in component:
            raise PathSecurityError(
                f"Path component '{component}' contains potentially dangerous character '{char}'"
            )

    # Check for leading/trailing whitespace that could cause issues
    if component != component.strip():
        raise PathSecurityError(
            f"Path component '{component}' contains leading or trailing whitespace"
        )

    # Validate that the component is not empty after validation
    if not component or component.strip() == '':
        raise PathSecurityError("Path component is empty or contains only whitespace")

    return component


def validate_path_within_directory(path, base_directory):
    """Validate that a path stays within the specified base directory.

    This is the core security function that prevents path traversal attacks
    by ensuring the resolved absolute path is within the allowed directory tree.

    Parameters
    ----------
    path : str or Path
        The path to validate
    base_directory : str or Path
        The base directory that the path must stay within

    Returns
    -------
    Path : The validated absolute path

    Raises
    ------
    PathSecurityError : If the path escapes the base directory
    """
    if path is None or base_directory is None:
        raise PathSecurityError("Path and base_directory cannot be None")

    path = Path(path)
    base_directory = Path(base_directory)

    # Resolve to absolute paths
    try:
        resolved_path = path.resolve()
        resolved_base = base_directory.resolve()
    except (OSError, RuntimeError) as e:
        raise PathSecurityError(f"Failed to resolve paths: {e}")

    # Check if the resolved path is within the base directory
    try:
        # relative_to will raise ValueError if path is not under base
        resolved_path.relative_to(resolved_base)
    except ValueError:
        raise PathSecurityError(
            f"Path '{path}' resolves to '{resolved_path}' which is outside "
            f"the allowed directory '{resolved_base}'"
        )

    return resolved_path


def configure_logging(level=logging.INFO, format_string=None):
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


def read_csv(urlpath, *args, **kwargs):
    def rename_dask_index(df, name):
        df.index.name = name
        return df

    index_col = index_name = None

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


def datetime_to_int64(df):
    """ convert datetime index to epoch int
    allows for cross language/platform portability
    """

    if isinstance(df.index, dd.Index) and (
            isinstance(df.index, pd.DatetimeIndex) and
            any(df.index.nanosecond) > 0):
        df.index = df.index.astype(np.int64)  # / 1e9

    return df


def subdirs(d):
    """ use this to construct paths for future storage support """
    return [o.parts[-1] for o in Path(d).iterdir()
            if o.is_dir() and o.parts[-1] != "_snapshots"]


def path_exists(path):
    """ use this to construct paths for future storage support """
    return path.exists()


def read_metadata(path):
    """ use this to construct paths for future storage support """
    dest = make_path(path, "metadata.json")
    if path_exists(dest):
        with dest.open() as f:
            metadata = json.load(f)
            logger.debug(f"Read metadata from {dest}")
            return metadata


def write_metadata(path, metadata={}):
    """ use this to construct paths for future storage support """
    now = datetime.now()
    metadata["_updated"] = now.strftime("%Y-%m-%d %H:%I:%S.%f")
    meta_file = make_path(path, "metadata.json")
    with meta_file.open("w") as f:
        json.dump(metadata, f, ensure_ascii=False)
        logger.debug(f"Wrote metadata to {meta_file}")


def make_path(*args, validate=False):
    """Construct a path from multiple components with optional security validation.

    This is the recommended way to construct paths in PyStore. By default,
    it does NOT validate components (for backward compatibility and to allow
    absolute base paths). Use validate=True only for user-provided components.

    Security Note: The primary security check is validate_path_within_directory(),
    which should be called on the final path to ensure it stays within the
    allowed directory tree.

    Parameters
    ----------
    *args : str or Path
        Path components to join
    validate : bool, optional (default=False)
        If True, validate each component for security (use only for user input)

    Returns
    -------
    Path : The constructed path object

    Raises
    ------
    PathSecurityError : If validation is enabled and a component fails validation
    """
    if not args:
        return Path()

    if validate:
        validated_args = []
        for arg in args:
            if arg is not None:
                # Validate each component
                validated_arg = validate_path_component(arg)
                validated_args.append(validated_arg)
            else:
                validated_args.append(arg)
        return Path(*validated_args)

    return Path(*args)


def get_path(*args):
    """ use this to construct paths for future storage support """
    # return Path(os.path.join(config.DEFAULT_PATH, *args))
    return Path(config.DEFAULT_PATH, *args)


def set_path(path):
    """Set the base path for PyStore data storage.

    This function validates and sets the base directory for all PyStore operations.
    It ensures the path is a valid local filesystem path and creates it if needed.

    Parameters
    ----------
    path : str or Path
        The base path for PyStore storage

    Returns
    -------
    Path : The configured storage path

    Raises
    ------
    PathSecurityError : If the path is invalid or points to non-local storage
    """
    if path is None:
        path = get_path()
    else:
        # Sanitize the path
        if isinstance(path, Path):
            path_str = str(path)
        elif isinstance(path, str):
            path_str = path
        else:
            raise PathSecurityError(
                f"Path must be str or Path, got {type(path)}"
            )

        # Check for null bytes
        if '\x00' in path_str or '\0' in path_str:
            raise PathSecurityError("Path contains null bytes")

        # Strip whitespace and trailing separators
        path_str = path_str.strip().rstrip("/").rstrip("\\")

        # Check for remote storage schemes
        if "://" in path_str and "file://" not in path_str:
            raise ValueError(
                "PyStore currently only works with local file system")

        # Remove file:// prefix if present
        if path_str.startswith("file://"):
            path_str = path_str[7:]

        # Validate the path doesn't contain dangerous patterns
        # We allow absolute paths for the base storage path
        path = Path(path_str)

        # Check for path traversal in the base path itself
        if '..' in path_str:
            raise PathSecurityError(
                "Base storage path cannot contain '..' traversal sequences"
            )

    config.DEFAULT_PATH = path
    path = get_path()

    # if path does not exist - create it
    if not path_exists(get_path()):
        os.makedirs(get_path())

    return get_path()


def list_stores():
    if not path_exists(get_path()):
        os.makedirs(get_path())
    return subdirs(get_path())


def delete_store(store):
    validated_store = validate_path_component(store)
    store_path = get_path(validated_store)
    validate_path_within_directory(store_path, get_path())
    shutil.rmtree(store_path)
    return True


def delete_stores():
    shutil.rmtree(get_path())
    return True


def set_client(scheduler=None):
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


def get_client():
    return config._CLIENT


def set_partition_size(size=None):
    if size is None:
        size = config.DEFAULT_PARTITION_SIZE * 1
    config.PARTITION_SIZE = size
    return config.PARTITION_SIZE


def get_partition_size():
    return config.PARTITION_SIZE
