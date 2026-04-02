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

"""PyStore Item module for accessing stored data."""

from typing import Any, Dict, List, Optional, Union
import dask.dataframe as dd
import pandas as pd

from . import utils
from .utils import Path


class Item:
    """Represents a stored item in a PyStore collection.

    This class provides access to data and metadata for items stored
    in a PyStore collection.

    Attributes:
        datastore: Path to the datastore
        collection: Name of the collection
        item: Name of the item
        snapshot: Optional snapshot name
        metadata: Dictionary containing item metadata
        data: Dask DataFrame containing the item data
    """

    def __repr__(self) -> str:
        """Return string representation of the Item."""
        return "PyStore.item <%s/%s>" % (self.collection, self.item)

    def __init__(
        self,
        item: str,
        datastore: Union[str, Path],
        collection: str,
        snapshot: Optional[str] = None,
        filters: Optional[List[tuple]] = None,
        columns: Optional[List[str]] = None,
        engine: str = "fastparquet"
    ) -> None:
        """Initialize an Item instance.

        Args:
            item: Name of the item
            datastore: Path to the datastore
            collection: Name of the collection
            snapshot: Optional snapshot name to read from
            filters: Optional filters to apply when reading data
            columns: Optional list of columns to read
            engine: Parquet engine to use (default: fastparquet)

        Raises:
            ValueError: If item or snapshot doesn't exist
        """
        self.engine: str = engine
        self.datastore: Union[str, Path] = datastore
        self.collection: str = collection
        self.snapshot: Optional[str] = snapshot
        self.item: str = item

        self._path: Path = utils.make_path(datastore, collection, item)
        if not self._path.exists():
            raise ValueError(
                "Item `%s` doesn't exist. "
                "Create it using collection.write(`%s`, data, ...)" % (
                    item, item))
        if snapshot:
            snap_path = utils.make_path(
                datastore, collection, "_snapshots", snapshot)

            self._path = utils.make_path(snap_path, item)

            if not utils.path_exists(snap_path):
                raise ValueError("Snapshot `%s` doesn't exist" % snapshot)

            if not utils.path_exists(self._path):
                raise ValueError(
                    "Item `%s` doesn't exist in this snapshot" % item)

        self.metadata: Dict[str, Any] = utils.read_metadata(self._path) or {}
        self.data: dd.DataFrame = dd.read_parquet(
            self._path, engine=self.engine, filters=filters, columns=columns)

    def to_pandas(self, parse_dates: bool = True) -> pd.DataFrame:
        """Convert the item data to a Pandas DataFrame.

        Args:
            parse_dates: If True, attempt to parse the index as dates

        Returns:
            Pandas DataFrame with the item data
        """
        df = self.data.compute()

        if parse_dates and "datetime" not in str(df.index.dtype):
            df.index.name = ""
            if str(df.index.dtype) == "float64":
                df.index = pd.to_datetime(df.index, unit="s")
            elif len(df.index.values) > 0 and df.index.values[0] > 1e6:
                df.index = pd.to_datetime(df.index)

        return df

    def head(self, n: int = 5) -> pd.DataFrame:
        """Get the first n rows of the data.

        Args:
            n: Number of rows to return

        Returns:
            Pandas DataFrame with the first n rows
        """
        return self.data.head(n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """Get the last n rows of the data.

        Args:
            n: Number of rows to return

        Returns:
            Pandas DataFrame with the last n rows
        """
        return self.data.tail(n)
