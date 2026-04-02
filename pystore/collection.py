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

"""PyStore Collection module for managing data items."""

import os
import time
import shutil
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import dask.dataframe as dd
import multitasking
import pandas as pd

from . import utils
from .item import Item
from . import config
from .utils import Path


logger = logging.getLogger('pystore')


class Collection:
    """Represents a collection of items in a PyStore datastore.

    A collection is a namespace for storing related items (e.g., symbols).
    It provides methods to write, read, append, and manage data items.

    Attributes:
        engine: Parquet engine to use
        datastore: Path to the datastore
        collection: Name of the collection
        items: Set of item names in the collection
        snapshots: Set of snapshot names
    """

    def __repr__(self) -> str:
        """Return string representation of the Collection."""
        return "PyStore.collection <%s>" % self.collection

    def __init__(self, collection: str, datastore: str, engine: str = "fastparquet") -> None:
        """Initialize a Collection instance.

        Args:
            collection: Name of the collection
            datastore: Path to the datastore
            engine: Parquet engine to use (default: fastparquet)
        """
        self.engine: str = engine
        self.datastore: str = datastore
        self.collection: str = collection
        self.items: Set[str] = self.list_items()
        self.snapshots: Set[str] = self.list_snapshots()

    def _item_path(self, item: str, as_string: bool = False) -> Union[Path, str]:
        """Get the path to an item.

        Args:
            item: Name of the item
            as_string: If True, return path as string

        Returns:
            Path object or string to the item
        """
        p = utils.make_path(self.datastore, self.collection, item)
        if as_string:
            return str(p)
        return p

    @multitasking.task
    def _list_items_threaded(self, **kwargs: Any) -> None:
        """Update items list in a background thread.

        Args:
            **kwargs: Filter arguments for list_items
        """
        self.items = self.list_items(**kwargs)

    def list_items(self, **kwargs: Any) -> Set[str]:
        """List items in the collection, optionally filtered by metadata.

        Args:
            **kwargs: Metadata key-value pairs to filter by

        Returns:
            Set of item names matching the filter (or all items if no filter)
        """
        dirs = utils.subdirs(utils.make_path(self.datastore, self.collection))
        if not kwargs:
            return set(dirs)

        matched: List[str] = []
        for d in dirs:
            meta = utils.read_metadata(utils.make_path(
                self.datastore, self.collection, d))
            if meta is None:
                continue
            meta_copy = meta.copy()
            del meta_copy["_updated"]

            m = 0
            keys = list(meta_copy.keys())
            for k, v in kwargs.items():
                if k in keys and meta_copy[k] == v:
                    m += 1

            if m == len(kwargs):
                matched.append(d)

        return set(matched)

    def item(
        self,
        item: str,
        snapshot: Optional[str] = None,
        filters: Optional[List[tuple]] = None,
        columns: Optional[List[str]] = None
    ) -> Item:
        """Get an item from the collection.

        Args:
            item: Name of the item
            snapshot: Optional snapshot name to read from
            filters: Optional filters to apply when reading
            columns: Optional list of columns to read

        Returns:
            Item instance
        """
        return Item(item, self.datastore, self.collection,
                    snapshot, filters, columns, engine=self.engine)

    def index(self, item: str, last: bool = False) -> Union[pd.Index, float]:
        """Get the index of an item.

        Args:
            item: Name of the item
            last: If True, return only the last index value as float

        Returns:
            Pandas Index or last value as float
        """
        data = dd.read_parquet(self._item_path(item, as_string=True),
                               columns="index", engine=self.engine)
        if not last:
            return data.index.compute()

        return float(str(data.index).split(
                     "\nName")[0].split("\n")[-1].split(" ")[0])

    def delete_item(self, item: str, reload_items: bool = False) -> bool:
        """Delete an item from the collection.

        Args:
            item: Name of the item to delete
            reload_items: If True, reload the items list after deletion

        Returns:
            True on success
        """
        logger.info(f"Deleting item '{item}' from collection '{self.collection}'")
        shutil.rmtree(self._item_path(item))
        self.items.discard(item)
        if reload_items:
            self._list_items_threaded()
        logger.info(f"Successfully deleted item '{item}' from collection '{self.collection}'")
        return True

    def rename_item(self, old_item: str, new_item: str, reload_items: bool = False) -> bool:
        """Rename an item in the collection.

        Parameters
        ----------
        old_item : str
            The current name of the item
        new_item : str
            The new name for the item
        reload_items : bool, optional (default=False)
            If True, reload the list of items after renaming

        Returns
        -------
        bool : True if successful

        Raises
        ------
        ValueError : If old_item doesn't exist or new_item already exists
        """
        logger.info(f"Renaming item '{old_item}' to '{new_item}' in collection '{self.collection}'")

        # Check if old item exists
        if not utils.path_exists(self._item_path(old_item)):
            raise ValueError(f"Item '{old_item}' does not exist")

        # Check if new item doesn't already exist
        if utils.path_exists(self._item_path(new_item)):
            raise ValueError(f"Item '{new_item}' already exists")

        # Rename the item directory
        old_path = self._item_path(old_item, as_string=True)
        new_path = self._item_path(new_item, as_string=True)
        shutil.move(old_path, new_path)

        # Update items set
        self.items.discard(old_item)
        self.items.add(new_item)

        if reload_items:
            self._list_items_threaded()

        logger.info(f"Successfully renamed item '{old_item}' to '{new_item}' in collection '{self.collection}'")
        return True

    @multitasking.task
    def write_threaded(
        self,
        item: str,
        data: Union[pd.DataFrame, dd.DataFrame, Item],
        metadata: Optional[Dict[str, Any]] = None,
        npartitions: Optional[int] = None,
        chunksize: Optional[int] = None,
        overwrite: bool = False,
        epochdate: bool = False,
        reload_items: bool = False,
        **kwargs: Any
    ) -> None:
        """Write data to an item in a background thread.

        Args:
            item: Name of the item
            data: Data to write (Pandas or Dask DataFrame or Item)
            metadata: Optional metadata dictionary
            npartitions: Optional number of partitions
            chunksize: Optional chunk size
            overwrite: If True, overwrite existing item
            epochdate: If True, convert datetime index to epoch
            reload_items: If True, reload items list after write
            **kwargs: Additional arguments passed to dd.to_parquet
        """
        self.write(item, data, metadata,
                   npartitions, chunksize, overwrite,
                   epochdate, reload_items,
                   **kwargs)

    def write(
        self,
        item: str,
        data: Union[pd.DataFrame, dd.DataFrame, Item],
        metadata: Optional[Dict[str, Any]] = None,
        npartitions: Optional[int] = None,
        chunksize: Optional[int] = None,
        overwrite: bool = False,
        epochdate: bool = False,
        reload_items: bool = False,
        **kwargs: Any
    ) -> None:
        """Write data to an item in the collection.

        Args:
            item: Name of the item
            data: Data to write (Pandas or Dask DataFrame or Item)
            metadata: Optional metadata dictionary
            npartitions: Optional number of partitions
            chunksize: Optional chunk size
            overwrite: If True, overwrite existing item
            epochdate: If True, convert datetime index to epoch
            reload_items: If True, reload items list after write
            **kwargs: Additional arguments passed to dd.to_parquet

        Raises:
            ValueError: If item exists and overwrite is False
        """
        if metadata is None:
            metadata = {}

        logger.info(f"Writing item '{item}' to collection '{self.collection}'")

        if utils.path_exists(self._item_path(item)) and not overwrite:
            raise ValueError(
                "Item already exists. To overwrite, use `overwrite=True`. "
                "Otherwise, use `<collection>.append()`")

        if isinstance(data, Item):
            data = data.to_pandas()
        else:
            # work on copy
            data = data.copy()

        if epochdate or "datetime" in str(data.index.dtype):
            data = utils.datetime_to_int64(data)
            # Check if index has nanosecond attribute (DatetimeIndex)
            if hasattr(data.index, 'nanosecond'):
                nanoseconds = data.index.nanosecond
                if hasattr(nanoseconds, 'any'):
                    if nanoseconds.any() and "times" not in kwargs:
                        kwargs["times"] = "int96"
                elif any(nanoseconds) and "times" not in kwargs:
                    kwargs["times"] = "int96"

        if data.index.name == "":
            data.index.name = "index"

        if npartitions is None and chunksize is None:
            memusage = data.memory_usage(deep=True).sum()
            if isinstance(data, dd.DataFrame):
                npartitions = int(
                    1 + memusage.compute() // config.PARTITION_SIZE)
                data.repartition(npartitions=npartitions)
            else:
                npartitions = int(
                    1 + memusage // config.PARTITION_SIZE)
                data = dd.from_pandas(data, npartitions=npartitions)

        dd.to_parquet(data, self._item_path(item, as_string=True),
                      compression="snappy", engine=self.engine, **kwargs)

        utils.write_metadata(utils.make_path(
            self.datastore, self.collection, item), metadata)

        # update items
        self.items.add(item)
        if reload_items:
            self._list_items_threaded()

        logger.info(f"Successfully wrote item '{item}' to collection '{self.collection}'")

    def _get_item_schema(self, item: str) -> Dict[str, Any]:
        """Extract schema from existing item.

        Returns a dictionary containing column names, dtypes, and index type.

        Args:
            item: Name of the item

        Returns:
            Dictionary with 'columns', 'dtypes', 'index_name', 'index_type'
        """
        item_path = self._item_path(item, as_string=True)
        # Read metadata only (not full data) to get schema
        ddf = dd.read_parquet(item_path, engine=self.engine)
        schema: Dict[str, Any] = {
            'columns': list(ddf.columns),
            'dtypes': ddf.dtypes.to_dict(),
            'index_name': ddf.index.name,
            'index_type': str(type(ddf.index).__name__)
        }
        return schema

    def _validate_data_compatibility(
        self,
        new_data: pd.DataFrame,
        existing_schema: Dict[str, Any],
        strictness: str = 'strict',
        allow_extra_columns: bool = False
    ) -> Tuple[bool, List[str]]:
        """Validate that new data is compatible with existing data schema.

        Parameters
        ----------
        new_data : pandas.DataFrame
            The new data to validate
        existing_schema : dict
            Schema of existing data from _get_item_schema
        strictness : str
            'strict' - raises ValueError on mismatch
            'warn' - logs warning but proceeds
            'disabled' - no validation
        allow_extra_columns : bool
            If True, allows extra columns in new data

        Returns
        -------
        tuple : (is_valid, error_messages)
        """
        import warnings

        if strictness == 'disabled':
            return True, []

        errors: List[str] = []
        existing_columns: Set[str] = set(existing_schema['columns'])
        new_columns: Set[str] = set(new_data.columns)

        # Check for missing columns (columns in existing but not in new)
        missing_columns = existing_columns - new_columns
        if missing_columns:
            errors.append(
                f"Missing columns in new data: {sorted(missing_columns)}"
            )

        # Check for extra columns (columns in new but not in existing)
        extra_columns = new_columns - existing_columns
        if extra_columns and not allow_extra_columns:
            errors.append(
                f"Extra columns in new data not present in existing: "
                f"{sorted(extra_columns)}"
            )

        # Check dtype compatibility for common columns
        common_columns = existing_columns & new_columns
        for col in common_columns:
            existing_dtype = existing_schema['dtypes'].get(col)
            new_dtype = new_data[col].dtype
            if existing_dtype is not None:
                # Check if dtypes are compatible
                if not self._are_dtypes_compatible(existing_dtype, new_dtype):
                    errors.append(
                        f"dtype mismatch for column '{col}': "
                        f"existing={existing_dtype}, new={new_dtype}"
                    )

        # Check index type
        existing_index_type = existing_schema.get('index_type', 'Index')
        new_index_type = str(type(new_data.index).__name__)
        if existing_index_type != new_index_type:
            errors.append(
                f"index type mismatch: existing={existing_index_type}, "
                f"new={new_index_type}"
            )

        # Handle validation based on strictness
        if errors:
            if strictness == 'strict':
                return False, errors
            elif strictness == 'warn':
                warning_msg = "Schema validation warnings: " + "; ".join(errors)
                warnings.warn(warning_msg)
                return True, []  # Still valid in warn mode

        return True, []

    def _are_dtypes_compatible(self, existing_dtype: Any, new_dtype: Any) -> bool:
        """Check if two pandas dtypes are compatible.

        Parameters
        ----------
        existing_dtype : pandas dtype
            The existing dtype
        new_dtype : pandas dtype
            The new dtype

        Returns
        -------
        bool : True if compatible, False otherwise
        """
        # Use pandas API for dtype comparison
        if pd.api.types.is_dtype_equal(existing_dtype, new_dtype):
            return True

        # Check for numeric type compatibility
        existing_is_numeric = pd.api.types.is_numeric_dtype(existing_dtype)
        new_is_numeric = pd.api.types.is_numeric_dtype(new_dtype)

        if existing_is_numeric and new_is_numeric:
            # Both numeric - check if both are integer or both are float
            existing_is_float = pd.api.types.is_float_dtype(existing_dtype)
            new_is_float = pd.api.types.is_float_dtype(new_dtype)
            return existing_is_float == new_is_float

        # Check for string type compatibility
        existing_is_string = (
            pd.api.types.is_string_dtype(existing_dtype) or
            existing_dtype == 'object'
        )
        new_is_string = (
            pd.api.types.is_string_dtype(new_dtype) or
            new_dtype == 'object'
        )
        return existing_is_string == new_is_string

    def append(
        self,
        item: str,
        data: Union[pd.DataFrame, dd.DataFrame],
        npartitions: Optional[int] = None,
        epochdate: bool = False,
        threaded: bool = False,
        reload_items: bool = False,
        remove_duplicates: Optional[str] = None,
        validate_schema: bool = False,
        schema_strictness: str = 'strict',
        allow_extra_columns: bool = False,
        **kwargs: Any
    ) -> None:
        """Append new data to the collection.

        Saves new data to the collection and optionially removes duplicates
        within the data.

        Parameters
        ----------
        item : str
            Name of the item
        data : pandas.DataFrame or dask.DataFrame
            Data to append
        npartitions : int, optional
            Number of partitions for the combined data
        epochdate : bool, optional (default=False)
            If True, convert datetime index to epoch
        threaded : bool, optional (default=False)
            If True, write in a background thread
        reload_items : bool, optional (default=False)
            If True, reload items list after append
        remove_duplicates : str, optional (default=None)
            Defines how duplicates within the combined dataframe will be
                handled.
            None = no check for duplicated data. This is the fastest option
                but the user is responsible for not having an overlap
                between the new and old data
            "index" = For data with unique index but non unique row values.
                Rows with duplicated indices will be deleted. Ignores the
                values
            "values" = For data with non unique index but unique row values.
                Rows with duplicated values will be deleted. Ignores the index
            "all" = For data with unique index and unique row values. Rows
                with duplicated indices will be deleted first and then all
                rows with duplicated values will be deleted
            "values_in_index" = For data with non unique index but unique row
                values within index duplicates. Rows with duplicated values
                within the same index will be deleted
        validate_schema : bool, optional (default=False)
            When True, validates schema compatibility before appending data.
            This checks column names, dtypes, and index type compatibility.
        schema_strictness : str, optional (default='strict')
            Controls behavior on validation failure. Valid values:
            'strict' - raises ValueError on mismatch
            'warn' - logs warning but proceeds with append
            'disabled' - no validation performed
        allow_extra_columns : bool, optional (default=False)
            When True, allows appended data to have columns not present in
            existing data. Only relevant when validate_schema=True.
        **kwargs
            Additional arguments passed to write

        Returns
        -------
        None
        """
        logger.info(f"Appending data to item '{item}' in collection '{self.collection}'")

        if not utils.path_exists(self._item_path(item)):
            raise ValueError(
                "Item does not exist. Use `<collection>.write(...)`")

        # work on copy
        data = data.copy()

        # Validate schema if enabled
        if validate_schema:
            existing_schema = self._get_item_schema(item)
            is_valid, error_messages = self._validate_data_compatibility(
                data, existing_schema, schema_strictness, allow_extra_columns
            )
            if not is_valid:
                raise ValueError(
                    "Schema validation failed: " + "; ".join(error_messages)
                )

        try:
            should_convert = epochdate
            if not should_convert and "datetime" in str(data.index.dtype):
                # Check if index has nanosecond attribute (DatetimeIndex)
                if hasattr(data.index, 'nanosecond'):
                    nanoseconds = data.index.nanosecond
                    if hasattr(nanoseconds, 'any'):
                        should_convert = nanoseconds.any()
                    else:
                        should_convert = any(nanoseconds)
            
            if should_convert:
                data = utils.datetime_to_int64(data)
            old_index = dd.read_parquet(self._item_path(item, as_string=True),
                                        columns=[], engine=self.engine
                                        ).index.compute()
            data = data[~data.index.isin(old_index)]
        except Exception:
            return

        if data.empty:
            return

        if data.index.name == "":
            data.index.name = "index"

        # get old and new dataframe
        current = self.item(item)
        new = dd.from_pandas(data, npartitions=1)

        # combine old dataframe with new and optionally remove duplicates from
        # combined dataframe
        idx_name = data.index.name
        if remove_duplicates is None:
            combined = dd.concat([current.data, new])
        elif remove_duplicates == 'index':
            combined = dd.concat([current.data, new])\
                .reset_index()\
                .drop_duplicates(subset=idx_name, keep="last")\
                .set_index(idx_name)
        elif remove_duplicates == 'values':
            combined = dd.concat([current.data, new])\
                .drop_duplicates(keep="last")
        elif remove_duplicates == 'all':
            combined = dd.concat([current.data, new])\
                .reset_index()\
                .drop_duplicates(subset=idx_name, keep="last")\
                .set_index(idx_name)\
                .drop_duplicates(keep="last")
        elif remove_duplicates == 'values_in_index':
            combined = dd.concat([current.data, new])\
                .reset_index()\
                .drop_duplicates(keep="last")\
                .set_index(idx_name)
        else:
            raise ValueError(
                "argument remove_duplicates must either be None, 'index', "
                "'values', 'all' or 'values_in_index'")

        if npartitions is None:
            memusage = combined.memory_usage(deep=True).sum()
            if isinstance(combined, dd.DataFrame):
                memusage = memusage.compute()
            npartitions = int(1 + memusage // config.PARTITION_SIZE)

        # write data
        write = self.write_threaded if threaded else self.write
        write(item, combined, npartitions=npartitions, chunksize=None,
              metadata=current.metadata, overwrite=True,
              epochdate=epochdate, reload_items=reload_items, **kwargs)

        logger.info(f"Successfully appended data to item '{item}' in collection '{self.collection}'")

    def create_snapshot(self, snapshot: Optional[str] = None) -> bool:
        """Create a snapshot of the current collection state.

        Args:
            snapshot: Optional name for the snapshot. If None, uses timestamp.

        Returns:
            True on success
        """
        if snapshot:
            snapshot = "".join(
                e for e in snapshot if e.isalnum() or e in [".", "_"])
        else:
            snapshot = str(int(time.time() * 1000000))

        src = utils.make_path(self.datastore, self.collection)
        dst = utils.make_path(src, "_snapshots", snapshot)

        shutil.copytree(src, dst,
                        ignore=shutil.ignore_patterns("_snapshots"))

        self.snapshots = self.list_snapshots()
        return True

    def list_snapshots(self) -> Set[str]:
        """List all snapshots in the collection.

        Returns:
            Set of snapshot names
        """
        snapshots = utils.subdirs(utils.make_path(
            self.datastore, self.collection, "_snapshots"))
        return set(snapshots)

    def delete_snapshot(self, snapshot: str) -> bool:
        """Delete a snapshot from the collection.

        Args:
            snapshot: Name of the snapshot to delete

        Returns:
            True on success
        """
        if snapshot not in self.snapshots:
            # raise ValueError("Snapshot `%s` doesn't exist" % snapshot)
            return True

        shutil.rmtree(utils.make_path(self.datastore, self.collection,
                                      "_snapshots", snapshot))
        self.snapshots = self.list_snapshots()
        return True

    def delete_snapshots(self) -> bool:
        """Delete all snapshots from the collection.

        Returns:
            True on success
        """
        snapshots_path = utils.make_path(
            self.datastore, self.collection, "_snapshots")
        shutil.rmtree(snapshots_path)
        os.makedirs(snapshots_path)
        self.snapshots = self.list_snapshots()
        return True
