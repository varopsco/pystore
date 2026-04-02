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

"""PyStore Store module for managing datastore operations."""

import os
import shutil
from typing import List, Optional, Set

from . import utils
from .collection import Collection
from .item import Item
from .utils import Path


class Store:
    """Represents a PyStore datastore.

    A datastore is a container for collections, which in turn contain items.
    This class provides methods to create, access, and manage collections.

    Attributes:
        datastore: Path to the datastore directory
        engine: Parquet engine to use (fastparquet or pyarrow)
        collections: Set of collection names in the datastore
    """

    def __repr__(self) -> str:
        """Return string representation of the store."""
        return "PyStore.datastore <%s>" % self.datastore

    def __init__(self, datastore: str, engine: str = "fastparquet") -> None:
        """Initialize a store instance.

        Args:
            datastore: Name of the datastore
            engine: Parquet engine to use (default: fastparquet)
        """
        datastore_path: Path = utils.get_path()
        if not utils.path_exists(datastore_path):
            os.makedirs(datastore_path)

        self.datastore: Path = utils.make_path(datastore_path, datastore)
        if not utils.path_exists(self.datastore):
            os.makedirs(self.datastore)
            utils.write_metadata(self.datastore, {"engine": engine})
            self.engine: str = engine
        else:
            metadata = utils.read_metadata(self.datastore)
            if metadata:
                self.engine = metadata["engine"]
            else:
                # default / backward compatibility
                self.engine = "fastparquet"
                utils.write_metadata(self.datastore, {"engine": self.engine})

        self.collections: Set[str] = self.list_collections()

    def _create_collection(self, collection: str, overwrite: bool = False) -> Collection:
        """Create a new collection in the datastore.

        Args:
            collection: Name of the collection to create
            overwrite: If True, overwrite existing collection

        Returns:
            Collection instance for the created collection

        Raises:
            ValueError: If collection exists and overwrite is False
        """
        # create collection (subdir)
        collection_path: Path = utils.make_path(self.datastore, collection)
        if utils.path_exists(collection_path):
            if overwrite:
                self.delete_collection(collection)
            else:
                raise ValueError(
                    "Collection exists! To overwrite, use `overwrite=True`")

        os.makedirs(collection_path)
        os.makedirs(utils.make_path(collection_path, "_snapshots"))

        # update collections
        self.collections = self.list_collections()

        # return the collection
        return Collection(collection, str(self.datastore), self.engine)

    def delete_collection(self, collection: str) -> bool:
        """Delete a collection from the datastore.

        Args:
            collection: Name of the collection to delete

        Returns:
            True on success
        """
        # delete collection (subdir)
        shutil.rmtree(utils.make_path(self.datastore, collection))

        # update collections
        self.collections = self.list_collections()
        return True

    def list_collections(self) -> Set[str]:
        """List all collections in the datastore.

        Returns:
            Set of collection names
        """
        # lists collections (subdirs)
        return set(utils.subdirs(self.datastore))

    def collection(self, collection: str, overwrite: bool = False) -> Collection:
        """Get or create a collection.

        Args:
            collection: Name of the collection
            overwrite: If True, overwrite existing collection

        Returns:
            Collection instance
        """
        if collection in self.collections and not overwrite:
            return Collection(collection, str(self.datastore), self.engine)

        # create it
        self._create_collection(collection, overwrite)
        return Collection(collection, str(self.datastore), self.engine)

    def item(self, collection: str, item: str) -> Item:
        """Get an item from a collection.

        This bypasses the collection object and directly accesses the item.

        Args:
            collection: Name of the collection
            item: Name of the item

        Returns:
            Item instance
        """
        # bypasses collection
        return self.collection(collection).item(item)


# Backward compatibility alias (PEP8 recommends CapWords for class names)
store = Store