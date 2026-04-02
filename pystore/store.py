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
import shutil

from . import utils
from .collection import Collection
from .utils import PathSecurityError


class store(object):
    def __repr__(self):
        return "PyStore.datastore <%s>" % self.datastore

    def __init__(self, datastore, engine="fastparquet"):
        # Validate datastore name to prevent path traversal
        try:
            validated_datastore = utils.validate_path_component(datastore)
        except utils.PathSecurityError as e:
            raise ValueError(
                f"Invalid datastore name '{datastore}': {e}"
            )

        datastore_path = utils.get_path()
        if not utils.path_exists(datastore_path):
            os.makedirs(datastore_path)

        self.datastore = utils.make_path(datastore_path, validated_datastore)

        # Validate that the datastore path is within the base path
        utils.validate_path_within_directory(self.datastore, datastore_path)

        if not utils.path_exists(self.datastore):
            os.makedirs(self.datastore)
            utils.write_metadata(self.datastore, {"engine": engine})
            self.engine = engine
        else:
            metadata = utils.read_metadata(self.datastore)
            if metadata:
                self.engine = metadata["engine"]
            else:
                # default / backward compatibility
                self.engine = "fastparquet"
                utils.write_metadata(self.datastore, {"engine": self.engine})

        self.collections = self.list_collections()

    def _create_collection(self, collection, overwrite=False):
        # Validate collection name
        try:
            validated_collection = utils.validate_path_component(collection)
        except utils.PathSecurityError as e:
            raise ValueError(
                f"Invalid collection name '{collection}': {e}"
            )

        # create collection (subdir)
        collection_path = utils.make_path(self.datastore, validated_collection)

        # Validate path is within datastore
        utils.validate_path_within_directory(collection_path, self.datastore)

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
        return Collection(collection, self.datastore)

    def delete_collection(self, collection):
        # Validate collection name
        try:
            validated_collection = utils.validate_path_component(collection)
        except utils.PathSecurityError as e:
            raise ValueError(
                f"Invalid collection name '{collection}': {e}"
            )

        # delete collection (subdir)
        collection_path = utils.make_path(self.datastore, validated_collection)

        # Validate path is within datastore
        utils.validate_path_within_directory(collection_path, self.datastore)

        shutil.rmtree(collection_path)

        # update collections
        self.collections = self.list_collections()
        return True

    def list_collections(self):
        # lists collections (subdirs)
        return utils.subdirs(self.datastore)

    def collection(self, collection, overwrite=False):
        if collection in self.collections and not overwrite:
            return Collection(collection, self.datastore, self.engine)

        # create it
        self._create_collection(collection, overwrite)
        return Collection(collection, self.datastore, self.engine)

    def item(self, collection, item):
        # bypasses collection
        return self.collection(collection).item(item)