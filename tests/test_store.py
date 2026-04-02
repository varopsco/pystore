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
import tempfile
import pytest
import pandas as pd

import pystore


class TestStore:
    """Test Store class functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        # Create a temporary directory for pystore
        self.test_dir = tempfile.mkdtemp()
        pystore.set_path(self.test_dir)

        yield

        # Cleanup
        pystore.delete_stores()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_store_creation(self):
        """Test basic store creation."""
        store = pystore.store('test_store')
        
        # Verify store is created
        assert store.datastore is not None
        assert 'test_store' in pystore.list_stores()

    def test_store_repr(self):
        """Test store string representation."""
        store = pystore.store('test_store')
        repr_str = repr(store)
        assert 'PyStore.datastore' in repr_str
        assert 'test_store' in repr_str

    def test_store_default_engine(self):
        """Test that default engine is fastparquet."""
        store = pystore.store('test_store')
        assert store.engine == 'fastparquet'

    def test_store_custom_engine(self):
        """Test store with custom engine (pyarrow)."""
        store = pystore.store('test_store', engine='pyarrow')
        assert store.engine == 'pyarrow'

    def test_list_collections(self):
        """Test listing collections in a store."""
        store = pystore.store('test_store')
        
        # Initially no collections
        collections = store.list_collections()
        assert len(collections) == 0
        
        # Create a collection
        store.collection('test_collection')
        
        # Now should have one collection
        collections = store.list_collections()
        assert len(collections) == 1
        assert 'test_collection' in collections

    def test_collection_creation(self):
        """Test collection creation through store."""
        store = pystore.store('test_store')
        collection = store.collection('test_collection')
        
        assert collection is not None
        assert collection.collection == 'test_collection'
        assert 'test_collection' in store.collections

    def test_collection_retrieval(self):
        """Test retrieving an existing collection."""
        store = pystore.store('test_store')
        
        # Create collection
        collection1 = store.collection('test_collection')
        
        # Retrieve it again
        collection2 = store.collection('test_collection')
        
        assert collection1.collection == collection2.collection

    def test_collection_overwrite(self):
        """Test overwriting an existing collection."""
        store = pystore.store('test_store', engine='pyarrow')
        
        # Create collection with some data
        collection = store.collection('test_collection')
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        data.index = pd.Index([0, 1, 2])
        collection.write('item1', data)
        
        # Overwrite collection
        collection = store.collection('test_collection', overwrite=True)
        
        # Should be empty
        items = collection.list_items()
        assert len(items) == 0

    def test_delete_collection(self):
        """Test deleting a collection."""
        store = pystore.store('test_store')
        
        # Create collection
        store.collection('test_collection')
        assert 'test_collection' in store.collections
        
        # Delete it
        store.delete_collection('test_collection')
        assert 'test_collection' not in store.collections

    def test_item_access(self):
        """Test accessing an item through store."""
        store = pystore.store('test_store', engine='pyarrow')
        collection = store.collection('test_collection')
        
        # Write data
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        data.index = pd.Index([0, 1, 2])
        collection.write('item1', data)
        
        # Access through store
        item = store.item('test_collection', 'item1')
        assert item is not None
        assert item.item == 'item1'

    def test_multiple_stores(self):
        """Test creating multiple stores."""
        store1 = pystore.store('store1')
        store2 = pystore.store('store2')
        
        stores = pystore.list_stores()
        assert 'store1' in stores
        assert 'store2' in stores

    def test_store_persistence(self):
        """Test that store persists across instances."""
        # Create store and collection
        store = pystore.store('test_store')
        store.collection('test_collection')
        
        # Create new store instance
        store2 = pystore.store('test_store')
        
        # Should have the collection
        assert 'test_collection' in store2.collections

    def test_engine_persistence(self):
        """Test that engine setting persists."""
        # Create store with pyarrow
        store = pystore.store('test_store', engine='pyarrow')
        
        # Create new instance
        store2 = pystore.store('test_store')
        
        # Should still have pyarrow
        assert store2.engine == 'pyarrow'


class TestStoreErrors:
    """Test Store error handling."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        # Create a temporary directory for pystore
        self.test_dir = tempfile.mkdtemp()
        pystore.set_path(self.test_dir)

        yield

        # Cleanup
        pystore.delete_stores()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_collection_already_exists_error(self):
        """Test that creating existing collection raises error."""
        store = pystore.store('test_store')
        store.collection('test_collection')
        
        # Try to create again without overwrite - should fail
        with pytest.raises(ValueError):
            store._create_collection('test_collection', overwrite=False)


class TestStoreUtils:
    """Test store-related utility functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        # Create a temporary directory for pystore
        self.test_dir = tempfile.mkdtemp()
        pystore.set_path(self.test_dir)

        yield

        # Cleanup
        pystore.delete_stores()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_list_stores(self):
        """Test list_stores function."""
        # Initially empty
        stores = pystore.list_stores()
        assert len(stores) == 0
        
        # Create stores
        pystore.store('store1')
        pystore.store('store2')
        
        # Should have 2 stores
        stores = pystore.list_stores()
        assert len(stores) == 2
        assert 'store1' in stores
        assert 'store2' in stores

    def test_delete_store(self):
        """Test delete_store function."""
        # Create store
        pystore.store('test_store')
        assert 'test_store' in pystore.list_stores()
        
        # Delete it
        pystore.delete_store('test_store')
        assert 'test_store' not in pystore.list_stores()

    def test_delete_stores(self):
        """Test delete_stores function."""
        # Create multiple stores
        pystore.store('store1')
        pystore.store('store2')
        
        # Delete all
        pystore.delete_stores()
        
        # Should be empty
        stores = pystore.list_stores()
        assert len(stores) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
