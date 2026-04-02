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
    """Test store functionality."""

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

    def _create_sample_data(self):
        """Create sample DataFrame for testing."""
        data = {
            'a': [1, 2, 3],
            'b': [1.0, 2.0, 3.0],
            'c': ['x', 'y', 'z']
        }
        return pd.DataFrame(data)

    def test_store_repr(self):
        """Test store __repr__ method."""
        store = pystore.store('test_store')
        assert 'PyStore.datastore' in repr(store)
        assert 'test_store' in repr(store)

    def test_store_initialization_creates_datastore(self):
        """Test that store initialization creates datastore directory."""
        store = pystore.store('new_store')
        assert os.path.exists(store.datastore)

    def test_store_initialization_with_existing_datastore(self):
        """Test store initialization with existing datastore."""
        # Create first store
        store1 = pystore.store('store1')
        
        # Create second store in same datastore path (using same pystore path)
        store2 = pystore.store('store2')
        
        assert os.path.exists(store2.datastore)
        # Verify both stores exist in the list
        stores = pystore.list_stores()
        assert 'store1' in stores
        assert 'store2' in stores

    def test_store_list_collections(self):
        """Test listing collections in a store."""
        store = pystore.store('test_store')
        
        # Initially no collections
        collections = store.list_collections()
        assert len(collections) == 0
        
        # Create collections
        store.collection('coll1')
        store.collection('coll2')
        
        # List collections
        collections = store.list_collections()
        assert len(collections) == 2
        assert 'coll1' in collections
        assert 'coll2' in collections

    def test_store_create_collection(self):
        """Test creating a collection."""
        store = pystore.store('test_store')
        collection = store.collection('test_collection')
        
        assert collection is not None
        assert 'test_collection' in store.collections

    def test_store_create_collection_overwrite(self):
        """Test creating a collection with overwrite."""
        store = pystore.store('test_store', engine='pyarrow')
        
        # Create collection
        collection1 = store.collection('test_collection')
        data = self._create_sample_data()
        collection1.write('item1', data)
        
        # Verify item exists
        items = collection1.list_items()
        assert 'item1' in items
        
        # Overwrite collection using _create_collection directly
        collection2 = store._create_collection('test_collection', overwrite=True)
        
        # Verify collection was overwritten (should be empty)
        items = collection2.list_items()
        assert len(items) == 0

    def test_store_create_existing_collection_raises_error(self):
        """Test that creating existing collection without overwrite raises error."""
        store = pystore.store('test_store')
        
        # Create collection
        store.collection('test_collection')
        
        # Try to create again without overwrite
        with pytest.raises(ValueError) as exc_info:
            store._create_collection('test_collection', overwrite=False)
        
        assert 'Collection exists' in str(exc_info.value)

    def test_store_delete_collection(self):
        """Test deleting a collection."""
        store = pystore.store('test_store')
        
        # Create collection
        store.collection('test_collection')
        assert 'test_collection' in store.collections
        
        # Delete collection
        result = store.delete_collection('test_collection')
        assert result is True
        assert 'test_collection' not in store.collections

    def test_store_item_accessor(self):
        """Test accessing item directly from store."""
        store = pystore.store('test_store', engine='pyarrow')
        collection = store.collection('test_collection')
        
        # Write data
        data = self._create_sample_data()
        collection.write('test_item', data)
        
        # Access item directly from store
        item = store.item('test_collection', 'test_item')
        assert item is not None
        assert item.item == 'test_item'
        # Verify data can be read
        df = item.to_pandas()
        assert len(df) == 3

    def test_store_engine_persistence(self):
        """Test that store engine is persisted in metadata."""
        # Create store with pyarrow engine
        store1 = pystore.store('test_store', engine='pyarrow')
        assert store1.engine == 'pyarrow'
        
        # Reopen store (should load engine from metadata)
        store2 = pystore.store('test_store')
        assert store2.engine == 'pyarrow'

    def test_store_engine_backward_compatibility(self):
        """Test that store without engine metadata defaults to fastparquet."""
        import pystore.utils as utils
        
        # Create store directory manually without metadata
        store_path = utils.make_path(utils.get_path(), 'manual_store')
        os.makedirs(store_path)
        
        # Open store (should default to fastparquet)
        store = pystore.store('manual_store')
        assert store.engine == 'fastparquet'


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
        """Test listing all stores."""
        # Initially no stores
        stores = pystore.list_stores()
        assert len(stores) == 0
        
        # Create stores
        pystore.store('store1')
        pystore.store('store2')
        
        # List stores
        stores = pystore.list_stores()
        assert len(stores) == 2
        assert 'store1' in stores
        assert 'store2' in stores

    def test_delete_store(self):
        """Test deleting a store."""
        # Create store
        pystore.store('store_to_delete')
        
        # Verify store exists
        stores = pystore.list_stores()
        assert 'store_to_delete' in stores
        
        # Delete store
        result = pystore.delete_store('store_to_delete')
        assert result is True
        
        # Verify store is deleted
        stores = pystore.list_stores()
        assert 'store_to_delete' not in stores

    def test_delete_stores(self):
        """Test deleting all stores."""
        # Create multiple stores
        pystore.store('store1')
        pystore.store('store2')
        pystore.store('store3')
        
        # Delete all stores
        result = pystore.delete_stores()
        assert result is True
        
        # Verify all stores are deleted
        stores = pystore.list_stores()
        assert len(stores) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
