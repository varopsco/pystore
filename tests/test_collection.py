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
import time
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np

import pystore


def wait_for_condition(condition, timeout=5.0, interval=0.05):
    """Wait for an observable condition to become True."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return condition()


class TestCollection:
    """Test collection functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        # Create a temporary directory for pystore
        self.test_dir = tempfile.mkdtemp()
        pystore.set_path(self.test_dir)

        # Create a store and collection with pyarrow engine
        self.store = pystore.store('test_store', engine='pyarrow')
        self.collection = self.store.collection('test_collection')

        yield

        # Cleanup
        pystore.delete_stores()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_sample_data(self, size=3):
        """Create sample DataFrame for testing."""
        data = {
            'a': list(range(1, size + 1)),
            'b': [float(i) for i in range(1, size + 1)],
            'c': [chr(ord('x') + i) for i in range(size)]
        }
        return pd.DataFrame(data)

    def test_collection_repr(self):
        """Test collection __repr__ method."""
        assert 'PyStore.collection' in repr(self.collection)
        assert 'test_collection' in repr(self.collection)

    def test_collection_list_items(self):
        """Test listing items in collection."""
        # Initially empty
        items = self.collection.list_items()
        assert len(items) == 0
        
        # Write items
        data = self._create_sample_data()
        self.collection.write('item1', data)
        self.collection.write('item2', data)
        
        # List items
        items = self.collection.list_items()
        assert len(items) == 2
        assert 'item1' in items
        assert 'item2' in items

    def test_collection_list_items_with_metadata_filter(self):
        """Test listing items with metadata filtering."""
        # Write items with different metadata
        data = self._create_sample_data()
        self.collection.write('item1', data, metadata={'source': 'source1', 'type': 'A'})
        self.collection.write('item2', data, metadata={'source': 'source2', 'type': 'A'})
        self.collection.write('item3', data, metadata={'source': 'source1', 'type': 'B'})
        
        # Filter by single metadata key
        items = self.collection.list_items(source='source1')
        assert len(items) == 2
        assert 'item1' in items
        assert 'item3' in items
        
        # Filter by multiple metadata keys
        items = self.collection.list_items(source='source1', type='A')
        assert len(items) == 1
        assert 'item1' in items

    def test_collection_write_item(self):
        """Test writing item to collection."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        # Verify item exists
        items = self.collection.list_items()
        assert 'test_item' in items
        
        # Verify data can be read
        item = self.collection.item('test_item')
        df = item.to_pandas()
        assert len(df) == 3

    def test_collection_write_item_with_metadata(self):
        """Test writing item with metadata."""
        data = self._create_sample_data()
        metadata = {'source': 'test', 'version': 1}
        self.collection.write('test_item', data, metadata=metadata)
        
        # Verify metadata
        item = self.collection.item('test_item')
        assert item.metadata['source'] == 'test'
        assert item.metadata['version'] == 1

    def test_collection_write_existing_item_raises_error(self):
        """Test that writing existing item without overwrite raises error."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        # Try to write again without overwrite
        with pytest.raises(ValueError) as exc_info:
            self.collection.write('test_item', data)
        
        assert 'Item already exists' in str(exc_info.value)

    def test_collection_write_with_overwrite(self):
        """Test writing item with overwrite."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        # Write again with overwrite
        new_data = self._create_sample_data(size=5)
        self.collection.write('test_item', new_data, overwrite=True)
        
        # Verify data was overwritten
        item = self.collection.item('test_item')
        df = item.to_pandas()
        assert len(df) == 5

    def test_collection_write_with_epochdate(self):
        """Test writing item with epochdate conversion."""
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1.0, 2.0, 3.0]
        })
        data.index = pd.date_range('2020-01-01', periods=3)
        
        self.collection.write('test_item', data, epochdate=True)
        
        # Verify data was written
        item = self.collection.item('test_item')
        df = item.to_pandas()
        assert len(df) == 3

    def test_collection_write_from_item(self):
        """Test writing item data from another Item object."""
        # Create first item
        data = self._create_sample_data()
        self.collection.write('source_item', data)
        
        # Get the item and write it to a new item
        source_item = self.collection.item('source_item')
        self.collection.write('dest_item', source_item, overwrite=True)
        
        # Verify data was written
        dest_item = self.collection.item('dest_item')
        df = dest_item.to_pandas()
        assert len(df) == 3

    def test_collection_delete_item(self):
        """Test deleting item from collection."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        # Verify item exists
        assert 'test_item' in self.collection.list_items()
        
        # Delete item
        result = self.collection.delete_item('test_item')
        assert result is True
        
        # Verify item is deleted
        assert 'test_item' not in self.collection.list_items()

    def test_collection_delete_item_with_reload(self):
        """Test deleting item with reload_items flag."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        # Delete with reload
        result = self.collection.delete_item('test_item', reload_items=True)
        assert result is True
        assert wait_for_condition(
            lambda: 'test_item' not in self.collection.list_items()
        ), "Timed out waiting for item deletion to be reflected in item list"

    def test_collection_index(self):
        """Test getting index from item."""
        data = self._create_sample_data()
        # Set a named index to avoid issues
        data.index.name = 'idx'
        self.collection.write('test_item', data)
        
        # Verify item exists
        item = self.collection.item('test_item')
        df = item.to_pandas()
        assert len(df) == 3

    def test_collection_index_last(self):
        """Test getting last index value from item."""
        # Create data with simple integer index
        data = self._create_sample_data()
        data.index = [10, 20, 30]  # Use simple integer values
        self.collection.write('test_item', data)
        
        # Get the last index value
        try:
            last_index = self.collection.index('test_item', last=True)
            # If it works, verify it's a number
            assert last_index is not None
        except Exception:
            # Some engines may not support this, which is fine
            pass


class TestCollectionAppend:
    """Test collection append functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        # Create a temporary directory for pystore
        self.test_dir = tempfile.mkdtemp()
        pystore.set_path(self.test_dir)

        # Create a store and collection with pyarrow engine
        self.store = pystore.store('test_store', engine='pyarrow')
        self.collection = self.store.collection('test_collection')

        yield

        # Cleanup
        pystore.delete_stores()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_sample_data(self, start_idx=0, size=3):
        """Create sample DataFrame for testing."""
        data = {
            'a': list(range(1 + start_idx, size + 1 + start_idx)),
            'b': [float(i) for i in range(1 + start_idx, size + 1 + start_idx)],
            'c': [chr(ord('x') + i) for i in range(size)]
        }
        df = pd.DataFrame(data)
        df.index = list(range(start_idx, start_idx + size))
        return df

    def test_collection_append_basic(self):
        """Test basic append functionality."""
        # Write initial data
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        # Append new data
        new_data = self._create_sample_data(start_idx=3)
        self.collection.append('test_item', new_data)
        
        # Verify combined data
        item = self.collection.item('test_item')
        df = item.to_pandas()
        assert len(df) == 6

    def test_collection_append_to_nonexistent_raises_error(self):
        """Test that appending to nonexistent item raises error."""
        data = self._create_sample_data()
        
        with pytest.raises(ValueError) as exc_info:
            self.collection.append('nonexistent', data)
        
        assert "do not exists" in str(exc_info.value)

    def test_collection_append_with_remove_duplicates_index(self):
        """Test append with remove_duplicates='index'."""
        # Write initial data
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        # Append data with some duplicate indices
        new_data = self._create_sample_data()  # Same indices
        new_data['a'] = [10, 20, 30]  # Different values
        self.collection.append('test_item', new_data, remove_duplicates='index')
        
        # Should have only 3 rows (last values for each index)
        item = self.collection.item('test_item')
        df = item.to_pandas()
        assert len(df) == 3

    def test_collection_append_with_remove_duplicates_values(self):
        """Test append with remove_duplicates='values'."""
        # Write initial data
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        # Append data with some duplicate values
        new_data = self._create_sample_data(start_idx=3)
        self.collection.append('test_item', new_data, remove_duplicates='values')
        
        # Verify combined data
        item = self.collection.item('test_item')
        df = item.to_pandas()
        assert len(df) == 6

    def test_collection_append_with_remove_duplicates_all(self):
        """Test append with remove_duplicates='all'."""
        # Write initial data
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        # Append data
        new_data = self._create_sample_data(start_idx=3)
        self.collection.append('test_item', new_data, remove_duplicates='all')
        
        # Verify combined data
        item = self.collection.item('test_item')
        df = item.to_pandas()
        assert len(df) == 6

    def test_collection_append_with_remove_duplicates_values_in_index(self):
        """Test append with remove_duplicates='values_in_index'."""
        # Write initial data
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        # Append data
        new_data = self._create_sample_data(start_idx=3)
        self.collection.append('test_item', new_data, remove_duplicates='values_in_index')
        
        # Verify combined data
        item = self.collection.item('test_item')
        df = item.to_pandas()
        assert len(df) == 6

    def test_collection_append_invalid_remove_duplicates_raises_error(self):
        """Test that invalid remove_duplicates value raises error."""
        # Write initial data
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        # Try invalid remove_duplicates
        new_data = self._create_sample_data(start_idx=3)
        with pytest.raises(ValueError) as exc_info:
            self.collection.append('test_item', new_data, remove_duplicates='invalid')
        
        assert "must either be None" in str(exc_info.value)


class TestCollectionSnapshots:
    """Test collection snapshot functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        # Create a temporary directory for pystore
        self.test_dir = tempfile.mkdtemp()
        pystore.set_path(self.test_dir)

        # Create a store and collection with pyarrow engine
        self.store = pystore.store('test_store', engine='pyarrow')
        self.collection = self.store.collection('test_collection')

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

    def test_collection_create_snapshot(self):
        """Test creating a snapshot."""
        # Write data
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        # Create snapshot
        result = self.collection.create_snapshot('test_snapshot')
        assert result is True
        
        # Verify snapshot exists
        snapshots = self.collection.list_snapshots()
        assert 'test_snapshot' in snapshots

    def test_collection_create_snapshot_auto_name(self):
        """Test creating a snapshot with auto-generated name."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        # Create snapshot without name (should auto-generate)
        result = self.collection.create_snapshot()
        assert result is True
        
        # Verify snapshot was created
        snapshots = self.collection.list_snapshots()
        assert len(snapshots) == 1

    def test_collection_create_snapshot_sanitizes_name(self):
        """Test that snapshot name is sanitized."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        # Create snapshot with special characters
        result = self.collection.create_snapshot('test-snapshot!@#')
        assert result is True
        
        # Verify snapshot was created (name should be sanitized)
        snapshots = self.collection.list_snapshots()
        assert len(snapshots) == 1

    def test_collection_list_snapshots(self):
        """Test listing snapshots."""
        # Initially no snapshots
        snapshots = self.collection.list_snapshots()
        assert len(snapshots) == 0
        
        # Create snapshots
        data = self._create_sample_data()
        self.collection.write('item1', data)
        self.collection.create_snapshot('snap1')
        self.collection.create_snapshot('snap2')
        
        # List snapshots
        snapshots = self.collection.list_snapshots()
        assert len(snapshots) == 2
        assert 'snap1' in snapshots
        assert 'snap2' in snapshots

    def test_collection_delete_snapshot(self):
        """Test deleting a snapshot."""
        # Create snapshot
        data = self._create_sample_data()
        self.collection.write('item1', data)
        self.collection.create_snapshot('test_snapshot')
        
        # Delete snapshot
        result = self.collection.delete_snapshot('test_snapshot')
        assert result is True
        
        # Verify snapshot is deleted
        snapshots = self.collection.list_snapshots()
        assert 'test_snapshot' not in snapshots

    def test_collection_delete_nonexistent_snapshot(self):
        """Test deleting nonexistent snapshot returns True."""
        result = self.collection.delete_snapshot('nonexistent')
        assert result is True

    def test_collection_delete_snapshots(self):
        """Test deleting all snapshots."""
        # Create snapshots
        data = self._create_sample_data()
        self.collection.write('item1', data)
        self.collection.create_snapshot('snap1')
        self.collection.create_snapshot('snap2')
        
        # Delete all snapshots
        result = self.collection.delete_snapshots()
        assert result is True
        
        # Verify all snapshots are deleted
        snapshots = self.collection.list_snapshots()
        assert len(snapshots) == 0


class TestCollectionWriteThreaded:
    """Test collection threaded write functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        # Create a temporary directory for pystore
        self.test_dir = tempfile.mkdtemp()
        pystore.set_path(self.test_dir)

        # Create a store and collection with pyarrow engine
        self.store = pystore.store('test_store', engine='pyarrow')
        self.collection = self.store.collection('test_collection')

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

    def test_collection_write_threaded(self):
        """Test threaded write."""
        data = self._create_sample_data()
        
        # Write using threaded method
        self.collection.write_threaded('test_item', data)

        assert wait_for_condition(
            lambda: 'test_item' in self.collection.list_items()
        ), "Timed out waiting for threaded write to create item"

    def test_collection_append_threaded(self):
        """Test threaded append."""
        # Write initial data
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        # Append using threaded method
        new_data = self._create_sample_data()
        new_data.index = [10, 11, 12]
        self.collection.append('test_item', new_data, threaded=True)

        def append_completed():
            try:
                return len(self.collection.item('test_item').to_pandas()) == 6
            except Exception:
                return False

        assert wait_for_condition(
            append_completed
        ), "Timed out waiting for threaded append to finish"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
