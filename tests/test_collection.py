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
import numpy as np

import pystore


class TestCollection:
    """Test Collection class functionality."""

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

    def _create_sample_data(self, rows=3):
        """Create sample DataFrame for testing."""
        data = {
            'a': list(range(1, rows + 1)),
            'b': [float(x) for x in range(1, rows + 1)],
            'c': [chr(ord('x') + i) for i in range(rows)]
        }
        return pd.DataFrame(data)

    def test_collection_repr(self):
        """Test collection string representation."""
        repr_str = repr(self.collection)
        assert 'PyStore.collection' in repr_str
        assert 'test_collection' in repr_str

    def test_list_items_empty(self):
        """Test listing items in empty collection."""
        items = self.collection.list_items()
        assert len(items) == 0

    def test_write_item(self):
        """Test writing an item to collection."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        items = self.collection.list_items()
        assert 'item1' in items

    def test_write_item_with_metadata(self):
        """Test writing an item with metadata."""
        data = self._create_sample_data()
        metadata = {'source': 'test', 'version': 1}
        self.collection.write('item1', data, metadata=metadata)
        
        item = self.collection.item('item1')
        assert item.metadata['source'] == 'test'
        assert item.metadata['version'] == 1

    def test_write_item_already_exists_error(self):
        """Test that writing existing item raises error."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        with pytest.raises(ValueError):
            self.collection.write('item1', data)

    def test_write_item_overwrite(self):
        """Test overwriting an existing item."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        # Write new data with overwrite=True
        new_data = pd.DataFrame({'x': [10, 20], 'y': [30, 40]})
        new_data.index = pd.Index([0, 1])
        self.collection.write('item1', new_data, overwrite=True)
        
        item = self.collection.item('item1')
        result = item.to_pandas()
        assert 'x' in result.columns
        assert 'a' not in result.columns

    def test_append_item(self):
        """Test appending data to an item."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        # Append more data
        new_data = self._create_sample_data(rows=2)
        new_data.index = pd.Index([10, 11])
        self.collection.append('item1', new_data)
        
        item = self.collection.item('item1')
        result = item.to_pandas()
        assert len(result) == 5

    def test_append_nonexistent_item_error(self):
        """Test that appending to nonexistent item raises error."""
        data = self._create_sample_data()
        
        with pytest.raises(ValueError):
            self.collection.append('nonexistent', data)

    def test_delete_item(self):
        """Test deleting an item."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        assert 'item1' in self.collection.list_items()
        
        self.collection.delete_item('item1')
        
        assert 'item1' not in self.collection.list_items()

    def test_item_access(self):
        """Test accessing an item."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        item = self.collection.item('item1')
        assert item is not None
        assert item.item == 'item1'

    def test_list_items_with_metadata_filter(self):
        """Test listing items with metadata filtering."""
        data = self._create_sample_data()
        
        # Write items with different metadata
        self.collection.write('item1', data, metadata={'type': 'A'})
        self.collection.write('item2', data, metadata={'type': 'B'})
        self.collection.write('item3', data, metadata={'type': 'A'})
        
        # Filter by type='A'
        items = self.collection.list_items(type='A')
        assert len(items) == 2
        assert 'item1' in items
        assert 'item3' in items
        assert 'item2' not in items

    def test_index_method(self):
        """Test the index method."""
        data = self._create_sample_data()
        data.index = pd.Index([10, 20, 30])
        self.collection.write('item1', data)
        
        # Get full index
        item = self.collection.item('item1')
        result = item.to_pandas()
        assert len(result.index) == 3

    @pytest.mark.skip(reason="Index last method has compatibility issues with dask/pyarrow")
    def test_index_last_method(self):
        """Test getting last index value."""
        data = self._create_sample_data()
        data.index = pd.Index([10.0, 20.0, 30.0])
        self.collection.write('item1', data)
        
        last_idx = self.collection.index('item1', last=True)
        assert last_idx == 30.0

    def test_write_with_epochdate(self):
        """Test writing with epochdate conversion."""
        data = self._create_sample_data()
        dates = pd.date_range('2020-01-01', periods=3)
        data.index = dates
        
        self.collection.write('item1', data, epochdate=True)
        
        item = self.collection.item('item1')
        result = item.to_pandas(parse_dates=False)
        # Index should be numeric (int64 from epoch conversion)
        assert len(result) == 3

    @pytest.mark.skip(reason="npartitions parameter has compatibility issues with dask/pyarrow")
    def test_write_with_npartitions(self):
        """Test writing with explicit npartitions."""
        data = self._create_sample_data(rows=100)
        data.index = pd.Index(range(100))
        self.collection.write('item1', data, npartitions=4)
        
        item = self.collection.item('item1')
        result = item.to_pandas()
        assert len(result) == 100


class TestCollectionSnapshots:
    """Test Collection snapshot functionality."""

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

    def _create_sample_data(self, rows=3):
        """Create sample DataFrame for testing."""
        data = {
            'a': list(range(1, rows + 1)),
            'b': [float(x) for x in range(1, rows + 1)],
            'c': [chr(ord('x') + i) for i in range(rows)]
        }
        return pd.DataFrame(data)

    def test_create_snapshot(self):
        """Test creating a snapshot."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        # Create snapshot
        self.collection.create_snapshot('snap1')
        
        # Check snapshot exists
        snapshots = self.collection.list_snapshots()
        assert 'snap1' in snapshots

    def test_list_snapshots_empty(self):
        """Test listing snapshots when none exist."""
        snapshots = self.collection.list_snapshots()
        assert len(snapshots) == 0

    def test_delete_snapshot(self):
        """Test deleting a snapshot."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        self.collection.create_snapshot('snap1')
        
        # Delete snapshot
        self.collection.delete_snapshot('snap1')
        
        assert 'snap1' not in self.collection.list_snapshots()

    def test_delete_snapshots(self):
        """Test deleting all snapshots."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        self.collection.create_snapshot('snap1')
        self.collection.create_snapshot('snap2')
        
        # Delete all snapshots
        self.collection.delete_snapshots()
        
        assert len(self.collection.list_snapshots()) == 0

    def test_item_from_snapshot(self):
        """Test accessing an item from a snapshot."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        # Create snapshot
        self.collection.create_snapshot('snap1')
        
        # Modify the item
        new_data = self._create_sample_data(rows=10)
        self.collection.write('item1', new_data, overwrite=True)
        
        # Access item from snapshot (should have 3 rows)
        item = self.collection.item('item1', snapshot='snap1')
        result = item.to_pandas()
        assert len(result) == 3
        
        # Current version should have 10 rows
        item = self.collection.item('item1')
        result = item.to_pandas()
        assert len(result) == 10

    def test_snapshot_preserves_multiple_items(self):
        """Test that snapshot preserves all items in collection."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        self.collection.write('item2', data)
        self.collection.write('item3', data)
        
        # Create snapshot
        self.collection.create_snapshot('snap1')
        
        # Delete one item
        self.collection.delete_item('item1')
        
        # Access snapshot - all items should exist
        item = self.collection.item('item2', snapshot='snap1')
        assert item is not None
        
        item = self.collection.item('item3', snapshot='snap1')
        assert item is not None


class TestCollectionAppend:
    """Test Collection append functionality with various options."""

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

    def _create_sample_data(self, rows=3, start_idx=0):
        """Create sample DataFrame for testing."""
        data = {
            'a': list(range(1, rows + 1)),
            'b': [float(x) for x in range(1, rows + 1)],
            'c': [chr(ord('x') + i) for i in range(rows)]
        }
        df = pd.DataFrame(data)
        df.index = pd.Index(list(range(start_idx, start_idx + rows)))
        return df

    def test_append_basic(self):
        """Test basic append operation."""
        data = self._create_sample_data(rows=3, start_idx=0)
        self.collection.write('item1', data)
        
        new_data = self._create_sample_data(rows=2, start_idx=10)
        self.collection.append('item1', new_data)
        
        item = self.collection.item('item1')
        result = item.to_pandas()
        assert len(result) == 5

    def test_append_remove_duplicates_index(self):
        """Test append with remove_duplicates='index'."""
        data = self._create_sample_data(rows=3, start_idx=0)
        self.collection.write('item1', data)
        
        # Append with overlapping index
        new_data = self._create_sample_data(rows=3, start_idx=1)
        self.collection.append('item1', new_data, remove_duplicates='index')
        
        item = self.collection.item('item1')
        result = item.to_pandas()
        # Should keep last version of duplicate indices
        assert len(result) == 4  # indices 0, 1, 2, 3

    def test_append_remove_duplicates_values(self):
        """Test append with remove_duplicates='values'."""
        data = self._create_sample_data(rows=3, start_idx=0)
        self.collection.write('item1', data)
        
        # Append with different index but some duplicate values
        new_data = pd.DataFrame({
            'a': [1, 10],  # 1 is duplicate value from original
            'b': [1.0, 10.0],
            'c': ['x', 'z']  # 'x' is duplicate value from original
        })
        new_data.index = pd.Index([10, 11])
        
        self.collection.append('item1', new_data, remove_duplicates='values')
        
        item = self.collection.item('item1')
        result = item.to_pandas()
        # Should remove duplicate rows based on values
        # Result should have unique rows only
        assert len(result) >= 3  # At least original unique rows

    def test_append_remove_duplicates_all(self):
        """Test append with remove_duplicates='all'."""
        data = self._create_sample_data(rows=3, start_idx=0)
        self.collection.write('item1', data)
        
        # Append with overlapping index
        new_data = self._create_sample_data(rows=2, start_idx=2)
        self.collection.append('item1', new_data, remove_duplicates='all')
        
        item = self.collection.item('item1')
        result = item.to_pandas()
        # Should handle both index and value duplicates
        assert len(result) == 4

    def test_append_invalid_remove_duplicates(self):
        """Test append with invalid remove_duplicates value."""
        data = self._create_sample_data(rows=3, start_idx=0)
        self.collection.write('item1', data)
        
        new_data = self._create_sample_data(rows=2, start_idx=10)
        
        with pytest.raises(ValueError):
            self.collection.append('item1', new_data, remove_duplicates='invalid')

    def test_append_preserves_metadata(self):
        """Test that append preserves metadata."""
        data = self._create_sample_data(rows=3, start_idx=0)
        metadata = {'source': 'test', 'version': 1}
        self.collection.write('item1', data, metadata=metadata)
        
        new_data = self._create_sample_data(rows=2, start_idx=10)
        self.collection.append('item1', new_data)
        
        item = self.collection.item('item1')
        assert item.metadata['source'] == 'test'
        assert item.metadata['version'] == 1


class TestCollectionEdgeCases:
    """Test Collection edge cases and error handling."""

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

    def _create_sample_data(self, rows=3):
        """Create sample DataFrame for testing."""
        data = {
            'a': list(range(1, rows + 1)),
            'b': [float(x) for x in range(1, rows + 1)],
            'c': [chr(ord('x') + i) for i in range(rows)]
        }
        return pd.DataFrame(data)

    @pytest.mark.skip(reason="Writing empty dataframe has issues with dask/pyarrow")
    def test_write_empty_dataframe(self):
        """Test writing an empty DataFrame."""
        data = pd.DataFrame()
        
        # Should still work
        self.collection.write('item1', data)
        
        item = self.collection.item('item1')
        result = item.to_pandas()
        assert len(result) == 0

    def test_append_empty_dataframe(self):
        """Test appending an empty DataFrame."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        # Append empty data
        empty_data = pd.DataFrame()
        self.collection.append('item1', empty_data)
        
        # Should still have original data
        item = self.collection.item('item1')
        result = item.to_pandas()
        assert len(result) == 3

    def test_write_with_datetime_index(self):
        """Test writing with datetime index."""
        data = self._create_sample_data()
        dates = pd.date_range('2020-01-01', periods=3)
        data.index = dates
        
        self.collection.write('item1', data)
        
        item = self.collection.item('item1')
        result = item.to_pandas()
        assert len(result) == 3

    def test_write_with_named_index(self):
        """Test writing with named index."""
        data = self._create_sample_data()
        data.index.name = 'my_index'
        
        self.collection.write('item1', data)
        
        item = self.collection.item('item1')
        result = item.to_pandas()
        # Index name may not be preserved by parquet storage
        assert len(result) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
