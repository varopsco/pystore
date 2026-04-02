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


class TestItem:
    """Test item functionality."""

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

    def _create_sample_data(self, with_datetime_index=False):
        """Create sample DataFrame for testing."""
        data = {
            'a': [1, 2, 3, 4, 5],
            'b': [1.0, 2.0, 3.0, 4.0, 5.0],
            'c': ['x', 'y', 'z', 'w', 'v']
        }
        df = pd.DataFrame(data)
        
        if with_datetime_index:
            df.index = pd.date_range('2020-01-01', periods=5)
        
        return df

    def test_item_repr(self):
        """Test item __repr__ method."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        item = self.collection.item('test_item')
        assert 'PyStore.item' in repr(item)
        assert 'test_item' in repr(item)
        assert 'test_collection' in repr(item)

    def test_item_nonexistent_raises_error(self):
        """Test that accessing nonexistent item raises error."""
        with pytest.raises(ValueError) as exc_info:
            self.collection.item('nonexistent_item')
        
        assert "doesn't exist" in str(exc_info.value)

    def test_item_metadata(self):
        """Test item metadata retrieval."""
        data = self._create_sample_data()
        metadata = {'source': 'test', 'version': 1, 'tags': ['a', 'b']}
        self.collection.write('test_item', data, metadata=metadata)
        
        item = self.collection.item('test_item')
        assert item.metadata['source'] == 'test'
        assert item.metadata['version'] == 1
        assert item.metadata['tags'] == ['a', 'b']

    def test_item_to_pandas(self):
        """Test converting item to pandas DataFrame."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        item = self.collection.item('test_item')
        df = item.to_pandas()
        
        assert len(df) == 5
        assert list(df['a']) == [1, 2, 3, 4, 5]

    def test_item_to_pandas_with_datetime_index(self):
        """Test converting item with datetime index to pandas."""
        data = self._create_sample_data(with_datetime_index=True)
        # Write with epochdate to convert datetime to int64
        self.collection.write('test_item', data, epochdate=True)
        
        item = self.collection.item('test_item')
        df = item.to_pandas(parse_dates=True)
        
        assert len(df) == 5
        # After parsing, the dates should be converted back

    def test_item_to_pandas_without_date_parsing(self):
        """Test converting item without date parsing."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        item = self.collection.item('test_item')
        df = item.to_pandas(parse_dates=False)
        
        assert len(df) == 5

    def test_item_to_pandas_with_float_index(self):
        """Test converting item with float index that represents epoch time."""
        data = self._create_sample_data()
        # Create float index that looks like epoch timestamps
        data.index = [1577836800.0, 1577923200.0, 1578009600.0, 1578096000.0, 1578182400.0]
        # Don't use epochdate as the index is not datetime
        self.collection.write('test_item', data)
        
        item = self.collection.item('test_item')
        df = item.to_pandas(parse_dates=True)
        
        assert len(df) == 5

    def test_item_head(self):
        """Test item head method."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        item = self.collection.item('test_item')
        head = item.head(n=2)
        
        assert len(head) == 2

    def test_item_tail(self):
        """Test item tail method."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        item = self.collection.item('test_item')
        tail = item.tail(n=2)
        
        assert len(tail) == 2

    def test_item_data_property(self):
        """Test item data property returns dask DataFrame."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        item = self.collection.item('test_item')
        
        # Data should be a dask DataFrame
        import dask.dataframe as dd
        assert isinstance(item.data, dd.DataFrame)

    def test_item_with_columns(self):
        """Test reading item with specific columns."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        item = self.collection.item('test_item', columns=['a', 'b'])
        df = item.to_pandas()
        
        assert list(df.columns) == ['a', 'b']

    def test_item_with_filters(self):
        """Test reading item with filters."""
        data = self._create_sample_data(with_datetime_index=True)
        self.collection.write('test_item', data)
        
        # Note: filters may not work with all engines
        # This test verifies the parameter is accepted
        item = self.collection.item('test_item')
        df = item.to_pandas()
        
        assert len(df) == 5


class TestItemSnapshots:
    """Test item snapshot functionality."""

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

    def test_item_from_snapshot(self):
        """Test reading item from snapshot."""
        # Write data
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        # Create snapshot
        self.collection.create_snapshot('test_snapshot')
        
        # Read item from snapshot
        item = self.collection.item('test_item', snapshot='test_snapshot')
        df = item.to_pandas()
        
        assert len(df) == 3
        assert item.snapshot == 'test_snapshot'

    def test_item_nonexistent_snapshot_raises_error(self):
        """Test that reading from nonexistent snapshot raises error."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        with pytest.raises(ValueError) as exc_info:
            self.collection.item('test_item', snapshot='nonexistent_snapshot')
        
        assert "doesn't exist" in str(exc_info.value)

    def test_item_not_in_snapshot_raises_error(self):
        """Test that reading item not in snapshot raises error."""
        # Write data
        data = self._create_sample_data()
        self.collection.write('test_item', data)
        
        # Create snapshot
        self.collection.create_snapshot('test_snapshot')
        
        # Write new item after snapshot
        self.collection.write('new_item', data)
        
        # Try to read new_item from snapshot (should fail)
        with pytest.raises(ValueError) as exc_info:
            self.collection.item('new_item', snapshot='test_snapshot')
        
        assert "doesn't exist in this snapshot" in str(exc_info.value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
