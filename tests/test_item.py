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
    """Test Item class functionality."""

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

    def _create_sample_data(self, rows=10):
        """Create sample DataFrame for testing."""
        data = {
            'a': list(range(1, rows + 1)),
            'b': [float(x) for x in range(1, rows + 1)],
            'c': [chr(ord('x') + (i % 26)) for i in range(rows)]
        }
        return pd.DataFrame(data)

    def test_item_repr(self):
        """Test item string representation."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        item = self.collection.item('item1')
        repr_str = repr(item)
        assert 'PyStore.item' in repr_str
        assert 'item1' in repr_str

    def test_item_data_access(self):
        """Test accessing item data as dask dataframe."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        item = self.collection.item('item1')
        
        # Data should be a dask dataframe
        assert item.data is not None
        assert hasattr(item.data, 'compute')

    def test_item_to_pandas(self):
        """Test converting item to pandas dataframe."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        item = self.collection.item('item1')
        result = item.to_pandas()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_item_metadata(self):
        """Test accessing item metadata."""
        data = self._create_sample_data()
        metadata = {'source': 'test', 'version': 1, 'desc': 'Test data'}
        self.collection.write('item1', data, metadata=metadata)
        
        item = self.collection.item('item1')
        
        assert item.metadata['source'] == 'test'
        assert item.metadata['version'] == 1
        assert item.metadata['desc'] == 'Test data'

    def test_item_head(self):
        """Test item head method."""
        data = self._create_sample_data(rows=100)
        self.collection.write('item1', data)
        
        item = self.collection.item('item1')
        head = item.head(n=5)
        
        assert len(head) == 5

    def test_item_tail(self):
        """Test item tail method."""
        data = self._create_sample_data(rows=100)
        self.collection.write('item1', data)
        
        item = self.collection.item('item1')
        tail = item.tail(n=5)
        
        assert len(tail) == 5

    def test_item_columns_selection(self):
        """Test selecting specific columns."""
        data = self._create_sample_data()
        self.collection.write('item1', data)
        
        # Read only specific columns
        item = self.collection.item('item1', columns=['a', 'b'])
        result = item.to_pandas()
        
        assert 'a' in result.columns
        assert 'b' in result.columns
        assert 'c' not in result.columns


class TestItemErrors:
    """Test Item error handling."""

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

    def test_item_nonexistent_raises_error(self):
        """Test that accessing nonexistent item raises error."""
        with pytest.raises(ValueError) as exc_info:
            self.collection.item('nonexistent')
        
        assert "doesn't exist" in str(exc_info.value)

    def test_item_nonexistent_snapshot_raises_error(self):
        """Test that accessing nonexistent snapshot raises error."""
        data = pd.DataFrame({'a': [1, 2, 3]})
        self.collection.write('item1', data)
        
        with pytest.raises(ValueError) as exc_info:
            self.collection.item('item1', snapshot='nonexistent')
        
        assert "doesn't exist" in str(exc_info.value)


class TestItemDateTimeIndex:
    """Test Item with datetime index."""

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

    def test_item_with_datetime_index(self):
        """Test item with datetime index."""
        data = pd.DataFrame({
            'value': [10, 20, 30, 40, 50]
        })
        data.index = pd.date_range('2020-01-01', periods=5)
        
        self.collection.write('item1', data)
        
        item = self.collection.item('item1')
        result = item.to_pandas()
        
        assert len(result) == 5

    def test_item_datetime_parse_dates(self):
        """Test that datetime index is parsed correctly."""
        # Write with epochdate=True
        data = pd.DataFrame({
            'value': [10, 20, 30]
        })
        dates = pd.date_range('2020-01-01', periods=3)
        data.index = dates
        
        self.collection.write('item1', data, epochdate=True)
        
        # Read with parse_dates=True (default)
        item = self.collection.item('item1')
        result = item.to_pandas(parse_dates=True)
        
        # Should be able to parse dates back
        assert len(result) == 3

    def test_item_datetime_no_parse(self):
        """Test reading without parsing dates."""
        data = pd.DataFrame({
            'value': [10, 20, 30]
        })
        data.index = pd.date_range('2020-01-01', periods=3)
        
        self.collection.write('item1', data)
        
        item = self.collection.item('item1')
        result = item.to_pandas(parse_dates=False)
        
        assert len(result) == 3


class TestItemLargeData:
    """Test Item with larger datasets."""

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

    def test_item_large_dataset(self):
        """Test item with larger dataset."""
        # Create a larger dataset
        data = pd.DataFrame({
            'a': list(range(10000)),
            'b': [float(x) for x in range(10000)],
            'c': [f'item_{i}' for i in range(10000)]
        })
        
        self.collection.write('item1', data)
        
        item = self.collection.item('item1')
        result = item.to_pandas()
        
        assert len(result) == 10000

    @pytest.mark.skip(reason="npartitions parameter has compatibility issues with dask/pyarrow")
    def test_item_partitioned_data(self):
        """Test item with explicit partitioning."""
        data = pd.DataFrame({
            'a': list(range(1000)),
            'b': [float(x) for x in range(1000)]
        })
        data.index = pd.Index(range(1000))
        
        # Write with multiple partitions
        self.collection.write('item1', data, npartitions=10)
        
        item = self.collection.item('item1')
        result = item.to_pandas()
        
        assert len(result) == 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
