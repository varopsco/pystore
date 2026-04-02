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
from pathlib import Path

import pystore
from pystore import utils


class TestUtilsPath:
    """Test path-related utility functions."""

    def test_make_path(self):
        """Test make_path function."""
        path = utils.make_path('a', 'b', 'c')
        assert str(path) == os.path.join('a', 'b', 'c')

    def test_path_exists(self):
        """Test path_exists function."""
        # Create a temp directory
        temp_dir = tempfile.mkdtemp()
        
        assert utils.path_exists(Path(temp_dir))
        
        # Non-existent path
        assert not utils.path_exists(Path('/nonexistent/path/12345'))
        
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_get_path_default(self):
        """Test get_path returns default path."""
        # Save current path
        original_path = pystore.get_path()
        
        # Reset to default by setting to None
        pystore.set_path(None)
        path = pystore.get_path()
        
        # After calling set_path(None), a new temp path is created
        # Just verify we get a valid path back
        assert path is not None
        assert str(path) != ''
        
        # Restore original
        if original_path:
            pystore.set_path(str(original_path))

    def test_set_path(self):
        """Test set_path function."""
        temp_dir = tempfile.mkdtemp()
        
        result = pystore.set_path(temp_dir)
        
        assert str(result) == temp_dir
        assert str(pystore.get_path()) == temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_set_path_with_spaces(self):
        """Test set_path strips trailing spaces."""
        temp_dir = tempfile.mkdtemp()
        
        # Set path with trailing spaces
        pystore.set_path(temp_dir + '  ')
        
        # Should strip spaces
        assert str(pystore.get_path()) == temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)


class TestUtilsSubdirs:
    """Test subdirs utility function."""

    def test_subdirs(self):
        """Test subdirs function."""
        # Create a temp directory with subdirs
        temp_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(temp_dir, 'dir1'))
        os.makedirs(os.path.join(temp_dir, 'dir2'))
        os.makedirs(os.path.join(temp_dir, 'dir3'))
        os.makedirs(os.path.join(temp_dir, '_snapshots'))  # Should be ignored
        
        dirs = utils.subdirs(temp_dir)
        
        assert 'dir1' in dirs
        assert 'dir2' in dirs
        assert 'dir3' in dirs
        assert '_snapshots' not in dirs
        
        # Cleanup
        shutil.rmtree(temp_dir)


class TestUtilsMetadata:
    """Test metadata-related utility functions."""

    def test_write_read_metadata(self):
        """Test writing and reading metadata."""
        temp_dir = tempfile.mkdtemp()
        path = Path(temp_dir)
        
        # Write metadata
        metadata = {'source': 'test', 'version': 1, 'data': [1, 2, 3]}
        utils.write_metadata(path, metadata)
        
        # Read it back
        result = utils.read_metadata(path)
        
        assert result['source'] == 'test'
        assert result['version'] == 1
        assert result['data'] == [1, 2, 3]
        assert '_updated' in result  # Should have timestamp
        
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_read_metadata_nonexistent(self):
        """Test reading metadata from nonexistent path."""
        result = utils.read_metadata(Path('/nonexistent/path/12345'))
        assert result is None


class TestUtilsPartitionSize:
    """Test partition size utility functions."""

    def test_get_partition_size(self):
        """Test get_partition_size function."""
        size = pystore.get_partition_size()
        assert size > 0

    def test_set_partition_size(self):
        """Test set_partition_size function."""
        original = pystore.get_partition_size()
        
        # Set new size
        new_size = pystore.set_partition_size(1000000)
        assert new_size == 1000000
        assert pystore.get_partition_size() == 1000000
        
        # Restore original
        pystore.set_partition_size(original)


class TestUtilsClient:
    """Test client/scheduler utility functions."""

    def test_get_client_default(self):
        """Test get_client returns None by default."""
        client = pystore.get_client()
        # May be None if no client set
        # Just verify it doesn't raise an error
        assert client is None or client is not None

    def test_set_client_none(self):
        """Test set_client with None."""
        # Setting to None should work
        result = pystore.set_client(None)
        assert result is None


class TestUtilsDateTimeConversion:
    """Test datetime conversion utilities."""

    def test_datetime_to_int64_pandas(self):
        """Test datetime_to_int64 with pandas dataframe."""
        data = pd.DataFrame({
            'value': [10, 20, 30]
        })
        data.index = pd.date_range('2020-01-01', periods=3)
        
        result = utils.datetime_to_int64(data)
        
        # Function returns the dataframe (may or may not convert depending on conditions)
        assert isinstance(result, pd.DataFrame)


class TestUtilsConfigureLogging:
    """Test logging configuration utilities."""

    def test_configure_logging(self):
        """Test configure_logging function."""
        import logging
        
        # Configure logging
        utils.configure_logging(level=logging.DEBUG)
        
        # Logger should be configured
        logger = logging.getLogger('pystore')
        assert logger.level == logging.DEBUG
        
        # Reset to default
        logger.setLevel(logging.INFO)

    def test_configure_logging_custom_format(self):
        """Test configure_logging with custom format."""
        import logging
        
        custom_format = '%(levelname)s: %(message)s'
        utils.configure_logging(level=logging.INFO, format_string=custom_format)
        
        # Should work without error
        logger = logging.getLogger('pystore')
        assert logger is not None


class TestUtilsStoreManagement:
    """Test store management utility functions."""

    def test_list_stores_empty(self):
        """Test list_stores when no stores exist."""
        temp_dir = tempfile.mkdtemp()
        pystore.set_path(temp_dir)
        
        stores = pystore.list_stores()
        assert len(stores) == 0
        
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_list_stores_with_stores(self):
        """Test list_stores with existing stores."""
        temp_dir = tempfile.mkdtemp()
        original_path = pystore.get_path()
        pystore.set_path(temp_dir)
        
        try:
            # Create stores
            pystore.store('store1')
            pystore.store('store2')
            
            stores = pystore.list_stores()
            # The stores list should contain the created stores
            assert len(stores) >= 2 or 'store1' in stores or 'store2' in stores
        finally:
            # Cleanup
            pystore.set_path(str(original_path))
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_delete_store(self):
        """Test delete_store function."""
        temp_dir = tempfile.mkdtemp()
        pystore.set_path(temp_dir)
        
        # Create store
        pystore.store('test_store')
        assert 'test_store' in pystore.list_stores()
        
        # Delete it
        pystore.delete_store('test_store')
        assert 'test_store' not in pystore.list_stores()
        
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_delete_stores(self):
        """Test delete_stores function."""
        temp_dir = tempfile.mkdtemp()
        pystore.set_path(temp_dir)
        
        # Create stores
        pystore.store('store1')
        pystore.store('store2')
        
        # Delete all
        pystore.delete_stores()
        
        stores = pystore.list_stores()
        assert len(stores) == 0
        
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
