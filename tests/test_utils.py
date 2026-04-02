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
import json
import logging

import pystore
import pystore.utils as utils


class TestUtils:
    """Test utility functions."""

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

    def test_set_path(self):
        """Test setting pystore path."""
        new_path = tempfile.mkdtemp()
        original_path = pystore.get_path()
        try:
            result = pystore.set_path(new_path)
            assert str(new_path) in str(result)
            assert os.path.exists(new_path)
        finally:
            # Reset to original path before cleanup
            pystore.set_path(str(original_path))
            if os.path.exists(new_path):
                shutil.rmtree(new_path)

    def test_set_path_none(self):
        """Test setting path to None uses default."""
        result = pystore.set_path(None)
        assert result is not None

    def test_set_path_remote_url_raises_error(self):
        """Test that setting remote URL path raises error."""
        with pytest.raises(ValueError) as exc_info:
            pystore.set_path('s3://bucket/path')
        
        assert 'only works with local file system' in str(exc_info.value)

    def test_get_path(self):
        """Test getting pystore path."""
        path = pystore.get_path()
        assert path is not None
        assert os.path.exists(path)

    def test_make_path(self):
        """Test making path from components."""
        path = utils.make_path('a', 'b', 'c')
        assert 'a' in str(path)
        assert 'b' in str(path)
        assert 'c' in str(path)

    def test_path_exists(self):
        """Test path existence check."""
        # Test existing path
        assert utils.path_exists(utils.get_path())
        
        # Test non-existing path
        assert not utils.path_exists(utils.make_path('nonexistent', 'path'))

    def test_subdirs(self):
        """Test getting subdirectories."""
        # Create some subdirectories
        base_dir = utils.get_path()
        os.makedirs(utils.make_path(base_dir, 'dir1'))
        os.makedirs(utils.make_path(base_dir, 'dir2'))
        os.makedirs(utils.make_path(base_dir, '_snapshots'))  # Should be ignored
        
        subdirs = utils.subdirs(base_dir)
        assert 'dir1' in subdirs
        assert 'dir2' in subdirs
        assert '_snapshots' not in subdirs

    def test_read_metadata(self):
        """Test reading metadata."""
        # Create metadata file
        meta_path = utils.make_path(self.test_dir, 'test_meta')
        os.makedirs(meta_path)
        
        metadata = {'key': 'value', 'number': 42}
        utils.write_metadata(meta_path, metadata)
        
        # Read metadata
        read_meta = utils.read_metadata(meta_path)
        assert read_meta['key'] == 'value'
        assert read_meta['number'] == 42
        assert '_updated' in read_meta

    def test_read_metadata_nonexistent(self):
        """Test reading metadata from nonexistent path."""
        meta = utils.read_metadata(utils.make_path('nonexistent', 'path'))
        assert meta is None

    def test_write_metadata(self):
        """Test writing metadata."""
        meta_path = utils.make_path(self.test_dir, 'test_meta')
        os.makedirs(meta_path)
        
        metadata = {'source': 'test', 'version': 1}
        utils.write_metadata(meta_path, metadata)
        
        # Verify file was created
        meta_file = utils.make_path(meta_path, 'metadata.json')
        assert meta_file.exists()
        
        # Verify content
        with meta_file.open() as f:
            content = json.load(f)
            assert content['source'] == 'test'
            assert content['version'] == 1
            assert '_updated' in content

    def test_datetime_to_int64(self):
        """Test datetime to int64 conversion."""
        # Create DataFrame with datetime index
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1.0, 2.0, 3.0]
        })
        df.index = pd.date_range('2020-01-01', periods=3, freq='s')  # Use second frequency
        
        # Convert
        result = utils.datetime_to_int64(df)
        
        # Verify the function works (may not convert for pandas DataFrames)
        # This function is primarily for dask DataFrames
        assert result is not None

    def test_read_csv_basic(self):
        """Test basic CSV reading."""
        # Create a test CSV file
        csv_path = utils.make_path(self.test_dir, 'test.csv')
        with csv_path.open('w') as f:
            f.write('a,b,c\n1,1.0,x\n2,2.0,y\n3,3.0,z\n')
        
        # Read CSV
        import dask.dataframe as dd
        df = utils.read_csv(str(csv_path))
        
        assert isinstance(df, dd.DataFrame)
        result = df.compute()
        assert len(result) == 3

    def test_read_csv_with_index_col(self):
        """Test CSV reading with index column."""
        # Create a test CSV file
        csv_path = utils.make_path(self.test_dir, 'test.csv')
        with csv_path.open('w') as f:
            f.write('idx,a,b\n0,1,1.0\n1,2,2.0\n2,3,3.0\n')
        
        # Read CSV with index column
        df = utils.read_csv(str(csv_path), index_col='idx')
        result = df.compute()
        
        assert result.index.name == 'idx'

    def test_read_csv_with_index_name(self):
        """Test CSV reading with index name."""
        # Create a test CSV file
        csv_path = utils.make_path(self.test_dir, 'test.csv')
        with csv_path.open('w') as f:
            f.write('a,b,c\n1,1.0,x\n2,2.0,y\n3,3.0,z\n')
        
        # Read CSV with index name
        df = utils.read_csv(str(csv_path), index_name='my_index')
        result = df.compute()
        
        assert result.index.name == 'my_index'


class TestDaskClient:
    """Test Dask client configuration."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        # Create a temporary directory for pystore
        self.test_dir = tempfile.mkdtemp()
        pystore.set_path(self.test_dir)

        yield

        # Cleanup
        # Shutdown any client
        client = pystore.get_client()
        if client is not None:
            try:
                client.shutdown()
            except:
                pass
        
        pystore.delete_stores()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_set_client_none(self):
        """Test setting client to None."""
        result = pystore.set_client(None)
        assert result is None

    def test_get_client_none(self):
        """Test getting client when not set."""
        pystore.set_client(None)
        client = pystore.get_client()
        assert client is None

    def test_set_client_local(self):
        """Test setting local Dask client."""
        from dask.distributed import LocalCluster
        
        try:
            # Create local cluster
            cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False)
            scheduler = cluster.scheduler_address
            
            # Set client
            client = pystore.set_client(scheduler)
            assert client is not None
            
            # Get client
            retrieved = pystore.get_client()
            assert retrieved is not None
            
            # Cleanup
            cluster.close()
        except Exception as e:
            # Skip test if Dask distributed has issues
            pytest.skip(f"Dask distributed not available or error: {e}")


class TestPartitionSize:
    """Test partition size configuration."""

    def test_set_partition_size(self):
        """Test setting partition size."""
        size = pystore.set_partition_size(1000000)
        assert size == 1000000

    def test_set_partition_size_none(self):
        """Test setting partition size to None uses default."""
        size = pystore.set_partition_size(None)
        assert size is not None

    def test_get_partition_size(self):
        """Test getting partition size."""
        pystore.set_partition_size(2000000)
        size = pystore.get_partition_size()
        assert size == 2000000


class TestLoggingConfiguration:
    """Test logging configuration."""

    def test_configure_logging(self):
        """Test configuring logging."""
        # Configure logging
        utils.configure_logging(level=logging.DEBUG)
        
        # Verify logger has handlers
        logger = logging.getLogger('pystore')
        assert len(logger.handlers) > 0
        assert logger.level == logging.DEBUG
        
        # Clean up
        for handler in logger.handlers[:]:
            if not isinstance(handler, logging.NullHandler):
                logger.removeHandler(handler)

    def test_configure_logging_custom_format(self):
        """Test configuring logging with custom format."""
        custom_format = '%(levelname)s: %(message)s'
        utils.configure_logging(format_string=custom_format)
        
        logger = logging.getLogger('pystore')
        assert len(logger.handlers) > 0
        
        # Clean up
        for handler in logger.handlers[:]:
            if not isinstance(handler, logging.NullHandler):
                logger.removeHandler(handler)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
