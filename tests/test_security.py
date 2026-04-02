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

"""
Tests for security hardening - path traversal protection and input sanitization.
"""

import os
import shutil
import tempfile
import pytest
import pandas as pd
from pathlib import Path

import pystore
from pystore.utils import (
    validate_path_component,
    validate_path_within_directory,
    PathSecurityError,
    make_path
)


class TestPathValidation:
    """Test path component validation."""

    def test_valid_simple_names(self):
        """Test that valid simple names pass validation."""
        valid_names = [
            'item1',
            'collection',
            'my_data',
            'test-store',
            'NASDAQ',
            'AAPL',
            'data123',
        ]
        for name in valid_names:
            result = validate_path_component(name)
            assert result == name

    def test_path_traversal_dots_blocked(self):
        """Test that path traversal with '..' is blocked."""
        malicious_names = [
            '../etc/passwd',
            '..\\windows\\system32',
            'foo/../../../etc/passwd',
            'foo..bar',  # Should be blocked - contains '..'
        ]
        for name in malicious_names:
            with pytest.raises(PathSecurityError) as exc_info:
                validate_path_component(name)
            assert "path traversal" in str(exc_info.value).lower()

    def test_path_separators_blocked(self):
        """Test that path separators are blocked."""
        malicious_names = [
            'foo/bar',
            'foo\\bar',
            '/etc/passwd',
            'C:\\Windows',
        ]
        for name in malicious_names:
            with pytest.raises(PathSecurityError) as exc_info:
                validate_path_component(name)
            assert "path separator" in str(exc_info.value).lower() or \
                   "absolute path" in str(exc_info.value).lower()

    def test_null_bytes_blocked(self):
        """Test that null bytes are blocked."""
        malicious_names = [
            'item\x00.txt',
            'coll\0ection',
        ]
        for name in malicious_names:
            with pytest.raises(PathSecurityError) as exc_info:
                validate_path_component(name)
            assert "null" in str(exc_info.value).lower()

    def test_shell_characters_blocked(self):
        """Test that dangerous shell characters are blocked."""
        dangerous_names = [
            'item$(whoami)',
            'item`id`',
            'item&exit',
            'item|cat',
            'item<file',
            'item>file',
            'item(cmd)',
        ]
        for name in dangerous_names:
            with pytest.raises(PathSecurityError) as exc_info:
                validate_path_component(name)
            # Could be blocked for dangerous char OR other security reasons (like path separators)
            error_msg = str(exc_info.value).lower()
            assert "dangerous" in error_msg or "separator" in error_msg or "traversal" in error_msg

    def test_shell_characters_with_separators_blocked(self):
        """Test that shell characters combined with path separators are blocked."""
        # These contain both shell chars AND path separators
        dangerous_names = [
            'item;rm -rf /',  # contains both ; and /
            'item|cat /etc/passwd',  # contains both | and /
        ]
        for name in dangerous_names:
            with pytest.raises(PathSecurityError):
                validate_path_component(name)

    def test_whitespace_handling(self):
        """Test whitespace handling."""
        # Leading/trailing whitespace should be blocked
        with pytest.raises(PathSecurityError):
            validate_path_component(' item ')

        with pytest.raises(PathSecurityError):
            validate_path_component('item ')

        with pytest.raises(PathSecurityError):
            validate_path_component(' item')

    def test_empty_component_blocked(self):
        """Test that empty components are blocked."""
        with pytest.raises(PathSecurityError):
            validate_path_component('')

        with pytest.raises(PathSecurityError):
            validate_path_component('   ')

        with pytest.raises(PathSecurityError):
            validate_path_component(None)

    def test_path_object_accepted(self):
        """Test that Path objects are accepted and converted."""
        result = validate_path_component(Path('valid_name'))
        assert result == 'valid_name'

    def test_absolute_path_blocked(self):
        """Test that absolute paths are blocked."""
        with pytest.raises(PathSecurityError) as exc_info:
            validate_path_component('/etc/passwd')
        assert "absolute path" in str(exc_info.value).lower()

    def test_windows_absolute_path_blocked(self):
        """Test that Windows absolute paths are blocked."""
        with pytest.raises(PathSecurityError) as exc_info:
            validate_path_component('C:\\Windows\\System32')
        assert "absolute path" in str(exc_info.value).lower()

    def test_type_validation(self):
        """Test that invalid types raise appropriate errors."""
        with pytest.raises(TypeError):
            validate_path_component(123)

        with pytest.raises(TypeError):
            validate_path_component(['list'])

    def test_allow_path_separators_flag(self):
        """Test the allow_path_separators flag."""
        # Without flag, path separators are blocked
        with pytest.raises(PathSecurityError):
            validate_path_component('foo/bar')

        # With flag, path separators are allowed (but still validated for other issues)
        result = validate_path_component('foo/bar', allow_path_separators=True)
        assert result == 'foo/bar'


class TestPathWithinDirectory:
    """Test validation that paths stay within allowed directories."""

    def test_valid_subpath_accepted(self):
        """Test that valid subpaths are accepted."""
        with tempfile.TemporaryDirectory() as base_dir:
            subdir = os.path.join(base_dir, 'subdir')
            os.makedirs(subdir)

            result = validate_path_within_directory(subdir, base_dir)
            assert str(result).startswith(str(Path(base_dir).resolve()))

    def test_path_traversal_blocked(self):
        """Test that path traversal is blocked."""
        with tempfile.TemporaryDirectory() as base_dir:
            # Try to escape the base directory
            escape_path = os.path.join(base_dir, '..', 'etc', 'passwd')

            with pytest.raises(PathSecurityError) as exc_info:
                validate_path_within_directory(escape_path, base_dir)
            assert "outside" in str(exc_info.value).lower()

    def test_absolute_path_outside_blocked(self):
        """Test that absolute paths outside base are blocked."""
        with tempfile.TemporaryDirectory() as base_dir:
            with pytest.raises(PathSecurityError):
                validate_path_within_directory('/etc/passwd', base_dir)

    def test_none_path_rejected(self):
        """Test that None paths are rejected."""
        with pytest.raises(PathSecurityError):
            validate_path_within_directory(None, '/tmp')

        with pytest.raises(PathSecurityError):
            validate_path_within_directory('/tmp', None)


class TestMakePath:
    """Test the make_path function with validation."""

    def test_make_path_with_valid_components(self):
        """Test make_path with valid components."""
        result = make_path('base', 'collection', 'item')
        assert str(result) == 'base/collection/item' or \
               str(result) == 'base\\collection\\item'  # Windows

    def test_make_path_validates_components(self):
        """Test that make_path validates components when validate=True."""
        with pytest.raises(PathSecurityError):
            make_path('base', '../escape', 'item', validate=True)

    def test_make_path_skip_validation(self):
        """Test that validation can be skipped."""
        # This should not raise when validation is disabled
        result = make_path('base', '../escape', 'item', validate=False)
        assert result is not None

    def test_make_path_empty(self):
        """Test make_path with no arguments."""
        result = make_path()
        assert result == Path()


class TestStoreSecurity:
    """Test security at the store level."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        pystore.set_path(self.test_dir)
        yield
        # Cleanup
        pystore.delete_stores()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_malicious_store_name_rejected(self):
        """Test that malicious store names are rejected."""
        malicious_names = [
            '../escape',
            'store/../../../etc',
            'store$name',
            'store;malicious',
        ]
        for name in malicious_names:
            with pytest.raises((PathSecurityError, ValueError)) as exc_info:
                pystore.store(name)
            # Should mention invalid or security

    def test_malicious_collection_name_rejected(self):
        """Test that malicious collection names are rejected."""
        store = pystore.store('test_store')

        malicious_names = [
            '../escape',
            'collection/../../../etc',
            'collection$name',
            'collection;malicious',
        ]
        for name in malicious_names:
            with pytest.raises((PathSecurityError, ValueError)) as exc_info:
                store.collection(name)

    def test_delete_store_path_traversal_rejected(self):
        """Test that delete_store cannot delete paths outside the store root."""
        outside_dir = os.path.join(
            os.path.dirname(self.test_dir),
            'outside_' + os.path.basename(self.test_dir)
        )
        os.makedirs(outside_dir, exist_ok=True)

        try:
            with pytest.raises((PathSecurityError, ValueError)):
                pystore.delete_store('../' + os.path.basename(outside_dir))
            assert os.path.isdir(outside_dir)
        finally:
            if os.path.exists(outside_dir):
                shutil.rmtree(outside_dir)


class TestCollectionSecurity:
    """Test security at the collection level."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        pystore.set_path(self.test_dir)
        self.store = pystore.store('test_store', engine='pyarrow')
        self.collection = self.store.collection('test_collection')
        yield
        # Cleanup
        pystore.delete_stores()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_sample_data(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1.0, 2.0, 3.0],
            'c': ['x', 'y', 'z']
        })

    def test_malicious_item_name_rejected_in_write(self):
        """Test that malicious item names are rejected in write."""
        malicious_names = [
            '../escape',
            'item/../../../etc',
            'item$name',
            'item;malicious',
        ]
        data = self._create_sample_data()
        for name in malicious_names:
            with pytest.raises((PathSecurityError, ValueError)):
                self.collection.write(name, data)

    def test_malicious_item_name_rejected_in_read(self):
        """Test that malicious item names are rejected in read."""
        malicious_names = [
            '../escape',
            'item/../../../etc',
            'item$name',
            'item;malicious',
        ]
        for name in malicious_names:
            with pytest.raises((PathSecurityError, ValueError)):
                self.collection.item(name)

    def test_malicious_item_name_rejected_in_delete(self):
        """Test that malicious item names are rejected in delete."""
        malicious_names = [
            '../escape',
            'item/../../../etc',
        ]
        for name in malicious_names:
            with pytest.raises((PathSecurityError, ValueError)):
                self.collection.delete_item(name)

    def test_malicious_snapshot_name_rejected(self):
        """Test that malicious snapshot names are rejected."""
        # First write some data
        data = self._create_sample_data()
        self.collection.write('test_item', data)

        # Names with path separators should be rejected
        with pytest.raises((PathSecurityError, ValueError)):
            self.collection.create_snapshot('snap/../../../etc')

        # Names that become empty after sanitization should be rejected
        with pytest.raises((PathSecurityError, ValueError)):
            self.collection.create_snapshot('$$$')

    def test_cannot_escape_datastore_via_item_name(self):
        """Test that we cannot escape the datastore via item name."""
        # This is the critical security test
        # Try to write to a path outside the datastore
        with pytest.raises((PathSecurityError, ValueError)):
            self.collection.write('../../../escaped', self._create_sample_data())

    def test_snapshot_valid_names(self):
        """Test that valid snapshot names are preserved and usable."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)

        # Valid characters should work
        self.collection.create_snapshot('valid_snapshot')
        assert 'valid_snapshot' in self.collection.list_snapshots()

        # Dots/underscores/numbers should work
        self.collection.create_snapshot('valid.name_123')
        assert 'valid.name_123' in self.collection.list_snapshots()

    def test_snapshot_name_round_trip_with_hyphen(self):
        """Test snapshot names are not silently rewritten."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)

        self.collection.create_snapshot('snap-1')
        assert 'snap-1' in self.collection.list_snapshots()

        item = self.collection.item('test_item', snapshot='snap-1')
        assert len(item.to_pandas()) == 3

    def test_snapshot_names_do_not_collide(self):
        """Test distinct snapshot names remain distinct."""
        data = self._create_sample_data()
        self.collection.write('test_item', data)

        self.collection.create_snapshot('snap1')
        self.collection.create_snapshot('snap-1')

        snapshots = self.collection.list_snapshots()
        assert 'snap1' in snapshots
        assert 'snap-1' in snapshots

    def test_path_traversal_in_rename_rejected(self):
        """Test that path traversal in rename is rejected."""
        data = self._create_sample_data()
        self.collection.write('valid_item', data)

        with pytest.raises((PathSecurityError, ValueError)):
            self.collection.rename_item('valid_item', '../escaped')


class TestItemSecurity:
    """Test security at the item level."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        pystore.set_path(self.test_dir)
        self.store = pystore.store('test_store', engine='pyarrow')
        self.collection = self.store.collection('test_collection')
        yield
        # Cleanup
        pystore.delete_stores()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_sample_data(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1.0, 2.0, 3.0],
            'c': ['x', 'y', 'z']
        })

    def test_item_validates_name_on_access(self):
        """Test that Item validates name on access."""
        data = self._create_sample_data()
        self.collection.write('valid_item', data)

        # Try to create an Item with malicious name
        with pytest.raises((PathSecurityError, ValueError)):
            from pystore.item import Item
            Item('../escape', self.collection._item_path('valid_item').parent,
                 'test_collection')


class TestSetPathSecurity:
    """Test security for set_path function."""

    def test_set_path_validates_path(self):
        """Test that set_path validates the path."""
        test_dir = tempfile.mkdtemp()
        try:
            # Valid path should work
            result = pystore.set_path(test_dir)
            assert result is not None

            # Path with null bytes should be rejected
            with pytest.raises((PathSecurityError, ValueError)):
                pystore.set_path('/tmp/test\x00')

            # Path with traversal should be rejected
            with pytest.raises((PathSecurityError, ValueError)):
                pystore.set_path('/tmp/../etc')

        finally:
            # Cleanup
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def test_set_path_rejects_remote_schemes(self):
        """Test that set_path rejects remote storage schemes."""
        with pytest.raises(ValueError) as exc_info:
            pystore.set_path('s3://bucket/path')
        assert "local file system" in str(exc_info.value).lower()

    def test_set_path_accepts_file_scheme(self):
        """Test that set_path accepts file:// scheme."""
        test_dir = tempfile.mkdtemp()
        try:
            # file:// scheme should work
            result = pystore.set_path(f'file://{test_dir}')
            assert result is not None
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)


class TestBackwardCompatibility:
    """Test that valid existing usage patterns still work."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        pystore.set_path(self.test_dir)
        yield
        # Cleanup
        pystore.delete_stores()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_valid_names_still_work(self):
        """Test that valid names from existing code still work."""
        # Create a store with pyarrow engine (fastparquet not supported)
        store = pystore.store('my_datastore', engine='pyarrow')

        # Create a collection
        collection = store.collection('NASDAQ')

        # Write data
        data = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200]
        })
        collection.write('AAPL', data)

        # Read it back
        item = collection.item('AAPL')
        result = item.to_pandas()
        assert len(result) == 3

    def test_special_characters_allowed(self):
        """Test that safe special characters are allowed."""
        store = pystore.store('test_store', engine='pyarrow')
        collection = store.collection('test_collection')

        # Underscores, hyphens, and dots should be allowed
        data = pd.DataFrame({'a': [1, 2, 3]})

        collection.write('item_name', data)
        collection.write('item-name', data)

        # Items should be accessible
        assert 'item_name' in collection.list_items()
        assert 'item-name' in collection.list_items()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
