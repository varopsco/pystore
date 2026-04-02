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
from io import StringIO
from unittest.mock import patch, Mock
import sys

import pystore
from pystore.__main__ import main


class TestCLI:
    """Test CLI functionality."""

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

    def test_cli_no_arguments_shows_help(self, capsys):
        """Test that CLI shows help when no arguments are provided."""
        with patch('sys.argv', ['pystore']):
            result = main()
            assert result == 0
            captured = capsys.readouterr()
            assert 'usage: pystore' in captured.out or 'PyStore' in captured.out

    def test_cli_version(self, capsys):
        """Test that CLI version flag works."""
        with patch('sys.argv', ['pystore', '--version']):
            with pytest.raises(SystemExit):
                main()

    def test_cli_list_empty(self, capsys):
        """Test list command when no stores exist."""
        with patch('sys.argv', ['pystore', 'list']):
            result = main()
            assert result == 0
            captured = capsys.readouterr()
            assert 'No stores found' in captured.out

    def test_cli_list_with_stores(self, capsys):
        """Test list command when stores exist."""
        # Create a store
        store = pystore.store('test_store')
        store.collection('test_collection')

        with patch('sys.argv', ['pystore', 'list']):
            result = main()
            assert result == 0
            captured = capsys.readouterr()
            assert 'test_store' in captured.out

    def test_cli_list_with_custom_path(self, capsys):
        """Test list command with custom path."""
        custom_dir = tempfile.mkdtemp()
        original_path = pystore.get_path()
        try:
            pystore.set_path(custom_dir)
            store = pystore.store('custom_store')
            store.collection('test_collection')

            with patch('sys.argv', ['pystore', 'list', '--path', custom_dir]):
                result = main()
                assert result == 0
                captured = capsys.readouterr()
                assert 'custom_store' in captured.out
        finally:
            # Reset to original path before cleanup
            pystore.set_path(str(original_path))
            if os.path.exists(custom_dir):
                shutil.rmtree(custom_dir)

    def test_cli_info(self, capsys):
        """Test info command."""
        with patch('sys.argv', ['pystore', 'info']):
            result = main()
            assert result == 0
            captured = capsys.readouterr()
            assert 'PyStore version:' in captured.out
            assert 'Storage path:' in captured.out

    def test_cli_info_with_custom_path(self, capsys):
        """Test info command with custom path."""
        custom_dir = tempfile.mkdtemp()
        original_path = pystore.get_path()
        try:
            with patch('sys.argv', ['pystore', 'info', '--path', custom_dir]):
                result = main()
                assert result == 0
                captured = capsys.readouterr()
                assert 'PyStore version:' in captured.out
        finally:
            # Reset to original path before cleanup
            pystore.set_path(str(original_path))
            if os.path.exists(custom_dir):
                shutil.rmtree(custom_dir)

    def test_cli_delete_with_force(self, capsys):
        """Test delete command with force flag."""
        # Create a store
        store = pystore.store('delete_me')
        store.collection('test_collection')

        with patch('sys.argv', ['pystore', 'delete', 'delete_me', '--force']):
            result = main()
            assert result == 0
            captured = capsys.readouterr()
            assert 'deleted successfully' in captured.out

        # Verify store is deleted
        stores = pystore.list_stores()
        assert 'delete_me' not in stores

    def test_cli_delete_with_confirmation_yes(self, capsys):
        """Test delete command with user confirmation (yes)."""
        # Create a store
        store = pystore.store('delete_me_confirmed')
        store.collection('test_collection')

        with patch('builtins.input', return_value='y'):
            with patch('sys.argv', ['pystore', 'delete', 'delete_me_confirmed']):
                result = main()
                assert result == 0
                captured = capsys.readouterr()
                assert 'deleted successfully' in captured.out

        # Verify store is deleted
        stores = pystore.list_stores()
        assert 'delete_me_confirmed' not in stores

    def test_cli_delete_with_confirmation_no(self, capsys):
        """Test delete command with user cancellation (no)."""
        # Create a store
        store = pystore.store('keep_me')
        store.collection('test_collection')

        with patch('builtins.input', return_value='n'):
            with patch('sys.argv', ['pystore', 'delete', 'keep_me']):
                result = main()
                assert result == 0
                captured = capsys.readouterr()
                assert 'Deletion cancelled' in captured.out

        # Verify store still exists
        stores = pystore.list_stores()
        assert 'keep_me' in stores

    def test_cli_delete_nonexistent_store_with_force(self, capsys):
        """Test delete command for nonexistent store with force flag."""
        with patch('sys.argv', ['pystore', 'delete', 'nonexistent', '--force']):
            result = main()
            # Should handle error gracefully
            assert result == 1 or result == 0

    def test_cli_delete_with_custom_path(self, capsys):
        """Test delete command with custom path."""
        custom_dir = tempfile.mkdtemp()
        original_path = pystore.get_path()
        try:
            pystore.set_path(custom_dir)
            store = pystore.store('custom_delete')
            store.collection('test_collection')

            with patch('sys.argv', ['pystore', 'delete', 'custom_delete', '--force', '--path', custom_dir]):
                result = main()
                assert result == 0
                captured = capsys.readouterr()
                assert 'deleted successfully' in captured.out
        finally:
            # Reset to original path before cleanup
            pystore.set_path(str(original_path))
            if os.path.exists(custom_dir):
                shutil.rmtree(custom_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
