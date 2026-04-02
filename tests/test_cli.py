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
import subprocess
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

    def test_cli_help(self, capsys, monkeypatch):
        """Test CLI help option."""
        # Run with --help
        monkeypatch.setattr(sys, 'argv', ['pystore', '--help'])
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # Should exit with 0
        assert exc_info.value.code == 0

    def test_cli_version(self, capsys, monkeypatch):
        """Test CLI version option."""
        # Run with --version
        monkeypatch.setattr(sys, 'argv', ['pystore', '--version'])
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # Should exit with 0
        assert exc_info.value.code == 0

    def test_cli_no_args(self, capsys, monkeypatch):
        """Test CLI with no arguments shows help."""
        monkeypatch.setattr(sys, 'argv', ['pystore'])
        result = main()
        
        # Should return 0 and show help
        assert result == 0
        
        captured = capsys.readouterr()
        # Help text should be shown
        assert 'usage:' in captured.out.lower() or 'pystore' in captured.out.lower()

    def test_cli_list_empty(self, capsys, monkeypatch):
        """Test CLI list command with no stores."""
        monkeypatch.setattr(sys, 'argv', ['pystore', 'list'])
        result = main()
        
        assert result == 0
        
        captured = capsys.readouterr()
        assert 'No stores found' in captured.out or 'Available stores:' not in captured.out

    def test_cli_list_with_stores(self, capsys, monkeypatch):
        """Test CLI list command with existing stores."""
        # Create some stores
        pystore.store('store1')
        pystore.store('store2')
        
        monkeypatch.setattr(sys, 'argv', ['pystore', 'list'])
        result = main()
        
        assert result == 0
        
        captured = capsys.readouterr()
        assert 'store1' in captured.out
        assert 'store2' in captured.out

    def test_cli_list_with_path(self, capsys, monkeypatch):
        """Test CLI list command with custom path."""
        monkeypatch.setattr(sys, 'argv', ['pystore', 'list', '--path', self.test_dir])
        result = main()
        
        assert result == 0

    def test_cli_info(self, capsys, monkeypatch):
        """Test CLI info command."""
        monkeypatch.setattr(sys, 'argv', ['pystore', 'info'])
        result = main()
        
        assert result == 0
        
        captured = capsys.readouterr()
        assert 'PyStore version:' in captured.out
        assert 'Storage path:' in captured.out

    def test_cli_info_with_path(self, capsys, monkeypatch):
        """Test CLI info command with custom path."""
        monkeypatch.setattr(sys, 'argv', ['pystore', 'info', '--path', self.test_dir])
        result = main()
        
        assert result == 0
        
        captured = capsys.readouterr()
        assert 'PyStore version:' in captured.out

    def test_cli_delete_force(self, capsys, monkeypatch):
        """Test CLI delete command with --force."""
        # Create a store
        pystore.store('test_store')
        assert 'test_store' in pystore.list_stores()
        
        # Delete with --force
        monkeypatch.setattr(sys, 'argv', ['pystore', 'delete', 'test_store', '--force'])
        result = main()
        
        assert result == 0
        
        captured = capsys.readouterr()
        assert 'deleted successfully' in captured.out
        
        # Store should be deleted
        assert 'test_store' not in pystore.list_stores()

    def test_cli_delete_with_path(self, capsys, monkeypatch):
        """Test CLI delete command with custom path."""
        # Create a store
        pystore.store('test_store')
        
        # Delete with --force and path
        monkeypatch.setattr(sys, 'argv', ['pystore', 'delete', 'test_store', '--force', '--path', self.test_dir])
        result = main()
        
        assert result == 0

    def test_cli_delete_nonexistent_force(self, capsys, monkeypatch):
        """Test CLI delete command with nonexistent store."""
        monkeypatch.setattr(sys, 'argv', ['pystore', 'delete', 'nonexistent', '--force'])
        result = main()
        
        # Should return error code
        assert result == 1
        
        captured = capsys.readouterr()
        assert 'Error:' in captured.out


class TestCLISubprocess:
    """Test CLI via subprocess (integration tests)."""

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

    def test_cli_subprocess_version(self):
        """Test CLI version via subprocess."""
        result = subprocess.run(
            [sys.executable, '-m', 'pystore', '--version'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert pystore.__version__ in result.stdout

    def test_cli_subprocess_help(self):
        """Test CLI help via subprocess."""
        result = subprocess.run(
            [sys.executable, '-m', 'pystore', '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'pystore' in result.stdout.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
