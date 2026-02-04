"""
File Manager Module - Comprehensive file and directory operations.

Provides high-level operations that go beyond single shell commands,
with proper error handling and integration with the permission system.
"""

import os
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from fnmatch import fnmatch


@dataclass
class FileInfo:
    """Information about a file or directory."""
    path: str
    name: str
    is_dir: bool
    size: int
    modified: str
    permissions: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OperationResult:
    """Result of a file operation."""
    success: bool
    message: str
    path: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return asdict(self)


class FileManager:
    """
    Comprehensive file management with safety checks.

    All operations respect trusted paths and provide detailed feedback.
    """

    def __init__(self, trusted_paths: Optional[List[str]] = None):
        """
        Initialize FileManager.

        Args:
            trusted_paths: List of paths where operations are allowed.
                          Defaults to home directory and /tmp.
        """
        self.home = Path.home()
        self.trusted_paths = trusted_paths or [
            str(self.home),
            '/tmp',
            '/var/tmp'
        ]
        # Expand all trusted paths
        self.trusted_paths = [os.path.expanduser(p) for p in self.trusted_paths]

    def _is_trusted(self, path: str) -> bool:
        """Check if path is within trusted directories."""
        abs_path = os.path.abspath(os.path.expanduser(path))
        return any(abs_path.startswith(tp) for tp in self.trusted_paths)

    def _resolve_path(self, path: str) -> Path:
        """Resolve path, expanding ~ and making absolute."""
        return Path(os.path.expanduser(path)).resolve()

    def _get_file_info(self, path: Path) -> FileInfo:
        """Get detailed file information."""
        stat = path.stat()
        return FileInfo(
            path=str(path),
            name=path.name,
            is_dir=path.is_dir(),
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            permissions=oct(stat.st_mode)[-3:]
        )

    # === Directory Operations ===

    def create_directory(self, path: str, parents: bool = True) -> OperationResult:
        """
        Create a directory.

        Args:
            path: Directory path to create
            parents: Create parent directories if needed (default True)
        """
        resolved = self._resolve_path(path)

        if not self._is_trusted(str(resolved)):
            return OperationResult(
                success=False,
                message=f'Path not in trusted directories: {resolved}'
            )

        if resolved.exists():
            if resolved.is_dir():
                return OperationResult(
                    success=True,
                    message=f'Directory already exists: {resolved}',
                    path=str(resolved)
                )
            return OperationResult(
                success=False,
                message=f'Path exists but is not a directory: {resolved}'
            )

        try:
            resolved.mkdir(parents=parents, exist_ok=True)
            return OperationResult(
                success=True,
                message=f'Created directory: {resolved}',
                path=str(resolved)
            )
        except PermissionError:
            return OperationResult(
                success=False,
                message=f'Permission denied: {resolved}'
            )
        except Exception as e:
            return OperationResult(
                success=False,
                message=f'Failed to create directory: {e}'
            )

    def delete_directory(self, path: str, recursive: bool = False) -> OperationResult:
        """
        Delete a directory.

        Args:
            path: Directory to delete
            recursive: Delete contents recursively (required for non-empty dirs)
        """
        resolved = self._resolve_path(path)

        if not self._is_trusted(str(resolved)):
            return OperationResult(
                success=False,
                message=f'Path not in trusted directories: {resolved}'
            )

        if not resolved.exists():
            return OperationResult(
                success=False,
                message=f'Directory does not exist: {resolved}'
            )

        if not resolved.is_dir():
            return OperationResult(
                success=False,
                message=f'Path is not a directory: {resolved}'
            )

        # Safety check for important directories
        important_dirs = [self.home, self.home / 'Documents', self.home / 'Downloads']
        if resolved in important_dirs:
            return OperationResult(
                success=False,
                message=f'Cannot delete protected directory: {resolved}'
            )

        try:
            if recursive:
                shutil.rmtree(resolved)
            else:
                resolved.rmdir()  # Only works on empty directories
            return OperationResult(
                success=True,
                message=f'Deleted directory: {resolved}',
                path=str(resolved)
            )
        except OSError as e:
            if 'not empty' in str(e).lower():
                return OperationResult(
                    success=False,
                    message=f'Directory not empty. Use recursive=True to delete: {resolved}'
                )
            return OperationResult(
                success=False,
                message=f'Failed to delete: {e}'
            )

    def list_directory(self, path: str = '.', pattern: str = '*',
                       show_hidden: bool = False) -> OperationResult:
        """
        List directory contents.

        Args:
            path: Directory to list
            pattern: Glob pattern to filter results
            show_hidden: Include hidden files (starting with .)
        """
        resolved = self._resolve_path(path)

        if not resolved.exists():
            return OperationResult(
                success=False,
                message=f'Directory does not exist: {resolved}'
            )

        if not resolved.is_dir():
            return OperationResult(
                success=False,
                message=f'Path is not a directory: {resolved}'
            )

        try:
            entries = []
            for item in sorted(resolved.iterdir()):
                if not show_hidden and item.name.startswith('.'):
                    continue
                if not fnmatch(item.name, pattern):
                    continue
                entries.append(self._get_file_info(item).to_dict())

            return OperationResult(
                success=True,
                message=f'Listed {len(entries)} items in {resolved}',
                path=str(resolved),
                details={'entries': entries, 'count': len(entries)}
            )
        except PermissionError:
            return OperationResult(
                success=False,
                message=f'Permission denied: {resolved}'
            )

    # === File Operations ===

    def create_file(self, path: str, content: str = '') -> OperationResult:
        """
        Create a file with optional content.

        Args:
            path: File path to create
            content: Optional initial content
        """
        resolved = self._resolve_path(path)

        if not self._is_trusted(str(resolved)):
            return OperationResult(
                success=False,
                message=f'Path not in trusted directories: {resolved}'
            )

        # Ensure parent directory exists
        parent = resolved.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        try:
            resolved.write_text(content)
            return OperationResult(
                success=True,
                message=f'Created file: {resolved}',
                path=str(resolved),
                details={'size': len(content)}
            )
        except Exception as e:
            return OperationResult(
                success=False,
                message=f'Failed to create file: {e}'
            )

    def delete_file(self, path: str) -> OperationResult:
        """Delete a file."""
        resolved = self._resolve_path(path)

        if not self._is_trusted(str(resolved)):
            return OperationResult(
                success=False,
                message=f'Path not in trusted directories: {resolved}'
            )

        if not resolved.exists():
            return OperationResult(
                success=False,
                message=f'File does not exist: {resolved}'
            )

        if resolved.is_dir():
            return OperationResult(
                success=False,
                message=f'Path is a directory, use delete_directory: {resolved}'
            )

        try:
            resolved.unlink()
            return OperationResult(
                success=True,
                message=f'Deleted file: {resolved}',
                path=str(resolved)
            )
        except Exception as e:
            return OperationResult(
                success=False,
                message=f'Failed to delete file: {e}'
            )

    def copy(self, source: str, destination: str) -> OperationResult:
        """
        Copy file or directory.

        Args:
            source: Source path
            destination: Destination path
        """
        src = self._resolve_path(source)
        dst = self._resolve_path(destination)

        if not self._is_trusted(str(dst)):
            return OperationResult(
                success=False,
                message=f'Destination not in trusted directories: {dst}'
            )

        if not src.exists():
            return OperationResult(
                success=False,
                message=f'Source does not exist: {src}'
            )

        try:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                # If destination is a directory, copy into it
                if dst.is_dir():
                    dst = dst / src.name
                shutil.copy2(src, dst)

            return OperationResult(
                success=True,
                message=f'Copied {src} to {dst}',
                path=str(dst)
            )
        except Exception as e:
            return OperationResult(
                success=False,
                message=f'Copy failed: {e}'
            )

    def move(self, source: str, destination: str) -> OperationResult:
        """
        Move file or directory.

        Args:
            source: Source path
            destination: Destination path
        """
        src = self._resolve_path(source)
        dst = self._resolve_path(destination)

        if not self._is_trusted(str(dst)):
            return OperationResult(
                success=False,
                message=f'Destination not in trusted directories: {dst}'
            )

        if not src.exists():
            return OperationResult(
                success=False,
                message=f'Source does not exist: {src}'
            )

        try:
            # If destination is a directory, move into it
            if dst.is_dir():
                dst = dst / src.name
            shutil.move(str(src), str(dst))

            return OperationResult(
                success=True,
                message=f'Moved {src} to {dst}',
                path=str(dst)
            )
        except Exception as e:
            return OperationResult(
                success=False,
                message=f'Move failed: {e}'
            )

    # === Search Operations ===

    def search(self, path: str, pattern: str, recursive: bool = True,
               file_type: Optional[str] = None, max_results: int = 100) -> OperationResult:
        """
        Search for files matching a pattern.

        Args:
            path: Starting directory
            pattern: Glob pattern to match (e.g., '*.py', 'test_*')
            recursive: Search subdirectories
            file_type: 'file', 'dir', or None for both
            max_results: Maximum results to return
        """
        resolved = self._resolve_path(path)

        if not resolved.exists():
            return OperationResult(
                success=False,
                message=f'Search path does not exist: {resolved}'
            )

        results = []
        try:
            glob_pattern = f'**/{pattern}' if recursive else pattern
            for item in resolved.glob(glob_pattern):
                if len(results) >= max_results:
                    break
                if file_type == 'file' and not item.is_file():
                    continue
                if file_type == 'dir' and not item.is_dir():
                    continue
                results.append(self._get_file_info(item).to_dict())

            return OperationResult(
                success=True,
                message=f'Found {len(results)} matches',
                path=str(resolved),
                details={'matches': results, 'count': len(results), 'truncated': len(results) >= max_results}
            )
        except Exception as e:
            return OperationResult(
                success=False,
                message=f'Search failed: {e}'
            )

    def find_duplicates(self, path: str, by: str = 'hash') -> OperationResult:
        """
        Find duplicate files in a directory.

        Args:
            path: Directory to search
            by: 'hash' (content) or 'name' (filename)
        """
        resolved = self._resolve_path(path)

        if not resolved.exists() or not resolved.is_dir():
            return OperationResult(
                success=False,
                message=f'Invalid directory: {resolved}'
            )

        seen: Dict[str, List[str]] = {}

        try:
            for item in resolved.rglob('*'):
                if not item.is_file():
                    continue

                if by == 'hash':
                    # Only hash first 8KB for speed
                    with open(item, 'rb') as f:
                        key = hashlib.md5(f.read(8192)).hexdigest()
                else:
                    key = item.name

                if key not in seen:
                    seen[key] = []
                seen[key].append(str(item))

            duplicates = {k: v for k, v in seen.items() if len(v) > 1}

            return OperationResult(
                success=True,
                message=f'Found {len(duplicates)} sets of duplicates',
                path=str(resolved),
                details={'duplicates': duplicates}
            )
        except Exception as e:
            return OperationResult(
                success=False,
                message=f'Duplicate search failed: {e}'
            )

    # === Backup Operations ===

    def backup(self, source: str, backup_dir: Optional[str] = None,
               timestamp: bool = True) -> OperationResult:
        """
        Create a backup of a file or directory.

        Args:
            source: Path to back up
            backup_dir: Where to store backup (default: same directory)
            timestamp: Add timestamp to backup name
        """
        src = self._resolve_path(source)

        if not src.exists():
            return OperationResult(
                success=False,
                message=f'Source does not exist: {src}'
            )

        # Determine backup destination
        if backup_dir:
            backup_base = self._resolve_path(backup_dir)
        else:
            backup_base = src.parent

        if not self._is_trusted(str(backup_base)):
            return OperationResult(
                success=False,
                message=f'Backup directory not trusted: {backup_base}'
            )

        # Create backup name
        if timestamp:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f'{src.stem}.backup_{ts}{src.suffix}'
        else:
            backup_name = f'{src.stem}.backup{src.suffix}'

        dst = backup_base / backup_name

        try:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

            return OperationResult(
                success=True,
                message=f'Backup created: {dst}',
                path=str(dst)
            )
        except Exception as e:
            return OperationResult(
                success=False,
                message=f'Backup failed: {e}'
            )

    # === Organization Operations ===

    def organize_by_extension(self, source_dir: str, target_dir: Optional[str] = None,
                               move_files: bool = False) -> OperationResult:
        """
        Organize files into subdirectories by extension.

        Args:
            source_dir: Directory containing files to organize
            target_dir: Where to create organized structure (default: source_dir)
            move_files: Move files instead of copying
        """
        src = self._resolve_path(source_dir)
        dst = self._resolve_path(target_dir) if target_dir else src

        if not src.exists() or not src.is_dir():
            return OperationResult(
                success=False,
                message=f'Invalid source directory: {src}'
            )

        if not self._is_trusted(str(dst)):
            return OperationResult(
                success=False,
                message=f'Target directory not trusted: {dst}'
            )

        organized = {'moved': [], 'copied': [], 'errors': []}

        try:
            for item in src.iterdir():
                if not item.is_file():
                    continue

                ext = item.suffix.lower().lstrip('.') or 'no_extension'
                ext_dir = dst / ext

                if not ext_dir.exists():
                    ext_dir.mkdir(parents=True)

                target_file = ext_dir / item.name

                if move_files:
                    shutil.move(str(item), str(target_file))
                    organized['moved'].append(str(item))
                else:
                    shutil.copy2(item, target_file)
                    organized['copied'].append(str(item))

            action = 'moved' if move_files else 'copied'
            count = len(organized[action])
            return OperationResult(
                success=True,
                message=f'Organized {count} files by extension',
                path=str(dst),
                details=organized
            )
        except Exception as e:
            return OperationResult(
                success=False,
                message=f'Organization failed: {e}',
                details=organized
            )

    def get_disk_usage(self, path: str = '~') -> OperationResult:
        """Get disk usage statistics for a path."""
        resolved = self._resolve_path(path)

        if not resolved.exists():
            return OperationResult(
                success=False,
                message=f'Path does not exist: {resolved}'
            )

        try:
            total_size = 0
            file_count = 0
            dir_count = 0

            if resolved.is_file():
                total_size = resolved.stat().st_size
                file_count = 1
            else:
                for item in resolved.rglob('*'):
                    if item.is_file():
                        total_size += item.stat().st_size
                        file_count += 1
                    elif item.is_dir():
                        dir_count += 1

            # Format size
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if total_size < 1024:
                    size_str = f'{total_size:.2f} {unit}'
                    break
                total_size /= 1024
            else:
                size_str = f'{total_size:.2f} PB'

            return OperationResult(
                success=True,
                message=f'{size_str} in {file_count} files, {dir_count} directories',
                path=str(resolved),
                details={
                    'total_bytes': int(total_size * (1024 ** ['B', 'KB', 'MB', 'GB', 'TB', 'PB'].index(unit))),
                    'size_human': size_str,
                    'file_count': file_count,
                    'dir_count': dir_count
                }
            )
        except Exception as e:
            return OperationResult(
                success=False,
                message=f'Failed to calculate disk usage: {e}'
            )


# Convenience instance with default settings
file_manager = FileManager()
