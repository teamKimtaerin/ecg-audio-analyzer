"""
Output Manager - Handle JSON file output with automatic directory creation
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Union, Optional, Dict, Any
from ..utils.logger import get_logger


class OutputManager:
    """Manage output file creation and JSON serialization"""

    def __init__(self, base_output_dir: str = "output"):
        """
        Initialize output manager

        Args:
            base_output_dir: Base directory for output files
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = get_logger().bind_context(service="output_manager")

        # Ensure output directory exists
        self._ensure_output_directory()

    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        try:
            self.base_output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("output_directory_ready", path=str(self.base_output_dir))
        except Exception as e:
            self.logger.error(
                "failed_to_create_output_directory",
                path=str(self.base_output_dir),
                error=str(e),
            )
            raise

    def generate_output_filename(
        self,
        input_filename: str,
        suffix: str = "analysis",
        extension: str = "json",
        include_timestamp: bool = True,
    ) -> str:
        """
        Generate output filename based on input filename

        Args:
            input_filename: Original input filename
            suffix: Suffix to add to filename
            extension: File extension
            include_timestamp: Whether to include timestamp

        Returns:
            Generated filename
        """
        # Get base name without extension
        input_path = Path(input_filename)
        base_name = input_path.stem

        # Create output filename
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{suffix}_{timestamp}.{extension}"
        else:
            filename = f"{base_name}_{suffix}.{extension}"

        return filename

    def get_output_path(
        self,
        input_filename: str,
        output_filename: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> Path:
        """
        Get full output path for a file

        Args:
            input_filename: Original input filename
            output_filename: Custom output filename (optional)
            subfolder: Subfolder within output directory

        Returns:
            Full path for output file
        """
        # Determine output filename
        if output_filename is None:
            output_filename = self.generate_output_filename(input_filename)

        # Determine output directory
        output_dir = self.base_output_dir
        if subfolder:
            output_dir = output_dir / subfolder
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir / output_filename

    def save_json(
        self,
        data: Union[Dict[str, Any], Any],
        output_path: Union[str, Path],
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> Path:
        """
        Save data as JSON file

        Args:
            data: Data to save (must be JSON serializable)
            output_path: Output file path
            indent: JSON indentation
            ensure_ascii: Whether to ensure ASCII encoding

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)

        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict if it has to_dict method
            if hasattr(data, "to_dict"):
                data = data.to_dict()

            # Save JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    data, f, indent=indent, ensure_ascii=ensure_ascii, default=str
                )

            self.logger.info(
                "json_file_saved",
                path=str(output_path),
                size_bytes=output_path.stat().st_size,
            )

            return output_path

        except Exception as e:
            self.logger.error(
                "failed_to_save_json", path=str(output_path), error=str(e)
            )
            raise

    def save_analysis_result(
        self,
        result: Any,
        input_filename: str,
        custom_output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Save analysis result with automatic path generation

        Args:
            result: Analysis result object
            input_filename: Original input filename
            custom_output_path: Custom output path (optional)

        Returns:
            Path to saved file
        """
        if custom_output_path:
            output_path = Path(custom_output_path)
            # If it's just a filename, put it in output directory
            if not output_path.is_absolute() and output_path.parent == Path("."):
                output_path = self.base_output_dir / output_path
        else:
            output_path = self.get_output_path(input_filename)

        return self.save_json(result, output_path)

    def load_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load JSON data from file

        Args:
            file_path: Path to JSON file

        Returns:
            Loaded JSON data
        """
        file_path = Path(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.logger.info("json_file_loaded", path=str(file_path))
            return data

        except Exception as e:
            self.logger.error("failed_to_load_json", path=str(file_path), error=str(e))
            raise

    def list_output_files(
        self, pattern: str = "*.json", subfolder: Optional[str] = None
    ) -> list[Path]:
        """
        List output files matching pattern

        Args:
            pattern: File pattern to match
            subfolder: Subfolder to search in

        Returns:
            List of matching file paths
        """
        search_dir = self.base_output_dir
        if subfolder:
            search_dir = search_dir / subfolder

        if not search_dir.exists():
            return []

        return list(search_dir.glob(pattern))

    def cleanup_old_files(
        self,
        max_age_days: int = 30,
        pattern: str = "*.json",
        subfolder: Optional[str] = None,
    ) -> int:
        """
        Clean up old output files

        Args:
            max_age_days: Maximum age in days
            pattern: File pattern to match
            subfolder: Subfolder to clean

        Returns:
            Number of files deleted
        """
        import time

        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60

        files = self.list_output_files(pattern, subfolder)
        deleted_count = 0

        for file_path in files:
            try:
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    deleted_count += 1
                    self.logger.info("old_file_deleted", path=str(file_path))
            except Exception as e:
                self.logger.warning(
                    "failed_to_delete_old_file", path=str(file_path), error=str(e)
                )

        if deleted_count > 0:
            self.logger.info("cleanup_completed", deleted_files=deleted_count)

        return deleted_count

    def get_output_info(self) -> Dict[str, Any]:
        """
        Get information about output directory

        Returns:
            Dictionary with output directory information
        """
        try:
            files = self.list_output_files("*")
            json_files = self.list_output_files("*.json")

            total_size = sum(f.stat().st_size for f in files if f.is_file())

            return {
                "output_directory": str(self.base_output_dir),
                "total_files": len(files),
                "json_files": len(json_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "exists": self.base_output_dir.exists(),
                "writable": self.base_output_dir.exists()
                and os.access(self.base_output_dir, os.W_OK),
            }

        except Exception as e:
            self.logger.error("failed_to_get_output_info", error=str(e))
            return {"error": str(e)}


# Global output manager instance
_global_output_manager: Optional[OutputManager] = None


def get_output_manager(base_output_dir: str = "output") -> OutputManager:
    """Get or create global output manager instance"""
    global _global_output_manager

    if (
        _global_output_manager is None
        or str(_global_output_manager.base_output_dir) != base_output_dir
    ):
        _global_output_manager = OutputManager(base_output_dir)

    return _global_output_manager
