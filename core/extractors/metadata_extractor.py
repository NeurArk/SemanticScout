from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict


class MetadataExtractor:
    """Extract basic file metadata."""

    def extract(self, file_path: str) -> Dict[str, Any]:
        stat = os.stat(file_path)
        return {
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "size": stat.st_size,
        }
