from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class DocumentNode(BaseModel):
    """Represents a document in visualization."""

    document_id: str
    label: str
    position: Tuple[float, float]
    size: float = Field(default=10.0)
    color: str = Field(default="#3498db")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentEdge(BaseModel):
    """Represents similarity between documents."""

    source_id: str
    target_id: str
    weight: float = Field(..., ge=0.0, le=1.0)
    label: Optional[str] = None


class VisualizationData(BaseModel):
    """Data for document visualization."""

    nodes: List[DocumentNode]
    edges: List[DocumentEdge]
    layout_type: str = Field(default="force")

    def to_plotly_format(self) -> Dict[str, Any]:
        """Convert to Plotly graph format."""
        # Placeholder for full implementation
        raise NotImplementedError


__all__ = ["DocumentNode", "DocumentEdge", "VisualizationData"]
