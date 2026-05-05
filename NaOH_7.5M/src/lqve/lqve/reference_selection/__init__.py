"""ViSNet descriptor based reference selection for frame-level LQVE."""

from .config import ReferenceSelectionConfig
from .features import extract_query_feature
from .library import ReferenceLibrary, load_reference_library
from .selector import ReferenceSelectionResult, select_reference

__all__ = [
    "ReferenceLibrary",
    "ReferenceSelectionConfig",
    "ReferenceSelectionResult",
    "extract_query_feature",
    "load_reference_library",
    "select_reference",
]
