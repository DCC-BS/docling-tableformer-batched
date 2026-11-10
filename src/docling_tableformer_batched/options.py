"""Options of the ``tableformer-v1-batched`` table structure engine."""

from __future__ import annotations

from typing import ClassVar

from docling.datamodel.pipeline_options import TableStructureOptions


class TableFormerBatchedOptions(TableStructureOptions):
    """docling's TableFormer v1 options (``mode``, ``do_cell_matching``) under the kind ``tableformer-v1-batched``."""

    kind: ClassVar[str] = "tableformer-v1-batched"
