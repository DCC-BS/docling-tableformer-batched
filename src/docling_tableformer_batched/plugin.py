"""Docling plugin entry point registering the batched TableFormer v1 engine."""

from docling.models.base_table_model import BaseTableStructureModel

from docling_tableformer_batched.model import TableFormerBatchedModel


def table_structure_engines() -> dict[str, list[type[BaseTableStructureModel]]]:
    """Return the table structure engine classes provided by this plugin."""
    return {"table_structure_engines": [TableFormerBatchedModel]}
