"""docling's TableFormer v1 table model with one TableFormer call per page batch (kind ``tableformer-v1-batched``).

docling's ``TableStructureModel`` calls TableFormer once per table, resizes the page image for every
table, and collects each table's PDF words by deep-copying every word cell of the page. This engine
keeps docling's model, options and output and changes only how a page batch is run:

* All tables of the page batch go to TableFormer in one ``multi_page_predict`` call, so each page is
  prepared once and the tag decoder runs several tables at once. That call exists in DCC's
  docling-ibm-models build (``4.0.4+dcc.1``, see its README). The page loop reproduces docling
  2.142's ``predict_tables``; with other docling or docling-ibm-models releases the engine runs
  docling's own per-table loop.
* A table's PDF words are found with the same filter and coordinate conversion as
  ``SegmentedPage.get_cells_in_bbox``, without copying the page's word cells.

Select it with a table structure preset of kind ``tableformer-v1-batched``, e.g.
``{"kind": "tableformer-v1-batched", "mode": "accurate"}``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from docling.datamodel.base_models import Table, TableStructurePrediction
from docling.datamodel.settings import settings
from docling.models.stages.table_structure.table_structure_model import TableStructureModel
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, TableCell
from docling_core.types.doc.page import BoundingRectangle, TextCellUnit

from docling_tableformer_batched.options import TableFormerBatchedOptions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from docling.datamodel.base_models import Cluster, Page
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import TableStructureOptions
    from docling_core.types.doc.page import SegmentedPage, TextCell

#: Share of a word cell that must lie inside the table box, as in ``get_cells_in_bbox``
_INSIDE = 0.8
#: The docling release whose ``predict_tables`` the batched path reproduces (2.146 collects table
#: words differently); other releases run docling's own loop
_DOCLING_RELEASE = "2.142."


def _docling_version() -> str:
    for dist in ("docling", "docling-slim"):
        try:
            return version(dist)
        except PackageNotFoundError:
            continue
    return ""


_DOCLING_SUPPORTED = _docling_version().startswith(_DOCLING_RELEASE)


def _cells_in_bbox(sp: SegmentedPage, bbox: BoundingBox) -> tuple[list[TextCell], list[BoundingRectangle]]:
    """``sp.get_cells_in_bbox(TextCellUnit.WORD, bbox)`` without deep-copying every word cell of the page.

    Returns the matching cells and their rects in ``bbox``'s coordinate origin, which is what the
    copies ``get_cells_in_bbox`` returns carry.
    """
    cells, rects = [], []
    height = sp.dimension.height
    for cell in sp.iterate_cells(TextCellUnit.WORD):
        rect = cell.rect
        if rect.coord_origin != bbox.coord_origin:
            if bbox.coord_origin == CoordOrigin.TOPLEFT:
                rect = rect.to_top_left_origin(height)
            elif bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
                rect = rect.to_bottom_left_origin(height)
        if rect.to_bounding_box().intersection_over_self(bbox) > _INSIDE:
            cells.append(cell)
            rects.append(rect)
    return cells, rects


class _PerPageTime:
    """docling's ``TimeRecorder`` for a page batch: one sample per page, the batch time split evenly."""

    def __init__(self, conv_res: ConversionResult, key: str, pages: int) -> None:
        self._recorder = TimeRecorder(conv_res, key)
        self._conv_res, self._key, self._pages = conv_res, key, pages

    def __enter__(self) -> Self:
        self._recorder.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._recorder.__exit__(*exc)
        item = self._conv_res.timings.get(self._key) if settings.debug.profile_pipeline_timings else None
        if item is None or not item.times or self._pages <= 1:
            return
        share = item.times.pop() / self._pages
        start = item.start_timestamps[-1]
        item.times.extend([share] * self._pages)
        item.start_timestamps.extend([start] * (self._pages - 1))
        item.count += self._pages - 1


class TableFormerBatchedModel(TableStructureModel):
    """docling's ``TableStructureModel`` with all tables of a page batch in one TableFormer call."""

    @classmethod
    def get_options_type(cls) -> type[TableStructureOptions]:
        """Return the options class for this table structure engine."""
        return TableFormerBatchedOptions

    def predict_tables(self, conv_res: ConversionResult, pages: Sequence[Page]) -> Sequence[TableStructurePrediction]:
        """Table structure of every table in the page batch, with one TableFormer call when the predictor has it."""
        if not (_DOCLING_SUPPORTED and hasattr(getattr(self, "tf_predictor", None), "multi_page_predict")):
            return super().predict_tables(conv_res, pages)
        pages = list(pages)
        predictions: list[TableStructurePrediction] = []
        requests: list[tuple[dict[str, Any], list[list[float]]]] = []  # (page input, [table box]) per table
        owners: list[tuple[Page, TableStructurePrediction, Cluster]] = []
        valid = sum(1 for page in pages if page._backend is not None and page._backend.is_valid())  # noqa: SLF001
        with _PerPageTime(conv_res, "table_structure", valid):
            for page in pages:
                prediction, tables = self._page_requests(page)
                predictions.append(prediction)
                for page_input, box, cluster in tables:
                    requests.append((page_input, [box]))
                    owners.append((page, prediction, cluster))
            if requests:
                outputs = self.tf_predictor.multi_page_predict(requests, do_matching=self.do_cell_matching)
                for (page, prediction, cluster), tf_output in zip(owners, outputs, strict=True):
                    prediction.table_map[cluster.id] = self._make_table(page, cluster, tf_output[0])
            if settings.debug.visualize_tables:
                for page, prediction in zip(pages, predictions, strict=True):
                    if prediction.table_map:
                        self.draw_table_and_cells(conv_res, page, prediction.table_map.values())
        return predictions

    def _page_requests(
        self, page: Page
    ) -> tuple[TableStructurePrediction, list[tuple[dict[str, Any], list[float], Cluster]]]:
        """The page's prediction (set on the page) and the TableFormer input of each of its tables."""
        if page._backend is None or not page._backend.is_valid():  # noqa: SLF001
            existing = page.predictions.tablestructure or TableStructurePrediction()
            page.predictions.tablestructure = existing
            return existing, []
        prediction = TableStructurePrediction()
        page.predictions.tablestructure = prediction
        if page.predictions.layout is None or page.size is None:
            return prediction, []
        tables = [
            (
                cluster,
                [
                    round(cluster.bbox.l) * self.scale,
                    round(cluster.bbox.t) * self.scale,
                    round(cluster.bbox.r) * self.scale,
                    round(cluster.bbox.b) * self.scale,
                ],
            )
            for cluster in page.predictions.layout.clusters
            if cluster.label in {DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX}
        ]
        if not tables:
            return prediction, []
        width, height = page.size.width * self.scale, page.size.height * self.scale
        image = np.asarray(page.get_image(scale=self.scale))
        sp = page._backend.get_segmented_page()  # noqa: SLF001
        return prediction, [
            (
                {"width": width, "height": height, "image": image, "tokens": self._table_tokens(sp, cluster)},
                box,
                cluster,
            )
            for cluster, box in tables
        ]

    def _table_tokens(self, sp: SegmentedPage | None, cluster: Cluster) -> list[dict[str, Any]]:
        """The PDF words of a table as TableFormer tokens, as docling's ``predict_tables`` builds them."""
        rects: list[BoundingRectangle] | None = None
        if sp is not None:
            cells, rects = _cells_in_bbox(sp, cluster.bbox)
            if not cells:  # no word cells: the cluster's line cells
                cells, rects = list(cluster.cells), None
        else:
            cells = list(cluster.cells)
        tokens = []
        for k, cell in enumerate(cells):
            if cell.text.strip():  # only non-empty strings go into the cells of a table
                rect = rects[k] if rects is not None else cell.rect
                bbox = BoundingRectangle.from_bounding_box(rect.to_bounding_box().scaled(scale=self.scale))
                tokens.append({"id": cell.index, "text": cell.text, "bbox": bbox.to_bounding_box().model_dump()})
        return tokens

    def _make_table(self, page: Page, cluster: Cluster, table_out: dict[str, Any]) -> Table:
        """Docling's ``Table`` of one TableFormer output, as ``predict_tables`` builds it."""
        backend = page._backend  # noqa: SLF001
        table_cells = []
        for element in table_out["tf_responses"]:
            if not self.do_cell_matching and backend is not None:
                the_bbox = BoundingBox.model_validate(element["bbox"]).scaled(1 / self.scale)
                element["bbox"]["token"] = backend.get_text_in_rect(the_bbox)
            cell = TableCell.model_validate(element)
            if cell.bbox is not None:
                cell.bbox = cell.bbox.scaled(1 / self.scale)
            table_cells.append(cell)
        details = table_out["predict_details"]
        return Table(
            otsl_seq=details.get("prediction", {}).get("rs_seq", []),
            table_cells=table_cells,
            num_rows=details.get("num_rows", 0),
            num_cols=details.get("num_cols", 0),
            id=cluster.id,
            page_no=page.page_no,
            cluster=cluster,
            label=cluster.label,
        )
