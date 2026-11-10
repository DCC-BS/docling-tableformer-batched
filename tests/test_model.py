"""The batched engine gives docling's TableStructureModel inputs and tables, with one TableFormer call per batch."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.datamodel.base_models import Cluster, InputFormat, LayoutPrediction, Page
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import TableStructureOptions
from docling.models.base_table_model import BaseTableStructureModel
from docling.models.stages.table_structure.table_structure_model import TableStructureModel
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel
from docling_core.types.doc.page import TextCellUnit

from docling_tableformer_batched import TableFormerBatchedModel, TableFormerBatchedOptions
from docling_tableformer_batched.model import _cells_in_bbox
from docling_tableformer_batched.plugin import table_structure_engines

if TYPE_CHECKING:
    from pathlib import Path


def _text_pdf(path: Path, rows: int = 6, cols: int = 4) -> Path:
    """A one-page PDF with a text layer: a grid of words, like a table."""
    words = []
    for r in range(rows):
        for c in range(cols):
            words.append(f"BT /F1 10 Tf {72 + c * 110} {700 - r * 24} Td (r{r}c{c} value) Tj ET")
    words.append("BT /F1 12 Tf 72 740 Td (Title above the table) Tj ET")
    stream = "\n".join(words).encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.write_bytes(bytes(out))
    return path


def _pages(tmp_path: Path, n: int = 3) -> list[Page]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pages = []
    for i in range(n):
        doc = InputDocument(
            path_or_stream=_text_pdf(tmp_path / f"p{i}.pdf", rows=4 + i),
            format=InputFormat.PDF,
            backend=ThreadedDoclingParseDocumentBackend,
        )
        backend = next(iter(doc._backend.iter_pages()))
        page = Page(page_no=i)
        page._backend = backend
        page.size = backend.get_size()
        table = Cluster(id=7, label=DocItemLabel.TABLE, bbox=BoundingBox(l=60, t=80, r=520, b=100 + 24 * (4 + i)))
        title = Cluster(id=1, label=DocItemLabel.TEXT, bbox=BoundingBox(l=60, t=40, r=300, b=60))
        page.predictions.layout = LayoutPrediction(clusters=[title, table] if i != 1 else [title])
        pages.append(page)
    return pages


class _Predictor:
    """Stands in for TFPredictor: a table derived from its input, and a record of the calls."""

    def __init__(self, *, batched: bool) -> None:
        self.calls: list[tuple] = []
        if batched:
            self.multi_page_predict = self._multi_page_predict

    def multi_table_predict(self, page_input, boxes, *, do_matching=True):
        self.calls.append(("table", copy.deepcopy(page_input["tokens"]), boxes, page_input["image"].shape))
        cells = [
            {"bbox": {**tok["bbox"], "token": tok["text"]}, "row_span": 1, "col_span": 1, "start_row_offset_idx": k,
             "end_row_offset_idx": k + 1, "start_col_offset_idx": 0, "end_col_offset_idx": 1, "text": tok["text"]}
            for k, tok in enumerate(page_input["tokens"][:5])
        ]  # fmt: skip
        details = {"num_rows": len(cells), "num_cols": 1, "prediction": {"rs_seq": ["fcel", "nl"] * len(cells)}}
        return [{"tf_responses": cells, "predict_details": details}]

    def _multi_page_predict(self, requests, *, do_matching=True):
        self.calls.append(("batch", len(requests)))
        return [self.multi_table_predict(page_input, boxes, do_matching=do_matching) for page_input, boxes in requests]


def _model(cls, predictor) -> TableStructureModel:
    model = object.__new__(cls)
    model.enabled = True
    model.options = TableFormerBatchedOptions() if cls is TableFormerBatchedModel else TableStructureOptions()
    model.do_cell_matching = True
    model.scale = 2.0
    model.tf_predictor = predictor
    return model


def test_options_and_registration():
    assert TableFormerBatchedOptions.kind == "tableformer-v1-batched"
    assert isinstance(TableFormerBatchedOptions(), TableStructureOptions)
    assert TableFormerBatchedModel.get_options_type() is TableFormerBatchedOptions
    engines = table_structure_engines()["table_structure_engines"]
    assert engines == [TableFormerBatchedModel]
    assert issubclass(engines[0], BaseTableStructureModel)


def test_words_found_as_get_cells_in_bbox_finds_them(tmp_path):
    page = _pages(tmp_path, 1)[0]
    sp = page._backend.get_segmented_page()
    for bbox in (
        BoundingBox(l=60, t=80, r=520, b=196, coord_origin=CoordOrigin.TOPLEFT),
        BoundingBox(l=60, t=720, r=300, b=600, coord_origin=CoordOrigin.BOTTOMLEFT),
    ):
        expected = sp.get_cells_in_bbox(cell_unit=TextCellUnit.WORD, bbox=bbox)
        cells, rects = _cells_in_bbox(sp, bbox)
        assert [c.index for c in cells] == [c.index for c in expected]
        assert rects == [c.rect for c in expected]
        assert len(cells) > 0


def test_same_inputs_and_tables_as_docling_with_one_call(tmp_path):
    pages_a, pages_b = _pages(tmp_path / "a", 3), _pages(tmp_path / "b", 3)
    reference, batched = _Predictor(batched=False), _Predictor(batched=True)
    conv = SimpleNamespace(timings={})

    want = _model(TableStructureModel, reference).predict_tables(conv, pages_a)
    got = _model(TableFormerBatchedModel, batched).predict_tables(conv, pages_b)

    assert batched.calls[0] == ("batch", 2)  # one call for the two pages with a table
    assert batched.calls[1:] == reference.calls  # the same tokens, boxes and images per table
    assert [p.model_dump() for p in got] == [p.model_dump() for p in want]
    assert [p.predictions.tablestructure for p in pages_b] == list(got)


def test_without_the_batched_call_it_is_doclings_loop(tmp_path):
    predictor = _Predictor(batched=False)
    _model(TableFormerBatchedModel, predictor).predict_tables(SimpleNamespace(timings={}), _pages(tmp_path, 3))
    assert [call[0] for call in predictor.calls] == ["table", "table"]


def test_invalid_pages_keep_their_prediction(tmp_path):
    page = _pages(tmp_path, 1)[0]
    page._backend = SimpleNamespace(is_valid=lambda: False)
    out = _model(TableFormerBatchedModel, _Predictor(batched=True)).predict_tables(SimpleNamespace(timings={}), [page])
    assert out[0] is page.predictions.tablestructure
    assert out[0].table_map == {}


def test_table_time_is_recorded_per_page(tmp_path, monkeypatch):
    from docling.datamodel.settings import settings

    monkeypatch.setattr(settings.debug, "profile_pipeline_timings", True)
    conv = SimpleNamespace(timings={})
    _model(TableFormerBatchedModel, _Predictor(batched=True)).predict_tables(conv, _pages(tmp_path, 3))
    item = conv.timings["table_structure"]
    assert item.count == 3
    assert len(item.times) == len(item.start_timestamps) == 3


@pytest.mark.e2e
def test_real_tableformer_gives_doclings_tables(tmp_path):
    """With DCC's docling-ibm-models build: the batched call gives the tables of docling's per-table loop."""
    from docling.datamodel.accelerator_options import AcceleratorOptions

    accel = AcceleratorOptions(device="cpu")
    reference = TableStructureModel(
        enabled=True, artifacts_path=None, options=TableStructureOptions(), accelerator_options=accel
    )
    batched = TableFormerBatchedModel(
        enabled=True, artifacts_path=None, options=TableFormerBatchedOptions(), accelerator_options=accel
    )
    assert hasattr(batched.tf_predictor, "multi_page_predict")
    conv = SimpleNamespace(timings={})
    want = reference.predict_tables(conv, _pages(tmp_path / "a", 3))
    got = batched.predict_tables(conv, _pages(tmp_path / "b", 3))
    assert [p.model_dump() for p in got] == [p.model_dump() for p in want]
    assert sum(len(p.table_map) for p in got) == 2


def test_other_docling_releases_run_doclings_loop(tmp_path, monkeypatch):
    from docling_tableformer_batched import model as module

    monkeypatch.setattr(module, "_DOCLING_SUPPORTED", False)
    predictor = _Predictor(batched=True)
    _model(TableFormerBatchedModel, predictor).predict_tables(SimpleNamespace(timings={}), _pages(tmp_path, 3))
    assert [call[0] for call in predictor.calls] == ["table", "table"]
