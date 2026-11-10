# docling-tableformer-batched

A [docling](https://github.com/docling-project/docling) table structure plugin: docling's own
TableFormer v1 model with all tables of a page batch in one TableFormer call.

<p align="center">
  <a href="https://github.com/DCC-BS/docling-tableformer-batched">GitHub</a>
  &nbsp;|&nbsp;
  <a href="https://pypi.org/project/docling-tableformer-batched/">PyPI</a>
</p>

## What it changes

docling's `TableStructureModel` calls TableFormer once per table, prepares the page image for every
table and collects a table's PDF words by deep-copying every word cell of the page. The engine kind
`tableformer-v1-batched` keeps docling's model, options and output and changes only how a page batch
is run:

- all tables of the page batch go to TableFormer in one `multi_page_predict` call, so each page is
  prepared once and the tag decoder runs several tables at once;
- a table's PDF words are found with the same filter and coordinate conversion as docling's
  `get_cells_in_bbox`, without copying the page's word cells.

The tables are the same as with docling's `docling_tableformer` kind.

`multi_page_predict` exists in DCC's docling-ibm-models build
([DCC-BS/docling-ibm-models](https://github.com/DCC-BS/docling-ibm-models), release `v4.0.4+dcc.1`),
which also carries the faster TableFormer v1 decoder of
[docling-project/docling-ibm-models#210](https://github.com/docling-project/docling-ibm-models/pull/210).
The batched path reproduces docling 2.142's table loop (docling 2.146 collects table words
differently); with other docling or docling-ibm-models releases the engine runs docling's own per-table loop.

## Installation

```bash
pip install docling-tableformer-batched
# DCC's docling-ibm-models build, for the batched call
pip install --no-deps "https://github.com/DCC-BS/docling-ibm-models/releases/download/v4.0.4%2Bdcc.1/docling_ibm_models-4.0.4+dcc.1-py3-none-any.whl"
```

## Usage

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_tableformer_batched import TableFormerBatchedOptions

pipeline_options = PdfPipelineOptions(allow_external_plugins=True)
pipeline_options.table_structure_options = TableFormerBatchedOptions(mode="accurate")
converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
```

With **docling-serve** (`DOCLING_SERVE_ALLOW_EXTERNAL_PLUGINS=true`), as a table structure preset:

```bash
DOCLING_SERVE_CUSTOM_TABLE_STRUCTURE_PRESETS='{"tableformer_v1_batched": {"kind": "tableformer-v1-batched", "mode": "accurate"}}'
DOCLING_SERVE_DEFAULT_TABLE_STRUCTURE_PRESET=tableformer_v1_batched
```

It takes every option of docling's TableFormer v1 (`mode`, `do_cell_matching`).

## Development

```bash
make install   # uv sync (with DCC's docling-ibm-models build) + pre-commit hooks
make check     # ruff, ty
make test      # pytest; TABLEFORMER_E2E=1 also runs TableFormer itself (downloads the models)
```

## License

MIT
