# mellea-grounded-qa

## Grounded Q&A Assistant
Ask questions over a real set of documents and get answers that cite their source — and an honest "I don't know" when the answer isn't there, instead of a made-up one. Built with [Mellea](https://docs.mellea.ai).

## Setup

Requires Python 3.12+, `pip`, and a backend: Ollama running an IBM Granite model locally, or a hosted Granite endpoint.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama pull granite4.1:3b

```

## Document set

The corpus is the FIU computer science course syllabi. Place the source files in the `corpus/` folder. Supported formats: `.docx`, `.doc`, and `.png` (read with OCR).

Other document sets work too, e.g. a public product manual or government benefits and tax rules (such as IRS publications at `irs.gov`).

## Folder structure

```text
mellea-grounded-qa/
|-- corpus/                   # Source documents to process
|-- textjson/                 # Extracted document text (generated)
|   `-- text.json
|-- chunkeddata/              # Text chunks for retrieval (generated)
|   `-- chunked_data.json
|-- config.py                 # Chunking and JSON settings
|-- dbloader.py               # Extracts text from source documents
|-- chunker.py                # Splits extracted text into chunks
|-- main.py                   # Runs the full processing pipeline
|-- requirements.txt          # Python dependencies
`-- README.md                 # Project documentation
```

The `textjson/` and `chunkeddata/` folders and their JSON files are created when the pipeline runs. The `.venv/` folder is also present after setting up the local Python environment; it is not part of the project source.


## Run the pipeline

```bash
python3 main.py
```

This runs the document loader (`dbloader.py`) and then the chunker (`chunker.py`).

### Output

| File | Contents |
| --- | --- |
| `textjson/text.json` | Extracted text with source filenames |
| `chunkeddata/chunked_data.json` | Text chunks with source filenames and chunk IDs |

### Configuration

Chunking settings live in `config.py`:

```python
CHUNK_SIZE = 500          # tokens per chunk
CHUNK_OVERLAP = 50        # tokens shared between adjacent chunks
TOKENIZER = "cl100k_base" # tiktoken encoding used to count tokens
```

## Contributing

**Contribute:** fork, pick an issue, open a PR. Interface (CLI / notebook / web) is your team's choice.

