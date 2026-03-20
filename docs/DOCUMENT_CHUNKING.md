# Document Chunking

This document describes the chunking pipeline and the intended data structures for processing financial filings (PDF/Markdown) into retrievable chunks suitable for embeddings and vector store ingestion.

## End-to-end pipeline (matches the paper)

```text
PDF/Markdown
  ↓
MinerU parsing
  ↓
Multimodal block extraction (text / tables / images)
  ↓
Textualization of tables & figures
  ↓
Bundle creation
  ↓
Chunk segmentation
  ↓
Deduplication
  ↓
Coreference resolution
  ↓
Section metadata generation
  ↓
Embeddings
  ↓
Vector store ingestion
```

## Resulting hierarchy

```text
Document
  ↓
Sections
  ↓
Bundles
  ↓
Chunks
  ↓
Embeddings
```

## 1. What a bundle represents

A **bundle** is a semantic unit larger than a chunk.

Hierarchy:

```text
Document
  ↓
Section
  ↓
Bundle
  ↓
Chunks
```

### Example (financial filing)

Section: **Vehicle Deliveries**

- Paragraph 1
- Paragraph 2
- Paragraph 3
- Table
- Table explanation

Token limits force us to split this into chunks:

- `chunk_1`
- `chunk_2`
- `chunk_3`
- `chunk_4`
- `chunk_5`

But these chunks must be retrieved together, so we assign:

```text
bundle_id = vehicle_deliveries_001
```

Result:

```text
chunk_1 → bundle vehicle_deliveries_001
chunk_2 → bundle vehicle_deliveries_001
chunk_3 → bundle vehicle_deliveries_001
chunk_4 → bundle vehicle_deliveries_001
chunk_5 → bundle vehicle_deliveries_001
```

During retrieval:

```text
retrieve chunk_2
  ↓
system detects bundle_id
  ↓
returns all bundle chunks
```

Bundles are typically built around:

| Content type | Bundle strategy |
| --- | --- |
| Table | entire table + description |
| Section paragraph block | paragraphs together |
| Figure explanation | figure + caption |
| Disclosure | section block |

## 2. Bundle generation logic

We assign bundle IDs based on:

```text
section_title + incremental counter
```

Example:

- `lotus_6k_2024_vehicle_deliveries_001`

Or simply:

- `bundle_001`
- `bundle_002`
- `bundle_003`

Important rule:

- All chunks derived from the same logical section share `bundle_id`.

Example bundle:

```text
bundle_004
  ├ chunk_21
  ├ chunk_22
  ├ chunk_23
  └ chunk_24
```

## 3. Production FFP architecture

Pipeline modules:

```text
src/doc_processing
│
├── ffp/
│   ├── ingestion/
│   │   ├── mineru_parser.py
│   │   └── markdown_parser.py
│   │
│   ├── multimodal/
│   │   ├── table_extractor.py
│   │   └── image_captioner.py
│   │
│   ├── processing/
│   │   ├── chunk_segmenter.py
│   │   ├── deduplicator.py
│   │   ├── coreference_resolver.py
│   │   └── section_summarizer.py
│   │
│   └── pipeline.py
│
└── data/
```

## 4. Core data structures

Before coding, we define canonical JSON schemas.

### Raw MinerU block

These come from `content_list.json` produced by MinerU.

```json
{
  "type": "text | table | image",
  "text": "...",
  "text_level": 0,
  "page_idx": 3,
  "img_path": "images/table1.png"
}
```

### Chunk object

```json
{
  "chunk_id": "doc_00021",
  "text": "...",
  "doc_id": "lotus_2024_6k",
  "page": 3,
  "bundle_id": 1,
  "section_title": "Vehicle Deliveries",
  "title_summary": "Lotus achieved major delivery growth",
  "content": "delivery growth came from extra machines from the washing machine",
  "publish_date": "2024-09-19",
  "prev_chunk": "doc_00020",
  "next_chunk": "doc_00022"
}
```

## 5. MinerU parsing layer

Parser implementation:

```python
# ingestion/mineru_parser.py

import json
from pathlib import Path


class MineruParser:
    def __init__(self, json_path):
        self.json_path = json_path

    def load_blocks(self):
        with open(self.json_path) as f:
            blocks = json.load(f)

        parsed = []

        for b in blocks:
            parsed.append(
                {
                    "type": b.get("type"),
                    "text": b.get("text"),
                    "level": b.get("text_level"),
                    "page": b.get("page_idx"),
                    "img_path": b.get("img_path", None),
                }
            )

        return parsed
```

The above parser implementation is for PDF. We need a similar implementation to process Markdown as well.


## 6. Multimodal conversion

Financial filings contain:

- tables
- charts
- diagrams

The pipeline converts them into textual narratives using vision-language models.

Tables and figures must be converted to textual narratives.

Implement:

- `table_extractor.py`: Use a vision-language model (OpenAI GPT-4o or similar) to convert table images into text describing:
  - key metrics
  - trends
  - relationships
- `image_captioner.py`: Generate captions describing financial insights from charts or diagrams.


### Table → text converter

```python
# multimodal/table_extractor.py

import base64
from openai import OpenAI


class TableExtractor:
    def __init__(self):
        self.client = OpenAI()

    def encode_image(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def convert(self, img_path):
        image = self.encode_image(img_path)

        prompt = """
Convert this financial table into a concise textual explanation.
Describe trends and key numbers.
"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image}"},
                        }
                    ],
                },
            ],
        )

        return response.choices[0].message.content
```

### Image captioner

```python
# multimodal/image_captioner.py


class ImageCaptioner:
    def __init__(self, client):
        self.client = client

    def caption(self, img_path):
        prompt = """
Describe the key financial insights of this figure.
Focus on trends or metrics.
"""

        return self.client.chat.completions.create(...)
```

## 7. Chunk segmentation

Implement chunking with the following rules:

- chunk size ≈ 200–900 characters
- segmentation must remain inside the same section
- chunks should maintain reading order

Introduce bundle_id during chunking.

bundle_id indicates that several chunks belong to the same logical unit.

All chunks that belong to the same semantic unit must share the same bundle_id.

Examples:

- tables split across multiple chunks
- multi-paragraph explanations
- figure + caption pairs
- financial statement sections

Bundle structure:

```text
bundle_001
chunk_001
chunk_002
chunk_003
```

Segmenter:

- Bundle ID is added in the `chunk_segmenter`.

```python
# processing/chunk_segmenter.py

import uuid


class ChunkSegmenter:
    def __init__(self, max_chars=900):
        self.max_chars = max_chars

    def segment_with_bundles(self, blocks):
        bundles = []
        current_bundle = []
        bundle_id = None

        for block in blocks:
            text = block.get("text")
            level = block.get("level")

            if not text:
                continue

            # Start new bundle when encountering new section title
            if level == 1:
                if current_bundle:
                    bundles.append({"bundle_id": bundle_id, "texts": current_bundle})

                bundle_id = str(uuid.uuid4())
                current_bundle = []

            current_bundle.append(text)

        if current_bundle:
            bundles.append({"bundle_id": bundle_id, "texts": current_bundle})

        return bundles
```

## 8. Deduplication engine

Implement duplicate chunk removal using cosine similarity.

Use TF-IDF embeddings.

If similarity > `0.7` (threshold), remove the duplicate chunk.

This step reduces redundant financial disclosures.

Let’s have the threshold as an environment variable in `.env.sample` and use it across.

```python
# processing/deduplicator.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Deduplicator:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def run(self, chunks):
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(chunks)

        keep = []

        for i in range(len(chunks)):
            duplicate = False

            for j in keep:
                sim = cosine_similarity(matrix[i], matrix[j])[0][0]
                if sim > self.threshold:
                    duplicate = True
                    break

            if not duplicate:
                keep.append(i)

        return [chunks[i] for i in keep]
```

## 9. Splitting a bundle into chunks

Each bundle may still exceed token limits, so we split within the bundle.

```python
def split_bundle_into_chunks(bundle, max_chars=900):
    texts = bundle["texts"]
    bundle_id = bundle["bundle_id"]

    chunks = []
    buffer = ""

    for text in texts:
        if len(buffer) + len(text) < max_chars:
            buffer += " " + text
        else:
            chunks.append({"bundle_id": bundle_id, "text": buffer.strip()})
            buffer = text

    if buffer:
        chunks.append({"bundle_id": bundle_id, "text": buffer.strip()})

    return chunks
```

## 10. Coreference resolution

Implement coreference resolution using an LLM.

Goal:

Replace pronouns with explicit entities.

Example:

"It achieved strong revenue growth."

Becomes:

"Lotus Technology achieved strong revenue growth."


```text
It delivered 7,617 vehicles
```

Becomes:

```text
Lotus Technology delivered 7,617 vehicles
```


Algorithm:

- each chunk looks at previous `k = 4` chunks
- only consider chunks with the same section title
- rewrite the chunk replacing pronouns



The pipeline uses previous \(k\) chunks as context (\(k = 4\)).

The previous \(k\) value should be configured via `.env.sample` and used from there.

```python
# processing/coreference_resolver.py


class CoreferenceResolver:
    def __init__(self, client):
        self.client = client

    def resolve(self, chunk, context):
        prompt = f"""
Resolve pronouns in the text.

Context:
{context}

Text:
{chunk}

Rewrite replacing pronouns with entities.
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content
```

## 11. Section metadata summaries

Each chunk receives section-level summary metadata.

```python
# processing/section_summarizer.py


class SectionSummarizer:
    def __init__(self, client):
        self.client = client

    def summarize(self, section_text):
        prompt = f"""
Summarize this financial filing section in one sentence.

{section_text}
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content
```

## 12. Full FFP pipeline

Now we orchestrate the full system.

Also add all the metadata fields as defined above.

```python
# pipeline.py


class FinancialFilingsPipeline:
    def __init__(self):
        self.parser = MineruParser(...)
        self.segmenter = ChunkSegmenter()
        self.deduplicator = Deduplicator()
        self.embedder = Embedder()

    def run(self, json_path):
        blocks = self.parser.load_blocks()

        chunks = self.segmenter.segment(blocks)
        chunks = self.deduplicator.run(chunks)

        return chunks
```

The final output is the chunk object.