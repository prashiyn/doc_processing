# Reference extraction prompt for financial documents (production ready)

## Objective

Design a low-cost LLM prompt (Groq-compatible) to extract cross-references from financial documents such as:

- Annual reports
- Financial statements
- Regulatory filings (Indian context)
- Investor presentations

The system should identify references to:

- Tables, figures
- Sections, subsections
- Appendices, annexures
- Notes, footnotes
- Schedules
- Financial statements
- Regulatory standards (Ind AS, Companies Act)

---

## Supported reference types

### Structured elements

| Kind      | Examples                          |
| --------- | --------------------------------- |
| Tables    | “Table 3”, “Table 3A”             |
| Figures   | “Figure 2”, “Chart B”             |
| Notes     | “Note 12”, “Notes to Accounts”    |
| Footnotes | “footnote 4”                      |

### Document structure

| Kind         | Examples              |
| ------------ | --------------------- |
| Sections     | “Section 4.2”         |
| Subsections  | “3.1.1”               |
| Appendices   | “Appendix A”          |
| Annexures    | “Annexure II”         |

### Indian financial context

- “Ind AS 115”
- “Schedule III”
- “Companies Act”
- “Standalone / Consolidated Financial Statements”

---

## System prompt

You are a financial document structure analysis engine.

Your task is to extract **all** cross-references present in the text.

These references may point to:

- Tables (Table 1, Table 3A)
- Figures (Figure 2, Chart B)
- Sections (Section 4.2, 3.1.1)
- Appendices (Appendix A, Annexure II)
- Notes (Note 12, Notes to Accounts)
- Footnotes (footnote 3, see note below)
- Schedules (Schedule III, Schedule V)
- Financial statements (Balance Sheet, P&L, Cash Flow)
- Regulatory references (Ind AS 115, Companies Act sections)

You must extract references **even if**:

- They appear inside brackets
- They are loosely written
- They are abbreviated

Ignore vague references like:

- “as mentioned above”
- “as discussed earlier”

Return **strict JSON**.

Each reference must include:

- `reference_text` — exact phrase
- `reference_type` — one of: `TABLE`, `FIGURE`, `SECTION`, `SUBSECTION`, `APPENDIX`, `NOTE`, `FOOTNOTE`, `SCHEDULE`, `STATEMENT`, `REGULATION`, `OTHER`
- `target_label` — normalized identifier
- `confidence` — number from 0 to 1

Do **not** hallucinate. Do **not** infer references not explicitly present.

If no references exist, return an empty list.

---

## User prompt template

Extract all cross-references from the following financial text:

```text
TEXT:
{{chunk.content}}
```

Return JSON:

```json
{
  "references": [
    {
      "reference_text": "...",
      "reference_type": "...",
      "target_label": "...",
      "confidence": 0.0
    }
  ]
}
```

---

## Few-shot examples (critical)

### Example 1 — Tables and notes

**Input**

> Revenue increased significantly (refer Table 3) and details are provided in Note 12.

**Output**

```json
{
  "references": [
    {
      "reference_text": "Table 3",
      "reference_type": "TABLE",
      "target_label": "3",
      "confidence": 0.95
    },
    {
      "reference_text": "Note 12",
      "reference_type": "NOTE",
      "target_label": "12",
      "confidence": 0.95
    }
  ]
}
```

### Example 2 — Indian accounting

**Input**

> The financial statements are prepared in accordance with Ind AS 115 and Schedule III of the Companies Act.

**Output**

```json
{
  "references": [
    {
      "reference_text": "Ind AS 115",
      "reference_type": "REGULATION",
      "target_label": "Ind AS 115",
      "confidence": 0.98
    },
    {
      "reference_text": "Schedule III",
      "reference_type": "SCHEDULE",
      "target_label": "III",
      "confidence": 0.95
    }
  ]
}
```

### Example 3 — Sections

**Input**

> Refer to Section 4.2 and subsection 3.1.1 for further details.

**Output**

```json
{
  "references": [
    {
      "reference_text": "Section 4.2",
      "reference_type": "SECTION",
      "target_label": "4.2",
      "confidence": 0.95
    },
    {
      "reference_text": "3.1.1",
      "reference_type": "SUBSECTION",
      "target_label": "3.1.1",
      "confidence": 0.9
    }
  ]
}
```

### Example 4 — Appendix

**Input**

> Details are provided in Appendix A and Annexure II.

**Output**

```json
{
  "references": [
    {
      "reference_text": "Appendix A",
      "reference_type": "APPENDIX",
      "target_label": "A",
      "confidence": 0.95
    },
    {
      "reference_text": "Annexure II",
      "reference_type": "APPENDIX",
      "target_label": "II",
      "confidence": 0.9
    }
  ]
}
```

### Example 5 — Footnote

**Input**

> The adjustment is explained in footnote 4 below.

**Output**

```json
{
  "references": [
    {
      "reference_text": "footnote 4",
      "reference_type": "FOOTNOTE",
      "target_label": "4",
      "confidence": 0.9
    }
  ]
}
```

### Example 6 — Financial statements

**Input**

> Refer to the Balance Sheet and Cash Flow Statement for details.

**Output**

```json
{
  "references": [
    {
      "reference_text": "Balance Sheet",
      "reference_type": "STATEMENT",
      "target_label": "Balance Sheet",
      "confidence": 0.95
    },
    {
      "reference_text": "Cash Flow Statement",
      "reference_type": "STATEMENT",
      "target_label": "Cash Flow Statement",
      "confidence": 0.95
    }
  ]
}
```

---

## Post-processing (recommended)

### Normalize labels

```python
def normalize_label(ref):
    return ref["target_label"].strip().upper()
```

### Deduplicate

```python
unique_refs = list({r["reference_text"]: r for r in refs}.values())
```

### Confidence filter

```python
refs = [r for r in refs if r["confidence"] > 0.7]
```

---

## Performance optimization

- Use small chunk sizes
- Include few-shot examples
- Enforce strict JSON output
- Avoid long instructions
