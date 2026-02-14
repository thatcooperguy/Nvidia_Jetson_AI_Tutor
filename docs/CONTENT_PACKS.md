# EdgeTutor AI — Content Packs Guide

## What Are Content Packs?

Content packs are collections of educational documents (PDF, TXT, Markdown)
that EdgeTutor can use to provide curriculum-specific answers. When a student
asks a question, EdgeTutor searches the content packs for relevant material
and uses it to inform its response.

This is called **RAG (Retrieval-Augmented Generation)** — the tutor "retrieves"
relevant content before generating its answer, making responses more accurate
and grounded in your chosen curriculum.

## How It Works

1. You place documents in the `content/` folder
2. Run the ingest script to index them
3. EdgeTutor automatically searches the index when answering questions

## Adding Content

### Step 1: Place Documents

Put your files in the `content/` directory (or any subdirectory):

```
content/
├── starter_pack/          # Included with EdgeTutor
│   ├── basic_math.md
│   ├── basic_science.md
│   └── reading_comprehension.md
├── grade_5_math/          # Your custom pack
│   ├── chapter_1.pdf
│   ├── chapter_2.pdf
│   └── vocabulary.txt
└── science_curriculum/
    ├── biology_notes.md
    └── chemistry_basics.txt
```

### Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain text | `.txt` | Best for simple content |
| Markdown | `.md` | Supports formatting |
| PDF | `.pdf` | Extracts text from pages |

### Step 2: Run Ingestion

```bash
./scripts/ingest_content.sh
```

Or for a specific directory:
```bash
./scripts/ingest_content.sh /path/to/your/content
```

This process:
1. Reads all documents
2. Splits them into chunks (~500 characters each)
3. Creates embeddings using the AI model
4. Stores everything in a local FAISS index

### Step 3: Use It

Just ask questions! EdgeTutor will automatically search the content packs.

Example:
- **Student**: "What is photosynthesis?"
- **EdgeTutor** *(with science content pack)*: Uses your specific curriculum
  material to explain photosynthesis in age-appropriate language.

## Creating Good Content Packs

### Tips

1. **Be specific**: Content about "Grade 5 fractions" is better than
   "all of math"
2. **Use clear headings**: Helps with chunking and retrieval
3. **Keep it factual**: The tutor treats your content as ground truth
4. **Use the right format**:
   - Markdown for structured content (headings, lists, tables)
   - PDF for textbook pages
   - TXT for simple text

### Example: Creating a Math Pack

```markdown
# Fractions for Grade 5

## What is a Fraction?
A fraction represents a part of a whole. It has two numbers:
- **Numerator**: The top number (how many parts we have)
- **Denominator**: The bottom number (how many equal parts in total)

Example: 3/4 means 3 out of 4 equal parts.

## Adding Fractions with Same Denominators
To add fractions with the same denominator:
1. Add the numerators
2. Keep the denominator the same
3. Simplify if possible

Example: 1/5 + 2/5 = 3/5
```

## Configuration

In `.env`:

```bash
# Content directory (all subdirectories are scanned)
RAG_CONTENT_DIR=content/

# Index storage location
RAG_INDEX_DIR=edgetutor/core/faiss_index/

# Number of content chunks to retrieve per query
RAG_TOP_K=3

# Chunk size in characters (smaller = more precise, larger = more context)
RAG_CHUNK_SIZE=500
```

## Included Starter Pack

EdgeTutor ships with a small starter pack (`content/starter_pack/`) containing
public domain educational content:

- **basic_math.md**: Arithmetic, fractions, PEMDAS
- **basic_science.md**: Scientific method, states of matter, solar system
- **reading_comprehension.md**: Main idea, inferences, vocabulary

These are meant to demonstrate the pipeline. Replace or supplement them with
your own curriculum materials.

## Re-Indexing

If you add, remove, or modify content, re-run the ingest script:

```bash
./scripts/ingest_content.sh
```

The old index is replaced entirely. This is fast (usually under a minute
for typical curriculum-sized packs).

## Public Domain Content Sources

Looking for free educational content to use? These sources offer
public domain or openly licensed material:

- **Project Gutenberg**: Classic literature and reference texts
- **CK-12**: Free STEM textbooks (check license)
- **OpenStax**: Free peer-reviewed textbooks
- **Khan Academy** transcripts (personal use)
- **Simple English Wikipedia**: Great for younger students

Always verify the license before distributing content packs.
