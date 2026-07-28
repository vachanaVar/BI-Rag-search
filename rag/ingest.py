"""
Ingestion: load raw documents from disk and split them into overlapping chunks.
"""

import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable

# Check for optional dependencies
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("⚠️  pdfplumber not installed. Install with: pip install pdfplumber")

try:
    from bs4 import BeautifulSoup
    HAS_HTML = True
except ImportError:
    HAS_HTML = False
    print("⚠️  BeautifulSoup not installed. HTML loading disabled. Install with: pip install beautifulsoup4")


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    text: str
    source_file: str = ""
    file_type: str = ""
    chunk_index: int = 0
    num_pages: int = 0


# ============ HELPERS ============

def get_title_from_filename(filepath: str) -> str:
    """Extract title from filename."""
    return os.path.splitext(os.path.basename(filepath))[0].replace("_", " ").title()


def make_metadata(filepath: str, file_type: str, extra: dict = None) -> dict:
    """Create metadata dict with common fields."""
    meta = {
        "source_file": os.path.basename(filepath),
        "file_type": file_type,
    }
    if extra:
        meta.update(extra)
    return meta


def extract_facts_text(item: dict) -> str:
    """Extract text from a facts list or dict."""
    if "facts" in item and isinstance(item["facts"], list):
        return " ".join(item["facts"])
    elif "text" in item:
        return item["text"]
    else:
        return json.dumps(item, indent=2)


# ============ SINGLE DOCUMENT LOADER ============

def load_documents(filepath: str) -> List[Dict]:
    """Load documents from a file or folder."""
    if os.path.isdir(filepath):
        docs = []
        for filename in os.listdir(filepath):
            file_path = os.path.join(filepath, filename)
            if os.path.isfile(file_path):
                docs.extend(load_single_document(file_path))
        return docs
    elif os.path.isfile(filepath):
        return load_single_document(filepath)
    else:
        print(f"❌ Path not found: {filepath}")
        return []


def load_single_document(filepath: str) -> List[Dict]:
    """Route to appropriate loader based on extension."""
    ext = os.path.splitext(filepath)[1].lower()
    loaders = {
        ".json": load_json_file,
        ".html": load_html_file,
        ".htm": load_html_file,
        ".pdf": load_pdf_file,
        ".txt": load_txt_file,
        ".md": load_md_file,
        ".markdown": load_md_file,
    }
    loader = loaders.get(ext)
    if loader:
        return loader(filepath)
    print(f"⏭️  Skipping unsupported: {os.path.basename(filepath)}")
    return []


# ============ FILE LOADERS ============

def load_json_file(filepath: str) -> List[Dict]:
    """Load JSON file(s) with metadata."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        docs = []
        for item in data:
            title = item.get("species", item.get("title", get_title_from_filename(filepath)))
            text = extract_facts_text(item)

            docs.append({
                "title": title,
                "text": text,
                "metadata": make_metadata(filepath, "json", {
                    "species": item.get("species", ""),
                    "genus": item.get("genome", {}).get("details", {}).get("genus", ""),
                    "accession": item.get("genome", {}).get("details", {}).get("accession", ""),
                })
            })
        return docs
    except Exception as e:
        print(f"❌ Error loading JSON {filepath}: {e}")
        return []


def load_txt_file(filepath: str) -> List[Dict]:
    """Load text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return []
        return [{
            "title": get_title_from_filename(filepath),
            "text": text,
            "metadata": make_metadata(filepath, "txt")
        }]
    except Exception as e:
        print(f"❌ Error loading TXT {filepath}: {e}")
        return []


def load_html_file(filepath: str) -> List[Dict]:
    """Load HTML file."""
    if not HAS_HTML:
        print(f"⚠️  BeautifulSoup not installed")
        return []

    try:
        soup = BeautifulSoup(open(filepath, "r", encoding="utf-8"), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        if not text:
            return []

        html_title = soup.title.string.strip() if soup.title and soup.title.string else ""
        title = html_title if html_title else get_title_from_filename(filepath)

        return [{
            "title": title,
            "text": text,
            "metadata": make_metadata(filepath, "html", {"html_title": html_title})
        }]
    except Exception as e:
        print(f"❌ Error loading HTML {filepath}: {e}")
        return []


def load_pdf_file(filepath: str) -> List[Dict]:
    """Load PDF file."""
    if not HAS_PDF:
        print(f"❌ pdfplumber not installed")
        return []

    try:
        text_parts = []
        metadata = make_metadata(filepath, "pdf")

        with pdfplumber.open(filepath) as pdf:
            metadata["num_pages"] = len(pdf.pages)

            if pdf.metadata:
                for key in ['Title', 'Author', 'Subject']:
                    if pdf.metadata.get(key):
                        metadata[key.lower()] = pdf.metadata.get(key)

            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(f"--- Page {page_num} ---\n{text.strip()}")

        if not text_parts:
            print(f"⚠️  No text extracted from PDF: {filepath}")
            return []

        full_text = "\n\n".join(text_parts)
        title = metadata.get("title", get_title_from_filename(filepath))
        metadata["extraction_method"] = "pdfplumber"

        return [{"title": title, "text": full_text, "metadata": metadata}]
    except Exception as e:
        print(f"❌ Error loading PDF {filepath}: {e}")
        return []


def load_md_file(filepath: str) -> List[Dict]:
    """Load Markdown file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return []
        return [{
            "title": get_title_from_filename(filepath),
            "text": text,
            "metadata": make_metadata(filepath, "markdown")
        }]
    except Exception as e:
        print(f"❌ Error loading Markdown {filepath}: {e}")
        return []


# ============ CHUNKING ============

def chunk_text_word_based(text: str, chunk_size: int = 80, overlap: int = 20) -> List[str]:
    """Simple word-based chunking."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def chunk_text_sentences(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Sentence-aware chunking."""
    if not text or not text.strip():
        return []

    # Handle abbreviations
    abbr = r'\b(?:Mr|Mrs|Ms|Dr|Prof|Rev|Gen|Col|Sgt|Capt|Jr|Sr|vs|etc|e\.g|i\.e|al|St|Ave|Blvd|Rd|Mt)\b'
    def replace_abbr(m): return m.group(0).replace('.', '☃')

    text_temp = re.sub(abbr, replace_abbr, text)
    sentences = [s.replace('☃', '.').strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])', text_temp) if s.strip()]

    if not sentences:
        return [text]

    chunks, current_chunk, current_len = [], [], 0

    for sentence in sentences:
        s_len = len(sentence) + 1
        if s_len > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_len = [], 0

            # Split long sentence
            parts = re.split(r'(?<=[,;])\s+', sentence) if ',' in sentence or ';' in sentence else sentence.split()
            temp_chunk, temp_len = [], 0

            for part in parts:
                p_len = len(part) + 1
                if temp_len + p_len > chunk_size and temp_chunk:
                    chunks.append(" ".join(temp_chunk))
                    temp_chunk, temp_len = [], 0
                temp_chunk.append(part)
                temp_len += p_len
            if temp_chunk:
                chunks.append(" ".join(temp_chunk))
            continue

        if current_len + s_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_chunk, overlap_len = [], 0
            for s_rev in reversed(current_chunk):
                r_len = len(s_rev) + 1
                if overlap_len + r_len <= overlap:
                    overlap_chunk.insert(0, s_rev)
                    overlap_len += r_len
                else:
                    break
            current_chunk, current_len = overlap_chunk, overlap_len

        current_chunk.append(sentence)
        current_len += s_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def chunk_text_hybrid(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Hybrid: try sentence, fallback to word-based."""
    chunks = chunk_text_sentences(text, chunk_size, overlap)
    if len(chunks) <= 1:
        return chunk_text_word_based(text, chunk_size, overlap)
    return chunks


def build_chunk_records(
    docs: List[Dict],
    chunk_size: int = 500,
    overlap: int = 50,
    chunker: str = "hybrid"
) -> List[Chunk]:
    """Build chunk records from documents."""
    chunkers = {
        "sentence": chunk_text_sentences,
        "word": chunk_text_word_based,
        "hybrid": chunk_text_hybrid,
    }
    chunk_func = chunkers.get(chunker, chunk_text_hybrid)

    records = []
    for doc in docs:
        pieces = chunk_func(doc["text"], chunk_size=chunk_size, overlap=overlap)
        meta = doc.get("metadata", {})
        for i, piece in enumerate(pieces):
            records.append(Chunk(
                chunk_id=f"{doc['title']}::{i}",
                doc_title=doc["title"],
                text=piece,
                source_file=meta.get("source_file", ""),
                file_type=meta.get("file_type", ""),
                chunk_index=i,
                num_pages=meta.get("num_pages", 0),
            ))
    return records


# ============ FOLDER LOADER (Alias) ============

def load_documents_from_folder(folder: str) -> List[Dict]:
    """Alias for load_documents."""
    return load_documents(folder)


# ============ MAIN ============
if __name__ == "__main__":
    folder_path = "C:/final_project_starter/data/sample_docs/"

    try:
        docs = load_documents_from_folder(folder_path)
        print(f"\n✅ Loaded {len(docs)} documents total")

        chunks = build_chunk_records(docs, chunk_size=500, overlap=100, chunker="sentence")
        print(f"✅ Created {len(chunks)} chunks using sentence chunker")

        print("\n📄 First 3 chunks:")
        for chunk in chunks[:3]:
            print(f"  - ID: {chunk.chunk_id}")
            print(f"    Title: {chunk.doc_title}")
            print(f"    Source: {chunk.source_file}")
            print(f"    Type: {chunk.file_type}")
            if chunk.num_pages:
                print(f"    Pages: {chunk.num_pages}")
            print(f"    Text: {chunk.text[:150]}...")
            print()

        print("\n📊 Statistics:")
        type_stats = {}
        for chunk in chunks:
            type_stats[chunk.file_type] = type_stats.get(chunk.file_type, 0) + 1
        for ftype, count in type_stats.items():
            print(f"  - {ftype}: {count} chunks")

        if chunks:
            sizes = [len(c.text) for c in chunks]
            print(f"\n📊 Chunk Size: Avg {sum(sizes)/len(sizes):.0f}, Min {min(sizes)}, Max {max(sizes)}")

    except Exception as e:
        print(f"❌ Error: {e}")