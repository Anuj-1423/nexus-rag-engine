"""
Chunking Engine
===============
Advanced chunking strategies including Semantic Chunking and Recursive Character splitting.
"""

import logging
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

from document_parser import DocumentStructure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Toggle between 'semantic' and 'recursive'
CHUNKING_STRATEGY = "semantic"  

# Recursive Splitter Config
CHUNK_SIZE = 1000           # Characters per chunk
CHUNK_OVERLAP = 200         # Overlap between adjacent chunks
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Semantic Chunking Config
# Options: "percentile", "standard_deviation", "interquartile", "gradient"
BREAKPOINT_THRESHOLD_TYPE = "percentile"
BREAKPOINT_THRESHOLD_VALUE = 95  # 95th percentile of distances as split points


# ---------------------------------------------------------------------------
# Main Chunking Function
# ---------------------------------------------------------------------------

def chunk_document(doc_structure: DocumentStructure, embeddings=None) -> List[Document]:
    """
    Convert a DocumentStructure into a list of LangChain Documents using either
    Semantic or Recursive chunking.
    """
    
    if CHUNKING_STRATEGY == "semantic" and embeddings:
        logger.info(f"Using Semantic Chunking (Threshold: {BREAKPOINT_THRESHOLD_VALUE} {BREAKPOINT_THRESHOLD_TYPE})")
        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=BREAKPOINT_THRESHOLD_TYPE,
            breakpoint_threshold_amount=BREAKPOINT_THRESHOLD_VALUE
        )
    else:
        logger.info(f"Using Recursive Character Chunking (Size: {CHUNK_SIZE})")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
            length_function=len,
        )

    all_chunks: List[Document] = []
    
    # If we have sections, process each section separately to maintain boundaries
    if doc_structure.sections:
        for section in doc_structure.sections:
            if not section.content.strip():
                continue

            metadata = {
                "filename": doc_structure.filename,
                "doc_title": doc_structure.title,
                "section_heading": section.heading,
                "page_number": section.page_number
            }
            
            # SemanticChunker.create_documents handles the splitting
            sub_chunks = splitter.create_documents(
                [section.content],
                metadatas=[metadata],
            )
            all_chunks.extend(sub_chunks)
    else:
        # Fallback to raw text
        all_chunks = splitter.create_documents(
            [doc_structure.raw_text],
            metadatas=[{
                "filename": doc_structure.filename,
                "doc_title": doc_structure.title,
                "page_number": 1
            }],
        )

    return all_chunks


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_chunking_stats(chunks: List[Document]) -> dict:
    """Return summary statistics about the chunking result."""
    if not chunks:
        return {"total_chunks": 0, "avg_chunk_size": 0}

    total_length = sum(len(c.page_content) for c in chunks)

    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": round(total_length / len(chunks)),
        "total_characters": total_length,
    }
