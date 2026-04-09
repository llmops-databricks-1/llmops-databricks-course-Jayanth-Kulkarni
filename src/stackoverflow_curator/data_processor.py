"""
hf_doc_sources (raw markdown from Week 1 ingestion)
   ↓ (process_chunks)
hf_doc_chunks (cleaned, chunked text with metadata)
   ↓ (VectorSearchManager — separate class)
Vector Search Index (embeddings)
"""

import re

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

from stackoverflow_curator.config import ProjectConfig


class DataProcessor:
    """Processes raw HuggingFace Transformers markdown docs into chunks.

    Reads from the ``hf_doc_sources`` Bronze table (populated in Week 1),
    splits each document into semantic chunks by markdown heading, cleans
    the text, and writes the result to the ``hf_doc_chunks`` Silver table.

    The chunking strategy is **heading-aware**: each ``##`` section becomes
    its own chunk.  Sections longer than ``max_chunk_chars`` are further
    split using fixed-size windowing with overlap.
    """

    DEFAULT_MAX_CHUNK_CHARS = 2000
    DEFAULT_OVERLAP_CHARS = 200

    def __init__(
        self,
        spark: SparkSession,
        config: ProjectConfig,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    ) -> None:
        self.spark = spark
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

        self.sources_table = f"{self.catalog}.{self.schema}.hf_doc_sources"
        self.chunks_table = f"{self.catalog}.{self.schema}.hf_doc_chunks"

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_by_headings(markdown: str) -> list[tuple[str, str]]:
        """Split markdown into (heading, body) pairs on ``##`` boundaries.

        Top-level content before the first heading is assigned the heading
        ``"Introduction"``.
        """
        pattern = r"^(#{1,3})\s+(.+)$"
        sections: list[tuple[str, str]] = []
        current_heading = "Introduction"
        current_lines: list[str] = []

        for line in markdown.splitlines():
            m = re.match(pattern, line)
            if m:
                # Flush previous section
                if current_lines:
                    sections.append(
                        (current_heading, "\n".join(current_lines).strip())
                    )
                current_heading = m.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Flush last section
        if current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_heading, body))

        # Remove sections with empty bodies
        return [(h, b) for h, b in sections if b]

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Remove markdown artefacts that add noise for embeddings."""
        # Remove image tags ![alt](url)
        t = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        # Convert links [text](url) → text
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
        # Remove HTML tags
        t = re.sub(r"<[^>]+>", "", t)
        # Remove fenced code block markers (keep the code itself)
        t = re.sub(r"^```\w*\s*$", "", t, flags=re.MULTILINE)
        # Collapse repeated blank lines
        t = re.sub(r"\n{3,}", "\n\n", t)
        # Collapse repeated whitespace on a single line
        t = re.sub(r"[ \t]+", " ", t)
        return t.strip()

    def _fixed_size_split(self, text: str) -> list[str]:
        """Split text exceeding *max_chunk_chars* with overlap."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.max_chunk_chars
            chunks.append(text[start:end].strip())
            start += self.max_chunk_chars - self.overlap_chars
        return [c for c in chunks if c]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(
        self, doc_id: str, raw_content: str
    ) -> list[tuple[str, str, str]]:
        """Chunk a single document into ``(chunk_id, section_title, text)`` tuples."""
        sections = self._split_by_headings(raw_content)
        results: list[tuple[str, str, str]] = []
        chunk_idx = 0

        for heading, body in sections:
            cleaned = self._clean_markdown(body)
            if not cleaned:
                continue

            if len(cleaned) <= self.max_chunk_chars:
                results.append(
                    (f"{doc_id}_{chunk_idx}", heading, cleaned)
                )
                chunk_idx += 1
            else:
                for sub in self._fixed_size_split(cleaned):
                    results.append(
                        (f"{doc_id}_{chunk_idx}", heading, sub)
                    )
                    chunk_idx += 1

        return results

    def process_chunks(self) -> None:
        """Read ``hf_doc_sources``, chunk every fetched doc, write to ``hf_doc_chunks``."""
        logger.info(f"Reading source documents from {self.sources_table}")

        sources_df = (
            self.spark.table(self.sources_table)
            .filter("is_fetched = true")
            .select("id", "title", "url", "primary_category", "raw_content")
        )

        rows = sources_df.collect()
        logger.info(f"Found {len(rows)} fetched document(s) to chunk")

        chunk_rows: list[dict] = []
        for row in rows:
            chunks = self.chunk_document(row["id"], row["raw_content"])
            for chunk_id, section_title, text in chunks:
                chunk_rows.append(
                    {
                        "id": chunk_id,
                        "doc_id": row["id"],
                        "chunk_id": chunk_id,
                        "section_title": section_title,
                        "text": text,
                        "title": row["title"],
                        "url": row["url"],
                        "primary_category": row["primary_category"],
                    }
                )

        if not chunk_rows:
            logger.warning("No chunks produced — nothing to write")
            return

        logger.info(f"Produced {len(chunk_rows)} chunk(s)")

        chunk_schema = StructType(
            [
                StructField("id", StringType(), False),
                StructField("doc_id", StringType(), False),
                StructField("chunk_id", StringType(), False),
                StructField("section_title", StringType(), True),
                StructField("text", StringType(), True),
                StructField("title", StringType(), True),
                StructField("url", StringType(), True),
                StructField("primary_category", StringType(), True),
            ]
        )

        chunks_df = self.spark.createDataFrame(
            chunk_rows, schema=chunk_schema
        ).withColumn("chunked_at", current_timestamp())

        # Overwrite the whole table each run — idempotent
        chunks_df.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(self.chunks_table)
        logger.info(f"Wrote {len(chunk_rows)} chunk(s) to {self.chunks_table}")

        # Enable Change Data Feed for vector search delta-sync
        self.spark.sql(f"""
            ALTER TABLE {self.chunks_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        logger.info("Change Data Feed enabled")

    def process_and_save(self) -> None:
        """Full pipeline: chunk source documents and write to Delta."""
        self.process_chunks()
        logger.info("DataProcessor pipeline complete")
