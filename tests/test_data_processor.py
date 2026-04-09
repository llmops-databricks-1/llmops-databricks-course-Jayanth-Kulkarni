"""Tests for stackoverflow_curator.data_processor module.

Only tests the static/pure helper methods that don't require Spark.
"""

import pytest

from stackoverflow_curator.data_processor import DataProcessor


# ---------------------------------------------------------------------------
# _split_by_headings
# ---------------------------------------------------------------------------


class TestSplitByHeadings:
    def test_basic_split(self):
        md = "## Section 1\nHello world\n## Section 2\nGoodbye world"
        sections = DataProcessor._split_by_headings(md)
        assert len(sections) == 2
        assert sections[0] == ("Section 1", "Hello world")
        assert sections[1] == ("Section 2", "Goodbye world")

    def test_intro_before_heading(self):
        md = "Some intro text\n## First Heading\nContent"
        sections = DataProcessor._split_by_headings(md)
        assert len(sections) == 2
        assert sections[0][0] == "Introduction"
        assert sections[0][1] == "Some intro text"
        assert sections[1][0] == "First Heading"

    def test_h3_headings(self):
        md = "### Sub Section\nContent here"
        sections = DataProcessor._split_by_headings(md)
        assert sections[0][0] == "Sub Section"
        assert sections[0][1] == "Content here"

    def test_empty_sections_skipped(self):
        md = "## A\n\n## B\nContent"
        sections = DataProcessor._split_by_headings(md)
        # Section A has empty body → skipped
        assert len(sections) == 1
        assert sections[0][0] == "B"

    def test_multiline_body(self):
        md = "## Title\nLine 1\nLine 2\nLine 3"
        sections = DataProcessor._split_by_headings(md)
        assert sections[0][1] == "Line 1\nLine 2\nLine 3"


# ---------------------------------------------------------------------------
# _clean_markdown
# ---------------------------------------------------------------------------


class TestCleanMarkdown:
    def test_remove_images(self):
        text = "Before ![alt text](http://img.png) After"
        result = DataProcessor._clean_markdown(text)
        assert "![" not in result
        assert "Before" in result
        assert "After" in result

    def test_convert_links(self):
        text = "See [AutoModel](https://hf.co/docs) for details"
        result = DataProcessor._clean_markdown(text)
        assert result == "See AutoModel for details"

    def test_remove_html_tags(self):
        text = "Hello <br> World <div>content</div>"
        result = DataProcessor._clean_markdown(text)
        assert "<br>" not in result
        assert "<div>" not in result
        assert "Hello" in result

    def test_remove_code_fence_markers(self):
        text = "```python\nprint('hello')\n```"
        result = DataProcessor._clean_markdown(text)
        assert "```" not in result
        assert "print('hello')" in result

    def test_collapse_blank_lines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = DataProcessor._clean_markdown(text)
        assert "\n\n\n" not in result

    def test_collapse_whitespace(self):
        text = "Too   many    spaces"
        result = DataProcessor._clean_markdown(text)
        assert result == "Too many spaces"


# ---------------------------------------------------------------------------
# _fixed_size_split
# ---------------------------------------------------------------------------


class TestFixedSizeSplit:
    def test_short_text_single_chunk(self):
        proc = DataProcessor.__new__(DataProcessor)
        proc.max_chunk_chars = 100
        proc.overlap_chars = 20
        result = proc._fixed_size_split("short text")
        assert len(result) == 1
        assert result[0] == "short text"

    def test_long_text_multiple_chunks(self):
        proc = DataProcessor.__new__(DataProcessor)
        proc.max_chunk_chars = 50
        proc.overlap_chars = 10
        text = "A" * 120
        result = proc._fixed_size_split(text)
        assert len(result) == 3  # 0-50, 40-90, 80-120

    def test_overlap_exists(self):
        proc = DataProcessor.__new__(DataProcessor)
        proc.max_chunk_chars = 20
        proc.overlap_chars = 5
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        result = proc._fixed_size_split(text)
        # Second chunk should start 5 chars before end of first
        assert result[0][-5:] == result[1][:5]


# ---------------------------------------------------------------------------
# chunk_document (integration of helpers)
# ---------------------------------------------------------------------------


class TestChunkDocument:
    def _make_processor(self, max_chunk=2000, overlap=200):
        proc = DataProcessor.__new__(DataProcessor)
        proc.max_chunk_chars = max_chunk
        proc.overlap_chars = overlap
        return proc

    def test_basic_document(self):
        proc = self._make_processor()
        md = (
            "## Quick Tour\n"
            "The pipeline API is the easiest way to use models.\n"
            "## Installation\n"
            "Install with pip: `pip install transformers`"
        )
        chunks = proc.chunk_document("quicktour", md)
        assert len(chunks) == 2
        # Check chunk_id format
        assert chunks[0][0] == "quicktour_0"
        assert chunks[1][0] == "quicktour_1"
        # Check section titles preserved
        assert chunks[0][1] == "Quick Tour"
        assert chunks[1][1] == "Installation"

    def test_oversized_section_split(self):
        proc = self._make_processor(max_chunk=50, overlap=10)
        md = "## Big Section\n" + "A" * 120
        chunks = proc.chunk_document("doc", md)
        assert len(chunks) > 1
        # All chunks should reference the same section
        for _, section_title, _ in chunks:
            assert section_title == "Big Section"

    def test_empty_document(self):
        proc = self._make_processor()
        chunks = proc.chunk_document("empty", "")
        assert chunks == []

    def test_markdown_cleaned_in_chunks(self):
        proc = self._make_processor()
        md = "## Links\nSee [AutoModel](https://hf.co) for details"
        chunks = proc.chunk_document("doc", md)
        assert len(chunks) == 1
        assert "AutoModel" in chunks[0][2]
        assert "](https://" not in chunks[0][2]
