# Databricks notebook source

import time
from datetime import datetime

import backoff
import requests
import yaml
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import BooleanType, StringType, StructField, StructType

from stackoverflow_curator.config import get_env, load_config


# COMMAND ----------
# Create Spark session and load config

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
TABLE_NAME = "hf_doc_sources"

table_path = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

# Source: _toctree.yml defines all doc pages; raw markdown files are on GitHub.
# Using GitHub raw content avoids the HuggingFace docs site rate limits entirely.
TOCTREE_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers"
    "/main/docs/source/en/_toctree.yml"
)
GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/huggingface/transformers"
    "/main/docs/source/en"
)
HF_DOCS_BASE = "https://huggingface.co/docs/transformers"

# Seconds to wait between fetch requests (polite crawling)
REQUEST_DELAY = 0.5

logger.info(f"Target table: {table_path}")

# COMMAND ----------
# Discover all guide pages by parsing _toctree.yml from the transformers repo.
# API reference sections (model_doc/, main_classes/, internal/) are excluded.

# URL path prefixes that indicate API reference pages (not useful for RAG)
API_PREFIXES = ("model_doc/", "main_classes/", "internal/", "quantization/")


def _extract_pages(items: list, category: str) -> list[dict]:
    """Recursively extract page entries from a _toctree.yml section tree."""
    results = []
    for item in items:
        if "local" in item:
            local = item["local"]
            if any(local.startswith(p) for p in API_PREFIXES):
                continue
            slug = local.replace("/", "_")
            results.append(
                {
                    "id": slug,
                    "local": local,  # path relative to docs/source/en/
                    "url": f"{HF_DOCS_BASE}/{local}",
                    "title": item.get("title", slug),
                    "primary_category": category,
                }
            )
        # Recurse into nested sub-sections
        if "sections" in item:
            sub_category = item.get("title", category)
            results.extend(_extract_pages(item["sections"], sub_category))
    return results


def discover_doc_urls(toctree_url: str) -> list[dict]:
    """
    Fetch _toctree.yml from the HuggingFace transformers GitHub repo and
    return a flat list of all guide pages with their URLs and categories.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LLMOps-Course/1.0)"}
    resp = requests.get(toctree_url, headers=headers, timeout=30)
    resp.raise_for_status()

    toctree = yaml.safe_load(resp.text)

    results = []
    for section in toctree:
        title = section.get("title", "General")
        # Skip top-level API reference sections entirely
        if "API" in title or title in ("Quantization", "Internal Helpers"):
            continue
        if "sections" in section:
            results.extend(_extract_pages(section["sections"], title))

    return results


discovered = discover_doc_urls(TOCTREE_URL)
logger.info(f"Discovered {len(discovered)} guide pages from _toctree.yml")
for doc in discovered[:5]:
    logger.info(f"  [{doc['primary_category']}] {doc['title']}")
logger.info(f"  ... and {len(discovered) - 5} more")

# COMMAND ----------
# Idempotency: skip URLs already successfully fetched in the Delta table.

already_fetched_urls: set[str] = set()

if spark.catalog.tableExists(table_path):
    already_fetched_urls = {
        row.url
        for row in spark.table(table_path)
        .filter("is_fetched = true")
        .select("url")
        .collect()
    }
    logger.info(f"Already fetched: {len(already_fetched_urls)} page(s)")
else:
    logger.info("Table does not exist yet — will create on first run")

to_fetch = [d for d in discovered if d["url"] not in already_fetched_urls]
logger.info(f"Pages to fetch this run: {len(to_fetch)}")

# COMMAND ----------
# Fetch raw markdown from GitHub for each new page.
# GitHub raw content has no rate limits for public repos and returns clean
# markdown that is better for RAG than HTML-scraped text.

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LLMOps-Course/1.0)"}


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.HTTPError,
    max_tries=5,
    giveup=lambda e: e.response is not None and e.response.status_code not in (429, 503),
)
def fetch_hf_doc(local: str) -> str:
    """
    Fetch the raw markdown source for a HuggingFace Transformers doc page
    from GitHub (docs/source/en/{local}.md).
    Retries automatically on 429/503 with exponential backoff.
    """
    raw_url = f"{GITHUB_RAW_BASE}/{local}.md"
    resp = requests.get(raw_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


schema_struct = StructType(
    [
        StructField("id", StringType(), False),
        StructField("title", StringType(), False),
        StructField("url", StringType(), False),
        StructField("primary_category", StringType(), True),
        StructField("raw_content", StringType(), True),
        StructField("is_fetched", BooleanType(), False),
        StructField("fetched_at", StringType(), True),
        StructField("ingestion_timestamp", StringType(), False),
    ]
)

rows = []
for i, source in enumerate(to_fetch, 1):
    now = datetime.now().isoformat()
    try:
        content = fetch_hf_doc(source["local"])
        rows.append(
            {
                "id": source["id"],
                "title": source["title"],
                "url": source["url"],
                "primary_category": source["primary_category"],
                "raw_content": content,
                "is_fetched": True,
                "fetched_at": now,
                "ingestion_timestamp": now,
            }
        )
        logger.info(f"  [{i}/{len(to_fetch)}] OK  {source['title']}  ({len(content):,} chars)")
    except Exception as e:
        logger.warning(f"  [{i}/{len(to_fetch)}] FAIL  {source['url']}  — {e}")
        rows.append(
            {
                "id": source["id"],
                "title": source["title"],
                "url": source["url"],
                "primary_category": source["primary_category"],
                "raw_content": None,
                "is_fetched": False,
                "fetched_at": None,
                "ingestion_timestamp": now,
            }
        )
    time.sleep(REQUEST_DELAY)

# COMMAND ----------
# Write results to Delta table.

if not rows:
    logger.info("Nothing new to write — all discovered pages already fetched")
else:
    df = spark.createDataFrame(rows, schema=schema_struct)

    if spark.catalog.tableExists(table_path):
        df.write.format("delta").mode("append").saveAsTable(table_path)
        logger.info(f"Appended {len(rows)} row(s) to {table_path}")
    else:
        df.write.format("delta").mode("overwrite").saveAsTable(table_path)
        logger.info(f"Created {table_path} with {len(rows)} row(s)")

# COMMAND ----------
# Summary.

result_df = spark.table(table_path)
total = result_df.count()
fetched = result_df.filter("is_fetched = true").count()

logger.info(f"Table {table_path}: {fetched}/{total} pages fetched")

result_df.select("id", "title", "primary_category", "is_fetched", "fetched_at").orderBy(
    "primary_category", "id"
).show(50, truncate=60)

# COMMAND ----------
