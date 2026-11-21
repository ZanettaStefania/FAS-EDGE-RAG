"""Rebuild Chroma databases from current PDFs (not legacy-comparable)."""

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


logging.Formatter.converter = time.gmtime
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03dZ] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DATABASE_ROOT = ROOT / "DB"
PDF_ROOT = ROOT / "PDF_files"
DATABASES = {
    "new500_db": (500, "BAAI/bge-small-en-v1.5"),
    "new1000_db": (1000, "BAAI/bge-small-en-v1.5"),
    "new1500_db": (1500, "BAAI/bge-small-en-v1.5"),
    "new500_db_med": (500, "abhinand/MedEmbed-small-v0.1"),
    "new1000_db_med": (1000, "abhinand/MedEmbed-small-v0.1"),
    "new1500_db_med": (1500, "abhinand/MedEmbed-small-v0.1"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="Embedding device (default: cuda).")
    parser.add_argument(
        "--allow-non-comparable-rebuild",
        action="store_true",
        help="Confirm that replacing the migrated legacy chunks is intentional.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=DATABASES,
        default=list(DATABASES),
        help="Databases to rebuild (default: all).",
    )
    return parser.parse_args()


def archive_databases(database_names):
    existing = [DATABASE_ROOT / name for name in database_names if (DATABASE_ROOT / name).exists()]
    if not existing:
        return None

    archive_root = ROOT / "DB_old"
    archive_root.mkdir(exist_ok=True)
    archive_directory = archive_root / datetime.now().strftime("%Y%m%dT%H%M%S")
    archive_directory.mkdir()
    for database_path in existing:
        destination = archive_directory / database_path.name
        logger.info("Archiving %s -> %s", database_path, destination)
        shutil.move(str(database_path), str(destination))
    return archive_directory


def load_documents():
    loader = DirectoryLoader(
        str(PDF_ROOT),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    if not documents:
        raise RuntimeError(f"No PDF pages found in {PDF_ROOT}")
    logger.info("Loaded %s PDF pages", len(documents))
    return documents


def rebuild_database(name, chunk_size, model_name, documents, device):
    destination = DATABASE_ROOT / name
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    logger.info(
        "Building %s: %s chunks, chunk_size=%s, embedding=%s",
        name, len(chunks), chunk_size, model_name,
    )
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(destination),
        collection_name="langchain",
    )
    count = vectorstore._collection.count()
    logger.info("Created %s with %s embeddings", destination, count)
    if count != len(chunks):
        raise RuntimeError(f"{name}: expected {len(chunks)} embeddings, found {count}")


def main():
    args = parse_args()
    if not args.allow_non_comparable_rebuild:
        raise SystemExit(
            "Refusing to replace the legacy-comparable databases. Pass "
            "--allow-non-comparable-rebuild only when different chunks are acceptable."
        )
    database_names = list(dict.fromkeys(args.only))
    archive_directory = archive_databases(database_names)
    if archive_directory:
        logger.info("Legacy databases preserved in %s", archive_directory)

    documents = load_documents()
    for name in database_names:
        chunk_size, model_name = DATABASES[name]
        rebuild_database(name, chunk_size, model_name, documents, args.device)
    logger.info("All requested databases rebuilt successfully.")


if __name__ == "__main__":
    main()


# Get process ID

