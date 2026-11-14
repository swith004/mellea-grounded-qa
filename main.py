import logging

import dbloader
import chunker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)


def main():
    logging.info("Starting document processing pipeline")
    dbloader.load_documents()
    chunker.chunk_documents()
    logging.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()
