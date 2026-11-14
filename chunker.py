import sys
import config as c
import os
import json
import logging
import pathlib
import dbloader as dbl
from langchain_text_splitters import RecursiveCharacterTextSplitter as rcts

# Destination for the chunked output consumed by later pipeline steps
chunk_data_file = pathlib.Path("chunkeddata/chunked_data.json")

def chunk_documents():

    # dbloader.py produces the input JSON, so it must run first
    if not dbl.outputfile.exists():
        raise FileNotFoundError("the json file doesn't exist. Try running dbloader.py first.")

    # Each document is expected to have "file" and "text" keys
    with open(dbl.outputfile, "r") as f:
        documents = json.load(f)
    
    if not documents:
        logging.warning("No documents available for chunking")
        return

    # Ensure the output directory exists before writing
    if not chunk_data_file.parent.exists():
        os.makedirs(chunk_data_file.parent)    
        
    # Sizes are measured in tokens (via tiktoken), not characters; overlap preserves context across chunk boundaries
    splitter = rcts.from_tiktoken_encoder(
        encoding_name=c.TOKENIZER,
        chunk_size=c.CHUNK_SIZE,
        chunk_overlap=c.CHUNK_OVERLAP
    )


    try:
        with open(chunk_data_file, "w", encoding="utf-8") as output:
            # Write the JSON array by hand so entries stream out without holding all chunks in memory
            output.write("[\n")
            first = True
            total_chunks = 0
            logging.info("Starting document chunking")
            
            for doc in documents:
                chunks = splitter.split_text(doc["text"])

                for chunk_id, chunk in enumerate(chunks):
                    # chunk_id is per-file; file + chunk_id identifies a chunk for citations
                    entry = {
                        "file": doc["file"],
                        "chunk_id": chunk_id,
                        "text": chunk
                    }

                    # Comma goes between entries, not before the first one
                    if not first:
                        output.write(",\n")
                    else:
                        first = False
                    
                    output.write(
                        json.dumps(entry, ensure_ascii=c.JSON_ENSURE_ASCII)
                    )
                    total_chunks += 1
            output.write("\n]")
        logging.info("Generated %d chunks", total_chunks)
    except FileNotFoundError as e:
        logging.error("Error: %s", e)
        



if __name__ == "__main__":
    chunk_documents()