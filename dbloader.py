import config as c
import json
import logging
import pathlib
from docx import Document
import easyocr
from spire.doc import Document as SpireDocument

# Paths for the output JSON file and the folder containing Word documents
outputfile = pathlib.Path("textjson/text.json")

def load_documents():

    folder = pathlib.Path("corpus")
    folder.mkdir(parents=True, exist_ok=True)

    # Make sure the output folder exists
    outputfile.parent.mkdir(parents=True, exist_ok=True)

    # Create the JSON file if it doesn't exist
    if not outputfile.exists():
        with open(outputfile, "w") as f:
            json.dump([], f)

    # Store the documents
    documents = []
    
    # to read text from images using OCR
    reader = easyocr.Reader(["en"], gpu=False)    

    # Read each Word document
    try:
        logging.info("Starting document extraction")
        for file in folder.iterdir():

            # for docx files
            if file.suffix.lower() == ".docx":
                doc = Document(file)
                text = "\n".join([p.text for p in doc.paragraphs])

            # for doc files
            elif file.suffix.lower() == ".doc":
                doc = SpireDocument()
                try:
                    doc.LoadFromFile(str(file))
                    text = doc.GetText()
                finally:
                    doc.Close()

            #for png files
            elif file.suffix.lower() == ".png":
                results = reader.readtext(str(file), detail=0)
                text = "\n".join(results)

            else:
                continue

            documents.append({
                "file": str(file),
                "text": text
            })
            logging.info("Processed: %s", file.name)

        logging.info("Extracted %d documents", len(documents))

    # If any error occurs while processing a file, it will be raised with the file name included
    except Exception as e:
        raise Exception("Error processing file: " + str(file)) from e
    
    
    # Warn if no documents were loaded
    if not documents:
        logging.warning("No documents found in the corpus folder")
        return

        
    # Write the documents to the JSON file
    try:
        with open(outputfile, "w", encoding="utf-8") as f:
            json.dump(
                documents, 
                f, 
                indent=c.JSON_INDENT,
                ensure_ascii=c.JSON_ENSURE_ASCII,
                sort_keys=c.JSON_SORT_KEYS
            )
    # If any error occurs while writing to the JSON file, it will be raised with the file name included
    except Exception as e:
        raise Exception("Error writing to JSON file: " + str(outputfile)) from e
    

        

if __name__ == "__main__":
    load_documents()
