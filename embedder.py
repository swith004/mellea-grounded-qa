import ollama
import chromadb
import json

#initialize ChromaDB client
client = chromadb.PersistentClient("./.chromadb")

EMBED_MODEL = "nomic-embed-text"
CHUNKS_FILE = "chunkeddata/chunked_data.json"

collection = client.get_or_create_collection("chunks")

def build_index():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    response = ollama.embed(
        model=EMBED_MODEL,
        input=["search_document: " + t for t in texts],
    )

    collection.upsert(
        ids=[f'{c["file"]}-{c["chunk_id"]}' for c in chunks],
        documents=texts,
        embeddings=response["embeddings"],
        metadatas=[{"file": c["file"], "chunk_id": c["chunk_id"]} for c in chunks],
    )
    print(f"Indexed {len(chunks)} chunks")

def retrieve(question, k=5):
    q = ollama.embed(model=EMBED_MODEL, input="search_query: " + question)

    results = collection.query(
        query_embeddings=q["embeddings"],
        n_results=k,
    )

    passages = []
    for text, meta, dist in zip(results["documents"][0],
                                results["metadatas"][0],
                                results["distances"][0]):
        passages.append({"file": meta["file"], "chunk_id": meta["chunk_id"],
                         "distance": dist, "text": text})
    return passages

if __name__ == "__main__":
    if collection.count() == 0:
        build_index()

    question = "Which course teaches TCP/IP networking?"
    for p in retrieve(question):
        print(f'[{p["file"]} #{p["chunk_id"]}] distance={p["distance"]:.3f}')
        print(p["text"][:200], "\n")
