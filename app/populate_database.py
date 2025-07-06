import argparse
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from app.get_embedding_function import get_embedding_function
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import hashlib
import json

CHROMA_PATH = "chroma_vectordb"
DATA_PATH = "docs"
HASH_RECORD_FILE = "pdf_hashes.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the Chroma vector store.")
    args = parser.parse_args()

    if args.reset:
        print(" Resetting Chroma vector store...")
        clear_database()
        return

    print(" Loading HR Policy Documents...")
    documents = load_documents()

    print(f" Splitting {len(documents)} documents into chunks...")
    chunks = split_documents(documents)

    print("Assigning unique chunk IDs...")
    chunks_with_ids = assign_chunk_ids(chunks)

    # print("Adding chunks with id's to txt file...")
    # export_chunks_to_txt(chunks_with_ids, output_file="chunks_debug_output.txt")

    print("Embedding and storing in Chroma...")
    add_to_chroma(chunks_with_ids)

# def load_documents() -> list[Document]:
#     loader = PyPDFDirectoryLoader(DATA_PATH)
#     docs = loader.load()

#     for doc in docs:
#         if not doc.page_content.strip():
#             print(f"[⚠️] Empty page found: {doc.metadata.get('source')} page {doc.metadata.get('page')}")
#     return docs

def load_documents() -> list[Document]:
    loader = PyPDFDirectoryLoader(DATA_PATH)
    all_docs = loader.load()

    current_hashes = load_hashes()
    new_hashes = {}
    updated_docs = []

    for doc in all_docs:
        source = doc.metadata.get("source")
        file_path = os.path.join(DATA_PATH, os.path.basename(source))

        file_hash = get_file_hash(file_path)
        new_hashes[source] = file_hash

        if current_hashes.get(source) != file_hash:
            updated_docs.append(doc)
        else:
            print(f"[⏩] Skipping unchanged file: {source}")

    save_hashes(new_hashes)

    print(f"[📄] Loaded {len(updated_docs)} updated documents.")
    return updated_docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.split_documents(documents)

    #Filter out empty or whitespace-only content
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    print(f"[+] Filtered non-empty chunks: {len(chunks)}")
    return chunks

def assign_chunk_ids(chunks: list[Document]) -> list[Document]:
    last_page_id = None
    chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown.pdf")
        page = chunk.metadata.get("page", "0")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            chunk_index += 1
        else:
            chunk_index = 0

        chunk_id = f"{current_page_id}:{chunk_index}"
        chunk.metadata["id"] = chunk_id

        last_page_id = current_page_id

    return chunks

# def export_chunks_to_txt(chunks: list[Document], output_file="chunk_output.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.metadata.get("id", f"chunk_{i}")
            f.write(f"--- Chunk {i} | ID: {chunk_id} ---\n")
            f.write(chunk.page_content.strip() + "\n\n")
    print(f"[📄] Exported {len(chunks)} chunks to {output_file}")



def load_hashes():
    if os.path.exists(HASH_RECORD_FILE):
        with open(HASH_RECORD_FILE, "r") as f:
            return json.load(f)
    return {}

def save_hashes(hash_dict):
    with open(HASH_RECORD_FILE, "w") as f:
        json.dump(hash_dict, f, indent=4)

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def add_to_chroma(chunks: list[Document]):
    
    if not chunks:
        print("[🚫] No chunks to embed. Skipping Chroma update.")
        return
    
    embedding = get_embedding_function()

    print(f"Generating embeddings for {len(chunks)} chunks...")
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [meta["id"] for meta in metadatas]  # 🆕 Extract all IDs

    print(f"Connecting to persistent Chroma DB at {CHROMA_PATH}")
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="rag_collection")

    existing_items = collection.get(ids=ids)  # 🆕 Fetch only those IDs
    existing_ids = set(existing_items.get("ids", []))  # 🆕 Set for fast lookup
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing_ids]  # 🆕 Only new ones

    if not new_indices:
        print("✅ No new documents to add.")
        return

    print(f"👉 New chunks to add: {len(new_indices)}")
    texts_to_embed = [texts[i] for i in new_indices]
    metadatas_to_embed = [metadatas[i] for i in new_indices]
    ids_to_embed = [ids[i] for i in new_indices]

    embeddings = embedding.embed_documents(texts_to_embed)  # 🧠 Embed only new

    valid_texts = []
    valid_embeddings = []
    valid_metadatas = []
    valid_ids = []

    for i, emb in enumerate(embeddings):
        if emb and isinstance(emb, list) and len(emb) > 0:
            valid_texts.append(texts_to_embed[i])
            valid_embeddings.append(emb)
            valid_metadatas.append(metadatas_to_embed[i])
            valid_ids.append(ids_to_embed[i])
        else:
            print(f"Skipping chunk (embedding failed): {ids_to_embed[i]}")

    if not valid_embeddings:
        print("⚠️ No valid embeddings found. Skipping DB write.")
        return

    collection.add(  # ✅ Only new chunks added
        ids=valid_ids,
        embeddings=valid_embeddings,
        documents=valid_texts,
        metadatas=valid_metadatas
    )

    print("✅ New documents added to Chroma successfully.")


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Cleared {CHROMA_PATH} directory.")
    else:
        print("No existing Chroma DB to clear.")

if __name__ == "__main__":
    main()