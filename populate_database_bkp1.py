import argparse
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from get_embedding_function import get_embedding_function
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

CHROMA_PATH = "chroma_vectordb"
DATA_PATH = "docs"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the Chroma vector store.")
    args = parser.parse_args()

    if args.reset:
        print(" Resetting Chroma vector store...")
        clear_database()

    print(" Loading HR Policy Documents...")
    documents = load_documents()

    print(f" Splitting {len(documents)} documents into chunks...")
    chunks = split_documents(documents)

    print("Assigning unique chunk IDs...")
    chunks_with_ids = assign_chunk_ids(chunks)

    print("Embedding and storing in Chroma...")
    add_to_chroma(chunks_with_ids)

def load_documents() -> list[Document]:
    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()

    for doc in docs:
        if not doc.page_content.strip():
            print(f"[⚠️] Empty page found: {doc.metadata.get('source')} page {doc.metadata.get('page')}")
    return docs

def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
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

def add_to_chroma(chunks: list[Document]):
    embedding = get_embedding_function()

    print(f"Generating embeddings for {len(chunks)} chunks...")
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    embeddings = embedding.embed_documents(texts)

    valid_texts = []
    valid_embeddings = []
    valid_metadatas = []
    valid_ids = []

    for i, emb in enumerate(embeddings):
        if emb and isinstance(emb, list) and len(emb) > 0:
            valid_texts.append(texts[i])
            valid_embeddings.append(emb)
            valid_metadatas.append(metadatas[i])
            valid_ids.append(metadatas[i]["id"])
        else:
            print(f"Skipping chunk {i} (embedding failed): {metadatas[i]['id']}")

    print(f"Valid embeddings: {len(valid_embeddings)} / {len(chunks)}")

    if not valid_embeddings:
        print("No valid embeddings found. Skipping DB creation.")
        return

    # Use chromadb PersistentClient directly
    print(f"Saving to persistent Chroma DB at {CHROMA_PATH}")
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="rag_collection")

    collection.add(
        ids=valid_ids,
        embeddings=valid_embeddings,
        documents=valid_texts,
        metadatas=valid_metadatas
    )

    print("Chroma vector store created and persisted successfully via chromadb.")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Cleared {CHROMA_PATH} directory.")
    else:
        print("No existing Chroma DB to clear.")

if __name__ == "__main__":
    main()