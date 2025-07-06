import argparse
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

CHROMA_PATH = "chroma_vectordb"
COLLECTION_NAME = "rag_collection"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_text", type=str, help="The query text.")
    args = parser.parse_args()   
    # query_rag(args.query_text)
    # query_text = input("Enter your search query to the pdf repo: ")
    if args.query_text:
        query_text = args.query_text
    else:
        query_text = input("Enter your search query: ")

    query_text=expand_query(query_text)
    print("Epanded query is:", query_text)
    # return
    # Writing below to check is query is not null
    if not query_text.strip():
        return
    query_rag(str(query_text))


def expand_query(query_text):
    query_expander_prompt = PromptTemplate.from_template("""
    You are a query rewriter. Convert vague user queries into short, focused questions that retrieve HR policy information. Avoid extra explanation. Just return the rewritten query.
    ❗Only output the rewritten query. Do NOT explain or include anything else.
    Original Query: {query_text}

    Expanded Query:
    """)
    model = OllamaLLM(model="llama3.2")
    query_expander_chain = query_expander_prompt | model
    return query_expander_chain.invoke({"query_text": query_text})


def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(
        collection_name=COLLECTION_NAME,  #Explicitly match the populate step
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    print(f"Loaded Chroma DB with {len(db.get()['ids'])} documents")

    retriever = db.as_retriever(search_kwargs={"k": 10})
    results = retriever.invoke(query_text)

    if not results:
        print("No documents matched your query.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("\n[Prompt Sent to LLM]:\n", prompt)

    model = OllamaLLM(model="llama3.2")
    response_text = model.invoke(prompt)

    sources = [
        f"{doc.metadata.get('source')} (Page {doc.metadata.get('page', '?')}, ID {doc.metadata.get('id', '?')})"
        for doc in results
    ]
    formatted_response = f"\nResponse:\n{response_text}\n\nSources:\n{sources}"
    print(formatted_response)

    # return response_text
    return {
    "answer": response_text,
    "sources": sources
            }

if __name__ == "__main__":
    main()
