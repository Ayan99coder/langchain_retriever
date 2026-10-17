import re
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

docs = [
    Document(page_content="Python is GREAT!"),
    Document(page_content="python is great"),
    Document(page_content="Python programming is useful."),
    Document(page_content="Machine learning uses Python."),
    Document(page_content="FAISS searches vector embeddings.")
]

# Text cleaning function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return tokens

# Preprocess documents
tokenized_docs = [
    preprocess(doc.page_content)
    for doc in docs
]

for i, tokens in enumerate(tokenized_docs, start=1):
    print(f"Document {i}: {tokens}")

# BM25 uses the cleaned tokens
from rank_bm25 import BM25Okapi

bm25 = BM25Okapi(tokenized_docs)

query = "PYTHON, GREAT!"
query_tokens = preprocess(query)

scores = bm25.get_scores(query_tokens)

for i, score in enumerate(scores, start=1):
    print(f"Document {i} Score: {score:.4f}")