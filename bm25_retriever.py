from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
# Step 1: PDF read karo
loader = PyPDFLoader("machine_learning.pdf")
docs = loader.load()


test_docs = [
    Document(
        page_content="Information Gain helps select the best decision tree split."
    ),
    Document(
        page_content=(
            "Information Gain Information Gain Information Gain "
            "is used for decision tree feature selection."
        )
    ),
    Document(
        page_content=(
            "Neural networks learn patterns through interconnected neurons."
        )
    ),
]

# Step 2: Text ko chunks mein divide karo
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    separators=['\n\n','\n','.',' ']
)
chunks = splitter.split_documents(test_docs)
bm25 = BM25Retriever.from_documents(chunks, k=5)

# Step 4: Search karo
results = bm25.invoke("Information Gain in Decision Trees")

for i, doc in enumerate(results, start=1):
    print(f"\nDocument {i}")
    print(doc.page_content)
    print("Page:", doc.metadata.get("page"))
