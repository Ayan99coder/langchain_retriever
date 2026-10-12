from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
loader = PyPDFLoader(file_path='machine_learning.pdf')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", " "]
)

chunks = splitter.split_documents(docs)



print("Total pages:", len(docs))
print("Total chunks:", len(chunks))

vector_store = Chroma(
embedding_function = embeddings,
persist_directory = 'chroma_db',
collection_name = 'sample'
)

vector_store.add_documents(chunks)
print("Documents successfully stored!")
retriver = vector_store.as_retriever(kwargs={'k':2})
query = 'What is decision tree?'
results = retriver.invoke(query)
for i, doc in enumerate(results, start=1):
    print(f"\n--- Document {i} ---")
  
    print("Content:", doc.page_content[:1000])