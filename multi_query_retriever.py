from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
#define models
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash')

#load pdf file in documents
loader = PyPDFLoader(file_path='machine_learning.pdf')
docs = loader.load()

#text splitter 
splitter = RecursiveCharacterTextSplitter( 
    chunk_size=400,
    chunk_overlap=20,
    separators=["\n\n", "\n", " "])
chunks = splitter.split_documents(docs)

#storing chunks into vectore_store
vector_store = FAISS.from_documents(
documents= chunks,
embedding=embeddings
)

#retreive
similarity_search = vector_store.as_retriever(search_type = 'similarity',search_kwargs={"k":5})

multi_querySearch = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k":5}),
    llm=model
)
query = "What is a decision tree in machine learning?"

# Normal retrieval
normal_docs = similarity_search.invoke(query)
print("\n--- NORMAL RETRIEVER ---")
for i, doc in enumerate(normal_docs, start=1):
      print(f"\nDocument {i}:")
      print(doc.page_content)
      print("Page:", doc.metadata.get("page"))
  
  # Multi-query retrieval
multi_docs = multi_querySearch.invoke(query)
  
print("\n--- MULTI-QUERY RETRIEVER ---")
for i, doc in enumerate(multi_docs, start=1):
    print(f"\nDocument {i}:")
    print(doc.page_content)
    print("Page:", doc.metadata.get("page"))












