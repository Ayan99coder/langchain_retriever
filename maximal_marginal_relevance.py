from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
#mmr
#lambda_mult = 1.0
#Relevance priority
#Question se sab se zyada related chunks ko priority. Similar information repeat ho sakti hai
#lambda_mult = 0.5
#Balanced
#Relevant chunks bhi chahiye aur unmein unnecessary repetition bhi kam honi chahiye.
#lambda_mult = 0.0
#Diversity priority
#Selected chunks ke darmiyan difference ko priority milti hai, lekin question se relevance kam ho sakti hai.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

loader = PyPDFLoader(file_path='machine_learning.pdf')

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    separators=["\n\n", "\n", " "]
    )

chunks=splitter.split_documents(docs)


print("Total pages:", len(docs))
print("Total chunks:", len(chunks))

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory = 'chroma_db_mmr',
    collection_name = 'samples' 
)
vector_store.add_documents(chunks)
vector = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs = {"k":3,"fetch-k":20,"lambda_mult":0.0}
)
query = 'What is a decision tree in machine learning?'
results = vector.invoke(query)

for i,doc in enumerate(results):
      print(f"\n--- Document {i+1} ---")
      print("Content:", doc.page_content)
      
    


