from langchain_classic.retrievers import ContextualCompressionRetriever,MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv
load_dotenv()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
model = ChatGoogleGenerativeAI(model = 'gemini-3.7-flash')

loader = PyPDFLoader(file_path='machine_learning.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
chunk_size =800,
chunk_overlap = 20,
separators=["\n\n","\n","."," "]
)
chunks = splitter.split_documents(docs)

vector_store =  FAISS.from_documents(
documents= chunks,
embedding=embeddings
)

base_retriever = vector_store.as_retriever( search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.5
    })
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=model
)
compressor = LLMChainExtractor.from_llm(model)


contextual_compressor = ContextualCompressionRetriever(base_compressor=compressor,base_retriever=multi_query_retriever)

query = ' tell me about decision tree'

results = contextual_compressor.invoke(query)

for i, doc in enumerate(results, start=1):
    print(f"\nDocument: {i}")
    print(f"Answer/context:\n{doc.page_content}")





