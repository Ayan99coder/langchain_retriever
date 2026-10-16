import wikipedia

wikipedia.set_user_agent(
    "AyanLangChainProject/1.0 (learning project)"
)
from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=3,
    lang="en"
)


query = "Artificial Intelligence"


documents = retriever.invoke(query)


for i, doc in enumerate(documents, start=1):
    print(f"\n--- Document {i} ---")
  
    print("Content:", doc.page_content[:1000])
