from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import shutil

# Read URLs and SOAP requests from CSV files
urls_df = pd.read_csv("urls.csv")
soap_df = pd.read_csv("soap_requests.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"

# Always refresh the database to ensure we have the latest data
if os.path.exists(db_location):
    shutil.rmtree(db_location)

documents = []
ids = []

# Add URL documents
for i, row in urls_df.iterrows():
    document = Document(
        page_content=f"URL: {row['URL']}\nDescription: {row['Description']}",
        metadata={"type": "url", "url": row["URL"]},
        id=f"url_{i}"
    )
    ids.append(document.id)
    documents.append(document)

# Add SOAP request documents
for i, row in soap_df.iterrows():
    # Format the XML with proper indentation
    sample_request = row['SampleRequest'].replace('><', '>\n<')
    document = Document(
        page_content=f"RequestName: {row['RequestName']}\nSampleRequest:\n{sample_request}",
        metadata={"type": "soap", "request_name": row["RequestName"]},
        id=f"soap_{i}"
    )
    ids.append(document.id)
    documents.append(document)

vector_store = Chroma(
    collection_name="knowledge_base",
    persist_directory=db_location,
    embedding_function=embeddings
)

vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)