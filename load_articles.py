from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Load and split documents
def load_and_split_docs(data_path="data/"):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for fname in os.listdir(data_path):
        loader = TextLoader(os.path.join(data_path, fname))
        docs.extend(splitter.split_documents(loader.load()))
    return docs

# Store in Chroma
def embed_to_chroma(docs):
    db = Chroma(persist_directory="./vector_store", embedding_function=OpenAIEmbeddings())
    db.add_documents(docs)
    db.persist()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    documents = load_and_split_docs()
    embed_to_chroma(documents)
