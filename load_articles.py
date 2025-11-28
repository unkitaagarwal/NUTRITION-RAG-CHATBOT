from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# Load and split documents
def load_and_split_docs(data_path="data"):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for fname in os.listdir(data_path):
        file_path = os.path.join(data_path, fname)

        # Skip hidden/system files (e.g. .DS_Store) that break decoding
        if fname.startswith("."):
            print(f"⚠️  Skipping hidden/system file: {fname}")
            continue

        try:
            if fname.lower().endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif fname.lower().endswith(".csv"):
                loader = CSVLoader(file_path)
            else:
                print(f"ℹ️  Skipping unsupported file type: {fname}")
                continue

            docs.extend(splitter.split_documents(loader.load()))
            print(f"✅ Loaded {fname}")
        except Exception as e:
            print(f"❌ Error loading {fname}: {e}")
            continue
    return docs

# Store in Chroma
def embed_to_chroma(docs):
    db = Chroma(persist_directory="./vector_store", embedding_function=OpenAIEmbeddings())
    db.add_documents(docs)

if __name__ == "__main__":
    load_dotenv()
    documents = load_and_split_docs()
    embed_to_chroma(documents)
    print(f"✅ {len(documents)} chunks embedded")
