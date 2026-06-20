import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

class RAGEngine:
    def __init__(self, data_dir="data", db_dir="db"):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        self.vector_store = None
        
    def ingest_documents(self):
        print("Ingesting documents from:", self.data_dir)
        
        # Loaders for different file types
        loaders = {
            ".txt": TextLoader,
            ".md": TextLoader, # TextLoader works fine for MD in simple cases
            ".pdf": PyPDFLoader
        }
        
        documents = []
        for file in os.listdir(self.data_dir):
            ext = os.path.splitext(file)[1].lower()
            if ext in loaders:
                loader_cls = loaders[ext]
                loader = loader_cls(os.path.join(self.data_dir, file))
                documents.extend(loader.load())
        
        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        splits = text_splitter.split_documents(documents)
        
        # Vector Store
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        print(f"Ingested {len(splits)} chunks into {self.db_dir}")

    def load_vector_store(self):
        if os.path.exists(self.db_dir):
            self.vector_store = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )
            return True
        return False

    def retrieve(self, query, k=3):
        if not self.vector_store:
            if not self.load_vector_store():
                self.ingest_documents()
        
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        return results

if __name__ == "__main__":
    # Test
    engine = RAGEngine()
    engine.ingest_documents()
    res = engine.retrieve("How to reset password?")
    for doc, score in res:
        print(f"Score: {score:.4f} | Source: {doc.metadata.get('source')}")
        printContent = doc.page_content[:100] + "..."
        print(f"Content: {printContent}")
