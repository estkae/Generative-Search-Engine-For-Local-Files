import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
import os
import json
from typing import List, Dict, Any
import PyPDF2

class DocumentSearch:
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = 768
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []
        
    def read_pdf(self, file_path: str) -> str:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return ' '.join(page.extract_text() for page in reader.pages)
            
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) 
                for i in range(0, len(words), chunk_size)]

    def index_documents(self, directory: str) -> None:
        self.metadata = []
        documents = []
        
        for file in os.listdir(directory):
            if not file.lower().endswith('.pdf'):
                continue
                
            file_path = os.path.join(directory, file)
            content = self.read_pdf(file_path)
            
            for i, chunk in enumerate(self.chunk_text(content)):
                documents.append(chunk)
                self.metadata.append({
                    "path": file_path,
                    "chunk_id": i,
                    "total_chunks": len(self.chunk_text(content))
                })
        
        if documents:
            embeddings = self.model.encode(documents)
            self.index.add(np.array(embeddings))
            self.save_index()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.model.encode([query])[0]
        scores, indices = self.index.search(np.array([query_vector]), k)
        
        return [{
            "id": int(idx),
            "metadata": self.metadata[idx],
            "score": float(scores[0][i])
        } for i, idx in enumerate(indices[0])]
    
    def save_index(self, index_path: str = "index.faiss", metadata_path: str = "metadata.json") -> None:
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def load_index(self, index_path: str = "index.faiss", metadata_path: str = "metadata.json") -> None:
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

def main():
    st.set_page_config(page_title="Document Search", layout="wide")
    st.title("Document Search System")
    
    search_system = DocumentSearch()
    
    docs_path = st.text_input("Documents Directory Path:")
    if st.button("Index Documents") and docs_path:
        with st.spinner("Indexing documents..."):
            search_system.index_documents(docs_path)
        st.success("Indexing complete!")
        
    query = st.text_input("Search Query:")
    if st.button("Search") and query:
        results = search_system.search(query)
        for result in results:
            st.write(f"Document: {os.path.basename(result['metadata']['path'])}")
            st.write(f"Score: {result['score']:.2f}")
            st.write("---")

if __name__ == "__main__":
    main()