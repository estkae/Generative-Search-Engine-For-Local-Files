import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
import json
import PyPDF2
from typing import List, Dict, Any
import ollama
import logging

class DocumentSearch:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.dimension = 768
        self.index = None
        self.metadata = []
        self.initialize_index()
        
    def initialize_index(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        
    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() + " "
                
                text_chunks = self.chunk_text(full_text)
                for i, chunk in enumerate(text_chunks):
                    chunks.append({
                        "content": chunk,
                        "path": file_path,
                        "chunk_id": i,
                        "total_chunks": len(text_chunks)
                    })
                return chunks
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            return chunks

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1
            if current_length > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word) + 1
            else:
                current_chunk.append(word)
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def index_documents(self, directory: str) -> None:
        self.initialize_index()
        self.metadata = []
        all_chunks = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        total_files = len(pdf_files)
        
        for i, file in enumerate(pdf_files):
            file_path = os.path.join(directory, file)
            status_text.text(f"Processing {file}...")
            
            chunks = self.process_pdf(file_path)
            all_chunks.extend(chunks)
            
            progress_bar.progress((i + 1) / total_files)
        
        if all_chunks:
            self.metadata = all_chunks
            embeddings = self.model.encode([chunk["content"] for chunk in all_chunks])
            self.index.add(np.array(embeddings))
            self.save_index()
            
        status_text.text(f"Indexing complete! Processed {len(all_chunks)} chunks from {total_files} files.")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.index or self.index.ntotal == 0:
            return []
            
        query_vector = self.model.encode([query])[0]
        scores, indices = self.index.search(np.array([query_vector]), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(scores[0][i])
                result["id"] = int(idx)
                results.append(result)
        return results

    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""Systemanweisung: Du bist ein hilfsbereiter Assistent. Antworte immer auf Deutsch. 
Beantworte die Frage basierend auf dem gegebenen Kontext. Verwende Dokumenten-IDs [0], [1], etc.
Falls die Antwort nicht im Kontext zu finden ist, sage das ehrlich.

Kontext: {context}

Frage: {query}

Antwort:"""
        try:
            response = ollama.generate(model='llama3.2', prompt=prompt)
            return response['response']
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            return "Entschuldigung, es gab einen Fehler bei der Antwortgenerierung."

    def save_index(self):
        try:
            faiss.write_index(self.index, "search_index.faiss")
            with open("search_metadata.json", "w", encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving index: {str(e)}")

    def load_index(self) -> bool:
        try:
            if os.path.exists("search_index.faiss") and os.path.exists("search_metadata.json"):
                self.index = faiss.read_index("search_index.faiss")
                with open("search_metadata.json", "r", encoding='utf-8') as f:
                    self.metadata = json.load(f)
                return True
        except Exception as e:
            logging.error(f"Error loading index: {str(e)}")
            self.initialize_index()
        return False

def main():
    st.set_page_config(page_title="Dokumentensuche", layout="wide")
    
    if 'search_system' not in st.session_state:
        st.session_state.search_system = DocumentSearch()
        st.session_state.search_system.load_index()

    st.title("🔍 Dokumentensuche")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        docs_path = st.text_input("📁 Dokumentenordner:", help="Pfad zum Ordner mit PDF-Dokumenten")
        
    with col2:
        if st.button("🔄 Dokumente indexieren"):
            if not docs_path:
                st.error("Bitte Pfad angeben!")
            elif not os.path.exists(docs_path):
                st.error("Pfad existiert nicht!")
            else:
                with st.spinner("Indexiere Dokumente..."):
                    st.session_state.search_system.index_documents(docs_path)

    st.divider()
    query = st.text_input("🔎 Ihre Frage:")
    
    if st.button("Suchen") and query:
        results = st.session_state.search_system.search(query)
        
        if results:
            context = "\n\n".join([f"{i}: {r['content']}" for i, r in enumerate(results)])
            
            with st.spinner("Generiere Antwort..."):
                answer = st.session_state.search_system.generate_answer(query, context)
                
            st.markdown("### 💡 Antwort:")
            st.write(answer)
            
            st.markdown("### 📄 Gefundene Dokumente:")
            for i, result in enumerate(results):
                with st.expander(f"Dokument {i} - {os.path.basename(result['path'])}"):
                    st.write(result['content'])
                    st.write(f"Relevanz: {result['score']:.2f}")
        else:
            st.warning("Keine Ergebnisse gefunden.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()