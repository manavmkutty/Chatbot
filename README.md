# KTU Academics RAG Chatbot
An AI-based chatbot built using Retrieval-Augmented Generation (RAG) that enables students to ask questions related to KTU (Kerala Technological University) academics. The chatbot leverages document embeddings and semantic search to provide accurate answers from a curated knowledge base of notes, previous year questions (PYQs), and academic materials. 

## Setup Instructions

### Prerequisites
- Python 3.11.9
- Virtual environment

### Installation
1. Create and activate a virtual environment with Python 3.11.9:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. Place your PDF documents (Notes, PYQs, etc.) in the `data/` folder

## Project Structure

- **datastore.py**: Handles document loading, text chunking, and embedding generation
  - Loads PDF files from the `data/` directory
  - Splits documents into chunks for optimal embedding
  - Generates embeddings using the `all-MiniLm-L6-v2` SentenceTransformer model

- **main.py**: Main application entry point (in development)

- **data/**: Directory for storing PDF documents (Notes, PYQs, etc.)

## Current Progress ✅

- [x] Document loading pipeline (PDFs from local directory)
- [x] Text chunking with configurable overlap
- [x] Embedding generation using SentenceTransformer
- [x] Project structure and dependencies
- [x] README documentation

## Next Steps (TODO) 📋

- [ ] **ChromaDB Integration**: Store embeddings and documents in ChromaDB for efficient semantic search
- [ ] **Query Embedding Pipeline**: Create a function to embed user queries using the same model
- [ ] **Retrieval System**: Implement similarity search to retrieve relevant documents based on query embeddings
- [ ] **LLM Integration**: Integrate a language model (OpenAI, Claude, Ollama, etc.) for generating contextual answers
- [ ] **RAG Pipeline**: Connect the retrieval and generation components to create the complete RAG pipeline
- [ ] **API/Interface**: Build a REST API or web interface (FastAPI, Streamlit, etc.) for user interaction
- [ ] **Testing & Validation**: Test with real academic questions and refine retrieval/generation quality
- [ ] **Optimization**: Fine-tune chunk sizes, models, and search parameters for better performance

## Technologies Used

- **LangChain**: Document loading and text splitting
- **SentenceTransformers**: Embedding generation
- **ChromaDB**: Vector database (to be integrated)
- **Python 3.11.9**: Runtime environment