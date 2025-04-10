# RAG Chat Application

This project implements a Retrieval-Augmented Generation (RAG) system that integrates document processing, vector embeddings, and OpenAI's language models to provide context-aware responses to user queries.

## Features

- **Document Processing**: Upload and process PDF and TXT files
- **Semantic Search**: Find relevant information using embeddings-based similarity
- **Conversation History**: Maintain context across multiple user interactions
- **Query Enhancement**: Improve search results with automatic query expansion
- **Multi-Source Retrieval**: Get information from diverse documents
- **Interactive Interface**: User-friendly Gradio web interface

## Architecture

The system consists of several key components:

- **Document Processor**: Extracts and chunks text from documents
- **Embedding Generator**: Creates vector representations using Sentence Transformers
- **Document Store**: Stores document chunks and embeddings in ChromaDB
- **Query Enhancer**: Improves search queries using OpenAI
- **Retriever**: Finds relevant document chunks based on semantic similarity
- **LLM Interface**: Generates responses based on retrieved context
- **Conversation History**: Maintains conversation state and context

## Directory Structure

```
rag-chat/
├── modules/
│   ├── __init__.py
│   ├── config.py
│   ├── conversation_history.py
│   ├── document_processor.py
│   ├── document_store.py
│   ├── embeddings.py
│   ├── llm_interface.py
│   ├── query_enhancer.py
│   └── retriever.py
├── gradio_app.py
├── requirements.txt
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- OpenAI API Key

### Environment Setup

1. Clone the repository:
   ```
   git clone https://github.com/jigarthummar/rag-chat.git
   cd rag-chat
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   ```

3. Activate the virtual environment:
   ```
   source .venv/bin/activate  # On Linux/Mac
   .venv\Scripts\activate     # On Windows
   ```

4. Install requirements:
   ```
   pip install -r requirements.txt
   ```

5. Set up your .env file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application

Start the Gradio web interface:
```
python gradio_app.py
```

The application will be available at `http://127.0.0.1:7860`

## Usage

1. **Upload Documents**: Use the file upload area to add PDF or TXT files to the knowledge base
2. **Ask Questions**: Type your questions in the message box
3. **View Sources**: See which documents and sections were used to answer your questions
4. **Manage Conversations**: Start new conversations or save the current one

## Configuration

You can customize the system by adjusting parameters in `config.py`:

- `CHUNK_SIZE`: Size of text chunks for document processing
- `CHUNK_OVERLAP`: Overlap between adjacent chunks
- `EMBEDDING_MODEL`: Model for generating embeddings
- `TOP_K`: Number of documents to retrieve
- `SIMILARITY_THRESHOLD`: Minimum similarity score for relevant documents
- `LLM_MODEL`: OpenAI model to use for generating responses