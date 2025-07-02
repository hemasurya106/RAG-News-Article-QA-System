# RAG News Article QA System

This project is a Retrieval-Augmented Generation (RAG) system that leverages Large Language Models (LLMs) and vector search to answer questions based on a collection of news articles. It uses LangChain, ChromaDB, and HuggingFace embeddings to provide context-aware answers from your own document collection.

## Features
- Loads and indexes news articles from the `news_articles/` directory
- Embeds and stores document chunks in a persistent Chroma vector database
- Uses a Groq-hosted LLM (Llama 4) for question answering
- Simple script-based interface for querying the knowledge base

## Project Structure
```
RAG/
├── app.py                  # Main script for document ingestion and QA
├── requirements.txt        # Python dependencies
├── news_articles/          # Directory of news article .txt files
├── chroma_persistent_storage/ # Persistent vector database storage
```

## Setup
1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables**
   - Create a `.env` file in the project root with your Groq API key:
     ```env
     GROQ_API_KEY=your_groq_api_key_here
     ```

## Usage
Run the main script to ingest articles and perform a sample query:
```bash
python app.py
```
- The script will load all `.txt` files from `news_articles/`, embed and store them, and then run a sample query (`how ai replace tv writers`).
sample output:-
The Writers Guild of America proposed that AI output not be considered writers' work and that writers not be required to adapt AI-generated content. However, the Alliance of Motion Picture and Television Producers (AMPTP) refused to engage with this proposal. Currently, it seems AI won't directly replace TV writers, but rather be used as a tool that writers' unions want to ensure doesn't undermine their working conditions.
- You can modify the `question` variable in `app.py` to ask your own questions.
-If no relevant content is found, the system will respond with an out-of-context message.

## Dependencies
- langchain
- langchain-huggingface
- langchain-chroma
- langchain-groq
- sentence-transformers
- chromadb
- python-dotenv

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Adding Data
- Place your `.txt` files in the `news_articles/` directory. Each file will be loaded and indexed automatically on the next run.

## Notes
- The vector database is stored in `chroma_persistent_storage/` and will persist between runs.
- Make sure your API keys are kept secure and **never commit your `.env` file** to version control.


