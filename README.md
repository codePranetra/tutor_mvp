# RAG-Based Chatbot with LangChain

A Flask web application that uses LangChain to create a RAG (Retrieval-Augmented Generation) chatbot. The bot can answer questions based on the content in `data.txt` using OpenAI's language models and vector embeddings.

## Features

- 🤖 **RAG-based Q&A**: Answers questions using context from `data.txt`
- 🔍 **Semantic Search**: Uses FAISS vector store for efficient document retrieval
- 💬 **Web Interface**: Beautiful, modern chat UI
- 📝 **Conversation History**: Maintains context across the conversation
- 🌐 **Multi-language Support**: Can handle questions in Hindi and other languages

## Prerequisites

- Python 3.8 or higher
- OpenAI API Key
- Virtual environment (`.venv`)

## Setup Instructions

1. **Activate your virtual environment:**
   ```bash
   # On Windows
   .venv\Scripts\activate

   # On Linux/Mac
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API Key:**
   ```bash
   # On Windows (PowerShell)
   $env:OPENAI_API_KEY="your-api-key-here"

   # On Windows (CMD)
   set OPENAI_API_KEY=your-api-key-here

   # On Linux/Mac
   export OPENAI_API_KEY="your-api-key-here"
   ```

   Or create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Ensure `data.txt` is in the project root directory**

## Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Start chatting!** Ask questions about the content in `data.txt`

## How It Works

1. **Document Loading**: The app loads `data.txt` and splits it into chunks
2. **Vector Store**: Creates embeddings using OpenAI and stores them in FAISS
3. **Retrieval**: When you ask a question, the system searches for relevant context
4. **Generation**: The LangChain agent uses the retrieved context to generate accurate answers

## API Endpoints

- `GET /` - Main chat interface
- `POST /api/chat` - Send a message and get a response
  ```json
  {
    "message": "Your question here",
    "user_id": "optional_user_id"
  }
  ```
- `GET /api/health` - Health check endpoint

## Project Structure

```
.
├── app.py              # Main Flask application with RAG service
├── data.txt            # Knowledge base file
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Web chat interface
└── README.md           # This file
```

## Troubleshooting

- **"OPENAI_API_KEY not set"**: Make sure you've set the environment variable
- **"Knowledge base not loaded"**: Ensure `data.txt` exists in the project root
- **Import errors**: Make sure all dependencies are installed in your virtual environment

## Notes

- The first run may take a moment to load and index the documents
- The vector store is created in memory and will be rebuilt on each restart
- For production use, consider persisting the vector store to disk

