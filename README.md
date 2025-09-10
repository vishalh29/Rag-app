# RAG Document Q&A System

A powerful Retrieval-Augmented Generation (RAG) system built with Streamlit, LangChain, OpenAI, and Pinecone for document-based question answering.

## 🌟 Features

- **Document Upload**: Upload multiple PDF files to create your knowledge base
- **Intelligent Q&A**: Ask questions and get AI-powered answers with source references
- **Vector Search**: Uses Pinecone for efficient semantic search
- **Modern UI**: Clean, intuitive Streamlit interface
- **Source Tracking**: Shows which documents and pages answers come from

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project directory:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone API Key  
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 3. Run the Application

**Option A: Using the launcher script**
```bash
python run_app.py
```

**Option B: Direct Streamlit command**
```bash
streamlit run streamlit_app.py
```

### 4. Use the App

1. **Initialize System**: Click the "Initialize System" button in the sidebar
2. **Upload Documents**: Upload your PDF files using the file uploader
3. **Process Documents**: Click "Process Documents" to create embeddings
4. **Ask Questions**: Enter your questions in the main area and get answers!

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Pinecone API key
- PDF documents to query

## 🛠️ How It Works

1. **Document Processing**: PDFs are loaded and split into manageable chunks
2. **Embedding Creation**: Text chunks are converted to vector embeddings using OpenAI
3. **Vector Storage**: Embeddings are stored in Pinecone for fast retrieval
4. **Query Processing**: User questions are converted to embeddings and matched against stored documents
5. **Answer Generation**: Relevant documents are sent to OpenAI GPT for answer generation

## 🔧 Configuration

- **Chunk Size**: Documents are split into 800-character chunks with 50-character overlap
- **Model**: Uses `gpt-3.5-turbo-instruct` for answer generation
- **Embeddings**: Uses OpenAI's `text-embedding-ada-002` model
- **Vector DB**: Pinecone with cosine similarity metric

## 🐛 Troubleshooting

**Common Issues:**

1. **API Key Errors**: Make sure your `.env` file is properly configured
2. **Pinecone Index**: The app automatically creates an index named "langchanivector"
3. **PDF Processing**: Ensure PDFs are readable and not encrypted
4. **Dependencies**: Run `pip install -r requirements.txt` if you get import errors

## 📝 Example Queries

- "What are the main topics discussed in the documents?"
- "Summarize the key findings from the research paper"
- "What recommendations are made in the report?"
- "Tell me about the methodology used in the study"

## 🔒 Security

- API keys are loaded from environment variables
- Documents are processed locally before being sent to external services
- No sensitive data is stored permanently

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this RAG system!

## 📄 License

This project is open source and available under the MIT License.
