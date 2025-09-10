import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Load environment variables for local development
load_dotenv()

def get_api_key(key_name):
    """Get API key from Streamlit secrets (cloud) or environment variables (local)"""
    # First try Streamlit secrets (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    
    # Fall back to environment variables (for local development)
    return os.getenv(key_name)

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RAGSystem:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.index_name = "langchanivector"
        
    def initialize_components(self):
        """Initialize all RAG components"""
        try:
            # Check for API keys
            openai_key = get_api_key("OPENAI_API_KEY")
            pinecone_key = get_api_key("PINECONE_API_KEY")
            
            if not openai_key:
                st.error("❌ OpenAI API key not found.")
                st.info("🔧 **For Streamlit Cloud**: Add OPENAI_API_KEY to your app's secrets in the Streamlit Cloud dashboard")
                st.info("🏠 **For local development**: Set OPENAI_API_KEY in your .env file")
                return False
            
            if not pinecone_key:
                st.error("❌ Pinecone API key not found.")
                st.info("🔧 **For Streamlit Cloud**: Add PINECONE_API_KEY to your app's secrets in the Streamlit Cloud dashboard")
                st.info("🏠 **For local development**: Set PINECONE_API_KEY in your .env file")
                return False
            
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(api_key=openai_key)
            
            # Initialize Pinecone
            pc = PineconeClient(api_key=pinecone_key)
            
            # Create index if it doesn't exist
            if self.index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embeddings dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                st.info(f"✅ Created new Pinecone index: {self.index_name}")
            
            # Initialize QA chain
            llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.5)
            self.qa_chain = load_qa_chain(llm, chain_type="stuff")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error initializing components: {str(e)}")
            return False
    
    def process_documents(self, uploaded_files):
        """Process uploaded PDF documents"""
        try:
            with st.spinner("🔄 Processing documents..."):
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files
                    for uploaded_file in uploaded_files:
                        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Load documents
                    loader = PyPDFDirectoryLoader(temp_dir)
                    documents = loader.load()
                    
                    if not documents:
                        st.error("❌ No documents could be loaded. Please check your PDF files.")
                        return False
                    
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=50
                    )
                    chunks = text_splitter.split_documents(documents)
                    
                    # Create vector store
                    self.vector_store = PineconeVectorStore.from_documents(
                        documents=chunks,
                        embedding=self.embeddings,
                        index_name=self.index_name,
                        pinecone_api_key=get_api_key('PINECONE_API_KEY')
                    )
                    
                    st.success(f"✅ Successfully processed {len(documents)} documents into {len(chunks)} chunks!")
                    return True
                    
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
            return False
    
    def query_documents(self, query, k=3):
        """Query the vector store for relevant documents"""
        try:
            if not self.vector_store:
                st.error("❌ No documents loaded. Please upload and process documents first.")
                return None
            
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(query, k=k)
            
            if not relevant_docs:
                st.warning("⚠️ No relevant documents found for your query.")
                return None
            
            # Generate answer using QA chain
            response = self.qa_chain.run(input_documents=relevant_docs, question=query)
            
            return {
                'answer': response,
                'source_documents': relevant_docs
            }
            
        except Exception as e:
            st.error(f"❌ Error querying documents: {str(e)}")
            return None

def main():
    # Add custom CSS for better formatting
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
        margin: 0.5rem 0;
    }
    .stTextInput > div > div > input {
        font-size: 16px !important;
    }
    .status-info {
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
    }
    .ready-status {
        background-color: #f8f9fa;
    }
    .init-status {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
    }
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 1rem;
        border-top: 1px solid #dee2e6;
        margin-top: 2rem;
        font-size: 14px;
    }
    /* Fix text wrapping in expanders */
    .streamlit-expanderContent {
        overflow-wrap: break-word;
        word-wrap: break-word;
    }
    /* Improve readability */
    .element-container {
        text-align: justify;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("📚 RAG Document Q&A System")
    st.markdown("""
    <div class="main-header">
        <h3 style="color: white; margin: 0;">RAG (Retrieval-Augmented Generation) Document Q&A System! Using Pinecone Database</h3>

    </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
        st.session_state.system_initialized = False
        st.session_state.documents_processed = False
    
    # Sidebar for configuration and file upload
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Initialize system
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing RAG system..."):
                st.session_state.system_initialized = st.session_state.rag_system.initialize_components()
        
        if st.session_state.system_initialized:
            st.success("✅ System initialized successfully!")
            
            st.header("📄 Document Upload")
            uploaded_files = st.file_uploader(
                "Upload PDF documents",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or more PDF files to create your knowledge base"
            )
            
            if uploaded_files and st.button("📤 Process Documents"):
                st.session_state.documents_processed = st.session_state.rag_system.process_documents(uploaded_files)
            
            # Query settings
            if st.session_state.documents_processed:
                st.header("⚙️ Query Settings")
                num_results = st.slider(
                    "Number of source documents to retrieve",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="More documents provide more context but may be slower"
                )
        else:
            st.warning("⚠️ Please initialize the system first")
            st.info("💡 **For Streamlit Cloud**: Add your API keys to the app secrets in the Streamlit dashboard")
            st.info("💡 **For local development**: Create a .env file with OPENAI_API_KEY and PINECONE_API_KEY")
    
    # Main content area
    if st.session_state.system_initialized:
        if st.session_state.documents_processed:
            st.header("Ask Questions")
            
            # Query input
            query = st.text_input(
                "Enter your question:",
                placeholder="e.g., What are the main topics discussed in the documents?",
                help="Ask any question about the uploaded documents"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("🔍 Ask Question", type="primary")
            
            if ask_button and query:
                with st.spinner("🤔 Thinking..."):
                    result = st.session_state.rag_system.query_documents(query, k=num_results)
                
                if result:
                    # Display answer in a nice container
                    st.markdown("### 💡 Answer")
                    # Escape HTML in the answer to prevent issues
                    escaped_answer = result['answer'].replace('<', '&lt;').replace('>', '&gt;')
                    st.markdown(f"""
                    <div class="answer-box">
                        <div style="color: #2c3e50; font-size: 16px; line-height: 1.6; white-space: pre-wrap;">
                            {escaped_answer}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display source documents with better formatting
                    st.markdown("### 📋 Source Documents")
                    
                    for i, doc in enumerate(result['source_documents'], 1):
                        # Extract filename from path for cleaner display
                        source_path = doc.metadata.get('source', 'Unknown')
                        filename = source_path.split('/')[-1] if '/' in source_path else source_path
                        page_num = doc.metadata.get('page', 'Unknown')
                        
                        with st.expander(f"📄 Source {i}: {filename} (Page {page_num})", expanded=False):
                            # Escape HTML in source content as well
                            escaped_content = doc.page_content.replace('<', '&lt;').replace('>', '&gt;')
                            st.markdown(f"""
                            <div class="source-box">
                                <div style="color: #495057; font-size: 14px; line-height: 1.5; white-space: pre-wrap;">
                                    {escaped_content}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            elif ask_button and not query:
                st.warning("⚠️ Please enter a question first!")
        
        else:
            st.markdown("""
            <div class="status-info ready-status">
                <h3 style="color: #495057; margin-bottom: 1rem;">📤 Ready to Get Started?</h3>
                <p style="color: #6c757d; font-size: 16px;">Please upload and process some PDF documents to begin asking questions!</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="status-info init-status">
            <h3 style="color: #856404; margin-bottom: 1rem;">🚀 Let's Get Started!</h3>
            <p style="color: #856404; font-size: 16px;">Please initialize the system using the sidebar to get started!</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
