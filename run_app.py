#!/usr/bin/env python3
"""
Simple launcher script for the RAG Streamlit app
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app"""
    print("🚀 Starting RAG Document Q&A System...")
    print("📱 The app will open in your browser shortly...")
    print("🛑 Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Run the streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py", 
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Stopping the app... Goodbye!")
    except Exception as e:
        print(f"❌ Error running the app: {e}")
        print("💡 Make sure you have installed all requirements: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
