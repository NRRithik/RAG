import os
import sys
from dotenv import load_dotenv
from rag_engine import RAGPipeline

# Load API key securely from .env file
load_dotenv()

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        print("Please add your GOOGLE_API_KEY to the .env file.")
        return

    print("Initializing LangChain RAG with Gemini...")
    rag = RAGPipeline("sample_document.txt", api_key=api_key)
    print("Ready!\n")

    # Command line argument support: python main.py "Who is Rithik?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Q: {query}\nA: {rag.ask(query)}")
        return

    # Interactive loop
    while True:
        try:
            q = input("Ask a question (or 'exit'): ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                break
            print(f"\nAI: {rag.ask(q)}\n")
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
