from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
from transformers import pipeline
import google.generativeai as genai


def build_vectorstore_from_documents(documents, chunk_size=200, chunk_overlap=40, model_name="BAAI/bge-base-en-v1.5"):
    """Build a FAISS vectorstore from a list of documents."""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = []
    for doc in documents:
        splits.extend(text_splitter.split_text(doc))
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(splits, embeddings)
    return vectorstore, splits

def semantic_search(query, vectorstore, k=3):
    """Search for the k most similar documents to the query using a provided vectorstore."""
    results = vectorstore.similarity_search(query, k=k)
    return [r.page_content for r in results]

# 4. Chatbot/QA: Use a local text-generation model (DialoGPT-medium)
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro')
    except Exception as e:
        print(f"[WARNING] Could not initialize Gemini: {e}")

def chat_with_bot(user_input):
    if gemini_model:
        try:
            response = gemini_model.generate_content(user_input)
            if hasattr(response, 'text'):
                return response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].text.strip()
        except Exception as e:
            print(f"Gemini chat error: {e}")
    # Fallback to local model
    response = chatbot(user_input, max_length=100, pad_token_id=50256, truncation=True)
    return response[0]['generated_text']

# 5. Summarization: Use a local summarization model (facebook/bart-large-cnn)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
def summarize_text(text):
    if gemini_model:
        try:
            prompt = f"Summarize the following text:\n{text}"
            response = gemini_model.generate_content(prompt)
            if hasattr(response, 'text'):
                return response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].text.strip()
        except Exception as e:
            print(f"Gemini summarize error: {e}")
    # Fallback to local model
    input_length = len(text.split())
    max_len = min(130, input_length + 20)
    summary = summarizer(text, max_length=max_len, min_length=10, do_sample=False)
    return summary[0]['summary_text']

# Example usage:
if __name__ == "__main__":
    user_query = "How does semantic search work?"
    print("--- Top Semantic Search Results ---")
    for i, doc in enumerate(semantic_search(user_query, vectorstore), 1):
        print(f"{i}. {doc}")

    print("\n--- Chatbot/QA Example ---")
    print(chat_with_bot("What is semantic search?"))

    print("\n--- Summarization Example ---")
    long_text = """Semantic search uses machine learning models to understand the meaning behind search queries and documents, rather than relying solely on keyword matching. This allows for more accurate and relevant search results."""
    print(summarize_text(long_text)) 