import gradio as gr
import os
print("Current COHERE_API_KEY:", os.getenv("COHERE_API_KEY"))
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from transformers import pipeline, AutoTokenizer
import nltk
import cohere
import re
import difflib
nltk.data.path.append(r"C:\Users\bindi\AppData\Roaming\nltk_data")
nltk.data.path.append(r"C:\Users\bindi\OneDrive\Documents\chitra\OneDrive\Desktop\Project\nltk_data")
print("NLTK data paths:", nltk.data.path)
# Remove the problematic punkt download and import
# nltk.download('punkt', quiet=True)
# from nltk.tokenize import sent_tokenize

# Models
EMBED_MODEL = 'BAAI/bge-base-en-v1.5'  # Stronger semantic embedding
# EMBED_MODEL = 'sentence-transformers/gtr-t5-base'  # Use gtr-t5-base for semantic search
GEN_QA_MODEL = 'google/flan-t5-large'  # Generative QA
SUMMARIZER_MODEL = 'facebook/bart-large-cnn'  # Summarizer

embedder = SentenceTransformer(EMBED_MODEL)
qa_pipeline = pipeline("text2text-generation", model=GEN_QA_MODEL, tokenizer=GEN_QA_MODEL)
summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)
summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)

# Use cross-encoder for reranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

DIM = 768  # all-mpnet-base-v2 output dim
index = faiss.IndexFlatL2(DIM)
documents = []
embeddings = []

# Synonym dictionary for query expansion
SYNONYMS = {
    "benefits": ["advantages", "pros", "features", "value", "importance"],
    "drawbacks": ["disadvantages", "cons", "limitations", "challenges"],
    # Add more as needed
}

COHERE_API_KEY = os.getenv('COHERE_API_KEY')
cohere_client = None
if COHERE_API_KEY:
    try:
        cohere_client = cohere.Client(COHERE_API_KEY)
    except Exception as e:
        print(f"[WARNING] Could not initialize Cohere: {e}")

paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

try:
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except Exception as e:
    zero_shot_classifier = None
    print(f"[WARNING] Could not load zero-shot classifier: {e}")

# Add HuggingFace LLM pipeline for fallback
try:
    hf_llm = pipeline("text2text-generation", model="google/flan-t5-large")
except Exception as e:
    hf_llm = None
    print(f"[WARNING] Could not load HuggingFace LLM: {e}")

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from llm import summarize_documents_separately, generate_synthesized_answer_from_summaries

def get_dynamic_max_tokens(intent):
    # Set max tokens based on intent
    if intent in ["definition", "full-form"]:
        return 80
    elif intent in ["list", "examples", "summary"]:
        return 200
    elif intent in ["comparison", "pros-cons"]:
        return 400
    else:  # explanation, how-to, general
        return 512

def cohere_answer(full_prompt, intent=None, context=None, abbreviation=None, allow_continue=True):
    if not cohere_client:
        print("Cohere client not initialized!")
        return None
    if intent == "full-form" and context and abbreviation:
        extracted = extract_full_form_from_context(context, abbreviation)
        if extracted:
            print(f"[Regex Extraction] Found full form for {abbreviation}: {extracted}")
            return extracted
    if not intent:
        intent = detect_intent(full_prompt)
    max_tokens = get_dynamic_max_tokens(intent)
    if intent in ["definition", "full-form"]:
        model = 'command'
    elif intent in ["list", "examples", "summary"]:
        model = 'command-light'
    elif intent in ["comparison", "pros-cons"]:
        model = 'command-light'
    else:
        model = 'command'
    try:
        response = cohere_client.generate(prompt=full_prompt, model=model, max_tokens=max_tokens, temperature=0.7)
        print(f"Cohere response (model={model}, max_tokens={max_tokens}):", response)
        if hasattr(response, 'generations') and response.generations:
            text = response.generations[0].text
            print("Cohere generations text:", repr(text))
            if text and text.strip():
                answer = text.strip()
                # If answer is cut off or ends mid-sentence, try to continue
                if allow_continue and (len(answer.split()) > 10 and (answer.endswith(',') or answer.endswith('which') or answer.endswith('and') or not answer[-1] in '.!?')):
                    continue_prompt = f"Continue the answer from where it left off. Do not repeat previous sentences.\n\nPrevious answer:\n{answer}\n\nContinue:"
                    response2 = cohere_client.generate(prompt=continue_prompt, model=model, max_tokens=max_tokens, temperature=0.7)
                    if hasattr(response2, 'generations') and response2.generations:
                        cont = response2.generations[0].text
                        if cont and cont.strip():
                            answer += ' ' + cont.strip()
                return answer
            else:
                print("Cohere returned empty or whitespace-only text.")
                return None
        else:
            print("Cohere returned no generations.")
            return None
    except Exception as e:
        print("Cohere LLM error:", e)
        return None

def huggingface_answer(prompt, intent=None, allow_continue=True):
    if not hf_llm:
        return None
    max_length = get_dynamic_max_tokens(intent)
    try:
        result = hf_llm(prompt, max_length=max_length, do_sample=False)
        if result and isinstance(result, list):
            answer = result[0]['generated_text'].strip()
            # If answer is cut off or ends mid-sentence, try to continue
            if allow_continue and (len(answer.split()) > 10 and (answer.endswith(',') or answer.endswith('which') or answer.endswith('and') or not answer[-1] in '.!?')):
                continue_prompt = f"Continue the answer from where it left off. Do not repeat previous sentences.\n\nPrevious answer:\n{answer}\n\nContinue:"
                result2 = hf_llm(continue_prompt, max_length=max_length, do_sample=False)
                if result2 and isinstance(result2, list):
                    cont = result2[0]['generated_text'].strip()
                    if cont:
                        answer += ' ' + cont
            return answer
        return None
    except Exception as e:
        print(f"HuggingFace LLM error: {e}")
        return None

def detect_intent(question):
    # Advanced keyword-based analysis for precise intent
    q = question.lower().strip()
    if any(phrase in q for phrase in ["full form of", "expand", "abbreviation of", "stands for"]):
        return "full-form"
    if any(word in q for word in ["advantage", "advantages", "pro", "pros", "benefit", "benefits", "disadvantage", "disadvantages", "con", "cons"]):
        return "pros-cons"
    if any(word in q for word in ["difference", "differentiate", "vs", "versus", "compare", "comparison"]):
        return "comparison"
    if any(word in q for word in ["step", "steps", "process", "procedure", "how to", "how do", "how does"]):
        return "how-to"
    if any(word in q for word in ["example", "examples", "instance", "instances"]):
        return "examples"
    if any(word in q for word in ["summary", "summarize", "in short", "briefly"]):
        return "summary"
    # Zero-shot classification if available
    candidate_labels = ["definition", "full-form", "explanation", "list", "comparison", "how-to", "pros-cons", "examples", "summary", "general"]
    if zero_shot_classifier:
        try:
            result = zero_shot_classifier(question, candidate_labels)
            return result['labels'][0]
        except Exception as e:
            print(f"Zero-shot intent detection error: {e}")
    # Fallback to regex-based detection
    if re.match(r"^(what is|what's|define|stands for|full form of|expand|abbreviation of|meaning of)\b", q):
        return "definition"
    if re.search(r"\bdefinition of\b|\bstands for\b|\bmeaning of\b|\bexpand\b|\babbreviation\b", q):
        return "definition"
    if re.match(r"^(list|show|give|what are|which are|name|types of|examples of)\b", q):
        return "list"
    if re.search(r"\btypes of\b|\bexamples of\b|\blist\b|\bshow me\b|\bgive some\b", q):
        return "list"
    if re.match(r"^(how|explain|describe|process of|steps of|how do|how does|how to)\b", q):
        return "explanation"
    if re.search(r"\bprocess\b|\bsteps\b|\bexplain\b|\bdescribe\b|\bhow\b", q):
        return "explanation"
    return "general"

def expand_query(query):
    words = re.findall(r'\w+', query.lower())
    expanded = set(words)
    for word in words:
        if word in SYNONYMS:
            expanded.update(SYNONYMS[word])
    return ' '.join(expanded)

# --- Document Processing ---
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        with pdfplumber.open(file_path) as pdf:
            return '\n'.join(page.extract_text() or '' for page in pdf.pages)
    elif ext == '.docx':
        doc = Document(file_path)
        return '\n'.join([p.text for p in doc.paragraphs])
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    else:
        return ''

def simple_sentence_split(text):
    """Simple sentence splitting using punctuation and heuristics"""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:
            cleaned_sentences.append(sentence)
    return cleaned_sentences

def definition_likeness_score(chunk, abbr):
    chunk_lower = chunk.lower()
    abbr_lower = abbr.lower()
    score = 0
    if f"{abbr_lower} is" in chunk_lower: score += 3
    if f"{abbr_lower} stands for" in chunk_lower: score += 4
    if f"is called {abbr_lower}" in chunk_lower: score += 2
    if f"is abbreviated as {abbr_lower}" in chunk_lower: score += 2
    if f"({abbr_lower})" in chunk_lower: score += 1
    if abbr_lower in chunk_lower: score += 1
    return score

def paragraph_chunking(text):
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    return paragraphs

def synthesize_chunk(chunk):
    # Use the summarizer to condense the chunk
    try:
        summary = summarizer(chunk, max_length=60, min_length=20, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Summarization failed: {e}")
        return chunk  # fallback to original if summarization fails

def synthesize_chunks(chunks):
    synthesized = []
    batch_size = 8  # Tune this based on your hardware
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        # Only summarize if chunk is long enough
        to_summarize = [c for c in batch if len(c) > 400]
        to_keep = [c for c in batch if len(c) <= 400]
        try:
            if to_summarize:
                summaries = summarizer(to_summarize, max_length=60, min_length=20, do_sample=False)
                synthesized.extend([s['summary_text'] for s in summaries])
            synthesized.extend(to_keep)
        except Exception as e:
            print(f"Batch summarization failed: {e}")
            synthesized.extend(batch)  # fallback to original
    return synthesized

# --- Gradio Functions ---
def upload_files(files):
    global documents, embeddings, index
    new_texts = []
    new_sources = []
    for file in files:
        file_path = file.name
        text = extract_text(file_path)
        print(f"[DEBUG] Extracted text from {file_path}:\n{text[:1000]}\n---END OF EXTRACT---")
        if text:
            paragraphs = paragraph_chunking(text)
            for chunk in paragraphs:
                new_texts.append(chunk)
                new_sources.append(os.path.basename(file_path))
    if new_texts:
        new_embs = embedder.encode(new_texts, show_progress_bar=True)
        index.add(np.array(new_embs, dtype='float32'))
        documents.extend(zip(new_texts, new_sources))
        embeddings.extend(new_embs)
    return f"Uploaded and indexed {len(new_texts)} chunks from {len(files)} file(s)."

def extract_full_form_from_context(context, abbreviation):
    # Look for patterns like IDS (Intrusion Detection System) or IDS stands for Intrusion Detection System
    pattern1 = re.compile(rf"{re.escape(abbreviation)}\s*\(([^)]+)\)", re.IGNORECASE)
    pattern2 = re.compile(rf"{re.escape(abbreviation)}\s+stands for\s+([A-Za-z ]+)", re.IGNORECASE)
    match1 = pattern1.search(context)
    if match1:
        return match1.group(1).strip()
    match2 = pattern2.search(context)
    if match2:
        return match2.group(1).strip()
    return None

# Update cohere_answer to use 'command' model for definition/full-form

def keyword_retrieve(question, documents, top_k=5):
    # Simple keyword search: return chunks containing all keywords from the question
    keywords = re.findall(r'\w+', question.lower())
    scored = []
    for i, (chunk, _) in enumerate(documents):
        chunk_lower = chunk.lower()
        score = sum(1 for kw in keywords if kw in chunk_lower)
        if score > 0:
            scored.append((score, i))
    scored.sort(reverse=True)
    return [documents[i][0] for score, i in scored[:top_k]]

def extract_key_sentences(context, question):
    # Extract sentences from context that contain any keyword from the question
    keywords = re.findall(r'\w+', question.lower())
    sentences = re.split(r'(?<=[.!?])\s+', context)
    key_sents = [s for s in sentences if any(kw in s.lower() for kw in keywords)]
    # If nothing found, return the whole context
    return ' '.join(key_sents) if key_sents else context

def context_is_relevant(context, question):
    # Check if any main keyword from the question is in the context
    keywords = set(re.findall(r'\w+', question.lower()))
    context_lower = context.lower()
    return any(kw in context_lower for kw in keywords if len(kw) > 2)

def is_too_similar(answer, context, threshold=0.7):
    seq = difflib.SequenceMatcher(None, answer.lower(), context.lower())
    return seq.ratio() > threshold

def paraphrase_with_hf(answer):
    result = paraphraser(f"paraphrase: {answer}", max_length=256, num_return_sequences=1)
    return result[0]['generated_text']

def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set(['the', 'is', 'in', 'at', 'of', 'a', 'and', 'to', 'for', 'on', 'with', 'as', 'by', 'an', 'be', 'are', 'from', 'or', 'that', 'this', 'it', 'has', 'have', 'was', 'were', 'but', 'not', 'can', 'will', 'if', 'may', 'shall', 'he', 'she', 'they', 'their', 'his', 'her'])
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return ' '.join(list(set(keywords)))

def is_definition_question(question):
    return any(q in question.lower() for q in ["what is", "define", "meaning of", "explain"])

def rephrase_answer(answer, context, question=None):
    if not cohere_client:
        return answer
    if question is None:
        question = ""
    print(f"[DEBUG] Context sent to LLM: {context}")
    if is_definition_question(question):
        # Fallback: if the context does not contain the key term, use LLM's own knowledge
        key_term = question.lower().replace('what is', '').replace('define', '').replace('meaning of', '').replace('explain', '').strip(' ?')
        if key_term and key_term not in context.lower():
            prompt = (
                f"Answer the following question as accurately as possible using your own knowledge.\nQuestion: {question}\nAnswer:"
            )
        else:
            prompt = (
                f"Based on the following context, define the term or answer the question '{question}' in your own words. "
                "Do not copy any sentences or phrases from the original document. Write a single, clear paragraph.\n\n"
                f"Context: {context}\n"
                f"Question: {question}\n"
                "Answer:\n"
            )
    else:
        keywords = extract_keywords(context)
        prompt = (
            "Using only the following keywords, answer the user's question in your own words, as if explaining to a friend. "
            "Do not copy any sentences or phrases from the original document. Write a single, clear paragraph.\n\n"
            f"Keywords: {keywords}\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    response = cohere_client.generate(prompt=prompt, max_tokens=256, temperature=0.7)
    if hasattr(response, 'generations') and response.generations:
        result = response.generations[0].text.strip()
    else:
        result = answer
    return paraphrase_with_hf(result)

def all_word_chunks(query):
    words = query.split()
    n = len(words)
    chunks = []
    for size in range(1, n+1):
        for i in range(n - size + 1):
            chunk = " ".join(words[i:i+size])
            chunks.append(chunk)
    return chunks

def build_powerful_prompt(question, context, intent=None):
    if not intent:
        intent = detect_intent(question)
    
    # Check if the user explicitly asks for advantages/disadvantages
    wants_advantages = any(word in question.lower() for word in ["advantage", "advantages", "pros"])
    wants_disadvantages = any(word in question.lower() for word in ["disadvantage", "disadvantages", "cons"])
    wants_pros_cons = wants_advantages or wants_disadvantages
    
    # Synthesis-focused, explicit instruction for all answer types
    base_instruction = (
        "Using the information below, write a clear, well-structured, and complete answer to the user's question. "
        "Synthesize the information, do not copy or repeat sentences, and explain in your own words as if teaching a beginner. "
        "If the answer requires a comparison, use a table or bullet points for clarity. "
    )
    if intent in ["definition", "full-form"]:
        instruction = (
            base_instruction +
            "Only provide the full form (expanded version) of the abbreviation or acronym mentioned in the question. "
            "Do NOT provide any explanation, context, or extra information. "
            "If the full form is not found, say 'Not found.'"
        )
    elif intent == "explanation":
        instruction = (
            base_instruction +
            "Explain the concept in detail, using examples or analogies if helpful. "
            "Synthesize information from the context and write in a way that's easy to understand."
        )
    elif intent == "list":
        instruction = (
            base_instruction +
            "List the relevant items, features, or points from the context in a clear, organized manner. "
            "Do NOT copy verbatim; paraphrase and group related items."
        )
    elif intent == "comparison":
        instruction = (
            base_instruction +
            "Clearly and concisely explain the main differences between the two concepts mentioned in the question. "
            "Do NOT include advantages or disadvantages unless the question specifically asks for them."
        ) if not wants_pros_cons else (
            base_instruction +
            "Compare the two concepts mentioned in the question, including any advantages or disadvantages if relevant."
        )
    elif intent == "how-to":
        instruction = (
            base_instruction +
            "Describe the process or steps required to accomplish the task, using the context. "
            "Present the steps in a logical order and explain each step clearly."
        )
    elif intent == "pros-cons":
        instruction = (
            base_instruction +
            "List the advantages and disadvantages, or pros and cons, of the topic using the context. "
            "Present them in a balanced and organized manner."
        )
    elif intent == "examples":
        instruction = (
            base_instruction +
            "Provide relevant examples from the context to illustrate the concept. "
            "Explain each example briefly."
        )
    elif intent == "summary":
        instruction = (
            base_instruction +
            "Summarize the main points from the context in a concise paragraph."
        )
    else:
        instruction = (
            base_instruction +
            "Write a comprehensive, well-structured answer to the user's question using ONLY the information in the context below. "
            "Synthesize, paraphrase, and connect information from different parts of the context. "
            "If the answer is not found, say so."
        )
    prompt = (
        f"{instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt

def filter_pros_cons(answer, question):
    # Only filter if the question does NOT ask for pros/cons/advantages/disadvantages
    if not any(word in question.lower() for word in ["advantage", "advantages", "pros", "disadvantage", "disadvantages", "cons"]):
        # Remove sections containing those words (case-insensitive)
        pattern = re.compile(
            r'(^.*(advantage|disadvantage|pros|cons)[^\n]*(:)?(\n|$)(.*\n)*?)(?=\n\s*\n|^\s*[A-Z][a-z]+:|$)',
            re.IGNORECASE | re.MULTILINE
        )
        answer = pattern.sub('', answer)
        # Remove any line containing those words as a fallback
        answer = '\n'.join(
            [line for line in answer.split('\n') if not re.search(r'(advantage|disadvantage|pros|cons)', line, re.IGNORECASE)]
        )
        # Remove extra blank lines
        answer = re.sub(r'\n\s*\n', '\n\n', answer)
        answer = answer.strip()
    return answer

# Remove 'top_k' and 'paraphrase' parameters from answer_question and Gradio UI
# Set top_k=5 and paraphrase=True internally

def synthesize_chunks_for_context(chunks):
    try:
        summaries = summarizer(chunks, max_length=60, min_length=20, do_sample=False)
        return [s['summary_text'] for s in summaries]
    except Exception as e:
        print(f"Context summarization failed: {e}")
        return chunks  # fallback to original



def answer_question(question):
    top_k = 7
    paraphrase = True
    intent = detect_intent(question)
    if intent == "summary":
        # Get all unique document sources
        doc_files = list(set([src for _, src in documents]))
        # Extract full text for each document
        full_texts = []
        for doc_file in doc_files:
            try:
                text = extract_text(doc_file)
                if not text.strip():
                    text = '\n'.join([chunk for chunk, src in documents if src == doc_file])
            except Exception:
                text = '\n'.join([chunk for chunk, src in documents if src == doc_file])
            full_texts.append(text)
        summaries = summarize_documents_separately(full_texts)
        # Label each summary for clarity
        labeled_summaries = []
        for idx, summary in enumerate(summaries):
            label = f"Summary of Document {idx+1}:"
            labeled_summaries.append(f"{label}\n{summary}")
        return '\n\n'.join(labeled_summaries)
    if not documents or not embeddings:
        prompt = (
            f"Answer the following question as accurately and informatively as possible using your own knowledge.\n"
            f"Question: {question}"
        )
        print("Prompt sent to LLM (no docs):", prompt)
        intent = detect_intent(question)
        answer = cohere_answer(prompt, intent)
        if answer:
            return paraphrase_with_hf(answer)
        answer_hf = huggingface_answer(prompt, intent)
        if answer_hf:
            return answer_hf
        fallback_prompt = f"Answer this question: {question}"
        print("Retrying Cohere with fallback prompt:", fallback_prompt)
        answer = cohere_answer(fallback_prompt, intent)
        if answer:
            return paraphrase_with_hf(answer)
        answer_hf = huggingface_answer(fallback_prompt, intent)
        if answer_hf:
            return answer_hf
        return "Sorry, I couldn't find an answer to your question."
    # Use synthesized summaries for all documents as context for the LLM
    doc_files = list(set([src for _, src in documents]))
    full_texts = []
    for doc_file in doc_files:
        try:
            text = extract_text(doc_file)
            if not text.strip():
                text = '\n'.join([chunk for chunk, src in documents if src == doc_file])
        except Exception:
            text = '\n'.join([chunk for chunk, src in documents if src == doc_file])
        full_texts.append(text)
    # Use backend LLM synthesis
    return generate_synthesized_answer_from_summaries(question, full_texts)



# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Document Chatbot \n Ask questions about your documents.")
    with gr.Row():
        file_input = gr.File(file_count="multiple", file_types=[".pdf", ".docx", ".txt", ".ppt"], label="Upload Documents")
        upload_btn = gr.Button("Upload & Index")
    upload_output = gr.Textbox(label="Upload & Index Status")
    upload_btn.click(upload_files, inputs=file_input, outputs=upload_output)
    gr.Markdown("---")
    question = gr.Textbox(label="Ask a Question")
    answer = gr.Textbox(label="Answer")
    ask_btn = gr.Button("Get Answer")
    ask_btn.click(answer_question, inputs=[question], outputs=answer)

demo.launch(share=True)

