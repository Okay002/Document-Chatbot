import os
from transformers import pipeline
import datetime
import json
import cohere
import re
from transformers import pipeline as hf_pipeline
import concurrent.futures

# Authenticate with Hugging Face Hub
hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Authenticate with Cohere
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
cohere_client = None
if COHERE_API_KEY:
    try:
        cohere_client = cohere.Client(COHERE_API_KEY)
    except Exception as e:
        print(f"[WARNING] Could not initialize Cohere: {e}")

# Fallback models
try:
    qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")
except Exception as e:
    qa_pipeline = None
    print(f"[WARNING] Could not load BERT QA model: {e}")

try:
    generative_pipeline = pipeline(
        "text-generation",
        model="meta-llama/Llama-2-13b-chat-hf",  # You can change this to any available HF model
        device_map="auto",
        torch_dtype="auto",
        use_auth_token=hf_token
    )
except Exception as e:
    generative_pipeline = None
    print(f"[WARNING] Could not load Llama-2-13B-Chat: {e}")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Using ONLY the information in the provided context, "
    "answer the user's question. If the answer is not in the context, say 'I don't know.'\n\n"
    "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
)

# Load once at module level
try:
    intent_classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased",
        top_k=3
    )
except Exception as e:
    intent_classifier = None
    print(f"[WARNING] Could not load intent classification model: {e}")

# Load once at module level
try:
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except Exception as e:
    zero_shot_classifier = None
    print(f"[WARNING] Could not load zero-shot classification model: {e}")

# Load summarization model (facebook/bart-large-cnn)
summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")

def is_summary_query(query):
    summary_keywords = ["summarize", "summary", "overview", "brief", "main points", "key points", "short version", "in short"]
    ql = query.lower()
    return any(kw in ql for kw in summary_keywords)

def analyze_query_intent(query):
    """Analyze the intent and type of the query"""
    intent_analysis = {
        'type': 'general',  # general, comparison, explanation, definition, how_to
        'requires_context': True,
        'keywords': [],
        'entities': []
    }
    
    query_lower = query.lower()
    
    # Detect query types
    if any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning']):
        intent_analysis['type'] = 'definition'
    elif any(word in query_lower for word in ['how', 'steps', 'process', 'procedure']):
        intent_analysis['type'] = 'how_to'
    elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'between']):
        intent_analysis['type'] = 'comparison'
    elif any(word in query_lower for word in ['explain', 'why', 'reason', 'cause']):
        intent_analysis['type'] = 'explanation'
    elif any(word in query_lower for word in ['list', 'examples', 'types', 'kinds']):
        intent_analysis['type'] = 'listing'
    
    # Extract keywords (simple approach)
    keywords = re.findall(r'\b\w+\b', query_lower)
    intent_analysis['keywords'] = [kw for kw in keywords if len(kw) > 3 and kw not in ['what', 'when', 'where', 'which', 'that', 'this', 'with', 'from', 'about', 'into', 'during', 'before', 'after', 'above', 'below']]
    
    return intent_analysis

def advanced_analyze_query_intent(query):
    """
    Use a transformer model to classify the intent of the query.
    Returns top predicted intents/labels.
    """
    if intent_classifier is None:
        # Fallback to basic intent analysis
        intent_analysis = analyze_query_intent(query)
        return [intent_analysis['type']]
    
    try:
        results = intent_classifier(query)
        # results is a list of dicts: [{'label': '...', 'score': ...}, ...]
        top_intents = [r['label'] for r in results[0]]
        return top_intents
    except Exception as e:
        print(f"Intent classification error: {e}")
        # Fallback to basic intent analysis
        intent_analysis = analyze_query_intent(query)
        return [intent_analysis['type']]

def zero_shot_intent(query, candidate_labels=None):
    if candidate_labels is None:
        candidate_labels = ["definition", "explanation", "comparison", "how-to", "listing"]
    
    if zero_shot_classifier is None:
        # Fallback to basic intent analysis
        intent_analysis = analyze_query_intent(query)
        return [intent_analysis['type']]
    
    try:
        result = zero_shot_classifier(query, candidate_labels)
        return result['labels'][:3]  # Top 3 intents
    except Exception as e:
        print(f"Zero-shot classification error: {e}")
        # Fallback to basic intent analysis
        intent_analysis = analyze_query_intent(query)
        return [intent_analysis['type']]

def build_semantic_prompt(query, chunks, history=None, intent_labels=None):
    """Build a comprehensive prompt for semantic understanding"""
    # Set instruction based on detected intent
    instruction = ""
    if intent_labels:
        top_intent = intent_labels[0]
        if top_intent == "definition":
            instruction = "Provide a concise definition based on the documents."
        elif top_intent == "explanation":
            instruction = "Provide a detailed explanation using the documents."
        elif top_intent == "comparison":
            instruction = "Compare the relevant concepts using the documents."
        elif top_intent == "how-to":
            instruction = "Describe the process or steps using the documents."
        elif top_intent == "listing":
            instruction = "List the relevant items or points from the documents."
    if not instruction:
        instruction = "Provide a comprehensive answer based only on the information in the documents."

    # --- BEGIN: Prompt Engineering Update ---
    # Add explicit instruction for own words, synthesis, and examples
    instruction = (
        "Using the information below, explain in your own words, as if teaching a beginner. "
        "Do not copy or paraphrase directly. Synthesize and summarize the key points, and provide examples if possible.\n"
        + instruction +
        " Write a complete, detailed answer. Do not stop until all relevant points are covered. If the answer is long, continue until fully complete."
    )
    # --- END: Prompt Engineering Update ---

    # Merge chunks with better context management
    context_parts = []
    for i, chunk in enumerate(chunks[:12]):  # Increased from 6 to 12
        context_parts.append(f"Document Section {i+1}:\n{chunk['text']}\n")
    
    context = '\n'.join(context_parts)
    
    # Build conversation history
    conversation_history = ""
    if history:
        conversation_history = "\n\nPrevious conversation:\n"
        for turn in history[-3:]:  # Last 3 turns
            conversation_history += f"User: {turn['query']}\nAssistant: {turn['answer']}\n"
    
    prompt = f"""You are an expert assistant with deep knowledge of the provided documents. Your task is to provide comprehensive, accurate answers based ONLY on the information in the documents.\n\n{instruction}\n\nIMPORTANT GUIDELINES:\n- Use ONLY information from the provided document sections\n- Provide comprehensive, detailed answers that fully address the question\n- If information is not available in the documents, clearly state \"I don't have enough information to answer this question\"\n- Connect related information from different sections when relevant\n- Use specific details and examples from the documents\n- Maintain academic/professional tone\n- Structure your answer logically with clear sections if needed\n- EXPLAIN IN YOUR OWN WORDS. Do NOT copy or paraphrase directly. Synthesize and summarize the key points. Provide examples if possible.\n\nDocument Context:\n{context}\n\n{conversation_history}\n\nUser Question: {query}\n\nPlease provide a comprehensive answer:"""

    return prompt

# Aggressive continue logic for completeness
import re

def is_answer_complete_strict(answer, min_words=250):
    # Check for sentence-ending punctuation and minimum word count
    if len(answer.split()) < min_words:
        return False
    if re.search(r'[.!?]["\']?$|\n$', answer.strip()):
        abrupt = [
            'and', 'or', 'such as', 'including', 'based on', 'with', 'by', 'for example', 'to', 'from', 'as well as', 'in order to', 'because', 'due to', 'so that', 'but', 'while', 'whereas', 'although', 'however', 'therefore', 'thus', 'since', 'after', 'before', 'if', 'when', 'then', 'which', 'that', 'who', 'whose', 'whom', 'where', 'while', 'until', 'unless', 'except', 'like', 'among', 'between', 'during', 'within', 'without', 'through', 'over', 'under', 'about', 'against', 'amongst', 'beside', 'besides', 'beyond', 'despite', 'inside', 'outside', 'onto', 'upon', 'via', 'according to', 'regarding', 'concerning', 'because of', 'instead of', 'regardless of', 'apart from', 'as far as', 'as soon as', 'as long as', 'as much as', 'as well as', 'even though', 'even if', 'in case', 'in spite of', 'in addition to', 'in front of', 'in place of', 'in spite of', 'on account of', 'on behalf of', 'on top of', 'out of', 'owing to', 'prior to', 'subsequent to', 'such as', 'thanks to', 'up to', 'with reference to', 'with regard to', 'with respect to', 'with a view to', 'with the exception of', 'yet'
        ]
        last_words = answer.strip().split()[-4:]
        last_phrase = ' '.join(last_words).lower()
        for ab in abrupt:
            if last_phrase.endswith(ab) or last_phrase == ab:
                return False
        return True
    return False

# --- Updated Cohere Answer Generation with Continue Logic ---
def generate_answer_with_cohere(query, chunks, history=None, max_continues=8):
    if not cohere_client:
        return None
    try:
        intent_labels = zero_shot_intent(query)
        prompt = build_semantic_prompt(query, chunks, history, intent_labels)
        response = cohere_client.generate(
            model='command',
            prompt=prompt,
            max_tokens=1200,
            temperature=0.7
        )
        if hasattr(response, 'generations') and response.generations:
            answer = response.generations[0].text.strip()
            continue_count = 0
            while not is_answer_complete_strict(answer) and continue_count < max_continues:
                followup = cohere_client.generate(
                    model='command',
                    prompt=f"Continue the answer.\n{answer}",
                    max_tokens=600,
                    temperature=0.7
                )
                if hasattr(followup, 'generations') and followup.generations:
                    addition = followup.generations[0].text.strip()
                    answer += ' ' + addition
                else:
                    break
                continue_count += 1
            return answer
        else:
            return None
    except Exception as e:
        print(f"Cohere LLM error: {e}")
        return None

# --- Updated HuggingFace Answer Generation with Continue Logic ---
def generate_answer_with_hf(query, chunks, history=None, max_continues=8):
    """Generate answer using Hugging Face LLM for better semantic understanding"""
    try:
        intent_analysis = analyze_query_intent(query)
        prompt = build_semantic_prompt(query, chunks, history, intent_analysis)
        if generative_pipeline:
            response = generative_pipeline(
                prompt,
                max_length=len(prompt.split()) + 1200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=generative_pipeline.tokenizer.eos_token_id
            )
            answer = response[0]['generated_text'][len(prompt):].strip()
            continue_count = 0
            while not is_answer_complete_strict(answer) and continue_count < max_continues:
                followup = generative_pipeline(
                    f"Continue the answer.\n{answer}",
                    max_length=600,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=generative_pipeline.tokenizer.eos_token_id
                )
                addition = followup[0]['generated_text'].strip()
                answer += ' ' + addition
                continue_count += 1
            return answer
        else:
            return None
    except Exception as e:
        print(f"Hugging Face LLM error: {e}")
        return None

# --- Updated Local LLM Answer Generation with Continue Logic ---
def generate_answer_with_local_llm(query, chunks, history=None, max_continues=8):
    """Generate answer using local LLM as fallback"""
    try:
        # Use zero-shot intent detection
        intent_labels = zero_shot_intent(query)
        print("Detected intent labels:", intent_labels)
        prompt = build_semantic_prompt(query, chunks, history, intent_labels)
        
        if generative_pipeline:
            response = generative_pipeline(
                prompt,
                max_length=len(prompt.split()) + 1200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=generative_pipeline.tokenizer.eos_token_id
            )
            answer = response[0]['generated_text'][len(prompt):].strip()
            continue_count = 0
            while not is_answer_complete_strict(answer) and continue_count < max_continues:
                followup = generative_pipeline(
                    f"Continue the answer.\n{answer}",
                    max_length=600,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=generative_pipeline.tokenizer.eos_token_id
                )
                addition = followup[0]['generated_text'].strip()
                answer += ' ' + addition
                continue_count += 1
            return answer
        else:
            return None
    except Exception as e:
        print(f"Local LLM error: {e}")
        return None

def generate_answer_with_bert(query, chunks, history=None):
    """Fallback to BERT QA for simple questions"""
    try:
        # Merge chunks for BERT
        context_parts = []
        for chunk in chunks[:5]:  # Use fewer chunks for BERT
            context_parts.append(chunk['text'])
        context = ' '.join(context_parts)
        
        if qa_pipeline and context.strip():
            result = qa_pipeline(question=query, context=context)
            return result['answer']
        else:
            return "I don't have enough information to answer this question."
    except Exception as e:
        print(f"BERT QA error: {e}")
        return "I don't have enough information to answer this question."

def merge_chunks(chunks, max_words=800):  # Increased from 300 to 800
    """Merge chunks with better overlap handling"""
    merged = []
    current = ""
    current_chunks = []
    
    for chunk in chunks:
        chunk_text = chunk['text']
        words = chunk_text.split()
        
        if len((current + " " + chunk_text).split()) <= max_words:
            current = (current + " " + chunk_text).strip()
            current_chunks.append(chunk)
        else:
            if current:
                merged.append({
                    'text': current,
                    'chunks': current_chunks
                })
            current = chunk_text
            current_chunks = [chunk]
    
    if current:
        merged.append({
            'text': current,
            'chunks': current_chunks
        })
    
    return merged

# Replace format_answer to always return a single paragraph

def format_answer(answer):
    """Always return the answer as a single, complete paragraph (no bullet points or lists)."""
    lines = [line.strip() for line in answer.split('\n') if line.strip()]
    return ' '.join(lines)

# Remove general knowledge fallback logic and restore document-only answer generation

def generate_answer(query, chunks, history=None):
    """Main answer generation function with multiple fallback options, always paragraph formatting, and dedicated summarization for summary queries."""
    if history is None:
        history = []
    if not chunks:
        return "I don't have enough information to answer this question."
    if is_summary_query(query):
        # Merge all chunks into one text for summarization
        merged_text = ' '.join([chunk['text'] for chunk in chunks])
        # BART has a max token limit, so chunk if needed
        max_input_len = 1024  # tokens, but we'll use words as a proxy
        words = merged_text.split()
        if len(words) > max_input_len:
            merged_text = ' '.join(words[:max_input_len])
        summary = summarizer(merged_text, max_length=250, min_length=50, do_sample=False)
        answer = summary[0]['summary_text']
        answer = format_answer(answer)
        with open('backend/query_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now()}\nQ: {query}\nA: {answer}\nMethod: BART Summarizer\n\n")
        return answer
    # Try Cohere LLM first
    answer = generate_answer_with_cohere(query, chunks, history)
    if answer:
        answer = format_answer(answer)
        with open('backend/query_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now()}\nQ: {query}\nA: {answer}\nMethod: Cohere LLM\n\n")
        return answer
    # Try Hugging Face LLM next
    answer = generate_answer_with_hf(query, chunks, history)
    if answer:
        answer = format_answer(answer)
        with open('backend/query_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now()}\nQ: {query}\nA: {answer}\nMethod: HuggingFace LLM\n\n")
        return answer
    # Try local LLM as fallback
    answer = generate_answer_with_local_llm(query, chunks, history)
    if answer:
        answer = format_answer(answer)
        with open('backend/query_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now()}\nQ: {query}\nA: {answer}\nMethod: Local LLM\n\n")
        return answer
    # Fallback to BERT QA
    answer = generate_answer_with_bert(query, chunks, history)
    answer = format_answer(answer)
    with open('backend/query_log.txt', 'a', encoding='utf-8') as f:
        f.write(f"{datetime.datetime.now()}\nQ: {query}\nA: {answer}\nMethod: BERT QA\n\n")
    return answer

def recursive_summarize(text, summarizer, max_chunk_words=600, max_length=350, min_length=120):
    instruction = (
        "Summarize the following document in your own words, focusing on the main arguments, findings, and conclusions. "
        "Be concise but cover all key points. Do not copy sentences directly.\n\n"
    )
    input_text = instruction + text
    words = text.split()
    if len(words) <= max_chunk_words:
        try:
            summary = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"[WARNING] Summarization failed for chunk: {e}")
            sentences = text.split('. ')
            return '. '.join(sentences[:3]) + '.'
    # Split into smaller chunks
    chunks = [' '.join(words[i:i+max_chunk_words]) for i in range(0, len(words), max_chunk_words)]
    chunk_summaries = []
    for idx, chunk in enumerate(chunks):
        print(f"[INFO] Summarizing chunk {idx} of document: {len(chunk.split())} words. First 100 chars: {chunk[:100]}")
        chunk_summaries.append(recursive_summarize(chunk, summarizer, max_chunk_words, max_length, min_length))
    combined = ' '.join(chunk_summaries)
    # Summarize the combined summaries
    return recursive_summarize(combined, summarizer, max_chunk_words, max_length, min_length)


def preprocess_text(text):
    # Remove lines with metadata
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not re.match(r'^(Name|RollNo|Branch|Section|Assignment|Class Test|Department|Course|Prof|Professor|IIT|Kharagpur|Test|Date|Subject|Email|Phone|ID|RegNo|Reg No|Student|Teacher|Instructor|Batch|Year|Semester|Session|Code|Title|Page|Signature|Marks|Total|Score|Grade|Exam|Paper|Sheet|Number|ID):', line.strip(), re.IGNORECASE):
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def remove_instructions(text):
    # Remove lines that look like summarization instructions or prompts
    instruction_patterns = [
        r'^Summarize the following text.*',
        r'^Do not copy sentences directly.*',
        r'^Be concise but cover all key points.*',
        r'^Write a case study.*',
        r'^1\.',
        r'^Ans\.',
        r'^Q[0-9]+\.',
        r'^Question.*',
        r'^Answer.*',
        r'^\d+\.',
        r'^- ',
        r'^o ',
        r'^• ',
    ]
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not any(re.match(pat, line.strip(), re.IGNORECASE) for pat in instruction_patterns):
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def chunk_text(text, chunk_size=200, max_chunk_size=400):
    # Split by single newlines, then group into chunks of ~chunk_size words
    paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len((current + " " + para).split()) <= chunk_size:
            current = (current + " " + para).strip()
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    # Further split any chunk that is too large
    final_chunks = []
    for chunk in chunks:
        words = chunk.split()
        if len(words) > max_chunk_size:
            for i in range(0, len(words), max_chunk_size):
                final_chunks.append(' '.join(words[i:i+max_chunk_size]))
        else:
            final_chunks.append(chunk)
    return final_chunks

def chunk_by_section(text, min_section_words=600):
    # Split by common section/heading keywords
    section_pattern = re.compile(r'(?=\b(Case Study|Unit|Section|Conclusion|Findings|Overview|Summary|Introduction|Background|Analysis|Discussion|Result|Results|Recommendation|Recommendations|Lesson|Lessons|Topic|Chapter|Part|Subsection|Objective|Aim|Purpose|Scope|Method|Methods|Approach|Experiment|Observation|Observations|Test|Testing|Evaluation|Assessment|Review|Insight|Insights|Implication|Implications|Future Work|Limitation|Limitations|Appendix|Reference|References|Bibliography)\b)', re.IGNORECASE)
    sections = [s.strip() for s in section_pattern.split(text) if s.strip()]
    # Re-attach headings to their content
    chunks = []
    i = 0
    while i < len(sections):
        if re.match(r'^(Case Study|Unit|Section|Conclusion|Findings|Overview|Summary|Introduction|Background|Analysis|Discussion|Result|Results|Recommendation|Recommendations|Lesson|Lessons|Topic|Chapter|Part|Subsection|Objective|Aim|Purpose|Scope|Method|Methods|Approach|Experiment|Observation|Observations|Test|Testing|Evaluation|Assessment|Review|Insight|Insights|Implication|Implications|Future Work|Limitation|Limitations|Appendix|Reference|References|Bibliography)\b', sections[i], re.IGNORECASE):
            if i+1 < len(sections):
                chunk = sections[i] + ' ' + sections[i+1]
                i += 2
            else:
                chunk = sections[i]
                i += 1
        else:
            chunk = sections[i]
            i += 1
        # Merge with next section if too small
        while len(chunk.split()) < min_section_words and i < len(sections):
            chunk += ' ' + sections[i]
            i += 1
        chunks.append(chunk)
    return chunks

def final_synthesize_with_llm(text):
    prompt = (
        "Write a detailed, synthesized summary of the following content, focusing on the unique arguments, findings, and conclusions presented in the document. "
        "Do not just define terms or repeat general knowledge. Highlight what is specific and important in this document. "
        "Be thorough, clear, and use original language.\n\nContent:\n" + text
    )
    try:
        if cohere_client:
            response = cohere_client.generate(
                model='command',
                prompt=prompt,
                max_tokens=800,
                temperature=0.7
            )
            if hasattr(response, 'generations') and response.generations:
                return response.generations[0].text.strip()
    except Exception as e:
        print(f"[WARNING] Cohere LLM final synthesis failed: {e}")
    return text

def process_chunk(chunk, idx, cidx, max_length, min_length):
    print(f"[INFO] Synthesizing section chunk {cidx} of document {idx}: {len(chunk.split())} words. First 100 chars: {chunk[:100]}")
    synthesized = synthesize_chunk_with_llm(chunk)
    try:
        instruction = (
            "Summarize the following section in your own words, focusing on the unique arguments, findings, and conclusions. "
            "Do not just define terms or repeat general knowledge. Highlight what is specific and important in this section. "
            "Be thorough, clear, and use original language.\n\n"
        )
        input_text = instruction + synthesized
        summary = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"[WARNING] Summarization failed for section chunk {cidx} of document {idx}: {e}")
        sentences = synthesized.split('. ')
        fallback = '. '.join(sentences[:3]) + '.'
        return fallback

def summarize_documents_separately(documents, max_length=800, min_length=300):
    """
    Preprocess, chunk by section/heading, summarize each chunk (no LLM synthesis per chunk), then combine and synthesize again for a single concise summary.
    """
    summaries = []
    for idx, doc in enumerate(documents):
        # Preprocess to remove metadata
        if isinstance(doc, list):
            text = ' '.join(doc)
        else:
            text = str(doc)
        text = preprocess_text(text)
        words = text.split()
        if not text.strip() or len(words) < 10:
            print(f"[WARNING] Skipping empty or too-short document at index {idx}")
            summaries.append("Document too short to summarize.")
            continue
        # Chunk the cleaned document by section/heading
        chunks = chunk_by_section(text)
        # Limit number of chunks to 8 by merging extras into the last chunk
        max_chunks = 8
        if len(chunks) > max_chunks:
            # Merge all extra chunks into the last chunk
            chunks = chunks[:max_chunks-1] + [' '.join(chunks[max_chunks-1:])]
        chunk_summaries = []
        for cidx, chunk in enumerate(chunks):
            print(f"[INFO] Summarizing section chunk {cidx} of document {idx}: {len(chunk.split())} words. First 100 chars: {chunk[:100]}")
            try:
                instruction = (
                    "Summarize the following section in your own words, focusing on the unique arguments, findings, and conclusions. "
                    "Do not just define terms or repeat general knowledge. Highlight what is specific and important in this section. "
                    "Be thorough, clear, and use original language.\n\n"
                )
                input_text = instruction + chunk
                summary = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
                chunk_summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"[WARNING] Summarization failed for section chunk {cidx} of document {idx}: {e}")
                sentences = chunk.split('. ')
                fallback = '. '.join(sentences[:3]) + '.'
                chunk_summaries.append(fallback)
        # Combine all section summaries and synthesize again for a single concise summary
        combined_summary = ' '.join(chunk_summaries)
        final_summary = final_synthesize_with_llm(combined_summary)
        summaries.append(final_summary)
    return summaries

def generate_synthesized_answer_from_summaries(query, documents, history=None):
    """
    Summarize each document in own words, combine summaries, and pass as context to the LLM for a synthesized answer.
    Args:
        query: User's question
        documents: List of documents (each as list of chunks or string)
        history: Conversation history
    Returns:
        Synthesized answer string
    """
    # Step 1: Summarize each document in own words
    summaries = summarize_documents_separately(documents)
    # Step 2: Combine summaries into a single context string
    combined_context = '\n'.join(summaries)
    # Step 3: Use the combined summary as the only chunk for the LLM
    summary_chunk = {'text': combined_context}
    # Use the main answer generation logic, but with the summary chunk
    answer = generate_answer(query, [summary_chunk], history)
    return answer

def synthesize_chunk_with_llm(chunk, question=None):
    # Use Cohere LLM to paraphrase/synthesize the chunk in own words if available
    prompt = (
        "Rewrite the following text in your own words, condensing it to the main ideas and making it clear and concise. "
        "Do not copy sentences directly.\n\nText:\n" + chunk
    )
    try:
        if cohere_client:
            response = cohere_client.generate(
                model='command',
                prompt=prompt,
                max_tokens=200,
                temperature=0.7
            )
            if hasattr(response, 'generations') and response.generations:
                return response.generations[0].text.strip()
    except Exception as e:
        print(f"[WARNING] Cohere LLM synthesis failed for chunk: {e}")
    # Fallback: return original chunk
    return chunk