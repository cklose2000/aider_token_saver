import os
import json
import time
import re
import math
import logging
import datetime
from collections import Counter
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import PCA
import hashlib

# Import our logger
try:
    from rag_logger import RagLogger
except ImportError:
    # Define a fallback logger if rag_logger.py is missing
    class RagLogger:
        def log_vector_search(self, *args, **kwargs):
            pass
        def log_token_reduction(self, *args, **kwargs):
            pass
        def log_retrieval_metrics(self, *args, **kwargs):
            pass

class SimpleVectorStore:
    """A simple vector store that uses TF-IDF and cosine similarity without external dependencies."""
    
    def __init__(self, store_path="simple_vector_store.json", enable_logging=True, model_name="bert-base-uncased"):
        self.store_path = store_path
        self.documents = []
        self.word_to_idx = {}  # Maps words to their indices in the vector
        self.idx_to_word = {}  # Maps indices to words
        self.idf = {}  # Inverse document frequency
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.load_store()
        
        # Initialize logger
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = RagLogger()
        
        self.file_vectors = {}  # Store file vectors by path
        self.file_hashes = {}   # Store file hashes to detect changes
    
    def load_store(self):
        """Load the vector store from disk if it exists."""
        logging.debug(f"Loading vector store from path: {self.store_path}")
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, 'r') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', [])
                    self.word_to_idx = data.get('word_to_idx', {})
                    self.idx_to_word = {int(k): v for k, v in data.get('idx_to_word', {}).items()}
                    self.idf = data.get('idf', {})
            except Exception as e:
                print(f"Failed to load vector store: {e}")
                self.documents = []
                self.word_to_idx = {}
                self.idx_to_word = {}
                self.idf = {}
    
    def save_store(self):
        """Save the vector store to disk."""
        data = {
            'documents': self.documents,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'idf': self.idf
        }
        with open(self.store_path, 'w') as f:
            json.dump(data, f)
    
    def tokenize(self, text):
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        return re.findall(r'\w+', text)
    
    def compute_tf(self, tokens):
        """Compute term frequency."""
        tf_dict = {}
        token_count = Counter(tokens)
        total_tokens = len(tokens)
        
        for token, count in token_count.items():
            tf_dict[token] = count / total_tokens
        
        return tf_dict
    
    def update_vocabulary(self, tokens):
        """Update the vocabulary with new tokens."""
        for token in tokens:
            if token not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[token] = idx
                self.idx_to_word[idx] = token
    
    def update_idf(self):
        """Update inverse document frequency for all words."""
        N = len(self.documents)
        
        # Count document frequency for each word
        doc_freq = {}
        for doc in self.documents:
            # Get unique words in document
            unique_words = set(self.tokenize(doc['content']))
            for word in unique_words:
                doc_freq[word] = doc_freq.get(word, 0) + 1
        
        # Compute IDF
        for word, freq in doc_freq.items():
            self.idf[word] = math.log(N / (1 + freq))
    
    def compute_tfidf_vector(self, text):
        """Compute TF-IDF vector for text."""
        tokens = self.tokenize(text)
        tf = self.compute_tf(tokens)
        
        # Initialize vector with zeros
        vector = [0.0] * len(self.word_to_idx)
        
        # Fill in TF-IDF values
        for token, tf_value in tf.items():
            if token in self.word_to_idx and token in self.idf:
                idx = self.word_to_idx[token]
                vector[idx] = tf_value * self.idf.get(token, 0)
        
        return vector
    
    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def vectorize_text(self, text):
        """Vectorize text using a pre-trained model."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def add_document(self, doc_id, content, metadata):
        """Add a document to the vector store."""
        # Tokenize the content
        tokens = self.tokenize(content)
        
        # Update vocabulary
        self.update_vocabulary(tokens)
        
        # Store the document
        document = {
            'doc_id': doc_id,
            'content': content,
            'metadata': metadata,
            'timestamp': time.time()
        }
        
        # Check if document already exists
        for i, doc in enumerate(self.documents):
            if doc['doc_id'] == doc_id:
                # Update existing document
                self.documents[i] = document
                self.update_idf()
                self.save_store()
                return
        
        # Add new document
        self.documents.append(document)
        
        # Update IDF values
        self.update_idf()
        
        # Save to disk
        self.save_store()
    
    def get_vector_count(self):
        """Return the number of vectors currently stored."""
        return len(self.documents)

    def search(self, query, top_k=5):
        # Use BERT for semantic understanding
        bert_results = self.search_bert(query, top_k=top_k*2)
        # Use TF-IDF for keyword matching
        tfidf_results = self.search_tfidf(query, top_k=top_k*2)
        # Combine results with weights
        combined_results = self.merge_results(bert_results, tfidf_results)
        return combined_results[:top_k]
    
    def chunk_code_file(self, file_path, file_content):
        """Split code file into semantic chunks."""
        # Use regex to detect function, class, or import declarations as boundaries
        pattern = re.compile(r'^(def |class |import )')
        lines = file_content.splitlines()
        chunks = []
        current_chunk = []
        start_line = 0

        for i, line in enumerate(lines):
            if pattern.match(line.lstrip()):
                # If a new definition is detected and there is an existing chunk, store it
                if current_chunk:
                    chunks.append({
                        'file_path': file_path,
                        'chunk_content': "\n".join(current_chunk),
                        'start_line': start_line,
                        'end_line': i - 1,
                    })
                    current_chunk = []
                start_line = i
            current_chunk.append(line)
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append({
                'file_path': file_path,
                'chunk_content': "\n".join(current_chunk),
                'start_line': start_line,
                'end_line': len(lines) - 1,
            })
        
        return chunks

    def compress_vector(self, vector, n_components=100):
        """Compress vector while preserving semantic relationships"""
        if not hasattr(self, 'pca'):
            self.pca = PCA(n_components=n_components)
            self.pca.fit([vector])  # Would need to batch train this
        return self.pca.transform([vector])[0]

    def smart_chunk_content(self, content, max_chunk_size=512):
        """Break content into semantic chunks before vectorization"""
        chunks = []
        # Use natural boundaries like function definitions, class declarations
        # or paragraph breaks to create chunks
        return chunks

    def retrieve_relevant_context(self, query_vector, threshold=0.7):
        """Retrieve only the most relevant pre-vectorized content"""
        relevant_chunks = []
        for doc in self.documents:
            similarity = self.cosine_similarity(query_vector, doc['bert_vector'])
            if similarity > threshold:
                relevant_chunks.append(doc['content'])
        return relevant_chunks

    def monitor_vector_stats(self):
        """Monitor vector store performance"""
        stats = {
            'total_vectors': len(self.documents),
            'avg_vector_size': np.mean([len(d.get('bert_vector', [])) for d in self.documents]),
            'storage_size': os.path.getsize(self.store_path),
            'retrieval_times': self.retrieval_times  # Need to track this
        }
        return stats

    def process_file_for_context(self, file_path, content):
        """Process a file for context window, using vectorization if possible"""
        # Generate hash of current content
        current_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Check if we have a cached vector and if file hasn't changed
        if (file_path in self.file_vectors and 
            file_path in self.file_hashes and 
            self.file_hashes[file_path] == current_hash):
            # Return compressed vector representation
            return {
                'type': 'vector',
                'vector': self.file_vectors[file_path],
                'file_path': file_path,
                'summary': self.generate_file_summary(content)
            }
        
        # If file is new or changed, store its vector and hash
        vector = self.vectorize_text(content)
        self.file_vectors[file_path] = vector
        self.file_hashes[file_path] = current_hash
        
        # Return full content for first time or when changed
        return {
            'type': 'full',
            'content': content,
            'file_path': file_path
        }
    
    def generate_file_summary(self, content, max_length=100):
        """Generate a brief summary of the file content"""
        # Extract key elements like function names, class definitions
        # This helps maintain context while using much fewer tokens
        summary_lines = []
        for line in content.split('\n'):
            if re.match(r'^(def |class |import )', line.strip()):
                summary_lines.append(line.strip())
        return '\n'.join(summary_lines[:max_length])


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def on_modified(self, event):
        if not event.is_directory:
            self.vector_store.update_file(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self.vector_store.update_file(event.src_path)


class FileWatcher:
    def __init__(self, vector_store, directory_to_watch):
        self.vector_store = vector_store
        self.directory_to_watch = directory_to_watch
        self.event_handler = FileChangeHandler(vector_store)
        self.observer = Observer()

    def start(self):
        self.observer.schedule(self.event_handler, self.directory_to_watch, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()


class FileProcessor:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.file_timestamps = {}  # Store file paths and their last modified timestamps
        
    def update_file(self, file_path):
        """Process a file that has been created or modified."""
        logging.debug(f"Updating file: {file_path}")
        try:
            if not os.path.exists(file_path):
                return False
                
            # Get file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            # Remove old chunks for this file from the vector store
            self._remove_file_chunks(file_path)
            
            # Chunk the file and add to vector store
            chunks = self.vector_store.chunk_code_file(file_path, content)
            for i, chunk in enumerate(chunks):
                doc_id = f"{file_path}_{i}"
                self.vector_store.add_document(
                    doc_id=doc_id,
                    content=chunk['chunk_content'],
                    metadata={
                        "type": "code",
                        "file": file_path,
                        "start_line": chunk['start_line'],
                        "end_line": chunk['end_line']
                    }
                )
            
            # Update timestamp
            self.file_timestamps[file_path] = os.path.getmtime(file_path)
            return True
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return False
    
    def _remove_file_chunks(self, file_path):
        """Remove all chunks related to a specific file from the vector store."""
        # Mark existing chunks as deleted
        for doc in self.vector_store.documents:
            if doc.get('metadata', {}).get('file') == file_path:
                # Mark as deleted
                doc['metadata']['deleted'] = True
        
        self.vector_store.save_store()
    
    def scan_directory(self, directory_path, file_extensions=None):
        """Scan a directory for files that have changed."""
        if file_extensions is None:
            file_extensions = ['.py', '.js', '.ts', '.html', '.css', '.json']
            
        changed_files = []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    
                    try:
                        last_modified = os.path.getmtime(file_path)
                        
                        # Check if file is new or modified
                        if (file_path not in self.file_timestamps or 
                            last_modified > self.file_timestamps.get(file_path, 0)):
                            changed_files.append(file_path)
                    except Exception as e:
                        print(f"Error checking file {file_path}: {e}")
        
        # Process changed files
        processed_files = []
        for file_path in changed_files:
            if self.update_file(file_path):
                processed_files.append(file_path)
            
        return processed_files


class AiderPromptOptimizer:
    def optimize_messages(self, original_messages, current_query, current_files_context):
        """Simple optimization method to reduce tokens.""" 
        # Just keep system messages and the current query
        system_messages = [m for m in original_messages if m.get('role') == 'system']
        user_messages = [m for m in original_messages if m.get('role') == 'user']
        
        optimized = system_messages.copy()
        if user_messages:
            optimized.append(user_messages[-1])
        return optimized
    
    def optimize_messages(self, original_messages, current_query, current_files_context):
        """Simple optimization method to reduce tokens."""
        # Just keep system messages and the current query
        system_messages = [m for m in original_messages if m.get('role') == 'system']
        user_messages = [m for m in original_messages if m.get('role') == 'user']
        
        # Create optimized message list
        optimized = system_messages.copy()
        
        # Always add the current query (last user message)
        if user_messages:
            optimized.append(user_messages[-1])
        
        return optimized

    def optimize_messages(self, original_messages, current_query, current_files_context):
        """Simple optimization method to reduce tokens."""
        # Just keep system messages and the current query
        system_messages = [m for m in original_messages if m.get('role') == 'system']
        user_messages = [m for m in original_messages if m.get('role') == 'user']
        
        # Create optimized message list
        optimized = system_messages.copy()
        
        # Always add the current query (last user message)
        if user_messages:
            optimized.append(user_messages[-1])
        
        return optimized

    def optimize_messages(self, original_messages, current_query, current_files_context):
        """Simple optimization method to reduce tokens."""
        # Just keep system messages and the current query
        system_messages = [m for m in original_messages if m.get('role') == 'system']
        user_messages = [m for m in original_messages if m.get('role') == 'user']
        
        # Create optimized message list
        optimized = system_messages.copy()
        
        # Always add the current query (last user message)
        if user_messages:
            optimized.append(user_messages[-1])
        
        return optimized

    def __init__(self, vector_store, recent_memory_limit=3, retrieval_top_k=5, relevance_threshold=0.7, enable_logging=True):
        self.vector_store = vector_store
        self.recent_memory_limit = recent_memory_limit
        self.retrieval_top_k = retrieval_top_k
        self.relevance_threshold = relevance_threshold
        self.recent_interactions = []  # Maintains a small working memory of interactions
        
        # Initialize logger
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = RagLogger()

    def update_recent_interactions(self, interaction):
        # Append a new interaction and retain only the most recent ones
        self.recent_interactions.append(interaction)
        if len(self.recent_interactions) > self.recent_memory_limit:
            self.recent_interactions = self.recent_interactions[-self.recent_memory_limit:]
    
    def get_total_tokens_used(self):
        """Calculate the total number of tokens used in recent interactions."""
        return sum(len(interaction.split()) for interaction in self.recent_interactions)

def optimize_messages(self, original_messages, current_query, current_files_context):
    """Optimize the messages by REPLACING context with semantically relevant retrieved context."""
    try:
        # Count original tokens 
        orig_token_count = sum(len(m.get("content", "").split()) for m in original_messages)
        
        # 1. Identify different message types
        system_instruction_messages = []
        user_messages = []
        assistant_messages = []
        files_context_message = None
        
        # Categorize messages
        for m in original_messages:
            role = m.get('role', '')
            content = m.get('content', '')
            
            # Keep essential system instructions (likely at the beginning)
            if role == 'system' and not files_context_message:
                if "Current Files:" in content:
                    files_context_message = m
                else:
                    system_instruction_messages.append(m)
            elif role == 'user':
                user_messages.append(m)
            elif role == 'assistant':
                assistant_messages.append(m)
        
        # 2. Keep only most recent exchanges to preserve conversation flow
        # At most keep the last 1-2 exchanges before the current query
        latest_exchanges = []
        if len(user_messages) > 1:  # If there are previous user messages
            # Handle the case where we have alternating user/assistant messages
            recent_exchanges_count = min(self.recent_memory_limit, len(user_messages) - 1)
            
            for i in range(recent_exchanges_count):
                user_idx = len(user_messages) - 2 - i  # Second-to-last user message, going backward
                if user_idx >= 0:
                    latest_exchanges.append(user_messages[user_idx])
                    # Find corresponding assistant message
                    assistant_idx = min(user_idx, len(assistant_messages) - 1)
                    if assistant_idx >= 0:
                        latest_exchanges.append(assistant_messages[assistant_idx])
            
            # Reverse to maintain chronological order
            latest_exchanges.reverse()
        
        # 3. Always include the most recent user message (current query)
        latest_exchanges.append(user_messages[-1])
        
        # 4. Retrieve relevant context
        retrieved_contexts = self.vector_store.search(current_query, top_k=self.retrieval_top_k)
        
        # 5. Filter by relevance threshold - be more selective!
        relevant_contexts = [
            ctx for ctx in retrieved_contexts 
            if ctx.get('distance', 1.0) < self.relevance_threshold
        ]
        
        # 6. Initialize optimized message sequence with essential components
        optimized_messages = system_instruction_messages.copy()
        
        # 7. Add context from vector store AS A REPLACEMENT for old context
        # Only if we have relevant contexts and they're better than what we have
        if relevant_contexts:
            # Build a single system message with retrieved context instead of multiple history messages
            context_lines = ["Relevant context from previous interactions:"]
            
            # Limit total context to avoid making things worse
            total_context_size = 0
            max_context_size = 1500  # Adjust based on your typical message sizes
            
            for ctx in relevant_contexts:
                ctx_text = ctx.get('content', '')
                # Truncate each context if needed
                if len(ctx_text) > 300:
                    ctx_text = ctx_text[:300] + "..."
                
                # Check if adding this would exceed our budget
                if total_context_size + len(ctx_text) > max_context_size:
                    break
                    
                ctx_type = ctx.get('metadata', {}).get('type', 'context')
                ctx_file = ctx.get('metadata', {}).get('file', 'unknown')
                
                context_lines.append(f"--- {ctx_type} from {ctx_file} ---")
                context_lines.append(ctx_text)
                context_lines.append("")  # Blank line for spacing
                
                total_context_size += len(ctx_text)
            
            # Only add this if we have meaningful context
            if len(context_lines) > 1:
                context_content = "\n".join(context_lines)
                optimized_messages.append({
                    'role': 'system',
                    'content': context_content
                })
        
        # 8. Add recent exchanges (limited history)
        optimized_messages.extend(latest_exchanges)
        
        # 9. Add current files context if available (needed for coding context)
        if current_files_context and len(current_files_context) > 0:
            # If current files context is very large, consider truncating it
            if len(current_files_context) > 2000:
                # Extract the file names and only include the most recent or relevant ones
                import re
                files = re.findall(r'```\w+\s+([\w\d./_-]+)', current_files_context)
                if files:
                    # Create a simpler context that just lists the files
                    simplified_context = "Current Files Context (abbreviated):\n" + "\n".join(files[:5])
                    optimized_messages.append({
                        'role': 'system',
                        'content': simplified_context
                    })
            else:
                optimized_messages.append({
                    'role': 'system',
                    'content': current_files_context
                })
        
        # Count optimized tokens 
        opt_token_count = sum(len(m.get("content", "").split()) for m in optimized_messages)
        
        # Only return optimized messages if they're actually smaller
        if opt_token_count < orig_token_count:
            return optimized_messages
        else:
            # If optimization didn't reduce tokens, only keep essential messages
            essential_messages = system_instruction_messages.copy()
            
            # Keep only the current query
            essential_messages.append(user_messages[-1])
            
            # Add abbreviated context if available
            if relevant_contexts:
                brief_context = "Relevant context (abbreviated): " + relevant_contexts[0].get('content', '')[:300]
                essential_messages.append({
                    'role': 'system',
                    'content': brief_context
                })
                
            return essential_messages
            
    except Exception as e:
        print(f"Error optimizing messages: {e}")
        return original_messages  # Fall back to original messages on error

# For compatibility with the wrapper script
VectorStore = SimpleVectorStore

class MessageOptimizer:
    def preprocess_message(self, message):
        # 1. Break into chunks
        chunks = self.smart_chunk_content(message)
        
        # 2. Pre-vectorize each chunk
        vectors = [self.vectorize_text(chunk) for chunk in chunks]
        
        # 3. Store vectors for future use
        return {
            'original': message,
            'chunks': chunks,
            'vectors': vectors
        }
