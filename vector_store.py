import os
import json
import time
import re
import math
from collections import Counter
import numpy as np

class SimpleVectorStore:
    """A simple vector store that uses TF-IDF and cosine similarity without external dependencies."""
    
    def __init__(self, store_path="simple_vector_store.json"):
        self.store_path = store_path
        self.documents = []
        self.word_to_idx = {}  # Maps words to their indices in the vector
        self.idx_to_word = {}  # Maps indices to words
        self.idf = {}  # Inverse document frequency
        self.load_store()
    
    def load_store(self):
        """Load the vector store from disk if it exists."""
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
    
    def search(self, query, top_k=5, doc_type=None):
        """Search for similar documents."""
        if not self.documents:
            return []
        
        # Compute query vector
        query_vector = self.compute_tfidf_vector(query)
        
        # Compute similarity with all documents
        results = []
        for doc in self.documents:
            # Skip deleted documents
            if doc.get('metadata', {}).get('deleted', False):
                continue
            
            # Apply document type filter if specified
            if doc_type and doc.get('metadata', {}).get('type') != doc_type:
                continue
            
            # Compute document vector
            doc_vector = self.compute_tfidf_vector(doc['content'])
            
            # Compute similarity
            similarity = self.cosine_similarity(query_vector, doc_vector)
            
            # Add to results
            result = dict(doc)
            result['distance'] = 1.0 - similarity  # Convert to distance (lower is better)
            results.append(result)
        
        # Sort by similarity (lowest distance first)
        results.sort(key=lambda x: x['distance'])
        
        # Return top_k results
        return results[:top_k]
    
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


class FileWatcher:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.file_timestamps = {}  # Store file paths and their last modified timestamps
        
    def update_file(self, file_path):
        """Process a file that has been created or modified."""
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
    def __init__(self, vector_store, recent_memory_limit=3, retrieval_top_k=5, relevance_threshold=0.8):
        self.vector_store = vector_store
        self.recent_memory_limit = recent_memory_limit
        self.retrieval_top_k = retrieval_top_k
        self.relevance_threshold = relevance_threshold
        self.recent_interactions = []  # Maintains a small working memory of interactions

    def update_recent_interactions(self, interaction):
        # Append a new interaction and retain only the most recent ones
        self.recent_interactions.append(interaction)
        if len(self.recent_interactions) > self.recent_memory_limit:
            self.recent_interactions = self.recent_interactions[-self.recent_memory_limit:]

    def optimize_messages(self, original_messages, current_query, current_files_context):
        """Optimize the messages by replacing context with semantically relevant retrieved context."""
        try:
            # Preserve existing system messages and recent exchanges (user/assistant)
            system_messages = [m for m in original_messages if m.get('role') == 'system']
            
            # Keep a minimum of recent exchanges
            recent_exchanges = []
            user_assistant_pairs = []
            
            # Group user/assistant messages into pairs
            for i in range(len(original_messages) - 1):
                if (original_messages[i].get('role') == 'user' and 
                    original_messages[i+1].get('role') == 'assistant'):
                    user_assistant_pairs.append((original_messages[i], original_messages[i+1]))
            
            # Keep the most recent pairs
            if user_assistant_pairs:
                recent_exchanges = [msg for pair in user_assistant_pairs[-self.recent_memory_limit:] for msg in pair]
            
            # Always include the most recent user message if it exists
            if original_messages and original_messages[-1].get('role') == 'user':
                if not recent_exchanges or recent_exchanges[-1] != original_messages[-1]:
                    recent_exchanges.append(original_messages[-1])
            
            # Retrieve relevant context from the vector store
            retrieved_contexts = self.vector_store.search(current_query, top_k=self.retrieval_top_k)
            relevant_contexts = [
                ctx for ctx in retrieved_contexts 
                if ctx.get('distance', 1.0) < self.relevance_threshold
            ]
            
            # Start with preserved system messages and recent interactions
            optimized_messages = system_messages + recent_exchanges
            
            # Append current file context as a system message, if available
            if current_files_context:
                optimized_messages.append({
                    'role': 'system',
                    'content': f"Current Files Context:\n{current_files_context}"
                })
            
            # Append retrieved relevant context as a separate system message
            if relevant_contexts:
                context_lines = ["Relevant context from previous interactions:"]
                for ctx in relevant_contexts:
                    ctx_type = ctx.get('metadata', {}).get('type', 'context')
                    ctx_file = ctx.get('metadata', {}).get('file', 'unknown')
                    context_lines.append(f"--- {ctx_type} from {ctx_file} ---")
                    context_lines.append(ctx.get('content', ''))
                    context_lines.append("")  # Blank line for spacing
                context_content = "\n".join(context_lines)
                optimized_messages.append({
                    'role': 'system',
                    'content': context_content
                })
            
            return optimized_messages
        except Exception as e:
            print(f"Error optimizing messages: {e}")
            return original_messages  # Fall back to original messages on error

# For compatibility with the wrapper script
VectorStore = SimpleVectorStore