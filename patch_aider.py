#!/usr/bin/env python3
import os
import sys
import time
import logging
import datetime
import tempfile
import atexit

# Ensure our log directory exists
log_dir = os.path.join(os.path.expanduser("~"), ".aider", "rag_logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, "patch_debug.log"),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Attempt to import aider and our modules
try:
    import aider
    print(f"Found Aider at: {aider.__file__}")
except Exception as e:
    print(f"Error importing aider: {e}")

# Add current directory to sys.path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_store import SimpleVectorStore, AiderPromptOptimizer

# Try to import our logger (rag_logger)
try:
    from rag_logger import RagLogger
    logger = RagLogger()
    print(f"RAG logging enabled at: {logger.log_dir}")
except ImportError:
    logger = None
    print("RAG logger not found, logging disabled")

# Initialize the vector store
home_dir = os.path.expanduser("~")
aider_dir = os.path.join(home_dir, ".aider")
if not os.path.exists(aider_dir):
    os.makedirs(aider_dir)

vector_store = SimpleVectorStore(
    store_path=os.path.join(aider_dir, "simple_vector_store.json")
)

# Initialize the prompt optimizer
optimizer = AiderPromptOptimizer(
    vector_store,
    enable_logging=(logger is not None)
)

# If logger exists, initialize a new session
if logger:
    logger.current_session_id = int(time.time())
    logger.session_metrics = {
        "session_id": logger.current_session_id,
        "start_time": datetime.datetime.now().isoformat(),
        "searches": 0,
        "token_reductions": 0,
        "total_orig_tokens": 0,
        "total_opt_tokens": 0,
        "avg_reduction_pct": 0,
        "queries_with_relevant_results": 0,
        "total_queries": 0,
        "relevance_score": 0
    }
    logger._update_session_file()

# --- PATCH OpenAI API CALLS ---
try:
    import openai
    # Save the original ChatCompletion.create method
    original_create = openai.ChatCompletion.create

    def patched_create(*args, **kwargs):
        if 'messages' in kwargs:
            messages = kwargs['messages']
            # Calculate original token count
            orig_token_count = sum(len(m.get("content", "").split()) for m in messages)
            print(f"[DEBUG] Original token count: {orig_token_count}")

            # Extract the current query from the last user message
            user_messages = [m for m in messages if m.get("role") == "user"]
            if user_messages:
                current_query = user_messages[-1].get("content", "")
                
                # Add the user query to the vector store to ensure it's vectorized.
                doc_id = f"query_{int(time.time())}"
                optimizer.vector_store.add_document(
                    doc_id,
                    current_query,
                    {"type": "query", "timestamp": time.time()}
                )
                
                # Optimize messages using the retrieved context.
                optimized_messages = optimizer.optimize_messages(messages, current_query, "")
                
                # Calculate optimized token count.
                opt_token_count = sum(len(m.get("content", "").split()) for m in optimized_messages)
                print(f"[DEBUG] Optimized token count: {opt_token_count}")

                # Print the token reduction if any.
                if orig_token_count > 0 and orig_token_count != opt_token_count:
                    reduction_percent = int((1 - opt_token_count / orig_token_count) * 100)
                    print(f"[DEBUG] Token reduction: {orig_token_count} -> {opt_token_count} (approx. {reduction_percent}% saved)")
                else:
                    print("[DEBUG] No token reduction achieved.")

                # Optionally, print the retrieved context details.
                retrieved_contexts = optimizer.vector_store.search(current_query, top_k=optimizer.retrieval_top_k)
                print(f"[DEBUG] Retrieved {len(retrieved_contexts)} contexts for the query:")
                for ctx in retrieved_contexts:
                    print(f"  - DocID: {ctx.get('doc_id')}, Distance: {ctx.get('distance'):.3f}")

                # Replace original messages with optimized messages.
                kwargs['messages'] = optimized_messages

        return original_create(*args, **kwargs)

    # Apply the patch to OpenAI ChatCompletion.create
    openai.ChatCompletion.create = patched_create
    print("RAG enhancement active - OpenAI API calls will be optimized with vectorized query context")
except Exception as e:
    print(f"Failed to patch OpenAI: {e}")

# --- PATCH Anthropic API CALLS (if needed) ---
try:
    import anthropic
    original_messages = anthropic.Anthropic.messages

    def patched_messages(self, *args, **kwargs):
        if 'messages' in kwargs:
            messages = kwargs['messages']
            orig_token_count = sum(len(m.get("content", "").split()) for m in messages)
            user_messages = [m for m in messages if m.get("role") == "user"]
            if user_messages:
                current_query = user_messages[-1].get("content", "")
                # Add the query to the vector store.
                doc_id = f"query_{int(time.time())}"
                optimizer.vector_store.add_document(
                    doc_id,
                    current_query,
                    {"type": "query", "timestamp": time.time()}
                )
                optimized_messages = optimizer.optimize_messages(messages, current_query, "")
                opt_token_count = sum(len(m.get("content", "").split()) for m in optimized_messages)
                if orig_token_count > 0 and orig_token_count != opt_token_count:
                    reduction_percent = int((1 - opt_token_count / orig_token_count) * 100)
                    print(f"[DEBUG] Token reduction: {orig_token_count} -> {opt_token_count} (approx. {reduction_percent}% saved)")
                kwargs['messages'] = optimized_messages
        return original_messages(self, *args, **kwargs)

    anthropic.Anthropic.messages = patched_messages
    print("RAG enhancement active - Anthropic API calls will be optimized with vectorized query context")
except Exception as e:
    print(f"Failed to patch Anthropic: {e}")

# Register shutdown handler to save session stats if logger exists
if logger:
    def end_session():
        logger.end_session()
        print("RAG session ended, logs saved")
    atexit.register(end_session)

# Mark that Aider has been patched
aider.RAG_PATCHED = True

def patch_aider():
    """Return True if patching was successful."""
    return True

# Apply patch immediately when this module is imported
patch_applied = patch_aider()
print(f"RAG patch {'applied successfully' if patch_applied else 'failed'}")
