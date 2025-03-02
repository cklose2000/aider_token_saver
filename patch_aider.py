#!/usr/bin/env python3

import os
import sys
import logging
import time
import tempfile
import datetime

# Ensure the log directory exists
log_dir = os.path.join(os.path.expanduser("~"), ".aider", "rag_logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(
    filename=os.path.join(log_dir, "patch_debug.log"),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Then in your patch functions for OpenAI and Anthropic, add:
def patched_create(*args, **kwargs):
    logging.debug("OpenAI ChatCompletion.create was called")
    # [existing code]
    if 'messages' in kwargs:
        # Get original token count
        messages = kwargs['messages']
        orig_token_count = sum(len(m.get("content", "").split()) for m in messages)
        logging.debug(f"Original token count: {orig_token_count}")
        # [existing code]
        # Get optimized token count
        opt_token_count = sum(len(m.get("content", "").split()) for m in messages)  # Assuming messages are optimized here
        logging.debug(f"Optimized token count: {opt_token_count}")
    # [rest of the function]
    if 'messages' in kwargs:
        logging.debug(f"Original token count: {orig_token_count}")
        # [existing code]
        logging.debug(f"Optimized token count: {opt_token_count}")
    # [rest of the function]

def patch_aider():
    """Apply monkey patches to aider to add RAG functionality."""
    try:
        # Try to import aider's modules
        import aider
        print(f"Found Aider at: {aider.__file__}")
        
        # Try to import our vector store module from the current directory
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from vector_store import SimpleVectorStore, AiderPromptOptimizer
        
        # Also import our logger
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
        
        # Initialize the optimizer
        optimizer = AiderPromptOptimizer(
            vector_store,
            enable_logging=logger is not None
        )
        
        # Start a new session if logger exists
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
        
        # Patch OpenAI
        try:
            import openai
            original_create = openai.ChatCompletion.create
            
            def patched_create(*args, **kwargs):
                if 'messages' in kwargs:
                    # Get original token count
                    messages = kwargs['messages']
                    orig_token_count = sum(len(m.get("content", "").split()) for m in messages)
                    
                    # Extract the current query from the last user message
                    user_messages = [m for m in messages if m.get("role") == "user"]
                    if user_messages:
                        current_query = user_messages[-1].get("content", "")
                        # Optimize messages
                        optimized_messages = optimizer.optimize_messages(messages, current_query, "")
                        # Get optimized token count
                        opt_token_count = sum(len(m.get("content", "").split()) for m in optimized_messages)
                        # Print token reduction
                        if orig_token_count > 0 and orig_token_count != opt_token_count:
                            reduction_percent = int((1 - opt_token_count/orig_token_count) * 100)
                            print(f"Token reduction: {orig_token_count} → {opt_token_count} (approx. {reduction_percent}% saved)")
                        kwargs['messages'] = optimized_messages
                
                return original_create(*args, **kwargs)
            
            # Apply the patch
            openai.ChatCompletion.create = patched_create
            print("RAG enhancement active - OpenAI API calls will be optimized")
            
        except Exception as e:
            print(f"Failed to patch OpenAI: {e}")
        
        # Patch Anthropic
        try:
            import anthropic
            original_messages = anthropic.Anthropic.messages
            
            def patched_messages(self, *args, **kwargs):
                if 'messages' in kwargs:
                    # Get original token count
                    messages = kwargs['messages']
                    orig_token_count = sum(len(m.get("content", "").split()) for m in messages)
                    
                    # Extract the current query from the last user message
                    user_messages = [m for m in messages if m.get("role") == "user"]
                    if user_messages:
                        current_query = user_messages[-1].get("content", "")
                        # Optimize messages
                        optimized_messages = optimizer.optimize_messages(messages, current_query, "")
                        # Get optimized token count
                        opt_token_count = sum(len(m.get("content", "").split()) for m in optimized_messages)
                        # Print token reduction
                        if orig_token_count > 0 and orig_token_count != opt_token_count:
                            reduction_percent = int((1 - opt_token_count/orig_token_count) * 100)
                            print(f"Token reduction: {orig_token_count} → {opt_token_count} (approx. {reduction_percent}% saved)")
                        kwargs['messages'] = optimized_messages
                
                return original_messages(self, *args, **kwargs)
            
            # Apply the patch
            anthropic.Anthropic.messages = patched_messages
            print("RAG enhancement active - Anthropic API calls will be optimized")
            
        except Exception as e:
            print(f"Failed to patch Anthropic: {e}")
        
        # Register shutdown handler to end the session and save stats
        if logger:
            import atexit
            def end_session():
                logger.end_session()
                print("RAG session ended, logs saved")
            atexit.register(end_session)
        
        # Flag that we've applied the patches
        aider.RAG_PATCHED = True
        
        return True
        
    except Exception as e:
        print(f"Failed to patch Aider: {e}")
        return False

# Apply the patches when this module is imported
patch_applied = patch_aider()
print(f"RAG patch {'applied successfully' if patch_applied else 'failed'}")
