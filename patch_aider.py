#!/usr/bin/env python3

import os
import sys
import tempfile

# Create this file as patch_aider.py in the same directory as aider's installed location
# or in the same directory as your vector_store.py

def patch_aider():
    """Apply monkey patches to aider to add RAG functionality."""
    try:
        # Try to import aider's modules
        import aider
        print(f"Found Aider at: {aider.__file__}")
        
        # Try to import our vector store module from the current directory
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from vector_store import SimpleVectorStore, AiderPromptOptimizer
        
        # Initialize the vector store
        home_dir = os.path.expanduser("~")
        aider_dir = os.path.join(home_dir, ".aider")
        if not os.path.exists(aider_dir):
            os.makedirs(aider_dir)
        
        vector_store = SimpleVectorStore(
            store_path=os.path.join(aider_dir, "simple_vector_store.json")
        )
        
        # Initialize the optimizer
        optimizer = AiderPromptOptimizer(vector_store)
        
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
        
        # Flag that we've applied the patches
        aider.RAG_PATCHED = True
        
        return True
        
    except Exception as e:
        print(f"Failed to patch Aider: {e}")
        return False

# Apply the patches when this module is imported
patch_applied = patch_aider()
print(f"RAG patch {'applied successfully' if patch_applied else 'failed'}")