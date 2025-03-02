from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Store the original method
original_create = client.chat.completions.create

def patched_create(*args, **kwargs):
    """Intercepts OpenAI API calls, optimizes context, and logs usage."""
    print("\n🔥 [DEBUG] Intercepting API call!")  # ✅ Force debug log to appear

    if "messages" in kwargs:
        messages = kwargs["messages"]
        orig_token_count = sum(len(m.get("content", "").split()) for m in messages)
        print(f"📜 [DEBUG] Original messages: {messages}")
        print(f"🔢 [DEBUG] Original token count: {orig_token_count}")

        # Extract last user query
        user_messages = [m for m in messages if m.get("role") == "user"]  
        if user_messages:
            current_query = user_messages[-1].get("content", "")
            print(f"📝 [DEBUG] User query: {current_query[:60]}{'...' if len(current_query) > 60 else ''}")

        # Optimize messages (Dummy for now, replace with real function)
        optimized_messages = messages  # Placeholder

        kwargs["messages"] = optimized_messages  # Inject optimized messages

    response = original_create(*args, **kwargs)

    # 🔥 Force debug logs before returning response
    print(f"\n✅ [DEBUG] API Response received!")
    print(f"📡 [DEBUG] Model Used: {response.model}")
    print(f"💬 [DEBUG] Assistant Response: {response.choices[0].message.content}")
    print(f"📊 [DEBUG] Token Usage: {response.usage.total_tokens} total")
    print(f"🔄 [DEBUG] Returning modified API response...")

    return response

# Apply the patch
client.chat.completions.create = patched_create
print("\n🛠 [INFO] Successfully patched OpenAI API calls at Client level!\n")

# ✅ Ensure test.py imports this patched client
def get_patched_client():
    return client
