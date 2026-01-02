import os
import google.generativeai as genai

# Read API key from PowerShell environment
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not set.\n"
        "Run in PowerShell:\n"
        "$env:GEMINI_API_KEY=\"your_api_key_here\""
    )

genai.configure(api_key=api_key)

print("\nAvailable Gemini Models:\n")

for model in genai.list_models():
    print(f"Model name: {model.name}")
    print(f"Supported methods: {model.supported_generation_methods}")
    print("-" * 60)
