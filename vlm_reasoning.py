# vlm_reasoning.py

import os
import google.generativeai as genai

class GeminiReasoner:
    def __init__(self, api_key=None, model_name="gemini-2.0-flash-lite"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            return
            raise ValueError("Missing Gemini API key. Pass it or set GEMINI_API_KEY env var.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"🔮 Gemini model initialized: {model_name}")

    def ask(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error: {str(e)}"

# 🔁 Test
if __name__ == "__main__":
    reasoner = GeminiReasoner(api_key="GEMINI_API_KEY")
    result = reasoner.ask("A person is walking in the middle of a highway. Is this safe?")
    print("Gemini Flash says:", result)
