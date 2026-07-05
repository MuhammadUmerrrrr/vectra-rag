"""
ollama_client.py
=================
Rewritten to use Groq API for Chat (Generation) and HuggingFace for Embeddings.
Needs GROQ_API_KEY and HF_API_KEY environment variables.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class OllamaClient:
    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.hf_key = os.getenv("HF_API_KEY")
        
        # Initialize Groq if key exists
        if self.groq_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.groq_key)
            except ImportError:
                self.client = None
        else:
            self.client = None
            
        self.embed_model = "all-MiniLM-L6-v2 (HF)"
        self.gen_model = "llama3-8b-8192 (Groq)"

    def is_available(self) -> bool:
        # Consider available if we at least have Groq key
        return bool(self.groq_key)

    def embed(self, text: str) -> list[float] | None:
        headers = {"Authorization": f"Bearer {self.hf_key}"} if self.hf_key else {}
        try:
            # Using HuggingFace free inference API for embeddings
            model = "sentence-transformers/all-MiniLM-L6-v2"
            r = requests.post(
                f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}",
                headers=headers,
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=30,
            )
            if r.status_code == 200:
                res = r.json()
                if isinstance(res, list) and len(res) > 0:
                    # In some cases HF returns a nested list
                    return res[0] if isinstance(res[0], list) else res
            return None
        except Exception:
            return None

    def generate(self, prompt: str) -> str:
        if not self.client:
            return "ERROR: Groq is not configured. Please set GROQ_API_KEY."
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"ERROR: Groq API failed - {str(e)}"
