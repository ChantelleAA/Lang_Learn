import os
from openai import OpenAI
import language_tool_python
import textstat
from dotenv import load_dotenv

load_dotenv()
class SentenceGenerator:
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.tool = language_tool_python.LanguageTool("en-US")
    
    def create_prompt(self, obj, level, structure, culture="global"):
        return f"Create a {level} level {structure} sentence using '{obj}' appropriate for a {culture} audience."

    def validate(self, sentence):
        return len(self.tool.check(sentence)) == 0

    def score(self, sentence):
        return textstat.flesch_reading_ease(sentence)
    
    def close(self):
        self.tool.close()

    def describe_object(self, obj, level="beginner", structure="declarative", culture="global"):
        prompt = (
            f"Give a simple explanation, beginner-friendly synonyms, and a {structure} sentence using the word '{obj}'.\n"
            f"Use this format:\n"
            f"- Meaning: <short explanation>\n"
            f"- Synonyms: <comma-separated synonyms>\n"
            f"- Sentence: <example sentence appropriate for a {culture} audience>"
        )
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        response_ = self.parse_response(res.choices[0].message.content.strip())
        print(f"Response from OpenAI: {response_}")
        return response_

    def parse_response(self, response_text):
        # Robust parsing of formatted response
        meaning, synonyms, sentence = "", [], ""
        lines = response_text.split('\n')
        
        for line in lines:
            if line.lower().startswith("- meaning:"):
                meaning = line.split(":", 1)[1].strip()
                print(f"Meaning: {meaning}")
            elif line.lower().startswith("- synonyms:"):
                synonyms = [syn.strip() for syn in line.split(":", 1)[1].split(",")]
                print(f"Synonyms: {synonyms}")
            elif line.lower().startswith("- sentence:"):
                sentence = line.split(":", 1)[1].strip()
                print(f"Sentence: {sentence}")
        return {
            "chatgpt_meaning": meaning,
            "chatgpt_synonyms": synonyms,
            "chatgpt_sentence": sentence
        }
