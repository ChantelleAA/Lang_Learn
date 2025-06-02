import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import TRANSLATOR_MODEL, DEFAULT_SOURCE_LANG

class ObjectTranslator:
    def __init__(self, target_lang='aka_Latn'):
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_MODEL)
        self.tokenizer.src_lang = DEFAULT_SOURCE_LANG
        self.target_lang = target_lang
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def set_language(self, new_target_lang):
        self.target_lang = new_target_lang

    def translate(self, texts):
        if not texts:
            return []

        # Tokenize all texts at once (batching)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256  # or 512, depending on model
        ).to(self.device)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
        if forced_bos_token_id is None:
            raise ValueError(f"Invalid target language token: {self.target_lang}")

        # Generate translations in batch
        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

        # Decode all at once
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        return decoded
