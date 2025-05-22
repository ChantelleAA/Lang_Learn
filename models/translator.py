from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import TRANSLATOR_MODEL, DEFAULT_SOURCE_LANG

class ObjectTranslator:
    def __init__(self, target_lang):
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_MODEL)
        self.tokenizer.src_lang = DEFAULT_SOURCE_LANG
        self.target_lang = target_lang

    def translate(self, texts):
        outputs = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt")
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
            generated = self.model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.append(decoded[0])
        return outputs
