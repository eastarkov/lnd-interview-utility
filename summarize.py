"""Local abstractive summary. All inference stays on the user's computer."""
from pathlib import Path
import numpy as np
from semantic import summary_indices

MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'

class Summarizer:
    def __init__(self, cache):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(4)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=str(Path(cache)), local_files_only=True)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=str(Path(cache)))
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        dtype = torch.float16 if self.device == 'mps' else torch.float32
        from huggingface_hub import hf_hub_download
        try:
            hf_hub_download(MODEL, 'model.safetensors', cache_dir=str(Path(cache)), local_files_only=True)
        except OSError:
            hf_hub_download(MODEL, 'model.safetensors', cache_dir=str(Path(cache)))
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, cache_dir=str(Path(cache)), torch_dtype=dtype,
            use_safetensors=True, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def summarize(self, passages, vectors):
        import torch
        if not passages:
            return 'Речь не найдена.'
        all_text = '\n'.join(p['text'] for p in passages)
        sampled = len(all_text) > 9000
        if sampled:
            indices = set(summary_indices(vectors, 8))
            indices.update(int(x) for x in np.linspace(0, len(passages)-1, 8))
            all_text = '\n'.join(passages[i]['text'] for i in sorted(indices))[:9000]
        messages = [
            {'role': 'system', 'content': 'Ты редактор. Сохраняй факты исходного текста.'},
            {'role': 'user', 'content': 'Краткое содержание этого видео:\n' + all_text + '\nО чём говорит человек?'},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=5000).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=220, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        blocks = answer.split('\n\n')
        if len(blocks) > 1 and len(blocks[0]) < 150 and 'содержани' in blocks[0].lower() and blocks[0].rstrip().endswith(':'):
            answer = '\n\n'.join(blocks[1:])
        return answer or 'Не удалось сформировать содержание.'
