"""Local multilingual embeddings, with no remote inference or PyTorch dependency."""
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import onnxruntime as ort

REPO = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2'

class SemanticIndex:
    def __init__(self, cache):
        cache = str(Path(cache))
        def cached_file(name):
            try:
                return hf_hub_download(REPO, name, cache_dir=cache, local_files_only=True)
            except OSError:
                return hf_hub_download(REPO, name, cache_dir=cache)
        tokenizer_path = cached_file('tokenizer.json')
        model_path = cached_file('onnx/model_quantized.onnx')
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])

    def encode(self, texts):
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        batches = []
        for offset in range(0, len(texts), 16):
            encodings = self.tokenizer.encode_batch(texts[offset:offset+16])
            values = {
                'input_ids': np.array([e.ids for e in encodings], dtype=np.int64),
                'attention_mask': np.array([e.attention_mask for e in encodings], dtype=np.int64),
                'token_type_ids': np.array([e.type_ids for e in encodings], dtype=np.int64),
            }
            output = self.session.run(None, {i.name: values[i.name] for i in self.session.get_inputs()})[0]
            if output.ndim == 3:
                mask = values['attention_mask'][..., None]
                output = (output * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
            output = output / np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-9)
            batches.append(output)
        return np.concatenate(batches)


def passages(segments, max_seconds=25):
    result, current = [], []
    for segment in segments:
        if current and segment['end'] - current[0]['start'] > max_seconds:
            result.append({'start': current[0]['start'], 'end': current[-1]['end'],
                           'text': ' '.join(s['text'].strip() for s in current)})
            current = []
        current.append(segment)
    if current:
        result.append({'start': current[0]['start'], 'end': current[-1]['end'],
                       'text': ' '.join(s['text'].strip() for s in current)})
    return result


def summary_indices(vectors, limit=3):
    """Central, diverse excerpts: no invented claims or generative model."""
    if len(vectors) == 0:
        return []
    center = vectors.mean(axis=0)
    relevance = vectors @ center
    chosen = []
    for _ in range(min(limit, len(vectors))):
        diversity = np.max(vectors @ vectors[chosen].T, axis=1) if chosen else np.zeros(len(vectors))
        scores = 0.7 * relevance - 0.3 * diversity
        scores[chosen] = -np.inf
        chosen.append(int(np.argmax(scores)))
    return sorted(chosen)
