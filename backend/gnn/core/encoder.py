
"""宏观+GNN 专用文本编码器。同时支持 BERT 和哈希回退。"""

import hashlib, math, re
from functools import lru_cache

_CJK = re.compile("[\u3400-\u9fff]+")
_LATIN = re.compile(r"[a-z0-9_]+")

def _tokenize(text):
    txt = text.casefold().replace("_", " ")
    tokens = _LATIN.findall(txt)
    for block in _CJK.findall(txt):
        tokens.extend(block)
        tokens.extend(block[i:i+2] for i in range(len(block)-1))
    return tokens


class BertTextEncoder:
    """
    文本编码器，宏观+GNN 专用。

    策略:
      - transformers 已安装 + bert-base-chinese 已缓存 → 用真实 BERT (768维)
      - 否则 → 用哈希编码回退 (128维)

    使用:
        encoder = BertTextEncoder()
        vec = encoder.encode("阿司匹林")
        dim = encoder.dim
    """

    def __init__(self, model_name: str = "bert-base-chinese"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._try_bert()

    def _try_bert(self):
        try:
            import huggingface_hub
            cache = huggingface_hub.constants.HUGGINGFACE_HUB_CACHE
            model_id = "models--" + self.model_name.replace("/", "--")
            import os
            if os.path.isdir(os.path.join(cache, model_id)):
                from transformers import AutoTokenizer, AutoModel
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()
        except Exception:
            self._model = None

    @property
    def dim(self) -> int:
        if self._model is not None:
            return 768
        return 128

    def encode(self, text: str) -> list[float]:
        if self._model is not None:
            return self._encode_bert(text)
        return self._encode_hash(text)

    def _encode_bert(self, text: str) -> list[float]:
        import torch
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
        with torch.no_grad():
            out = self._model(**inputs)
        return out.last_hidden_state[:, 0, :].squeeze().tolist()

    @lru_cache(maxsize=2048)
    def _encode_hash(self, text: str) -> list[float]:
        vec = [0.0] * 128
        for token in _tokenize(text):
            d = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            v = int.from_bytes(d, "little")
            idx = v % 128
            vec[idx] += 1.0 if (v >> 8) & 1 else -1.0
        norm = math.sqrt(sum(x*x for x in vec))
        return [x/norm for x in vec] if norm else vec
