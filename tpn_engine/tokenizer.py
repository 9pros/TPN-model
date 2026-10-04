"""
Tokenizer integration for the TPN engine.

Wraps the HuggingFace ``tokenizers`` library (auto-detected) for real
checkpoint vocabularies, with a pure-Python char-level fallback so the
engine works without any extra install.

Also provides sampling helpers (top-k / top-p / temperature) so
``model.generate`` can turn logits into next tokens.
"""

import math
import os
import random
from typing import List, Optional, Tuple

try:
    from tokenizers import Tokenizer as HFTokenizer  # type: ignore
    _HAS_HF_TOKENIZERS = True
except ImportError:
    HFTokenizer = None  # type: ignore
    _HAS_HF_TOKENIZERS = False


class Tokenizer:
    """
    Tokenizer abstraction over a real checkpoint vocabulary.

    Args:
        hf_tokenizer: pre-loaded ``tokenizers.Tokenizer`` (from a
            ``tokenizer.json`` on disk), or ``None`` to use the
            char-level fallback.
    """

    def __init__(self, hf_tokenizer=None):
        self._hf = hf_tokenizer
        self._eos_token_id: Optional[int] = None
        self._pad_token_id: Optional[int] = None
        self._unk_token_id: Optional[int] = None
        if hf_tokenizer is not None:
            self._load_special_ids(hf_tokenizer)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str) -> "Tokenizer":
        """
        Load a tokenizer from a checkpoint directory that contains a
        ``tokenizer.json`` (HuggingFace format).

        Falls back to a char-level tokenizer if the file is missing or
        cannot be parsed, so ``from_checkpoint`` never raises.
        """
        path = os.path.join(checkpoint_dir, "tokenizer.json")
        if not _HAS_HF_TOKENIZERS or not os.path.isfile(path):
            return cls._char_level()

        try:
            hf = HFTokenizer.from_file(path)
            return cls(hf)
        except Exception:
            return cls._char_level()

    @classmethod
    def _char_level(cls) -> "Tokenizer":
        """Pure-Python char-level tokenizer (no dependencies)."""
        return cls(None)

    def _load_special_ids(self, hf) -> None:
        """Extract special-token ids from the loaded HF tokenizer."""
        if not hasattr(hf, "get_vocab"):
            return
        vocab = hf.get_vocab()
        for special in ("eos_token", "pad_token", "unk_token"):
            getfn = getattr(hf, f"get_{special}", None)
            if callable(getfn):
                tid = getfn()
                if tid is not None and tid in vocab:
                    if special == "eos_token":
                        self._eos_token_id = tid
                    elif special == "pad_token":
                        self._pad_token_id = tid
                    elif special == "unk_token":
                        self._unk_token_id = tid

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #

    def encode(self, text: str, add_eos: bool = True,
               max_length: Optional[int] = None) -> List[int]:
        """Encode text into token ids."""
        if self._hf is not None:
            enc = self._hf.encode(text)
            ids = list(enc.ids)
        else:
            ids = [ord(ch) for ch in text]

        if max_length is not None:
            ids = ids[:max_length]

        if add_eos and self._eos_token_id is not None:
            ids.append(self._eos_token_id)
        return ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts."""
        if self._hf is not None:
            encs = self._hf.encode_batch(texts)
            return [list(e.ids) for e in encs]
        return [[ord(ch) for ch in t] for t in texts]

    def decode(self, ids: List[int], skip_special: bool = False
               ) -> str:
        """Decode token ids into text."""
        if self._hf is not None:
            ids = [i for i in ids
                   if not (self._is_special(i) and skip_special)]
            return self._hf.decode(ids)
        return "".join(chr(i) for i in ids if 32 <= i < 127)

    def _is_special(self, tid: int) -> bool:
        """Check if an id is one of the known special tokens."""
        return tid in (self._eos_token_id, self._pad_token_id,
                       self._unk_token_id)

    # ------------------------------------------------------------------ #
    # Vocabulary introspection
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        if self._hf is not None:
            return self._hf.get_vocab_size(with_added_tokens=True)
        return 128  # ascii range for the char-level fallback

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._pad_token_id


# --------------------------------------------------------------------- #
# Sampling helpers (logits -> next token)
# --------------------------------------------------------------------- #

def top_k_sample(logits: List[float], top_k: int,
                 temperature: float = 1.0) -> int:
    """Sample from the top-k largest logits."""
    if top_k <= 0:
        return greedy_sample(logits, temperature)
    scored = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)\
        [:top_k]
    max_val = max(v for _, v in scored)
    exps = [math.exp(v - max_val) for _, v in scored]
    total = sum(exps)
    r = random.uniform(0.0, total)
    cum = 0.0
    for (i, v), e in zip(scored, exps):
        cum += e
        if r <= cum:
            return i
    return scored[-1][0]


def top_p_sample(logits: List[float], top_p: float,
                 temperature: float = 1.0) -> int:
    """Sample from the smallest set of logits whose cumulative prob >= p."""
    if not (0.0 < top_p <= 1.0):
        return greedy_sample(logits, temperature)
    max_val = max(logits)
    scored = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    acc = 0.0
    chosen = []
    for i, v in scored:
        acc += math.exp(v - max_val)
        chosen.append((i, acc))
        if acc >= top_p:
            break
    total = chosen[-1][1]
    r = random.uniform(0.0, total)
    cum = 0.0
    for i, acc in chosen:
        cum += acc
        if r <= cum:
            return i
    return chosen[-1][0]


def greedy_sample(logits: List[float], temperature: float = 1.0) -> int:
    """Pick the highest logit (optionally after temperature scaling)."""
    if temperature <= 0.0:
        temperature = 1e-9
    scaled = [v / temperature for v in logits]
    max_val = max(scaled)
    exps = [math.exp(v - max_val) for v in scaled]
    total = sum(exps)
    r = random.uniform(0.0, total)
    cum = 0.0
    for i, e in enumerate(exps):
        cum += e
        if r <= cum:
            return i
    return len(exps) - 1
