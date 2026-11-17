"""A character-level tokenizer over printable ASCII.

Real LLMs use BPE or SentencePiece tokenizers with ~32k-200k entries. A serving
engine does not care how tokens are produced, only that text maps to a list of
integer ids and back, so the simplest possible tokenizer is enough here: one id
per printable ASCII character, plus a few special tokens.
"""

from __future__ import annotations

import string

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
_NUM_SPECIAL = 4

# Space through tilde, plus newline and tab: 97 characters.
_CHARS = "".join(chr(c) for c in range(32, 127)) + "\n\t"
assert set(_CHARS) <= set(string.printable)


class CharTokenizer:
    """Maps printable ASCII characters to ids ``4..100`` and back.

    Unknown characters map to ``UNK_ID``. Special tokens decode to the empty string.
    """

    pad_id = PAD_ID
    bos_id = BOS_ID
    eos_id = EOS_ID
    unk_id = UNK_ID

    def __init__(self) -> None:
        self._char_to_id = {ch: i + _NUM_SPECIAL for i, ch in enumerate(_CHARS)}
        self._id_to_char = {i: ch for ch, i in self._char_to_id.items()}

    @property
    def vocab_size(self) -> int:
        """Total number of token ids, including special tokens."""
        return _NUM_SPECIAL + len(_CHARS)

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        """Turn ``text`` into token ids, optionally prefixed with ``BOS_ID``."""
        ids = [self._char_to_id.get(ch, UNK_ID) for ch in text]
        return [BOS_ID, *ids] if add_bos else ids

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into text, dropping special tokens."""
        return "".join(self._id_to_char.get(i, "") for i in ids)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token. Handy for streaming one token at a time."""
        return self._id_to_char.get(token_id, "")
