import json
import os
import regex as re
from typing import Iterable, Iterator

from cs336_basics.bpe_trainer import BPETrainer
from cs336_basics.utils import DoublyLinkedList

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        self.decode_vocab = vocab
        self.encode_vocab = {v: k for k, v in self.decode_vocab.items()}
        self.merges = merges
        self.merge_ranks = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens = special_tokens
        self.PAT =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = json.load(f)
        if special_tokens is None:
            special_tokens = []
        return cls(vocab, merges, special_tokens)

    def _bpe_ids_for_bytes(self, blob: bytes) -> list[int]:
        # Convert the byte sequence into a list of tokens, with each token initialized as a single byte.
        word = [bytes([b]) for b in blob]
        if len(word) <= 1:
            return [self.encode_vocab[word[0]]] if word else []

        def get_pairs(seq):
            # Return a set of adjacent pairs.
            return {(seq[i], seq[i+1]) for i in range(len(seq)-1)}

        pairs = get_pairs(word)
        while True:
            # Select the mergeable pair with the lowest rank.
            best = None
            best_rank = None
            for p in pairs:
                r = self.merge_ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best = r, p
            if best is None:
                break

            A, B = best
            new_word = []
            i = 0
            n = len(word)
            while i < n:
                if i < n - 1 and word[i] == A and word[i+1] == B:
                    new_word.append(A + B)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        return [self.encode_vocab[tok] for tok in word]


    def encode(self, text: str) -> list[int]:
        out_ids: list[int] = []
        if not text:
            return out_ids

        # If there are special tokens, split the text by the longest token first and keep the matches;
        # otherwise, process the entire text as a single segment.
        if getattr(self, "special_tokens", None):
            specials = sorted(self.special_tokens, key=len, reverse=True)
            special_re = re.compile("(" + "|".join(re.escape(s) for s in specials) + ")")
            parts = special_re.split(text)
        else:
            parts = [text]

        for piece in parts:
            if not piece:
                continue

            # If a special token is matched, look it up in the table and output directly.
            if getattr(self, "special_tokens", None) and piece in self.special_tokens:
                b = piece.encode("utf-8")
                tid = self.encode_vocab.get(b)
                if tid is None:
                    raise KeyError(f"Special token not in vocab: {piece!r}")
                out_ids.append(tid)
                continue

            # For regular text, first pre-tokenize using the GPT-2 pattern,
            # then perform BPE merging on each sub-segment.
            for m in re.finditer(self.PAT, piece):
                sub = m.group(0)
                if not sub:
                    continue
                blob = sub.encode("utf-8")
                out_ids.extend(self._bpe_ids_for_bytes(blob))
        return out_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        list_of_bytes = []
        for id in ids:
            list_of_bytes.append(self.decode_vocab[id])
        nodes = DoublyLinkedList()
        for node_bytes in list_of_bytes:
            nodes.add_node(node_bytes)
        for merge in reversed(self.merges):
            nodes.expand_pair(merge)
        byte_stream = b"".join(node for node in nodes)
        return byte_stream.decode("utf-8", errors="replace")

if __name__ == "__main__":
    bpe_trainer = BPETrainer(input_path="data/TinyStoriesV2-GPT4-train.txt", vocab_size=257 + 100, special_tokens=["<|endoftext|>"], pretokenized_words_path="data/pretokenized_words.json")
    bpe_trainer.train_bpe()
    vocab, merges = bpe_trainer.decode_vocab, bpe_trainer.merges
    tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    print(tokenizer.encode("Hello, world!"))