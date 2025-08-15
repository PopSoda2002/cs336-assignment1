class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        self.vocab = vocab
        self.merges = merges

    def encode(self, text: str) -> list[int]:
        return [self.vocab[token] for token in text.split()]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[id] for id in ids)