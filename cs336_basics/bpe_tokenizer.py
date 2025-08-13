import os
from collections import defaultdict
from typing import BinaryIO
import regex as re
import json
from tqdm import tqdm

class BPETokenizer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pretokenized_words = defaultdict(int)
        self.chunk_boundaries = []
        
    def pretokenize(self, pretokenized_words_path: str):
        with open(self.input_path, "rb") as input_file:
            self.input_file = input_file
            self.chunk_boundaries = self._find_chunk_boundaries_v2(desired_num_chunks=100, split_special_token=self.special_tokens[0].encode("utf-8"))
            print("finished finding chunk boundaries, length: ", len(self.chunk_boundaries))
            self._save_pretokenized_words(pretokenized_words_path)

    def _find_chunk_boundaries_v2(
        self,
        desired_num_chunks: int,
        split_special_token: bytes) -> list[int]:
        
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        self.input_file.seek(0, os.SEEK_END)
        file_size = self.input_file.tell()
        self.file_size = file_size
        self.input_file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            end_position = chunk_boundaries[bi + 1] if bi + 1 < len(chunk_boundaries) else self.file_size
            self.input_file.seek(initial_position)  # Start at boundary guess
            while initial_position < end_position:
                mini_chunk = self.input_file.read(mini_chunk_size)  # Read a mini chunk
                
                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = self.file_size
                    break

                # Find the special token in the mini chunk, if it exists
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at + len(split_special_token)
                    # print(f"step {bi} Found special token at {initial_position + found_at + len(split_special_token)}")
                    break
                initial_position += mini_chunk_size
                if initial_position > end_position:
                    chunk_boundaries[bi] = 0

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def _save_pretokenized_words(self, pretokenized_words_path: str) -> dict[bytes, int]:
        """
        Count the number of tokens in each chunk.
        """
        for chunk_start, chunk_end in tqdm(zip(self.chunk_boundaries[:-1], self.chunk_boundaries[1:]), total=len(self.chunk_boundaries) - 1, desc="Pretokenizing words"):
            self.input_file.seek(chunk_start)
            chunk = self.input_file.read(chunk_end - chunk_start)
            print(f"chunk size: {chunk_end - chunk_start}")
            
            for match in re.finditer(self.PAT, chunk.decode("utf-8")):
                words = match.group(0)
                self.pretokenized_words[words] += 1
        print("finished getting pretokenized words, length: ", len(self.pretokenized_words))
        with open(pretokenized_words_path, "w") as f:
            json.dump({k: v for k, v in self.pretokenized_words.items()}, f)
        return

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Train the BPE tokenizer.
        """
        # Load the pretokenized words
        with open(self.pretokenized_words_path, "r") as f:
            self.pretokenized_words = json.load(f)
        
        pass


if __name__ == "__main__":
    tokenizer = BPETokenizer(input_path="data/TinyStoriesV2-GPT4-train.txt", vocab_size=1000, special_tokens=["<|endoftext|>"])
    tokenizer.pretokenize(pretokenized_words_path="data/pretokenized_words.json")
