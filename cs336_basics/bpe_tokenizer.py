import os
from collections import defaultdict
from typing import BinaryIO
import regex as re
import json
from tqdm import tqdm
import time

from cs336_basics.utils import DoublyLinkedList

class BPETokenizer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str], pretokenized_words_path: str):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pretokenized_words = defaultdict(int)
        self.pretokenized_words_path = pretokenized_words_path
        self.chunk_boundaries = []
        self.encode_vocab = {}
        self.decode_vocab = {}
        self.merges = []
        INITIAL_VOCAB_SIZE = 256
        self.merge_count = self.vocab_size - INITIAL_VOCAB_SIZE - 1
        for i in range(INITIAL_VOCAB_SIZE):
            self.encode_vocab[bytes([i])] = i
            self.decode_vocab[i] = bytes([i])
        self.encode_vocab[self.special_tokens[0].encode("utf-8")] = INITIAL_VOCAB_SIZE
        self.decode_vocab[INITIAL_VOCAB_SIZE] = self.special_tokens[0].encode("utf-8")
        
    def pretokenize(self):
        with open(self.input_path, "rb") as input_file:
            self.input_file = input_file
            self.chunk_boundaries = self._find_chunk_boundaries_v2(desired_num_chunks=100, split_special_token=self.special_tokens[0].encode("utf-8"))
            print("finished finding chunk boundaries, length: ", len(self.chunk_boundaries))
            self._save_pretokenized_words(self.pretokenized_words_path)

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

    def _save_pretokenized_words(self, pretokenized_words_path: str) -> dict[str, int]:
        """
        Count the number of tokens in each chunk.
        """
        for chunk_start, chunk_end in tqdm(zip(self.chunk_boundaries[:-1], self.chunk_boundaries[1:]), total=len(self.chunk_boundaries) - 1, desc="Pretokenizing words"):
            self.input_file.seek(chunk_start)
            chunk = self.input_file.read(chunk_end - chunk_start)
            pat = re.escape(self.special_tokens[0])
            parts = re.split(pat, chunk.decode("utf-8"))
            for part in parts:
                for match in re.finditer(self.PAT, part):
                    words = match.group(0)
                    self.pretokenized_words[words] += 1
        print("finished getting pretokenized words, length: ", len(self.pretokenized_words))
        with open(pretokenized_words_path, "w") as f:
            json.dump({k: v for k, v in self.pretokenized_words.items()}, f)
        return

    def train_bpe(self) -> list[tuple[bytes, bytes]]:
        """
        Compute the merges for the BPE tokenizer.
        """
        # Load the pretokenized words
        with open(self.pretokenized_words_path, "r") as f:
            self.pretokenized_words = json.load(f)
        dll_vector = []
        for i, (word, count) in enumerate(self.pretokenized_words.items()):
            bytes_of_word = word.encode("utf-8")
            node = DoublyLinkedList(bytes_of_word, count)
            dll_vector.append(node)
        print(f"length of dll_vector: {len(dll_vector)}")
        pair_map = defaultdict(int)
        for node in dll_vector:
            pair = node.get_pairs()
            for p in pair:
                pair_map[p] += node.freq
        time_start = time.time()
        time_merge_pair_time = 0
        time_get_pair_time = 0
        for i in tqdm(range(self.merge_count), desc="Training BPE"):
            time_start_get_pair = time.time()
            most_common_pair = max(pair_map.items(), key=lambda x: (x[1], x[0]))
            time_get_pair_end = time.time()
            time_get_pair_time += time_get_pair_end - time_start_get_pair
            del pair_map[most_common_pair[0]]
            self.merges.append(most_common_pair[0])
            next_token = most_common_pair[0][0] + most_common_pair[0][1]
            next_token_id = len(self.encode_vocab)
            self.encode_vocab[next_token] = next_token_id
            self.decode_vocab[next_token_id] = next_token
            time_merge_pair_start = time.time()
            for node in dll_vector:
                deltas = node.merge_pair_and_get_deltas(most_common_pair[0])  # {pair: delta_count}，delta 可正可负
                if not deltas:
                    continue
                w = node.freq
                for p, d in deltas.items():
                    if d:
                        pair_map[p] += d * w
                        if pair_map[p] <= 0:
                            pair_map.pop(p, None)
            time_merge_pair_end = time.time()
            time_merge_pair_time += time_merge_pair_end - time_merge_pair_start
        time_end = time.time()
        print(f"encode_vocab length: {len(self.encode_vocab)}")
        print(f"decode_vocab length: {len(self.decode_vocab)}")
        print(f"merges length: {len(self.merges)}")
        print(f"time taken to get pair: {time_get_pair_time}")
        print(f"time taken to merge pair: {time_merge_pair_time}")
        print(f"time taken: {time_end - time_start}")
        return

if __name__ == "__main__":
    tokenizer = BPETokenizer(input_path="data/TinyStoriesV2-GPT4-train.txt", vocab_size=257 + 100, special_tokens=["<|endoftext|>"], pretokenized_words_path="data/pretokenized_words.json")
    tokenizer.pretokenize()
    tokenizer.train_bpe()