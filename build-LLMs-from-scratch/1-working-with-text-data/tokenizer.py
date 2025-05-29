import re
import json
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tiktoken import Encoding

class SimpleTokenizerV1:  # this is not the real tokenizer used by GPT models, but it is a simplified version
    """
    A first simplified version of tokenizer.
    """

    def __init__(self, vocab: dict) -> None:  # the class takes a vocabulary as input
        self.str_to_int = vocab  # regular vocabulary (string to integer mapping)
        self.int_to_str = {
            i: s for s, i in vocab.items()
        }  # inverted vocabulary (integer to string mapping)

    def encode(self, text: str) -> list[int]:
        """
        A function to break down text into tokens using regular expressions.
        """
        preprocessed = re.split(
            r'([,.:;?_!"()\']|--|\s)', text
        )  # creation of tokenized version of input text
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [
            self.str_to_int[s] for s in preprocessed
        ]  # creation of token ID for each token/string
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        A function to re-create the original text starting from the token IDs.
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(
            r'\s+([,.?!"()\'])', r"\1", text
        )  # replace spaces before the specified punctuations
        return text


class SimpleTokenizerV2:  # advanced version of our tokenizer including special tokens
    """
    A second version of tokenizer which is also able to deal with unknown words.
    """

    def __init__(self, vocab: dict) -> None:  # the class takes a vocabulary as input
        self.str_to_int = vocab  # regular vocabulary (string to integer mapping)
        self.int_to_str = {
            i: s for s, i in vocab.items()
        }  # inverted vocabulary (integer to string mapping)

    def encode(self, text: str) -> list[int]:
        """
        A function to break down text into tokens using regular expressions and including special tokens.
        """
        preprocessed = re.split(
            r'([,.:;?_!"()\']|--|\s)', text
        )  # creation of tokenized version of input text
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]  # replace the string if it isn't in the vocabulary
        ids = [
            self.str_to_int[s] for s in preprocessed
        ]  # creation of token ID for each token/string
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        A function to re-create the original text starting from the token IDs.
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(
            r'\s+([,.?!"()\'])', r"\1", text
        )  # replace spaces before the specified punctuations
        return text


class BPETokenizerSimple:
    """
    A simplified version of the BPE (Byte Pair Encoding) tokenizer by Sebastian Raschka.

    Links:
        https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb
        https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
    """

    def __init__(self) -> None:
        self.vocab = {}  # map token_id to token_str (e.g., {11246: "some"})
        self.inverse_vocab = {}  # map token_str to token_id (e.g., {"some": 11246})
        self.bpe_merges = {}  # dictionary of BPE merges/pairs (e.g., {(token_id1, token_id2): merged_token_id})
        self.bpe_ranks = {}  # TODO

    def encode(self, text: str, allowed_special: set = None) -> list[int]:
        """
        Encode the input text into a list of token IDs.

        Args:
            text (str): The input text to encode.
            allowed_special (set or None): Special tokens to allow passthrough.

        Returns:
            token_ids (list[int]): List of token IDs.
        """
        import re

        token_ids = []

        # TODO: Implement the logic if special tokens are present

        # If no special tokens are present
        tokens = []
        lines = text.split(
            "\n"
        )  # break the string at each occurrence of the "\n" character
        for i, line in enumerate(lines):
            if i > 0:
                tokens.append(
                    "\n"
                )  # only from the second line onwards we append the "\n"
            words = line.split()  # split the string with each occurrence of whitespace
            for j, word in enumerate(words):
                if (
                    j == 0 and i > 0
                ):  # replace whitespaces with "Ġ" (e.g., "Hello world" might be tokenized as ["Hello", "Ġworld"])
                    tokens.append(
                        "Ġ" + word
                    )  # also for the first word of each new line from the second onwards we append the "Ġ"
                elif j == 0:
                    tokens.append(
                        word
                    )  # if there is only one line, the first word (j == 0) is appended without the "Ġ"
                else:
                    tokens.append(
                        "Ġ" + word
                    )  # from the second word onwards we append the "Ġ" to show the presence of a whitespace

        for token in tokens:  # "token" is a string here
            if token in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[token])
            else:
                token_ids.extend(self.tokenize_with_bpe(token))

        return token_ids

    def tokenize_with_bpe(self, token: str) -> list[int]:
        """
        Tokenize a single token using BPE merges.

        Args:
            token (str): The token to tokenize.

        Returns:
            token_ids (list[int]): The list of token IDs after applying BPE.
        """
        # tokenization starts always by splitting each word into its characters
        token_ids = [
            self.inverse_vocab.get(char, None) for char in token
        ]  # tokenize the token into individual characters
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        # if OpenAI's GPT-2 merges have not been loaded
        if not self.bpe_ranks:
            can_merge = True
            while (
                can_merge and len(token_ids) > 1
            ):  # token_ids contains the IDs of the word's single characters
                can_merge = False
                new_tokens = []
                i = 0
                while (
                    i < len(token_ids) - 1
                ):  # scan through the entire length of the word
                    pair = (token_ids[i], token_ids[i + 1])
                    if pair in self.bpe_merges:
                        merged_token_id = self.bpe_merges[pair]
                        new_tokens.append(merged_token_id)
                        i += 2
                        can_merge = True
                    else:
                        new_tokens.append(token_ids[i])
                        i += 1
                if i < len(token_ids):
                    new_tokens.append(token_ids[i])
                token_ids = new_tokens  # once we substitute the pairs, we loop again to find new pairs
            return token_ids

    def load_vocab_and_merges(self, vocab_path: str, bpe_merges_path: str) -> None:
        """
        Load the vocabulary and BPE merges from JSON files.

        Args:
            vocab_path (str): Path to the vocabulary (JSON) file.
            bpe_merges_path (str): Path to the BPE merges (text) file.
        """
        # load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(v): k for k, v in loaded_vocab.items()}
            self.inverse_vocab = {k: int(v) for k, v in loaded_vocab.items()}

        # handle newline character without adding a new token (first prio for "fallback_token" is "<|endoftext|>")
        if "\n" not in self.inverse_vocab:
            fallback_token = next(
                (
                    token
                    for token in ["<|endoftext|>", "Ġ", ""]
                    if token in self.inverse_vocab
                ),
                None,
            )
            if fallback_token is not None:
                newline_token_id = self.inverse_vocab[
                    fallback_token
                ]  # assign <|endoftext|>'s token ID to \n's one
            else:
                raise KeyError("No suitable token found in vocabulary to map '\\n'.")

            self.inverse_vocab["\n"] = newline_token_id
            self.vocab[newline_token_id] = "\n"

        # load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            if lines and lines[0].startswith("#"):
                lines = lines[1:]  # skip header line if present

            for rank, line in enumerate(lines):
                pair = tuple(
                    line.strip().split()
                )  # .strip() removes the whitespaces before and after, while .split() divides the two pairs
                if len(pair) != 2:
                    print(f"Line {rank + 1} has more than 2 entries: {line.strip()}")
                    continue
                token1, token2 = pair
                if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                    token_id1 = self.inverse_vocab[token1]
                    token_id2 = self.inverse_vocab[token2]
                    merged_token = token1 + token2
                    if merged_token in self.inverse_vocab:
                        merged_token_id = self.inverse_vocab[merged_token]
                        self.bpe_merges[(token_id1, token_id2)] = merged_token_id
                    else:
                        print(
                            f"Merged token '{merged_token}' not found in vocab. Skipping."
                        )
                else:
                    print(
                        f"Skipping pair {pair} as one of the tokens is not in the vocabulary."
                    )

    def train(
        self, text: str, vocab_size: int, allowed_special: set = {"<|endoftext|>"}
    ) -> None:
        """
        Train the BPE tokenizer from scratch (without using any existing vocabulary).

        Args:
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            allowed_special (set): A set of special tokens to include in the vocabulary.
        """
        # TODO


class GPTDatasetV1(Dataset):
    """
    Create the dataset (namely input and target chunks) needed to pre-train the LLM.

    Args:
        txt (str): The training dataset (the text to tokenize and to divide into chunks).
        tokenizer (Encoding): The tokenizer to use.
        max_length (int): The context's length (namely the chunk size).
        stride (int): ...
    """

    def __init__(self, txt: str, tokenizer: Encoding, max_length: int, stride: int) -> None:
        """
        Link: https://github.com/openai/tiktoken/blob/4560a8896f5fb1d35c6f8fd6eee0399f9a1a27ca/tiktoken/registry.py#L63
        """
        self.input_ids = []
        self.target_ids = []

        # tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # use a sliding window to chunk the book into overlapping seuences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[
                i + 1 : i + max_length + 1
            ]  # target = input shifted by 1 position
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """
        Return the total number of rows in the dataset.

        Returns:
            len(self.input_ids) (int): The total number of rows in the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx) -> list[int]:
        """
        Return a single row from the dataset.

        Args:
            idx (int): The row index.

        Returns:
            self.input_ids[idx] (list[int]): The input row from the dataset.
            self.target_ids[idx] (list[int]): The target row from the dataset.
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Write here and add type hints to the function ...

    Args:
        txt (str): The training dataset.
        batch_size (int): The number of samples are in each batch.
        max_length (int): The context's length (namely the chunk size). 
        stride (int): ...
        shuffle (bool): ...
        drop_last (bool): If true it truncates the last batch (chunk) whenever the dataset size is not
        divisible by the batch_size. Leaving this last (small) batch could result in a loss spike and
        thus in an unstable training. It is usually a good idea to have batches of the same size.
        num_workers (int): It is for using multiple background processors (multiple subprocesses).

    Returns:
        dataloader (DataLoader): ...
    """

    # initialize the (tiktoken's) tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader
