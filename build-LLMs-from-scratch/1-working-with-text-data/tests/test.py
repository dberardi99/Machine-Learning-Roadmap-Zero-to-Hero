import sys
sys.path.insert(1, '/Users/danieleberardi/Documents/Dati/Machine-Learning-Roadmap-Zero-to-Hero/build-LLMs-from-scratch/1-working-with-text-data')
from tokenizer import BPETokenizerSimple

text = "Hello how are you?\n I am ok thanks and you \n I am ok too!!!!!!!!!!!!!!"
vocab_path = "/Users/danieleberardi/Documents/Dati/Machine-Learning-Roadmap-Zero-to-Hero/build-LLMs-from-scratch/1-working-with-text-data/utils/gpt2-vocab.json"
bpe_path = "/Users/danieleberardi/Documents/Dati/Machine-Learning-Roadmap-Zero-to-Hero/build-LLMs-from-scratch/1-working-with-text-data/utils/gpt2-merges.txt"
tok = BPETokenizerSimple()
tok.load_vocab_and_merges(vocab_path=vocab_path, bpe_merges_path=bpe_path)
print(tok.encode(text))
