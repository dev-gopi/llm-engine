class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.rev = {v:k for k,v in vocab.items()}
    def encode(self, text):
        return [self.vocab.get(w,0) for w in text.split()]
    def decode(self, ids):
        return " ".join(self.rev.get(i,"<unk>") for i in ids)
