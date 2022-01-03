import torch
from torch import nn
import collections
import re
from d2l import torch as d2l


tokens = d2l.tokenize(d2l.read_time_machine())
# Since each text line is not necessarily a sentence or a paragraph, we
# concatenate all text lines
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)


def main():
    freqs = [freq for token, freq in vocab.token_freqs]
    bigram_freqs = n_gram(2)
    trigram_freqs = n_gram(3)
    d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
             ylabel='frequency: n(x)', xscale='log', yscale='log',
             legend=['unigram', 'bigram', 'trigram'])

def n_gram(n):
    corpus = [token for line in tokens for token in line]
    if n == 1:
        vocab = d2l.Vocab(corpus)
        return vocab.token_freqs
    if n == 2:
        bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
        bigram_vocab = d2l.Vocab(bigram_tokens)
        return bigram_vocab.token_freqs
    if n == 3:
        trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1],
                                                   corpus[2:])]
        trigram_vocab = d2l.Vocab(trigram_tokens)
        return trigram_vocab.token_freqs


if __name__ == "__main__":
    main()