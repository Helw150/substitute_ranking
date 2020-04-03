import re
import torch
from torch import nn
import substitute_ranking.bert_ranking as br


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, original_embeddings, new_embeddings, attention):
        original_embeddings = original_embeddings.squeeze(0)
        new_embeddings = new_embeddings.squeeze(0)
        attention = attention.squeeze(0)
        similarity = self.cos(original_embeddings, new_embeddings)
        x = (attention * similarity).sum(1)
        outputs = self.linear(x)
        return torch.sigmoid(outputs)

    def predict(self, sentence, original_word, sub_word):
        substitution_sentence = re.sub(
            r"\b" + re.escape(original_word) + r"\b", sub_word, sentence, count=1
        )
        original_embeddings, attention = br.pre_process(sentence, original_word)
        new_embeddings = br.pre_process(substitution_sentence, sub_word)[0]
        return self.forward(original_embeddings, new_embeddings, attention)

    def multi_predict(self, sentence, original_words, sub_word_seq):
        original_embeddings, attention = br.pre_process(sentence, original_word)
        results = []
        for sub_word in sub_word_seq:
            new_embeddings = br.pre_process(
                sentence.replace(original_word, sub_word), sub_word
            )[0]
            results.append(self.forward(original_embeddings, new_embeddings, attention))
        return results
