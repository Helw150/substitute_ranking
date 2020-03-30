import torch
from torch import nn


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


# Taken From https://stackoverflow.com/questions/2250633/python-find-a-list-within-members-of-another-listin-order
def find_sublist(sub, bigger):
    if not bigger:
        return -1
    if not sub:
        return 0
    first, rest = sub[0], sub[1:]
    pos = 0
    try:
        while True:
            pos = bigger.index(first, pos) + 1
            if not rest or bigger[pos : pos + len(rest)] == rest:
                return pos - 1, pos + len(rest)
    except ValueError:
        return -1


def average(list_avg):
    return sum(list_avg) / len(list_avg)


def replace_with_average(vectors, range_to_average):
    vector_to_average = vectors[range_to_average[0] : range_to_average[1]]
    averaged_vector = vectors.mean(0).unsqueeze(0)
    return torch.cat(
        [
            vectors[: range_to_average[0]],
            averaged_vector,
            vectors[range_to_average[1] :],
        ],
        dim=0,
    )


def tokenize_word(word, tokenizer):
    token_encoding = tokenizer.encode_plus(
        word, return_tensors="pt", add_special_tokens=False
    )
    token_ids = token_encoding["input_ids"][0].tolist()
    return tokenizer.convert_ids_to_tokens(token_ids)
