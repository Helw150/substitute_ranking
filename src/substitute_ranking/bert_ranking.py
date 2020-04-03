import re
import substitute_ranking.util as util
import substitute_ranking.weight_model as wm
from transformers import BertTokenizer, BertModel
import torch
from torch.autograd import Variable
from torch import nn
from tqdm import tqdm


model_version = "bert-base-uncased"
do_lower_case = True
model = BertModel.from_pretrained(model_version, output_attentions=True).eval().cuda()
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
cos = nn.CosineSimilarity(dim=0, eps=1e-6)


def train(
    input_sentences,
    original_words,
    substitution_words,
    labels,
    eval_function=None,
    batch_size=100,
    n_iters=3000,
    lr_rate=0.001,
):
    assert len(input_sentences) == len(original_words)
    assert len(input_sentences) == len(substitution_words)
    assert len(input_sentences) == len(labels)
    training_data = []
    for sentence, original_word, substitution_word_seq, label in zip(
        input_sentences, original_words, substitution_words, labels
    ):
        original_embeddings, attention = pre_process(sentence, original_word)
        original_embeddings = original_embeddings.cpu()
        attention = attention.cpu()
        new_embeddings_seq = [
            pre_process(
                re.sub(
                    r"\b" + re.escape(original_word) + r"\b", sub, sentence, count=1
                ),
                sub,
            )[0]
            for sub in substitution_word_seq
        ]
        for new_embeddings in new_embeddings_seq:
            training_data.append(
                (original_embeddings, new_embeddings, attention, label)
            )

    epochs = n_iters / (len(training_data) / batch_size)
    predictor = wm.LogisticRegression(
        144, 1
    ).cuda()  # Number of Attention Heads x binary prediction
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(predictor.parameters(), lr=lr_rate)
    train_loader = torch.utils.data.DataLoader(
        dataset=training_data, batch_size=batch_size, shuffle=True
    )

    iter = 0
    for epoch in tqdm(range(int(epochs))):
        for i, (orig, new, attn, labels) in enumerate(train_loader):
            orig = Variable(orig).cuda()
            new = Variable(new).cuda()
            attn = Variable(attn).cuda()
            labels = Variable(labels.float()).cuda()

            optimizer.zero_grad()
            outputs = predictor(orig, new, attn)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iter += 1
            if eval_function != None and iter % 500:
                eval_function(predictor)
    return predictor


def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # num_heads x seq_len x seq_len
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def self_attention_averages(attention, tokens, sub_tokens):
    attn = format_attention(attention)
    attn_for_embedding = [
        sum(
            [
                attn[:, :, sub_token, j].mean()
                for sub_token in range(sub_tokens[0], sub_tokens[1])
            ]
        )
        for j in range(0, len(tokens))
    ]
    attn_for_embedding[sub_tokens[0] : sub_tokens[1]] = [
        util.average(attn_for_embedding[sub_tokens[0] : sub_tokens[1]])
    ]
    total_attn = sum(attn_for_embedding)
    attn_for_embedding = [attn / total_attn for attn in attn_for_embedding]
    assert abs(sum(attn_for_embedding).item() - 1) < 0.01, (
        sum(attn_for_embedding),
        total_attn,
    )
    return attn_for_embedding


def pre_process(input_sentence, target_word):
    token_subset = util.tokenize_word(target_word, tokenizer)
    inputs = tokenizer.encode_plus(
        input_sentence, return_tensors="pt", add_special_tokens=False
    )
    token_type_ids = inputs["token_type_ids"].cuda()
    input_ids = inputs["input_ids"].cuda()
    embeddings, _, attention = model(input_ids, token_type_ids=token_type_ids)
    input_id_list = input_ids[0].tolist()  # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    range_of_target = util.find_sublist(token_subset, tokens)
    sub_tokens = util.find_sublist(token_subset, tokens)
    attn = format_attention(attention)
    attn = torch.stack(
        [
            attn[:, :, sub_tokens[0] : sub_tokens[1], j].mean(-1)
            for j in range(0, len(tokens))
        ]
    )
    attn = attn.permute([1, 2, 0]).flatten(0, 1)
    attn = attn.tolist()
    for i, head in enumerate(attn):
        head[sub_tokens[0] : sub_tokens[1]] = [
            util.average(torch.FloatTensor(head)[sub_tokens[0] : sub_tokens[1]])
        ]
        attn[i] = head
    attn = torch.FloatTensor(attn)
    attn = (attn.permute([1, 0]) / attn.sum(1)).permute([1, 0])
    return util.replace_with_average(embeddings[-1], range_of_target), attn


def stats(input_sentence, target_word, attention_needed):
    token_subset = util.tokenize_word(target_word, tokenizer)
    inputs = tokenizer.encode_plus(
        input_sentence, return_tensors="pt", add_special_tokens=False
    )
    token_type_ids = inputs["token_type_ids"]
    input_ids = inputs["input_ids"]
    embeddings, _, attention = model(input_ids, token_type_ids=token_type_ids)
    input_id_list = input_ids[0].tolist()  # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    range_of_target = util.find_sublist(token_subset, tokens)
    if attention_needed:
        sub_tokens = util.find_sublist(token_subset, tokens)
        return (
            util.replace_with_average(embeddings[-1], range_of_target),
            self_attention_averages(attention, tokens, sub_tokens),
        )
    else:
        return util.replace_with_average(embeddings[-1], range_of_target), None


def score_token(original_embedding, new_embedding, attention):
    return attention * cos(original_embedding, new_embedding)


def raw_substitution_data(input_sentence, original_word, substitution_word):
    substitution_sentence = re.sub(
        r"\b" + re.escape(original_word) + r"\b",
        substitution_word,
        input_sentence,
        count=1,
    )
    original_embeddings, attention = stats(input_sentence, original_word, True)
    new_embeddings, _ = stats(substitution_sentence, substitution_word, False)
    assert len(original_embeddings) == len(attention), (
        len(original_embeddings),
        len(attention),
    )
    assert len(original_embeddings) == len(new_embeddings), (
        len(original_embeddings),
        len(new_embeddings),
    )
    return original_embeddings, new_embeddings, attention


def score_substitution(input_sentence, original_word, substitution_word):
    substitution_sentence = re.sub(
        r"\b" + re.escape(original_word) + r"\b",
        substitution_word,
        input_sentence,
        count=1,
    )
    original_embeddings, attention = stats(input_sentence, original_word, True)
    new_embeddings, _ = stats(substitution_sentence, substitution_word, False)
    assert len(original_embeddings) == len(attention), (
        len(original_embeddings),
        len(attention),
    )
    assert len(original_embeddings) == len(new_embeddings), (
        len(original_embeddings),
        len(new_embeddings),
    )
    original_embeddings, new_embeddings, attention = raw_substitution_data(
        input_sentence, original_word, substitution_word
    )
    token_scores = [
        score_token(original_embeddings[i], new_embeddings[i], attention[i])
        for i in range(0, len(original_embeddings))
    ]
    return sum(token_scores).item()
