import re
import substitute_ranking.util as util
from transformers import BertTokenizer, BertModel
import torch
from torch import nn


model_version = 'bert-base-uncased'
do_lower_case = True
model = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
cos = nn.CosineSimilarity(dim=0, eps=1e-6)

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # num_heads x seq_len x seq_len
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

def self_attention_averages(attention, tokens, sub_tokens):
    attn = format_attention(attention)
    attn_for_embedding = [sum([attn[:, :, sub_token, j].mean() for sub_token in range(sub_tokens[0], sub_tokens[1])]) for j in range(0, len(tokens))]
    attn_for_embedding[sub_tokens[0]:sub_tokens[1]] = [util.average(attn_for_embedding[sub_tokens[0]:sub_tokens[1]])]
    total_attn = sum(attn_for_embedding)
    attn_for_embedding = [attn/total_attn for attn in attn_for_embedding]
    assert(abs(sum(attn_for_embedding).item() - 1) < 0.01), (sum(attn_for_embedding), total_attn)
    return attn_for_embedding

def stats(input_sentence, target_word, attention_needed):
    token_subset = util.tokenize_word(target_word, tokenizer)
    inputs = tokenizer.encode_plus(input_sentence, return_tensors='pt', add_special_tokens=True)
    token_type_ids = inputs['token_type_ids']
    input_ids = inputs['input_ids']
    embeddings, _, attention = model(input_ids, token_type_ids=token_type_ids)
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    range_of_target = util.find_sublist(token_subset, tokens)
    if attention_needed:
        sub_tokens = util.find_sublist(token_subset, tokens)
        return util.replace_with_average(embeddings[0], range_of_target), self_attention_averages(attention, tokens, sub_tokens)
    else:
        return util.replace_with_average(embeddings[0], range_of_target), None



def score_token(original_embedding, new_embedding, attention):
    return attention * cos(original_embedding, new_embedding)
    

def score_substitution(input_sentence, original_word, substitution_word):
    substitution_sentence = re.sub(r"\b" + re.escape(original_word) + r"\b", substitution_word, input_sentence, count = 1)
    original_embeddings, attention = stats(input_sentence, original_word, True)
    new_embeddings, _ = stats(substitution_sentence, substitution_word, False)
    assert(len(original_embeddings) == len(attention)), (len(original_embeddings), len(attention))
    assert(len(original_embeddings) == len(new_embeddings)), (len(original_embeddings), len(new_embeddings))
    return sum([score_token(original_embeddings[i], new_embeddings[i], attention[i]) for i in range(0, len(original_embeddings))]).item()
    
