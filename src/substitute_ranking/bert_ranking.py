from transformers import BertTokenizer, BertModel
import torch
from torch import nn
import substitute_ranking.util as util

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

def self_attention_averages(attention, tokens, sub_token):
    attn = format_attention(attention)
    attn_for_embedding = [attn[:, :, sub_token, j].mean() for j in range(0, len(tokens))]
    return attn_for_embedding

def original_stats(input_sentence, target_word):
    token_subset = util.tokenize_word(target_word, tokenizer)
    inputs = tokenizer.encode_plus(input_sentence, return_tensors='pt', add_special_tokens=True)
    token_type_ids = inputs['token_type_ids']
    input_ids = inputs['input_ids']
    embeddings, _, attention = model(input_ids, token_type_ids=token_type_ids)
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    range_of_target = util.find_sublist(token_subset, tokens)
    sub_token = tokens.index(target_word)
    return util.replace_with_average(embeddings[0], range_of_target), self_attention_averages(attention, tokens, sub_token)

def substitution_stats(substitution_sentence, substitution_word):
    token_subset = util.tokenize_word(substitution_word, tokenizer)
    inputs = tokenizer.encode_plus(substitution_sentence, return_tensors='pt', add_special_tokens=True)
    token_type_ids = inputs['token_type_ids']
    input_ids = inputs['input_ids']
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    range_of_target = util.find_sublist(token_subset, tokens)
    embeddings, _, _ = model(input_ids, token_type_ids=token_type_ids)
    return util.replace_with_average(embeddings[0], range_of_target)

def score_token(original_embedding, new_embedding, attention):
    return attention * cos(original_embedding, new_embedding)
    

def score_substitution(input_sentence, original_word, substitution_word):
    substitution_sentence = input_sentence.replace(original_word, substitution_word)
    original_embeddings, attention = original_stats(input_sentence, original_word)
    new_embeddings = substitution_stats(substitution_sentence, substitution_word)
    assert(len(original_embeddings) == len(attention))
    assert(len(original_embeddings) == len(new_embeddings))
    return sum([score_token(original_embeddings[i], new_embeddings[i], attention[i]) for i in range(0, len(original_embeddings))]).item()
    
