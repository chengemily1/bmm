import torch
from google.colab import drive
drive.mount('/content/drive')

# Navigate to the directory where your script is located
import sys
sys.path.append('/content/drive/My Drive/bmm_language_tutorial/')

from control_wrapper import LinearControlWrapper

def retrofit_model(Ws, layerlist, p):
    num_layers = len(layerlist)

    # Wrap all of the layers of the model
    for layer in range(num_layers):
        if type(layerlist[layer]) != LinearControlWrapper:
            layerlist[layer] = LinearControlWrapper(
                layerlist[layer],
                linear_probe=Ws[layer],
                p=p
            )
        else:
            layerlist[layer] = LinearControlWrapper(
                layerlist[layer].base_layer,
                linear_probe=Ws[layer],
                p=p
            )

def last_token_rep(x, attention_mask, padding='right'):
        seq_len = attention_mask.sum(dim=1)
        indices = (seq_len - 1)
        last_token_rep = x[torch.arange(x.size(0)), indices] if padding=='right' else x[torch.arange(x.size(0)), -1]
        return last_token_rep.cpu()

def encode_data(tokenizer, data, batch_size, max_length, device):
    N = len(data)
    
    # If the input data is text
    if type(data[0]) == str:
        encodings = tokenizer(data, padding=True, truncation=True, max_length=max_length, return_length=True, return_tensors="pt") # output variable length encodings
        encodings = [
            {'input_ids': encodings['input_ids'][i: i + batch_size].to(device),
            'attention_mask': encodings['attention_mask'][i: i + batch_size].to(device),
            'length': encodings['length'][i: i + batch_size] }
            for i in range(0, N, batch_size)
        ]
    else: # input data is tokens-- manually pad and batch.
        max_len = max([len(sentence) for sentence in data])
        data = [sentence for sentence in data if len(sentence) > 2]
        encodings = [tokenizer.encode(sentence[1:], padding='max_length', max_length=max_len, return_tensors="pt") \
                     for sentence in data]
        batched_encodings = [torch.stack(encodings[i: i + batch_size]).squeeze(1).to(device) for i in range(0, len(data), batch_size)]
        batched_attention_masks = [(tokens != 1).to(device).long() for tokens in batched_encodings]
        encodings = [
            {'input_ids': batched_encodings[j], 'attention_mask': batched_attention_masks[j]}
            for j in range(len(batched_encodings))
        ]

    return encodings
