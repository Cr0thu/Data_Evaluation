import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset
import sys
from tqdm import tqdm

def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# Load the IMDB dataset
dataset = load_dataset('imdb', split = 'test')

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True, torch_dtype=torch.bfloat16, num_labels = 2)
model = model.cuda()

max_length = 2048

all_embeddings = []
all_labels = []

with torch.no_grad():
    for example in tqdm(dataset):
        text = example['text']
        label = example['label']
        inputs = tokenizer(text, max_length=max_length, padding=True, truncation=True, return_tensors='pt')

        inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = model(**inputs)
        embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())
        all_labels.append(label)

all_embeddings = torch.cat(all_embeddings, dim=0)
all_labels = torch.tensor(all_labels)

print(f'Generated embeddings shape: {all_embeddings.shape}')
# print(all_labels[:20])
torch.save(all_embeddings, "/gfshome/IMDB_test_embeddeings.pt")
torch.save(all_labels, '/gfshome/IMDB_test_labels.pt')