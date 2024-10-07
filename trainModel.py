import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_


# Conditional imports for AMP based on CUDA availability
use_cuda = torch.cuda.is_available()
if use_cuda:
    from torch.cuda.amp import GradScaler, autocast
    print("Using CUDA")

class SQLDataset(Dataset):
    def __init__(self, tokenizer, file_path='CodingBot/trainingData.json', max_len=512):
        self.max_len = max_len
        self.tokenizer = tokenizer
        try:
            with open(file_path, 'r') as file:
                self.data = json.load(file)
        except Exception as e:
            raise IOError(f"Error opening {file_path}: {str(e)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source = self.tokenizer.encode_plus(
            item['prompt'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        target = self.tokenizer.encode_plus(
            item['sql'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': source['input_ids'].squeeze(),  # Use squeeze to remove unnecessary dimensions
            'attention_mask': source['attention_mask'].squeeze(),
            'labels': target['input_ids'].squeeze()
        }


tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

dataset = SQLDataset(tokenizer)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device('cuda' if use_cuda else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(loader)*3)

scaler = torch.amp.GradScaler(device=device) if use_cuda else None

model.train()
for epoch in range(3):
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

model.save_pretrained('CodingBot/fine_tuned_t5_sql')
tokenizer.save_pretrained('CodingBot/fine_tuned_t5_sql')
