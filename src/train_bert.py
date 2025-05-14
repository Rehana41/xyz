import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import numpy as np

class EssayDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_len):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        score = self.scores[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(score, dtype=torch.float)
        }

class BERTRegressor(nn.Module):
    def __init__(self):
        super(BERTRegressor, self).__init__()
        self.bert = BertModel(BertConfig())  # From scratch
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return torch.sigmoid(self.out(output).squeeze()) * 10

def create_data_loaders(df, tokenizer, max_len, batch_size):
    dataset = EssayDataset(
        texts=df['essay'].to_numpy(),
        scores=df['score'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    losses = []
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids, attention_mask)
            predictions.extend(outputs.cpu().numpy().tolist())
            actuals.extend(targets.cpu().numpy().tolist())

    mse = mean_squared_error(actuals, predictions)
    return mse, predictions, actuals

def run_training(data_path='data/essays.csv', model_path='models/bert_essay_model.pt'):
    df = pd.read_csv(data_path)[['essay', 'score']].dropna()
    df['score'] = df['score'].astype(float)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    MAX_LEN = 512
    BATCH_SIZE = 4
    EPOCHS = 3
    LR = 2e-5

    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)

    train_loader = create_data_loaders(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_loader = create_data_loaders(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTRegressor().to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_mse, _, _ = eval_model(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}, Val MSE: {val_mse:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == '__main__':
    run_training()
