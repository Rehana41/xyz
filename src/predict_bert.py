import torch
from transformers import BertTokenizer, BertConfig, BertModel
import torch.nn as nn

class BERTRegressor(nn.Module):
    def __init__(self):
        super(BERTRegressor, self).__init__()
        self.bert = BertModel(BertConfig())
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return torch.sigmoid(self.out(output).squeeze()) * 10

class EssayScorer:
    def __init__(self, model_path='models/bert_essay_model.pt'):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BERTRegressor()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, essay):
        inputs = self.tokenizer(
            essay,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding='max_length'
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        return round(outputs.item(), 2)

if __name__ == '__main__':
    scorer = EssayScorer()
    sample = "This is a well-structured essay demonstrating high coherence and vocabulary."
    print("Predicted Score:", scorer.predict(sample))
