from torch import nn
from transformers import AutoModel


class BERT(nn.Module):

    def __init__(self, num_ner_labels, model_name):
        super(BERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        # For NER
        self.ner_dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.ner_output = nn.Linear(self.bert.config.hidden_size, num_ner_labels)

    def forward(self, input_ids=None):
        embeddings = self.bert(input_ids)

        sequence_output = embeddings[0]
        sequence_output = self.ner_dropout(sequence_output)
        logits = self.ner_output(sequence_output)
        return logits
    
    