import os
import argparse
import pandas as pd
import torch
from nltk.tag.stanford import StanfordNERTagger
from tqdm import tqdm
from transformers import AutoTokenizer

from bert_model.model import BERT

parser = argparse.ArgumentParser()
parser.add_argument("--week", type=str, required=True)
parser.add_argument("--entity", type=str, default="Action", required=True)
args = parser.parse_args()

# Configure java path
JAVA_PATH = "/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.292.b10-1.el7_9.x86_64"
os.environ['JAVA_HOME'] = JAVA_PATH
# Check GPU, if not available use cpu instead
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_dir = "/home/projects/Active_Learning_Mobility/"

_DATA_VERSION = "Week" + args.week
_ENT = args.ent
pool_csv = root_dir + f"data/active_data/round_{str(int(args.week) - 1)}_indexing.csv"

df = pd.read_csv(pool_csv, keep_default_na=False)
df = df.drop(['Unnamed: 0'], axis=1)
df["bert_tag"] = None
df["crf_tag"] = None
df["diff_count"] = 0 
df["length"] = 0 
df = df[df.batch_id == 0].reset_index(drop=True)

jar = root_dir + "crf_model/stanford-ner-2020-11-17/stanford-ner.jar"
crf_path = root_dir + f"crf_model/saved_models/{_DATA_VERSION}/{_ENT.lower()}.ser.gz"
pretrained = "emilyalsentzer/Bio_Discharge_Summary_BERT"
# pretrained = "bert-base-uncased"

bert_path = root_dir + f"bert_model/saved_models/{_DATA_VERSION}/{_ENT}/best_model_state.bin"
tag_list = [f"B-{_ENT}", f"I-{_ENT}", "O"]

ner_tagger = StanfordNERTagger(crf_path, jar, encoding='utf8')

tokenizer = AutoTokenizer.from_pretrained(pretrained)
model = BERT(num_ner_labels=3, model_name=pretrained)
model.load_state_dict(torch.load(bert_path))
model.to(device)
model.eval()

for i in tqdm(range(df.shape[0])):
    sentence = df.loc[i, "sent"]
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=256,
        return_token_type_ids=False,
        padding='longest',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
        )
    input_ids = encoding["input_ids"].to(device)
    outputs = model(input_ids)
    _, preds = torch.max(outputs[0], dim=1)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])

    new_tokens, bert_outputs = [], []
    for token, tag_idx in zip(tokens, preds):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            if token not in ['[PAD]', '[CLS]', '[SEP]']:
                bert_outputs.append(tag_list[tag_idx])
                new_tokens.append(token)    
    
    crf_outputs =  ner_tagger.tag(new_tokens)
    crf_outputs = [out[1] for out in crf_outputs]
    diff = [1 if bo != co else 0 for bo, co in zip(bert_outputs, crf_outputs)]
    length = len(crf_outputs)    
        
    df.at[i, "bert_tag"]  = bert_outputs
    df.at[i, "crf_tag"]  = crf_outputs
    df.loc[i, "diff_count"]  = sum(diff)
    df.loc[i, "length"]  = length

df.to_csv(f"./data/{_DATA_VERSION}_{_ENT}.csv", index=False)
