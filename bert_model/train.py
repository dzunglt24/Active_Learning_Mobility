import argparse
import os

import numpy as np
import torch
from datasets import ClassLabel, load_dataset
from model import BERT
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (AdamW, AutoTokenizer, default_data_collator,
                          get_linear_schedule_with_warmup)
from torch.cuda.amp import GradScaler

from utils import custom_loss, eval_epoch, train_epoch_amp

parser = argparse.ArgumentParser()
parser.add_argument("--week", type=str, required=True)
parser.add_argument("--entity", type=str, required=True)
parser.add_argument("--model_name", type=str, default="emilyalsentzer/Bio_Discharge_Summary_BERT")
parser.add_argument("--save_dir", type=str, default="saved_models")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--early_stop", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--warm_up_steps", type=int, default=0)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--padding", type=str, default="max_length")
parser.add_argument("--lr", type=float, default=5e-6)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cv", type=int, default=0)


args = parser.parse_args()

SAVE_DIR = os.path.join(args.save_dir , args.week, args.entity)
if not os.path.isdir(os.path.join(args.save_dir, args.week)):
    os.mkdir(os.path.join(args.save_dir, args.week))
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

if args.cv == 0:
    CROSS_VALIDATION = False
    NUM_FOLD = 1
else:
    CROSS_VALIDATION = True
    NUM_FOLD = 5

np.random.seed(args.seed)
torch.manual_seed(args.seed)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        max_length=args.max_length,
        padding=args.padding,
        truncation=True,
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

fold_results = []

for fold in range(NUM_FOLD):
    # LOAD DATA
    if not CROSS_VALIDATION:
        raw_datasets = load_dataset("load_data/load_mf.py", name=args.entity, week=args.week, data_dir="../../data/NER/")
    else:
        raw_datasets = load_dataset("load_data/load_mf.py", name=args.entity, has_dev=True, week="fold" + str(fold) , data_dir="../../data/NER/final_dataset/")
    print(raw_datasets)
    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    elif raw_datasets["validation"] is not None:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features
    else: 
        column_names = raw_datasets["test"].column_names
        features = raw_datasets["test"].column_names
    text_column_name = "tokens"
    label_column_name = "ner_tags"

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}

    num_ner_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BERT(num_ner_labels=num_ner_labels, model_name=args.model_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels, batched=True, remove_columns=['id', 'ner_tags', 'tokens']
    )

    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]
    test_dataset = processed_raw_datasets["test"]
    data_collator = default_data_collator

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size  
    )
    eval_dataloader = DataLoader(eval_dataset,
                                collate_fn=data_collator,
                                batch_size=args.batch_size  
                                )
    test_dataloader = DataLoader(test_dataset,
                                collate_fn=data_collator,
                                batch_size=args.batch_size  
                                )
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)  
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=total_steps
    )

    loss_ner_fn = custom_loss(num_ner_labels, device)

    best_f1_score = 0
    early_stopping = 0
    best_results = []

    scaler = GradScaler(enabled=True)
    for epoch in tqdm(range(args.epochs)):
        ner_acc, train_loss = train_epoch_amp(
            model,
            train_dataloader,
            loss_ner_fn,
            optimizer,
            label_list,
            device,
            scheduler,
            scaler
        )
        ner_val_acc, val_loss, precision, recall, f1_score, support = eval_epoch(
            model,
            eval_dataloader,
            loss_ner_fn,
            label_list,
            device
        )

        if f1_score > best_f1_score:
            early_stopping = 0
            if not os.path.isdir(os.path.join(SAVE_DIR)):
                os.mkdir(os.path.join(SAVE_DIR))
            if not CROSS_VALIDATION:
                if args.seed == 42:
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model_state.bin"))
            best_f1_score = f1_score

            ner_test_acc, test_loss, test_precision, test_recall, test_f1_score, test_support = eval_epoch(
                model,
                test_dataloader,
                loss_ner_fn,
                label_list,
                device
            )

            best_results = [test_precision, test_recall, test_f1_score]
            print(f"Epoch {epoch} - fold{fold} - P: {test_precision:.4f}, R: {test_recall:.4f}, F1: {test_f1_score:.4f}, Support: {test_support} ")
        else:
            early_stopping += 1
        if early_stopping == args.early_stop:
            break
    
    fold_results.append(best_results)
    print(f"Fold {str(fold)}: Best validate F1_score: {best_f1_score:.4f}")
    print(f"Best test results: {best_results}")
    
print(fold_results)
