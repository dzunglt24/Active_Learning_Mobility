import numpy as np
import torch
from seqeval.metrics import classification_report
from torch import nn, autocast

def get_labels(predictions, references, device):
    if device.type == "cpu":
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

    prediction_ids = [
        [p for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]

    label_ids = [
        [l for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]

    return prediction_ids, label_ids

def get_preds(predictions, ner_label_list, device):
    if device.type == "cpu":
        y_pred = predictions.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()

    true_predictions = [ner_label_list[p] for p in y_pred[0]]

    return true_predictions


def get_true_labels(prediction_ids, label_ids, ner_label_list):
    true_predictions = [ner_label_list[p] for p in prediction_ids]
    true_labels = [ner_label_list[l] for l in label_ids]
    return true_predictions, true_labels


def custom_loss(num_ner_labels, device):
    ner_weights = num_ner_labels * [1]
    ner_weights[0] = 0.1#∫1
    loss_ner_fn = nn.CrossEntropyLoss(
        weight=torch.cuda.FloatTensor(ner_weights)).to(device)
    loss_ner_fn.ignore_index = -100
    return loss_ner_fn


def train_epoch(
    model,
    data_loader,
    loss_ner_fn,
    optimizer,
    ner_label_list,
    device,
    scheduler
):
    model = model.train()
    losses = []
    ner_correct_predictions = 0
    len_ner_predict = 0
    pred_list = []
    label_list = []

    for d in data_loader:
        optimizer.zero_grad()
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)

        ner_outputs = model(input_ids=input_ids)
        
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_outputs.view(-1, ner_outputs.shape[2])
            active_labels = torch.where(
                active_loss, targets.view(-1), torch.tensor(
                    loss_ner_fn.ignore_index).type_as(targets)
            )
            loss = loss_ner_fn(active_logits, active_labels)
        else:
            loss =  loss_ner_fn(ner_outputs.view(-1, ner_outputs.shape[2]), targets.view(-1))

        ner_preds = torch.argmax(ner_outputs, dim=-1)
        ner_pred_ids, ner_label_ids = get_labels(ner_preds, targets, device)

        for (ner_pred, ner_label) in zip(ner_pred_ids, ner_label_ids):
            ner_correct_predictions += torch.sum(torch.tensor(ner_pred) == torch.tensor(ner_label))
            len_ner_predict += len(ner_pred)

            preds, labels = get_true_labels(ner_pred, ner_label, ner_label_list)
            pred_list.append(preds)
            label_list.append(labels)


        losses.append(loss.item())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # report = classification_report(label_list, pred_list, mode='strict', output_dict=True)
    return ner_correct_predictions.double()/len_ner_predict, np.mean(losses)


def eval_epoch(model,
               data_loader,
               loss_ner_fn,
               ner_label_list,
               device):
    model = model.eval()
    losses = []

    ner_correct_predictions = 0
    len_ner_predict = 0
    pred_list = []
    label_list = []


    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)

        ner_outputs = model(input_ids=input_ids)

        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_outputs.view(-1, ner_outputs.shape[2])
            active_labels = torch.where(
                active_loss, targets.view(-1), torch.tensor(
                    loss_ner_fn.ignore_index).type_as(targets)
            )
            loss = loss_ner_fn(active_logits, active_labels)
        else:
            loss = loss_ner_fn(ner_outputs.view(-1, ner_outputs.shape[2]), targets.view(-1))

        ner_preds = torch.argmax(ner_outputs, dim=-1)
        ner_pred_ids, ner_label_ids = get_labels(ner_preds, targets, device)
        for (ner_pred, ner_label) in zip(ner_pred_ids, ner_label_ids):
            ner_correct_predictions += torch.sum(torch.tensor(ner_pred) == torch.tensor(ner_label))
            len_ner_predict += len(ner_pred)

            preds, labels = get_true_labels(ner_pred, ner_label, ner_label_list)
            pred_list.append(preds)
            label_list.append(labels)
        losses.append(loss.item())
        
    report = classification_report(label_list, pred_list, mode='strict', output_dict=True)

    return ner_correct_predictions.double()/len_ner_predict, np.mean(losses), report["micro avg"]["precision"], report["micro avg"]["recall"], report["micro avg"]["f1-score"], report["micro avg"]["support"]


def extract_entities(label_list, tokens, tags, offsets):
    pred_dict = {k:[] for k in label_list}
    prev_tag = "O"
    curr_ent = [0, 0]
    for idx, (token, tag, offset) in enumerate(zip(tokens, tags, offsets)):
        if tag.startswith("B-"):
            if prev_tag == "O":
                curr_ent = offset.tolist()
            else:
                if curr_ent != [0, 0]:
                    pred_dict[prev_tag[2:]].append(curr_ent)
                curr_ent = offset.tolist() 
            
            prev_tag = tag

        elif tag.startswith("I-"):
            if tag[2:] == prev_tag[2:]:
                if curr_ent != [0, 0]:
                    
                    curr_ent[1] = offset[1] 
            else:
                if prev_tag != "O" and curr_ent != [0, 0]:
                    pred_dict[prev_tag[2:]].append(curr_ent)
                    curr_ent = [0, 0] 
            prev_tag = tag
        else:
            if prev_tag != "O" and curr_ent != [0, 0]:
                pred_dict[prev_tag[2:]].append(curr_ent)
                curr_ent = [0, 0]
            prev_tag = "O"
        
        if idx == len(tokens) - 1:
            if curr_ent != [0, 0]:
                pred_dict[prev_tag[2:]].append(curr_ent)
        
    return pred_dict


def train_epoch_amp(
    model,
    data_loader,
    loss_ner_fn,
    optimizer,
    ner_label_list,
    device,
    scheduler, 
    scaler,
):
    model = model.train()
    losses = []
    ner_correct_predictions = 0
    len_ner_predict = 0
    pred_list = []
    label_list = []

  
    for d in data_loader:
        optimizer.zero_grad()
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)

        with autocast(device_type="cuda"):
            ner_outputs = model(input_ids=input_ids)
            
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = ner_outputs.view(-1, ner_outputs.shape[2])
                active_labels = torch.where(
                    active_loss, targets.view(-1), torch.tensor(
                        loss_ner_fn.ignore_index).type_as(targets)
                )
                loss = loss_ner_fn(active_logits, active_labels)
            else:
                loss =  loss_ner_fn(ner_outputs.view(-1, ner_outputs.shape[2]), targets.view(-1))

        ner_preds = torch.argmax(ner_outputs, dim=-1)
        ner_pred_ids, ner_label_ids = get_labels(ner_preds, targets, device)

        for (ner_pred, ner_label) in zip(ner_pred_ids, ner_label_ids):
            ner_correct_predictions += torch.sum(torch.tensor(ner_pred) == torch.tensor(ner_label))
            len_ner_predict += len(ner_pred)

            preds, labels = get_true_labels(ner_pred, ner_label, ner_label_list)
            pred_list.append(preds)
            label_list.append(labels)


        losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    # report = classification_report(label_list, pred_list, mode='strict', output_dict=True)
    return ner_correct_predictions.double()/len_ner_predict, np.mean(losses)