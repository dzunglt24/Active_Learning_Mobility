import os
from  xml.dom import minidom

import torch
from model import BERT
from transformers import AutoTokenizer

from utils import extract_entities

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define paths
_DATA_VERSION = "36"
_MODEL_VERSION = "Week35"
_DATA_DIR = f"./data/active_data/{_DATA_VERSION}/"
_MODEL_DIR = f"./bert_model/saved_models/{_MODEL_VERSION}"
_SAVE_DIR = f"./data/active_data/{_DATA_VERSION}_xmi/"
os.makedirs(_SAVE_DIR, exist_ok=True)
# Load models
pretrained = "emilyalsentzer/Bio_Discharge_Summary_BERT"
models = []
for entity in ["Action", "Mobility", "Assistance", "Quantification"]:
    bert_path = os.path.join(_MODEL_DIR, f"{entity}/best_model_state.bin")
    model = BERT(num_ner_labels=3, model_name=pretrained)
    model.load_state_dict(torch.load(bert_path))
    model.to(device)
    model.eval()
    models.append(model)

tokenizer = AutoTokenizer.from_pretrained(pretrained)

sent_paths = os.listdir(_DATA_DIR)
for path in sent_paths:
    xmi_path = os.path.join(_SAVE_DIR, path.replace(".txt", ".xmi"))
    f = open(os.path.join(_DATA_DIR, path), "r")
    input_sent = f.read()
    input_sent = input_sent.replace("&#10;", "\n").replace("&#13;", "\n")

    list_entities = []
    for idx, entity in enumerate(["Action", "Mobility", "Assistance", "Quantification"]):
        tag_list = [f"B-{entity}", f"I-{entity}", "O"]

        encoding = tokenizer.encode_plus(
            input_sent,
            add_special_tokens=True,
            max_length=256,
            return_token_type_ids=False,
            padding='longest',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            return_offsets_mapping=True
            )

        # Prepare input then use model to predict
        input_ids = encoding["input_ids"].to(device)
        model = models[idx]
        outputs = model(input_ids)

        tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        offsets = encoding["offset_mapping"][0].detach().cpu().numpy()
        # STEP: Post-process predictions
        _, preds = torch.max(outputs[0], dim=1)
        
        # Merge BPE tokens to word tokens (BPE-Byte Pair Encoding, subword token -> word)
        new_tokens, new_offsets, new_preds = [], [], []
        for (token, offset, pred) in zip(tokens, offsets, preds):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
                new_offsets[-1][1] = offset[1] 
            else:
                if token not in ['[PAD]', '[CLS]', '[SEP]']:
                    new_preds.append(tag_list[pred])
                    new_tokens.append(token)
                    new_offsets.append(offset)

        pred_dict = extract_entities([entity], new_tokens, new_preds, new_offsets)
        
        count_entity = 0
        # Show all extracted entities:
        for k, v in pred_dict.items():
            # Loop through all entity types, check if any types has detected entity
            if v != []:
                for pos in v:
                    # pos[0]: start character index of entity
                    # pos[1]: end character index of entity
                    # use input_sent[pos[0]: pos[1]] to get entity from input sentence
                    print(f"Entity: {k:25}  || Start: {pos[0]:5} || End: {pos[1]:5}  || Text: <{input_sent[pos[0]: pos[1]]}>")
                    count_entity += 1
                    list_entities.append([k, pos[0], pos[1]])

    root = minidom.Document()
    xml = root.createElement('xmi:XMI')
    root.appendChild(xml)
    xml.setAttribute("xmlns:pos", "http:///de/tudarmstadt/ukp/dkpro/core/api/lexmorph/type/pos.ecore")
    xml.setAttribute("xmlns:tcas", "http:///uima/tcas.ecore")
    xml.setAttribute("xmlns:xmi", "http://www.omg.org/XMI")
    xml.setAttribute("xmlns:cas", "http:///uima/cas.ecore")
    xml.setAttribute("xmlns:tcas", "http:///uima/tcas.ecore")
    xml.setAttribute("xmlns:tweet", "http:///de/tudarmstadt/ukp/dkpro/core/api/lexmorph/type/pos/tweet.ecore")
    xml.setAttribute("xmlns:morph", "http:///de/tudarmstadt/ukp/dkpro/core/api/lexmorph/type/morph.ecore")
    xml.setAttribute("xmlns:type", "http:///de/tudarmstadt/ukp/clarin/webanno/api/type.ecore")
    xml.setAttribute("xmlns:dependency", "http:///de/tudarmstadt/ukp/dkpro/core/api/syntax/type/dependency.ecore")
    xml.setAttribute("xmlns:type6", "http:///de/tudarmstadt/ukp/dkpro/core/api/semantics/type.ecore")
    xml.setAttribute("xmlns:type9", "http:///de/tudarmstadt/ukp/dkpro/core/api/transform/type.ecore")
    xml.setAttribute("xmlns:type8", "http:///de/tudarmstadt/ukp/dkpro/core/api/syntax/type.ecore")
    xml.setAttribute("xmlns:type3", "http:///de/tudarmstadt/ukp/dkpro/core/api/metadata/type.ecore")
    xml.setAttribute("xmlns:type10", "http:///org/dkpro/core/api/xml/type.ecore")
    xml.setAttribute("xmlns:type4", "http:///de/tudarmstadt/ukp/dkpro/core/api/ner/type.ecore")
    xml.setAttribute("xmlns:type5", "http:///de/tudarmstadt/ukp/dkpro/core/api/segmentation/type.ecore")
    xml.setAttribute("xmlns:type2", "http:///de/tudarmstadt/ukp/dkpro/core/api/coref/type.ecore")
    xml.setAttribute("xmlns:type7", "http:///de/tudarmstadt/ukp/dkpro/core/api/structure/type.ecore")
    xml.setAttribute("xmlns:constituent", "http:///de/tudarmstadt/ukp/dkpro/core/api/syntax/type/constituent.ecore")
    xml.setAttribute("xmlns:chunk", "http:///de/tudarmstadt/ukp/dkpro/core/api/syntax/type/chunk.ecore")
    xml.setAttribute("xmlns:custom", "http:///webanno/custom.ecore")
    xml.setAttribute("xmi:version", "2.0")

    i = 0
    casChild = root.createElement("cas:NULL")
    casChild.setAttribute("xmi:id", str(i))
    xml.appendChild(casChild)
    i += 1
    casChild = root.createElement("cas:Sofa")
    casChild.setAttribute("xmi:id", str(i))
    casChild.setAttribute("sofaID", "_InitialView")
    casChild.setAttribute("mimeType", "text")
    casChild.setAttribute("sofaString", input_sent)
    xml.appendChild(casChild)
    i += 1
    sentChild = root.createElement("type5:Sentence")
    sentChild.setAttribute("xmi:id", str(i))
    sentChild.setAttribute("sofa", "1")
    sentChild.setAttribute("begin", str(new_offsets[0][0]))
    sentChild.setAttribute("end", str(new_offsets[-1][1]))
    xml.appendChild(sentChild)

    i += 1
    for offset in new_offsets:
        tokChild = root.createElement("type5:Token")
        tokChild.setAttribute("xmi:id", str(i))
        tokChild.setAttribute("sofa", "1")
        tokChild.setAttribute("begin", str(offset[0]))
        tokChild.setAttribute("end", str(offset[1]))
        tokChild.setAttribute("order", "0")
        xml.appendChild(tokChild)
        i += 1

    for e in list_entities:
        entChild = root.createElement(f"custom:Pre_{e[0].lower()}")
        entChild.setAttribute("xmi:id", str(i))
        entChild.setAttribute("sofa", "1")
        entChild.setAttribute("begin", str(e[1]))
        entChild.setAttribute("end", str(e[2]))
        xml.appendChild(entChild)
        i += 1

    casChild = root.createElement("cas:View")
    casChild.setAttribute("sofa", "1")
    list_ids = [str(j) for j in range(i)]
    list_ids = " ".join(list_ids)
    casChild.setAttribute("members", list_ids)

    xml.appendChild(casChild)
    xml_str = root.toprettyxml(indent ="\t")
    with open(xmi_path, "w") as f:
        f.write(xml_str)
    





