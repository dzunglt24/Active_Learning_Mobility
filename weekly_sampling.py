import os
import pandas as pd
import argparse as args
from ast import arg, literal_eval
from utils import calc_information_density, calc_agree_density

parser = args.ArgumentParser()
parser.add_argument("--data_dir", default="./data/active_data/")
parser.add_argument("--round", type=str, required=True)
parser.add_argument("--num_sample", type=int, default=100)
parser.add_argument("--entity", type=str, default="Action")

args = parser.parse_args()

def calc_score(row, batch):
    # keep score of previous round is < 0
    if row["batch_id"] <= batch and row["batch_id"] > 0:
        return -1
    else:
        return calc_information_density(row["bert_tag"], row["crf_tag"], row["density"])

def agree_score(row, batch):
    # keep score of previous round is < 0
    if row["batch_id"] <= batch and row["batch_id"] > 0:
        return -1
    else:
        return calc_agree_density(row["bert_tag"], row["crf_tag"], row["density"])
    
if __name__ == "__main__":
    data_path = os.path.join(args.data_dir, f"round_{args.round}_{args.ent}.csv")
    df = pd.read_csv(data_path, index_col=0)
    df["bert_tag"] = df["bert_tag"].apply(literal_eval)
    df["crf_tag"] = df["crf_tag"].apply(literal_eval)
    # Remember: round 0 => batch_id 1
    batch = int(args.round) + 1
    df["score"] = df.apply(lambda row: calc_score(row, batch), axis=1)
    df["agree_score"] = df.apply(lambda row: agree_score(row, batch), axis=1)
    print(df.head(10))
    
    # Keep this number to continue indexing new picked sentences
    current_sent_index = df["sent_num_index"].max() + 3400
    print(f"Max sentence index: {current_sent_index}")
    print(f"Number of total sentences: {df.shape[0]}")
    print(f"Number of total sentences in previous batches: {df[df['batch_id'] < batch].shape[0]}")
    df = df[df["sent"].str.split().str.len() > 4]
    df = df[df["batch_id"] == 0].sort_values(by=["score"], ascending=False).reset_index()
    print(f"Number of sentences in unlabeled pool: {df.shape[0]}")
    
    txt_dir = os.path.join(args.data_dir, str(batch) + f"_{args.ent}")


    if not os.path.isdir(txt_dir):
        os.mkdir(txt_dir)

    num_sample = args.num_sample
    if df.shape[0] < num_sample:
        num_sample = df.shape[0]
    for i in range(0, num_sample):
        # print(i + 1 + current_sent_index)
        df.loc[i, "sent_num_index"] = i + 1 + current_sent_index
        df.loc[i, "batch_id"] = batch
        f = open(os.path.join(txt_dir, str(i + 1 + current_sent_index).zfill(4) + ".txt"), "w")
        f.write(df.loc[i, "sent"])
        f.close()
    
    # Get more 25 examples that BERT and CRF produce same results and much entities
    current_sent_index += num_sample
    df = df.sort_values(by=["agree_score"], ascending=False).reset_index(drop=True)
    for i in range(25):
        # print(i + 1 + current_sent_index)
        df.loc[i, "sent_num_index"] = i + 1 + current_sent_index
        df.loc[i, "batch_id"] = batch
        f = open(os.path.join(txt_dir, str(i + 1 + current_sent_index).zfill(4) + ".txt"), "w")
        f.write(df.loc[i, "sent"])
        f.close()


    df.to_csv(os.path.join(args.data_dir, "round_" + args.round + f"_indexing_{args.ent}.csv"))
    print(df.head(10))

