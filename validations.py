import os
import csv
import json
from glob import glob
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.metrics import \
    precision_recall_fscore_support, \
    accuracy_score

ap = ArgumentParser()
ap.add_argument('--version', type=str, default="v3debug")
args = ap.parse_args()
version = args.version

BASEPATH = "/home/jimena/work/dev/polpostann"
GTFILE = os.path.join(BASEPATH, "ground_truth_v2_400.csv")
OUTPUTSFOLDER = os.path.join(BASEPATH, "outputs", version)
RESULTSFOLDER = os.path.join(BASEPATH, "results", version)
METRICSFOLDER = os.path.join(RESULTSFOLDER, "metrics")
os.makedirs(METRICSFOLDER, exist_ok=True)

MODELS =  [
    os.path.split(model_path)[-1]
    for model_path in glob(os.path.join(OUTPUTSFOLDER, "*"))
]

NBPARAMS = {
    "zephyr-7b-beta": 7,
    "gpt-oss-20b": 20,
    "Ministral-8B-Instruct-2410": 8,
    "Mistral-Small-24B-Instruct-2501": 24,
    "Mistral-Small-3.1-24B-Instruct-2503": 24,
    "Magistral-Small-2506": 24,
    "Mistral-Small-24B-Instruct-2501": 24,
    "max_model_len_8000_Qwen3-30B-A3B-Instruct-2507": 30,
    "gpt-oss-120b": 120,
    "Llama-3.3-70B-Instruct": 70,
    "max_model_len_7000_Llama-3.3-70B-Instruct": 70,
    "DeepSeek-R1-Distill-Llama-70B": 70,
    "Mistral-Large-Instruct-2411": 123,
    "Unknown": -1
}

def family(name):
    for model in NBPARAMS:
        if name.startswith(model):
            return model
    return 'Unknown'

def nbparams(name):
    return NBPARAMS[family(name)]


SUPPORTCHOICES = {
    "multiple": ["Macron", "Mélenchon", "Le Pen", "None"],
    "binary" : ["YES"]
}

NBDECIMALS = 2

ANNOTATIONS = [
    "voteintention/multiple/all",
    # "support/multiple/all",
    # "criticism/multiple/all",
    # "criticism/binary/lepen",
    # "criticism/binary/macron",
    # "criticism/binary/melenchon",
    # "support/binary/lepen",
    # "support/binary/macron",
    # "support/binary/melenchon",
    # "voteintention/binary/lepen",
    # "voteintention/binary/macron",
    # "voteintention/binary/melenchon"
]


for annotation in ANNOTATIONS:

    file = os.path.join(RESULTSFOLDER, f"{annotation.replace('/', '_')}.csv")
    candidate = annotation.split('/')[-1]
    setting = annotation.split('/')[1]
    task = annotation.split('/')[0]
    column = f"{candidate.upper()} {task.upper()}"

    with open(GTFILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        gt = [r[column] for r in reader]

    with open(file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        annotations = [r for r in reader]

    idxs = range(len(gt))

    metrics = []
    for model in MODELS:

        model_abb = model.split('000_')[-1]

        ann = [a[model] for a in annotations]

        # binary classification
        if setting == "binary":
            res = precision_recall_fscore_support(
                y_true=gt,
                y_pred=ann,
                pos_label='YES',
                average='binary',
                zero_division=np.nan)

            res = list(res)
            support =  sum([1 if _ == 'YES' else 0 for _ in gt])

            metrics.append({
                "model": model_abb,
                "kind": kind,
                "params": NBPARAMS[model],
                "precision": str(res[0])[:NBDECIMALS + 2],
                "recall": str(res[1])[:NBDECIMALS + 2],
                "f1_weighted": str(res[2])[:NBDECIMALS + 2],
                "support": support,
                "labels": ' | '.join(SUPPORTCHOICES[setting])
                })

        # multiclass classification
        if setting == "multiple":
            # See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

            acc = accuracy_score(
                y_true=gt,
                y_pred=ann,
                normalize=True)

            # binary
            # Reports results for the classes specified by labels
            res = precision_recall_fscore_support(
                y_true=gt,
                y_pred=ann,
                labels=SUPPORTCHOICES['multiple'],
                average=None,
                zero_division=np.nan)

            # 'macro'
            # Calculate metrics for each label, and find their unweighted mean.
            # This does not take label imbalance into account.
            f1_macro = precision_recall_fscore_support(
                y_true=gt,
                y_pred=ann,
                labels=SUPPORTCHOICES['multiple'],
                average='macro',
                zero_division=np.nan)[2]

            # 'micro'
            # Calculate metrics globally by counting the total true positives,
            # false negatives and false positives
            f1_micro = precision_recall_fscore_support(
                y_true=gt,
                y_pred=ann,
                labels=SUPPORTCHOICES['multiple'],
                average='micro',
                zero_division=np.nan)[2]

            # 'weighted'
            # Calculate metrics for each label,
            # and find their average weighted by support (the number of true instances for each label).
            # This alters ‘macro’ to account for label imbalance;
            # it can result in an F-score that is not between precision and recall.
            f1_weighted = precision_recall_fscore_support(
                y_true=gt,
                y_pred=ann,
                labels=SUPPORTCHOICES['multiple'],
                average='weighted',
                zero_division=np.nan)[2]

            metrics.append({
                "model": model_abb,
                "family": family(model_abb),
                "params": nbparams(model),
                "support": ' | '.join(map(str, res[3])),
                "labels": ' | '.join(SUPPORTCHOICES[setting]),
                "accuracy": acc,
                "precision": ' | '.join([str(r)[:NBDECIMALS + 2] for r in res[0]]),
                "recall": ' | '.join([str(r)[:NBDECIMALS + 2] for r in res[1]]),
                "f1_binary": ' | '.join([str(r)[:NBDECIMALS + 2] for r in res[2]]),
                "f1_macro": str(f1_macro)[:NBDECIMALS + 2],
                "f1_micro":  str(f1_micro)[:NBDECIMALS + 2],
                "f1_weighted": str(f1_weighted)[:NBDECIMALS + 2],
                })


    df = pd.DataFrame.from_records(metrics).sort_values(by=["params"])
    path = os.path.join(METRICSFOLDER, f"{annotation.replace('/', '_')}.csv")
    df.to_csv(path, index=False)
    print(f"Metrics for annotation {annotation} experiment saved at {path}\n")

    # cmd = f"xan select model,params,labels,support,accuracy,f1_macro,f1_binary {path} | xan sort -s f1_macro | xan v"
    # print(cmd)
    # os.system(cmd)

    cmd = f"xan groupby family --along-cols params,accuracy,f1_macro 'mean(_)' {path} | xan v"
    print(cmd)
    os.system(cmd)

