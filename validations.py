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

ANNOTATIONS = [

    # multiclass
    "voteintention/multiple/all",
    "support/multiple/all",
    "criticism/multiple/all",

    # binary criticism
    "criticism/binary/lepen",
    "criticism/binary/macron",
    "criticism/binary/melenchon",

    # binary support
    "support/binary/lepen",
    "support/binary/macron",
    "support/binary/melenchon",

    # binary voteintention
    "voteintention/binary/lepen",
    "voteintention/binary/macron",
    "voteintention/binary/melenchon",
]

BASEPATH = "/home/jimena/work/dev/polpostann"
DEFAULTGTFILE = os.path.join(BASEPATH, "ground_truth_v4_400.csv")
DEFAULTGTINDEX = "idx_all"

ap = ArgumentParser()
ap.add_argument('--version', type=str, default="v3ModelSelectionfrench")
ap.add_argument('--gt_file', type=str, default=DEFAULTGTFILE)
ap.add_argument('--gt_index', type=str, default=DEFAULTGTINDEX)
ap.add_argument('--annotation', type=str, default=ANNOTATIONS[0], choices=ANNOTATIONS)
ap.add_argument('--server', type=str, default='in2p3')
ap.add_argument('--doparsing', action='store_true')
ap.add_argument('--language',type=str, default='french')
ap.add_argument('--filename', type=str, default="llm_answer_0.csv")
args = ap.parse_args()
version = args.version
annotation = args.annotation
server = args.server
filename = args.filename
doparsing = args.doparsing
language = args.language
gt_file = args.gt_file
gt_index = args.gt_index

OUTPUTSFOLDER = os.path.join(BASEPATH, f"outputs_{server}", version)

RESULTSFOLDER = os.path.join(BASEPATH, "results", version)
os.makedirs(RESULTSFOLDER, exist_ok=True)

METRICSFOLDER = os.path.join(RESULTSFOLDER, "metrics")
os.makedirs(METRICSFOLDER, exist_ok=True)

parameters = vars(args)
parameters.update({
    "OUTPUTSFOLDER": OUTPUTSFOLDER,
    "RESULTSFOLDER": RESULTSFOLDER,
    "METRICSFOLDER": METRICSFOLDER,
})
dumped_parameters = json.dumps(parameters, sort_keys=True, indent=4)
print("---------------------------------------------------------")
print(f"PARAMETERS:\n{dumped_parameters[2:-2]}")
print("---------------------------------------------------------")

# hard core :S
PATTERN = os.path.join(OUTPUTSFOLDER, "*", "guided", annotation, filename)
MODELS =  [
    model_path.split("/")[8]
    for model_path in glob(PATTERN)
]

if len(MODELS) == 0:
    print(f"Didn't find any result at pattern {PATTERN}")
    exit()

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


CHOICES = {
    "multiple": ["Macron", "Mélenchon", "LePen", "None"],
    "binary" : ["OUI", "NON"]
}

SUPPORTCHOICES = {
    "french": {
        "multiple": ["Macron", "Mélenchon", "LePen", "None"],
        "binary" : ["OUI", "NON"]
    },
    "english": {
        "multiple": ["Macron", "Mélenchon", "LePen", "None"],
        "binary" : ["YES", "NO"]
    },
}

BINARYPOSLABEL = {
    "french": "OUI",
    "english": "YES",
}

BINARYMAP = {
    "french": {"YES": "OUI", "NO": "NON"},
    "english": {"YES": "YES", "NO": "NO"},
}

NBDECIMALS = 2

SETTINGS = ["binary", "multiple"]

def parseAnwers(whole_answer):
    return whole_answer.replace(" ", "")

def extract_data(model, annotation, df=None, columns=[]):

    setting = annotation.split('/')[1]

    assert model in MODELS
    assert setting in SETTINGS

    folders = glob(os.path.join(OUTPUTSFOLDER, model))
    assert len(folders) == 1
    folder = folders[0]

    # get answers
    colname = model
    path_guided = os.path.join(folder, "guided", annotation, filename)
    df_guided = pd.read_csv(path_guided) \
        .fillna("None") \
        .rename(columns={"answer": colname})

    df_guided[colname] = df_guided[colname].apply(lambda a: parseAnwers(a))
    print(f"Answers loaded from {path_guided}")

    if df is not None:
        df = df.merge(df_guided, on=['idx', 'tweet'])
    else:
        df = df_guided

    columns.extend([model])

    return df, columns

def getPath(annotation):
    return os.path.join(RESULTSFOLDER, f"{annotation.replace('/', '_')}.csv")

def parseAndJoin(annotation):

    setting = annotation.split('/')[1]

    df = None
    columns = ["idx", "tweet"]

    for model in MODELS:
        df, columns = extract_data(model, annotation, df, columns)

    path = getPath(annotation)

    df[columns].to_csv(path, index=False)

    print(f"Results for annotation {annotation} results saved at {path}")
    # os.system(f"xan shuffle {path} | xan slice -l 5 | xan v")


def computeValidationMetrics(annotation):

    file = getPath(annotation)

    candidate = annotation.split('/')[-1]
    setting = annotation.split('/')[1]
    task = annotation.split('/')[0]
    column = f"{candidate.upper()} {task.upper()}"
    print(f"Using column {column} for ground truth")

    ground_truth = pd.read_csv(gt_file, dtype=str, keep_default_na=False, na_values=['NaN'])
    ground_truth = ground_truth[["idx_all", "idx", "english", "french", column]]
    ground_truth[column] = ground_truth[column].apply(lambda a: parseAnwers(a))
    assert ground_truth.isna().sum().sum() == 0

    allannotations = pd.read_csv(file, dtype=str,  keep_default_na=False, na_values=['NaN'])
    assert allannotations.isna().sum().sum() == 0

    annotations = ground_truth.merge(allannotations, left_on=gt_index, right_on='idx', how='left')
    assert len(annotations) == len(ground_truth)

    for model in MODELS:
        annotations[column] = annotations[column].apply(lambda a: parseAnwers(a))

    if annotations['idx_all'].isna().sum() > 0:
        raise ValueError(f"There are NAN indexes:{annotations[annotations['idx_all'].isna()]}")

    gtpath = os.path.join(RESULTSFOLDER, f"{annotation.replace('/', '_')}_with_ground_truth.csv")
    annotations.drop(columns=["tweet"], inplace=True)
    annotations.to_csv(gtpath, index=False)
    print(f"Results for annotation with ground_truth saved at {gtpath}")

    metrics = []
    for model in MODELS:

        model_abb = model.split('000_')[-1]

        if annotations[model].isna().sum() > 0:
            print(f"Removing NAN annotations:{annotations[annotations[model].isna()]}")

        ann = annotations[model][~annotations[model].isna()].tolist()
        gt = ground_truth[~annotations[model].isna()][column].tolist()

        # binary classification
        if setting == "binary":

            # map gt answers to language (this is the same for english)
            gt = [BINARYMAP[language][g] for g in gt]

            P = [sum([a == SUPPORTCHOICES[language][setting][0] for a in ann])]
            TP = [sum([g == SUPPORTCHOICES[language][setting][0] for g in gt])]

            acc = accuracy_score(
                y_true=gt,
                y_pred=ann,
                normalize=True)

            res = precision_recall_fscore_support(
                y_true=gt,
                y_pred=ann,
                pos_label=BINARYPOSLABEL[language],
                average='binary',
                zero_division=np.nan)

            res = list(res)


            metrics.append({
                "model": model_abb,
                "family": family(model_abb),
                "version": version,
                "task": task,
                "params": nbparams(model),
                "TP": str(TP[0]),
                "P": str(P[0]),
                "accuracy": acc,
                "precision": str(res[0])[:NBDECIMALS + 2],
                "recall": str(res[1])[:NBDECIMALS + 2],
                "f1": str(res[2])[:NBDECIMALS + 2],
                })

        # multiclass classification
        if setting == "multiple":
            # See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

            positives = [sum([a == l for a in ann]) for l in SUPPORTCHOICES[language]['multiple']]

            acc = accuracy_score(
                y_true=gt,
                y_pred=ann,
                normalize=True)

            # binary
            # Reports results for the classes specified by labels
            res = precision_recall_fscore_support(
                y_true=gt,
                y_pred=ann,
                labels=SUPPORTCHOICES[language]['multiple'],
                average=None,
                zero_division=np.nan)

            # 'macro'
            # Calculate metrics for each label, and find their unweighted mean.
            # This does not take label imbalance into account.
            f1_macro = precision_recall_fscore_support(
                y_true=gt,
                y_pred=ann,
                labels=SUPPORTCHOICES[language]['multiple'],
                average='macro',
                zero_division=np.nan)[2]

            # 'micro'
            # Calculate metrics globally by counting the total true positives,
            # false negatives and false positives
            f1_micro = precision_recall_fscore_support(
                y_true=gt,
                y_pred=ann,
                labels=SUPPORTCHOICES[language]['multiple'],
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
                labels=SUPPORTCHOICES[language]['multiple'],
                average='weighted',
                zero_division=np.nan)[2]

            metrics.append({
                "model": model_abb,
                "family": family(model_abb),
                "version": version,
                "task": task,
                "params": nbparams(model),
                "TP": ' | '.join(map(str, res[3])),
                "P": ' | '.join(map(str, positives)),
                "labels": ' | '.join(SUPPORTCHOICES[language][setting]),
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

    if setting == "binary":
        cmd = f"xan select model,version,TP,P,precision,recall,accuracy,f1 {path} | xan v -I"
        os.system(cmd)

    if setting == "multiple":
        cmd = f"xan select model,labels,TP,P {path} | xan sort -R -s model | xan v -I"
        os.system(cmd)

        cmd = f"xan select model,precision,recall,accuracy,f1_binary,f1_macro,f1_micro {path} | xan sort -R -s model | xan v -I"
        os.system(cmd)

        cmd = f"xan groupby family --along-cols accuracy,f1_macro 'mean(_)' {path} | xan map '\"{version}\" as cross_validation' | xan sort -R -s family | xan v -I -S 4"
        os.system(cmd)

# parse and join results
if doparsing:
    parseAndJoin(annotation)

# make validations
computeValidationMetrics(annotation)


