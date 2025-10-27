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
    # multiclass annotations
    "voteintention/multiple/all",
    "support/multiple/all",
    "criticism/multiple/all",
    # binary annotations
    "criticism/binary/lepen",
    "criticism/binary/macron",
    "criticism/binary/melenchon",
    "support/binary/lepen",
    "support/binary/macron",
    "support/binary/melenchon",
    "voteintention/binary/lepen",
    "voteintention/binary/macron",
    "voteintention/binary/melenchon",
]

BASEPATH = "/home/jimena/work/dev/polpostann"
GTFILE = os.path.join(BASEPATH, "ground_truth_v2_400.csv")

ap = ArgumentParser()
ap.add_argument('--version', type=str, default="v3ModelSelectionfrench")
ap.add_argument('--annotation', type=str, default=ANNOTATIONS[0], choices=ANNOTATIONS)
args = ap.parse_args()
version = args.version
annotation = args.annotation

OUTPUTSFOLDER = os.path.join(BASEPATH, "outputs", version)

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


CHOICES = {
    "multiple": ["Macron", "Mélenchon", "Le Pen", "None"],
    "binary" : ["YES", "NO"]
}

SUPPORTCHOICES = CHOICES

DEFAULTANSWER = {
    "multiple": "None",
    "binary" : "NO"
}

NBDECIMALS = 2

SETTINGS = ["binary", "multiple"]

def parseAnwers(whole_answer, model, setting):

    whole_answer = whole_answer \
        .strip() \
        .split('\n')[0] \
        .replace("(", "") \
        .replace(")", "") \
        .replace(":", "") \
        .replace("'", "") \
        .replace('"', "") \
        .replace(".", "") \
        .replace("·", "") \
        .replace(",", "")

    if "gpt-oss" in model:
        whole_answer = whole_answer.split("assistantfinal")[-1]

    if setting == "binary":
        whole_answer = whole_answer.upper()

    annotation = DEFAULTANSWER[setting]
    for choice in CHOICES[setting]:
        if whole_answer[:len(choice)] == choice:
            annotation = choice

    return annotation

def extract_data(model, annotation, df=None, columns=[]):

    setting = annotation.split('/')[1]

    assert model in MODELS
    assert setting in SETTINGS

    folders = glob(os.path.join(OUTPUTSFOLDER, model))
    assert len(folders) == 1
    folder = folders[0]

    # get answers
    colname = model
    path_guided = os.path.join(folder, "guided", annotation, "llm_answer_0.csv")
    df_guided = pd.read_csv(path_guided) \
        .fillna("None") \
        .rename(columns={"answer": colname})

    df_guided[colname] = df_guided[colname].apply(lambda a: parseAnwers(a, model, setting))

    if df is not None:
        df = df.merge(df_guided, on=['idx', 'tweet'])
    else:
        df = df_guided

    columns.extend([model])

    return df, columns

def parseAndJoin(annotation):

    setting = annotation.split('/')[1]

    df = None
    columns = ["idx", "tweet"]

    for model in MODELS:
        df, columns = extract_data(model, annotation, df, columns)

    df = df.assign(guidedMayorityVote=df[df.columns[2:]].mode(axis=1)[0])
    df = df.rename(columns={
        'guidedMayorityVote':"mayorityVote",
        })
    columns.append("mayorityVote")

    path = os.path.join(RESULTSFOLDER, f"{annotation.replace('/', '_')}.csv")

    df[columns].to_csv(path, index=False)

    print(f"Results for annotation {annotation} results saved at {path}")
    # os.system(f"xan shuffle {path} | xan slice -l 5 | xan v")


def computeValidationMetrics(annotation):
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
                "version": version,
                "task": task,
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
    # print(f"Metrics for annotation {annotation} experiment saved at {path}\n")

    cmd = f"xan select model,params,labels,support,accuracy,f1_macro,f1_binary {path} | xan sort -s f1_macro | xan v"
    # print(cmd)
    os.system(cmd)

    cmd = f"xan groupby family --along-cols params,accuracy,f1_macro 'mean(_)' {path} | xan map '\"{version}\" as cross_validation' | xan map '\"{version}\" as cross_validation' | xan v"
    # print(cmd)
    os.system(cmd)


# parse and join results
parseAndJoin(annotation)

# make validations
computeValidationMetrics(annotation)


