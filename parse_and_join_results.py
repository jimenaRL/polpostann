import os
from glob import glob
import  pandas as pd

BASEPATH = "/home/jimena/work/dev/polpostann"
VERSION = "v3debugFrench"
OUTPUTSFOLDER = os.path.join(BASEPATH, "outputs", VERSION)
RESULTSFOLDER = os.path.join(BASEPATH, "results", VERSION)
os.makedirs(RESULTSFOLDER, exist_ok=True)

MODELS = {
    "v3debug": [
        "zephyr-7b-beta",
        "Mistral-Small-24B-Instruct-2501",
        "Llama-3.3-70B-Instruct",
        "Mistral-Large-Instruct-2411",
    ],
    "v3debugFrench": [
        "zephyr-7b-beta",
        "Mistral-Small-24B-Instruct-2501",
        "Llama-3.3-70B-Instruct",
        "Mistral-Large-Instruct-2411",
    ],
}

ANNOTATIONS = {

    "multiple": [
        "voteintention/multiple/all",
        # "support/multiple/all",
        # "criticism/multiple/all"
    ],

    "binary" : [
        # "criticism/binary/lepen",
        # "criticism/binary/macron",
        # "criticism/binary/melenchon",
        # "support/binary/lepen",
        # "support/binary/macron",
        # "support/binary/melenchon",
        # "voteintention/binary/lepen",
        # "voteintention/binary/macron",
        # "voteintention/binary/melenchon",
    ]
}

CHOICES = {
    "multiple": ["Macron", "Mélenchon", "Le Pen", "None"],
    "binary" : ["YES", "NO"]
}

DEFAULTANSWER = {
    "multiple": "None",
    "binary" : "NO"
}

SETTINGS = ANNOTATIONS.keys()

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

def extract_data(model, annotation, setting, df=None, columns=[]):

    assert model in MODELS[VERSION]
    assert setting in SETTINGS

    folders = glob(os.path.join(OUTPUTSFOLDER, f"*{model}*"))
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

for setting in SETTINGS:

    for annotation in ANNOTATIONS[setting]:

        df = None
        columns = ["idx", "tweet"]

        for model in MODELS[VERSION]:
            df, columns = extract_data(model, annotation, setting, df, columns)

        df = df.assign(guidedMayorityVote=df[df.columns[2:]].mode(axis=1)[0])
        df = df.rename(columns={
            'guidedMayorityVote':"mayorityVote",
            })
        columns.append("mayorityVote")

        path = os.path.join(RESULTSFOLDER, f"{annotation.replace('/', '_')}.csv")

        df[columns].to_csv(path, index=False)

        print(f"Results for annotation {annotation} results saved at {path}")
        os.system(f"xan shuffle {path} | xan slice -l 5 | xan v")
