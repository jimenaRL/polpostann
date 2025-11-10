import os
import csv
import json
import time
import asyncio
import numpy as np
from argparse import ArgumentParser

from openai import OpenAI

BASEPATH = ""
if "SERVER" in os.environ:
    if os.environ["SERVER"] == "jeanzay":
        BASEPATH =  "/lustre/fswork/projects/rech/nmf/umu89ib/dev/polpostann"
    if os.environ["SERVER"] == "in2p3":
        BASEPATH =  "/sps/humanum/user/jroyolet/dev/polpostann"

DEFAULTRESFOLDER = os.path.join(BASEPATH, 'translations_v2/mistralai_Mistral-7B-Instruct-v0.2')
DEFAULTINPUTSFILE = os.path.join(BASEPATH, 'cleaned_text2annotate_2022-03-27_2022-04-25.csv')
DEFAULTMODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULTINSTRUCTIONS = "Please translate the following text from French to English. Do not provide any explanation or note. Do not produce anything other than a literal translation."
DEFAULTCOLUMNS = "idx_all,french"
DEFAULTCONTENTCOLUMN = "french"
DEFAULTBATCHSIZE = 1000

ap = ArgumentParser()
ap.add_argument('--model', type=str, default=DEFAULTMODEL)
ap.add_argument('--instructions', type=str, default=DEFAULTINSTRUCTIONS)
ap.add_argument('--results_folder', type=str, default=DEFAULTRESFOLDER)
ap.add_argument('--input_file', type=str, default=DEFAULTINPUTSFILE)
ap.add_argument('--content_column', type=str, default=DEFAULTCONTENTCOLUMN)
ap.add_argument('--columns_to_keep', type=str, default=DEFAULTCOLUMNS)
ap.add_argument('--batch_size', type=int, default=DEFAULTBATCHSIZE)
ap.add_argument('--reverse_batch_order',  action='store_true')
ap.add_argument('--nbgpus', type=int, default=1)

args = ap.parse_args()
model = args.model
instructions = args.instructions
nbgpus = args.nbgpus
input_file = args.input_file
content_column = args.content_column
columns_to_keep = args.columns_to_keep.split(",")
reverse_batch_order = args.reverse_batch_order
results_folder = args.results_folder
batch_size = args.batch_size

parameters = vars(args)
dumped_parameters = json.dumps(parameters, sort_keys=True, indent=4)
print("---------------------------------------------------------")
print(f"PARAMETERS:\n{dumped_parameters[2:-2]}")
print("---------------------------------------------------------")

if not content_column in columns_to_keep:
    columns_to_keep.append(content_column)

# Load inputs
with open(input_file, 'r') as f:
    reader = csv.reader(f)
    inputs = [{c: d[c] for c in columns_to_keep} for d in csv.DictReader(f)]

# Run vllm server
vllm_serve_command = f'vllm serve "{model}" --tensor-parallel-size {nbgpus} &'
print(f"[RUNNING] {vllm_serve_command}")
os.system(vllm_serve_command)

# Wait for vllm server to be available and retrive model
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)
model = None
while not model:
    try:
        model = client.models.list().data[0].id
    except Exception as e:
        print(f"Model not ready: {e}. Waiting 30 more seconds...")
        time.sleep(30)
print(f"Model is ready: {model} !")


async def doCompletetion(input_):
    # make request
    res = client.responses.create(
        model=model,
        instructions=instructions,
        input=input_[content_column])
    # format result
    input_.update({'res': res.output_text.strip()})
    # and return
    return input_

async def inputsIterator(start, end):
    for input_ in inputs[start:end]:
        yield input_

async def run_all(start, end):
    # Asynchronously call the function for each prompt
    tasks = [
        doCompletetion(input_)
        async for input_ in inputsIterator(start, end)
    ]
    # Gather and run the tasks concurrently
    results = await asyncio.gather(*tasks)
    return results

data_length = len(inputs)
batchl = [
    [i * batch_size, (i + 1) * batch_size]
    for i in range(np.int32(data_length / batch_size + 1))
]
batchl[-1][1] = min(data_length,  batchl[-1][1])

enumbatchl = list(enumerate(batchl))
if reverse_batch_order:
    enumbatchl.reverse()

for batch_idx, b in enumbatchl:

        file = os.path.join(results_folder, f"translations_bsize_{batch_size}_bindex_{batch_idx}.csv")
        lockfile = file + ".lock"

        # If batch already computed continue or lock is granted, continue
        if os.path.exists(file):
            print(f"Already computed file at {file}. Continuing.")
            continue

        if os.path.exists(lockfile):
            print(f"Found active lock for file at {file}. Continuing.")
            continue

        # If not, create lock and start batch computation
        with open(lockfile, 'w') as f:
            f.write("")
        print(f"Computing batch at index {batch_idx}...")

        # Run all courutines
        start = time.time()
        results = asyncio.run(run_all(start=b[0], end=b[1]))
        end = time.time()
        print(f"Took {end - start} seconds.")

        # save to file
        with open(file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"LLM answers (={len(results)}) saved to {file}")

        # and release lock
        os.system(f"rm {lockfile}")
