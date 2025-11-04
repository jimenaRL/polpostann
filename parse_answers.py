import os
from glob import glob

folders = glob("outputs_in2p3/french/*/guided/*/binary/*")

for folder in folders:
    csv_files_pattern = os.path.join(folder, "llm_answer_*.csv")
    paths_file = os.path.join(folder, "csvfiles.txt")
    print(f"ls {csv_files_pattern} > {paths_file}")
    out_csv_file = os.path.join(folder, "concatenated_answers.csv")
    print(f"xan cat rows --paths {paths_file} > {out_csv_file}")
