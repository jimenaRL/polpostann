import os
import csv
import tempfile

currentdir = os.path.dirname(os.path.realpath(__file__))

file = "/home/jimena/work/cleaned_text2annotate_2022-03-27_2022-04-25.csv"
seed = 999
outprefix = "123456"

def xanSearch(cmd, outfile):
    mssg = f"Preforming a xan search the command is:\n\t{cmd}"
    print(mssg)
    os.system(cmd)
    os.system(f"xan head {outfile} | xan f")

# random sample
nbrandom = 100
outfile_random = f"{outprefix}_random.csv"

cmd = f"xan sample --seed={seed} {nbrandom} {file} "
cmd += f" > {outfile_random}"

xanSearch(cmd, outfile_random)

# oriented sample

candidates = ['Macron', 'Mélenchon', 'Le Pen']
patterns = ['\\bvote|é\\b', '\\bappel\\b', '\\bsoutien\\b']
nb = 100

patternsfile = f"{outprefix}_patterns.csv"
with open(patternsfile, 'w') as f:
    f.writelines('\n'.join(patterns)+'\n')

for candidate in candidates:

        outfile = f"{outprefix}_{candidate.replace(' ', '')}_sampled.csv"

        cmd = f"xan search -i -r --patterns {patternsfile} {file} | "
        cmd += f"xan search je | "
        cmd += f"xan search '{candidate}' | "
        cmd += f"xan sample --seed={seed} {nb}"
        cmd += f" > {outfile}"

        xanSearch(cmd, outfile)

totalnb = nbrandom + nb * len(candidates)
finalfile = f"{totalnb}_balanced_sampled_xan_seed_{seed}.csv"
cmd = f"xan cat rows "
cmd += f" {outfile_random} "
for candidate in candidates:
    cmd += f" {outprefix}_{candidate.replace(' ', '')}_sampled.csv "
cmd += f" > {finalfile}"

xanSearch(cmd, finalfile)


