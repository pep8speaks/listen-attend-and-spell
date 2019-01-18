import numpy as np
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--labels', help='File with saved list of labels.', required=True)
parser.add_argument('--output', help='Output path.', required=True)
args = parser.parse_args()

s = set()
f = np.load(args.labels).item()

for line in f.values():
    s.update(line)

d = sorted(list(s))

with open(args.output, 'w') as f:
    print('\n'.join(d), file=f)
