import os
from argparse import ArgumentParser
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('--vctk_path', help='Path to VCTK corpus.', required=True, type=str)
parser.add_argument('--output', help='Path to output file.', required=True, type=str)

args = parser.parse_args()

output = open(args.output, 'w')
for root, _, files in tqdm(os.walk(args.vctk_path), desc='Collecting filenames'):
    for audio_filename in filter(lambda x: x.endswith('.wav'), files):
        text_root = root.replace('wav48', 'txt')
        text_filename = audio_filename.replace('.wav', '.txt')
        text_path = os.path.join(text_root, text_filename)
        if not os.path.exists(text_path):
            continue
        with open(text_path, 'r') as f:
            text = f.read().strip()
        audio_path = os.path.join(root, audio_filename)
        output.write('{}|||{}\n'.format(audio_path, text_path))
