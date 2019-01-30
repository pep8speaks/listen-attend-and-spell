import os
from argparse import ArgumentParser
from tqdm import tqdm
import random


parser = ArgumentParser()
parser.add_argument('--timit_path', help='Path to converted TIMIT corpus.', required=True, type=str)
parser.add_argument('--val_fraction', help='What fraction of train to use in validation.', required=True, type=float)
parser.add_argument('--output_dir', help='Path to output directory.', required=True, type=str)
parser.add_argument('--read_phonemes', help='Red phonetic representation', action='store_true')

args = parser.parse_args()

output_train = open(os.path.join(args.output_dir, 'train.csv'), 'w')
output_val = open(os.path.join(args.output_dir, 'val.csv'), 'w')
output_test = open(os.path.join(args.output_dir, 'test.csv'), 'w')
annotation_ext = '.PHN' if args.read_phonemes else '.TXT'
for root, _, files in tqdm(os.walk(args.timit_path), desc='Collecting filenames'):
    for audio_filename in filter(lambda x: x.endswith('.WAV'), files):
        if 'SA' in audio_filename:
            # skip dialect sentences
            continue
        text_filename = audio_filename.replace('.WAV', annotation_ext)
        text_path = os.path.join(root, text_filename)
        if not os.path.exists(text_path):
            continue
        with open(text_path, 'r') as f:
            if args.read_phonemes:
                text = ' '.join([line.split(' ')[-1]  for line in f.read().strip().split('\n')])
            else:
                text = f.read().strip().replace(',', '')
                text = ' '.join(text.split()[2:])
        audio_path = os.path.join(root, audio_filename)
        write_text = '{},en,{}\n'.format(audio_path, text)
        if 'TEST' in root:
            output_test.write(write_text)
        elif random.random() < args.val_fraction:
            output_val.write(write_text)
        else:
            output_train.write(write_text)
