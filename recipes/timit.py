import os
from argparse import ArgumentParser
from tqdm import tqdm
import random


parser = ArgumentParser()
parser.add_argument('--timit_path', help='Path to converted TIMIT corpus.', required=True, type=str)
parser.add_argument('--val_fraction', help='What fraction of train to use in validation.', required=True, type=float)
parser.add_argument('--output_dir', help='Path to output directory.', required=True, type=str)
parser.add_argument('--labels_type', help='What type of labels to use.',
    type=str, default='text', choices=['text', 'phones60', 'phones48', 'phones39'], )
parser.add_argument('--phone_map', help='Path to phoneme map file (for phones48 or phones39)',
    default='misc/phones.60-48-39.map')

args = parser.parse_args()

output_train = open(os.path.join(args.output_dir, 'train.csv'), 'w')
output_val = open(os.path.join(args.output_dir, 'val.csv'), 'w')
output_test = open(os.path.join(args.output_dir, 'test.csv'), 'w')
annotation_ext = '.TXT' if args.labels_type == 'text' else '.PHN'

mapping = None
mapping_ind = ['phones60', 'phones48', 'phones39'].index(args.labels_type)
if mapping_ind > 0:
    with open(args.phone_map, 'r') as f:
        lines = [line.split('\t') for line in f.read().strip().split('\n')]
        mapping = {line[0]: (line[mapping_ind] if len(line) > mapping_ind else None) for line in lines}

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
            if args.labels_type == 'text':
                text = f.read().strip().replace(',', '')
                text = ' '.join(text.split()[2:])
            else:
                text = [line.split(' ')[-1]  for line in f.read().strip().split('\n')]
                if mapping is not None:
                    text = [mapping[t] for t in text]
                text = ' '.join([t for t in text if t is not None])
        audio_path = os.path.join(root, audio_filename)
        write_text = '{},en,{}\n'.format(audio_path, text)
        if 'TEST' in root:
            output_test.write(write_text)
        elif random.random() < args.val_fraction:
            output_val.write(write_text)
        else:
            output_train.write(write_text)
