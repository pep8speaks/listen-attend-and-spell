import tensorflow as tf
from tqdm import tqdm
from multiprocessing import Lock
from joblib import Parallel, delayed
from argparse import ArgumentParser

from utils import get_ipa


par_handle = None
tfrecord_mutex = Lock()


def make_example(input, label):
    feature_lists = tf.train.FeatureLists(feature_list={
        'labels': tf.train.FeatureList(feature=[
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[p.encode()]))
            for p in label
        ]),
        'inputs': tf.train.FeatureList(feature=[
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[f.encode()]))
            for f in input
        ])
    })

    return tf.train.SequenceExample(feature_lists=feature_lists)


def read_text(inputs):
    text = inputs['text']
    language = inputs['language']
    text = ' '.join(text.split())
    for p in ',.:;?!-_':
        text = text.replace(p, '')
    text = text.lower().split()
    return {
        'text': text,
        'language': language
    }


def build_features_fn(args, inputs):
    text = inputs['text']
    language = inputs['language']
    if args.targets == 'phones':
        text = list(' '.join([get_ipa(t, language) for t in text]))
        text = [x if x != ' ' else '<space>' for x in text]
    return {
        'text': text
    }


def write_tf_output(writer, inputs):
    with tfrecord_mutex:
        writer.write(make_example(inputs['text'], inputs['text']).SerializeToString())
    par_handle.update()


def process_line(args, writer, line):
    _, language, text = line.split(',')
    inputs = {
        'text': text.strip(),
        'language': language
    }
    out = read_text(inputs)
    out = build_features_fn(args, out)
    write_tf_output(writer, out)


if __name__ == "__main__":
    parser = ArgumentParser(description='Collects textual data for auxiliary text auto-encoder.')
    parser.add_argument('--input_file', help='File with audio paths and texts.', required=True)
    parser.add_argument('--output_file', help='Target TFRecord file name.', required=True)
    parser.add_argument('--n_jobs', help='Number of parallel jobs.', type=int, default=4)
    parser.add_argument('--targets', help='Determines targets type.', type=str,
                        choices=['words', 'phones'], default='words')
    args = parser.parse_args()
    print('Processing text dataset from file {}.'.format(args.input_file))
    lines = open(args.input_file, 'r').readlines()
    par_handle = tqdm(unit='file')
    with tf.io.TFRecordWriter(args.output_file) as writer:
        Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(process_line)(args, writer, x) for x in lines)
    par_handle.close()
