import argparse
import os
import tensorflow as tf
from joblib import dump

import utils

from model_helper import las_model_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description='Listen, Attend and Spell(LAS) implementation based on Tensorflow. '
                    'The model utilizes input pipeline and estimator API of Tensorflow, '
                    'which makes the training procedure truly end-to-end.')

    parser.add_argument('--data', type=str, required=True,
                        help='inference data in TFRecord format')
    parser.add_argument('--vocab', type=str, required=True,
                        help='vocabulary table, listing vocabulary line by line')
    parser.add_argument('--norm', type=str, default=None,
                        help='normalization params')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='path of imported model')

    parser.add_argument('--beam_width', type=int, default=0,
                        help='number of beams (default 0: using greedy decoding)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--num_channels', type=int, default=39,
                        help='number of input channels')
    parser.add_argument('--delimiter', help='Symbols delimiter. Default: " "', type=str, default=' ')
    parser.add_argument('--take', help='Use this number of elements (0 for all).', type=int, default=0)

    return parser.parse_args()


def input_fn(dataset_filename, vocab_filename, norm_filename=None, num_channels=39, batch_size=8, take=0):
    dataset = utils.read_dataset(dataset_filename, num_channels)
    vocab_table = utils.create_vocab_table(vocab_filename)

    if norm_filename is not None:
        means, stds = utils.load_normalization(args.norm)
    else:
        means = stds = None

    dataset = utils.process_dataset(
        dataset, vocab_table, utils.SOS, utils.EOS, means, stds, batch_size, 1)

    if args.take > 0:
        dataset = dataset.take(take)
    return dataset


def to_text(vocab_list, sample_ids):
    sym_list = [vocab_list[x] for x in sample_ids] + [utils.EOS]
    return args.delimiter.join(sym_list[:sym_list.index(utils.EOS)])


def main(args):
    vocab_list = utils.load_vocab(args.vocab)
    vocab_size = len(vocab_list)

    config = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = utils.create_hparams(
        args, vocab_size, utils.SOS_ID, utils.EOS_ID)

    hparams.decoder.set_hparam('beam_width', args.beam_width)

    model = tf.estimator.Estimator(
        model_fn=las_model_fn,
        config=config,
        params=hparams)

    predictions = model.predict(
        input_fn=lambda: input_fn(
            args.data, args.vocab, args.norm, num_channels=args.num_channels,
            batch_size=args.batch_size, take=args.take),
        predict_keys=['sample_ids', 'embedding'])

    if args.beam_width > 0:
        predictions = [{
            'transcription': to_text(vocab_list, y['sample_ids'][:, 0]),
            'embedding': y['embedding']
        } for y in predictions]
    else:
        predictions = [{
            'transcription': to_text(vocab_list, y['sample_ids']),
            'embedding': y['embedding']
        } for y in predictions]

    save_to = os.path.join(args.model_dir, 'infer.txt')
    with open(save_to, 'w') as f:
        f.write('\n'.join(p['transcription'] for p in predictions))

    save_to = os.path.join(args.model_dir, 'infer.dmp')
    dump(predictions, save_to)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    main(args)
