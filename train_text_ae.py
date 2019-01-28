import argparse
import tensorflow as tf

from utils.params_utils import create_hparams
from utils.vocab_utils import *
from text_ae.text_dataset import read_dataset, process_dataset
from text_ae.model_helper import text_ae_model_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description='Trains auxiliary text-to-text auto-encoder.')

    parser.add_argument('--train', type=str, required=True,
                        help='training data in TFRecord format')
    parser.add_argument('--valid', type=str,
                        help='validation data in TFRecord format')
    parser.add_argument('--vocab', type=str, required=True,
                        help='vocabulary table, listing vocabulary line by line')
    parser.add_argument('--mapping', type=str,
                        help='additional mapping when evaluation')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='path of saving model')
    parser.add_argument('--eval_secs', type=int, default=300,
                        help='evaluation every N seconds, only happening when `valid` is specified')

    parser.add_argument('--encoder_units', type=int, default=128,
                        help='rnn hidden units of encoder')
    parser.add_argument('--encoder_layers', type=int, default=3,
                        help='rnn layers of encoder')
    parser.add_argument('--use_pyramidal', action='store_true',
                        help='whether to use pyramidal rnn')

    parser.add_argument('--decoder_units', type=int, default=128,
                        help='rnn hidden units of decoder')
    parser.add_argument('--decoder_layers', type=int, default=2,
                        help='rnn layers of decoder')
    parser.add_argument('--embedding_size', type=int, default=0,
                        help='embedding size of target vocabulary, if 0, one hot encoding is applied')
    parser.add_argument('--sampling_probability', type=float, default=0.1,
                        help='sampling probabilty of decoder during training')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate of rnn cell')

    return parser.parse_args()


def input_fn(dataset_filename, vocab_filename, batch_size=8, num_epochs=1):
    dataset = read_dataset(dataset_filename,)
    vocab_table = create_vocab_table(vocab_filename)
    dataset = process_dataset(dataset, vocab_table, SOS, EOS, batch_size, num_epochs)
    return dataset


def main(args):
    vocab_list = load_vocab(args.vocab)
    vocab_size = len(vocab_list)

    config = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = create_hparams(args, vocab_size, SOS_ID, EOS_ID)

    model = tf.estimator.Estimator(
        model_fn=text_ae_model_fn,
        config=config,
        params=hparams)

    if args.valid:
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(
                args.train, args.vocab, batch_size=args.batch_size, num_epochs=args.num_epochs))

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(
                args.valid or args.train, args.vocab, batch_size=args.batch_size),
            start_delay_secs=60,
            throttle_secs=args.eval_secs)

        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    else:
        model.train(
            input_fn=lambda: input_fn(
                args.train, args.vocab, batch_size=args.batch_size, num_epochs=args.num_epochs))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    main(args)
