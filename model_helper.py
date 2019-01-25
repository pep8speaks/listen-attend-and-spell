import os
import tensorflow as tf
import tensorflow.contrib as tf_contrib

import las
import utils

__all__ = [
    'las_model_fn',
]


def compute_loss(logits, targets, final_sequence_length, target_sequence_length, mode):

    assert mode != tf.estimator.ModeKeys.PREDICT

    if mode == tf.estimator.ModeKeys.TRAIN:
        target_weights = tf.sequence_mask(
            target_sequence_length, dtype=tf.float32)
        loss = tf_contrib.seq2seq.sequence_loss(
            logits, targets, target_weights)
    else:
        '''
        # Reference: https://github.com/tensorflow/nmt/issues/2
        # Note that this method always trim the tensor with larger length to shorter one, 
        # and I think it is unfair. 
        # Consider targets = [[3, 3, 2]], and logits with shape [1, 2, VOCAB_SIZE]. 
        # This method will trim targets to [[3, 3]] and compute sequence_loss on new targets and logits.
        # However, I think the truth is that the model predicts less word than ground truth does,
        # and hence, both targets and logits should be padded to the same sequence length (dimension 1)
        # to compute loss.

        current_sequence_length = tf.to_int32(
            tf.minimum(tf.shape(targets)[1], tf.shape(logits)[1]))
        targets = tf.slice(targets, begin=[0, 0],
                           size=[-1, current_sequence_length])
        logits = tf.slice(logits, begin=[0, 0, 0],
                          size=[-1, current_sequence_length, -1])
        target_weights = tf.sequence_mask(
            target_sequence_length, maxlen=current_sequence_length, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits, targets, target_weights)
        '''

        max_ts = tf.reduce_max(target_sequence_length)
        max_fs = tf.reduce_max(final_sequence_length)

        max_sequence_length = tf.to_int32(
            tf.maximum(max_ts, max_fs))

        logits = tf.slice(logits, begin=[0, 0, 0], size=[-1, max_fs, -1])

        # pad EOS to make targets and logits have same shape
        targets = tf.pad(targets, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(targets)[1])]], constant_values=utils.EOS_ID)
        logits = tf.pad(logits, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(logits)[1])], [0, 0]], constant_values=0)

        # find larger length between predictions and targets
        sequence_length = tf.reduce_max(
            [target_sequence_length, final_sequence_length], 0)

        target_weights = tf.sequence_mask(
            sequence_length, maxlen=max_sequence_length, dtype=tf.float32)

        loss = tf_contrib.seq2seq.sequence_loss(
            logits, targets, target_weights)

    return loss

def sequence_loss_sigmoid(logits, targets, weights):
    with tf.name_scope(name="sequence_loss",
                       values=[logits, targets, weights]):
        num_classes = tf.shape(logits)[2]
        logits_flat = tf.reshape(logits, [-1, num_classes])
        targets = tf.cast(tf.reshape(targets, [-1, num_classes]), tf.float32)
        crossent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=targets, logits=logits_flat), axis=1)

        crossent *= tf.reshape(weights, [-1])
        crossent = tf.reduce_sum(crossent)
        total_size = tf.reduce_sum(weights)
        total_size += 1e-12  # to avoid division by 0 for all-0 weights
        crossent /= total_size
    return crossent

def compute_loss_sigmoid(logits, targets, final_sequence_length, target_sequence_length, mode):

    assert mode != tf.estimator.ModeKeys.PREDICT

    if mode == tf.estimator.ModeKeys.TRAIN:
        target_weights = tf.sequence_mask(
            target_sequence_length, dtype=tf.float32)
        loss = sequence_loss_sigmoid(logits, targets, target_weights)
    else:
        max_ts = tf.reduce_max(target_sequence_length)
        max_fs = tf.reduce_max(final_sequence_length)

        max_sequence_length = tf.to_int32(
            tf.maximum(max_ts, max_fs))

        logits = tf.slice(logits, begin=[0, 0, 0], size=[-1, max_fs, -1])

        # pad EOS to make targets and logits have same shape
        targets = tf.pad(targets, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(targets)[1])], [0, 0]], constant_values=0)
        logits = tf.pad(logits, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(logits)[1])], [0, 0]], constant_values=0)

        # find larger length between predictions and targets
        sequence_length = tf.reduce_max(
            [target_sequence_length, final_sequence_length], 0)

        target_weights = tf.sequence_mask(
            sequence_length, maxlen=max_sequence_length, dtype=tf.float32)

        loss = sequence_loss_sigmoid(logits, targets, target_weights)

    return loss


def las_model_fn(features,
                 labels,
                 mode,
                 config,
                 params,
                 binf2phone=None):

    encoder_inputs = features['encoder_inputs']
    source_sequence_length = features['source_sequence_length']

    decoder_inputs = None
    targets = None
    target_sequence_length = None

    binf_embedding = None
    if binf2phone is not None and params.decoder.binary_outputs:
        binf_embedding = tf.constant(binf2phone, dtype=tf.float32, name='binf2phone')
    is_binf_outputs = params.decoder.binary_outputs and (
        binf_embedding is None or mode == tf.estimator.ModeKeys.TRAIN)

    if mode != tf.estimator.ModeKeys.PREDICT:
        decoder_inputs = labels['targets_inputs']
        targets = labels['targets_outputs']
        target_sequence_length = labels['target_sequence_length']

    tf.logging.info('Building listener')

    with tf.variable_scope('listener'):
        (encoder_outputs, source_sequence_length), encoder_state = las.model.listener(
            encoder_inputs, source_sequence_length, mode, params.encoder)

    tf.logging.info('Building speller')

    with tf.variable_scope('speller'):
        decoder_outputs, final_context_state, final_sequence_length = las.model.speller(
            encoder_outputs, encoder_state, decoder_inputs,
            source_sequence_length, target_sequence_length,
            mode, params.decoder, binf_embedding)

    with tf.name_scope('prediction'):
        if mode == tf.estimator.ModeKeys.PREDICT and params.decoder.beam_width > 0:
            logits = tf.no_op()
            sample_ids = decoder_outputs.predicted_ids
        else:
            logits = decoder_outputs.rnn_output
            if is_binf_outputs:
                sample_ids = tf.to_int32(tf.round(tf.sigmoid(logits)))
            else:
                sample_ids = tf.to_int32(tf.argmax(logits, -1))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'sample_ids': sample_ids,
            'embedding': tf.stack([x.c for x in encoder_state], axis=1),
            'encoder_out': encoder_outputs,
            'source_length': source_sequence_length
        }
        try:
            predictions['alignment'] = tf.transpose(final_context_state.alignment_history.stack(), perm=[1, 0, 2])
        except AttributeError:
            alignment_history = tf.transpose(final_context_state.cell_state.alignment_history, perm=[1, 0, 2])
            shape = tf.shape(alignment_history)
            predictions['alignment'] = tf.reshape(alignment_history,
                [-1, params.decoder.beam_width, shape[1], shape[2]])
        if params.decoder.beam_width == 0:
            if params.decoder.binary_outputs and binf_embedding is None:
                predictions['probs'] = tf.nn.sigmoid(logits)
            else:
                predictions['probs'] = tf.nn.softmax(logits)

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    metrics = None

    if binf_embedding is not None and not is_binf_outputs:
        binf_to_ipa_tiled = tf.cast(
            tf.tile(binf_embedding[None, :, :], [tf.shape(targets)[0], 1, 1]), tf.int32)
        targets_transformed = tf.cast(
            tf.argmax(tf.matmul(targets, binf_to_ipa_tiled), -1), tf.int32)
    else:
        targets_transformed = targets

    if not is_binf_outputs:
        with tf.name_scope('metrics'):
            edit_distance = utils.edit_distance(
                sample_ids, targets_transformed, utils.EOS_ID, params.mapping)

            metrics = {
                'edit_distance': tf.metrics.mean(edit_distance),
            }

        tf.summary.scalar('edit_distance', metrics['edit_distance'][1])

    with tf.name_scope('cross_entropy'):
        loss_fn = compute_loss_sigmoid if is_binf_outputs else compute_loss
        loss = loss_fn(
            logits, targets_transformed, final_sequence_length, target_sequence_length, mode)

    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.name_scope('alignment'):
            attention_images = utils.create_attention_images(
                final_context_state)

        attention_summary = tf.summary.image(
            'attention_images', attention_images)

        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=10,
            output_dir=os.path.join(config.model_dir, 'eval'),
            summary_op=attention_summary)

        hooks = [eval_summary_hook]
        if not is_binf_outputs:
            logging_hook_editdistance = tf.train.LoggingTensorHook({
                'edit_distance': tf.reduce_mean(edit_distance),
                'max_edit_distance': tf.reduce_max(edit_distance),
                'min_edit_distance': tf.reduce_min(edit_distance),
            }, every_n_iter=10)
            hooks += [logging_hook_editdistance]

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=hooks)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())

    logging_hook_vocab = {
        'loss': loss
    }
    if not is_binf_outputs:
        logging_hook_vocab['edit_distance'] = tf.reduce_mean(edit_distance)
    logging_hook = tf.train.LoggingTensorHook(logging_hook_vocab, every_n_secs=10)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
