import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import categorical
from warnings import warn


class TPUScheduledEmbeddingTrainingHelper(tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper):
    def __init__(self, inputs, sequence_length, embedding, sampling_probability):
        super().__init__(inputs, sequence_length, embedding, sampling_probability)
        self._orig_batch_size = sequence_length.get_shape()[0].value
        warn('This helper is not yet TPU compatible.')

    def sample(self, time, outputs, state, name=None):
        with ops.name_scope(name, "ScheduledEmbeddingTrainingHelper",
                            [time, outputs, state]):
            # Return -1s where we did not sample, and sample_ids elsewhere
            select_sampler = bernoulli.Bernoulli(
                probs=self._sampling_probability, dtype=dtypes.bool)
            select_sample = select_sampler.sample(
                sample_shape=self._orig_batch_size, seed=self._scheduling_seed)
            sample_id_sampler = categorical.Categorical(logits=outputs)
            return array_ops.where(
                select_sample,
                sample_id_sampler.sample(seed=self._seed),
                gen_array_ops.fill([self._orig_batch_size], -1))
