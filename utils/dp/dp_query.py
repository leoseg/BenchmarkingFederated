"""Implements DPQuery interface for distributed discrete Gaussian mechanism."""

import collections
import tensorflow_federated as tff
import dp_accounting
import tensorflow as tf
from tensorflow_federated.python.aggregators.discretization import _stochastic_rounding
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_privacy.privacy.dp_query import discrete_gaussian_utils
from tensorflow_privacy.privacy.dp_query import dp_query
import numpy as np

class DistributedDiscreteGaussianSumQuery(dp_query.SumAggregationDPQuery):
  """Implements DPQuery for discrete distributed Gaussian sum queries.

  For each local record, we check the L2 norm bound and add discrete Gaussian
  noise. In particular, this DPQuery does not perform L2 norm clipping and the
  norms of the input records are expected to be bounded.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple('_GlobalState',
                                        ['l2_norm_bound', 'local_stddev'])

  # pylint: disable=invalid-name
  _SampleParams = collections.namedtuple('_SampleParams',
                                         ['l2_norm_bound', 'local_stddev'])

  def __init__(self, l2_norm_bound, local_stddev,scale_factor = 1.0, stochastic= False,beta=np.exp(-0.5), prior_norm_bound=0.0):
    """Initializes the DistributedDiscreteGaussianSumQuery.

    Args:
      l2_norm_bound: The L2 norm bound to verify for each record.
      local_stddev: The stddev of the local discrete Gaussian noise.
    """
    self._l2_norm_bound = l2_norm_bound
    self._local_stddev = local_stddev
    self.scale_factor = float(scale_factor)
    self.stochastic = stochastic
    self.beta = float(beta)
    self.prior_norm_bound = prior_norm_bound

  def initial_global_state(self):
    return self._GlobalState(
        tf.cast(self._l2_norm_bound, tf.float32),
        tf.cast(self._local_stddev, tf.float32))

  def derive_sample_params(self, global_state):
    return self._SampleParams(global_state.l2_norm_bound,
                              global_state.local_stddev)

  def _add_local_noise(self, record, local_stddev, shares=1):
    """Add local discrete Gaussian noise to the record.

    Args:
      record: The record to which we generate and add local noise.
      local_stddev: The stddev of the local discrete Gaussian noise.
      shares: Number of shares of local noise to generate. Should be 1 for each
        record. This can be useful when we want to generate multiple noise
        shares at once.

    Returns:
      The record with local noise added.
    """
    # Round up the noise as the TF discrete Gaussian sampler only takes
    # integer noise stddevs for now.
    ceil_local_stddev = tf.cast(tf.math.ceil(local_stddev), tf.int32)

    def add_noise(v):
      # Adds an extra dimension for `shares` number of draws.
      shape = tf.concat([[shares], tf.shape(v)], axis=0)
      dgauss_noise = discrete_gaussian_utils.sample_discrete_gaussian(
          scale=ceil_local_stddev, shape=shape, dtype=v.dtype)
      # Sum across the number of noise shares and add it.
      noised_v = v + tf.reduce_sum(dgauss_noise, axis=0)
      # Set shape as TF shape inference may fail due to custom noise sampler.
      noised_v.set_shape(v.shape.as_list())
      return noised_v

    return tf.nest.map_structure(add_noise, record)

  def _discretize_struct(
            self,struct, scale_factor, stochastic, beta, prior_norm_bound
    ):
    """Scales and rounds each tensor of the structure to the integer grid."""

    def discretize_tensor(x):
        x = tf.cast(x, tf.float32)
        # Scale up the values.
        scaled_x = x * scale_factor
        scaled_bound = prior_norm_bound * scale_factor  # 0 if no prior bound.
        # Round to integer grid.
        if stochastic:
            discretized_x = _stochastic_rounding(
                scaled_x, scaled_bound, scale_factor, beta
            )
        else:
            discretized_x = tf.round(scaled_x)

        return tf.cast(discretized_x, tf.int32)

    return tf.nest.map_structure(discretize_tensor, struct)

  def _undiscretize_struct(self,struct, scale_factor, tf_dtype_struct):
      """Unscales the discretized structure and casts back to original dtypes."""

      def undiscretize_tensor(x, original_dtype):
        unscaled_x = tf.cast(x, tf.float32) / scale_factor
        return tf.cast(unscaled_x, original_dtype)

      return tf.nest.map_structure(undiscretize_tensor, struct, tf_dtype_struct)

  def preprocess_record(self, params, record):
    """Check record norm and add noise to the record."""
    record_as_list = tf.nest.flatten(record)

    self._discretize_struct(record_as_list, self.scale_factor, self.stochastic, self.beta,self.prior_norm_bound)
    tf_dtype = type_conversions.structure_from_tensor_type_tree(
        lambda x: x.dtype, tff.types.to_type(record)
    )
    record_as_float_list = [tf.cast(x, tf.float32) for x in record_as_list]
    tf.nest.map_structure(lambda x: tf.compat.v1.assert_type(x, tf.int32),
                          record_as_list)
    dependencies = [
        tf.compat.v1.assert_less_equal(
            tf.linalg.global_norm(record_as_float_list),
            params.l2_norm_bound,
            message=f'Global L2 norm exceeds {params.l2_norm_bound}.')
    ]
    with tf.control_dependencies(dependencies):
      result = tf.cond(
          tf.equal(params.local_stddev, 0), lambda: record,
          lambda: self._add_local_noise(record, params.local_stddev))
      return self._undiscretize_struct(result,self.scale_factor,tf_dtype)

  def get_noised_result(self, sample_state, global_state):
    # Note that by directly returning the aggregate, this assumes that there
    # will not be missing local noise shares during execution.
    event = dp_accounting.UnsupportedDpEvent()
    return sample_state, global_state, event


class LocalGaussianSumQuery(dp_query.SumAggregationDPQuery):
  """Implements DPQuery for discrete distributed Gaussian sum queries.

  For each local record, we check the L2 norm bound and add discrete Gaussian
  noise. In particular, this DPQuery does not perform L2 norm clipping and the
  norms of the input records are expected to be bounded.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple('_GlobalState',
                                        ['l2_norm_bound', 'local_stddev'])

  # pylint: disable=invalid-name
  _SampleParams = collections.namedtuple('_SampleParams',
                                         ['l2_norm_bound', 'local_stddev'])

  def __init__(self, l2_norm_bound, local_stddev):
    """Initializes the DistributedDiscreteGaussianSumQuery.

    Args:
      l2_norm_bound: The L2 norm bound to verify for each record.
      local_stddev: The stddev of the local discrete Gaussian noise.
    """
    self._l2_norm_bound = l2_norm_bound
    self._local_stddev = local_stddev

  def initial_global_state(self):
    return self._GlobalState(
        tf.cast(self._l2_norm_bound, tf.float32),
        tf.cast(self._local_stddev, tf.float32))

  def derive_sample_params(self, global_state):
    return self._SampleParams(global_state.l2_norm_bound,
                              global_state.local_stddev)

  def _add_local_noise(self, record, local_stddev, shares=1):
    """Add local discrete Gaussian noise to the record.

    Args:
      record: The record to which we generate and add local noise.
      local_stddev: The stddev of the local discrete Gaussian noise.
      shares: Number of shares of local noise to generate. Should be 1 for each
        record. This can be useful when we want to generate multiple noise
        shares at once.

    Returns:
      The record with local noise added.
    """
    # Round up the noise as the TF discrete Gaussian sampler only takes
    # integer noise stddevs for now.
    #ceil_local_stddev = tf.cast(tf.math.ceil(local_stddev), tf.int32)
    random_normal = tf.random_normal_initializer(stddev=local_stddev)

    def add_noise(v):
        return v + tf.cast(random_normal(tf.shape(input=v)), dtype=v.dtype)
    # def add_noise(v):
    #   # Adds an extra dimension for `shares` number of draws.
    #   shape = tf.concat([[shares], tf.shape(v)], axis=0)
    #
    #   dgauss_noise = discrete_gaussian_utils.sample_discrete_gaussian(
    #       scale=ceil_local_stddev, shape=shape, dtype=v.dtype)
    #   # Sum across the number of noise shares and add it.
    #   noised_v = v + tf.reduce_sum(dgauss_noise, axis=0)
    #   # Set shape as TF shape inference may fail due to custom noise sampler.
    #   noised_v.set_shape(v.shape.as_list())
    #   return noised_v

    return tf.nest.map_structure(add_noise, record)

  def preprocess_record(self, params, record):
    """Check record norm and add noise to the record."""
    record_as_list = tf.nest.flatten(record)

    record_as_float_list = [tf.cast(x, tf.float32) for x in record_as_list]
    dependencies = [
        tf.compat.v1.assert_less_equal(
            tf.linalg.global_norm(record_as_float_list),
            params.l2_norm_bound,
            message=f'Global L2 norm exceeds {params.l2_norm_bound}.')
    ]
    with tf.control_dependencies(dependencies):
      result = tf.cond(
          tf.equal(params.local_stddev, 0), lambda: record,
          lambda: self._add_local_noise(record, params.local_stddev))
      return result



  def get_noised_result(self, sample_state, global_state):
    # Note that by directly returning the aggregate, this assumes that there
    # will not be missing local noise shares during execution.
    event = dp_accounting.UnsupportedDpEvent()
    return sample_state, global_state, event
