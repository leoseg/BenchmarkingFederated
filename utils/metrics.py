""" Metrics is derived from keras metric to use in FL setting for multi class AUC"""
import tensorflow as tf
import numpy as np
from keras import activations
from keras import backend
from keras.dtensor import utils as dtensor_utils
from keras.metrics import base_metric
from keras.utils import metrics_utils
from keras.utils.tf_utils import is_tensor_or_variable

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.metrics.AUC")
class AUC(base_metric.Metric):
    """Approximates the AUC (Area under the curve) of the ROC or PR curves.
    Derived from original AUC from keras but does onehotencoding before evaluation.
    The AUC (Area under the curve) of the ROC (Receiver operating
    characteristic; default) or PR (Precision Recall) curves are quality
    measures of binary classifiers. Unlike the accuracy, and like cross-entropy
    losses, ROC-AUC and PR-AUC evaluate all the operational points of a model.

    This class approximates AUCs using a Riemann sum. During the metric
    accumulation phrase, predictions are accumulated within predefined buckets
    by value. The AUC is then computed by interpolating per-bucket averages.
    These buckets define the evaluated operational points.

    This metric creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the AUC.  To discretize the AUC curve, a linearly spaced set of
    thresholds is used to compute pairs of recall and precision values. The area
    under the ROC-curve is therefore computed using the height of the recall
    values by the false positive rate, while the area under the PR-curve is the
    computed using the height of the precision values by the recall.

    This value is ultimately returned as `auc`, an idempotent operation that
    computes the area under a discretized curve of precision versus recall
    values (computed using the aforementioned variables). The `num_thresholds`
    variable controls the degree of discretization with larger numbers of
    thresholds more closely approximating the true AUC. The quality of the
    approximation may vary dramatically depending on `num_thresholds`. The
    `thresholds` parameter can be used to manually specify thresholds which
    split the predictions more evenly.

    For a best approximation of the real AUC, `predictions` should be
    distributed approximately uniformly in the range [0, 1] (if
    `from_logits=False`). The quality of the AUC approximation may be poor if
    this is not the case. Setting `summation_method` to 'minoring' or 'majoring'
    can help quantify the error in the approximation by providing lower or upper
    bound estimate of the AUC.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
      num_thresholds: (Optional) Defaults to 200. The number of thresholds to
        use when discretizing the roc curve. Values must be > 1.
      curve: (Optional) Specifies the name of the curve to be computed, 'ROC'
        [default] or 'PR' for the Precision-Recall-curve.
      summation_method: (Optional) Specifies the [Riemann summation method](
          https://en.wikipedia.org/wiki/Riemann_sum) used.
          'interpolation' (default) applies mid-point summation scheme for
          `ROC`.  For PR-AUC, interpolates (true/false) positives but not the
          ratio that is precision (see Davis & Goadrich 2006 for details);
          'minoring' applies left summation for increasing intervals and right
          summation for decreasing intervals; 'majoring' does the opposite.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      thresholds: (Optional) A list of floating point values to use as the
        thresholds for discretizing the curve. If set, the `num_thresholds`
        parameter is ignored. Values should be in [0, 1]. Endpoint thresholds
        equal to {-epsilon, 1+epsilon} for a small positive epsilon value will
        be automatically included with these to correctly handle predictions
        equal to exactly 0 or 1.
      multi_label: boolean indicating whether multilabel data should be
        treated as such, wherein AUC is computed separately for each label and
        then averaged across labels, or (when False) if the data should be
        flattened into a single label before AUC computation. In the latter
        case, when multilabel data is passed to AUC, each label-prediction pair
        is treated as an individual data point. Should be set to False for
        multi-class data.
      num_labels: (Optional) The number of labels, used when `multi_label` is
        True. If `num_labels` is not specified, then state variables get created
        on the first call to `update_state`.
      label_weights: (Optional) list, array, or tensor of non-negative weights
        used to compute AUCs for multilabel data. When `multi_label` is True,
        the weights are applied to the individual label AUCs when they are
        averaged to produce the multi-label AUC. When it's False, they are used
        to weight the individual label predictions in computing the confusion
        matrix on the flattened data. Note that this is unlike class_weights in
        that class_weights weights the example depending on the value of its
        label, whereas label_weights depends only on the index of that label
        before flattening; therefore `label_weights` should not be used for
        multi-class data.
      from_logits: boolean indicating whether the predictions (`y_pred` in
        `update_state`) are probabilities or sigmoid logits. As a rule of thumb,
        when using a keras loss, the `from_logits` constructor argument of the
        loss should match the AUC `from_logits` constructor argument.

    Standalone usage:

    >>> m = tf.keras.metrics.AUC(num_thresholds=3)
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
    >>> # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
    >>> # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
    >>> # tp_rate = recall = [1, 0.5, 0], fp_rate = [1, 0, 0]
    >>> # auc = ((((1+0.5)/2)*(1-0)) + (((0.5+0)/2)*(0-0))) = 0.75
    >>> m.result().numpy()
    0.75

    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
    ...                sample_weight=[1, 0, 0, 1])
    >>> m.result().numpy()
    1.0

    Usage with `compile()` API:

    ```python
    # Reports the AUC of a model outputting a probability.
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC()])

    # Reports the AUC of a model outputting a logit.
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.AUC(from_logits=True)])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(
        self,
        num_thresholds=200,
        curve="ROC",
        summation_method="interpolation",
        name=None,
        dtype=None,
        thresholds=None,
        multi_label=False,
        num_labels=None,
        label_weights=None,
        from_logits=False,
    ):
        """
        Initializes `AUC` metric
        """
        if isinstance(curve, metrics_utils.AUCCurve) and curve not in list(
            metrics_utils.AUCCurve
        ):
            raise ValueError(
                f'Invalid `curve` argument value "{curve}". '
                f"Expected one of: {list(metrics_utils.AUCCurve)}"
            )
        if isinstance(
            summation_method, metrics_utils.AUCSummationMethod
        ) and summation_method not in list(metrics_utils.AUCSummationMethod):
            raise ValueError(
                "Invalid `summation_method` "
                f'argument value "{summation_method}". '
                f"Expected one of: {list(metrics_utils.AUCSummationMethod)}"
            )

        # Update properties.
        self._init_from_thresholds = thresholds is not None
        if thresholds is not None:
            # If specified, use the supplied thresholds.
            self.num_thresholds = len(thresholds) + 2
            thresholds = sorted(thresholds)
            self._thresholds_distributed_evenly = (
                metrics_utils.is_evenly_distributed_thresholds(
                    np.array([0.0] + thresholds + [1.0])
                )
            )
        else:
            if num_thresholds <= 1:
                raise ValueError(
                    "Argument `num_thresholds` must be an integer > 1. "
                    f"Received: num_thresholds={num_thresholds}"
                )

            # Otherwise, linearly interpolate (num_thresholds - 2) thresholds in
            # (0, 1).
            self.num_thresholds = num_thresholds
            thresholds = [
                (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
            ]
            self._thresholds_distributed_evenly = True

        # Add an endpoint "threshold" below zero and above one for either
        # threshold method to account for floating point imprecisions.
        self._thresholds = np.array(
            [0.0 - backend.epsilon()] + thresholds + [1.0 + backend.epsilon()]
        )

        if isinstance(curve, metrics_utils.AUCCurve):
            self.curve = curve
        else:
            self.curve = metrics_utils.AUCCurve.from_str(curve)
        if isinstance(summation_method, metrics_utils.AUCSummationMethod):
            self.summation_method = summation_method
        else:
            self.summation_method = metrics_utils.AUCSummationMethod.from_str(
                summation_method
            )
        super().__init__(name=name, dtype=dtype)

        # Handle multilabel arguments.
        self.multi_label = multi_label
        self.num_labels = num_labels
        if label_weights is not None:
            label_weights = tf.constant(label_weights, dtype=self.dtype)
            tf.debugging.assert_non_negative(
                label_weights,
                message="All values of `label_weights` must be non-negative.",
            )
            self.label_weights = label_weights

        else:
            self.label_weights = None

        self._from_logits = from_logits

        self._built = False
        if self.multi_label:
            if num_labels:
                shape = tf.TensorShape([None, num_labels])
                self._build(shape)
        else:
            if num_labels:
                raise ValueError(
                    "`num_labels` is needed only when `multi_label` is True."
                )
            self._build(None)

    @property
    def thresholds(self):
        """The thresholds used for evaluating AUC."""
        return list(self._thresholds)

    def _build(self, shape):
        """Initialize TP, FP, TN, and FN tensors, given the shape of the
        data."""
        if self.multi_label:
            if shape.ndims != 2:
                raise ValueError(
                    "`y_true` must have rank 2 when `multi_label=True`. "
                    f"Found rank {shape.ndims}. "
                    f"Full shape received for `y_true`: {shape}"
                )
            self._num_labels = shape[1]
            variable_shape = tf.TensorShape([self.num_thresholds, self._num_labels])
        else:
            variable_shape = tf.TensorShape([self.num_thresholds])

        self._build_input_shape = shape
        # Create metric variables
        self.true_positives = self.add_weight(
            "true_positives", shape=variable_shape, initializer="zeros"
        )
        self.true_negatives = self.add_weight(
            "true_negatives", shape=variable_shape, initializer="zeros"
        )
        self.false_positives = self.add_weight(
            "false_positives", shape=variable_shape, initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            "false_negatives", shape=variable_shape, initializer="zeros"
        )

        if self.multi_label:
            with tf.init_scope():
                # This should only be necessary for handling v1 behavior. In v2,
                # AUC should be initialized outside of any tf.functions, and
                # therefore in eager mode.
                if not tf.executing_eagerly():
                    backend._initialize_variables(backend._get_session())

        self._built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        y_true = tf.one_hot(tf.squeeze(y_true), depth=tf.shape(y_pred)[-1])
        if not self._built:
            self._build(tf.TensorShape(y_pred.shape))

        if self.multi_label or (self.label_weights is not None):
            # y_true should have shape (number of examples, number of labels).
            shapes = [(y_true, ("N", "L"))]
            if self.multi_label:
                # TP, TN, FP, and FN should all have shape
                # (number of thresholds, number of labels).
                shapes.extend(
                    [
                        (self.true_positives, ("T", "L")),
                        (self.true_negatives, ("T", "L")),
                        (self.false_positives, ("T", "L")),
                        (self.false_negatives, ("T", "L")),
                    ]
                )
            if self.label_weights is not None:
                # label_weights should be of length equal to the number of
                # labels.
                shapes.append((self.label_weights, ("L",)))
                tf.debugging.assert_shapes(
                    shapes, message="Number of labels is not consistent."
                )

        # Only forward label_weights to update_confusion_matrix_variables when
        # multi_label is False. Otherwise the averaging of individual label AUCs
        # is handled in AUC.result
        label_weights = None if self.multi_label else self.label_weights

        if self._from_logits:
            y_pred = activations.sigmoid(y_pred)

        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,  # noqa: E501
            },
            y_true,
            y_pred,
            self._thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            sample_weight=sample_weight,
            multi_label=self.multi_label,
            label_weights=label_weights,
        )

    def interpolate_pr_auc(self):
        """Interpolation formula inspired by section 4 of Davis & Goadrich 2006.

        https://www.biostat.wisc.edu/~page/rocpr.pdf

        Note here we derive & use a closed formula not present in the paper
        as follows:

          Precision = TP / (TP + FP) = TP / P

        Modeling all of TP (true positive), FP (false positive) and their sum
        P = TP + FP (predicted positive) as varying linearly within each
        interval [A, B] between successive thresholds, we get

          Precision slope = dTP / dP
                          = (TP_B - TP_A) / (P_B - P_A)
                          = (TP - TP_A) / (P - P_A)
          Precision = (TP_A + slope * (P - P_A)) / P

        The area within the interval is (slope / total_pos_weight) times

          int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}
          int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}

        where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in

          int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)

        Bringing back the factor (slope / total_pos_weight) we'd put aside, we
        get

          slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight

        where dTP == TP_B - TP_A.

        Note that when P_A == 0 the above calculation simplifies into

          int_A^B{Precision.dTP} = int_A^B{slope * dTP} = slope * (TP_B - TP_A)

        which is really equivalent to imputing constant precision throughout the
        first bucket having >0 true positives.

        Returns:
          pr_auc: an approximation of the area under the P-R curve.
        """
        dtp = self.true_positives[: self.num_thresholds - 1] - self.true_positives[1:]
        p = tf.math.add(self.true_positives, self.false_positives)
        dp = p[: self.num_thresholds - 1] - p[1:]
        prec_slope = tf.math.divide_no_nan(dtp, tf.maximum(dp, 0), name="prec_slope")
        intercept = self.true_positives[1:] - tf.multiply(prec_slope, p[1:])

        safe_p_ratio = tf.where(
            tf.logical_and(p[: self.num_thresholds - 1] > 0, p[1:] > 0),
            tf.math.divide_no_nan(
                p[: self.num_thresholds - 1],
                tf.maximum(p[1:], 0),
                name="recall_relative_ratio",
            ),
            tf.ones_like(p[1:]),
        )

        pr_auc_increment = tf.math.divide_no_nan(
            prec_slope * (dtp + intercept * tf.math.log(safe_p_ratio)),
            tf.maximum(self.true_positives[1:] + self.false_negatives[1:], 0),
            name="pr_auc_increment",
        )

        if self.multi_label:
            by_label_auc = tf.reduce_sum(
                pr_auc_increment, name=self.name + "_by_label", axis=0
            )
            if self.label_weights is None:
                # Evenly weighted average of the label AUCs.
                return tf.reduce_mean(by_label_auc, name=self.name)
            else:
                # Weighted average of the label AUCs.
                return tf.math.divide_no_nan(
                    tf.reduce_sum(tf.multiply(by_label_auc, self.label_weights)),
                    tf.reduce_sum(self.label_weights),
                    name=self.name,
                )
        else:
            return tf.reduce_sum(pr_auc_increment, name="interpolate_pr_auc")

    def result(self):
        """
        Returns metric result
        """
        if (
            self.curve == metrics_utils.AUCCurve.PR
            and self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION
        ):
            # This use case is different and is handled separately.
            return self.interpolate_pr_auc()

        # Set `x` and `y` values for the curves based on `curve` config.
        recall = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_negatives),
        )
        if self.curve == metrics_utils.AUCCurve.ROC:
            fp_rate = tf.math.divide_no_nan(
                self.false_positives,
                tf.math.add(self.false_positives, self.true_negatives),
            )
            x = fp_rate
            y = recall
        else:  # curve == 'PR'.
            precision = tf.math.divide_no_nan(
                self.true_positives,
                tf.math.add(self.true_positives, self.false_positives),
            )
            x = recall
            y = precision

        # Find the rectangle heights based on `summation_method`.
        if self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION:
            # Note: the case ('PR', 'interpolation') has been handled above.
            heights = (y[: self.num_thresholds - 1] + y[1:]) / 2.0
        elif self.summation_method == metrics_utils.AUCSummationMethod.MINORING:
            heights = tf.minimum(y[: self.num_thresholds - 1], y[1:])
        # self.summation_method = metrics_utils.AUCSummationMethod.MAJORING:
        else:
            heights = tf.maximum(y[: self.num_thresholds - 1], y[1:])

        # Sum up the areas of all the rectangles.
        if self.multi_label:
            riemann_terms = tf.multiply(x[: self.num_thresholds - 1] - x[1:], heights)
            by_label_auc = tf.reduce_sum(
                riemann_terms, name=self.name + "_by_label", axis=0
            )

            if self.label_weights is None:
                # Unweighted average of the label AUCs.
                return tf.reduce_mean(by_label_auc, name=self.name)
            else:
                # Weighted average of the label AUCs.
                return tf.math.divide_no_nan(
                    tf.reduce_sum(tf.multiply(by_label_auc, self.label_weights)),
                    tf.reduce_sum(self.label_weights),
                    name=self.name,
                )
        else:
            return tf.reduce_sum(
                tf.multiply(x[: self.num_thresholds - 1] - x[1:], heights),
                name=self.name,
            )

    def reset_state(self):
        """
        Resets state
        """
        if self._built:
            confusion_matrix_variables = (
                self.true_positives,
                self.true_negatives,
                self.false_positives,
                self.false_negatives,
            )
            if self.multi_label:
                backend.batch_set_value(
                    [
                        (v, np.zeros((self.num_thresholds, self._num_labels)))
                        for v in confusion_matrix_variables
                    ]
                )
            else:
                backend.batch_set_value(
                    [
                        (v, np.zeros((self.num_thresholds,)))
                        for v in confusion_matrix_variables
                    ]
                )

    def get_config(self):
        """
        Gets metric config
        """
        if is_tensor_or_variable(self.label_weights):
            label_weights = backend.eval(self.label_weights)
        else:
            label_weights = self.label_weights
        config = {
            "num_thresholds": self.num_thresholds,
            "curve": self.curve.value,
            "summation_method": self.summation_method.value,
            "multi_label": self.multi_label,
            "num_labels": self.num_labels,
            "label_weights": label_weights,
            "from_logits": self._from_logits,
        }
        # optimization to avoid serializing a large number of generated
        # thresholds
        if self._init_from_thresholds:
            # We remove the endpoint thresholds as an inverse of how the
            # thresholds were initialized. This ensures that a metric
            # initialized from this config has the same thresholds.
            config["thresholds"] = self.thresholds[1:-1]
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
