import tensorflow as tf
from main.utils.neural_utils.custom_preprocessors.tensor_factory import (
    TensorFactory,
)

from keras import backend as K


def masked_sparse_categorical_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor):
    """The masked sparse categorical cross-entropy method calculates the
    cross-entropy loss on the predictions for true label of the masked items.

    Args:
        y_true (tf.Tensor): A tensor containing the true identities of the masked
            items. The tensor looks as follows:
            true_data = [
                [PADDING_TARGET, PADDING_TARGET, PADDING_TARGET, 4]
                [PADDING_TARGET, PADDING_TARGET, PADDING_TARGET, 8]
                [9, PADDING_TARGET, PADDING_TARGET, 12]
            ]
            We use boolean_mask to convert this tensor to the tensor [4, 8, 9, 12]
            containing just the true identities.
        y_pred (tf.Tensor): The predicted probability distributions for each of the
            masked items.
            So, the shape of this tensor is (num_masked_items_in_batch, num_items)
            because the probability distributions have dimension (num_items).

    Returns:
        tf.Tensor: A tensor containing the cross-entropy losses.
    """
    mask = tf.not_equal(y_true, TensorFactory.PADDING_TARGET)
    y_true_masked = tf.boolean_mask(y_true, mask)

    # For fixed-shape training outputs, mask logits on the same positions as labels.
    if (
        y_pred.shape.rank is not None
        and y_true.shape.rank is not None
        and y_pred.shape.rank == y_true.shape.rank + 1
    ):
        y_pred_masked = tf.boolean_mask(y_pred, mask)
    else:
        y_pred_masked = y_pred

    return tf.cond(
        tf.equal(tf.size(y_true_masked), 0),
        lambda: tf.cast(0.0, y_pred.dtype),
        lambda: K.mean(K.sparse_categorical_crossentropy(y_true_masked, y_pred_masked)),
    )
