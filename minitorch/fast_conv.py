import numpy as np
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
    MAX_DIMS,
)
from .tensor_functions import Function
from numba import njit, prange


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
count = njit(inline="always")(count)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


@njit(parallel=True)
def tensor_conv1d(
    out,
    out_shape,
    out_strides,
    out_size,
    input,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `weight` tensor.
        weight_shape (array): shape for `weight` tensor.
        weight_strides (array): strides for `weight` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape
    batch_, out_channels, out_width = out_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # print("in shape: ", input_shape)
    # print("out shape: ", out_shape)
    # print("Weight: ", weight_shape)

    for i in prange(len(out)):
        out_index = np.empty(len(out_shape), np.int32)
        if reverse:
            a = len(out) - 1 - i
            count(a, out_shape, out_index)
        else:
            count(i, out_shape, out_index)
        for j in range(in_channels * kw):
            if reverse:
                j = (in_channels * kw) - 1 - j
            w_ind = np.empty(len(weight_shape), np.int32)
            count(j, weight_shape, w_ind)
            w_ind[0] = out_index[1]
            i_ind = np.copy(w_ind)
            i_ind[0] = out_index[0]
            if reverse:
                i_ind[-1] = out_index[-1] - (kw - 1 - w_ind[-1])
                print(
                    "out_ind: ",
                    out_index[-1],
                    " i_ind: ",
                    i_ind[-1],
                    " and w_ind: ",
                    w_ind[-1],
                )
            else:
                i_ind[-1] = out_index[-1] + w_ind[-1]
            if i_ind[-1] < width and i_ind[-1] >= 0:
                inp_pos = index_to_position(i_ind, s1)
                weight_pos = index_to_position(w_ind, s2)
                out_pos = index_to_position(out_index, out_strides)
                out[out_pos] += input[inp_pos] * weight[weight_pos]


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input (:class:`Tensor`) : batch x in_channel x h x w
            weight (:class:`Tensor`) : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


@njit(parallel=True)
def tensor_conv2d(
    out,
    out_shape,
    out_strides,
    out_size,
    input,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `input` tensor.
        weight_shape (array): shape for `input` tensor.
        weight_strides (array): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides

    if reverse:
        weight_range = range((in_channels * kw * kh) - 1, -1, -1)
    else:
        weight_range = range(in_channels * kw * kh)

    for i in prange(len(out)):
        out_index = np.empty(len(out_shape), np.int32)
        if reverse:
            a = len(out) - 1 - i
            count(a, out_shape, out_index)
        else:
            count(i, out_shape, out_index)
        out_pos = index_to_position(out_index, out_strides)
        for j in weight_range:
            w_ind = np.empty(len(weight_shape), np.int32)
            count(j, weight_shape, w_ind)
            w_ind[0] = out_index[1]
            i_ind = np.copy(out_index)
            i_ind[-3] = w_ind[-3]
            if reverse:
                i_ind[-1] = out_index[-1] - (kw - 1 - w_ind[-1])
                i_ind[-2] = out_index[-2] - (kh - 1 - w_ind[-2])
            else:
                i_ind[-2] = w_ind[-2] + out_index[-2]
                i_ind[-1] = w_ind[-1] + out_index[-1]
            if (
                i_ind[-1] < width
                and i_ind[-2] < height
                and i_ind[-1] >= 0
                and i_ind[-2] >= 0
            ):
                inp_pos = index_to_position(i_ind, s1)
                weight_pos = index_to_position(w_ind, s2)
                out[out_pos] += input[inp_pos] * weight[weight_pos]

    # TODO: Implement for Task 4.2.
    # raise NotImplementedError('Need to implement for Task 4.2')


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input (:class:`Tensor`) : batch x in_channel x h x w
            weight (:class:`Tensor`) : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
