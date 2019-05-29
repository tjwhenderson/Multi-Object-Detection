import torch
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

try:
    import _ext as _dcn_backend
except ImportError:
    print("Please compile source files before using DCN cuda extension.")
    raise


class _DCNv2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias,
                stride, padding, dilation, deformable_groups):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.deformable_groups = deformable_groups
        output = _dcn_backend.dcn_v2_forward(input, weight, bias,
                                             offset, mask,
                                             ctx.kernel_size[0], ctx.kernel_size[1],
                                             ctx.stride[0], ctx.stride[1],
                                             ctx.padding[0], ctx.padding[1],
                                             ctx.dilation[0], ctx.dilation[1],
                                             ctx.deformable_groups)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = \
            _dcn_backend.dcn_v2_backward(input, weight,
                                         bias,
                                         offset, mask,
                                         grad_output,
                                         ctx.kernel_size[0], ctx.kernel_size[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.deformable_groups)

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, \
               None, None, None, None,


dcn_v2_conv = _DCNv2.apply


class _DCNv2Pooling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rois, offset,
                spatial_scale,
                pooled_size,
                output_dim,
                no_trans,
                group_size=1,
                part_size=None,
                sample_per_part=4,
                trans_std=.0):
        ctx.spatial_scale = spatial_scale
        ctx.no_trans = int(no_trans)
        ctx.output_dim = output_dim
        ctx.group_size = group_size
        ctx.pooled_size = pooled_size
        ctx.part_size = pooled_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std

        output, output_count = \
            _dcn_backend.dcn_v2_psroi_pooling_forward(input, rois, offset,
                                                      ctx.no_trans, ctx.spatial_scale,
                                                      ctx.output_dim, ctx.group_size,
                                                      ctx.pooled_size, ctx.part_size,
                                                      ctx.sample_per_part, ctx.trans_std)
        ctx.save_for_backward(input, rois, offset, output_count)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, offset, output_count = ctx.saved_tensors
        grad_input, grad_offset = \
            _dcn_backend.dcn_v2_psroi_pooling_backward(grad_output,
                                                       input,
                                                       rois,
                                                       offset,
                                                       output_count,
                                                       ctx.no_trans,
                                                       ctx.spatial_scale,
                                                       ctx.output_dim,
                                                       ctx.group_size,
                                                       ctx.pooled_size,
                                                       ctx.part_size,
                                                       ctx.sample_per_part,
                                                       ctx.trans_std)

        return grad_input, None, grad_offset, \
               None, None, None, None, None, None, None, None


dcn_v2_pooling = _DCNv2Pooling.apply
