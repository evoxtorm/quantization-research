import torch
import triton
import triton.language as tl

A = torch.tensor([[ 1., -5.,  2., 0.],
                  [ 3.,  1., -4., 2.],
                  [-2.,  6.,  1., 1.],
                  [ 0., -1.,  3., 7.]], device="cuda")

@triton.jit
def per_token_max(x_ptr, out_ptr, D: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, D)
    x = tl.load(x_ptr + row * D + d)
    tl.store(out_ptr + row, tl.max(tl.abs(x), axis=0))

@triton.jit
def per_group_channel_max(x_ptr, out_ptr, G: tl.constexpr, D: tl.constexpr):
    g = tl.program_id(0)
    t = tl.arange(0, G)
    d = tl.arange(0, D)
    offs = g * G * D + t[:, None] * D + d[None, :]
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + g * D + d, tl.max(tl.abs(x), axis=0))

out = torch.empty(4, device="cuda")
print(out, "this is out")
per_token_max[(4,)](A, out, D=4)                       # grid = 4 programs, one per row
print("triton per-token :", out.tolist())
print("torch  dim=1     :", A.abs().amax(dim=1).tolist())

out2 = torch.empty(2, 4, device="cuda")
print(out2, "this si out 2")
per_group_channel_max[(2,)](A, out2, G=2, D=4)         # grid = 2 programs, one per group
print("triton per-group :", out2.tolist())
print("torch  dim=1     :", A.reshape(2, 2, 4).abs().amax(dim=1).tolist())



# @triton.jit
# def show_workers(x_ptr, D: tl.constexpr):
#     row = tl.program_id(0)                 # my badge number
#     d = tl.arange(0, D)                    # [0, 1, 2, 3]
#     positions = row * D + d                # which tape positions I read
#     x = tl.load(x_ptr + positions)         # the numbers at those positions
#     tl.device_print("positions", positions)
#     tl.device_print("values", x)

# show_workers[(4,)](A, D=4)
# torch.cuda.synchronize()                   # wait so all the prints appear


B2 = torch.tensor([[1., -5.],
                   [3.,  1.],
                   [-2., 6.],
                   [0., -1.]], device="cuda")

out3 = torch.full((4,), float("nan"), device="cuda")
per_token_max[(4,)](B2, out3, D=2)      # 4 rows -> 4 workers; each row has D=2 values
print(out3)                             # expect [5, 3, 6, 1]
