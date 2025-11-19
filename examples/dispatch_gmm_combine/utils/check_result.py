import argparse
import torch


def output_error_item(output_tensor, golden_tensor, threshold=0.1, max_print=100, epsilon=1e-8):
    print("原始形状:", output_tensor.shape, output_tensor.shape)

    if len(output_tensor.shape) != len(golden_tensor.shape):
        raise AssertionError("张量的维度数必须相同")

    slices = []
    for a_dim, b_dim in zip(output_tensor.shape, golden_tensor.shape):
        min_dim = min(a_dim, b_dim)
        slices.append(slice(0, min_dim))

    a_slice = output_tensor[slices]
    b_slice = golden_tensor[slices]
    print("比较的截取形状:", a_slice.shape)

    diff = (a_slice - b_slice).abs()
    relative_error = diff / (b_slice.abs() + epsilon)

    mask = relative_error >= threshold
    indices = torch.nonzero(mask, as_tuple=True)
    a_values = a_slice[indices]
    b_values = b_slice[indices]
    error_values = relative_error[indices]

    print(f"发现 {len(a_values)} 个差异超过{threshold * 100}%的元素")


def read_binary_file(file_path, dtype=torch.float32):
    try:
        with open(file_path, "rb") as f:
            binary_data = f.read()
        writable_data = bytearray(binary_data)
        tensor = torch.frombuffer(writable_data, dtype=dtype)
        return tensor
    except FileNotFoundError:
        print(f"The file {file_path} does not exist!")
        return None


parser = argparse.ArgumentParser()
parser.add_argument('--rank_size', type=int, required=True)
parser.add_argument('--dataType', type=int, required=True)
parser.add_argument('--m', type=int, required=True)
parser.add_argument('--k', type=int, required=True)
parser.add_argument('--n', type=int, required=True)
parser.add_argument('--expert_per_rank', type=int, required=True)
parser.add_argument('--EP', type=int, required=True)

args = parser.parse_args()
for i in range(args.rank_size):
    print(f"================{i} rank ================")
    a = read_binary_file(f"./out/output_{i}.bin", dtype=torch.float16)
    b = read_binary_file(f"./utils/test_data/unpermuted_token_{i}_{args.dataType}_1_{args.m}_{args.k}_{args.n}_{args.expert_per_rank}_{args.EP}_1.bin", dtype=torch.float16)
    output_error_item(a, b)
