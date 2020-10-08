import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt


def mode(obj, use_gpu=False):
    if use_gpu:
        obj = obj.cuda()

    return obj


def get_mask_from_lengths(lengths, pad=False):
    max_len = torch.max(lengths).item()
    if pad and max_len % 1 != 0:
        max_len += 1 - max_len % 1

    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def save_png(data, save_path):
    fig, axes = plt.subplots(1, len(data), figsize=(16,4))
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower',
                       interpolation='none')

    plt.savefig(save_path)






