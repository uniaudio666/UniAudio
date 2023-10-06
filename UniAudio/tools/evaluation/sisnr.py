import torch
import numpy as np

def estimate_si_sdr(
    clean_signal, enhanced_signal, sampling_rate=16000, zero_mean: bool = True):
    """Scale-Invariant Source-to-Noise Ratio(SI-SNR)
    or Scale-Invariant Signal Distortion Ratio(SI-SDR)

    References
    ----------
    - https://github.com/PyTorchLightning/metrics
    - Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
    and Signal Processing (ICASSP) 2019.
    """
    eps = np.finfo(dtype=np.float64).eps
    if zero_mean:
        clean_signal = clean_signal - np.mean(clean_signal, axis=-1, keepdims=True)
        enhanced_signal = enhanced_signal - np.mean(enhanced_signal, axis=-1, keepdims=True)
    alpha = (np.sum(enhanced_signal * clean_signal, axis=-1, keepdims=True) + eps) / (
        np.sum(clean_signal**2, axis=-1, keepdims=True) + eps
    )
    projection = alpha * clean_signal
    noise = enhanced_signal - projection
    ratio = (np.sum(projection**2, axis=-1) + eps) / (
        np.sum(noise**2, axis=-1) + eps
    )
    ratio = 10 * np.log10(ratio)
    return ratio

def get_sisnr(tarWav, refWav, eps=1e-8):

    """
    Arguments:
    x: separated signal, BS x S
    s: reference signal, BS x S
    Return:
    sisnr: BS tensor
    """
   # print('.............')
   # print('sis ',tarWav,refWav)
    from scipy.io import wavfile
    fs, s = wavfile.read(refWav)
    #print(fs,s.shape)
    fs, x = wavfile.read(tarWav)
    n = min(len(s), len(x))
    if len(x) != len(s):
        x = x[0:n]
        s = x[0:n]
    s = torch.from_numpy(s).unsqueeze(0).float()
    x = torch.from_numpy(x).unsqueeze(0).float()
    #print('s ',s.shape,x.shape) 
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimension mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True) ** 2 + eps)
    ans = 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))
    ans_ = ans.numpy()
    return ans_[0]

def sisnr_loss(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor, estimate value
          s: reference signal, N x S tensor, True value
    Return:
          sisnr: N tensor
    """
    if x.shape != s.shape:
        if x.shape[-1] > s.shape[-1]:
            x = x[:, :s.shape[-1]]
        else:
            s = s[:, :x.shape[-1]]
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)
    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    # x = torch.from_numpy(x)
    # x = torch.from_numpy(s)
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    ans = 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))
    ans_ = ans.numpy()
    return ans_[0]