# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Command-line for audio compression."""
import argparse
from pathlib import Path
import sys
import torchaudio
import os
import torch
import typing as tp
from collections import OrderedDict
from omegaconf import OmegaConf
from tqdm import tqdm

from models.soundstream import SoundStream


def build_codec_model(config):
    model = eval(config.generator.name)(**config.generator.config)
    return model


SUFFIX = '.ecdc'
def save_audio(wav: torch.Tensor, path: tp.Union[Path, str],
               sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(path, wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

def get_parser():
    parser = argparse.ArgumentParser(
        'encodec',
        description='High fidelity neural audio codec. '
                    'If input is a .ecdc, decompresses it. '
                    'If input is .wav, compresses it. If output is also wav, '
                    'do a compression/decompression cycle.')
    parser.add_argument(
        'input', type=Path,
        help='Input file, whatever is supported by torchaudio on your system.')
    parser.add_argument(
        'output', type=Path, nargs='?',
        help='Output file, otherwise inferred from input file.')
    parser.add_argument('--resume_path', type=str, required=True,
                        help='resume_path')
    parser.add_argument(
        '-r', '--rescale', action='store_true',
        help='Automatically rescale the output to avoid clipping.')
    parser.add_argument('-b', '--bw', type=str, default=None, help='Target bandwidth.')
    return parser


def fatal(*args):
    print(*args, file=sys.stderr)
    sys.exit(1)


def check_output_exists(args):
    if not args.output.parent.exists():
        fatal(f"Output folder for {args.output} does not exist.")
    if args.output.exists() and not args.force:
        fatal(f"Output file {args.output} exist. Use -f / --force to overwrite.")


def check_clipping(wav, args):
    if args.rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)


def check_clipping2(wav, rescale):
    if rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)


# def main():
    # args = get_parser().parse_args()
    # if not args.input.exists():
        # fatal(f"Input file {args.input} does not exist.")
    # # Compression
    # if args.output is None:
        # args.output = args.input.with_suffix(SUFFIX)
    # elif args.output.suffix.lower() not in [SUFFIX, '.wav']:
        # fatal(f"Output extension must be .wav or {SUFFIX}")
    # check_output_exists(args)
    # model = SoundStream(n_filters=32, D=256, ratios=[6, 5, 4, 2])
    # parameter_dict = torch.load(args.resume_path)
    # model.load_state_dict(parameter_dict) # load model

    # wav, sr = torchaudio.load(args.input)
    # print('wav:',wav.shape)
    # if sr != 24000:
        # wav = convert_audio(wav, sr, 16000, 1)
    # print('after convertion:',wav.shape)
    # wav = wav.cuda()
    # compressed = soundstream.encode(wav)
    # print('after compression:',compressed.shape)
    # out = soundstream.decode(compressed)
    # check_clipping(out, args)
    # save_audio(out, args.output, 24000, rescale=args.rescale)

def test_one(wav_root, store_root, rescale, args, config, soundstream):
    #compressing
    wav, sr = torchaudio.load(wav_root)
    if sr != soundstream.sample_rate:
        wav = torchaudio.transforms.Resample(sr, soundstream.sample_rate)(wav)
    if config.audio_norm_scale < 1.0:
        wav = wav * config.audio_norm_scale
    wav = wav.unsqueeze(1).cuda()
    print('wav ', wav.shape)
    compressed = soundstream.encode(wav, target_bw=args.bw)
    # print('compressed ', compressed.shape)
    # assert 1==2
    print(wav_root)
    print('finish compressing')
    out = soundstream.decode(compressed)
    out = out.detach().cpu().squeeze(0)
    check_clipping2(out, rescale)
    save_audio(out, store_root, 16000, rescale=rescale)
    print('finish decompressing')

def test_batch():
    args = get_parser().parse_args()

    args.bw = float(args.bw)
    print(f"Target bandwidth: {args.bw}")

    if not args.input.exists():
        fatal(f"Input file {args.input} does not exist.")

    input_lists = [i for i in os.listdir(args.input) if i.endswith('wav')]
    input_lists.sort()

    config_path = os.path.join(os.path.dirname(os.path.dirname(args.resume_path)), 'config.yaml')
    if not os.path.isfile(config_path):
        fatal(f"{config_path} file does not exist.")
    config = OmegaConf.load(config_path)
    soundstream = build_codec_model(config)
    # soundstream = SoundStream(n_filters=32, D=512, ratios=[8, 5, 4, 2]) 
    parameter_dict = torch.load(args.resume_path)
    # new_state_dict = OrderedDict()
    # for k, v in parameter_dict.items(): # k为module.xxx.weight, v为权重
        # name = k[7:] # 截取`module.`后面的xxx.weight
        # new_state_dict[name] = v
    # soundstream.load_state_dict(new_state_dict) # load model
    soundstream.load_state_dict(parameter_dict['codec_model']) # load model
    soundstream = soundstream.cuda()
    os.makedirs(args.output, exist_ok=True)
    for audio in input_lists:
        test_one(os.path.join(args.input,audio), os.path.join(args.output,audio), args.rescale, args, config, soundstream)

if __name__ == '__main__':
    #main()
    test_batch()
