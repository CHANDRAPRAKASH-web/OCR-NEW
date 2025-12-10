import os
import json
import random
from PIL import Image, ImageOps
import numpy as np
import torch

class Tokenizer:
    def __init__(self, chars):
        # chars: string of characters (e.g. "abcdefghijklmnopqrstuvwxyz0123456789")
        self.chars = chars
        self.blank = "<BLK>"
        self.pad = "<PAD>"
        self.idx2char = [self.pad, self.blank] + list(chars)
        self.char2idx = {c: i for i, c in enumerate(self.idx2char)}
        self.pad_idx = 0
        self.blank_idx = 1

    def encode(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, idxs):
        # collapse blanks and remove pad
        out = []
        prev = None
        for i in idxs:
            if i == self.blank_idx or i == self.pad_idx:
                prev = i
                continue
            if i != prev:
                out.append(self.idx2char[i])
            prev = i
        return ''.join(out)


def load_image_gray(path):
    im = Image.open(path).convert('L')
    return im


def resize_and_pad(img, target_h=32, max_w=320):
    # preserve aspect ratio by resizing to target_h then pad width to max_w
    w, h = img.size
    new_h = target_h
    new_w = max(1, int(w * (new_h / float(h))))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    if new_w < max_w:
        result = Image.new('L', (max_w, new_h), color=255)
        result.paste(img, (0, 0))
        img = result
    else:
        # if wider than max_w, center-crop (rare) or resize down
        img = img.resize((max_w, new_h), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # normalize
    return arr


def save_checkpoint(path, state):
    torch.save(state, path)


def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location
