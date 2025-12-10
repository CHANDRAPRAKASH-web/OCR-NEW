import os
import pandas as pd
from torch.utils.data import Dataset
import torch
from .utils import load_image_gray, resize_and_pad

class OCRDataset(Dataset):
    def __init__(self, annotations_file, img_root, tokenizer, target_h=32, max_w=320):
        self.df = pd.read_csv(annotations_file)
        self.img_root = img_root
        self.tokenizer = tokenizer
        self.target_h = target_h
        self.max_w = max_w

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row['image_path'])
        img = load_image_gray(img_path)
        arr = resize_and_pad(img, target_h=self.target_h, max_w=self.max_w)
        tensor = torch.from_numpy(arr).unsqueeze(0).float()  # 1, H, W
        label = row['transcription']
        label_ids = self.tokenizer.encode(label)
        return tensor, torch.tensor(label_ids, dtype=torch.long)


def ocr_collate(batch):
    images = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    images = torch.stack(images, dim=0)
    # Convert labels to concatenated vector + lengths (needed for CTC)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    concatenated = torch.cat(labels) if len(labels) > 0 else torch.tensor([], dtype=torch.long)
    return images, concatenated, label_lengths
