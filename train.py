import os
import math
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from .dataset import OCRDataset, ocr_collate
from .utils import Tokenizer, save_checkpoint, load_checkpoint
from .model import CRNN


def train_main(
    annotations_file,
    img_root,
    chars,
    out_dir='checkpoints',
    epochs=50,
    batch_size=16,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = Tokenizer(chars)
    dataset = OCRDataset(annotations_file, img_root, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=ocr_collate)

    model = CRNN(num_classes=len(tokenizer.idx2char)).to(device)
    criterion = nn.CTCLoss(blank=tokenizer.blank_idx, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f'Epoch {epoch}'):
            images, labels, label_lengths = batch
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(images)  # T x B x C
            input_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long).to(device)

            loss = criterion(logits.log_softmax(2), labels, input_lengths, label_lengths)
            if not torch.isfinite(loss):
                print('Non-finite loss encountered; skipping batch and saving debug state')
                save_checkpoint(os.path.join(out_dir, 'debug.pth'), {'epoch': epoch})
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / max(1, len(loader))
        print(f'Epoch {epoch} average loss: {avg:.4f}')
        save_checkpoint(os.path.join(out_dir, f'ckpt_epoch_{epoch}.pth'), {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'tokenizer': tokenizer.__dict__,
        })

if __name__ == '__main__':
    # example usage
    train_main('dataset/sample_annotations.csv', 'dataset/', chars='abcdefghijklmnopqrstuvwxyz0123456789')
