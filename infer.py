import torch
from .utils import Tokenizer, load_image_gray, resize_and_pad, load_checkpoint
from .model import CRNN
import sys


def infer(image_path, checkpoint, chars, device='cpu'):
    tokenizer = Tokenizer(chars)
    ckpt = load_checkpoint(checkpoint, device)
    model = CRNN(num_classes=len(tokenizer.idx2char))
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    img = load_image_gray(image_path)
    arr = resize_and_pad(img)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = logits.softmax(2)  # T x B x C
        _, max_idxs = probs.max(2)
        max_idxs = max_idxs.squeeze(1).cpu().numpy().tolist()
        text = tokenizer.decode(max_idxs)
        return text

if __name__ == '__main__':
    img = sys.argv[1]
    ckpt = sys.argv[2]
    print(infer(img, ckpt, chars='abcdefghijklmnopqrstuvwxyz0123456789'))
