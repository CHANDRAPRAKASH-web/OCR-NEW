import Levenshtein as Lev

def cer(pred, target):
    # character error rate
    return Lev.distance(pred, target) / max(1, len(target))

def wer(pred, target):
    # word error rate -- simple split
    p = pred.split()
    t = target.split()
    return Lev.distance(' '.join(p), ' '.join(t)) / max(1, len(t))
