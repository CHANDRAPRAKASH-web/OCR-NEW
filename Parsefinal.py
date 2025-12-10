# src/parser.py
import re
from typing import List, Dict, Any
from statistics import mean

EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}', re.I)
PHONE_RE = re.compile(
    r'(?:(?:\+?\d{1,4})?[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?(?:\d{3,4}[-.\s]?\d{3,4}|\d{6,14})'
)
PINCODE_RE = re.compile(r'\b\d{5,6}\b')
WEBSITE_RE = re.compile(r'(https?://[^\s,;]+|www\.[^\s,;]+)', re.I)
LINKEDIN_RE = re.compile(r'(linkedin\.com/[^\s,;]+)', re.I)
GSTIN_RE = re.compile(r'\b[0-9A-Z]{15}\b')  # simple GSTIN heuristic (uppercase alnum 15 chars)
CIN_RE = re.compile(r'\b[A-Z0-9]{16,21}\b')  # heuristic for Indian CIN (length varies)
DESIGNATION_KEYWORDS = [
    r'\bmanager\b', r'\bdirector\b', r'\bengineer\b', r'\bdeveloper\b', r'\bdesigner\b',
    r'\bhead\b', r'\bchief\b', r'\bcto\b', r'\bceo\b', r'\bcoo\b', r'\bfounder\b',
    r'\bpresident\b', r'\blead\b', r'\bsenior\b', r'\bjunior\b', r'\bintern\b',
    r'\bconsultant\b', r'\banalyst\b', r'\bspecialist\b', r'\bsales\b', r'\bmarketing\b'
]
DESIGNATION_RE = re.compile('|'.join(DESIGNATION_KEYWORDS), re.I)

def is_likely_name(line: str) -> bool:
    if not line or any(ch.isdigit() for ch in line):
        return False
    words = [w for w in line.strip().split() if w]
    if not words or len(words) > 6:
        return False
    capital_words = sum(1 for w in words if w[0].isupper())
    return capital_words >= max(1, len(words)//2)

def is_likely_address(line: str) -> bool:
    address_tokens = ['street','st','road','rd','lane','block','sector','area','city','state','pincode','zip','house','landmark','near','plot','colony','chennai','coimbatore','bangalore','bengaluru']
    ln = line.lower()
    if any(tok in ln for tok in address_tokens):
        return True
    if PINCODE_RE.search(line):
        return True
    if any(ch.isdigit() for ch in line) and len(line.split()) >= 2:
        return True
    return False

def group_lines_by_vertical_position(results: List[Dict[str,Any]], y_margin: int = 12):
    rows = []
    for r in results:
        text = (r.get("text_clean") or "").strip()
        if not text:
            continue
        box = r.get("box", [0,0,0,0])
        x0,y0,x1,y1 = box if len(box) >= 4 else (0,0,0,0)
        mid_y = (y0 + y1)//2
        placed = False
        for idx,(ry, items) in enumerate(rows):
            if abs(ry - mid_y) <= y_margin:
                items.append((r, mid_y))
                placed = True
                break
        if not placed:
            rows.append((mid_y, [(r, mid_y)]))
    rows = sorted(rows, key=lambda x: x[0])
    out_rows = []
    for ry, items in rows:
        items_sorted = sorted(items, key=lambda it: it[0].get("box", [0])[0])
        out_rows.append((ry, [it[0] for it in items_sorted]))
    return out_rows

def lines_from_rows(rows):
    lines = []
    for ry, items in rows:
        pieces = [it.get("text_clean","").strip() for it in items if it.get("text_clean")]
        if pieces:
            lines.append(" ".join(pieces))
    return lines

def parse_contact_fields(results: List[Dict[str,Any]]) -> Dict[str,Any]:
    parsed = {
        "name": None,
        "designation": None,
        "company": None,
        "mobile": [],
        "email": [],
        "address": None,
        "location": None,
        "website": [],
        "social": {},
        "extras": {},
        "language_detected": [],  # placeholder (can be filled with langdetect if added)
        "confidence": None,
        "raw_text": "",
        "notes": []
    }

    # raw_text: concatenate all text_clean in reading order
    rows = group_lines_by_vertical_position(results, y_margin=12)
    lines = lines_from_rows(rows)
    parsed["raw_text"] = "\n".join(lines)
    parsed["raw_lines"] = lines.copy()  # keep raw lines too

    # collect emails, phones, websites, socials, extras (gstin/cin)
    for ln in lines:
        for m in EMAIL_RE.findall(ln):
            if m not in parsed["email"]:
                parsed["email"].append(m)
        for p in PHONE_RE.findall(ln):
            if isinstance(p, tuple):
                pstr = ''.join(p)
            else:
                pstr = p
            pstr = re.sub(r'[^\d\+]', '', pstr)
            if pstr and len(re.sub(r'\D', '', pstr)) >= 6:
                if pstr not in parsed["mobile"]:
                    parsed["mobile"].append(pstr)
        for w in WEBSITE_RE.findall(ln):
            w = w.strip().rstrip(',.')
            if w not in parsed["website"]:
                parsed["website"].append(w)
        li = LINKEDIN_RE.search(ln)
        if li:
            parsed["social"]["linkedin"] = li.group(1)
        gst = GSTIN_RE.search(ln)
        if gst:
            parsed["extras"]["gstin"] = gst.group(0)
        cin = CIN_RE.search(ln)
        if cin:
            parsed["extras"]["cin"] = cin.group(0)

    # name and designation heuristics: top lines are prime candidates
    candidate_names = []
    for i, ln in enumerate(lines[:6]):
        if not ln:
            continue
        if EMAIL_RE.search(ln) or PHONE_RE.search(ln) or WEBSITE_RE.search(ln):
            continue
        if is_likely_name(ln):
            candidate_names.append((i, ln))
    if candidate_names:
        parsed["name"] = candidate_names[0][1]
        name_idx = candidate_names[0][0]
        # next lines could be designation and/or company
        if name_idx + 1 < len(lines):
            nxt = lines[name_idx + 1]
            if DESIGNATION_RE.search(nxt):
                parsed["designation"] = nxt
                # company maybe next
                if name_idx + 2 < len(lines):
                    cand_company = lines[name_idx + 2].strip()
                    if cand_company and cand_company not in parsed["email"] and cand_company not in parsed["mobile"]:
                        parsed["company"] = cand_company
            else:
                # if next line contains company-like tokens (Pvt Ltd, Ltd, Inc, LLC), treat as company
                if re.search(r'\b(pvt|ltd|limited|inc|llc|corporation|corp|co)\b', nxt, re.I):
                    parsed["company"] = nxt
                else:
                    # maybe designation in same line
                    if DESIGNATION_RE.search(nxt):
                        parsed["designation"] = nxt
    else:
        # fallback: find designation anywhere
        for i, ln in enumerate(lines[:8]):
            if DESIGNATION_RE.search(ln):
                parsed["designation"] = ln
                if i - 1 >= 0 and is_likely_name(lines[i-1]):
                    parsed["name"] = lines[i-1]
                # company guess: line above or below
                if i + 1 < len(lines):
                    parsed["company"] = lines[i+1]
                break

    # address detection: prefer middle->bottom blocks
    address_lines = []
    for ln in lines:
        if is_likely_address(ln):
            address_lines.append(ln)
    if address_lines:
        parsed["address"] = ", ".join(address_lines)
        # try to extract city, state from address (very heuristic: last comma-separated tokens)
        parts = parsed["address"].split(',')
        if len(parts) >= 2:
            parsed["location"] = ", ".join([p.strip() for p in parts[-2:]])
    else:
        # fallback: bottom lines (excluding email/phone)
        fallback = []
        for ln in reversed(lines[-6:]):
            if not ln:
                continue
            if ln in parsed["email"] or any(ln in m for m in parsed["mobile"]):
                continue
            fallback.append(ln)
            if len(fallback) >= 3:
                break
        if fallback:
            parsed["address"] = ", ".join(reversed(fallback))
            parsed["location"] = parsed["address"].split(',')[-1].strip()

    # ensure unique email/mobile/website arrays
    parsed["email"] = list(dict.fromkeys(parsed["email"]))
    parsed["mobile"] = list(dict.fromkeys(parsed["mobile"]))
    parsed["website"] = list(dict.fromkeys(parsed["website"]))

    # compute global confidence from results confidences (if present)
    confs = []
    for r in results:
        c = r.get("confidence", None)
        if isinstance(c, (int, float)) and c >= 0:
            confs.append(float(c))
    if confs:
        # normalize to 0-1 if confidences appear as 0-100 ints
        avg = mean(confs)
        if avg > 1.1:
            parsed["confidence"] = round(avg / 100.0, 2)
        else:
            parsed["confidence"] = round(avg, 2)
    else:
        parsed["confidence"] = None

    # notes: light heuristics
    if not parsed["name"]:
        parsed["notes"].append("name_not_detected")
    if not parsed["email"]:
        parsed["notes"].append("email_not_detected")
    if not parsed["mobile"]:
        parsed["notes"].append("mobile_not_detected")

    return parsed
