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
CIN_RE = re.compile(r'\b[A-Z0-9]{16,21}\b')  # heuristic for Indian CIN

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
    capital_words = sum(1 for w in words if w and w[0].isupper())
    return capital_words >= max(1, len(words)//2)

def is_likely_address(line: str) -> bool:
    address_tokens = ['street','st','road','rd','lane','block','sector','area','city','state','pincode','zip','house','landmark','near','plot','colony']
    ln = (line or "").lower()
    if any(tok in ln for tok in address_tokens):
        return True
    if PINCODE_RE.search(line or ""):
        return True
    if any(ch.isdigit() for ch in line or "") and len((line or "").split()) >= 2:
        return True
    return False

def group_lines_by_vertical_position(results: List[Dict[str,Any]], y_margin: int = 12):
    rows = []
    for r in results:
        text = (r.get("text_clean") or "").strip()
        if not text:
            continue
        box = r.get("box", [0,0,0,0])
        if len(box) < 4:
            x0,y0,x1,y1 = 0,0,0,0
        else:
            x0,y0,x1,y1 = box[0], box[1], box[2], box[3]
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
    """
    Robust parser — always creates `parsed` dict first so it cannot be referenced before assignment.
    """
    # create parsed immediately so it's always defined
    parsed: Dict[str,Any] = {
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
        "language_detected": [],
        "confidence": None,
        "raw_text": "",
        "raw_lines": [],
        "notes": []
    }

    try:
        # build raw_text & lines
        rows = group_lines_by_vertical_position(results or [], y_margin=12)
        lines = lines_from_rows(rows)
        parsed["raw_text"] = "\n".join(lines)
        parsed["raw_lines"] = lines.copy()

        # extract emails, phones, websites, socials, gstin/cin
        for ln in lines:
            # emails
            for m in EMAIL_RE.findall(ln):
                if m not in parsed["email"]:
                    parsed["email"].append(m)
            # phones
            for p in PHONE_RE.findall(ln):
                if isinstance(p, tuple):
                    pstr = ''.join(p)
                else:
                    pstr = p
                pstr = re.sub(r'[^\d\+]', '', pstr)
                if pstr and len(re.sub(r'\D', '', pstr)) >= 6:
                    if pstr not in parsed["mobile"]:
                        parsed["mobile"].append(pstr)
            # websites
            for wmatch in WEBSITE_RE.findall(ln):
                w = wmatch.strip().rstrip(',.')
                if w and w not in parsed["website"]:
                    parsed["website"].append(w)
            # linkedin
            li = LINKEDIN_RE.search(ln)
            if li and "linkedin" not in parsed["social"]:
                parsed["social"]["linkedin"] = li.group(1)
            # gstin/cin
            gst = GSTIN_RE.search(ln)
            if gst:
                parsed["extras"]["gstin"] = gst.group(0)
            cin = CIN_RE.search(ln)
            if cin:
                parsed["extras"]["cin"] = cin.group(0)

        # heuristics for name and designation
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
            if name_idx + 1 < len(lines):
                nxt = lines[name_idx + 1]
                if DESIGNATION_RE.search(nxt):
                    parsed["designation"] = nxt
                    if name_idx + 2 < len(lines):
                        cand_comp = lines[name_idx + 2].strip()
                        if cand_comp and not EMAIL_RE.search(cand_comp) and not PHONE_RE.search(cand_comp):
                            parsed["company"] = cand_comp
                else:
                    if re.search(r'\b(pvt|ltd|limited|inc|llc|corporation|corp|co)\b', nxt, re.I):
                        parsed["company"] = nxt
                    elif DESIGNATION_RE.search(nxt):
                        parsed["designation"] = nxt
        else:
            for i, ln in enumerate(lines[:8]):
                if DESIGNATION_RE.search(ln):
                    parsed["designation"] = ln
                    if i - 1 >= 0 and is_likely_name(lines[i-1]):
                        parsed["name"] = lines[i-1]
                    if i + 1 < len(lines):
                        parsed["company"] = lines[i+1]
                    break

        # address extraction
        address_lines = []
        for ln in lines:
            if is_likely_address(ln):
                address_lines.append(ln)
        if address_lines:
            parsed["address"] = ", ".join(address_lines)
            parts = parsed["address"].split(',')
            if len(parts) >= 2:
                parsed["location"] = ", ".join([p.strip() for p in parts[-2:]])
        else:
            fallback = []
            for ln in reversed(lines[-6:]):
                if not ln or ln in parsed["email"] or any(ln in m for m in parsed["mobile"]):
                    continue
                fallback.append(ln)
                if len(fallback) >= 3:
                    break
            if fallback:
                parsed["address"] = ", ".join(reversed(fallback))
                parsed["location"] = parsed["address"].split(',')[-1].strip()

        # unique lists
        parsed["email"] = list(dict.fromkeys(parsed["email"]))
        parsed["mobile"] = list(dict.fromkeys(parsed["mobile"]))
        parsed["website"] = list(dict.fromkeys(parsed["website"]))

        # compute confidence if present
        confs = []
        for r in results or []:
            c = r.get("confidence", None)
            if isinstance(c, (int, float)) and c >= 0:
                confs.append(float(c))
        if confs:
            avg = mean(confs)
            if avg > 1.1:  # likely 0-100 -> convert
                parsed["confidence"] = round(avg / 100.0, 2)
            else:
                parsed["confidence"] = round(avg, 2)
        else:
            parsed["confidence"] = None

        # notes
        if not parsed["name"]:
            parsed["notes"].append("name_not_detected")
        if not parsed["email"]:
            parsed["notes"].append("email_not_detected")
        if not parsed["mobile"]:
            parsed["notes"].append("mobile_not_detected")

    except Exception as ex:
        # if anything went wrong, record note and continue returning safe parsed dict
        parsed.setdefault("notes", [])
        parsed["notes"].append(f"parser_error:{type(ex).__name__}")
        # don't re-raise here — return the best possible partial result
    return parsed
