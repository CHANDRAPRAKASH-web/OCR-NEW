# src/parser.py
import re
from typing import List, Dict, Any, Tuple

# Common lists / regexes
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}', re.I)
PHONE_RE = re.compile(
    r'(?:(?:\+?\d{1,3})?[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?(?:\d{3,4}[-.\s]?\d{3,4}|\d{10,12})'
)
PINCODE_RE = re.compile(r'\b\d{5,6}\b')  # simple India/Intl postal length heuristic

# designation keywords (extend this list)
DESIGNATION_KEYWORDS = [
    r'\bmanager\b', r'\bdirector\b', r'\bengineer\b', r'\bdeveloper\b', r'\bdesigner\b',
    r'\bhead\b', r'\bchief\b', r'\bcto\b', r'\bceo\b', r'\bcoo\b', r'\bfounder\b',
    r'\bpresident\b', r'\blead\b', r'\bsenior\b', r'\bjunior\b', r'\bintern\b',
    r'\bconsultant\b', r'\banalyst\b', r'\bspecialist\b', r'\bsales\b', r'\bmarketing\b'
]
DESIGNATION_RE = re.compile('|'.join(DESIGNATION_KEYWORDS), re.I)

# heuristics helpers
def is_likely_name(line: str) -> bool:
    # Name: mostly letters, Title Case (heuristic), short (1-4 words), not phone/email or contain digits
    if not line or any(ch.isdigit() for ch in line):
        return False
    words = line.strip().split()
    if len(words) > 5 or len(words) == 0:
        return False
    # prefer lines with capitalized words (Title Case) or common name patterns
    capital_words = sum(1 for w in words if w and w[0].isupper())
    return capital_words >= max(1, len(words) // 2)

def is_likely_address(line: str) -> bool:
    # Contains address-like tokens or digits or common words
    address_tokens = ['street','st','road','rd','lane','block','sector','area','city','state','pincode','zip','house','h no','landmark','near','plot']
    line_lower = line.lower()
    if any(tok in line_lower for tok in address_tokens):
        return True
    if PINCODE_RE.search(line):
        return True
    # if line has both letters and digits and is longer than short phrase, may be address
    if any(ch.isdigit() for ch in line) and len(line.split()) >= 2:
        return True
    return False

def group_lines_by_vertical_position(results: List[Dict[str,Any]], y_margin: int = 10) -> List[Tuple[int, List[Dict[str,Any]]]]:
    """
    Group OCR boxes into rows by their top y coordinate (approx).
    Returns: list of (y, [items]) sorted top->bottom
    """
    rows = []
    for r in results:
        # each r expected to have "box": [x0,y0,x1,y1] and "text_clean"
        if not r.get("text_clean"):
            continue
        x0,y0,x1,y1 = r.get("box", [0,0,0,0])
        mid_y = (y0 + y1) // 2
        placed = False
        for idx,(ry, items) in enumerate(rows):
            if abs(ry - mid_y) <= y_margin:
                items.append((r, mid_y))
                placed = True
                break
        if not placed:
            rows.append((mid_y, [(r, mid_y)]))
    # normalize and sort rows by y
    rows = sorted(rows, key=lambda x: x[0])
    # collapse items to list of dicts sorted by x within row
    out_rows = []
    for ry, items in rows:
        items_sorted = sorted(items, key=lambda it: it[0].get("box", [0])[0])
        out_rows.append((ry, [it[0] for it in items_sorted]))
    return out_rows

def lines_from_rows(rows: List[Tuple[int, List[Dict[str,Any]]]]) -> List[str]:
    lines = []
    for ry, items in rows:
        # join texts in that row with a space
        pieces = [it.get("text_clean","").strip() for it in items if it.get("text_clean")]
        if pieces:
            lines.append(" ".join(pieces))
    return lines

def parse_contact_fields(results: List[Dict[str,Any]]) -> Dict[str,Any]:
    """
    Input: results from pipeline.process_image: list of {box, text_raw, text_clean, confidence, ...}
    Output: structured dict with name, designation, address, mobiles, emails, raw_lines
    """
    parsed = {
        "name": None,
        "designation": None,
        "address": None,
        "mobile_numbers": [],
        "emails": [],
        "raw_lines": []
    }

    # 1. create lines from boxes (top-to-bottom)
    rows = group_lines_by_vertical_position(results, y_margin=12)
    lines = lines_from_rows(rows)
    parsed['raw_lines'] = lines.copy()

    # 2. detect emails and phones first (global scan)
    for ln in lines:
        for m in EMAIL_RE.findall(ln):
            if m not in parsed['emails']:
                parsed['emails'].append(m)
        for p in PHONE_RE.findall(ln):
            # PHONE_RE.findall returns tuples if groups; convert to string
            if isinstance(p, tuple):
                pstr = ''.join(p)
            else:
                pstr = p
            pstr = re.sub(r'[^\d\+]', '', pstr)
            # keep numbers longer than 6 digits
            if pstr and len(re.sub(r'\D','', pstr)) >= 6:
                if pstr not in parsed['mobile_numbers']:
                    parsed['mobile_numbers'].append(pstr)

    # 3. Heuristic for name + designation + address
    # We'll examine the top lines first because name and designation often at top
    # Keep a candidate list for name-like lines
    candidate_names = []
    for i, ln in enumerate(lines[:6]):  # top 6 lines prioritized
        if not ln:
            continue
        # skip if contains emails or phones
        if EMAIL_RE.search(ln) or PHONE_RE.search(ln):
            continue
        if is_likely_name(ln):
            candidate_names.append((i, ln))

    if candidate_names:
        # choose the first candidate name (top-most)
        parsed['name'] = candidate_names[0][1]

        # next line(s) after name could be designation
        name_idx = candidate_names[0][0]
        if name_idx + 1 < len(lines):
            next_line = lines[name_idx + 1]
            if DESIGNATION_RE.search(next_line):
                parsed['designation'] = next_line
            else:
                # sometimes designation and company combined, check two-line lookahead
                if name_idx + 2 < len(lines) and DESIGNATION_RE.search(lines[name_idx + 2]):
                    parsed['designation'] = lines[name_idx + 2]
    else:
        # fallback: try to find any line with designation keyword as name area
        for i, ln in enumerate(lines[:6]):
            if DESIGNATION_RE.search(ln):
                parsed['designation'] = ln
                # potential name above if exists
                if i - 1 >= 0 and is_likely_name(lines[i-1]):
                    parsed['name'] = lines[i-1]
                break

    # 4. Address detection: prefer blocks toward bottom/mid that contain address tokens
    addresses = []
    for ln in lines:
        if is_likely_address(ln):
            addresses.append(ln)
    if addresses:
        # join contiguous address-like lines
        parsed['address'] = ", ".join(addresses)
    else:
        # fallback: take bottom-most 2-3 lines that are not phone/email/name
        fallback = []
        for ln in reversed(lines[-5:]):  # last up to 5 lines
            if ln and ln not in parsed['emails'] and not any(ln in m for m in parsed['mobile_numbers']):
                # skip if it's name or designation
                if parsed['name'] and ln.strip() == parsed['name'].strip():
                    continue
                if parsed['designation'] and ln.strip() == parsed['designation'].strip():
                    continue
                fallback.append(ln)
            if len(fallback) >= 3:
                break
        if fallback:
            parsed['address'] = ", ".join(reversed(fallback))

    # 5. Clean up: ensure empty lists -> None or empty
    if not parsed['emails']:
        parsed['emails'] = []
    if not parsed['mobile_numbers']:
        parsed['mobile_numbers'] = []

    # 6. Final normalization: strip whitespace
    for k in ['name','designation','address']:
        if parsed[k]:
            parsed[k] = parsed[k].strip()

    return parsed
