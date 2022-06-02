import argparse
from collections import Counter

def calculate_score(hyp_m2_list, ref_m2_list):
    assert len(hyp_m2_list) == len(ref_m2_list)

    # Store global corpus level best counts here
    best_dict = Counter({"tp":0, "fp":0, "fn":0})
    sents = zip(hyp_m2_list, ref_m2_list)
    for sent_id, sent in enumerate(sents):
        hyp_edits = simplify_edits(sent[0])
        ref_edits = simplify_edits(sent[1])
        hyp_dict = process_edits(hyp_edits)
        ref_dict = process_edits(ref_edits)
        count_dict, cat_dict = evaluate_edits(hyp_dict, ref_dict, best_dict)
        best_dict += Counter(count_dict)
    result = get_full_result(best_dict)
    return result

def simplify_edits(sent):
    out_edits = []
    # Get the edit lines from an m2 block.
    edits = sent.split("\n")
    # Loop through the edits
    for edit in edits:
        # Preprocessing
        if edit.startswith("A "):
            edit = edit[2:].split("|||") # Ignore "A " then split.
            span = edit[0].split()
            start = int(span[0])
            end = int(span[1])
            cat = edit[1]
            cor = edit[2].replace(" ", "")
            coder = int(edit[-1])
            out_edit = [start, end, cat, cor, coder]
            out_edits.append(out_edit)
    return out_edits

def process_edits(edits):
    coder_dict = {}
    # Add an explicit noop edit if there are no edits.
    if not edits: edits = [[-1, -1, "noop", "-NONE-", 0]]
    # Loop through the edits
    for edit in edits:
        # Name the edit elements for clarity
        start = edit[0]
        end = edit[1]
        cat = edit[2]
        cor = edit[3]
        coder = edit[4]
        if coder not in coder_dict:
            coder_dict[coder] = {}
        if (start, end, cor) in coder_dict[coder].keys():
            coder_dict[coder][(start, end, cor)].append(cat)
        else:
            coder_dict[coder][(start, end, cor)] = [cat]
    return coder_dict

def evaluate_edits(hyp_dict, ref_dict, best):
    # Store the best sentence level scores and hyp+ref combination IDs
    # best_f is initialised as -1 cause 0 is a valid result.
    best_tp, best_fp, best_fn, best_f, best_hyp, best_ref = 0, 0, 0, -1, 0, 0
    best_cat = {}
    # skip not annotatable sentence
    if len(ref_dict.keys()) == 1:
        ref_id = list(ref_dict.keys())[0]
        if len(ref_dict[ref_id].keys()) == 1:
            cat = list(ref_dict[ref_id].values())[0][0]
            if cat == "NA":
                best_dict = {"tp":best_tp, "fp":best_fp, "fn":best_fn}
                return best_dict, best_cat

    # Compare each hyp and ref combination
    for hyp_id in hyp_dict.keys():
        for ref_id in ref_dict.keys():
            # Get the local counts for the current combination.
            tp, fp, fn, cat_dict = compareEdits(hyp_dict[hyp_id], ref_dict[ref_id])
            # Compute the global sentence scores
            p, r, f = computeFScore(
                tp+best["tp"], fp+best["fp"], fn+best["fn"], 0.5)
            if     (f > best_f) or \
                (f == best_f and tp > best_tp) or \
                (f == best_f and tp == best_tp and fp < best_fp) or \
                (f == best_f and tp == best_tp and fp == best_fp and fn < best_fn):
                best_tp, best_fp, best_fn = tp, fp, fn
                best_f, best_hyp, best_ref = f, hyp_id, ref_id
                best_cat = cat_dict
    # Save the best TP, FP and FNs as a dict, and return this and the best_cat dict
    best_dict = {"tp":best_tp, "fp":best_fp, "fn":best_fn}
    return best_dict, best_cat

def compareEdits(hyp_edits, ref_edits):
    tp = 0    # True Positives
    fp = 0    # False Positives
    fn = 0    # False Negatives
    cat_dict = {} # {cat: [tp, fp, fn], ...}

    for h_edit, h_cats in hyp_edits.items():
        # noop hyp edits cannot be TP or FP
        if h_cats[0] == "noop": continue
        # TRUE POSITIVES
        if h_edit in ref_edits.keys():
            # On occasion, multiple tokens at same span.
            for h_cat in ref_edits[h_edit]: # Use ref dict for TP
                tp += 1
                # Each dict value [TP, FP, FN]
                if h_cat in cat_dict.keys():
                    cat_dict[h_cat][0] += 1
                else:
                    cat_dict[h_cat] = [1, 0, 0]
        # FALSE POSITIVES
        else:
            # On occasion, multiple tokens at same span.
            for h_cat in h_cats:
                fp += 1
                # Each dict value [TP, FP, FN]
                if h_cat in cat_dict.keys():
                    cat_dict[h_cat][1] += 1
                else:
                    cat_dict[h_cat] = [0, 1, 0]
    for r_edit, r_cats in ref_edits.items():
        # noop ref edits cannot be FN
        if r_cats[0] == "noop": continue
        # FALSE NEGATIVES
        if r_edit not in hyp_edits.keys():
            # On occasion, multiple tokens at same span.
            for r_cat in r_cats:
                fn += 1
                # Each dict value [TP, FP, FN]
                if r_cat in cat_dict.keys():
                    cat_dict[r_cat][2] += 1
                else:
                    cat_dict[r_cat] = [0, 0, 1]
    return tp, fp, fn, cat_dict

def computeFScore(tp, fp, fn, beta):
    p = float(tp)/(tp+fp) if fp else 1.0
    r = float(tp)/(tp+fn) if fn else 1.0
    f = float((1+(beta**2))*p*r)/(((beta**2)*p)+r) if p+r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)

def get_full_result(best):
    true_positive = best["tp"]
    false_positive = best["fp"]
    false_negative = best["fn"]
    precision, recall, f0_5 = computeFScore(best["tp"], best["fp"], best["fn"], 0.5)
    return {'tp':true_positive, 'fp':false_positive, 'fn': false_negative, 'precision':precision, 'recall': recall, 'f0_5':f0_5}

# hyp_m2_list = ['S 这 样 ， 你 就 会 尝 到 泰 国 人 死 爱 的 味 道 。\nT0-A0 这 样 ， 你 会 尝 到 泰 国 人 死 爱 的 味 道 。\nA 4 5|||R|||-NONE-|||REQUIRED|||-NONE-|||0', 'S 这 样 ， 你样 ， 你 会 尝 到 泰 国 人 死 爱 的 味 道 。\nA 4 5|||R|||-NONE-|||REQUIRED|||-NONE-|||0']
# ref_m2_list = ['S 这 样 ， 你 就 会 尝 到 泰 国 人 死 爱 的 味 道 。\nT0-A0 这 样 ， 你 会 尝 到 泰 国 人 死 爱 的 味 道 。\nA 4 5|||R|||-NONE-|||REQUIRED|||-NONE-|||0', 'S 这 样 ， 你样 ， 你 会 尝 到 泰 国 人 死 爱 的 味 道 。\nA 4 5|||R|||-NONE-|||REQUIRED|||-NONE-|||0']

# calculate_score(hyp_m2_list, ref_m2_list)