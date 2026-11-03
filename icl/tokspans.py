"""Tokenization-safe span/target helpers shared by the runners and tested in tests/test_tokspans.py."""
def is_bare_ws(tok,tid): d=tok.decode([tid]); return d!='' and d.strip()==''
def label_spans(tok,ids,pairs,labpos):
    """(s0,s1,label) per demo from the FULL prompt tokenization `ids`; skips a leading bare-whitespace token
    (Qwen: ' 7' -> [' ','7']) so the span starts on content; asserts the decoded span equals the label."""
    out=[]
    for (x,y),lp_ in zip(pairs,labpos):
        n=len(tok(' '+y)['input_ids']); s0,s1=lp_,lp_+n
        while s1-s0>1 and is_bare_ws(tok,ids[s0]): s0+=1
        dec=tok.decode(ids[s0:s1]); assert dec.strip()==y.strip() and dec.strip()!='', ('span mismatch',dec,y,ids[s0:s1]); out.append((s0,s1,y))
    return out
def target_ids(tok,ids,target):
    """token ids of ' '+target as the model would see them AFTER the prompt `ids` (full-text tokenization, suffix),
    falling back to standalone tokenization when the prompt does not re-tokenize to itself. Returns (ids, used_fallback)."""
    ptxt=tok.decode(ids); full=tok(ptxt+' '+target,add_special_tokens=False)['input_ids']; ids=list(ids); bos=tok.bos_token_id
    if bos is not None and ids and ids[0]==bos and (not full or full[0]!=bos): ids=ids[1:]   # Llama-style tokenizers prepend BOS to the prompt; decode() may drop it
    if full[:len(ids)]==ids: tgt,fb=full[len(ids):],False
    else: tgt,fb=tok(' '+target,add_special_tokens=False)['input_ids'],True
    dec=tok.decode(tgt); assert dec.strip()==target.strip() and dec.strip()!='' and len(tgt)>=1, ('target mismatch',dec,target)
    return tgt,fb
