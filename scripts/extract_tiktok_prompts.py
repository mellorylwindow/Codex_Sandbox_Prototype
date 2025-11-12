import csv, json, math, re
from collections import Counter, defaultdict, deque
from pathlib import Path

# ===========================
# PATHS
# ===========================
RAW_DIR = Path("data/tiktok_archive/raw")
OUT_DIR = Path("data/tiktok_archive/derived")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "tiktok_prompts.csv"
CAT_CSV = OUT_DIR / "tiktok_prompts_categorized.csv"
CLUSTER_JSON = OUT_DIR / "tiktok_clusters.json"
GROUPS_JSON = OUT_DIR / "tiktok_groups.json"
QUICKCAST_MD = OUT_DIR / "quickcast.md"
QUICKCAST_CLUSTER_MD = OUT_DIR / "quickcast_by_cluster.md"
AUTOCAST_MD = OUT_DIR / "autocast_scripts.md"
WHISPER_JSON = OUT_DIR / "autocast_whisper.json"
DEDUP_FILE = OUT_DIR / "seen_prompts.json"

SRT_DIR = OUT_DIR / "srt"
SRT_DIR.mkdir(exist_ok=True)

# ===========================
# CATEGORY PATTERNS
# ===========================
CATEGORY_PATTERNS = {
    "prompt":  [r"^\s*prompt[:\-–]\s*(.+)$"],
    "hook":    [r"^\s*hook[:\-–]\s*(.+)$"],
    "cta":     [r"^\s*cta[:\-–]\s*(.+)$"],
    "script":  [r"^\s*script[:\-–]\s*(.+)$"],
    "idea":    [r"^\s*idea[:\-–]\s*(.+)$"],
}
COMPILED_CATS = {c: [re.compile(p, re.IGNORECASE) for p in pats] for c, pats in CATEGORY_PATTERNS.items()}

def clean_text(t:str)->str:
    return re.sub(r"\s+", " ", t).strip()

def categorize_line(line:str):
    for cat, pats in COMPILED_CATS.items():
        for rx in pats:
            m = rx.match(line)
            if m:
                return cat, clean_text(m.group(1))
    return "unknown", clean_text(line)

# ===========================
# EMOTION + SENTIMENT
# ===========================
def detect_emotional_function(text:str, category:str)->str:
    t = text.lower()
    if category == "cta": return "directive"
    if category == "hook": return "activation"
    if category == "prompt": return "clarity"
    rules = [
        ("grounding",   ["deep breath","pause","ground","calm","breathe","slow down"]),
        ("directive",   ["stop","wait","listen","look","now","must"]),
        ("clarity",     ["explain","how to","why","understand","advice"]),
        ("regulation",  ["panic","overwhelm","can't","stuck","freeze"]),
        ("affirmation", ["you're okay","you got this","you can","it's fine","proud of you"]),
        ("imagination", ["imagine","what if","picture","visualize","story"]),
        ("humor",       ["lol","lmao","joke","funny","chaos"]),
    ]
    for label, keys in rules:
        if any(k in t for k in keys):
            return label
    return "unknown"

POSITIVE = set("good great love safe calm ease relief hope gentle connected aligned proud progress win helpful better steady clear grounded".split())
NEGATIVE = set("bad fear angry panic stuck freeze pain overwhelmed chaos heavy shame fail useless lost worse confused".split())
WORD_RE = re.compile(r"[a-z0-9']+")

STOPWORDS = set((
    "the a an and or of to in for with without on at from by into about as is are was were be been being "
    "it this that these those i you he she we they them me my your our their can could should would may might must do does did doing "
    "not no yes just really very so if then than when where how what why which who whom because but however though over under again more most "
    "up down out make makes made get got let lets will wont dont can't couldnt shouldn't wouldnt it's that's there's here's".split()
))

def tokenize(text:str):
    return [w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and not w.isdigit() and len(w)>2]

def sentiment(text:str):
    words = tokenize(text)
    pos = sum(1 for w in words if w in POSITIVE)
    neg = sum(1 for w in words if w in NEGATIVE)
    if pos == neg == 0: return ("neutral", 0.0)
    if pos > neg: return ("positive", (pos-neg)/(pos+neg))
    if neg > pos: return ("negative", (neg-pos)/(pos+neg))
    return ("mixed", 0.5)

# ===========================
# CLUSTERING (keyword)
# ===========================
CLUSTER_RULES = {
    "adhd_regulation": ["adhd","time blindness","executive","focus","dopamine","task","routine","switching"],
    "emotional_regulation": ["panic","freeze","overwhelm","ground","calm","breathe","regulate","soothe"],
    "relationship_clarity": ["boundaries","communication","apologize","feelings","relationship","expectations","misunderstood"],
    "self_validation": ["you're okay","you got this","worthy","shame","affirm","self-talk"],
    "productivity": ["hack","workflow","system","organize","optimize","checklist","step-by-step"],
    "humor_chaos": ["lol","lmao","meme","joke","funny","chaos","unhinged"],
    "imagination_story": ["imagine","visualize","story","narrative","scene","what if"],
}
def detect_cluster(text:str)->str:
    t = text.lower()
    scores = {k:0 for k in CLUSTER_RULES}
    for name, keys in CLUSTER_RULES.items():
        for kw in keys:
            if kw in t: scores[name]+=1
    best = max(scores, key=scores.get)
    return best if scores[best]>0 else "misc"

# ===========================
# EXTRACT
# ===========================
def extract_from_text(path:Path):
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            category, text = categorize_line(line)
            if text:
                yield {"source":str(path), "category":category, "text":text}

def walk_raw():
    for p in RAW_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".txt",".md",".json",".jsonl"):
            yield from extract_from_text(p)

# ===========================
# DEDUPE
# ===========================
def dedupe(rows):
    if DEDUP_FILE.exists():
        seen_global = set(json.loads(DEDUP_FILE.read_text()))
    else:
        seen_global = set()
    seen_run = set(); unique=[]
    for r in rows:
        key = f"{r['category']}::{r['text']}".lower()
        if key in seen_global or key in seen_run: continue
        seen_run.add(key); seen_global.add(key); unique.append(r)
    DEDUP_FILE.write_text(json.dumps(list(seen_global), indent=2))
    return unique, len(seen_global)

# ===========================
# TF-IDF + SIMILARITY GROUPS
# ===========================
def build_tfidf(items, top_k=5):
    docs=[]; df=Counter()
    for r in items:
        toks = tokenize(r["text"]); r["tokens"]=toks; docs.append(toks)
        for t in set(toks): df[t]+=1
    N=max(1,len(items))
    idf={t: math.log((N+1)/(df[t]+1))+1.0 for t in df}
    for r,toks in zip(items, docs):
        tf=Counter(toks)
        vec={t:(tf[t]/max(1,len(toks)))*idf.get(t,0.0) for t in tf}
        r["tfidf"]=vec
        r["keywords"]=[t for t,_ in sorted(vec.items(), key=lambda kv:kv[1], reverse=True)[:top_k]]
    return items, idf

def cosine(a:dict,b:dict)->float:
    if not a or not b: return 0.0
    if len(a)<len(b): small,large=a,b
    else: small,large=b,a
    dot=sum(small[k]*large.get(k,0.0) for k in small)
    na=math.sqrt(sum(v*v for v in a.values()))
    nb=math.sqrt(sum(v*v for v in b.values()))
    if na==0.0 or nb==0.0: return 0.0
    return dot/(na*nb)

def build_similarity_groups(items, threshold=0.42):
    for i,r in enumerate(items): r["_idx"]=i
    items,_idf = build_tfidf(items, top_k=5)
    nbrs=defaultdict(set)
    for i in range(len(items)):
        vi=items[i]["tfidf"]
        for j in range(i+1,len(items)):
            vj=items[j]["tfidf"]
            if cosine(vi,vj)>=threshold:
                nbrs[i].add(j); nbrs[j].add(i)
    visited=set(); groups=[]
    for i in range(len(items)):
        if i in visited: continue
        stack=[i]; comp=[]; visited.add(i)
        while stack:
            u=stack.pop(); comp.append(u)
            for v in nbrs[u]:
                if v not in visited:
                    visited.add(v); stack.append(v)
        groups.append(sorted(comp))
    groups_json={}
    for gid,comp in enumerate(groups, start=1):
        label=f"group_{gid:03d}"
        groups_json[label]=[]
        for idx in comp:
            items[idx]["group_id"]=label
            groups_json[label].append({
                "text": items[idx]["text"],
                "category": items[idx]["category"],
                "emotion": items[idx].get("emotion","unknown"),
                "cluster": items[idx].get("cluster","misc"),
                "keywords": items[idx].get("keywords",[]),
                "source": items[idx]["source"],
            })
    GROUPS_JSON.write_text(json.dumps(groups_json, indent=2))
    return items

# ===========================
# TITLES
# ===========================
def make_title(r):
    k=r.get("keywords",[])
    base=" / ".join(k[:3]) if k else r["text"][:40]
    parts=[]
    if r.get("cluster") and r["cluster"]!="misc":
        parts.append(r["cluster"].replace("_"," ").title())
    if r.get("emotion") and r["emotion"]!="unknown":
        parts.append(r["emotion"].title())
    if base: parts.append(base.title())
    return " — ".join(parts)[:120]

# ===========================
# QUICKCAST (lists)
# ===========================
def pick(items, where=None, sort_key=None, limit=7):
    pool=[r for r in items if (where(r) if where else True)]
    if sort_key: pool.sort(key=sort_key, reverse=True)
    return pool[:limit]

def render_md_list(title, rows):
    out=[f"## {title}"]
    if not rows: out+=["_none_",""]; return out
    for r in rows:
        k=" | ".join(r.get("keywords",[])[:3])
        emo=r.get("emotion","unknown")
        out.append(f"- **{r['text']}**  \n  _{emo}_  {('· ' + k) if k else ''}")
    out.append(""); return out

def write_quickcast(items):
    hooks= pick(items, where=lambda r:r["category"]=="hook",  sort_key=lambda r:r["sentiment"][1], limit=8)
    ctas = pick(items, where=lambda r:r["category"]=="cta",   sort_key=lambda r:r["sentiment"][1], limit=8)
    scripts=pick(items, where=lambda r:r["category"]=="script",sort_key=lambda r:len(r["tokens"]), limit=10)
    grounding = pick(items, where=lambda r:r["emotion"]=="grounding", sort_key=lambda r:r["sentiment"][1], limit=6)
    regulation= pick(items, where=lambda r:r["emotion"]=="regulation", sort_key=lambda r:-abs(r["sentiment"][1]-0.4), limit=6)
    affirmation=pick(items, where=lambda r:r["emotion"]=="affirmation", sort_key=lambda r:r["sentiment"][1], limit=6)
    clarity=    pick(items, where=lambda r:r["emotion"]=="clarity",     sort_key=lambda r:len(r["tokens"]), limit=6)
    humor=      pick(items, where=lambda r:r["emotion"]=="humor",       sort_key=lambda r:r["sentiment"][1], limit=6)
    lines=["# TikTok QuickCast",""]
    lines+=render_md_list("🎯 Hooks", hooks)
    lines+=render_md_list("👉 Calls to Action", ctas)
    lines+=render_md_list("🧠 Scripts / Talking Points", scripts)
    lines+=render_md_list("🌀 Grounding Lines", grounding)
    lines+=render_md_list("🧩 Regulation Helpers", regulation)
    lines+=render_md_list("❤️ Affirmations", affirmation)
    lines+=render_md_list("🔎 Clarity / Explanations", clarity)
    lines+=render_md_list("😈 Humor / Chaos", humor)
    QUICKCAST_MD.write_text("\n".join(lines), encoding="utf-8"); return QUICKCAST_MD

def write_quickcast_by_cluster(items):
    clusters=defaultdict(list)
    for r in items: clusters[r["cluster"]].append(r)
    title_map={
        "adhd_regulation":"ADHD / Regulation",
        "emotional_regulation":"Emotional Regulation",
        "relationship_clarity":"Relationship Clarity",
        "self_validation":"Self-Validation",
        "productivity":"Productivity",
        "humor_chaos":"Humor / Chaos",
        "imagination_story":"Imagination / Story",
        "misc":"Misc"
    }
    out=["# TikTok QuickCast — By Cluster",""]
    for key in sorted(clusters.keys()):
        pretty=title_map.get(key, key.replace("_"," ").title())
        out.append(f"## {pretty}")
        rows=pick(clusters[key], limit=10, sort_key=lambda r:r["sentiment"][1])
        for r in rows:
            k=" | ".join(r.get("keywords",[])[:3])
            out.append(f"- **{r['text']}**  \n  _{r['emotion']}_  {('· ' + k) if k else ''}")
        out.append("")
    QUICKCAST_CLUSTER_MD.write_text("\n".join(out), encoding="utf-8"); return QUICKCAST_CLUSTER_MD

# ===========================
# STORY ARC + AUTOSCRIPTS (V7)
# ===========================
def choose_best(items, where=None, sort_key=None):
    pool=[r for r in items if (where(r) if where else True)]
    if not pool: return None
    if sort_key: pool.sort(key=sort_key, reverse=True)
    return pool[0]

def beatmap_from_keywords(kws, max_beats=4):
    beats=[f"- {k.title()}: one concrete micro-action in 15–20s." for k in kws[:max_beats]]
    while len(beats)<3: beats.append("- Micro-step: one tiny action you can do right now.")
    return beats

def synth_problem(items):
    cand=choose_best(items, where=lambda r:r["emotion"]=="regulation", sort_key=lambda r:1-r["sentiment"][1])
    if cand: return cand["text"]
    cand=choose_best(items, where=lambda r:r["sentiment"][0]=="negative", sort_key=lambda r:r["sentiment"][1])
    return cand["text"] if cand else "When your brain is loud and time feels slippery."

def synth_reframe(items):
    cand=choose_best(items, where=lambda r:r["emotion"] in ("clarity","affirmation"), sort_key=lambda r:r["sentiment"][1])
    return cand["text"] if cand else "You're not broken; your system wants different switches."

def synth_cta(items):
    cand=choose_best(items, where=lambda r:r["category"]=="cta", sort_key=lambda r:r["sentiment"][1])
    return cand["text"] if cand else "Save this for hard days and share with someone who needs it."

def story_arc(title_seed:str, hook:str, problem:str, reframe:str, beats:list, cta:str, vibe:str):
    md=[]
    md.append(f"**Hook:** {hook}")
    md.append(f"**Problem:** {problem}")
    md.append(f"**Turn:** {reframe}")
    md.append("**Beat Map (0→60s):**"); md+=beats
    md.append(f"**CTA:** {cta}")
    md.append(f"_vibe: {vibe}_")
    return "\n".join(md)

def script_from_group(label, items):
    top=choose_best(items, sort_key=lambda r:r["sentiment"][1]) or items[0]
    title=top.get("title") or "Autocast"
    hook=choose_best(items, where=lambda r:r["category"]=="hook", sort_key=lambda r:r["sentiment"][1])
    if not hook:
        hook=choose_best(items, where=lambda r:r["emotion"] in ("activation","grounding","humor"), sort_key=lambda r:r["sentiment"][1])
    hook_line=hook["text"] if hook else "If time keeps slipping, try this 60-second reset."
    problem=synth_problem(items)
    reframe=synth_reframe(items)
    kwc=Counter()
    for r in items:
        for k in r.get("keywords", []): kwc[k]+=1
    beats=beatmap_from_keywords([k for k,_ in kwc.most_common(6)], max_beats=4)
    cta_line=synth_cta(items)
    cluster=Counter([r["cluster"] for r in items]).most_common(1)[0][0]
    mood=Counter([r["emotion"] for r in items]).most_common(1)[0][0]
    arc=story_arc(title, hook_line, problem, reframe, beats, cta_line, f"{cluster.replace('_',' ')} · {mood}")
    md=[f"### {label} — {title}","",arc,""]
    return "\n".join(md)

def write_autocast(items):
    groups=defaultdict(list)
    for r in items:
        gid=r.get("group_id","group_000")
        groups[gid].append(r)
    sections=["# TikTok AutoCast Scripts",""]
    for gid in sorted(groups.keys()):
        rows=sorted(groups[gid], key=lambda r:r["sentiment"][1], reverse=True)
        sections.append(script_from_group(gid, rows))
    AUTOCAST_MD.write_text("\n".join(sections), encoding="utf-8")
    return groups

# ===========================
# TIMING / SRT / WHISPER (V8/V9)
# ===========================
def seconds_to_timestamp(sec):
    h=sec//3600; m=(sec%3600)//60; s=sec%60
    return f"{h:02}:{m:02}:{s:02},000"

def estimate_line_duration(text:str, wpm=160, floor=1.2, ceil=6.0):
    # ~160 words/min ≈ 0.375s per word. Clamp for readability.
    wc=max(1, len(WORD_RE.findall(text)))
    dur=max(floor, min(ceil, wc*(60.0/max(100, wpm))))
    return dur

def generate_srt(script_text:str, wpm=160):
    lines=[ln for ln in script_text.split("\n") if ln.strip()]
    idx=1; t=0.0; entries=[]
    for line in lines:
        dur=estimate_line_duration(line, wpm=wpm)
        start=seconds_to_timestamp(int(round(t)))
        t+=dur
        end=seconds_to_timestamp(int(round(t)))
        entries.append((idx, start, end, line)); idx+=1
    return entries

def write_srt(entries, path:Path):
    with path.open("w", encoding="utf-8") as f:
        for idx, start, end, text in entries:
            f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")

def write_all_srt(group_scripts:dict, prefix=""):
    for gid, script in group_scripts.items():
        entries=generate_srt(script, wpm=165)
        out=SRT_DIR / (f"{prefix}{gid}.srt" if prefix else f"{gid}.srt")
        write_srt(entries, out)

def write_whisper_json(group_scripts:dict):
    # Whisper-friendly JSON: list of segments with start/end and text
    data={}
    for gid, script in group_scripts.items():
        segments=[]; t=0.0
        for line in [ln for ln in script.split("\n") if ln.strip()]:
            dur=estimate_line_duration(line, wpm=165)
            start=float(round(t,2)); end=float(round(t+dur,2))
            segments.append({"start":start,"end":end,"text":line})
            t+=dur
        data[gid]=segments
    WHISPER_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return WHISPER_JSON

# ===========================
# TELEPROMPTER (V8)
# ===========================
def write_teleprompter_html(groups_map:dict):
    tele=OUT_DIR/"teleprompter.html"
    html=[
        "<html><head><meta charset='utf-8'><title>Teleprompter</title>",
        "<style>body{background:#000;color:#0f0;font-size:36px;line-height:1.6;font-family:monospace;padding:40px}h2{color:#0ff;font-size:48px;border-bottom:2px solid #0ff}.section{margin-bottom:80px}</style>",
        "</head><body><h1>TikTok Teleprompter</h1>"
    ]
    for gid, script in groups_map.items():
        html.append(f"<div class='section'><h2>{gid}</h2><pre>{script}</pre></div>")
    html.append("</body></html>")
    tele.write_text("\n".join(html), encoding="utf-8")

# ===========================
# MARKOV FILL (V9)
# ===========================
def build_markov_corpus(lines, order=2):
    model=defaultdict(list)
    for line in lines:
        toks=WORD_RE.findall(line.lower())
        if len(toks)<=order: continue
        q=deque(maxlen=order)
        for w in toks:
            if len(q)==order:
                key=tuple(q); model[key].append(w)
            q.append(w)
    return model

def markov_generate(model, seed_words=None, length=12):
    if not model: return ""
    keys=list(model.keys())
    state = seed_words if seed_words in model else (keys[0] if keys else None)
    if state is None: return ""
    out=list(state)
    for _ in range(length):
        nxts=model.get(tuple(out[-len(state):]), [])
        if not nxts: break
        out.append(nxts[0])  # deterministic for paste-safety
    s=" ".join(out)
    return s.capitalize() + "."

# ===========================
# NEGATIVE SPACE INFERENCE (V9)
# ===========================
def ensure_script_parts(hook, problem, reframe, cta, corpus_lines):
    missing=[]
    if not hook:    missing.append("hook")
    if not problem: missing.append("problem")
    if not reframe: missing.append("reframe")
    if not cta:     missing.append("cta")
    if not missing: return hook, problem, reframe, cta

    model=build_markov_corpus(corpus_lines, order=2)
    if not hook:
        hook = "Try this if your brain won’t cooperate." if not model else markov_generate(model, next(iter(model)), 10)
    if not problem:
        problem = "Time slips, pressure spikes, and everything feels loud."
    if not reframe:
        reframe = "You're not failing—your switches are mismapped; let's remap two now."
    if not cta:
        cta = "Save this for your hard days and pass it to one friend."
    return hook, problem, reframe, cta

# ===========================
# 90-SECOND CUT (V9)
# ===========================
def compress_to_target_duration(script_text:str, target_sec=90, wpm=165):
    lines=[ln for ln in script_text.split("\n") if ln.strip()]
    if not lines: return script_text
    # keep lines while total duration <= target
    out=[]; t=0.0
    for ln in lines:
        d=estimate_line_duration(ln, wpm=wpm)
        if t+d > target_sec: break
        out.append(ln); t+=d
    # ensure it ends with CTA hint
    if not any("**CTA:**" in l for l in out):
        out.append("**CTA:** Save and share if this helped.")
    return "\n".join(out)

# ===========================
# OUTPUT WRITERS
# ===========================
def write_core_csvs(items, memory_size):
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["source","category","emotion","cluster","sentiment","sentiment_score","keywords","title","group_id","text"])
        w.writeheader()
        for r in items:
            w.writerow({
                "source":r["source"], "category":r["category"], "emotion":r["emotion"], "cluster":r["cluster"],
                "sentiment":r["sentiment"][0], "sentiment_score":round(r["sentiment"][1],3),
                "keywords":"|".join(r.get("keywords",[])), "title":r.get("title",""), "group_id":r.get("group_id",""), "text":r["text"]
            })
    with CAT_CSV.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["category","emotion","cluster","title","text","source"])
        w.writeheader()
        for r in sorted(items, key=lambda x:x["category"]):
            w.writerow({
                "category":r["category"], "emotion":r["emotion"], "cluster":r["cluster"],
                "title":r.get("title",""), "text":r["text"], "source":r["source"]
            })
    clusters=defaultdict(list)
    for r in items:
        clusters[r["cluster"]].append({
            "text":r["text"], "title":r.get("title",""), "emotion":r["emotion"],
            "sentiment":r["sentiment"][0], "keywords":r.get("keywords",[]), "source":r["source"]
        })
    CLUSTER_JSON.write_text(json.dumps(clusters, indent=2), encoding="utf-8")
    print(f"Added {len(items)} new prompts")
    print(f"Memory size now {memory_size} total unique prompts")
    print(f"Wrote clusters -> {CLUSTER_JSON}")
    print(f"Wrote groups   -> {GROUPS_JSON}")

# ===========================
# MAIN
# ===========================
def main():
    rows=list(walk_raw())
    rows, mem_size = dedupe(rows)

    # enrich
    for r in rows:
        r["emotion"]=detect_emotional_function(r["text"], r["category"])
        r["cluster"]=detect_cluster(r["text"])
        r["sentiment"]=sentiment(r["text"])

    # tfidf + groups
    rows,_idf = build_tfidf(rows, top_k=5)
    rows = build_similarity_groups(rows, threshold=0.42)

    # titles
    for r in rows: r["title"]=make_title(r)

    # core outputs
    write_core_csvs(rows, mem_size)
    write_quickcast(rows)
    write_quickcast_by_cluster(rows)
    groups = write_autocast(rows)

    # build map: group_id -> script (story arc text)
    group_scripts={}
    corpus_lines=[r["text"] for r in rows]
    grouped = defaultdict(list)
    for r in rows: grouped[r["group_id"]].append(r)

    # assemble scripts with negative-space fill
    for gid in sorted(grouped.keys()):
        items=sorted(grouped[gid], key=lambda r:r["sentiment"][1], reverse=True)
        # seeds
        hook = choose_best(items, where=lambda r:r["category"]=="hook", sort_key=lambda r:r["sentiment"][1])
        hook_line = hook["text"] if hook else ""
        problem = synth_problem(items)
        reframe = synth_reframe(items)
        cta = synth_cta(items)
        hook_line, problem, reframe, cta = ensure_script_parts(hook_line, problem, reframe, cta, corpus_lines)

        # beats
        kwc=Counter()
        for r in items:
            for k in r.get("keywords", []): kwc[k]+=1
        beats=beatmap_from_keywords([k for k,_ in kwc.most_common(6)], max_beats=4)

        cluster=Counter([r["cluster"] for r in items]).most_common(1)[0][0]
        mood=Counter([r["emotion"] for r in items]).most_common(1)[0][0]
        title_seed = items[0].get("title") or "Autocast"

        arc = story_arc(title_seed, hook_line, problem, reframe, beats, cta, f"{cluster.replace('_',' ')} · {mood}")
        group_scripts[gid]=arc

    # write SRT + whisper
    write_all_srt(group_scripts, prefix="")
    whisper_path = write_whisper_json(group_scripts)

    # teleprompter
    write_teleprompter_html(group_scripts)

    # 90-sec cut (global best-of script)
    all_script = "# QuickCast 90s\n\n" + "\n\n".join([f"## {gid}\n{txt}" for gid,txt in group_scripts.items()])
    quick_90 = compress_to_target_duration(all_script, target_sec=90, wpm=165)
    (OUT_DIR/"quickcast_90.md").write_text(quick_90, encoding="utf-8")
    write_srt(generate_srt(quick_90, wpm=165), SRT_DIR/"quickcast_90.srt")

    print(f"Wrote QuickCast  -> {QUICKCAST_MD}")
    print(f"Wrote QuickCastC -> {QUICKCAST_CLUSTER_MD}")
    print(f"Wrote AutoCast   -> {AUTOCAST_MD}")
    print(f"Wrote Whisper    -> {whisper_path}")
    print(f"Wrote Teleprompt -> {OUT_DIR/'teleprompter.html'}")
    print(f"Wrote 90s Cut    -> {OUT_DIR/'quickcast_90.md'} and {SRT_DIR/'quickcast_90.srt'}")
    print("✅ V9 pipeline complete.")

if __name__ == "__main__":
    main()
