"""Lab 9's worked examples for the parts this course added: A2b-A2d, A3b, A4b, A5b, B1b-B1d, C0, C1c, C2b,
C4b and C6.

Session 9 presents four decks: 9.A1-9.A3, and 9.B1, the walk-through of this lab. These functions let the lab
run the examples those decks work through, with the decks' own inputs, so that a number a slide prints is a
number a cell prints. Everything here is offline, runs on a CPU and is seeded.

Three kinds of input:
  - the illustrative inputs the slides use (three short documents, two topic bars, two-dimensional word
    vectors, nine hand-set points, the vector (2, 4, 6, 8)), copied here so that the slides' arithmetic can be redone;
  - this lab's own corpus, vocabulary and Machine B, through `session05_text` and `session05_attention`;
  - `recorded_deck12_examples.json`: one recorded, offline run of real tokenizers and pretrained models
    (cl100k_base, multilingual BERT, the XLM-R vocabulary, FinBERT) and of Machine B. Those models are not
    shipped with the lab, so their outputs are read from the record, whose sha256 is checked on loading.

Added 2026-11-18 (ECON 282E). This module may change; the four session05_* modules are kept as they came.
"""
from __future__ import annotations

import collections
import hashlib
import html
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

import session05_attention as at
import session05_text as tx

HERE = Path(__file__).resolve().parent
RECORD = HERE / "recorded_deck12_examples.json"
RECORD_SHA256 = "66c5f0571dc88ee41264818d7ffc2eb19947e74fb39b514b5d1eb3cc86891ff9"
SEED = 42


def load_record(path: Path = RECORD) -> dict:
    """The recorded run of real tokenizers and models, after its sha256 is checked."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != RECORD_SHA256:
        raise RuntimeError(f"{path.name} does not match its recorded sha256 ({digest[:16]}...)")
    return json.loads(data.decode("utf-8"))


def printed_as(value: float, printed: str) -> bool:
    """Does `value`, rounded to the decimals of `printed`, give the number a slide prints?"""
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    return abs(round(float(value), decimals) - float(printed)) < 1e-12


def as_printed(value: float, decimals: int = 3) -> str:
    """A number as the slides print it: trailing zeros dropped, so 0.250 is 0.25 and 0.000 is 0."""
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


# ======================================================================================================
# A2b -- one sentence, cleaned a step at a time (9.A1 "Cleaning, Worked on One Sentence", "Stop Words",
# "Stems, Lemmas, and a Language Without Inflection")
# ======================================================================================================
# The frame's three raw strings. The first is the course's running sentence, scraped with its mark-up.
RAW_STRINGS: tuple[tuple[str, str], ...] = (
    ("English, scraped from a web page", "<p>The central bank&nbsp;raised interest rates amid rising inflation.</p>"),
    ("Chinese, traditional characters", "<p>央行上調利率以遏制通脹。</p>"),
    ("Chinese, full-width", "ＣＰＩ同比上漲２.３％"),
)
# Traditional -> simplified for the three characters the frame converts. A real converter (OpenCC, say) carries
# thousands of entries and some one-to-many choices; three are enough to see the step.
TRAD_TO_SIMP: dict[str, str] = {"調": "调", "脹": "胀", "漲": "涨"}
# The words 9.A1's "Stop Words" frame says to keep, because they carry the direction or the sign of a sentence.
SIGNAL_WORDS: tuple[str, ...] = ("not", "up", "down", "more", "less")
STEM_PROBES: tuple[str, ...] = ("investing", "invests", "invested", "studies", "inflation", "policies", "housing")
# 9.A1 "Stems, Lemmas": a lemmatizer looks a word up, by its part of speech, and returns the dictionary form. The
# frame's two entries, then four of the stem probes read as the slide reads them. A real lemmatizer (spaCy's, or
# WordNet's in NLTK) carries tens of thousands of entries; this one shows the step, not the coverage.
LEMMAS: dict[str, str] = {"better": "good", "were": "be", "invested": "invest", "studies": "study",
                          "policies": "policy", "inflation": "inflation", "housing": "housing"}
# The frame's Chinese pair: the verb keeps its form, the tense is a separate word.
ZH_TENSE: tuple[tuple[str, str], ...] = (("央行已上调利率", "the central bank has raised rates"),
                                         ("央行将上调利率", "the central bank will raise rates"))


def strip_markup(text: str) -> str:
    """Drop HTML tags, decode entities (&nbsp; becomes U+00A0, then a plain space), squeeze the spaces."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", text)).replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def to_half_width(text: str) -> str:
    """Unicode NFKC: full-width Latin letters, digits and signs become their ordinary forms."""
    return unicodedata.normalize("NFKC", text)


def to_simplified(text: str, table: Mapping[str, str] = TRAD_TO_SIMP) -> str:
    return "".join(table.get(ch, ch) for ch in text)


def clean_raw(text: str) -> list[tuple[str, str]]:
    """The frame's three steps, in order, with the string after each."""
    steps = [("raw", text)]
    for label, step in (("strip mark-up and entities", strip_markup),
                        ("full-width to half-width (NFKC)", to_half_width),
                        ("traditional to simplified", to_simplified)):
        text = step(text)
        steps.append((label, text))
    return steps


def _removed(before: Sequence[str], after: Sequence[str]) -> list[str]:
    left = collections.Counter(after)
    out = []
    for token in before:
        if left[token]:
            left[token] -= 1
        else:
            out.append(token)
    return out


def seven_steps(text: str) -> list[tuple[str, object, str]]:
    """The lab's switches (block A2), applied one at a time to one sentence, with what each step did."""
    rows: list[tuple[str, object, str]] = []
    t = tx.to_lower(text); rows.append(("1 lower-case", t, ""))
    t = tx.drop_punctuation(t); rows.append(("2 drop punctuation", t, ""))
    t = tx.drop_numbers(t); rows.append(("3 drop numbers", t, ""))
    tokens = tx.split_tokens(t); rows.append(("4 tokenize", tokens, f"{len(tokens)} tokens"))
    kept = tx.drop_stopwords(tokens)
    rows.append(("5 drop stop-words", kept, "removed: " + ", ".join(_removed(tokens, kept))))
    stems = tx.crude_stem(kept)
    rows.append(("6 stem", stems, "changed: " + ", ".join(f"{a} -> {b}" for a, b in zip(kept, stems) if a != b)))
    short = tx.drop_short(stems)
    rows.append(("7 drop 1-character tokens", short, "removed: " + (", ".join(_removed(stems, short)) or "nothing")))
    return rows


# ======================================================================================================
# A2c -- where Chinese words begin (9.A1 "Tokens and the Vocabulary", "Chinese Segmentation Is a Labelling
# Problem", "Domain Compounds Need a User Dictionary")
# ======================================================================================================
# Maximum matching is the dictionary half of a segmenter such as Jieba, THULAC or LTP, without the statistical
# labeller those tools add for words the dictionary lacks. Forward matching takes the longest dictionary word
# from the left, backward matching from the right; a character no entry covers stands alone.
ZH_DICTIONARY = frozenset({
    "研究", "研究生", "生命", "科学", "生命科学",              # 研究生命科学: study / life science
    "行业", "政策", "支持", "行业政策", "政策支持",            # 行业政策支持: two sensible cuts
    "货币", "委员会", "政策委员会",                            # 货币政策委员会 without the user entry
    "央行", "上调", "下调", "利率", "遏制", "通胀", "决定", "存款准备金率",
    "社会", "融资", "规模", "增速", "放缓", "宏观", "审慎", "监管", "框架", "加强",
})
USER_DICTIONARY = frozenset({"货币政策委员会", "社会融资规模", "宏观审慎监管框架"})   # the frame's entries
ZH_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("央行上调利率以遏制通胀", "the central bank raises rates to curb inflation"),
    ("央行决定下调存款准备金率", "the central bank decides to cut the reserve requirement ratio"),
    ("研究生命科学", "study life science"),
    ("行业政策支持", "industry policy support"),
)
ZH_USER_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("货币政策委员会", "the monetary policy committee"),
    ("社会融资规模增速放缓", "growth of aggregate financing slowed"),
    ("加强宏观审慎监管框架", "strengthen the macro-prudential supervision framework"),
)


def max_match(text: str, dictionary: Iterable[str], backward: bool = False) -> list[str]:
    """Forward (or backward) maximum matching against one dictionary."""
    words = set(dictionary)
    longest = max(len(w) for w in words)
    out: list[str] = []
    if not backward:
        i = 0
        while i < len(text):
            for n in range(min(longest, len(text) - i), 0, -1):
                if n == 1 or text[i:i + n] in words:
                    out.append(text[i:i + n])
                    i += n
                    break
        return out
    j = len(text)
    while j > 0:
        for n in range(min(longest, j), 0, -1):
            if n == 1 or text[j - n:j] in words:
                out.append(text[j - n:j])
                j -= n
                break
    return out[::-1]


def bmes(words: Sequence[str]) -> list[tuple[str, str]]:
    """One label per character: B begins a word, M is inside it, E ends it, S is a one-character word."""
    out = []
    for w in words:
        tags = ["S"] if len(w) == 1 else ["B"] + ["M"] * (len(w) - 2) + ["E"]
        out.extend(zip(w, tags))
    return out


# ======================================================================================================
# A2d -- pairs as features (9.A1 "n-Grams: One Boundary Can Flip the Sign")
# ======================================================================================================
def ngram_counts(statements: Sequence, min_df: int = 2, pipeline=tx.BASELINE) -> dict:
    """Word and adjacent-pair columns on one set of statements: seen at all, and in at least `min_df` of them."""
    uni, bi = collections.Counter(), collections.Counter()
    uni_df, bi_df = collections.Counter(), collections.Counter()
    for s in statements:
        tokens = pipeline.run(s.text)
        pairs = list(zip(tokens, tokens[1:]))
        uni.update(tokens); bi.update(pairs)
        uni_df.update(set(tokens)); bi_df.update(set(pairs))
    V = len(uni)
    negations = sorted(((f"{a} {b}", n) for (a, b), n in bi.items() if a in ("not", "no", "less")),
                       key=lambda kv: (-kv[1], kv[0]))
    return dict(n_statements=len(statements), V=V, V_squared=V * V, bigrams_seen=len(bi),
                unigrams_min_df=sum(1 for w in uni_df if uni_df[w] >= min_df),
                bigrams_min_df=sum(1 for p in bi_df if bi_df[p] >= min_df), min_df=min_df,
                negation_bigrams=negations, unigram_counts=uni, bigram_counts=bi)


# ======================================================================================================
# A3b -- three ways to learn the pieces (9.A1 "Why Subwords", "Three Ways to Learn the Pieces", "Three Ways,
# One Phrase"). Simplified versions, trained on the window's word counts, and called simplified in the lab.
# ======================================================================================================
SCHEME_PHRASE = "disinflationary pressures eased"
SLIDE_PIECE_COUNTS = {"bpe": 6, "wordpiece": 9, "sentencepiece": 8}     # 9.A1 "Three Ways, One Phrase"
SUBWORD_PROBES: tuple[str, ...] = ("macroeconomics", "专精特新")             # 9.A1 "Why Subwords"
MARK = "▁"                                                          # the unigram word-start mark
SPACE = "␣"                                                         # a space shown inside a piece


def alphabetic(word_counts: Mapping[str, int]) -> dict[str, int]:
    return {w: int(c) for w, c in word_counts.items() if w.isalpha()}


def learn_wordpiece(word_counts: Mapping[str, int], n_merges: int = 200) -> list[tuple[str, str, float]]:
    """WordPiece's learning rule: glue the pair whose count most exceeds what its parts' counts predict.

    Score = count(ab) / (count(a) * count(b)). A piece that continues a word carries '##'. Ties go to the more
    frequent pair, then to the alphabetically first, so the run is deterministic.
    """
    words = {w: [w[0]] + ["##" + ch for ch in w[1:]] for w in word_counts}
    merges: list[tuple[str, str, float]] = []
    for _ in range(n_merges):
        unit, pair = collections.Counter(), collections.Counter()
        for w, pieces in words.items():
            c = word_counts[w]
            for p in pieces:
                unit[p] += c
            for ab in zip(pieces, pieces[1:]):
                pair[ab] += c
        if not pair:
            break
        best = min(pair, key=lambda ab: (-pair[ab] / (unit[ab[0]] * unit[ab[1]]), -pair[ab], ab))
        a, b = best
        glued = a + b[2:]
        merges.append((a, b, pair[best] / (unit[a] * unit[b])))
        for w, pieces in words.items():
            out, i = [], 0
            while i < len(pieces):
                if i + 1 < len(pieces) and pieces[i] == a and pieces[i + 1] == b:
                    out.append(glued); i += 2
                else:
                    out.append(pieces[i]); i += 1
            words[w] = out
    return merges


def wordpiece_vocabulary(word_counts: Mapping[str, int], merges: Sequence[tuple[str, str, float]]) -> set[str]:
    vocab = {w[0] for w in word_counts} | {"##" + ch for w in word_counts for ch in w[1:]}
    vocab |= {a + b[2:] for a, b, _ in merges}
    return vocab


def wordpiece_cut(word: str, vocab: set[str]) -> list[str]:
    """WordPiece's cutting rule: the longest known piece from the left, then the longest that continues it."""
    pieces, start = [], 0
    while start < len(word):
        for end in range(len(word), start, -1):
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                pieces.append(piece)
                start = end
                break
        else:
            return ["[UNK]"]
    return pieces


def _viterbi(word: str, logp: Mapping[str, float], max_len: int, banned: str | None = None
             ) -> tuple[list[str], float]:
    """The most probable split of `word` into known pieces, and its log-probability."""
    n = len(word)
    best = [0.0] + [-math.inf] * n
    back = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(max(0, j - max_len), j):
            piece = word[i:j]
            if piece == banned or piece not in logp or best[i] == -math.inf:
                continue
            score = best[i] + logp[piece]
            if score > best[j]:
                best[j], back[j] = score, i
    pieces, j = [], n
    while j > 0:
        pieces.append(word[back[j]:j])
        j = back[j]
    return pieces[::-1], best[n]


def learn_unigram(word_counts: Mapping[str, int], target: int = 300, max_len: int = 10,
                  seed_size: int = 4000, keep: float = 0.8) -> dict:
    """The unigram rule, simplified: start with many candidate pieces, then keep dropping the ones the corpus
    would miss least.

    Every word gets the start mark. The candidates are the `seed_size` commonest substrings of up to `max_len`
    characters, plus every single character (never dropped, so every word can still be cut). Each round splits
    every word the most probable way, re-estimates each piece's probability from those splits, and, for each
    longer piece, measures how much the corpus log-likelihood falls if the words using it are re-split without
    it; the cheapest fifth go. SentencePiece does the same with expected counts over all splits (EM); this
    version counts only the best split.
    """
    marked = {MARK + w: c for w, c in word_counts.items()}
    sub = collections.Counter()
    for w, c in marked.items():
        for i in range(len(w)):
            for j in range(i + 1, min(len(w), i + max_len) + 1):
                sub[w[i:j]] += c
    chars = {ch for w in marked for ch in w}
    vocab = {p for p, _ in sub.most_common(seed_size)} | chars
    total = sum(sub[p] for p in vocab)
    logp = {p: math.log(sub[p] / total) for p in vocab}
    rounds = []
    while True:
        use = collections.Counter()
        for w, c in marked.items():
            for p in _viterbi(w, logp, max_len)[0]:
                use[p] += c
        vocab = {p for p in vocab if use[p] > 0} | chars
        total = sum(use[p] for p in vocab) + 0.5 * sum(1 for p in chars if use[p] == 0)
        logp = {p: math.log((use[p] or 0.5) / total) for p in vocab}
        best = {w: _viterbi(w, logp, max_len) for w in marked}
        rounds.append(dict(pieces=len(vocab), loglik=round(sum(c * best[w][1] for w, c in marked.items()), 1)))
        if len(vocab) <= target:
            break
        users = collections.defaultdict(list)
        for w, (pieces, _) in best.items():
            for p in set(pieces) - chars:
                users[p].append(w)
        loss = {p: sum(marked[w] * (best[w][1] - _viterbi(w, logp, max_len, banned=p)[1]) for w in users.get(p, ()))
                for p in vocab - chars}
        n_drop = max(1, min(len(vocab) - target, int((1 - keep) * len(loss))))
        for p in sorted(loss, key=lambda q: (loss[q], q))[:n_drop]:
            vocab.discard(p)
        norm = math.log(sum(math.exp(logp[p]) for p in vocab))
        logp = {p: logp[p] - norm for p in vocab}
    return dict(logp=logp, max_len=max_len, rounds=rounds, vocabulary=len(logp))


def unigram_cut(word: str, model: Mapping) -> list[str]:
    return _viterbi(MARK + word, model["logp"], model["max_len"])[0]


def spells_back(pieces: Sequence[str], text: str, scheme: str) -> bool:
    """Do the pieces, markers removed, give the text back exactly?"""
    if scheme == "bpe":                       # cl100k_base: a leading space is part of the piece
        return "".join(pieces) == text
    if scheme == "lab-bpe":                   # block A3's merges: '</w>' closes each word
        return "".join(pieces).replace(tx.END, " ").strip() == text
    if scheme == "wordpiece":                 # '##' continues the word before it
        words: list[str] = []
        for p in pieces:
            if p.startswith("##") and words:
                words[-1] += p[2:]
            else:
                words.append(p)
        return " ".join(words) == text
    if scheme == "sentencepiece":             # the mark opens a word
        return "".join(pieces).replace(MARK, " ").strip() == text
    raise ValueError(scheme)


# ======================================================================================================
# A4b -- the slides' three documents (9.A1 "IDs and One-Hot Vectors" to "Three Weightings")
# ======================================================================================================
TOY_DOCS: tuple[str, ...] = tuple(tx.TOY_LDA_DOCS)        # d1, d2, d3: the same three documents as A5b
TOY_VOCAB: tuple[str, ...] = ("fed", "cut", "hike", "support", "curb", "growth", "inflation", "respond")
D2_RAISE = "fed raise curb inflation"                     # d2 with "raise" for "hike"
SLIDE_A4B = {                                             # what 9.A1 prints, as it prints it
    "d3_tfidf": ("0", "0", "0.405", "0", "0", "0", "0.811", "1.099"),
    "idf": {"fed": "0", "hike": "0.405", "inflation": "0.405", "respond": "1.099"},
    "cos_counts": {"d1, d2": "0.25", "d2, d3": "0.756", "d2', d3": "0.567"},
    "weightings_d2_d3": {"counts": "0.756", "presence": "0.75", "tfidf": "0.28"},
    "weightings_d1_d2": {"counts": "0.25", "presence": "0.25", "tfidf": "0"},
}
ORDER_PAIRS: tuple[tuple[str, str], ...] = (("GDP growth exceeded inflation", "inflation exceeded GDP growth"),
                                            ("the rate cut supports growth", "growth supports the rate cut"))


def count_matrix(docs: Sequence[str], vocab: Sequence[str]) -> np.ndarray:
    return np.array([[d.split().count(w) for w in vocab] for d in docs], dtype=float)


def cos(u, v) -> float:
    return tx.cosine(np.asarray(u, float), np.asarray(v, float))


def three_documents() -> dict:
    """Counts, presence and TF-IDF (the count times the natural log of D / df), and the cosines 9.A1 prints."""
    Z = count_matrix(TOY_DOCS, TOY_VOCAB)
    B = (Z > 0).astype(float)
    df = B.sum(axis=0)
    idf = np.log(len(TOY_DOCS) / df)
    T = Z * idf
    pairs = {"d1, d2": (0, 1), "d2, d3": (1, 2), "d1, d3": (0, 2)}
    weightings = {name: {k: cos(M[i], M[j]) for k, (i, j) in pairs.items()}
                  for name, M in (("counts", Z), ("presence", B), ("tfidf", T))}
    vocab9 = TOY_VOCAB + ("raise",)
    Z9 = count_matrix(TOY_DOCS + (D2_RAISE,), vocab9)
    one_hot = np.eye(len(TOY_VOCAB))
    return dict(Z=Z, presence=B, df=df, idf=idf, tfidf=T, weightings=weightings, vocab9=vocab9, Z9=Z9,
                cos_d2raise_d3=cos(Z9[3], Z9[2]), one_hot_fed_hike=float(one_hot[0] @ one_hot[2]))


def same_bag(a: str, b: str) -> bool:
    return collections.Counter(a.lower().split()) == collections.Counter(b.lower().split())


# ======================================================================================================
# A5b -- LDA by hand (9.A1 "The Assumption: How a Document Is Written", "How LDA Finds the Topics")
# ======================================================================================================
# The two topic bars the frame draws, over TOY_VOCAB, and d3's topic shares.
SLIDE_PHI = np.array([[0.25, 0.25, 0.02, 0.20, 0.02, 0.20, 0.04, 0.02],     # topic 1, "easing"
                      [0.25, 0.02, 0.20, 0.01, 0.15, 0.02, 0.25, 0.10]])    # topic 2, "tightening"
SLIDE_THETA_D3 = np.array([0.1, 0.9])
# d3 after step 1, as the frame colours it, in document order (fed hike respond inflation inflation). The
# frame lists the same five words as hike(2) inflation(1) respond(1) inflation(2) fed(1); LDA ignores order.
SLIDE_D3_TAGS = (1, 2, 1, 1, 2)


def lda_story(n_words: int = 5, phi: np.ndarray = SLIDE_PHI, theta: np.ndarray = SLIDE_THETA_D3,
              seed: int = SEED) -> list[tuple[int, str]]:
    """Write one document the way LDA assumes documents are written: per slot, a topic, then a word."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_words):
        z = int(rng.choice(len(theta), p=theta))
        out.append((z + 1, TOY_VOCAB[int(rng.choice(phi.shape[1], p=phi[z]))]))
    return out


def random_colours(seed: int = SEED) -> list[list[int]]:
    """Step 1: a random colour for every word of d1 and d2 (seeded); d3 coloured as the frame colours it."""
    rng = np.random.default_rng(seed)
    return ([[int(k) for k in rng.integers(1, 3, size=len(d.split()))] for d in TOY_DOCS[:2]]
            + [list(SLIDE_D3_TAGS)])


def _counts(words, tags, K: int, V: int):
    n_kv = np.zeros((K, V)); n_ik = np.zeros((len(words), K))
    for i, (d, t) in enumerate(zip(words, tags)):
        for v, k in zip(d, t):
            n_kv[k, v] += 1; n_ik[i, k] += 1
    return n_kv, n_ik


def redraw_odds(tags: Sequence[Sequence[int]], doc: int, pos: int, a: float = 0.1, b: float = 0.01) -> dict:
    """Steps 2-3 of the frame for one word: forget its colour, then the odds of each colour.

    odds(k) = (n_ik + a) x (n_kv + b) / (n_k + V b), every count taken without the word itself: how much its
    statement uses colour k, times how much colour k uses this word elsewhere. Colours count from 1.
    """
    words = [[TOY_VOCAB.index(w) for w in d.split()] for d in TOY_DOCS]
    K, V = 2, len(TOY_VOCAB)
    n_kv, n_ik = _counts(words, [[k - 1 for k in t] for t in tags], K, V)
    v, old = words[doc][pos], tags[doc][pos] - 1
    n_kv[old, v] -= 1; n_ik[doc, old] -= 1
    statement = n_ik[doc] + a
    topic = (n_kv[:, v] + b) / (n_kv.sum(axis=1) + V * b)
    odds = statement * topic
    return dict(word=TOY_VOCAB[v], old=old + 1, statement_counts=n_ik[doc].copy(), word_counts=n_kv[:, v].copy(),
                topic_sizes=n_kv.sum(axis=1), statement=statement, topic=topic, odds=odds, p=odds / odds.sum())


def toy_gibbs(start: Sequence[Sequence[int]], sweeps: int = 200, a: float = 0.1, b: float = 0.01,
              seed: int = SEED, watch: int = 2) -> dict:
    """Steps 2-4, swept over the three documents from a given colouring; records document `watch`'s colours."""
    rng = np.random.default_rng(seed)
    words = [[TOY_VOCAB.index(w) for w in d.split()] for d in TOY_DOCS]
    K, V = 2, len(TOY_VOCAB)
    tags = [[k - 1 for k in t] for t in start]
    n_kv, n_ik = _counts(words, tags, K, V)
    n_k = n_kv.sum(axis=1)
    trace = []
    for _ in range(sweeps):
        for i, d in enumerate(words):
            for ell, v in enumerate(d):
                k = tags[i][ell]
                n_kv[k, v] -= 1; n_k[k] -= 1; n_ik[i, k] -= 1
                p = (n_kv[:, v] + b) / (n_k + V * b) * (n_ik[i] + a)
                k = int(rng.choice(K, p=p / p.sum()))
                tags[i][ell] = k
                n_kv[k, v] += 1; n_k[k] += 1; n_ik[i, k] += 1
        trace.append(tuple(k + 1 for k in tags[watch]))
    phi = (n_kv + b) / (n_k[:, None] + V * b)
    theta = (n_ik + a) / (n_ik.sum(axis=1, keepdims=True) + K * a)
    return dict(trace=trace, phi=phi, theta=theta)


# ======================================================================================================
# B1b -- what word2vec does instead (9.A2 "Word2Vec, Step 1" to "One Step")
# ======================================================================================================
SGNS_U, SGNS_POS, SGNS_NEG, SGNS_ETA = (1.0, 1.0), (1.5, 0.5), (-0.5, 2.0), 0.25   # the frame's vectors
SLIDE_SGNS_AFTER = {"u_raised": ("1.147", "0.606"), "u_rates": ("1.530", "0.530"), "u_noise": ("-0.704", "1.796")}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def sgns_step(u=SGNS_U, pos=SGNS_POS, neg=SGNS_NEG, eta: float = SGNS_ETA) -> dict:
    """One gradient step on one real pair and one noise pair, for the objective log s(x+) + log s(-x-)."""
    u, cp, cn = (np.array(v, float) for v in (u, pos, neg))
    xp, xn = float(u @ cp), float(u @ cn)
    grad_u = (1 - sigmoid(xp)) * cp - sigmoid(xn) * cn
    u2 = u + eta * grad_u
    cp2 = cp + eta * (1 - sigmoid(xp)) * u
    cn2 = cn - eta * sigmoid(xn) * u
    xp2, xn2 = float(u2 @ cp2), float(u2 @ cn2)

    def objective(a: float, b: float) -> float:
        return math.log(sigmoid(a)) + math.log(sigmoid(-b))

    return dict(x_plus=xp, x_minus=xn, s_plus=sigmoid(xp), s_minus=sigmoid(xn), grad_u=grad_u,
                before=dict(u_raised=u, u_rates=cp, u_noise=cn),
                u_raised=u2, u_rates=cp2, u_noise=cn2, x_plus_after=xp2, x_minus_after=xn2,
                objective_before=objective(xp, xn), objective_after=objective(xp2, xn2))


# ======================================================================================================
# B1c -- window and dimension (9.A2 "Window, Dimension, Domain")
# ======================================================================================================
WINDOW_WORDS: tuple[str, ...] = ("balance", "discount")
DIMENSION_WORDS: tuple[str, ...] = ("inflation", "employment")


def overlap(a: Sequence[tuple[str, float]], b: Sequence[tuple[str, float]]) -> int:
    return len({w for w, _ in a} & {w for w, _ in b})


# ======================================================================================================
# B1d -- directions (9.A2 "The Geometry" and "Analogies, Worked")
# ======================================================================================================
# The slides' nine words, placed by hand on three named axes so that the arithmetic can be read by eye:
# illustrative, not fitted. Real vectors have 100-300 unnamed axes; B1d asks B1's the same question.
HAND_AXES: tuple[str, ...] = ("stance", "cycle", "topic")
HAND_SET: dict[str, tuple[float, float, float]] = {
    "hike": (5.5, 3.1, 3.9), "tighten": (6.2, 2.7, 3.2), "hawkish": (5.9, 3.7, 3.5),
    "cut": (0.5, 2.0, 2.7), "ease": (1.1, 1.6, 2.1), "dovish": (0.9, 2.5, 2.4),
    "recession": (3.1, 0.9, 0.9), "unemployment": (3.8, 0.6, 1.2), "layoffs": (3.6, 1.5, 0.6)}
CLUSTERS: dict[str, tuple[str, ...]] = {"tightening": ("hike", "tighten", "hawkish"),
                                        "easing": ("cut", "ease", "dovish"),
                                        "the cycle": ("recession", "unemployment", "layoffs")}
SLIDE_GEOMETRY = {"hike_cut": "(5.0, 1.1, 1.2)", "hawk_dove": "(5.0, 1.2, 1.1)", "cosine": "0.9996",
                  "query": "(0.9, 2.6, 2.3)", "distance": "0.14"}            # what 9.A2 prints, as it prints it
# A tightening word and an easing word, twice, each with a vector among B1's 624 words ('tightening', 'cut'
# and 'hike' fall below min_count there).
STANCE_PAIRS: tuple[tuple[str, str], ...] = (("raise", "lower"), ("firming", "easing"))


# ======================================================================================================
# C0 -- the recorded encoder (9.A2 "One Word, Two Sentences")
# ======================================================================================================
def contextual_rows(record: dict) -> list[dict]:
    rows = []
    for block in ("contextual", "contextual_en"):
        for p in record[block]["probes"]:
            rows.append(dict(target=p["target"], pieces=p["target_pieces"], contrast=p["contrast"],
                             anchor=p["anchor"]["text"], same=p["same"]["text"], other=p["other"]["text"],
                             cos_same=p["cos_same_last"], cos_other=p["cos_other_last"]))
    return rows


# ======================================================================================================
# C1c -- heads, order and the triangle (9.A2 "Many Heads", "Attention Cannot See Word Order"; 9.A3 "How BERT
# Differs From the Transformer")
# ======================================================================================================
def sinusoidal_positions(n: int, d: int) -> np.ndarray:
    """p[i, 2l] = sin(i / 10000^(2l/d)), p[i, 2l+1] = cos(i / 10000^(2l/d)), positions i = 1..n."""
    P = np.zeros((n, d))
    i = np.arange(1, n + 1)[:, None]
    rate = 10000.0 ** (np.arange(0, d, 2) / d)
    P[:, 0::2] = np.sin(i / rate)
    P[:, 1::2] = np.cos(i / rate)
    return P


def heads_order_triangle(X: np.ndarray, n_heads: int = 2, seed: int = SEED) -> dict:
    """Two heads on X; then X read backwards, with and without positions, unmasked (square) and masked (triangle)."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    W_Q, W_K, W_V, W_O = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4))
    order = np.arange(n)[::-1]                       # the tokens read backwards

    def run(M, causal):
        return at.multi_head_attention(M, W_Q, W_K, W_V, W_O, n_heads=n_heads, causal=causal)

    def gap(a, b):
        return float(np.abs(a - b).max())

    P = sinusoidal_positions(n, d)
    square, triangle = run(X, causal=False), run(X, causal=True)
    backwards = {
        "no positions, square": gap(run(X[order], False)["output"], square["output"][order]),
        "positions added, square": gap(run(X[order] + P, False)["output"], run(X + P, False)["output"][order]),
        "no positions, triangle": gap(run(X[order], True)["output"], triangle["output"][order]),
    }
    return dict(square=square, triangle=triangle, order=order, backwards=backwards, P=P, head_dim=d // n_heads,
                parameters=sum(W.size for W in (W_Q, W_K, W_V, W_O)),
                above_diagonal=max(float(np.triu(w, 1).max()) for w in triangle["weights"]))


# ======================================================================================================
# C2b -- inside one layer (9.A2 "Residual Connection", "Layer Normalisation", "The Feed-Forward Block")
# ======================================================================================================
LN_X = (2.0, 4.0, 6.0, 8.0)
SLIDE_LN = {"mu": "5", "sigma": "2.24", "out": ("-1.34", "-0.45", "0.45", "1.34")}
SLIDE_FFN_SHARE = "62.6"     # 9.A2 "Where the Parameters Are" and 9.B1's block C: the feed-forward share, in percent


def layer_norm(x: Sequence[float]) -> dict:
    """(x - mu) / sigma over the entries of one token's vector; sigma is the population standard deviation."""
    x = np.asarray(x, float)
    mu, sigma = float(x.mean()), float(x.std())
    return dict(mu=mu, sigma=sigma, out=(x - mu) / sigma)


def layer_by_hand(machine, text: str, layer: int = 0, probe: int = 10) -> dict:
    """Machine B's layer, rebuilt from its parts, and the rows a change to one character's vector reaches.

    Machine B normalises before each block (pre-LN): h = e + Attn(LN1(e)), out = h + FFN(LN2(h)).
    """
    import torch
    m = machine.model
    block = m.blocks[layer]
    ids = machine.encode(text)[None, :]
    with torch.no_grad():
        e = m.token_embedding(ids) + m.position_embedding(torch.arange(ids.shape[1]))
        for earlier in m.blocks[:layer]:
            e = earlier(e)
        n1 = block.norm1(e)
        z = (e - e.mean(-1, keepdim=True)) / torch.sqrt(e.var(-1, correction=0, keepdim=True) + block.norm1.eps)
        attended = block.attention(n1)
        h = e + attended
        f = block.feed_forward(block.norm2(h))
        out = h + f
        reference = block(e)
        e2 = e.clone()                                  # change one character's vector
        e2[0, probe] += torch.randn(e.shape[-1], generator=torch.Generator().manual_seed(SEED))
        norm_changed = block.norm1(e2) - n1
        ffn_changed = block.feed_forward(block.norm2(e2)) - block.feed_forward(block.norm2(e))
        attn_changed = block.attention(block.norm1(e2)) - attended
        formula_gap = (block.norm1.weight * z + block.norm1.bias - n1).abs().max()
        gamma = block.norm1.weight.detach().clone()
    others = [i for i in range(ids.shape[1]) if i != probe]
    moved = attn_changed.abs().amax(-1)[0]
    return dict(
        characters=ids.shape[1], d=e.shape[-1], hidden=block.feed_forward[0].out_features,
        activation=type(block.feed_forward[1]).__name__,
        by_hand_vs_block=float((out - reference).abs().max()),
        norm_matches_formula=float(formula_gap),
        z_mean=float(z.mean(-1).abs().max()), z_std=float(z.std(-1, correction=0).sub(1).abs().max()),
        gamma=(float(gamma.min()), float(gamma.max())),
        attention_update=float((attended.norm(dim=-1) / e.norm(dim=-1)).median()),
        ffn_update=float((f.norm(dim=-1) / h.norm(dim=-1)).median()),
        probe=probe,
        norm_other_rows=float(norm_changed[0, others].abs().max()),
        ffn_other_rows=float(ffn_changed[0, others].abs().max()),
        attention_before_probe=float(moved[:probe].max()),
        attention_after_probe=float(moved[probe + 1:].min()),
    )


# ======================================================================================================
# C4b -- one row, one distribution, one loop (9.A3 "What a Decoder Is Built For", "How a Decoder Writes")
# ======================================================================================================
DECODER_WORDS: tuple[str, ...] = ("maintain", "raise", "reduce", "keep", "lower")   # the schematic's words


def next_char_logits(machine, prefix: str) -> np.ndarray:
    import torch
    ids = machine.encode(prefix)[None, :]
    with torch.no_grad():
        return machine.model(ids[:, -machine.model.context:])[0, -1].double().numpy()


def greedy(machine, prefix: str, n_new: int = 60) -> str:
    """Predict, take the top character, append, repeat."""
    import torch
    ids = machine.encode(prefix)[None, :]
    with torch.no_grad():
        for _ in range(n_new):
            nxt = machine.model(ids[:, -machine.model.context:])[0, -1].argmax().reshape(1, 1)
            ids = torch.cat((ids, nxt), dim=1)
    return machine.decode(ids[0].tolist())


def word_probability(machine, prefix: str, word: str) -> float:
    """P(word, then a space | prefix) for a character model: one next-character probability per character."""
    p = 1.0
    for k, ch in enumerate(word + " "):
        p *= float(at.softmax(next_char_logits(machine, prefix + word[:k]))[machine.stoi[ch]])
    return p


def at_temperature(logits: np.ndarray, tau: float) -> np.ndarray:
    return at.softmax(np.asarray(logits, float) / tau)


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def top_p_size(p: np.ndarray, mass: float = 0.9) -> int:
    """The smallest number of characters whose probabilities add up to at least `mass`."""
    return int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), mass) + 1)
