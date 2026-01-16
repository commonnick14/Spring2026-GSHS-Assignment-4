import math

def tokenize(text: str) -> list[str]:
    return text.lower().split()

def compute_tf(document: str) -> dict:
    tf = {}
    tokens = tokenize(document)
    total = len(tokens)
    if total == 0:
        return tf

    counts = {}
    for w in tokens:
        counts[w] = counts.get(w, 0) + 1

    for w, c in counts.items():
        tf[w] = c / total

    return tf

def compute_idf(docs: list[str]) -> dict:
    idf = {}
    N = len(docs)
    all_words = set()

    tokenized_docs = []
    for doc in docs:
        toks = tokenize(doc)
        tokenized_docs.append(set(toks))
        all_words.update(toks)

    for w in all_words:
        df = 0
        for doc_set in tokenized_docs:
            if w in doc_set:
                df += 1
        idf[w] = math.log(N / df) if df > 0 else 0.0

    return idf

def compute_tf_idf(document: str, idf: dict) -> dict:
    tf_idf = {}
    tf = compute_tf(document)

    for w, tfv in tf.items():
        tf_idf[w] = tfv * idf.get(w, 0.0)

    return tf_idf

def cosine_similarity(vec1: dict, vec2: dict) -> float:
    dot = 0
    for word in vec1:
        dot += vec1[word] * vec2.get(word, 0)

    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0

    return dot / (mag1 * mag2)

def tf_idf_search(query: str, documents: list[str]) -> str:
    idf = compute_idf(documents)
    query_vec = compute_tf_idf(query, idf)
    scores = []

    for doc in documents:
        doc_vec = compute_tf_idf(doc, idf)
        score = cosine_similarity(query_vec, doc_vec)
        scores.append((doc, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]
