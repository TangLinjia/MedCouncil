"""Topic-held-out evaluation: audit -> development tuning -> frozen test.

Commands: python -m src.rag.benchmark audit|tune|test [--include-unreviewed]
Only tune reads development outcomes. Test results never select parameters.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
from langchain_core.documents import Document

from .chunker import corpus_fingerprint
from .evaluation import retrieval_metrics, select_queries, validate_corpus
from .retriever import reciprocal_rank_fusion, rerank, retrieve_guidelines, rewrite_query, sparse_search
from .vectorstore import get_vectorstore
from src.services.config import get_settings

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/evaluation'
LEXICAL_THRESHOLD = 0.65
SEMANTIC_THRESHOLD = 0.90
GRID = [(candidate_k, constant) for candidate_k in (20, 40) for constant in (20, 60, 100)]
METRICS = ('recall', 'hit_rate', 'mrr', 'ndcg')
AUDIT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def snapshot_digest(directory):
    files = sorted(path for path in directory.rglob('*')
                   if path.is_file() and '.cache' not in path.relative_to(directory).parts)
    if not files:
        raise ValueError(f'Missing local model artifacts: {directory}')
    snapshot = [(str(path.relative_to(directory)), digest(path)) for path in files]
    return hashlib.sha256(json.dumps(snapshot).encode()).hexdigest()


def model_directory(name):
    if Path(name).is_dir():
        return Path(name)
    cached = ROOT / '.cache/huggingface/hub' / ('models--' + name.replace('/', '--'))
    commit = (cached / 'refs/main').read_text().strip()
    return cached / 'snapshots' / commit


def audit_signature(report):
    return hashlib.sha256(json.dumps({key: value for key, value in report.items() if key != 'created_at'},
                                    sort_keys=True).encode()).hexdigest()


def groups(rows):
    return {group for row in rows for group in row.get('source_group_ids', [row['group_id']])}


def contextual_text(row):
    history = ' '.join(turn['role'] + ': ' + turn['content'] for turn in row['history'])
    return (history + ' ' + row['query']).strip()


def canonical(text):
    return re.sub(r'[^\w]', '', unicodedata.normalize('NFKC', text).casefold())


def audit_splits(dev, test, *, vectors=None):
    """Heuristic duplicate screening, NOT a proof of semantic independence."""
    def grams(text):
        return {text[i:i + 3] for i in range(max(1, len(text) - 2))}
    a, b = [canonical(contextual_text(r)) for r in dev], [canonical(contextual_text(r)) for r in test]
    exact, lexical, semantic, within = [], [], [], []
    max_lexical, max_semantic = 0., None
    ga, gb = [grams(s) for s in a], [grams(s) for s in b]
    if vectors is not None:
        vectors = np.asarray(vectors, dtype=float)
        if len(vectors) != len(dev) + len(test) or not np.isfinite(vectors).all():
            raise ValueError('Invalid audit embeddings')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError('Audit embeddings must not be zero')
        vectors = vectors / norms
    for i, left in enumerate(dev):
        for j, right in enumerate(test):
            pair = {'left': left['id'], 'right': right['id']}
            if a[i] == b[j]:
                exact.append(pair)
            score = len(ga[i] & gb[j]) / max(1, len(ga[i] | gb[j]))
            max_lexical = max(max_lexical, score)
            if score >= LEXICAL_THRESHOLD:
                lexical.append({**pair, 'jaccard': float(score)})
            if vectors is not None:
                similarity = float(vectors[i] @ vectors[len(dev) + j])
                max_semantic = max(max_semantic if max_semantic is not None else -1., similarity)
                if similarity >= SEMANTIC_THRESHOLD:
                    semantic.append({**pair, 'cosine': similarity})
    for rows in (dev, test):
        seen = {}
        for row in rows:
            key = canonical(contextual_text(row))
            if key in seen:
                within.append({'left': seen[key], 'right': row['id']})
            seen[key] = row['id']
    shared = sorted(groups(dev) & groups(test))
    return {'exact_overlap': exact, 'within_exact_duplicates': within, 'shared_groups': shared,
            'lexical_overlap': lexical, 'semantic_overlap': semantic,
            'lexical_threshold': LEXICAL_THRESHOLD, 'semantic_threshold': SEMANTIC_THRESHOLD,
            'max_cross_lexical_jaccard': max_lexical, 'max_cross_semantic_cosine': max_semantic,
            'semantic_model': AUDIT_MODEL if vectors is not None else None,
            'limitation': 'Threshold screening cannot guarantee absence of semantic duplicates or pretraining contamination.'}


def inventory():
    data = get_vectorstore().get(include=['documents', 'metadatas'])
    return [Document(page_content=text, metadata=meta)
            for text, meta in zip(data['documents'], data['metadatas'])]


def fingerprints(documents):
    paths = sorted((ROOT / 'src/rag').glob('*.py')) + [ROOT / 'src/services/config.py', ROOT / 'requirements.txt']
    code = [(str(path.relative_to(ROOT)), digest(path)) for path in paths]
    import ollama
    models = {entry.model.removesuffix(':latest'): entry.digest for entry in ollama.list().models}
    settings = get_settings()
    runtime = {'rewriter': settings.ollama_model, 'rewriter_digest': models.get(settings.ollama_model.removesuffix(':latest')),
               'embedding_digest': models.get('nomic-embed-text'), 'reranker': settings.rag_reranker_model,
               'reranker_artifacts': snapshot_digest(model_directory(settings.rag_reranker_model)),
               'audit_artifacts': snapshot_digest(model_directory(AUDIT_MODEL))}
    if runtime['rewriter_digest'] is None or runtime['embedding_digest'] is None:
        raise ValueError('Configured Ollama models are not installed')
    return {'code_sha256': hashlib.sha256(json.dumps(code).encode()).hexdigest(),
            'dev_sha256': digest(DATA / 'dev.json'), 'test_sha256': digest(DATA / 'test.json'),
            'corpus_sha256': corpus_fingerprint(documents),
            'runtime_sha256': hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()}


def check_frozen(frozen, current):
    for key, value in current.items():
        if frozen.get(key) != value:
            raise ValueError(f'Frozen experiment changed: {key}; final test must not be used for retuning.')


def validated_data(documents, include_unreviewed):
    dev_data, test_data = read(DATA / 'dev.json'), read(DATA / 'test.json')
    dev, test = dev_data['queries'], test_data['queries']
    ids = {d.metadata['passage_id'] for d in documents}
    for name, cases in [('dev', dev), ('test', test)]:
        if len(cases) != 120 or len({r['id'] for r in cases}) != 120:
            raise ValueError(f'{name} must contain exactly 120 unique query IDs')
        if len({r['intent_id'] for r in cases}) != 120:
            raise ValueError(f'{name} contains duplicate intent IDs')
        validate_corpus(cases, documents)
        for row in cases:
            gold = row['relevant_passage_ids']
            if not gold or len(gold) != len(set(gold)) or not set(gold) <= ids:
                raise ValueError(f'Invalid gold labels: {row["id"]}')
            if set(row['relevant_source_ids']) != {passage.split('#')[0] for passage in gold}:
                raise ValueError(f'Source labels do not match passage labels: {row["id"]}')
        if len(select_queries(cases, include_unreviewed)) != 120:
            raise ValueError(f'{name} has unreviewed labels. Review all 120 or explicitly use --include-unreviewed for synthetic experiments.')
    if {r['intent_id'] for r in dev} & {r['intent_id'] for r in test}:
        raise ValueError('Intent leakage across splits')
    if {r['id'] for r in dev} & {r['id'] for r in test}:
        raise ValueError('Query ID leakage across splits')
    return dev, test


def do_audit(dev, test):
    texts = [contextual_text(r) for r in dev + test]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(AUDIT_MODEL, cache_folder=str(ROOT / '.cache/huggingface/hub'), device='cpu')
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    report = audit_splits(dev, test, vectors=vectors)
    report.update(created_at=now(), dev_sha256=digest(DATA / 'dev.json'), test_sha256=digest(DATA / 'test.json'),
                  dev_count=len(dev), test_count=len(test),
                  dev_relevance_counts=dict(Counter(len(r['relevant_passage_ids']) for r in dev)),
                  test_relevance_counts=dict(Counter(len(r['relevant_passage_ids']) for r in test)))
    report['passed'] = not any(report[key] for key in ('exact_overlap', 'within_exact_duplicates', 'shared_groups',
                                                       'lexical_overlap', 'semantic_overlap'))
    write(DATA / 'split_audit.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report['passed']:
        raise ValueError('Split audit found potential leakage. Resolve before tuning; do not lower thresholds to pass.')
    return report


def mean_metrics(rows):
    return {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}


def tune(dev, test, documents):
    if (DATA / 'final_results.json').exists():
        raise ValueError('Final test already exposed. Create a new independent holdout before retuning.')
    audit = do_audit(dev, test)
    state = fingerprints(documents)
    store = get_vectorstore()
    trials = {f'{cap}:{constant}': [] for cap, constant in GRID}
    for row in dev:
        query = rewrite_query(row['query'], row['history'])
        dense = store.similarity_search(query, k=min(40, len(documents)))
        sparse = sparse_search(query, documents, min(40, len(documents)))
        candidates = {f'{cap}:{constant}': reciprocal_rank_fusion([sparse[:cap], dense[:cap]], constant)[:cap]
                      for cap, constant in GRID}
        union = {d.metadata['passage_id']: d for ranking in candidates.values() for d in ranking}
        scored = rerank(query, list(union.values()), len(union))
        scores = {d.metadata['passage_id']: d.metadata['rerank_score'] for d in scored}
        for key, ranking in candidates.items():
            ranking = sorted(ranking, key=lambda d: scores[d.metadata['passage_id']], reverse=True)[:5]
            ids = [d.metadata['passage_id'] for d in ranking]
            trials[key].append({'id': row['id'], **retrieval_metrics(ids, set(row['relevant_passage_ids']), 5)})
        print('DEV', row['id'], flush=True)
    summary = {key: mean_metrics(rows) for key, rows in trials.items()}
    best = max(summary, key=lambda key: (summary[key]['recall'], summary[key]['ndcg'],
                                         -int(key.split(':')[0]), -abs(int(key.split(':')[1]) - 60)))
    cap, constant = map(int, best.split(':'))
    check_frozen(state, fingerprints(inventory()))
    freeze = {**state, 'frozen_at': now(), 'parameters': {'candidate_k': cap, 'rrf_constant': constant},
              'top_k': 5, 'mode': 'hybrid', 'selected_using': 'dev only; macro Recall@5 then nDCG@5',
              'dev_summary': summary, 'audit_sha256': audit_signature(audit),
              'annotation_status': 'synthetic_pending_human_review' if any(r['annotation_status'] != 'human_reviewed' for r in dev + test) else 'human_reviewed',
              'models': {'embedding': 'nomic-embed-text', 'reranker': get_settings().rag_reranker_model,
                         'rewriter': get_settings().ollama_model}}
    write(DATA / 'dev_tuning.json', {'summary': summary, 'trials': trials})
    write(DATA / 'frozen_experiment.json', freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


def evaluate_test(test, documents):
    frozen_path, output = DATA / 'frozen_experiment.json', DATA / 'final_results.json'
    if not frozen_path.exists():
        raise ValueError('Run tune before evaluating the held-out test')
    frozen = read(frozen_path)
    check_frozen(frozen, fingerprints(documents))
    if audit_signature(read(DATA / 'split_audit.json')) != frozen['audit_sha256']:
        raise ValueError('Split audit differs from the frozen audit')
    if output.exists():
        report = read(output)
        if report['freeze_sha256'] != digest(frozen_path):
            raise ValueError('Cannot resume a different frozen experiment')
        if report['status'] == 'completed':
            raise ValueError('Final test already completed; inspect its saved result, do not retune on it.')
    else:
        report = {'started_at': now(), 'status': 'running', 'freeze_sha256': digest(frozen_path),
                  'top_k': 5, 'relevance_unit': 'passage', 'aggregation': 'macro over 120 queries',
                  'recall_definition': '|top5 intersect labeled relevant passages| / |all labeled relevant passages|',
                  'hit_rate_definition': '1 if top5 contains any labeled relevant passage, else 0',
                  'novelty_claim': 'Topic and intent held out from this tuning run; not a claim about model pretraining.',
                  'annotation_counts': dict(Counter(r['annotation_status'] for r in test)),
                  'indexed_passages': len(documents), 'results': []}
        write(output, report)
    done = {row['id'] for row in report['results']}
    for row in test:
        if row['id'] in done:
            continue
        docs = retrieve_guidelines(row['query'], 5, row['history'], **frozen['parameters'])
        ids = [d.metadata['passage_id'] for d in docs]
        report['results'].append({'id': row['id'], 'query': row['query'],
                                  'retrieval_query': docs[0].metadata['retrieval_query'] if docs else row['query'],
                                  'relevant_passage_ids': row['relevant_passage_ids'],
                                  'relevant_passage_count': len(row['relevant_passage_ids']),
                                  'relevant_source_count': len(row['relevant_source_ids']),
                                  'retrieved_ids': ids, **retrieval_metrics(ids, set(row['relevant_passage_ids']), 5)})
        write(output, report)
        print('TEST', row['id'], flush=True)
    check_frozen(frozen, fingerprints(inventory()))
    report.update(status='completed', completed_at=now(), total=len(report['results']), summary=mean_metrics(report['results']))
    write(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != 'results'}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'tune', 'test'])
    parser.add_argument('--include-unreviewed', action='store_true')
    args = parser.parse_args()
    documents = inventory()
    dev, test = validated_data(documents, args.include_unreviewed)
    if args.command == 'audit':
        do_audit(dev, test)
    elif args.command == 'tune':
        tune(dev, test, documents)
    else:
        evaluate_test(test, documents)


if __name__ == '__main__':
    main()
