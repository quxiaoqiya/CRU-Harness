"""Prepare immutable V3 visual regression/new-question cohorts locally.

Selection uses domain, image and exclusion hashes only, never reference content.
Labels are staged separately for the post-inference scoring process.
"""
from pathlib import Path
import hashlib
import json
import random
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'work/longmemeval_v2_qwen_20261103'
PRIOR = [BASE, ROOT/'work/longmemeval_v2_visual_holdout_20261103',
         ROOT/'work/longmemeval_v2_compact_holdout_20261103',
         ROOT/'work/longmemeval_v2_visual_anchor_holdout_20261103']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(4*1024*1024):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def normalized(text):
    return hashlib.sha256(re.sub(r'\s+', ' ', text).strip().casefold().encode()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def prepare():
    questions = rows(BASE/'data/questions.jsonl')
    by_id = {q['id']: q for q in questions}
    prior = [r for p in PRIOR for r in rows(p/'inputs.jsonl')]
    old_ids = {r['id'] for r in prior}
    old_text = {normalized(r['question']) for r in prior}
    old_images = {sha(BASE/'data'/r['image']) for r in prior if r['image']}
    assert len(old_ids) == 68
    dev = rows(ROOT/'work/longmemeval_v2_visual_anchor_dev_20261103/inputs.jsonl')
    dev += rows(PRIOR[-1]/'inputs.jsonl')
    assert len({q['id'] for q in dev}) == 20
    candidates = [q for q in questions if q['image'] and q['id'] not in old_ids
                  and normalized(q['question']) not in old_text]
    assert all((BASE/'data'/q['image']).is_file() for q in candidates), 'Missing original query images'
    candidates = [q for q in candidates if sha(BASE/'data'/q['image']) not in old_images]
    chosen = []
    rng = random.Random(20261105)
    for domain in ('enterprise', 'web'):
        pool = sorted((q for q in candidates if q['domain'] == domain), key=lambda q:q['id'])
        rng.shuffle(pool)
        assert len(pool) >= 4
        chosen += pool[:4]
    for stage, selected, arms in [
        ('development', dev, ['harness_visual_anchor']),
        ('holdout', chosen, ['direct_headtail_native', 'harness_visual_anchor_v1', 'harness_visual_anchor']),
    ]:
        work = ROOT/('work/longmemeval_v2_visual_handoff_'+('dev' if stage == 'development' else 'holdout')+'_20261104')
        assert not work.exists(), 'Never overwrite an existing experiment'
        work.mkdir()
        for name, source in [('data',BASE/'data'), ('upstream',BASE/'upstream'),
                             ('selected_trajectories.jsonl',BASE/'selected_trajectories.jsonl'),
                             ('source_preflight_counts.json',PRIOR[2]/'source_preflight_counts.json')]:
            (work/name).symlink_to(source, target_is_directory=source.is_dir())
        inputs = [{k:q.get(k) for k in ('id','domain','question','image')} for q in selected]
        (work/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inputs))
        haystacks = json.loads((BASE/'data/haystacks/lme_v2_small.json').read_text())
        write(work/'haystacks.json', {q['id']:haystacks[q['id']] for q in inputs})
        (work/'private_labels').mkdir()
        write(work/'private_labels/labels.json', {q['id']:{k:by_id[q['id']][k]
              for k in ('answer','eval_function','question_type')} for q in inputs})
        assets = json.loads((PRIOR[-1]/'assets_manifest.json').read_text())
        assets['query_images'] = {q['image']:sha(BASE/'data'/q['image']) for q in inputs}
        write(work/'assets_manifest.json', assets)
        manifest = {'stage':stage, 'n_questions':len(inputs), 'seed':20261105, 'arms':arms,
            'data_revision':assets['dataset_revision'],
            'upstream_code_commit':'2cc8c540bdb87fe6761629b585e727e1c4704520',
            'inputs_sha256':sha(work/'inputs.jsonl'), 'labels_sha256':sha(work/'private_labels/labels.json'),
            'selected':[{'id':r['id'],'domain':r['domain'],'question_image':True} for r in inputs],
            'source_histories_reused':True, 'selected_candidate_version':3,
            'prior_question_count':len(old_ids),
            'prior_id_sha256':hashlib.sha256(json.dumps(sorted(old_ids)).encode()).hexdigest(),
            'question_id_overlap_with_prior68':len({q['id'] for q in inputs}&old_ids),
            'normalized_question_hash_overlap_with_prior68':len({normalized(q['question']) for q in inputs}&old_text),
            'query_image_sha_overlap_with_prior68':len(set(assets['query_images'].values())&old_images),
            'selection':'20 consumed visual questions for diagnosis; eight remaining image questions selected by domain/seed only; no outcome-based candidate changes after freeze'}
        if stage == 'holdout':
            assert manifest['question_id_overlap_with_prior68'] == manifest['normalized_question_hash_overlap_with_prior68'] == manifest['query_image_sha_overlap_with_prior68'] == 0
        write(work/'selection_manifest.json', manifest)
        print(json.dumps({'stage':stage,'n':len(inputs),'arms':arms,'work':str(work)}))


if __name__ == '__main__':
    prepare()
