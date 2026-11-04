"""Register all 451 LME-V2 Small questions without inspecting answer content."""
from pathlib import Path
import json

from scripts.prepare_longmemeval_v2_visual_handoff import BASE, ROOT, PRIOR, rows, sha, normalized, write

WORK = ROOT/'work/longmemeval_v2_full_qwen_20261104'


def prepare():
    if WORK.exists():
        raise ValueError('Never overwrite a registered full experiment')
    questions = rows(BASE/'data/questions.jsonl')
    assert len(questions) == len({q['id'] for q in questions}) == 451
    prior = [r for p in PRIOR for r in rows(p/'inputs.jsonl')]
    prior += rows(ROOT/'work/longmemeval_v2_visual_handoff_holdout_20261104/inputs.jsonl')
    old_ids, old_text = {q['id'] for q in prior}, {normalized(q['question']) for q in prior}
    assert len(old_ids) == 76
    history = json.loads((BASE/'data/haystacks/lme_v2_small.json').read_text())
    assert set(history) == {q['id'] for q in questions}
    sources = {r['id'] for r in rows(BASE/'selected_trajectories.jsonl')}
    assert all(len(h)==100 and set(h)<=sources for h in history.values())
    assert all(not q['image'] or (BASE/'data'/q['image']).is_file() for q in questions)
    WORK.mkdir()
    for name, source in [('data',BASE/'data'),('upstream',BASE/'upstream'),
        ('selected_trajectories.jsonl',BASE/'selected_trajectories.jsonl'),
        ('source_preflight_counts.json',PRIOR[2]/'source_preflight_counts.json')]:
        (WORK/name).symlink_to(source,target_is_directory=source.is_dir())
    inputs = [{k:q.get(k) for k in ('id','domain','question','image')} for q in questions]
    (WORK/'inputs.jsonl').write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in inputs))
    write(WORK/'haystacks.json',history)
    (WORK/'private_labels').mkdir()
    write(WORK/'private_labels/labels.json',{q['id']:{k:q[k] for k in ('answer','eval_function','question_type')} for q in questions})
    cohorts = {q['id']:('previously_consumed' if q['id'] in old_ids or normalized(q['question']) in old_text else 'unused_question') for q in questions}
    write(WORK/'private_labels/cohorts.json',cohorts)
    assets = json.loads((PRIOR[-1]/'assets_manifest.json').read_text())
    assets['query_images'] = {q['image']:sha(BASE/'data'/q['image']) for q in questions if q['image']}
    write(WORK/'assets_manifest.json',assets)
    write(WORK/'selection_manifest.json',{'stage':'full_evaluation','n_questions':451,
        'arms':['direct_headtail_native','harness_visual_anchor_v1','harness_visual_anchor'],
        'data_revision':assets['dataset_revision'],
        'upstream_code_commit':'2cc8c540bdb87fe6761629b585e727e1c4704520',
        'inputs_sha256':sha(WORK/'inputs.jsonl'),'labels_sha256':sha(WORK/'private_labels/labels.json'),
        'cohorts_sha256':sha(WORK/'private_labels/cohorts.json'),
        'haystack_tier':'small','history_trajectories_per_question':100,
        'n_question_images':29,'n_text_only':422,'n_previously_consumed':sum(v=='previously_consumed' for v in cohorts.values()),
        'n_unused_questions':sum(v=='unused_question' for v in cohorts.values()),
        'source_histories_reused':True,'all_questions_selected_no_filtering':True,
        'shared_text_path_reuse':True,'logical_records':1353,'physical_inference_jobs':931,
        'prior_question_id_sha256':sha(ROOT/'work/longmemeval_v2_visual_handoff_holdout_20261104/inputs.jsonl')})
    print(json.dumps({'work':str(WORK),'questions':451,'logical_records':1353,
        'unique_inference_jobs':931,'previously_consumed':sum(v=='previously_consumed' for v in cohorts.values())},indent=2))


if __name__=='__main__':prepare()
