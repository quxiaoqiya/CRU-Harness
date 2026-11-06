"""Export allowlisted visual-overflow results without private inputs or replies.

Usage: python3 -S scripts/audits/export_visual_collective.py LOCAL_EXPERIMENT
No model calls, downloads, source changes, or regrading are performed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEM = 'visual_collective_20261105'
METHODS = ['long_context_agent/multimodal_page_harness.py',
           'long_context_agent/multimodal_page_harness_v2.py',
           'long_context_agent/visual_count_memory_v21.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def export(work, destination=ROOT):
    work, destination = Path(work), Path(destination)
    def read(name):
        return json.loads((work / name).read_text())
    original, final, clean = read('audit.json'), read('final_audit.json'), read('clean501_audit.json')
    assert read('completion_pipeline.json')['phase'] == 'complete'
    assert clean['status'] == 'pass_with_disclosed_pixel_duplicate'
    for name, sha in clean['immutable_artifact_sha256'].items():
        assert digest(work / name) == sha, name
    fixed = read('all_predictions_fixed_before_scoring.json')['fixed_unix']
    opened = read('formal/scoring_access.json')['opened_unix']
    assert fixed < opened
    frozen = read('formal/freeze.json')
    for name, sha in frozen['code_sha256'].items():
        assert digest(ROOT / name) == sha
    fresh_freeze = read('confirmation_2c/freeze.json')
    for name, sha in fresh_freeze['code_sha256'].items():
        assert digest(ROOT / name) == sha
    main_manifest = read('formal/run_summary.json')
    for name, sha in main_manifest['prediction_sha256'].items():
        assert digest(work/'formal/predictions'/name) == sha
    fresh_manifest = read('confirmation_2c/fixed_predictions.json')
    for name, sha in fresh_manifest['prediction_sha256'].items():
        assert digest(work/'confirmation_2c/predictions'/name) == sha
    fields = ['id', 'case_id', 'cohort', 'scale', 'arm', 'status', 'physical_calls', 'tokens',
              'all_usage_known', 'within_budget', 'empty_answer', 'unknown', 'capacity_unavailable',
              'correct', 'evidence_recall', 'all_pages_seen', 'source_pages', 'related_pixel_tokens',
              'after_filename_filter_pixel_tokens', 'facts_observed', 'per_page_exact_all',
              'per_page_exact_observed', 'relative_count_error']
    rows = [{k: r.get(k) for k in fields} for r in read('scored_records.json')]
    for row in rows:
        if row['capacity_unavailable']:
            row['correct'] = None
    for metric in final['fresh_confirmation_metrics']:
        record = read('confirmation_2c/predictions/' + metric['arm'] + '.json')
        row = {k: None for k in fields}
        row.update(id=record['id'], case_id='fresh2c', cohort='fresh_confirmation_502', scale=2,
                   arm=metric['arm'], status=record['status'], physical_calls=metric['physical_calls'],
                   tokens=metric['tokens'], all_usage_known=record['usage']['all_usage_known'],
                   within_budget=record['usage']['within_budget'], empty_answer=not bool(record['answer']),
                   unknown=not str(metric['answer']).strip().isdecimal(), capacity_unavailable=False,
                   correct=metric['correct'], evidence_recall=metric['evidence_recall'],
                   all_pages_seen=metric['evidence_recall']==1, source_pages=502,
                   related_pixel_tokens=515052, after_filename_filter_pixel_tokens=515052,
                   facts_observed=round(metric['fact_coverage']*502),
                   per_page_exact_all=metric['per_page_exact_all'],
                   per_page_exact_observed=metric['per_page_exact_observed'],
                   relative_count_error=metric['relative_count_error'])
        rows.append(row)
    # Per-page CSV exposes correctness/exposure only, not raw labels or count vectors.
    page_rows = []
    with (work/'confirmation_per_page_counts.csv').open() as stream:
        for row in csv.DictReader(stream):
            page_rows.append({k: row[k] for k in ['arm', 'source_id', 'pixel_sha256',
                              'in_pixel_disjoint_501', 'pixels_exposed', 'fact_emitted', 'both_fields_exact']})
    with (work/'all_physical_requests.csv').open() as stream:
        receipt_fields = ['purpose', 'task_id', 'arm', 'physical_request', 'status', 'prompt_tokens',
                          'completion_tokens', 'charged_tokens', 'usage_known', 'admission_prompt_tokens',
                          'requested_output_tokens', 'finish_reason']
        receipts = [{k: row[k] for k in receipt_fields} for row in csv.DictReader(stream)]
    source = {
        'initial_pages': [{k: p[k] for k in ['id', 'source_png', 'source_split', 'image_sha256']}
                          for p in read('doclaynet_pages.json')],
        'confirmation_pages': [{k: p[k] for k in ['id', 'source_png', 'image_sha256', 'small_sha256']}
                               for p in read('confirmation_2c/pages.json')],
    }
    # Anonymous response contents and reference answers are not copied into public metrics.
    fresh_keys = ['id', 'arm', 'scale', 'correct', 'relative_count_error', 'evidence_recall',
                  'fact_coverage', 'per_page_exact_all', 'per_page_exact_observed',
                  'physical_calls', 'tokens', 'input_resolution']
    fresh_metrics = [{k: m[k] for k in fresh_keys} for m in final['fresh_confirmation_metrics']]
    for value, metric in zip(fresh_metrics, final['fresh_confirmation_metrics']):
        answer = str(metric['answer']).strip()
        value['predicted_total'] = int(answer) if answer.isdecimal() else None
        value['reference_total'] = metric['reference']
    value = {
        'schema_version':1, 'date':'2026-11-05', 'model':original['model'],
        'same_budget_ceilings':True, 'actual_consumption_matched':False, 'budget':original['budget'],
        'old_controller_tool_execution_limit':24, 'new_count_batch_pages':16,
        'new_count_output_tokens':2048, 'all_predictions_fixed_before_labels':True,
        'fixed_unix':fixed, 'scoring_opened_unix':opened, 'official_leaderboard_score':False,
        'datasets':{
            'qa':{'name':'MMLongBench-Doc', 'url':'https://huggingface.co/datasets/yubo2333/MMLongBench-Doc',
                  'revision':'2ff6aa9237fc777b6627dc57a486e9225ac5fb86', 'indices':[15,164,407,549,625,782,844],
                  'n':7, 'target_documents_disjoint_from_prior_47':True,
                  'judge':'Qwen3.8-Flash-Next-FP8, method-blind semantic equivalence; local substitute',
                  'relevant_evidence_itself_above_C_required':False},
            'global':{'name':'DocLayNet 1.0 derived Picture+Table count census',
                      'url':'https://github.com/DS4SD/DocLayNet',
                      'archive_url':'https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip',
                      'archive_etag':'52242220284e12538949c5b06eb2777c-448', 'split':'test',
                      'resolution':[1025,1025], 'image_tokens_including_markers':1026,
                      'eligibility':'Every page has positive human Picture or Table count; no ranking by count magnitude',
                      'minimal_oracle_crops_proven_above_C':False,
                      'all_pages_required_after_filename_filter':True,
                      'original_scales_nested_not_independent':True}},
        'qa_summary':[s for s in original['summary'] if s['cohort']=='qa'],
        'rows':rows, 'fresh502_metrics':fresh_metrics,
        'pixel_deduplication':{
            'source_filename_overlap':0, 'pixel_overlap':1, 'exclusions':clean['exclusions'],
            'analysis_type':clean['analysis_type'], 'clean_pages':501,
            'related_original_tokens':514026, 'related_original_tokens_over_C':514026/262144,
            'generation_calls_for_slice':0, 'not_a_new_501_page_generation':True,
            'metrics':clean['metrics'], 'errata':clean['generated_artifact_errata']},
        'accounting':{
            'main_calls':original['main_calls'], 'main_tokens':original['main_tokens'],
            'judge_calls':original['judge_calls'], 'judge_tokens':original['judge_tokens'],
            'development_calls':original['development_calls'], 'development_tokens':original['development_tokens'],
            'confirmation_calls':final['additional_fresh_calls'], 'confirmation_tokens':final['additional_fresh_tokens'],
            'total_physical_calls':clean['total_physical_calls'], 'total_physical_tokens':clean['total_physical_tokens'],
            'prompt_tokens':clean['prompt_tokens'], 'completion_tokens':clean['completion_tokens'],
            'historical_calibration_charged_again':False, 'currency_cost':None},
        'offline_replays':{'original':46,'fresh':5,'total':51,'network_calls':0},
        'frozen_method_sha256':{n: digest(ROOT/n) for n in METHODS},
        'local_artifact_sha256':{n:digest(work/n) for n in ['formal/freeze.json','confirmation_2c/freeze.json',
                              'clean501_audit.json','final_audit.json','corrected_delivery_manifest.json']},
        'test_lineage':{
            'original_v21_test_sha256':digest(ROOT/'tests/test_visual_count_memory_v21.py'),
            'published_v21_test':'tests/test_visual_count_memory_public.py',
            'only_import_changed_from':'scripts.finalize_visual_collective_v21.explicit_global_answer',
            'only_import_changed_to':'long_context_agent.visual_count_outputs.explicit_global_answer'},
        'release_notes':[
            'Opt-in modules; existing defaults unchanged. Frozen inference method files are published byte-for-byte.',
            'Public v2.1 test imports a portable scalar-output helper instead of the private experiment finalizer; assertions are unchanged.',
            'Public verifier checks arithmetic, hashes and budgets, not semantic truth or full private trajectory replay.',
            'Original 502-page claim of zero image overlap is corrected to one duplicate; the 501-page slice is post-hoc.',
            'No exact global count is correct. Full evidence exposure is not equivalent to object recall.',
            'Raw images, labels, prompts, replies and credentials remain local.'],
    }
    out = destination/'benchmarks'; out.mkdir(exist_ok=True)
    write_csv(out/(STEM+'.csv'), rows)
    write_csv(out/(STEM+'_pages.csv'), page_rows)
    write_csv(out/(STEM+'_requests.csv'), receipts)
    dump(out/(STEM+'_sources.json'), source)
    value['public_data_sha256'] = {p.name:digest(p) for p in [out/(STEM+'.csv'),out/(STEM+'_pages.csv'),
                                          out/(STEM+'_requests.csv'),out/(STEM+'_sources.json')]}
    value['public_code_sha256'] = {n:digest(ROOT/n) for n in [*METHODS,
          'long_context_agent/experiment_budget.py','long_context_agent/visual_count_outputs.py','tests/test_multimodal_page_harness.py',
          'tests/test_multimodal_page_harness_v2.py','tests/test_visual_count_memory_public.py',
          'scripts/audits/export_visual_collective.py','scripts/verify_visual_collective_results.py']}
    dump(out/(STEM+'.json'),value)
    print(json.dumps({'public_rows':len(rows),'source_diagnostics':len(page_rows),'physical_receipts':len(receipts)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',type=Path)
    parser.add_argument('--destination',type=Path,default=ROOT)
    args=parser.parse_args()
    export(args.experiment,args.destination)
