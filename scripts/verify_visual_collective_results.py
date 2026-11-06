"""Offline arithmetic, pixel-lineage, budget and source checks on public data.

Does not call a provider or certify semantic labels or private replay claims.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = 'visual_collective_20261105'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def truth(value):
    return str(value).lower() == 'true'


def main():
    directory = ROOT/'benchmarks'
    result = json.loads((directory/(STEM+'.json')).read_text())
    for name, sha in result['public_data_sha256'].items():
        assert digest(directory/name) == sha, name
    for name, sha in result['public_code_sha256'].items():
        assert digest(ROOT/name) == sha, name
    for name, sha in result['frozen_method_sha256'].items():
        assert result['public_code_sha256'][name] == sha
    rows = result['rows']
    assert len(rows) == len({r['id'] for r in rows}) == 51
    assert result['fixed_unix'] < result['scoring_opened_unix']
    assert result['all_predictions_fixed_before_labels']
    assert result['same_budget_ceilings'] and not result['actual_consumption_matched']
    assert not result['official_leaderboard_score']
    budget = result['budget']; assert budget['context_tokens'] == 262144
    assert budget['total_tokens'] == 12*budget['context_tokens']
    for row in rows:
        assert row['within_budget'] and row['all_usage_known']
        assert row['tokens'] <= budget['total_tokens'] and row['physical_calls'] <= budget['max_calls']
    denied = [r for r in rows if r['capacity_unavailable']]
    assert len(denied) == 1 and denied[0]['arm'] == 'native_thumbnail'
    assert denied[0]['correct'] is None and denied[0]['physical_calls'] == denied[0]['tokens'] == 0
    for summary in result['qa_summary']:
        group = [r for r in rows if r['cohort']=='qa' and r['arm']==summary['arm']]
        assert len(group) == summary['n'] == 7
        assert sum(r['correct'] for r in group) == summary['correct']
        assert summary['accuracy'] == summary['correct']/7
        assert sum(r['tokens'] for r in group) == summary['tokens']
        assert sum(r['physical_calls'] for r in group) == summary['calls']
        assert abs(sum(r['evidence_recall'] for r in group)/7-summary['evidence_recall']) < 1e-12
    with (directory/(STEM+'_requests.csv')).open() as stream:
        receipts = list(csv.DictReader(stream))
    accounting = result['accounting']
    assert len(receipts) == accounting['total_physical_calls'] == 726
    assert len({(r['purpose'],r['task_id'],r['physical_request']) for r in receipts}) == len(receipts)
    for row in receipts:
        assert row['status']=='ok' and truth(row['usage_known'])
        assert int(row['admission_prompt_tokens'])+int(row['requested_output_tokens']) <= budget['context_tokens']
        assert int(row['requested_output_tokens']) <= budget['max_output_tokens']
        assert int(row['prompt_tokens'])+int(row['completion_tokens']) == int(row['charged_tokens'])
    for key in ['prompt_tokens', 'completion_tokens']:
        assert sum(int(r[key]) for r in receipts) == accounting[key]
    assert sum(int(r['charged_tokens']) for r in receipts) == accounting['total_physical_tokens'] == 13928581
    purpose_names = {'main':'main', 'judge':'judge', 'development':'development',
                     'independent_confirmation':'confirmation'}
    for purpose, prefix in purpose_names.items():
        group = [r for r in receipts if r['purpose']==purpose]
        assert len(group) == accounting[prefix+'_calls']
        assert sum(int(r['charged_tokens']) for r in group) == accounting[prefix+'_tokens']
    assert sum(r['tokens'] for r in rows if r['cohort']!='fresh_confirmation_502') == accounting['main_tokens']
    assert sum(r['tokens'] for r in rows if r['cohort']=='fresh_confirmation_502') == accounting['confirmation_tokens']
    source = json.loads((directory/(STEM+'_sources.json')).read_text())
    initial, fresh = source['initial_pages'], source['confirmation_pages']
    assert len(initial)==2106 and len(fresh)==502
    assert {p['source_png'] for p in initial}.isdisjoint(p['source_png'] for p in fresh)
    old_pixels = {p['image_sha256'] for p in initial}
    clean = {p['id'] for p in fresh if p['image_sha256'] not in old_pixels}
    duplicates = [p for p in fresh if p['image_sha256'] in old_pixels]
    assert len(clean)==501 and len(duplicates)==1 and duplicates[0]['id']=='c0024'
    assert len({p['image_sha256'] for p in fresh}) == 502
    dedup = result['pixel_deduplication']
    assert dedup['source_filename_overlap']==0 and dedup['pixel_overlap']==1
    assert dedup['clean_pages']==501 and dedup['generation_calls_for_slice']==0
    assert dedup['related_original_tokens']==501*1026 > budget['context_tokens']
    assert dedup['not_a_new_501_page_generation']
    with (directory/(STEM+'_pages.csv')).open() as stream:
        pages = list(csv.DictReader(stream))
    assert len(pages)==2510 and len({(p['arm'],p['source_id']) for p in pages})==2510
    sha_by_id = {p['id']:p['image_sha256'] for p in fresh}
    for page in pages:
        assert page['pixel_sha256'] == sha_by_id[page['source_id']]
        assert truth(page['in_pixel_disjoint_501']) == (page['source_id'] in clean)
    for metric in dedup['metrics']:
        group = [p for p in pages if p['arm']==metric['arm'] and p['source_id'] in clean]
        assert len(group)==501
        assert metric['pixel_page_exposure']==sum(truth(p['pixels_exposed']) for p in group)/501
        assert metric['complete_fact_coverage']==sum(truth(p['fact_emitted']) for p in group)/501
        if metric['per_page_both_fields_exact'] is not None:
            assert metric['correct_pages']==sum(truth(p['both_fields_exact']) for p in group)
            assert metric['per_page_both_fields_exact']==metric['correct_pages']/501
        number = metric['derived_subcohort_prediction']
        if number is not None:
            assert metric['relative_total_error']==abs(number-metric['reference'])/metric['reference']
        assert not metric['exact_total']
    assert result['offline_replays']=={'original':46,'fresh':5,'total':51,'network_calls':0}
    print('PASS: 51 outcomes, 726 physical receipts, 501 pixel-disjoint pages after one disclosed duplicate; public arithmetic/budgets/hashes only.')


if __name__=='__main__':
    main()
