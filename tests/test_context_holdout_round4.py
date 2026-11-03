"""Synthetic metadata only; this module never opens untouched inputs or labels."""
from copy import deepcopy
import pytest
from scripts import prepare_context_holdout_round4 as prepare


def rows():
    return [{'id': length + index, 'context_len': length, 'dataset': 'synthetic',
             'task_group': ('counting', 'timeline')[index % 2], 'task': 'fake', 'answer_type': 'fake',
             'context_window_id': length + index, '_row_idx': index}
            for length in prepare.LENGTHS for index in range(12)]


def prior():
    return [{'dataset_revision': prepare.REVISION, 'selected': [
        {'metadata': {'dataset': 'synthetic', 'context_window_id': prepare.LENGTHS[0]}, 'context_sha256': '1' * 64}]}]


def test_stage_two_selection_is_deterministic_unique_and_excludes_consumed():
    selected, identities, hashes = prepare.select_from_catalog(rows(), prior())
    assert len(selected) == 8
    assert {(r['dataset'], r['context_window_id']) for r in selected}.isdisjoint(identities)
    assert all(sum(r['context_len'] == length for r in selected) == 4 for length in prepare.LENGTHS)
    assert selected == prepare.select_from_catalog(rows(), prior())[0]


def test_question_answer_and_context_content_cannot_influence_selection():
    source = rows(); modified = deepcopy(source)
    for row in modified:
        row.update(question='injected semantics', answer='injected label', context_window_text='ignored')
    assert prepare.select_from_catalog(source, prior())[0] == prepare.select_from_catalog(modified, prior())[0]
    assert all(set(row) == set(prepare.META_KEYS) for row in prepare.select_from_catalog(modified, prior())[0])


def test_exhausted_length_fails_without_reusing_old_contexts():
    source = [row for row in rows() if row['context_len'] == prepare.LENGTHS[1] or row['_row_idx'] < 4]
    with pytest.raises(ValueError, match='need 4 independent contexts'):
        prepare.select_from_catalog(source, prior())


def test_prior_context_or_hash_duplicates_fail_closed():
    with pytest.raises(ValueError, match='not disjoint'):
        prepare.select_from_catalog(rows(), prior() * 2)


def test_pinned_revision_mismatch_fails_closed():
    consumed = prior(); consumed[0]['dataset_revision'] = 'different'
    with pytest.raises(ValueError, match='revision differs'):
        prepare.select_from_catalog(rows(), consumed)
