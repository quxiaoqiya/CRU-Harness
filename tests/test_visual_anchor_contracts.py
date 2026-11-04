from long_context_agent.trajectory_memory import TrajectoryStore
from long_context_agent.visual_anchor_memory_v2 import (
    VisualAnchorIndex, normalize_perception, observed_effect,
)


def test_focus_change_does_not_imply_the_checked_value_changed():
    before={'accessibility_tree':"[42] radio 'Choice', checked='false'"}
    after={'accessibility_tree':"[42] radio 'Choice', checked='false', focused"}
    effect=observed_effect(before,after,"click('42')")
    assert effect['same_id_state_attributes_unchanged'] is True
    assert effect['target_state_attributes_after']=={'checked':"'false'"}
    assert effect['effect_requires_semantic_check'] is True


def test_explicit_control_wins_over_other_controls_on_same_editor(tmp_path):
    tree="RootWebArea 'Record Editor'\n[1] button 'Return'\n[2] button 'Reset'"
    store=TrajectoryStore(tmp_path)
    store.insert({'id':'t','states':[
        {'url':'/edit','action':None,'accessibility_tree':tree},
        {'url':'/edit','action':"click('2')",'accessibility_tree':tree},
        {'url':'/all','action':"click('1')",'accessibility_tree':"RootWebArea 'All Records'"},
    ]})
    index=VisualAnchorIndex(store)
    p=normalize_perception('{}');p.update(visible_labels=['Record Editor'],requested_controls=['Return'],transition_needed=True)
    cards,links=index.retrieve_anchors(['Record Editor Return'],p)
    assert links[0]['action']=="click('1')"
    assert links[0]['effect']['target_before']=="[1] button 'Return'"


def test_keyboard_transition_indexes_original_focused_control(tmp_path):
    store=TrajectoryStore(tmp_path)
    store.insert({'id':'t','states':[
        {'url':'/form','action':None,'accessibility_tree':"RootWebArea 'Form'\n[7] slider 'Intensity', focused, value='2'"},
        {'url':'/form','action':"keyboard_press('ArrowRight')",'accessibility_tree':"RootWebArea 'Form'\n[7] slider 'Intensity', focused, value='3'"},
    ]})
    index=VisualAnchorIndex(store)
    t=index.transitions[0]
    assert 'intensity' in t['terms']
    assert t['effect']['focused_before']==["[7] slider 'Intensity', focused, value='2'"]
