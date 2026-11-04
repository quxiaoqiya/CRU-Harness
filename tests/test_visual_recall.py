from long_context_agent.visual_recall import image_distance,route_for_query


def test_local_visual_distance_is_zero_for_identical_features_and_separates_content():
    a={'horizontal':0,'vertical':0,'colors':[0]*192}
    b={'horizontal':2**64-1,'vertical':2**64-1,'colors':[255]*192}
    assert image_distance(a,a)==0 and image_distance(a,b)==1
    assert image_distance(a,b)==image_distance(b,a)


def test_absent_query_image_preserves_legacy_execution_route():
    assert route_for_query(None)=='legacy_trajectory_harness'
    assert route_for_query('query.png')=='perceptual_visual_guard'
