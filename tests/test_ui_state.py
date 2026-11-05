from src.ui_state import checklist_widget_key,next_analysis_generation,results_match_input

def test_checklist_keys_change_for_each_analysis():
    first=checklist_widget_key(next_analysis_generation(0),0)
    second=checklist_widget_key(next_analysis_generation(1),0)
    assert first!=second

def test_results_are_current_only_for_matching_input():
    result={"message":"first message"}
    assert results_match_input(result,"first message")
    assert not results_match_input(result,"changed message")
