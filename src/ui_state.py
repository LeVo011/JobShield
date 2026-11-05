"""Small deterministic helpers for preventing stale Streamlit result state."""

def next_analysis_generation(current):
    try: value=int(current)
    except (TypeError,ValueError): value=0
    return value+1

def results_match_input(analysis,current_text):
    return bool(analysis) and analysis.get("message")==current_text

def checklist_widget_key(generation,index):
    return f"verify_{generation}_{index}"
