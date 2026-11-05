import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation import evaluate,assess_30
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",help="Optional EMSCAD-style labelled CSV"); a=p.parse_args()
    print(json.dumps(evaluate(a.data),indent=2)); cases=assess_30(output_path="data/demo_assessments.json"); print(f"Demo scenario checks: {sum(x['correct'] for x in cases)}/{len(cases)} correct; recorded data/demo_assessments.json")
