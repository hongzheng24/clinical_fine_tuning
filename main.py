# Start gpu interact session: <interact -q gpu -g 2 -t 2:00:00>
# Activate conda env: <conda activate clinical-fine-tuning>


import argparse
import os
import sys

from src.preprocess.load_medcoder import build_dataset
from src.utils.pipeline import export_graph_diagram, run_pipeline
from src.utils.report import render_html, render_json

TEXT_FILEPATH = 'data/raw_data/text.csv'
DIAGNOSIS_FILEPATH = 'data/raw_data/diagnosis.csv'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", help="run a single cached note by key (see data/cached_notes.json)")
    parser.add_argument("--pdf", help="path to a PDF clinical document instead of a dataset note")
    parser.add_argument("--outdir", default="outputs/results")

    parser.add_argument('--train', default=False, help='train')
    parser.add_argument('--evaluate', default=False, help='evaluate')
    parser.add_argument('--model', default='base', help='Fine-tuned model or base model') # base or finetuned
    parser.add_argument('--nlp', default='True', help='Use full pipeline: Detect sections, detect entities, verify, and score confidence.')

    args = parser.parse_args()

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    '''
    ### Deprecated for asclepius dataset###
    if args.pdf:
        # jobs = [(os.path.basename(args.pdf), args.pdf, True)]
        pass
    else:
        notes = load_notes()  # downloads + caches on first run, reads cache after
        if args.note:
            jobs = [(args.note, notes[args.note], False)]
        else:
            jobs = [(name, text, False) for name, text in notes.items()]
    '''

    # For medcoder dataset
    if args.pdf:
        pass
    else:
        notes = build_dataset(TEXT_FILEPATH, DIAGNOSIS_FILEPATH)
        if args.note:
            jobs = [(args.note, notes[args.note], False)]
        else:
            jobs = [(example.doc_id, example.medical_record_text, False) for example in notes]


    for doc_id, raw_input, is_pdf in jobs:
        print(f"\n=== Running pipeline on: {doc_id} ===")
        state = run_pipeline(doc_id, raw_input, is_pdf)

        for trace in state.get('trace', []):
            print(' ', trace)

        if state.get('warnings'):
            print(' WARNINGS:')
            for warning in state.get('warnings'):
                print(' -', warning)

        html_path = os.path.join(args.outdir, f"{doc_id}.html")
        json_path = os.path.join(args.outdir, f"{doc_id}.json")
        with open(html_path, "w") as f:
            f.write(render_html(state))
        with open(json_path, "w") as f:
            f.write(render_json(state))
        print(f"  -> wrote {html_path}")
        print(f"  -> wrote {json_path}")


if __name__ == '__main__':
    sys.exit(main())