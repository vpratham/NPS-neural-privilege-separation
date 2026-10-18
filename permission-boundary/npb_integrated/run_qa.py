"""Single-request CLI. Omit --model-path for the deterministic host reference."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]

from npb_integrated.src.authorization import Authorization
from npb_integrated.src.baseline import BaselineAssistant
from npb_integrated.src.defended import DefendedAssistant
from npb_integrated.src.providers import LocalHFProvider, OllamaProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', required=True)
    parser.add_argument('--document', type=Path, required=True)
    parser.add_argument('--system', choices=['baseline', 'defended'], default='defended')
    parser.add_argument('--provider', choices=['hf', 'ollama'], default='hf')
    parser.add_argument('--model-path')
    parser.add_argument('--base-url', default='http://127.0.0.1:11436')
    args = parser.parse_args()
    if args.system == 'baseline' and not args.model_path:
        parser.error('baseline requires an existing model via --model-path')
    provider = None
    if args.model_path:
        provider = (LocalHFProvider(args.model_path) if args.provider == 'hf'
                    else OllamaProvider(args.model_path, base_url=args.base_url))
    authorization = Authorization()
    try:
        assistant = BaselineAssistant(provider, authorization) if args.system == 'baseline' else DefendedAssistant(provider, authorization)
        result = assistant.answer(args.question, args.document.read_text(encoding='utf-8'))
        print(json.dumps(result, indent=2, ensure_ascii=True))
    finally:
        authorization.close()


if __name__ == '__main__':
    main()
