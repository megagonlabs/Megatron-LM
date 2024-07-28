import argparse
import json
import sys
from typing import Any

from sentencepiece import SentencePieceProcessor


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def print_results(sp: SentencePieceProcessor, text: str):
    print(dumps(text))
    encoded_sp = sp.encode(text, out_type="immutable_proto")

    sp_pieces = [_.piece for _ in encoded_sp.pieces]
    print(dumps(sp_pieces))

    sp_result = sp.decode([_.id for _ in encoded_sp.pieces])
    if sp_result != text:
        sp_ids = [_.id for _ in encoded_sp.pieces]
        print(f">>NG")
        print(dumps(sp_result))
        print(dumps(sp_ids))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spm",
        required=True,
        type=str,
        help="path for sentencepiece model file",
    )
    parser.add_argument(
        "target_files",
        nargs='*',
        help="target text files (use default test strings if not specified)",
    )
    args = parser.parse_args()

    sp = SentencePieceProcessor(args.spm)

    if args.target_files:
        for file in args.target_files:
            with open(file, "r", encoding="utf8") as fin:
                for line in fin:
                    print_results(sp, line.rstrip("\n"))
    else:
        for line in sys.stdin:
            print_results(sp, line.rstrip("\n"))


if __name__ == "__main__":
    main()
