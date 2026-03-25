"""split_sentences.py

Used to assign gold_sentence_idx values to QA pairs
"""
from __future__ import annotations
import argparse
import sys
from lmk_train_data import load_text_file, remove_references, split_into_sentences


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split a Nougat .mmd paper into numbered sentences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    #mutually exclusive: either --paper_id OR --input (+ --output)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--paper_id",
        type=int,
        metavar="N",
        help="Paper index N → reads TEST_PAPER{N}.mmd, writes TEST_PAPER{N}_sentences.txt",
    )
    source.add_argument(
        "--input",
        type=str,
        metavar="FILE",
        help="Explicit input .mmd file (requires --output).",
    )

    ap.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Output .txt file (auto-set when --paper_id is used).",
    )
    ap.add_argument(
        "--start",
        type=int,
        default=0,
        metavar="N",
        help="First sentence index to write (default: 0).",
    )
    ap.add_argument(
        "--end",
        type=int,
        default=None,
        metavar="N",
        help="First sentence index NOT to write, i.e. exclusive end (default: all).",
    )

    args = ap.parse_args()


    if args.paper_id is not None:
        input_path = f"TEST_PAPER{args.paper_id}.mmd"
        output_path = args.output or f"TEST_PAPER{args.paper_id}_sentences.txt"
    else:
        input_path = args.input
        if not args.output:
            ap.error("--output is required when --input is used.")
        output_path = args.output

    text = load_text_file(input_path)
    text = remove_references(text)
    sents = split_into_sentences(text)

    n = len(sents)
    start = max(0, args.start)
    end = min(args.end if args.end is not None else n, n)

    if start >= end:
        print(f"No sentences to write (start={start}, end={end}, total={n}).", file=sys.stderr)
        sys.exit(1)

    print(f"Total sentences : {n}")
    print(f"Writing [{start}:{end}] → {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(start, end):
            f.write(f"[{i:04d}] {sents[i]}\n\n")

    print("Done.")


if __name__ == "__main__":
    main()
