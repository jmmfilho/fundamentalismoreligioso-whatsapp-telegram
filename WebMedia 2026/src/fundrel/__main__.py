"""Command line entry point.

    python -m fundrel datasets   # Sec. 3.1, Table 1, Figure 2
    python -m fundrel ner        # Sec. 4.1, Table 2
    python -m fundrel retrieval  # Sec. 4.2 (exports messages for manual reading)
    python -m fundrel polarity   # Sec. 4.3, Table 3
    python -m fundrel topics     # Sec. 4.4, Tables 4-7
    python -m fundrel liwc       # Sec. 4.5
    python -m fundrel verify     # compare results/generated with the paper
    python -m fundrel all        # everything above, in order
"""
from __future__ import annotations

import argparse
import logging

STEPS = ["datasets", "ner", "retrieval", "polarity", "topics", "liwc", "verify"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="fundrel", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["all"])
    ap.add_argument("--models", nargs="+", default=["lda", "ptm", "bertopic"],
                    help="topic models to run (topics step)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from . import descriptive, liwc, ner, polarity, search, topics, verify
    actions = {
        "datasets": descriptive.run, "ner": ner.run, "retrieval": search.run,
        "polarity": polarity.run, "topics": lambda: topics.run(tuple(args.models)),
        "liwc": liwc.run, "verify": lambda: print(verify.run().to_string(index=False)),
    }
    for step in (STEPS if args.step == "all" else [args.step]):
        logging.info("== %s", step)
        actions[step]()


if __name__ == "__main__":
    main()
