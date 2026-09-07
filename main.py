"""
main.py
=========
Single entry point.

Examples
--------
# 1) Smoke-test the whole pipeline on synthetic data (no download needed):
python scripts/make_synthetic_data.py
python main.py --transaction-file data/SYNTHETIC_train_transaction.csv \
                --identity-file data/SYNTHETIC_train_identity.csv \
                --seeds 0 1 2

# 2) Full run on the real IEEE-CIS data (after placing the two CSVs under
#    data/, see README.md) with all configured seeds:
python main.py
"""

import argparse
import logging

from src import config
from src.pipeline import analysis, experiment_runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help=f"Random seeds to run (default: config.SEEDS, currently {config.SEEDS})")
    parser.add_argument("--transaction-file", type=str, default=None,
                         help="Path to train_transaction.csv (default: config.TRANSACTION_FILE)")
    parser.add_argument("--identity-file", type=str, default=None,
                         help="Path to train_identity.csv (default: config.IDENTITY_FILE)")
    parser.add_argument("--skip-experiments", action="store_true",
                         help="Skip running experiments and only (re-)run analysis on existing results")
    args = parser.parse_args()

    if args.transaction_file:
        config.TRANSACTION_FILE = args.transaction_file
    if args.identity_file:
        config.IDENTITY_FILE = args.identity_file

    if not args.skip_experiments:
        experiment_runner.run_all_seeds(args.seeds)
    else:
        logger.info("Skipping experiment stage; analyzing existing results in %s", config.TABLES_DIR)

    analysis.run_full_analysis()


if __name__ == "__main__":
    main()
