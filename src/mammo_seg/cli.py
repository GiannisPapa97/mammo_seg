"""The ``mammo-seg`` command: the data steps and training behind one entry point.

    mammo-seg manifest | splits | masks | preprocess   # the data pipeline
    mammo-seg train                                    # train a model on one fold

The data steps are the commands of ``mammo_seg.data`` (``python -m
mammo_seg.data.splits ...`` does the same as ``mammo-seg splits ...``); each takes
its own options, see ``mammo-seg <step> --help``. ``train`` imports PyTorch only
when it runs, so the data steps work without it.

The same command runs on a laptop CPU and on a Kaggle or Colab GPU: the device is
chosen at run time (``--device``), ``--output-dir`` points at the notebook's
persistent output folder, ``--resume`` continues a session that ended, and
``--max-minutes`` makes the run stop cleanly before the session limit.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from mammo_seg.data import manifest, masks, preprocess_dataset, splits
from mammo_seg.data.preprocess_dataset import RECORD_FILE, read_record
from mammo_seg.data.splits import fold_name, read_splits

DEFAULT_OUTPUT_DIR = Path("runs")
# The choices of the engine, spelled out because importing the engine imports
# PyTorch; the tests check that they are the same.
ARCHITECTURES = ("unet", "unetplusplus", "deeplabv3", "deeplabv3plus", "fpn")
SELECTION_METRICS = tuple(
    f"{group}.{metric}"
    for group in ("with_mass", "without_mass", "all_images")
    for metric in ("dice", "iou", "precision", "sensitivity")
)

# The data steps: name -> (help, main). Their options are their own.
STEPS: dict[str, tuple[str, Callable[[Sequence[str] | None], int]]] = {
    "manifest": ("list the images and lesions of the dataset", manifest.main),
    "splits": ("make patient-level cross-validation splits", splits.main),
    "masks": ("write the mass masks", masks.main),
    "preprocess": ("crop, rescale and store the images", preprocess_dataset.main),
}


def build_parser() -> argparse.ArgumentParser:
    """The parser of the whole command; the data steps are listed for ``--help``
    only, because ``main`` hands their arguments over to the steps themselves."""
    parser = argparse.ArgumentParser(
        prog="mammo-seg",
        description="Segmentation of mammography masses. Research use only: "
        "not a medical device.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, (text, _) in STEPS.items():
        commands.add_parser(name, help=text, add_help=False)

    train = commands.add_parser(
        "train",
        help="train a model on one cross-validation fold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = train.add_argument_group("data")
    data.add_argument("--manifest", type=Path, default=manifest.DEFAULT_OUTPUT)
    data.add_argument("--splits", type=Path, default=splits.DEFAULT_OUTPUT)
    data.add_argument(
        "--preprocessed-dir", type=Path, default=preprocess_dataset.DEFAULT_OUTPUT_DIR
    )
    data.add_argument("--fold", type=int, default=0, help="the validation fold")

    run = train.add_argument_group("run")
    run.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="where runs are kept (on Kaggle: /kaggle/working); this fold's "
        "checkpoints and history go to <output-dir>/fold_<fold>",
    )
    run.add_argument(
        "--device",
        default=None,
        help="auto, cpu, cuda, cuda:<n> or mps (default: $MAMMO_SEG_DEVICE, else auto)",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="continue from last.pt in the output folder; the settings must match",
    )
    run.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="stop cleanly, after saving, when another epoch would not end within "
        "this time (for sessions with a time limit); resume afterwards",
    )

    training = train.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=60)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=1e-4)
    training.add_argument(
        "--positive-fraction",
        type=float,
        default=0.5,
        help="share of training images with a mass in an epoch",
    )
    training.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="images drawn per epoch (default: the number of training images)",
    )
    training.add_argument(
        "--patience",
        type=int,
        default=15,
        help="stop after this many epochs without improvement; 0 never stops early",
    )
    training.add_argument(
        "--selection-metric",
        choices=SELECTION_METRICS,
        default="all_images.dice",
        help="the validation metric that decides which epoch is the best",
    )
    training.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mixed precision (used on a CUDA GPU only)",
    )
    training.add_argument("--num-workers", type=int, default=0)
    training.add_argument("--seed", type=int, default=0)

    model = train.add_argument_group("model")
    model.add_argument("--architecture", choices=ARCHITECTURES, default="unet")
    model.add_argument("--encoder", default="resnet34")
    model.add_argument(
        "--encoder-weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="imagenet downloads pretrained weights the first time (needs internet)",
    )
    return parser


@contextmanager
def _training_log() -> Iterator[None]:
    """Show the training log (one line per epoch) on stderr while a run lasts."""
    logger = logging.getLogger("mammo_seg")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def _train(args: argparse.Namespace) -> int:
    try:
        from mammo_seg.data.dataset import canvas_of
        from mammo_seg.device import resolve_device
        from mammo_seg.engine.model import ModelSettings
        from mammo_seg.engine.train import TrainingSettings, train_fold
    except ImportError as error:
        print(
            f"error: training needs PyTorch and segmentation_models_pytorch ({error}); "
            "install an extra, see the README",
            file=sys.stderr,
        )
        return 1
    output = args.output_dir / fold_name(args.fold)
    try:
        settings = TrainingSettings(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            positive_fraction=args.positive_fraction,
            samples_per_epoch=args.samples_per_epoch,
            amp=args.amp,
            selection_metric=args.selection_metric,
            patience=args.patience or None,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        model_settings = ModelSettings(
            architecture=args.architecture,
            encoder=args.encoder,
            encoder_weights=None if args.encoder_weights == "none" else "imagenet",
        )
        # The images are read with the settings they were preprocessed with.
        record = read_record(args.preprocessed_dir / RECORD_FILE)
        with _training_log():
            result = train_fold(
                manifest.read_manifest(args.manifest),
                read_splits(args.splits),
                args.preprocessed_dir,
                args.fold,
                settings,
                model_settings,
                record.settings,
                device=resolve_device(args.device),
                output_dir=output,
                canvas=canvas_of(record),
                resume=args.resume,
                time_limit_minutes=args.max_minutes,
            )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    best = (
        f"best epoch {result.best_epoch} ({args.selection_metric} "
        f"{result.best_value:.4f}) in {output / 'best.pt'}"
        if result.best_epoch is not None
        else f"{args.selection_metric} was never defined"
    )
    epochs, reason = len(result.history), result.stop_reason
    print(f"Training ended after {epochs} epochs ({reason}): {best}")
    if reason == "time_limit":
        print("Run the same command with --resume to continue.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in STEPS:
        return STEPS[arguments[0]][1](arguments[1:])
    args = build_parser().parse_args(arguments)
    return _train(args)


if __name__ == "__main__":
    raise SystemExit(main())
