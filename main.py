"""Command-line entry point for Devanagari akshara segmentation."""

import argparse
from src.segmentation import SegmentConfig, segment_batch


def main():
    parser = argparse.ArgumentParser(
        description="Segment Devanagari/Sanskrit word images into aksharas."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing image/text pairs, e.g. line_0.jpg and line_0.txt",
    )
    parser.add_argument(
        "output_dir",
        help="Directory where character crops, debug images and manifests are saved.",
    )
    parser.add_argument(
        "--pattern",
        default="*.jpg",
        help="Image glob pattern (default: *.jpg). Use *.png for PNG input.",
    )
    args = parser.parse_args()

    results = segment_batch(
        args.input_dir,
        args.output_dir,
        pattern=args.pattern,
        cfg=SegmentConfig(),
    )
    print(f"Processed {len(results)} image(s).")


if __name__ == "__main__":
    main()
