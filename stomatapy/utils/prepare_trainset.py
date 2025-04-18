#!/usr/bin/env python
"""Script to prepare training data for stomata models"""

import os
import argparse
from stomatapy.utils.data4training import Data4Training


def main():
    """Main function to prepare training data"""
    parser = argparse.ArgumentParser(description='Prepare training dataset for stomata models')
    # Required parameters
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Root directory containing the dataset')
    # Optional parameters with defaults
    parser.add_argument('--r_train', type=float, default=0.8,
                        help='Ratio of training data (default: 0.8)')
    parser.add_argument('--r_test', type=float, default=0,
                        help='Ratio of test data (default: 0)')
    parser.add_argument('--aim', type=str, default='object detection',
                        choices=['semantic segmentation', 'object detection', 'instance segmentation'],
                        help='Training aim (default: object detection)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Custom output directory path (default: None)')
    parser.add_argument('--new_width', type=int, default=4352,
                        help='New width after resizing (default: 4352)')
    parser.add_argument('--new_height', type=int, default=1844,
                        help='New height after resizing (default: 1844)')
    parser.add_argument('--remove_subgroups', action='store_true', default=True,
                        help='Remove subgroups (default: True)')
    parser.add_argument('--if_resize_isat', action='store_true', default=True,
                        help='Resize ISAT files (default: True)')
    parser.add_argument('--use_sahi', action='store_true', default=True,
                        help='Use SAHI to slice images (default: True)')
    parser.add_argument('--slice_width', type=int, default=1280,
                        help='Slice width for SAHI (default: 1280)')
    parser.add_argument('--slice_height', type=int, default=1024,
                        help='Slice height for SAHI (default: 1024)')
    parser.add_argument('--sahi_overlap_ratio', type=float, default=0.2,
                        help='Overlap ratio for SAHI slicing (default: 0.2)')

    args = parser.parse_args()

    # Determine base directory and destination
    base_dir = os.path.dirname(os.getcwd())
    if args.output_dir:
        destination = args.output_dir
    else:
        destination = os.path.join(base_dir, "StomataPy", f"{os.path.basename(args.dataset_root)}_train")

    print(f'Destination directory: {destination}')

    # Walk through the dataset root
    for root, dirs, _ in os.walk(args.dataset_root):
        for dir_name in dirs:
            # Skip hidden directories
            if dir_name.startswith('.'):
                continue

            subfolder_dir = os.path.join(root, dir_name)
            if os.path.isdir(subfolder_dir):
                print(f'Processing {subfolder_dir}')
                output_rename = os.path.join(destination, os.path.basename(subfolder_dir))

                # Initialize Data4Training with custom parameters
                data_trainer = Data4Training(
                    input_dir=subfolder_dir,
                    aim=args.aim,
                    r_train=args.r_train,
                    r_test=args.r_test,
                    new_width=args.new_width,
                    new_height=args.new_height,
                    use_sahi=args.use_sahi,
                    slice_width=args.slice_width,
                    slice_height=args.slice_height,
                    sahi_overlap_ratio=args.sahi_overlap_ratio
                )

                # Process the data
                data_trainer.data4training(
                    remove_subgroups=args.remove_subgroups,
                    if_resize_isat=args.if_resize_isat,
                    output_rename=output_rename
                )


if __name__ == "__main__":
    main()
