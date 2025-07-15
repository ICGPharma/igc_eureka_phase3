import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def check_pickle_file(path):
    try:
        with open(path, 'rb') as f:
            pickle.load(f)
        return None  # No error
    except Exception:
        return path  # Return problematic path

def check_paths_in_parallel(paths):
    max_workers = os.cpu_count() or 4
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_pickle_file, path): path for path in paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking files"):
            result = future.result()
            if result is not None:
                print('Error with', result)
                errors.append(result)
    return errors

def main():
    parser = argparse.ArgumentParser(description="Check pickle integrity for exported cache.")
    parser.add_argument(
        "--dir", 
        type=str,
        required=True,
        help="Directory containing pickled files."
    )
    parser.add_argument(
        "--train", 
        action="store_true",
        help="Whether to check a train subfolder."
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Whether to check a test subfolder."
    )

    args = parser.parse_args()

    if not (args.train or args.test):
        parser.error('At least one of --train or --test is required.')

    # Check train paths
    if args.train:
        with open(os.path.join(args.dir, 'all_paths.pkl'), 'rb') as f:
            train_paths = pickle.load(f)
        check_paths_in_parallel(train_paths)

    # Check test paths
    if args.test:
        with open(os.path.join(args.dir, 'all_paths.pkl'), 'rb') as f:
            test_paths = pickle.load(f)
        check_paths_in_parallel(test_paths)

if __name__ == "__main__":
    main()