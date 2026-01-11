"""
Script to download and prepare the Brown Corpus for training
"""
import nltk
import os
import argparse
from tqdm import tqdm

def setup_brown_data(data_dir: str):
    """Download and format Brown Corpus"""
    print("Downloading Brown Corpus...")
    try:
        nltk.download('brown', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return

    from nltk.corpus import brown
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Get all file ids
    file_ids = brown.fileids()
    print(f"Found {len(file_ids)} documents in Brown Corpus")
    
    print(f"Processing and saving to {data_dir}...")
    for file_id in tqdm(file_ids):
        # Get raw text
        text = brown.raw(file_id)
        
        # Save to file
        output_path = os.path.join(data_dir, f"{file_id}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/brown', help='Output directory')
    args = parser.parse_args()
    
    setup_brown_data(args.data_dir)
