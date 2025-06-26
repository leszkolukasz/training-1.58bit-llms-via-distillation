import json
import random
from pathlib import Path

def select_k_random_samples(input_file: str, output_file: str, k: int, seed: int=42):
    
    random.seed(seed)
    
    buffer = []
    
    with open(input_file, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i < k:
                buffer.append(line)
            else:
                j = random.randint(0, i)
                if j < k:
                    buffer[j] = line
                    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f_out:
        for sample in buffer:
            f_out.write(sample)
            
if __name__ == "__main__":
    
    INPUT_FILE = "./data/amber_small/train_000.jsonl"
    OUTPUT_FILE = "./data/amber_small_100000/train_000_100000.jsonl."
    
    select_k_random_samples(INPUT_FILE, OUTPUT_FILE, 1e5)