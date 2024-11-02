import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import random

@dataclass
class FIMExample:
    prefix: str
    middle: str
    suffix: str
    original_text: str
    file_path: str
    split_index: int
    
    def to_training_format(self) -> Dict[str, str]:
        """Convert to training format with special FIM tokens"""
        return {
            "text": f"<fim_prefix>{self.prefix}<fim_suffix>{self.suffix}<fim_middle>{self.middle}",
            "original_text": self.original_text,
            "file_path": self.file_path,
            "split_index": self.split_index
        }

class FIMDatasetGenerator:
    def __init__(self, 
                 splits_per_file: int = 5,
                 fim_probability: float = 0.8,
                 random_seed: int = 42):
        self.splits_per_file = splits_per_file
        self.fim_probability = fim_probability
        self.examples = []
        random.seed(random_seed)
        
    def process_directory(self, directory_path: str) -> List[FIMExample]:
        """Process all Python files in directory"""
        processed_files = 0
        total_examples = 0
        
        for path in Path(directory_path).rglob('*.py'):
            if not any(x.startswith('.') for x in path.parts):
                try:
                    num_examples = self.process_file(str(path))
                    if num_examples > 0:
                        processed_files += 1
                        total_examples += num_examples
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    
        print(f"Processed {processed_files} files, generated {total_examples} examples")
        return self.examples

    def _split_document(self, content: str) -> Tuple[str, str, str]:
        """
        Split document into three parts completely randomly at character level.
        No minimum size constraints - pure random splits.
        """
        content_length = len(content)
        if content_length < 2:  
            raise ValueError("Document too short: needs at least 2 characters")
            
        split_points = sorted(random.sample(range(content_length + 1), 2))
        split1, split2 = split_points
        
        prefix = content[:split1]
        middle = content[split1:split2]
        suffix = content[split2:]
        
        return prefix, middle, suffix

    def process_file(self, file_path: str) -> int:
        """
        Process a single Python file, generating multiple FIM examples
        Returns: Number of examples generated for this file
        """
        examples_generated = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for split_idx in range(self.splits_per_file):
                if random.random() < self.fim_probability:
                    prefix, middle, suffix = self._split_document(content)
                    
                    example = FIMExample(
                        prefix=prefix,
                        middle=middle,
                        suffix=suffix,
                        original_text=content,
                        file_path=file_path,
                        split_index=split_idx
                    )
                    self.examples.append(example)
                    examples_generated += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
        return examples_generated

    def save_dataset(self, output_file: str, format: str = 'jsonl'):
        """
        Save dataset to file
        Args:
            output_file: Path to output file
            format: 'jsonl'
        """
        dataset = [example.to_training_format() for example in self.examples]
        
        if format == 'jsonl':
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in dataset:
                    f.write(json.dumps(example) + '\n')
        else:  # json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2)

    def get_statistics(self) -> Dict:
        """Get detailed dataset statistics"""
        total_examples = len(self.examples)
        
        if total_examples == 0:
            return {"error": "No examples generated"}
        
        unique_files = len(set(ex.file_path for ex in self.examples))
        
        prefix_lengths = [len(ex.prefix) for ex in self.examples]
        middle_lengths = [len(ex.middle) for ex in self.examples]
        suffix_lengths = [len(ex.suffix) for ex in self.examples]
        
        examples_per_file = {}
        for ex in self.examples:
            examples_per_file[ex.file_path] = examples_per_file.get(ex.file_path, 0) + 1
        
        ratios = []
        for ex in self.examples:
            total_len = len(ex.original_text)
            ratios.append({
                'prefix_ratio': len(ex.prefix) / total_len * 100,
                'middle_ratio': len(ex.middle) / total_len * 100,
                'suffix_ratio': len(ex.suffix) / total_len * 100
            })
        
        avg_ratios = {
            'avg_prefix_ratio': sum(r['prefix_ratio'] for r in ratios) / len(ratios),
            'avg_middle_ratio': sum(r['middle_ratio'] for r in ratios) / len(ratios),
            'avg_suffix_ratio': sum(r['suffix_ratio'] for r in ratios) / len(ratios)
        }
        
        return {
            "total_examples": total_examples,
            "unique_files": unique_files,
            "avg_examples_per_file": total_examples / unique_files if unique_files > 0 else 0,
            "max_examples_per_file": max(examples_per_file.values()) if examples_per_file else 0,
            "min_examples_per_file": min(examples_per_file.values()) if examples_per_file else 0,
            "length_stats": {
                "avg_prefix_length": sum(prefix_lengths) / total_examples,
                "avg_middle_length": sum(middle_lengths) / total_examples,
                "avg_suffix_length": sum(suffix_lengths) / total_examples,
                "min_prefix_length": min(prefix_lengths),
                "min_middle_length": min(middle_lengths),
                "min_suffix_length": min(suffix_lengths),
                "max_prefix_length": max(prefix_lengths),
                "max_middle_length": max(middle_lengths),
                "max_suffix_length": max(suffix_lengths),
            },
            "ratio_stats": avg_ratios
        }

def main():
    generator = FIMDatasetGenerator(
        splits_per_file=5,  
        fim_probability=0.8,
        random_seed=42
    )
    
    examples = generator.process_directory("assets")
    
    generator.save_dataset("datasets/fim_dataset.jsonl", format='jsonl')
    stats = generator.get_statistics()
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))
    

if __name__ == "__main__":
    main()