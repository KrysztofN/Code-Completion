from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path
import json
import random

@dataclass
class TextFIMExample:
    prefix: str
    middle: str
    suffix: str
    original_text: str
    page_number: int
    split_index: int
    
    def to_training_format(self) -> Dict[str, str]:
        """Convert to training format with special FIM tokens"""
        return {
            "text": f"<fim_prefix>{self.prefix}<fim_suffix>{self.suffix}<fim_middle>{self.middle}",
            "original_text": self.original_text,
            "page_number": self.page_number,
            "split_index": self.split_index
        }

class TextFIMGenerator:
    def __init__(self, 
                 lines_per_page: int = 50,
                 splits_per_page: int = 3,
                 fim_probability: float = 0.8,
                 random_seed: int = 42):
        self.lines_per_page = lines_per_page
        self.splits_per_page = splits_per_page
        self.fim_probability = fim_probability
        self.examples = []
        random.seed(random_seed)
        
    def process_text_file(self, file_path: str) -> List[TextFIMExample]:
        """Process text file and generate FIM examples"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.rstrip() for line in f] 
            
          
            pages = [lines[i:i + self.lines_per_page] for i in range(0, len(lines), self.lines_per_page)]
            
            total_examples = 0
            for page_num, page_lines in enumerate(pages, 1):
                page_text = '\n'.join(page_lines)
                num_examples = self._process_page(page_text, page_num)
                total_examples += num_examples
                
            print(f"Processed {len(pages)} pages, generated {total_examples} examples")
            return self.examples
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def _split_text(self, text: str) -> Tuple[str, str, str]:
        """Split text into three parts randomly, preserving line integrity"""
        lines = text.split('\n')
        if len(lines) < 3:
            raise ValueError("Text too short: needs at least 3 lines")
            
        split_points = sorted(random.sample(range(1, len(lines)), 2))
        split1, split2 = split_points
        
        prefix = '\n'.join(lines[:split1])
        middle = '\n'.join(lines[split1:split2])
        suffix = '\n'.join(lines[split2:])
        
        return prefix, middle, suffix

    def _process_page(self, page_content: str, page_number: int) -> int:
        """Process a single page, generating multiple FIM examples"""
        examples_generated = 0
        
        for split_idx in range(self.splits_per_page):
            if random.random() < self.fim_probability:
                try:
                    prefix, middle, suffix = self._split_text(page_content)
                    
                    example = TextFIMExample(
                        prefix=prefix,
                        middle=middle,
                        suffix=suffix,
                        original_text=page_content,
                        page_number=page_number,
                        split_index=split_idx
                    )
                    self.examples.append(example)
                    examples_generated += 1
                except ValueError as e:
                    print(f"Error processing split {split_idx} on page {page_number}: {e}")
                    
        return examples_generated

    def save_dataset(self, output_file: str, format: str = 'jsonl'):
        """Save dataset to file"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
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
        
        unique_pages = len(set(ex.page_number for ex in self.examples))
        
        prefix_lengths = [len(ex.prefix.split('\n')) for ex in self.examples]
        middle_lengths = [len(ex.middle.split('\n')) for ex in self.examples]
        suffix_lengths = [len(ex.suffix.split('\n')) for ex in self.examples]
        
        return {
            "total_examples": total_examples,
            "unique_pages": unique_pages,
            "avg_examples_per_page": total_examples / unique_pages if unique_pages > 0 else 0,
            "length_stats": {
                "avg_prefix_lines": sum(prefix_lengths) / total_examples,
                "avg_middle_lines": sum(middle_lengths) / total_examples,
                "avg_suffix_lines": sum(suffix_lengths) / total_examples,
                "min_prefix_lines": min(prefix_lengths),
                "min_middle_lines": min(middle_lengths),
                "min_suffix_lines": min(suffix_lengths),
                "max_prefix_lines": max(prefix_lengths),
                "max_middle_lines": max(middle_lengths),
                "max_suffix_lines": max(suffix_lengths),
            }
        }

def main():
    generator = TextFIMGenerator(
        lines_per_page=80,    
        splits_per_page=1,      
        fim_probability=0.8,    
        random_seed=42
    )
    
    generator.process_text_file("assets/text_based/hamlet.txt")
    generator.save_dataset("datasets/text_fim_dataset.jsonl", format='jsonl')
    stats = generator.get_statistics()
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()