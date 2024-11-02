from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import json
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from sacrebleu.metrics import CHRF
import evaluate
from rouge_score import rouge_scorer

@dataclass
class FIMPrediction:
    original_text: str
    prefix: str
    predicted_middle: str
    ground_truth_middle: str
    suffix: str
    exact_match: bool
    chrf_score: float
    jaccard_score: float
    bleu_score: float
    rouge: Dict[str, float]
    prediction_confidence: float
    file_path: str
    split_index: int

class StarCoderFIMPredictor:
    def __init__(self, model_name="bigcode/tiny_starcoder_py"):
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )
        self.chrf_metric = CHRF()
        self.bleu_metric = evaluate.load('bleu')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def calculate_chrf(self, prediction: str, reference: str) -> float:
        """Calculate CHRF score between prediction and reference"""
        score = self.chrf_metric.corpus_score([prediction], [[reference]])
        return score.score / 100  

    def calculate_bleu(self, prediction: str, reference: str) -> float:
        """Calculate BLEU score between prediction and reference"""
        pred_tokens = prediction.split()
        ref_tokens = [reference.split()]  
        
        result = self.bleu_metric.compute(predictions=[prediction], references=[[reference]])
        return result['bleu'] if result['bleu'] is not None else 0.0

    def token_overlap(self, prediction: str, reference: str) -> float:
        """Jaccard similarity score"""
        pred_tokens = set(prediction.split())
        ref_tokens = set(reference.split())

        if not pred_tokens and not ref_tokens:
            return 1.0
    
        intersection = pred_tokens.intersection(ref_tokens)
        union = pred_tokens.union(ref_tokens)
        return len(intersection)/len(union)

    def rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """ROUGE scores (1, 2, and L)"""
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def predict_middle(self, prefix: str, suffix: str, max_new_tokens: int = 128) -> tuple:
        """Generate the middle part given prefix and suffix"""
        prompt = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
        
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  
            return_attention_mask=True
        )
        
        inputs = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id, 
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                generation_config=generation_config
            )
        
        generated_tokens = outputs.sequences[0][len(inputs[0]):]
        completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        scores = torch.stack(outputs.scores, dim=1)
        probs = torch.softmax(scores[0], dim=-1)
        confidence = float(torch.mean(torch.max(probs, dim=-1).values))
        
        return completion.strip(), confidence

    def process_dataset(self, dataset_path: str, output_path: str):
        """Process entire FIM dataset and save results"""
        predictions = []
        
        with open(dataset_path, 'r') as f:
            examples = [json.loads(line) for line in f]
        
        print(f"Processing {len(examples)} examples...")
        
        for example in tqdm(examples):
            text = example['text']
            
            prefix = text.split('<fim_suffix>')[0].replace('<fim_prefix>', '')
            suffix = text.split('<fim_suffix>')[1].split('<fim_middle>')[0]
            ground_truth = text.split('<fim_middle>')[1]
            
            predicted, confidence = self.predict_middle(prefix, suffix)
            chrf_score = self.calculate_chrf(predicted, ground_truth)
            jaccard_score = self.token_overlap(predicted, ground_truth)
            bleu_score = self.calculate_bleu(predicted, ground_truth)
            rouge_scores = self.rouge_scores(predicted, ground_truth)

            prediction = FIMPrediction(
                original_text=example['original_text'],
                prefix=prefix,
                predicted_middle=predicted,
                ground_truth_middle=ground_truth,
                suffix=suffix,
                exact_match=predicted == ground_truth,
                chrf_score=chrf_score,
                jaccard_score=jaccard_score,
                bleu_score=bleu_score,
                rouge=rouge_scores,
                prediction_confidence=confidence,
                file_path=example['file_path'],
                split_index=example['split_index']
            )
            
            predictions.append(prediction)
        
        self._save_results(predictions, output_path)
        self._print_statistics(predictions)
    
    def _save_results(self, predictions: List[FIMPrediction], output_path: str):
        """Save predictions to file"""
        results = []
        for pred in predictions:
            results.append({
                'file_path': pred.file_path,
                'split_index': pred.split_index,
                'prefix': pred.prefix,
                'predicted_middle': pred.predicted_middle,
                'ground_truth_middle': pred.ground_truth_middle,
                'suffix': pred.suffix,
                'exact_match': pred.exact_match,
                'chrf_score': pred.chrf_score,
                'jaccard_score': pred.jaccard_score,
                'bleu_score': pred.bleu_score,
                'rouge1': pred.rouge['rouge1'],
                'rouge2': pred.rouge['rouge2'],
                'rougeL': pred.rouge['rougeL'],
                'confidence': pred.prediction_confidence,
                'original_text': pred.original_text
            })
            
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _print_statistics(self, predictions: List[FIMPrediction]):
        """Print summary statistics"""
        total = len(predictions)
        exact_matches = sum(1 for p in predictions if p.exact_match)
        avg_confidence = np.mean([p.prediction_confidence for p in predictions])
        avg_chrf = np.mean([p.chrf_score for p in predictions])
        avg_jaccard = np.mean([p.jaccard_score for p in predictions])
        avg_bleu = np.mean([p.bleu_score for p in predictions])
        avg_rouge1 = np.mean([p.rouge['rouge1'] for p in predictions])
        avg_rouge2 = np.mean([p.rouge['rouge2'] for p in predictions])
        avg_rougeL = np.mean([p.rouge['rougeL'] for p in predictions])
        
        print("\nResults Summary:")
        print(f"Total predictions: {total}")
        print(f"Exact matches: {exact_matches} ({exact_matches/total*100:.2f}%)")
        print(f"Average CHRF score: {avg_chrf:.3f}")
        print(f"Average Jaccard score: {avg_jaccard:.3f}")
        print(f"Average BLEU score: {avg_bleu:.3f}")
        print(f"Average Rouge1 score: {avg_rouge1:.3f}")
        print(f"Average Rouge2 score: {avg_rouge2:.3f}")
        print(f"Average RougeL score: {avg_rougeL:.3f}")
        print(f"Average confidence: {avg_confidence:.3f}")
        
        print("\nResults by file:")
        files_stats = {}
        for pred in predictions:
            if pred.file_path not in files_stats:
                files_stats[pred.file_path] = {
                    'total': 0, 
                    'correct': 0,
                    'chrf_scores': [],
                    'jaccard_scores': [],
                    'bleu_scores': []
                }
            stats = files_stats[pred.file_path]
            stats['total'] += 1
            stats['chrf_scores'].append(pred.chrf_score)
            stats['jaccard_scores'].append(pred.jaccard_score)
            stats['bleu_scores'].append(pred.bleu_score)
            if pred.exact_match:
                stats['correct'] += 1

        for file_path, stats in files_stats.items():
            print(f"\n{file_path}:")
            print(f"  Accuracy: {stats['correct']/stats['total']*100:.2f}% ({stats['correct']}/{stats['total']})")
            print(f"  Average CHRF: {np.mean(stats['chrf_scores']):.3f}")
            print(f"  Average Jaccard: {np.mean(stats['jaccard_scores']):.3f}")
            print(f"  Average BLEU: {np.mean(stats['bleu_scores']):.3f}")

def main():
    predictor = StarCoderFIMPredictor()
    
    predictor.process_dataset(
        dataset_path="datasets/hamlet_fim_dataset.jsonl",
        output_path="datasets/hamlet_fim_predictions.json"
    )

if __name__ == "__main__":
    main()