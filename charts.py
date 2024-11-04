import matplotlib.pyplot as plt
import numpy as np
import json

def load_predictions(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def extract_metrics(predictions):
    metrics = {
        'CHRF': np.mean([p['chrf_score'] for p in predictions]),
        'Jaccard': np.mean([p['jaccard_score'] for p in predictions]),
        'BLEU': np.mean([p['bleu_score'] for p in predictions]),
        'ROUGE-1': np.mean([p['rouge1'] for p in predictions]),
        'ROUGE-2': np.mean([p['rouge2'] for p in predictions]),
        'ROUGE-L': np.mean([p['rougeL'] for p in predictions])
    }
    return metrics

def plot_comparison(data1, data2, label1, label2, title):
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(data1))
    width = 0.35
    
    plt.bar(x - width/2, list(data1.values()), width, label=label1, color='skyblue')
    plt.bar(x + width/2, list(data2.values()), width, label=label2, color='lightcoral')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x, list(data1.keys()), rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def main():

    code_psm = load_predictions('datasets/code_fim_predictionsPSM.json')
    code_spm = load_predictions('datasets/code_fim_predictionsSPM.json')
    text_psm = load_predictions('datasets/text_fim_predictionsPSM.json')
    text_spm = load_predictions('datasets/text_fim_predictionsSPM.json')
    
    code_psm_metrics = extract_metrics(code_psm)
    code_spm_metrics = extract_metrics(code_spm)
    text_psm_metrics = extract_metrics(text_psm)
    text_spm_metrics = extract_metrics(text_spm)
    
    plot_comparison(
        code_psm_metrics, 
        code_spm_metrics,
        'PSM', 
        'SPM',
        'Code Completion: PSM vs SPM Comparison'
    )
    plt.savefig('charts/code_completion_comparison.png')
    
    plot_comparison(
        text_psm_metrics,
        text_spm_metrics,
        'PSM',
        'SPM',
        'Text Completion: PSM vs SPM Comparison'
    )
    plt.savefig('charts/text_completion_comparison.png')
    
    plot_comparison(
        code_spm_metrics,
        text_spm_metrics,
        'Code SPM',
        'Text SPM',
        'SPM Method: Code vs Text Completion Comparison'
    )
    plt.savefig('charts/spm_comparison.png')
    
    plt.close('all')

if __name__ == "__main__":
    main()