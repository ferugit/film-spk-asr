
"""
Compare evaluation results between base and fine-tuned Voxtral models.
"""

import json
import pandas as pd
import argparse
from pathlib import Path


def load_results(base_path, output_dir="results/evaluation"):
    """Load evaluation results from JSON files."""
    base_summary = json.load(open(f"{output_dir}/voxtral_base_evaluation_summary.json"))
    
    finetuned_path = f"{output_dir}/voxtral_finetuned_evaluation_summary.json"
    if Path(finetuned_path).exists():
        finetuned_summary = json.load(open(finetuned_path))
    else:
        finetuned_summary = None
    
    return base_summary, finetuned_summary


def print_comparison(base_summary, finetuned_summary):
    """Print comparison table between base and fine-tuned models."""
    
    print("\n" + "=" * 80)
    print("VOXTRAL MODEL COMPARISON")
    print("=" * 80)
    
    # NeuroVoz comparison
    print("\n📊 NeuroVoz Test Set (Spanish):")
    print("-" * 80)
    base_nv_wer = base_summary["neurovoz_test"]["wer"]
    print(f"  Base Model WER:       {base_nv_wer:.2f}%")
    
    if finetuned_summary:
        ft_nv_wer = finetuned_summary["neurovoz_test"]["wer"]
        improvement_nv = base_nv_wer - ft_nv_wer
        improvement_pct_nv = (improvement_nv / base_nv_wer) * 100 if base_nv_wer > 0 else 0
        
        print(f"  Fine-tuned Model WER: {ft_nv_wer:.2f}%")
        print(f"  Improvement:          {improvement_nv:.2f}% ({improvement_pct_nv:+.1f}% relative)")
    
    # TORGO comparison
    print("\n📊 TORGO Test Set (English):")
    print("-" * 80)
    base_torgo_wer = base_summary["torgo_test"]["wer"]
    print(f"  Base Model WER:       {base_torgo_wer:.2f}%")
    
    if finetuned_summary:
        ft_torgo_wer = finetuned_summary["torgo_test"]["wer"]
        improvement_torgo = base_torgo_wer - ft_torgo_wer
        improvement_pct_torgo = (improvement_torgo / base_torgo_wer) * 100 if base_torgo_wer > 0 else 0
        
        print(f"  Fine-tuned Model WER: {ft_torgo_wer:.2f}%")
        print(f"  Improvement:          {improvement_torgo:.2f}% ({improvement_pct_torgo:+.1f}% relative)")
    
    # Overall comparison
    print("\n📊 Overall (Combined):")
    print("-" * 80)
    base_combined_wer = base_summary["combined_wer"]
    print(f"  Base Model WER:       {base_combined_wer:.2f}%")
    
    if finetuned_summary:
        ft_combined_wer = finetuned_summary["combined_wer"]
        improvement_combined = base_combined_wer - ft_combined_wer
        improvement_pct_combined = (improvement_combined / base_combined_wer) * 100 if base_combined_wer > 0 else 0
        
        print(f"  Fine-tuned Model WER: {ft_combined_wer:.2f}%")
        print(f"  Improvement:          {improvement_combined:.2f}% ({improvement_pct_combined:+.1f}% relative)")
    
    print("\n" + "=" * 80)
    
    # Summary table
    if finetuned_summary:
        print("\n📋 Summary Table:")
        print("-" * 80)
        
        data = {
            "Test Set": ["NeuroVoz (ES)", "TORGO (EN)", "Combined"],
            "Base WER (%)": [
                f"{base_nv_wer:.2f}",
                f"{base_torgo_wer:.2f}",
                f"{base_combined_wer:.2f}"
            ],
            "Fine-tuned WER (%)": [
                f"{ft_nv_wer:.2f}",
                f"{ft_torgo_wer:.2f}",
                f"{ft_combined_wer:.2f}"
            ],
            "Absolute Improvement": [
                f"{improvement_nv:.2f}%",
                f"{improvement_torgo:.2f}%",
                f"{improvement_combined:.2f}%"
            ],
            "Relative Improvement": [
                f"{improvement_pct_nv:+.1f}%",
                f"{improvement_pct_torgo:+.1f}%",
                f"{improvement_pct_combined:+.1f}%"
            ]
        }
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned Voxtral model results")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
        help="Output directory with evaluation results (default: results/evaluation)"
    )
    args = parser.parse_args()
    
    try:
        base_summary, finetuned_summary = load_results(args.output_dir, args.output_dir)
        
        if not finetuned_summary:
            print("\n⚠️  Warning: Fine-tuned model results not found.")
            print("   Only base model results are available.")
            print(f"\n   Base Model WER:")
            print(f"     - NeuroVoz: {base_summary['neurovoz_test']['wer']:.2f}%")
            print(f"     - TORGO:    {base_summary['torgo_test']['wer']:.2f}%")
            print(f"     - Combined: {base_summary['combined_wer']:.2f}%")
            print("\n   Run the fine-tuned model evaluation first:")
            print("   ./src/fine-tune/voxtral/evaluate_finetuned.sh")
            return
        
        print_comparison(base_summary, finetuned_summary)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find evaluation results.")
        print(f"   {e}")
        print(f"\n   Please run evaluation first:")
        print(f"   ./src/fine-tune/voxtral/evaluate_finetuned.sh")


if __name__ == "__main__":
    main()
