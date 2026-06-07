import string
import evaluate
import jiwer
import pandas as pd
import argparse
import os
from datetime import datetime
import json

import re
from num2words import num2words

_NUM_RE = re.compile(r"\b\d+\b")


def digits_to_words(text: str, lang: str = "es") -> str:
    """Convert digit tokens to words in the specified language."""
    def repl(m):
        return num2words(int(m.group(0)), lang=lang)
    return _NUM_RE.sub(repl, text)


def normalize_text(text, lang: str = "es"):
    """Normalize text for WER calculation.

    Parameters
    ----------
    text : str
        The text to normalize.
    lang : str
        Language code for digit-to-word conversion ('es' for Spanish,
        'en' for English, etc.).
    """
    # Remove the "<unk>" tokens for WER calculation (unintelligible parts)
    text = str(text).lower().replace("<unk>", "")
    # digits to words (in the appropriate language)
    text = digits_to_words(text, lang=lang)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def compute_metrics(references, hypotheses):
    """Compute WER and error metrics."""
    wer_metric = evaluate.load("wer")
    standard_wer = wer_metric.compute(predictions=hypotheses, references=references)
    out = jiwer.process_words(references, hypotheses)
    return standard_wer, out


def compute_error_percentages(out):
    """Compute error type percentages from total errors."""
    total_errors = out.substitutions + out.deletions + out.insertions
    if total_errors == 0:
        return 0, 0, 0
    else:
        subs_pct = out.substitutions / total_errors * 100
        dels_pct = out.deletions / total_errors * 100
        ins_pct = out.insertions / total_errors * 100
    return subs_pct, dels_pct, ins_pct


if __name__ == "__main__":

    # Arguments and paths
    parser = argparse.ArgumentParser(description="Calculate WER and related metrics for ASR hypotheses.")
    parser.add_argument("--input_tsv", type=str, required=True,
                        help="Path to the input TSV file containing the hypotheses.")
    parser.add_argument("--reference_file", type=str, default=None,
                        help="Optional reference TSV with ground-truth transcriptions. "
                             "If omitted, references are read from the input TSV.")
    parser.add_argument("--hypothesis_column", type=str, default="Hypothesis",
                        help="Name of the hypothesis column in the input TSV.")
    parser.add_argument("--transcription_column", type=str, default="Transcription",
                        help="Name of the transcription/reference column in the reference TSV.")
    parser.add_argument("--sample_id_column", type=str, default="Sample_ID",
                        help="Name of the sample ID column used for merging (default: Sample_ID).")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Directory to save the WER report.")
    parser.add_argument("--no_merge", action="store_true",
                        help="Skip merging with a separate reference file. "
                             "The input TSV must already contain the reference/transcription column.")
    parser.add_argument("--language", type=str, default=None,
                        choices=["es", "en"],
                        help="Language for digit-to-word conversion ('es' for Spanish, "
                             "'en' for English). If not set, auto-detected from the "
                             "input filename: 'torgo' → 'en', otherwise → 'es'.")

    args = parser.parse_args()

    # Auto-detect language from filename if not explicitly set
    if args.language is None:
        if "torgo" in args.input_tsv.lower():
            lang = "en"
        else:
            lang = "es"
    else:
        lang = args.language
    print(f"Language for digit normalisation: {lang}")

    # Load the input TSV file with hypotheses
    input_path = args.input_tsv
    hypothesis_df = pd.read_csv(input_path, sep="\t")

    if args.no_merge:
        # Reference column is already in the input TSV — no merging needed
        df = hypothesis_df
        if args.transcription_column not in df.columns:
            raise ValueError(f"Transcription column '{args.transcription_column}' not found in input TSV. "
                             f"Available columns: {df.columns.tolist()}")
    else:
        # Check if the sample ID column exists in both dataframes
        if args.sample_id_column not in hypothesis_df.columns:
            # look for "sample_id"
            if "sample_id" in hypothesis_df.columns:
                hypothesis_df.rename(columns={"sample_id": args.sample_id_column}, inplace=True)
            else:
                raise ValueError(f"Sample ID column '{args.sample_id_column}' not found in hypothesis TSV. "
                                 f"Available columns: {hypothesis_df.columns.tolist()}")

        # Load the reference TSV file
        if not args.reference_file:
            raise ValueError("--reference_file is required unless --no_merge is set "
                             "(references read from the input TSV).")
        reference_df = pd.read_csv(args.reference_file, sep="\t")

        # Merge dataframes on the specified sample ID column
        df = reference_df.merge(hypothesis_df, on=args.sample_id_column, how="inner")

    # Normalize references and hypotheses
    norm = lambda t: normalize_text(t, lang=lang)
    references = df[args.transcription_column].apply(norm).tolist()
    hypotheses = df[args.hypothesis_column].apply(norm).tolist()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get model name from the input file name
    model_name = os.path.basename(input_path).replace("_hypothesis.tsv", "").replace("dss_", "")

    # Initialize results dictionary
    all_results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "overall": {},
        "by_disease": {},
        "by_intelligibility": {},
        "by_sex": {},
        "dataset_stats": {}
    }

    # ============ OVERALL METRICS ============
    standard_wer, out = compute_metrics(references, hypotheses)
    subs_pct, dels_pct, ins_pct = compute_error_percentages(out)

    all_results["overall"] = {
        "wer": float(standard_wer * 100),
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "hits": int(out.hits),
        "empty_hypotheses": sum(1 for hyp in hypotheses if hyp.strip() == ""),
        "error_distribution": {
            "substitutions_pct": float(subs_pct),
            "deletions_pct": float(dels_pct),
            "insertions_pct": float(ins_pct)
        }
    }

    print("=" * 60)
    print(f"Overall WER Results for {model_name}")
    print("=" * 60)
    print(f"WER: {all_results['overall']['wer']:.2f}%")
    print(f"Substitutions: {all_results['overall']['substitutions']}")
    print(f"Deletions: {all_results['overall']['deletions']}")
    print(f"Insertions: {all_results['overall']['insertions']}")
    print(f"Hits: {all_results['overall']['hits']}")
    print(f"Empty hypotheses: {all_results['overall']['empty_hypotheses']}")

    # ============ METRICS BY DISEASE ============

    # Check if 'Disease' column exists
    if "Disease" in df.columns:
        
        print("\n" + "=" * 60)
        print("WER by Disease")
        print("=" * 60)

        grouped = df.groupby("Disease")
        for disease, group in grouped:
            refs = group[args.transcription_column].apply(norm).tolist()
            hyps = group[args.hypothesis_column].apply(norm).tolist()
            wer, out = compute_metrics(refs, hyps)
            subs_pct, dels_pct, ins_pct = compute_error_percentages(out)
            
            all_results["by_disease"][disease] = {
                "n_samples": int(len(group)),
                "wer": float(wer * 100),
                "substitutions": int(out.substitutions),
                "deletions": int(out.deletions),
                "insertions": int(out.insertions),
                "hits": int(out.hits),
                "error_distribution": {
                    "substitutions_pct": float(subs_pct),
                    "deletions_pct": float(dels_pct),
                    "insertions_pct": float(ins_pct)
                }
            }
            print(f"{disease:15} | WER: {wer*100:6.2f}% | Samples: {len(group):3d}")

    # ============ METRICS BY INTELLIGIBILITY ============

    if "Intelligibility" in df.columns:
        print("\n" + "=" * 60)
        print("WER by Intelligibility")
        print("=" * 60)

        grouped = df.groupby("Intelligibility")
        for intel, group in grouped:
            refs = group[args.transcription_column].apply(norm).tolist()
            hyps = group[args.hypothesis_column].apply(norm).tolist()
            wer, out = compute_metrics(refs, hyps)
            subs_pct, dels_pct, ins_pct = compute_error_percentages(out)
            
            all_results["by_intelligibility"][intel] = {
                "n_samples": int(len(group)),
                "wer": float(wer * 100),
                "substitutions": int(out.substitutions),
                "deletions": int(out.deletions),
                "insertions": int(out.insertions),
                "hits": int(out.hits),
                "error_distribution": {
                    "substitutions_pct": float(subs_pct),
                    "deletions_pct": float(dels_pct),
                    "insertions_pct": float(ins_pct)
                }
            }
            print(f"{intel:15} | WER: {wer*100:6.2f}% | Samples: {len(group):3d}")

    if "Sex" in df.columns:
        # ============ METRICS BY SEX ============
        print("\n" + "=" * 60)
        print("WER by Sex")
        print("=" * 60)

        grouped = df.groupby("Sex")
        for sex, group in grouped:
            refs = group[args.transcription_column].apply(norm).tolist()
            hyps = group[args.hypothesis_column].apply(norm).tolist()
            wer, out = compute_metrics(refs, hyps)
            subs_pct, dels_pct, ins_pct = compute_error_percentages(out)
            
            sex_label = "Female" if sex == "Female" else "Male"
            all_results["by_sex"][sex_label] = {
                "n_samples": int(len(group)),
                "wer": float(wer * 100),
                "substitutions": int(out.substitutions),
                "deletions": int(out.deletions),
                "insertions": int(out.insertions),
                "hits": int(out.hits),
                "error_distribution": {
                    "substitutions_pct": float(subs_pct),
                    "deletions_pct": float(dels_pct),
                    "insertions_pct": float(ins_pct)
                }
            }
            print(f"{sex_label:15} | WER: {wer*100:6.2f}% | Samples: {len(group):3d}")

    # ============ METRICS BY UTTERANCE TYPE (for TORGO) ============
    # Check if this is TORGO dataset by looking for 'dataset_source' column and TORGO entries
    if "torgo" in args.input_tsv.lower():
        print("\n" + "=" * 60)
        print("WER by Utterance Type (TORGO)")
        print("=" * 60)

        # Classify utterances as single-word or multi-word based on transcription
        def classify_utterance(text):
            """Classify utterance as single-word or multi-word."""
            words = str(text).strip().split()
            return "single-word" if len(words) == 1 else "multi-word"
        
        df["utterance_type"] = df[args.transcription_column].apply(classify_utterance)
        
        all_results["by_utterance_type"] = {}
        
        grouped = df.groupby("utterance_type")
        for utt_type, group in grouped:
            refs = group[args.transcription_column].apply(norm).tolist()
            hyps = group[args.hypothesis_column].apply(norm).tolist()
            wer, out = compute_metrics(refs, hyps)
            subs_pct, dels_pct, ins_pct = compute_error_percentages(out)
            
            all_results["by_utterance_type"][utt_type] = {
                "n_samples": int(len(group)),
                "wer": float(wer * 100),
                "substitutions": int(out.substitutions),
                "deletions": int(out.deletions),
                "insertions": int(out.insertions),
                "hits": int(out.hits),
                "error_distribution": {
                    "substitutions_pct": float(subs_pct),
                    "deletions_pct": float(dels_pct),
                    "insertions_pct": float(ins_pct)
                }
            }
            print(f"{utt_type:15} | WER: {wer*100:6.2f}% | Samples: {len(group):4d}")

    # ============ DATASET STATISTICS ============
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    n_samples = len(df)
    if 'Sex' in df.columns:
        n_male = (df['Sex'] == 'Male').sum()
        n_female = (df['Sex'] == 'Female').sum()
    else:
        n_male = 0
        n_female = 0
    total_duration = df['Duration_sec'].sum() if 'Duration_sec' in df.columns else 0
    mean_duration = df['Duration_sec'].mean() if 'Duration_sec' in df.columns else 0

    all_results["dataset_stats"] = {
        "total_samples": int(n_samples),
        "male_samples": int(n_male),
        "female_samples": int(n_female),
        "male_percentage": float(n_male / n_samples * 100) if n_samples > 0 else 0,
        "female_percentage": float(n_female / n_samples * 100) if n_samples > 0 else 0,
        "total_duration_seconds": float(total_duration),
        "total_duration_hours": float(total_duration / 3600),
        "mean_duration_seconds": float(mean_duration)
    }

    print(f"Total samples: {n_samples}")
    print(f"Male: {n_male} ({n_male/n_samples*100:.1f}%) | Female: {n_female} ({n_female/n_samples*100:.1f}%)")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Mean audio length: {mean_duration:.2f} seconds")

    # ============ SAVE RESULTS ============
    # Save as JSON
    json_output_path = os.path.join(args.output_dir, f"wer_report_{model_name}.json")
    with open(json_output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n✓ JSON report saved to: {json_output_path}")

    """
    # Save as detailed CSV report
    csv_output_path = os.path.join(args.output_dir, f"wer_report_{model_name}.csv")
    
    # Create a summary dataframe
    summary_data = []
    
    # Overall metrics
    summary_data.append({
        "Category": "Overall",
        "Subcategory": "All Data",
        "Samples": n_samples,
        "WER (%)": all_results["overall"]["wer"],
        "Substitutions": all_results["overall"]["substitutions"],
        "Deletions": all_results["overall"]["deletions"],
        "Insertions": all_results["overall"]["insertions"],
        "Hits": all_results["overall"]["hits"]
    })
    
    # Disease metrics
    for disease, metrics in all_results["by_disease"].items():
        summary_data.append({
            "Category": "Disease",
            "Subcategory": disease,
            "Samples": metrics["n_samples"],
            "WER (%)": metrics["wer"],
            "Substitutions": metrics["substitutions"],
            "Deletions": metrics["deletions"],
            "Insertions": metrics["insertions"],
            "Hits": metrics["hits"]
        })
    
    # Intelligibility metrics
    for intel, metrics in all_results["by_intelligibility"].items():
        summary_data.append({
            "Category": "Intelligibility",
            "Subcategory": intel,
            "Samples": metrics["n_samples"],
            "WER (%)": metrics["wer"],
            "Substitutions": metrics["substitutions"],
            "Deletions": metrics["deletions"],
            "Insertions": metrics["insertions"],
            "Hits": metrics["hits"]
        })
    
    # Sex metrics
    for sex, metrics in all_results["by_sex"].items():
        summary_data.append({
            "Category": "Sex",
            "Subcategory": sex,
            "Samples": metrics["n_samples"],
            "WER (%)": metrics["wer"],
            "Substitutions": metrics["substitutions"],
            "Deletions": metrics["deletions"],
            "Insertions": metrics["insertions"],
            "Hits": metrics["hits"]
        })
    
    # Utterance type metrics (TORGO)
    if "by_utterance_type" in all_results:
        for utt_type, metrics in all_results["by_utterance_type"].items():
            summary_data.append({
                "Category": "Utterance Type",
                "Subcategory": utt_type,
                "Samples": metrics["n_samples"],
                "WER (%)": metrics["wer"],
                "Substitutions": metrics["substitutions"],
                "Deletions": metrics["deletions"],
                "Insertions": metrics["insertions"],
                "Hits": metrics["hits"]
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(csv_output_path, index=False)
    print(f"✓ CSV report saved to: {csv_output_path}")

    print("\n" + "=" * 60)
    """