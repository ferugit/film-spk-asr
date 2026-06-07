"""
This script applies text-level heuristics to detect and remove hallucination
patterns commonly produced by ASR models:

1. Error Filtering:
   - Words longer than max_word_length characters are likely hallucinations
     from repeated character sequences (e.g., "nononononononono").
   - Detected words are checked for internal repetition; if a repeating unit
     is found the word is replaced with its base unit, otherwise discarded.

2. Word-level Deduplication:
   - Detects runs of the same word repeated consecutively more than
     max_word_repeats times and collapses them to a single occurrence.
   - Punctuation-normalised comparison (e.g., "no," == "no" == "No,").

3. Phrase-level Deduplication:
   - For texts with more than 3 words, detects consecutive repeated multi-word
     phrases (2-6 word n-grams) and reduces them to a single occurrence.
"""

import argparse
import os
import re
import logging
import unicodedata
from typing import Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ===================================================================
# 1. Error Filtering – abnormally long words with internal repetition
# ===================================================================

def _find_repeating_unit(word: str) -> str | None:
    """Return the shortest repeating unit inside *word*, or None."""
    w = word.lower()
    length = len(w)
    for unit_len in range(1, length // 2 + 1):
        unit = w[:unit_len]
        if unit * (length // unit_len) == w[:unit_len * (length // unit_len)]:
            # Check that the leftover (if any) is also a prefix of unit
            leftover = w[unit_len * (length // unit_len):]
            if unit.startswith(leftover):
                return unit
    return None


def filter_long_words(text: str, max_word_length: int = 15, debug: bool = False) -> str:
    """
    Replace or remove words that exceed *max_word_length* characters.

    If the word contains a repeating sub-string it is replaced by a single
    copy of that unit; otherwise the word is discarded entirely.
    """
    words = text.split()
    result = []
    for w in words:
        # Strip punctuation to measure "real" length
        clean = re.sub(r"[^\w]", "", w, flags=re.UNICODE)
        if len(clean) > max_word_length:
            unit = _find_repeating_unit(clean)
            if unit:
                if debug:
                    logger.debug(f"  [long-word] '{w}' -> repeating unit '{unit}'")
                # Preserve leading/trailing punctuation from original token
                leading = re.match(r"^(\W*)", w, flags=re.UNICODE).group(1)
                trailing = re.search(r"(\W*)$", w, flags=re.UNICODE).group(1)
                result.append(f"{leading}{unit}{trailing}")
            else:
                if debug:
                    logger.debug(f"  [long-word] '{w}' -> discarded (no repeating unit)")
        else:
            result.append(w)
    return " ".join(result)


# ===================================================================
# 2. Word-level Deduplication – consecutive repeated words
# ===================================================================

def _normalise_word(word: str) -> str:
    """Lower-case, strip punctuation, normalise unicode for comparison."""
    w = unicodedata.normalize("NFC", word.lower())
    w = re.sub(r"[^\w]", "", w, flags=re.UNICODE)
    return w


def deduplicate_words(text: str, max_repeats: int = 2, debug: bool = False) -> str:
    """
    Collapse runs of the same word repeated more than *max_repeats* times
    into *max_repeats* occurrences.

    Comparison is case- and punctuation-insensitive, but the original tokens
    (with their casing / punctuation) are preserved.
    """
    words = text.split()
    if len(words) <= 1:
        return text

    result = []
    run_count = 1  # how many times current normalised word has been seen

    for i, w in enumerate(words):
        norm = _normalise_word(w)
        if i == 0:
            result.append(w)
            continue

        prev_norm = _normalise_word(words[i - 1])
        if norm == prev_norm and norm != "":
            run_count += 1
            if run_count <= max_repeats:
                result.append(w)
            else:
                if debug and run_count == max_repeats + 1:
                    logger.debug(f"  [word-dedup] dropping extra '{w}' (>{max_repeats} repeats)")
        else:
            run_count = 1
            result.append(w)

    return " ".join(result)


# ===================================================================
# 3. Phrase-level Deduplication – consecutive repeated n-grams
# ===================================================================

def deduplicate_phrases(text: str, min_phrase_len: int = 2, max_phrase_len: int = 6,
                        debug: bool = False) -> str:
    """
    Detect consecutive repeated multi-word phrases and reduce them to one
    occurrence.  Operates greedily from the longest n-gram down.

    Only applied when the text has more than 3 words.
    """
    words = text.split()
    if len(words) <= 3:
        return text

    changed = True
    while changed:
        changed = False
        for n in range(max_phrase_len, min_phrase_len - 1, -1):
            i = 0
            new_words = []
            while i < len(words):
                # Can we find a phrase of length n starting at i that repeats?
                if i + 2 * n <= len(words):
                    phrase = [_normalise_word(w) for w in words[i : i + n]]
                    next_phrase = [_normalise_word(w) for w in words[i + n : i + 2 * n]]
                    if phrase == next_phrase and all(p != "" for p in phrase):
                        # Count how many consecutive repeats
                        repeats = 1
                        j = i + n
                        while j + n <= len(words):
                            candidate = [_normalise_word(w) for w in words[j : j + n]]
                            if candidate == phrase:
                                repeats += 1
                                j += n
                            else:
                                break
                        if debug:
                            phrase_str = " ".join(words[i : i + n])
                            logger.debug(
                                f"  [phrase-dedup] '{phrase_str}' repeated {repeats}x -> keeping 1"
                            )
                        # Keep only the first occurrence
                        new_words.extend(words[i : i + n])
                        i = j  # skip all repeats
                        changed = True
                        continue
                new_words.append(words[i])
                i += 1
            words = new_words

    return " ".join(words)


# ===================================================================
# Combined pipeline
# ===================================================================

def postprocess_text(
    text: str,
    max_word_length: int = 15,
    max_word_repeats: int = 2,
    min_phrase_len: int = 2,
    max_phrase_len: int = 6,
    debug: bool = False,
) -> Tuple[str, dict]:
    """
    Run the full post-processing pipeline on a single text string.

    Returns:
        (cleaned_text, stats_dict)
    """
    if not isinstance(text, str) or text.strip() == "":
        return text if isinstance(text, str) else "", {
            "long_words_fixed": 0, "words_deduped": 0,
            "phrases_deduped": 0, "total_words_removed": 0,
        }

    original_word_count = len(text.split())

    # Step 1 – long-word filtering
    t1 = filter_long_words(text, max_word_length=max_word_length, debug=debug)
    after_step1 = len(t1.split())

    # Step 2 – word-level deduplication
    t2 = deduplicate_words(t1, max_repeats=max_word_repeats, debug=debug)
    after_step2 = len(t2.split())

    # Step 3 – phrase-level deduplication
    t3 = deduplicate_phrases(t2, min_phrase_len=min_phrase_len,
                             max_phrase_len=max_phrase_len, debug=debug)
    after_step3 = len(t3.split())

    # Clean up extra whitespace
    cleaned = re.sub(r"\s+", " ", t3).strip()

    stats = {
        "long_words_fixed": original_word_count - after_step1,
        "words_deduped": after_step1 - after_step2,
        "phrases_deduped": after_step2 - after_step3,
        "total_words_removed": original_word_count - len(cleaned.split()),
    }
    return cleaned, stats


# ===================================================================
# TSV processing
# ===================================================================

def process_tsv(
    input_tsv: str,
    output_tsv: str,
    hypothesis_column: str = "hypothesis",
    max_word_length: int = 15,
    max_word_repeats: int = 2,
    min_phrase_len: int = 2,
    max_phrase_len: int = 6,
    debug: bool = False,
) -> None:
    """Read a TSV, post-process the hypothesis column, write results."""
    logger.info(f"Reading {input_tsv}")
    df = pd.read_csv(input_tsv, sep="\t")
    logger.info(f"  {len(df)} rows, columns: {list(df.columns)}")

    if hypothesis_column not in df.columns:
        raise ValueError(
            f"Column '{hypothesis_column}' not found. Available: {list(df.columns)}"
        )

    cleaned_texts = []
    all_stats = []
    samples_modified = 0

    for idx, row in df.iterrows():
        text = row[hypothesis_column]
        sample_id = row.get("sample_id", row.get("Sample_ID", idx))

        if debug:
            logger.debug(f"\n--- Sample: {sample_id} ---")
            logger.debug(f"  ORIGINAL: {text}")

        cleaned, stats = postprocess_text(
            text,
            max_word_length=max_word_length,
            max_word_repeats=max_word_repeats,
            min_phrase_len=min_phrase_len,
            max_phrase_len=max_phrase_len,
            debug=debug,
        )

        if debug and cleaned != text:
            logger.debug(f"  CLEANED:  {cleaned}")
            logger.debug(f"  STATS:    {stats}")

        if cleaned != text:
            samples_modified += 1

        cleaned_texts.append(cleaned)
        all_stats.append(stats)

    # Add new column
    output_col = f"postprocessed_{hypothesis_column}"
    df[output_col] = cleaned_texts

    # Add stats columns
    df["pp_long_words_fixed"] = [s["long_words_fixed"] for s in all_stats]
    df["pp_words_deduped"] = [s["words_deduped"] for s in all_stats]
    df["pp_phrases_deduped"] = [s["phrases_deduped"] for s in all_stats]
    df["pp_total_words_removed"] = [s["total_words_removed"] for s in all_stats]

    # Save
    os.makedirs(os.path.dirname(output_tsv) or ".", exist_ok=True)
    df.to_csv(output_tsv, sep="\t", index=False)
    logger.info(f"Saved to {output_tsv}")

    # Summary
    total_removed = sum(s["total_words_removed"] for s in all_stats)
    total_long = sum(s["long_words_fixed"] for s in all_stats)
    total_word_dedup = sum(s["words_deduped"] for s in all_stats)
    total_phrase_dedup = sum(s["phrases_deduped"] for s in all_stats)

    print(f"\n{'='*60}")
    print(f"POST-PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples:           {len(df)}")
    print(f"Samples modified:        {samples_modified}")
    print(f"Total words removed:     {total_removed}")
    print(f"  - Long words fixed:    {total_long}")
    print(f"  - Word-level dedup:    {total_word_dedup}")
    print(f"  - Phrase-level dedup:  {total_phrase_dedup}")
    print(f"{'='*60}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Text-based post-processing to reduce ASR hallucinations"
    )
    parser.add_argument(
        "--input_tsv", type=str, required=True,
        help="Input TSV with hypothesis column",
    )
    parser.add_argument(
        "--output_tsv", type=str, required=True,
        help="Output TSV path",
    )
    parser.add_argument(
        "--hypothesis_column", type=str, default="hypothesis",
        help="Name of the hypothesis column (default: hypothesis)",
    )
    parser.add_argument(
        "--max_word_length", type=int, default=15,
        help="Words longer than this are checked for internal repetition (default: 15)",
    )
    parser.add_argument(
        "--max_word_repeats", type=int, default=2,
        help="Max allowed consecutive repetitions of the same word (default: 2)",
    )
    parser.add_argument(
        "--min_phrase_len", type=int, default=2,
        help="Minimum phrase length (in words) for phrase dedup (default: 2)",
    )
    parser.add_argument(
        "--max_phrase_len", type=int, default=6,
        help="Maximum phrase length (in words) for phrase dedup (default: 6)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output (shows per-sample details)",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    process_tsv(
        input_tsv=args.input_tsv,
        output_tsv=args.output_tsv,
        hypothesis_column=args.hypothesis_column,
        max_word_length=args.max_word_length,
        max_word_repeats=args.max_word_repeats,
        min_phrase_len=args.min_phrase_len,
        max_phrase_len=args.max_phrase_len,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
