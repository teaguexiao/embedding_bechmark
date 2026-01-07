#!/usr/bin/env python3
"""
Multilingual STS Benchmark: Compare Nova, Cohere, and OpenAI embedding models
Using official STS22 cross-lingual dataset from HuggingFace
"""

import boto3
import json
import os
import math
from typing import List, Callable, Dict
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# Load environment variables
load_dotenv()


# ============================================================================
# Embedding Functions
# ============================================================================

def embed_nova(text: str, dimension: int = 1024) -> List[float]:
    """Generate embedding using Nova model."""
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    # Truncate text if too long (Nova has input limits)
    text = text[:8000] if len(text) > 8000 else text
    request_body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": dimension,
            "text": {"truncationMode": "END", "value": text}
        }
    }
    response = client.invoke_model(
        modelId="amazon.nova-2-multimodal-embeddings-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(request_body)
    )
    result = json.loads(response['body'].read())
    return result['embeddings'][0]['embedding']


def embed_cohere(text: str) -> List[float]:
    """Generate embedding using Cohere Multilingual v3."""
    client = boto3.client('bedrock-runtime', region_name='us-west-2')
    # Truncate text if too long
    text = text[:8000] if len(text) > 8000 else text
    request_body = {
        "texts": [text],
        "input_type": "search_document",
        "truncate": "END"
    }
    response = client.invoke_model(
        modelId="cohere.embed-multilingual-v3",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(request_body)
    )
    result = json.loads(response['body'].read())
    return result['embeddings'][0]


def embed_openai_small(text: str) -> List[float]:
    """Generate embedding using OpenAI text-embedding-3-small."""
    client = OpenAI(api_key=os.getenv("OPENAI_API"))
    # Truncate text if too long
    text = text[:8000] if len(text) > 8000 else text
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def embed_openai_large(text: str) -> List[float]:
    """Generate embedding using OpenAI text-embedding-3-large."""
    client = OpenAI(api_key=os.getenv("OPENAI_API"))
    # Truncate text if too long
    text = text[:8000] if len(text) > 8000 else text
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def evaluate_model_on_lang_pair(
    embed_func: Callable,
    dataset,
    model_name: str,
    lang_pair: str
) -> dict:
    """
    Evaluate a model on cross-lingual STS dataset.
    Returns Spearman correlation, Pearson correlation.
    """
    human_scores = []
    predicted_scores = []

    samples = list(dataset)

    for item in tqdm(samples, desc=f"  {model_name} ({lang_pair})", ncols=80):
        sent1 = item['sentence1']
        sent2 = item['sentence2']
        human_score = item['score']

        try:
            emb1 = embed_func(sent1)
            emb2 = embed_func(sent2)
            cos_sim = cosine_similarity(emb1, emb2)

            human_scores.append(human_score)
            predicted_scores.append(cos_sim)
        except Exception as e:
            print(f"\n  Warning: Failed on sample - {str(e)[:50]}")
            continue

    if len(human_scores) < 2:
        return {"error": "Not enough valid samples"}

    # Calculate correlations
    spearman_corr, _ = spearmanr(human_scores, predicted_scores)
    pearson_corr, _ = pearsonr(human_scores, predicted_scores)

    return {
        "spearman": spearman_corr,
        "pearson": pearson_corr,
        "num_samples": len(human_scores)
    }


def main():
    print("=" * 70)
    print("Multilingual STS Benchmark: Cross-lingual Evaluation")
    print("Dataset: MTEB STS22 Cross-lingual STS")
    print("=" * 70)

    # Cross-lingual pairs to test
    LANG_PAIRS = ['zh-en', 'de-en', 'es-en', 'de-fr']
    LANG_PAIR_NAMES = {
        'zh-en': 'Chinese-English',
        'de-en': 'German-English',
        'es-en': 'Spanish-English',
        'de-fr': 'German-French'
    }

    # Define models to test
    models = {
        "Nova": lambda x: embed_nova(x, 1024),
        "Cohere": embed_cohere,
        "OpenAI-Small": embed_openai_small,
        "OpenAI-Large": embed_openai_large,
    }

    # Load datasets for each language pair
    print("\nLoading cross-lingual datasets from HuggingFace...")
    datasets = {}
    for lang_pair in LANG_PAIRS:
        try:
            datasets[lang_pair] = load_dataset(
                'mteb/sts22-crosslingual-sts',
                lang_pair,
                split='test'
            )
            print(f"  {LANG_PAIR_NAMES[lang_pair]}: {len(datasets[lang_pair])} samples")
        except Exception as e:
            print(f"  {LANG_PAIR_NAMES[lang_pair]}: Failed to load - {e}")

    # Results storage: {model_name: {lang_pair: result}}
    all_results: Dict[str, Dict[str, dict]] = {m: {} for m in models.keys()}

    # Evaluate each model on each language pair
    for lang_pair in LANG_PAIRS:
        if lang_pair not in datasets:
            continue

        print(f"\n{'=' * 70}")
        print(f"Testing: {LANG_PAIR_NAMES[lang_pair]} ({lang_pair})")
        print(f"{'=' * 70}")

        for model_name, embed_func in models.items():
            print(f"\n{'─' * 50}")
            print(f"Model: {model_name}")
            print('─' * 50)
            try:
                result = evaluate_model_on_lang_pair(
                    embed_func,
                    datasets[lang_pair],
                    model_name,
                    lang_pair
                )
                all_results[model_name][lang_pair] = result
                if "error" not in result:
                    print(f"  Spearman: {result['spearman']:.4f}")
                    print(f"  Pearson:  {result['pearson']:.4f}")
                    print(f"  Samples:  {result['num_samples']}")
                else:
                    print(f"  Error: {result['error']}")
            except Exception as e:
                print(f"  Failed: {e}")
                all_results[model_name][lang_pair] = {"error": str(e)}

    # Print summary by language pair
    print("\n" + "=" * 70)
    print("Results Summary - Cross-lingual STS Benchmark")
    print("=" * 70)

    for lang_pair in LANG_PAIRS:
        if lang_pair not in datasets:
            continue
        print(f"\n{LANG_PAIR_NAMES[lang_pair]} ({lang_pair}):")
        print(f"{'Model':<15} {'Spearman':<12} {'Pearson':<12} {'Samples':<10}")
        print("-" * 50)

        for model_name in models.keys():
            result = all_results[model_name].get(lang_pair, {})
            if "error" not in result and result:
                print(f"{model_name:<15} {result['spearman']:.4f}       {result['pearson']:.4f}       {result['num_samples']}")
            else:
                print(f"{model_name:<15} Failed       Failed       -")

    # Calculate and print overall averages
    print("\n" + "=" * 70)
    print("Overall Average Scores (across all language pairs)")
    print("=" * 70)
    print(f"\n{'Model':<15} {'Avg Spearman':<15} {'Avg Pearson':<15} {'Lang Pairs':<10}")
    print("-" * 55)

    model_averages = {}
    for model_name in models.keys():
        spearman_scores = []
        pearson_scores = []
        for lang_pair in LANG_PAIRS:
            result = all_results[model_name].get(lang_pair, {})
            if "error" not in result and result:
                spearman_scores.append(result['spearman'])
                pearson_scores.append(result['pearson'])

        if spearman_scores:
            avg_spearman = sum(spearman_scores) / len(spearman_scores)
            avg_pearson = sum(pearson_scores) / len(pearson_scores)
            model_averages[model_name] = avg_spearman
            print(f"{model_name:<15} {avg_spearman:.4f}          {avg_pearson:.4f}          {len(spearman_scores)}")
        else:
            print(f"{model_name:<15} N/A            N/A            0")

    # Determine winner
    if model_averages:
        winner = max(model_averages, key=model_averages.get)
        print(f"\n{'─' * 55}")
        print(f"Winner: {winner} (Avg Spearman = {model_averages[winner]:.4f})")
        print("=" * 70)


if __name__ == "__main__":
    main()
