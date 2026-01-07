#!/usr/bin/env python3
"""
MTEB Official Dataset Benchmark: Compare Nova, Cohere, and OpenAI embedding models
Using official STS22 dataset from HuggingFace
"""

import boto3
import json
import os
import math
from typing import List, Callable
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


def evaluate_model(
    embed_func: Callable,
    dataset,
    model_name: str,
    max_samples: int = None
) -> dict:
    """
    Evaluate a model on STS dataset.
    Returns Spearman correlation, Pearson correlation.
    """
    human_scores = []
    predicted_scores = []

    samples = list(dataset)
    if max_samples:
        samples = samples[:max_samples]

    print(f"  Evaluating {len(samples)} sentence pairs...")

    for item in tqdm(samples, desc=f"  {model_name}", ncols=80):
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
    print("MTEB Official Dataset Benchmark: STS22 (English)")
    print("=" * 70)

    # Load official MTEB STS22 dataset
    print("\nLoading MTEB STS22 dataset from HuggingFace...")
    dataset = load_dataset('mteb/sts22-crosslingual-sts', 'en', split='test')
    print(f"Dataset loaded: {len(dataset)} samples")

    # Show sample
    print(f"\nSample data:")
    print(f"  Sentence 1: {dataset[0]['sentence1'][:80]}...")
    print(f"  Sentence 2: {dataset[0]['sentence2'][:80]}...")
    print(f"  Human Score: {dataset[0]['score']}")

    # Define models to test
    models = {
        "Nova": lambda x: embed_nova(x, 1024),
        "Cohere": embed_cohere,
        "OpenAI-Small": embed_openai_small,
        "OpenAI-Large": embed_openai_large,
    }

    results = {}

    # Ask user for sample limit (full dataset = 197 samples, ~800 API calls total)
    print(f"\n" + "─" * 70)
    print(f"Full dataset has {len(dataset)} samples.")
    print(f"Testing all 4 models will require ~{len(dataset) * 4 * 2} API calls.")
    print("─" * 70)

    for model_name, embed_func in models.items():
        print(f"\n{'─' * 50}")
        print(f"Evaluating: {model_name}")
        print('─' * 50)
        try:
            result = evaluate_model(embed_func, dataset, model_name)
            results[model_name] = result
            if "error" not in result:
                print(f"\n  ✓ Spearman: {result['spearman']:.4f}")
                print(f"  ✓ Pearson:  {result['pearson']:.4f}")
                print(f"  ✓ Samples:  {result['num_samples']}")
            else:
                print(f"  ✗ Error: {result['error']}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[model_name] = {"error": str(e)}

    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary - MTEB STS22 Official Dataset (English)")
    print("=" * 70)
    print(f"\n{'Model':<15} {'Spearman':<12} {'Pearson':<12} {'Samples':<10}")
    print("-" * 50)

    valid_results = []
    for model_name, result in results.items():
        if "error" not in result:
            print(f"{model_name:<15} {result['spearman']:.4f}       {result['pearson']:.4f}       {result['num_samples']}")
            valid_results.append((model_name, result['spearman']))
        else:
            print(f"{model_name:<15} Failed       Failed       -")

    # Determine winner
    if valid_results:
        winner = max(valid_results, key=lambda x: x[1])
        print(f"\n{'─' * 50}")
        print(f"🏆 Winner: {winner[0]} (Spearman = {winner[1]:.4f})")
        print("=" * 70)


if __name__ == "__main__":
    main()
