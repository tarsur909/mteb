
"""
Evaluation script for running MTEB_SWE benchmark on code embedding models.
This script manually wraps models to avoid metadata auto-detection issues in MTEB.
"""

import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import mteb
from mteb import get_tasks
from mteb.cache import ResultCache
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.models.model_meta import ModelMeta
import torch
import gc

# Configure logging to show only info and above
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class SafeModelWrapper(SentenceTransformerEncoderWrapper):
    """
    A wrapper that safely initializes ModelMeta to bypass MTEB's auto-detection bugs
    when loading models with incomplete model cards (missing tags).
    """
    def __init__(self, model: SentenceTransformer, prompts: dict = None):
        # Do not call super().__init__ as it triggers the buggy metadata loading
        self.model = model
        
        # Use provided prompts or fall back to model's internal prompts
        self.model_prompts = prompts or getattr(model, "prompts", {})
        if self.model_prompts and not getattr(model, "prompts", None):
             self.model.prompts = self.model_prompts

        # Manually create safe metadata
        model_name = getattr(model.model_card_data, "model_name", "custom_model")
        self.mteb_model_meta = ModelMeta(
            loader=None,
            name=model_name,
            revision=None,
            release_date=None,
            languages=None,
            framework=["Sentence Transformers", "PyTorch"],
            similarity_fn_name=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            training_datasets=None,
            use_instructions=bool(self.model_prompts),
        )

        if hasattr(self.model, "similarity") and callable(self.model.similarity):
            self.similarity = self.model.similarity

def evaluate_model(model_name: str, tasks, output_dir: Path, prompts: dict = None, trust_remote_code: bool = True):
    """
    Load and evaluate a single model.
    """
    logger.info(f"Loading {model_name}...")
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    
    # Wrap model safely
    wrapper = SafeModelWrapper(model, prompts=prompts)
    
    logger.info(f"Evaluating {model_name}...")
    results = mteb.evaluate(
        wrapper,
        tasks,
        cache=ResultCache(cache_path=output_dir / f"cache_{model_name.split('/')[-1]}"),
        overwrite_strategy="always",
        show_progress_bar=True,
        encode_kwargs={"batch_size": 1},
    )
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    return results

def run_evaluation():
    # Task selection
    # Using MTEB(SWE, v1) benchmark
    benchmark = mteb.get_benchmark("MTEB(SWE, v1)")
    tasks = benchmark.tasks
    logger.info(f"Selected tasks: {[t.metadata.name for t in tasks]}")
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Salesforce/SweRankEmbed-Small
    # This model uses "query" prompt automatically if present in config
    evaluate_model(
        "Salesforce/SweRankEmbed-Small",
        tasks,
        output_dir
    )
    
    # 2. nomic-ai/CodeRankEmbed
    # Requires explicit query prefix
    evaluate_model(
        "nomic-ai/CodeRankEmbed",
        tasks,
        output_dir,
        prompts={"query": "Represent this query for searching relevant code: "}
    )
    
    logger.info("Evaluation Completed Successfully.")

if __name__ == "__main__":
    run_evaluation()
