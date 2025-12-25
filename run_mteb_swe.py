import mteb
from mteb import MTEB
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.benchmarks.benchmarks.benchmarks import MTEB_SWE

def run_benchmark():
    # Model 1: Salesforce/SweRankEmbed-Small
    # We rely on the model's built-in prompts configuration if available.
    # The user specifies prompt_name="query" for queries.
    # If the model has prompts defined in its config, MTEB will automatically use them 
    # when prompt_type="query" (which is default for retrieval queries in MTEB).
    print("Loading Salesforce/SweRankEmbed-Small...")
    sfr_model = SentenceTransformerEncoderWrapper(
        "Salesforce/SweRankEmbed-Small",
        trust_remote_code=True,
    )

    # Model 2: nomic-ai/CodeRankEmbed
    # The user specifies a custom prefix for queries.
    # We define this in model_prompts so MTEB uses it for queries.
    print("Loading nomic-ai/CodeRankEmbed...")
    nomic_prompts = {
        "query": "Represent this query for searching relevant code: "
    }
    nomic_model = SentenceTransformerEncoderWrapper(
        "nomic-ai/CodeRankEmbed",
        trust_remote_code=True,
        model_prompts=nomic_prompts
    )

    # Prepare MTEB tasks
    tasks = MTEB_SWE.tasks
    print(f"Benchmarking on tasks: {[t.metadata.name for t in tasks]}")

    # Run benchmark for Salesforce model
    print("Running benchmark for Salesforce/SweRankEmbed-Small...")
    evaluation_sfr = MTEB(tasks=tasks)
    evaluation_sfr.run(sfr_model, output_folder="results/salesforce")

    # Run benchmark for Nomic model
    print("Running benchmark for nomic-ai/CodeRankEmbed...")
    evaluation_nomic = MTEB(tasks=tasks)
    evaluation_nomic.run(nomic_model, output_folder="results/nomic")

if __name__ == "__main__":
    run_benchmark()

