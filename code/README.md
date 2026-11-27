# Experiment code

The three notebooks use the same Together AI request and evaluation code. Each notebook contains its own data preparation, prompts, experiments and exports. [Repository dataset guide](../README.md#dataset-guide)

| File | Purpose |
| --- | --- |
| [arguments_together.ipynb](arguments_together.ipynb) | Prepare Debate.org pairs and evaluate debate arguments |
| [abstracts_together.ipynb](abstracts_together.ipynb) | Prepare MDIC pairs and evaluate human summaries of scientific abstracts |
| [comments_together.ipynb](comments_together.ipynb) | Reconstruct the SHP pairs and evaluate replies to Reddit posts |
| [deploy/qwen3-8b/pyproject.toml](deploy/qwen3-8b/pyproject.toml) | Serve the exact Qwen3 8B weights on Together AI |
| [legacy/](legacy/) | Scripts for the earlier comments and Gemini experiments |

## Run a notebook

Use Python 3.10 or later. Clone the repository so the notebooks can find the original CSVs and the SHP pair manifest.

```bash
git clone --branch together-notebooks https://github.com/aimyr/Giuseppe_Aimer_research.git
cd Giuseppe_Aimer_research
python -m pip install jupyterlab
jupyter lab
```

Open the notebook for your dataset from `code/` and run the cells in order. The first cell sets the run options.

| Option | Meaning |
| --- | --- |
| `RUN_LIVE` | `False` loads the data and estimates requests. `True` sends requests to the configured Together endpoints. The first comments preview may fetch the public SHP records. |
| `ROW_LIMIT` | `1` selects the first pair for a short run. `None` selects the entire prepared dataset. |
| `MODEL_ALIASES` | `("qwen3",)` selects Qwen3 8B. Add `"mixtral_8x7b"` or `"llama_3_1_8b"` after configuring its endpoint. Every selected model generates variants and evaluates candidates. |
| `WORKERS` | Maximum number of API requests in parallel. The default is four. |

For a live run, set `TOGETHER_API_KEY` in the environment that launches Jupyter or enter it in the notebook's hidden prompt. Configure the endpoints before setting `RUN_LIVE = True`.

## Model setup

### Qwen3 8B

The default `qwen3` alias requests `Qwen/Qwen3-8B` from the Together deployment named `qwen3-8b`. The supplied configuration uses one H100, a pinned vLLM image and the same Hugging Face revision for the weights and tokenizer. [Pinned Qwen checkpoint](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218)

To create this deployment, your Together project needs Dedicated Container Inference access. Install the Together CLI with `uv tool install "together[cli]"`. With your project selected and `TOGETHER_API_KEY` set, run the following from the cloned repository. The GPU is billed while the replica runs. [Together vLLM deployment guide](https://docs.together.ai/docs/dedicated-containers-vllm)

```bash
cd code/deploy/qwen3-8b
tg beta jig deploy --detach
tg beta jig status
tg beta jig endpoint
```

Once the deployment is ready, the returned URL ends in `/v1/deployment-request/qwen3-8b`. The notebooks append `/v1/chat/completions`. If you changed the name in `pyproject.toml`, set `TOGETHER_QWEN3_DEPLOYMENT` to that name before starting Jupyter. If your project already has a managed endpoint serving the exact Qwen3 8B weights, set `TOGETHER_QWEN3_ENDPOINT` to its `project/endpoint` identifier. That setting takes precedence over the container name. [Managed Together endpoints](https://docs.together.ai/docs/dedicated-endpoints/quickstart)

### Additional models

| Alias | Model | Environment variable for the managed endpoint |
| --- | --- | --- |
| `mixtral_8x7b` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | `TOGETHER_MIXTRAL_ENDPOINT` |
| `llama_3_1_8b` | Llama 3.1 8B Instruct | `TOGETHER_LLAMA_ENDPOINT` |

Set each variable to the `project/endpoint` string shown in the Together console, then select the corresponding alias in `MODEL_ALIASES`. [Model availability and migration history](https://docs.together.ai/docs/deprecations)

The notebook uses one chat request format for all models. Each request sends a user message and `temperature=0`. The service and checkpoint context determine the maximum length because the request omits `max_tokens`. Qwen3 8B receives `chat_template_kwargs={"enable_thinking": False}` for direct answers. [Chat API parameters](https://docs.together.ai/docs/inference/chat/parameters) · [Qwen3 8B chat template](https://huggingface.co/Qwen/Qwen3-8B#enable_thinkingfalse)

## Dataset preparation

### 1. Debate arguments

The default input is [arguments_original.csv](../arguments_original.csv) with 106 selected pairs. The notebook uses the debate proposition as context and carries the source debate IDs, sides and vote totals into `texts.csv`.

- Set `DDO_SELECTION_CSV` to reuse the earlier `debate_themes_bestwin_worstlose_FULL.csv` export.
- Set `DDO_CROWD_CSV` and `DDO_VOTES_JSON` together to reconstruct that selection from the crowdsourcing export and filtered votes. The code joins debate rounds and tallies votes for the more convincing argument. Each proposition contributes the most voted text on the winning side and the least voted text on the other side. A tie selects the most voted Pro text and least voted Con text.

Generation asks the model to preserve the argument's stance, claims, evidence and certainty. Researcher-guided scoring considers relevance, reasoning, evidence, coherence and persuasive clarity. [Debate.org corpus](https://aclanthology.org/P19-1057/) · [DDO archive](https://doi.org/10.6084/m9.figshare.28327019)

### 2. Scientific abstract summaries

The default input is [abstracts_original.csv](../abstracts_original.csv) with 327 pairs of first-generation summaries. `level = 1` means each participant read the original abstract. The notebook uses DOI and topic as context and carries the candidate node IDs and human annotations into `texts.csv`.

Set `MDIC_PAIRWISE_CSV` to the earlier `pairwise_ready.csv` export to rebuild the study selection. The notebook filters for level 1, joins topics by the DOI in each node ID, and takes the first 327 rows in the export order. You can set `MDIC_CASCADES_CSV` to a local `control_cascades.csv`, or let the notebook fetch its pinned copy.

Generation preserves the population, interventions or exposures, comparisons, outcomes, numbers and conclusions present in the source. Researcher-guided scoring considers clarity, coherence and precision when reporting the findings. [MDIC source data](https://github.com/epfl-dlab/mdic/tree/f0c76aad5517a94c871b6dfd47601bac0b94fd78/data)

### 3. Reddit comments

The notebook matches the 300 archived row UUIDs to the same `r/AskAcademia` training pairs by Reddit post ID and both comment IDs. It reads the pinned SHP JSONL until it has found all matches and caches them under `raw_data/`. Set `SHP_INPUT` to a local copy of the SHP JSONL or the original `3rd_dataset.csv` to read the inputs offline. [Pair manifest](../comments_3_dataset/shp_pair_ids.csv) · [SHP data and field definitions](https://huggingface.co/datasets/stanfordnlp/SHP)

The default SHP ordering follows its published labels. `1` prefers comment A and `0` prefers comment B. Under `REPLAY_ARCHIVED_ORDER = False`, generation uses the SHP-less-preferred reply. Set `REPLAY_ARCHIVED_ORDER = True` to use the SHP-preferred reply, as the archived paraphrase and improve tasks did. The full Reddit post is the context. The export carries the post ID, comment IDs, Reddit scores, SHP label and selected ordering. The notebook's final cell verifies archive coverage and reply order with `RUN_LIVE = False`. [Comments dataset audit](../comments_3_dataset/README.md)

Generation preserves the author's viewpoint, advice and intended tone. Researcher-guided scoring weighs helpfulness, clarity, relevance and appropriateness of tone equally.

## Shared experiment stages

### 1. Initialize and validate

The notebook locates the repository root, resolves the model aliases and validates the selected pairs. Each pair has a unique `row_id`, a `context` and both human texts in `winning_text` and `losing_text`. The preview displays the source row count, the selected model and the estimated request count for each experiment.

### 2. Generate three versions of the less-preferred text

| Mode | Task |
| --- | --- |
| `paraphrase` | Change wording lightly while keeping the original structure and approximate length |
| `improve` | Improve clarity and flow while preserving meaning |
| `generate` | Rebuild the text with a different structure and phrasing while preserving meaning |

An identical `context` and `losing_text` share one generation request for each model and mode. The resulting `generation_id` links the generated text to every input pair that uses it. For example, Qwen's light rewrite appears in `losing_paraphrase_qwen3`.

### 3. Select the texts

Single-text tasks present both human texts and each available generated version. Pairwise tasks compare the two human texts, then compare each human text with each generated version. Every judge sees the same candidates and both presentation orders. The prompt includes the dataset context and candidate texts, while source and author labels are recorded in the results.

### 4. Measure self-recognition

- **Single text.** The judge answers `yes` or `no` when asked whether it wrote the text. `expected` is `yes` when the generator alias matches the judge alias.
- **Pair.** The judge selects text `1` or `2` in normal order (`N1`) and again in swapped order (`S1`). The notebook maps each answer back to its source column. `expected_source` identifies the judge's own version when exactly one candidate was generated by the judge.

### 5. Score individual texts

The judge scores each candidate from 1 to 5 using its own judgment, then scores it again with researcher instructions specific to the dataset. The `mode` column identifies the prompts as `own_judgment` and `researcher_instructions`.

### 6. Generate and apply scoring rubrics

Each model writes a JSON rubric with one description for each grade from 1 to 5. The code validates all five descriptions. Every judge then scores each candidate with every valid rubric, including the rubric it wrote. `model` identifies the judge and `rubric_author` identifies the model that wrote the rubric.

### 7. Measure pairwise preference

For abstract summaries, the judge is asked “Which abstract paraphrase is better written?”

Each judge sees the same pair in five rounds ordered `N1`, `S1`, `N2`, `S2` and `N3`. `N` uses the normal order and `S` uses the reversed order. Each round records the raw reply, the selected source and the request status.

- `majority_source` is filled when a source receives at least three votes. `majority_count_of_5` and `valid_votes` show the strongest vote count and number of valid choices.
- `normal_consistent` records whether all three valid normal rounds chose the same source. `order_influenced_decision` compares the sets of source choices between the two orders when every round is valid.

## Output files

Live runs write to `runs/<dataset>/<run_id>/`. `run_id` is derived from the selected inputs, temperature and model routing. Running the same settings again reuses successful replies from the SQLite checkpoint. Request failures and length truncations can be retried. Completed CSVs are written atomically.

| File | Contents |
| --- | --- |
| `run.json` | Dataset source, input hash and count, temperature and API model routing |
| `responses.sqlite3` | Successful raw replies keyed by the request and experiment stage |
| `generation.csv` | One row per generated variant with `generation_id`, `rewrite_mode`, `source_column` and the revised `text` |
| `texts.csv` | One row per study pair with human texts, source metadata, `generation_id` and generated text columns |
| `self_recognition_single.csv` | Per-text `answer` and `expected` by judge |
| `self_recognition_pair.csv` | Per-pair replies and chosen sources in both orders, with the expected source and order-effect field |
| `scoring.csv` | Per-text `score` for each researcher-instructed and own-judgment pass |
| `rubrics.csv` | One generated rubric per model with `score_1` through `score_5` |
| `scoring_with_rubrics.csv` | Per-text score for each judge and rubric-author combination |
| `preference.csv` | Five replies and choices per judge and pair, with majority and consistency fields |

In the results, `model` is the selected alias and `api_model` is the identifier sent in the API request. Join evaluations to `texts.csv` on `row_id`, using `source_column` or `A_source` and `B_source` to find the exact candidate. Blank answers and status fields such as `invalid_answer` and `truncated` keep malformed or partial replies visible without counting them as valid choices.

## Earlier experiment scripts

### Historical comments scripts

The scripts in `legacy/comments/` use the original `dataset3.csv` at the repository root by default. The CLI tools also accept `--csv`. They load provider credentials from the environment or the repository's `.env` when `python-dotenv` is available. Their saved model IDs describe the earlier experiment setup. [Comments scripts](legacy/comments/)

| Script | Task |
| --- | --- |
| `preference.py` | Collect direct five-round pairwise preferences |
| `preference_batch.py` | Submit, poll and fetch pairwise preference batches |
| `selfrec_batch.py` | Run single-text and pairwise self-recognition |
| `score_batch.py` | Score comments by own judgment or researcher instructions |
| `rubric_gen.py` | Generate rubrics for the historical scoring workflow |
| `score_rubric_batch.py` | Apply the generated rubrics across judges |
| `batch_lib.py` | Format, submit and retrieve provider batches |
| `judging_lib.py` | Resolve historical model aliases and send individual provider requests |

### Historical Gemini scripts

The sequential scripts in `legacy/gemini/seq/` use the adjacent `level1_merged.csv` and `rubric/` files. Run them from that directory with `GEMINI_API_KEY` in the environment and the Gemini Python library imported by the chosen script. [Sequential scripts and inputs](legacy/gemini/seq/)

| Script | Task |
| --- | --- |
| `self_recognition_one_seq.py` | Ask about authorship one summary at a time |
| `self_recognition_pair_seq.py` | Compare two summaries for authorship |
| `score_one_seq.py` | Score a single summary by own judgment |
| `score_one_seq_aimer_instruct.py` | Score a summary with researcher instructions |
| `score_one_seq_rubric.py` | Score summaries against the saved rubrics |
| `score_pair_seq.py` | Judge preference between summaries |
| `additional_preference/pairwise_gemini.py` | Submit Gemini pairwise preference batches for a supplied `--csv` file |
