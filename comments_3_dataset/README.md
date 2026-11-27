# Comments dataset audit

**Archived rewrite condition — rewrite the SHP-preferred reply.** The corrected notebook uses the SHP-less-preferred reply as the source for generation. Results for the corrected condition require a new generation and judging run. [Archived texts](dataset3.csv) · [Corrected notebook](../code/comments_together.ipynb)

## Pair order

Stanford Human Preferences (SHP) assigns label 1 when comment A is preferred and label 0 when comment B is preferred. The earlier preprocessing reversed that mapping. In all 300 archived rows, `losing_debate_id` and `losing_text` match the SHP-preferred comment. The archived `paraphrase` and `improve` prompts used `losing_text`. The recovered `generate` prompt asked for a reply to the topic. [SHP labels](https://huggingface.co/datasets/stanfordnlp/SHP#data-fields) · [Pair ID manifest](shp_pair_ids.csv)

With `REPLAY_ARCHIVED_ORDER = False`, the notebook puts the SHP-less-preferred reply in `losing_text` and records the source comment ID in `losing_comment_id`. Setting `REPLAY_ARCHIVED_ORDER = True` switches the input back to the archived orientation.

## Text coverage

| Item checked | Count |
| --- | ---: |
| Archived Reddit reply pairs | 300 |
| Available human text slots across the pairs | 600 of 600 |
| Available AI text slots across the pairs | 6,299 of 6,300 |
| Referenced AI variants with text in each pairwise preference file | 6,299 of 6,299 |
| Referenced AI variants with text in each single-comment self-recognition file | 6,299 of 6,299 |

The one empty slot is an `improve` variant in `losing_improve_mistral`. All six pairwise preference files and all six single-comment self-recognition files reference the same 6,899 candidates, counting both human and AI text slots. [Candidate texts](dataset3.csv) · [Pairwise preference files](preference/preference_between_2_comments/) · [Single-comment self-recognition files](self_rec/self_recognition_with_one_comment/)

To find a judged text, join the result's `row_id` to `id` in `dataset3.csv` and use the source column named by the result. The archive suffixes `_qwen` and `_mistral` match `_qwen3` and `_mixtral_8x7b` in the results.

## Exported judge answers

The six pairwise preference files contain 378,365 populated vote cells out of 386,940. The six single-comment self-recognition files contain 41,394 populated `raw_answer` cells out of 41,394. These counts include nonempty error markers. Candidate availability counts whether the text the judge saw is present. Any Figure 2B percentage needs to be checked against the figure's input and denominator.

## Reproduce the audit

Run the comments notebook with `RUN_LIVE = False`. Its final audit cell matches the archived pairs against the original SHP labels and checks every candidate reference in the twelve files above. The verification requires no model calls. [Comments notebook](../code/comments_together.ipynb)
