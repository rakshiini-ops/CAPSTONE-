# Instamart Customer Intelligence & Support Copilot

## Overview

This project cleans a messy (synthetic) Instamart dataset, segments customers, predicts which customers will stop ordering, triages support tickets with a mock LLM, and answers policy questions / handles refund requests with a small RAG + agent. All data is fictional and all money is in INR. Nothing needs internet or an API key: the "LLM" is the supplied `mock_llm.py`.

Flow of the project:

```
generate_data.py -> data/raw, data/policies
Part A  (src/part_a.py, validate_a.py, sql/queries.sql) -> data/clean, data/instamart.db
Part B  (src/part_b.py) -> outputs/customer_segments.csv, lapse_model.joblib, retention_list.csv
Part C  (src/part_c.py) -> outputs/triage_results.csv, triage_budget_demo.csv, triage_retries.log
Part D  (src/part_d.py) -> outputs/agent_runs.jsonl, refund_ledger.jsonl, state/*.json
```

Only the Python standard library, pandas, numpy, scikit-learn and joblib are used. `generate_data.py` and `mock_llm.py` are exactly as supplied.

## Reproduce

Tested with Python 3.12, pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, joblib 1.5.3.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_data.py            # last line: generated: customers=1200 products=40 orders_raw=13084 tickets=164
python src/run_all.py              # Part A, validation, Part B, C, D (about 15 seconds)
python src/validate_a.py           # prints ALL CHECKPOINTS PASSED
```

Every random step uses seed 42, so repeated runs give the same outputs (only the ledger timestamps and the elapsed time stored in `state/` change). Each part can also be run alone (`python src/part_b.py` etc.) but Part C needs Part B's output and Part D needs Part A's database.

## Part A

**Raw audit** (counted before cleaning): 13,084 order rows, 119 exact duplicates, 656 pending rows, 12,309 distinct order IDs, 120 non-numeric amounts, 106 negative delivery times, 2,541 blank delivery times, 61 blank ages, 6 ages outside 10-100, 24 city spellings, 10 gender spellings, 30 blank genders.

**Customers:** city is stripped and title-cased; gender is mapped to male / female / unknown (blank = unknown); ages outside 10-100 are set to missing and, together with the blank ages, filled with the median age of the customer's city using `groupby('city').transform('median')`. Result: 1,200 customers, 6 cities (Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184), female 610 / male 560 / unknown 30.

**Orders** (in this order): convert `order_ts` to datetime, drop exact duplicate rows, keep the latest record for each `order_id` (this removes the earlier "pending" copies), convert `order_amount` to numeric and refill the bad ones with `price x quantity x (1 - discount/100)`, set negative delivery times to missing, then fill missing delivery times of *completed* orders with the median of the product category. Cancelled orders were never delivered, so their delivery time stays missing.

Result: 12,309 orders = 11,110 completed + 1,199 cancelled, no pending, completed revenue INR 2,998,668.80.

**SQLite:** `data/instamart.db` has `customers`, `products` and `orders` with primary/foreign keys and column types (TEXT, INTEGER, REAL). `sql/queries.sql` has the seven queries:

| Query | Idea | Result |
|---|---|---|
| Q1 | WHERE + GROUP BY + HAVING | 30 customers |
| Q2 | 3-table JOIN, revenue per city and category | 30 rows, total 2,998,668.80, top = Bengaluru / Electronics 233,909.45 |
| Q3 | scalar subquery (price above the average price) | 11 products |
| Q4 | LEFT JOIN, customers with no completed order | 25 customers |
| Q5 | CTE, customers spending above the average | 550 customers (average spend 2,552.06) |
| Q6 | RANK / DENSE_RANK / ROW_NUMBER per city | top spenders: Bengaluru C0082, Chennai C0930, Delhi C0291, Hyderabad C0743, Mumbai C0693, Pune C0426 |
| Q7 | daily revenue with a cumulative SUM window | 183 days, final total 2,998,668.80 |

In Q6, 8 rows have a different RANK and ROW_NUMBER because of ties (tied customers share a rank).

`src/validate_a.py` checks the raw audit numbers, the cleaned data, the table types, all the SQL results above, and recomputes Q1, Q2, Q4, Q5 and Q7 in pandas to compare with SQL. It ends with `ALL CHECKPOINTS PASSED`.

**Pandas analysis:** Bengaluru completed revenue is 523,643.40 from 1,894 orders (mean 276.47); Mumbai is 479,058.50 from 1,613 orders (mean 297.00). Order amount has mean 269.91, median 135, skewness 4.61 (a few big electronics orders pull the mean up). The IQR upper fence is 645.30 and 897 orders are above it. Correlations are weak: amount vs discount -0.095, amount vs quantity +0.087.

## Part B

**RFM:** computed from completed orders with snapshot date 2025-10-01. 1,175 customers have at least one completed order. Recency median is 19 days, frequency mean 9.46, monetary mean 2,552.06.

**K-means:** the RFM columns are scaled with StandardScaler and K = 2 to 8 are compared.

| K | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| Inertia | 1820.8 | 1268.2 | 743.1 | 617.4 | 503.2 | 408.5 | 366.5 |
| Silhouette | 0.465 | 0.445 | **0.526** | 0.480 | 0.482 | 0.416 | 0.412 |

Inertia always falls when K grows, so it can't pick K by itself. K = 4 has the best silhouette, so I used 4. The segments:

| Segment | Customers | Avg recency (days) | Avg frequency | Avg monetary |
|---|---:|---:|---:|---:|
| Premium | 138 | 24.1 | 6.8 | 7,129.70 |
| Loyal | 419 | 8.8 | 19.4 | 3,410.02 |
| Regular | 436 | 28.9 | 3.8 | 1,057.62 |
| At Risk | 182 | 109.0 | 2.2 | 685.99 |

Cluster numbers are random, so the names are given from these averages (Premium = highest spend, At Risk = longest since last order, Loyal = highest frequency of the rest). PCA on the scaled RFM gives 63.4% + 21.8% = 85.2% variance in two components.

**Lapse model:** the cohort is the 1,156 customers with a completed order before 2025-09-01. A customer is *lapsed* if they have no completed order in September 2025: 443 lapsed (38.3%), 713 not lapsed. Stratified 80/20 split gives 924 train and 232 test rows.

**How leakage is avoided:**
- Features only use orders dated before 1 September (recency is measured to 1 September, not to the end of September). The label is the only thing that looks at September.
- The segment labels are not used as features, because they are built from the whole period including September.
- The scaler and one-hot encoder (`drop='first'`, i.e. k-1 columns) are inside a `Pipeline`, so they are fitted only on training data (and only on the training part inside each cross-validation fold).
- Class balancing is done on the training set only; the test set is never touched until the final scores.
- The final model is chosen with cross-validation F1, not test F1.
- One small leak remains: the Part A medians (city age, category delivery time) were computed on all rows, as the assignment's cleaning steps say.

**Models** (GridSearchCV, 5-fold, scoring = F1 of the lapsed class):

| Model | Best parameters | CV F1 | Test precision | Test recall | Test F1 |
|---|---|---:|---:|---:|---:|
| Baseline (always "not lapsed") | - | - | 0 | 0 | 0 |
| Logistic Regression (C in 0.01...100) | C = 100 | 0.651 | 0.644 | 0.652 | 0.648 |
| Random Forest (depth 3/5/8, min leaf 5/10/20) | depth 8, leaf 10 | 0.681 | 0.659 | 0.652 | 0.655 |

Random Forest is saved because it has the best CV F1 (it is only slightly better than Logistic Regression on the small test set). `lapse_model.joblib` is reloaded after saving and gives identical predictions.

**About C = 100 vs the reference C = 1 (F1 about 0.62):** I checked this. The data, split and seed are all fixed, so it is not randomness. The CV F1 for C = 1, 10, 100 is 0.645, 0.649, 0.651, which is almost flat, so the "best" C is decided by tiny differences. When I tried small changes (for example a 75/25 split, or dropping a few features) the best C and the test F1 moved around, and a 75/25 split gave C = 1 with test F1 0.624. So the reference value probably comes from a slightly different setup. I did not force C = 1; with the 80/20 split, seed 42 and these 11 features, C = 100 is what the code gives.

**Class balancing:** the lapsed rows in the training set are copied (random sampling with replacement) until both classes have 570 rows (1,140 total). Logistic Regression refit on this data: recall goes from 0.652 to 0.854 and F1 from 0.648 to 0.697, but precision drops from 0.644 to 0.589. So it finds more lapsers (76 instead of 58 out of 89) but also flags more people wrongly (53 instead of 32).

**Feature importance:** top Random Forest features are frequency (0.280), monetary (0.271), recency (0.115), average delivery time (0.061) and cancellation rate (0.052). In Logistic Regression, frequency has the biggest negative coefficient (-1.70): customers who ordered more often are less likely to lapse. Favourite category Electronics (-1.14) and Grocery (-0.45) are also lower risk than the dropped category. Frequency and monetary are correlated, so their individual coefficients should not be over-interpreted.

**Retention list:** `outputs/retention_list.csv` has the 10 Premium customers in the test set with the highest predicted lapse probability (1 of the 10 actually lapsed).

**Limitation:** the Premium flag comes from RFM over the whole period, which includes September (the period we are predicting). So the retention list is only an illustration, not a clean evaluation, even though the model features themselves are leak-free.

## Part C

Tickets are handled in file order. An empty ticket is marked `skipped_empty` and a ticket estimated above 400 tokens is `flagged_long` (4 of each; the long ones are about 734 tokens). Neither is sent to the model.

**PII:** before sending, the text is redacted in this order: email -> `[EMAIL]`, phone -> `[PHONE]`, order ID -> `[ORDER_ID]`. Total replaced: 55 emails, 91 phones, 132 order IDs (278). An `assert` checks that none of the patterns still match, and the user message sent to the model is only the redacted text.

**Prompts:** both system prompts start with `CLASSIFY` and ask for JSON with one of billing / delivery / technical / fraud. The few-shot prompt adds an `EXAMPLES:` section.

**Output checking:** valid JSON with an allowed category = `classified_json`. If the reply is not JSON, a regex looks for exactly one category word (`classified_keyword`). Anything else (including JSON with a category like "shipping") is `flagged_invalid_output` and goes to human review.

**Retries:** the mock fails every 9th call once. On a transient error the code waits and retries, at most 3 retries. Wait = 0.05 x 2^(retry-1) seconds x a random jitter between 0.9 and 1.15. Each retry is written to `outputs/triage_retries.log`. Each full run had 19 failures and every one worked on the first retry. `mock_llm.reset()` is called at the start of each run.

| Run | Classified | Accuracy | Total tokens | Tokens per ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 85.7% | 8,776 | 59.7 |
| Few-shot | 147 | 100% | 14,548 | 99.0 |

Zero-shot mistakes: fraud -> billing 11 and technical -> billing 10. Few-shot is about 66% more tokens but got everything right. Statuses in both runs: 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`, 4 empty, 4 long.

**Token budget:** before each call the code checks whether the tokens used so far have reached the budget; if yes the ticket becomes `budget_exhausted` (human review). The normal runs use 100,000 (never reached). The demo with a 5,000 budget classified 84 tickets and left 67 as `budget_exhausted`; it ended at 5,015 tokens because the budget is checked before a call, so the last call can go slightly over.

**Routing** (few-shot run): invalid / empty / long / budget-exhausted -> `human_review` NORMAL (17); fraud -> `security_team` HIGH (34); Premium customer -> `premium_desk` HIGH (11); everything else -> `support_queue` NORMAL (102). Fraud is checked before Premium, and a customer with no segment (no completed orders) counts as not Premium.

## Part D

**RAG:** the five policy files are cut into chunks of 40 words where each chunk starts 30 words after the previous one (10 words overlap); chunking stops when a chunk reaches the end of the file. This gives 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2), and each chunk remembers its file name. TF-IDF turns chunks into vectors that give more weight to words that are rare across chunks; cosine similarity between the question vector and the chunk vectors gives the top 3 chunks. For "refund policy late delivery" all three come from `refund_policy.txt`. The answer step sends a `POLICY ASSISTANT` system message and a user message in the form `CONTEXT:` ... `QUESTION:` ....

| | Chunks | Right document in top 3 | Expected fact in answer | Unsupported question -> "could not find" |
|---|---:|---:|---:|---:|
| 10-word overlap | 15 | 12/12 | 12/12 | 2/2 |
| No overlap | 13 | 12/12 | 10/12 | 2/2 |

Without overlap the right document is still found, but a chunk boundary can cut the answer sentence in half. For example, the refund-time sentence is split after "...in more than 45 minutes, the customer", so the answer comes back as the fragment "receives a full refund of the order amount"; and for "how soon is money returned" the "within 48 hours" part lands in a chunk that is not retrieved. Overlap fixes both. (An earlier version of this project also had a fixed similarity threshold of 0.32 which made no-overlap look worse, 7/12; I removed it so the comparison only shows the effect of overlap.)

**Tools and refund rules:** `TOOL_MAP` has five tools: `get_order` (reads SQLite), `search_policy`, `calculate_refund`, `issue_refund` and `escalate_to_human`. Only completed orders get refunds. Delivery over 45 minutes gives a full refund, 45 minutes or less gives 0. A refund above INR 1,000 is not issued automatically; it is escalated to a human. Issued and escalated refunds are written to `outputs/refund_ledger.jsonl`, and the ledger refuses a second entry for the same order, so a refund can't be paid twice (for example after a crash and resume).

**ReAct loop:** the agent gets the `AGENT` system message and the customer message, asks the LLM what to do, runs the tool it requests, adds the result to the conversation, and repeats until the LLM gives a final answer. Safety limits: 6 LLM calls (`FALLBACK_MAX_ITERATIONS`), 6,000 tokens (`FALLBACK_TOKEN_LIMIT`) and 10 seconds (`FALLBACK_TIMEOUT`). When a limit is hit the case is handed to a human.

| Scenario | Result | Refund (INR) | LLM calls |
|---|---|---:|---:|
| S1 | REFUND | 128.25 | 5 |
| S2 | REFUND | 135.00 | 5 |
| S3 | ESCALATE (over 1000) | 1,999.00 | 5 |
| S4 | DENIED (delivered within 45 min) | 0 | 4 |
| S5 | DENIED (cancelled order) | 0 | 3 |
| S6 | NOT_FOUND (no such order) | 0 | 2 |
| S7 | FALLBACK_MAX_ITERATIONS (`[LOOP]` request) | 0 | 6 |

The code asserts that each scenario matches its expected outcome from `agent_scenarios.csv`. `outputs/agent_runs.jsonl` has one line per scenario with its tool steps.

**Stop demos:** S1 with a token limit of 200 ends with `FALLBACK_TOKEN_LIMIT`; S1 with a time limit of 0 seconds ends with `FALLBACK_TIMEOUT`.

**Checkpoint and resume:** after every tool result the whole state (messages, steps, token count, call count) is saved to `state/<run_id>.json` (written to a temp file first and then renamed, so a crash can't leave a broken file). In the demo S1 is crashed on purpose after 2 tools (`get_order`, `search_policy`). `resume_agent` loads the checkpoint and continues, running only `calculate_refund` and `issue_refund`. The full tool order is `get_order -> search_policy -> calculate_refund -> issue_refund` and the final result is REFUND INR 128.25. Trying to issue the same refund again returns `duplicate: True`. The demos use a separate temporary ledger so the real ledger only has the 3 entries from S1-S3.

**ReAct vs plan-and-execute:** ReAct makes one LLM call per step (S1 needed 5 calls, S6 only 2) because it looks at each tool result before deciding the next step. Plan-and-execute would make a full plan in one call and use fewer calls when the path is predictable. For refunds ReAct is safer, because the result of `get_order` changes what should happen next (order missing, cancelled, delivered on time, amount over 1,000), and the agent can stop immediately instead of following a fixed plan.

## Limitations and Next Steps

- The data is synthetic, so the results (and the very clean 100% few-shot accuracy) will not carry over to real data.
- The lapse model is trained on one time window only and the test set is small (232 rows), so small F1 differences (like logistic vs random forest) are not reliable. Testing on a later month would be better.
- The Premium retention list uses a whole-period segment flag, so it is only illustrative.
- Class balancing is simple oversampling; other options (class weights, threshold tuning) could be compared.
- TF-IDF only matches words, not meaning, and the RAG evaluation has just 14 questions. Embeddings could be tried next.
- The token budget can be exceeded by one call, and the refund ledger is just a text file (not safe for several writers at once).
- The mock LLM is deterministic, so real LLM failures and wrong answers would be more varied.
