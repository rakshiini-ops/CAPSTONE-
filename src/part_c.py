# Part C: support ticket triage (guardrails, PII redaction, LLM classification, routing)
from pathlib import Path
import sys
import json
import random
import re
import time
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mock_llm

RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'outputs'

ALLOWED = {'billing', 'delivery', 'technical', 'fraud'}
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
SEED = 42

# PII patterns, applied in this order: email -> phone -> order id
EMAIL_RE = r'[\w.+-]+@[\w-]+\.[\w.]+'
PHONE_RE = r'\b[6-9]\d{9}\b'
ORDER_RE = r'ORD-\d{6}'

ZERO_SYSTEM = '''CLASSIFY
You classify support tickets into exactly one category: billing, delivery, technical, or fraud. Return JSON only in the form {"category":"...","reason":"..."}.'''
FEW_SYSTEM = '''CLASSIFY
You classify support tickets into exactly one category: billing, delivery, technical, or fraud. Return JSON only in the form {"category":"...","reason":"..."}.
EXAMPLES:
Ticket: Someone used my account without permission and my money was stolen. -> {"category":"fraud","reason":"unauthorized activity"}
Ticket: The app crashes when I try to log in. -> {"category":"technical","reason":"app failure"}'''

NOT_SENT = {'skipped_empty', 'flagged_long', 'flagged_invalid_output', 'flagged_llm_error', 'budget_exhausted'}


def redact(text):
    counts = {}
    text, counts['email'] = re.subn(EMAIL_RE, '[EMAIL]', text)
    text, counts['phone'] = re.subn(PHONE_RE, '[PHONE]', text)
    text, counts['order_id'] = re.subn(ORDER_RE, '[ORDER_ID]', text)
    return text, counts


def parse_output(text):
    """Returns (category, status). Try JSON first, then look for a single category word."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            if obj.get('category') in ALLOWED:
                return obj['category'], 'classified_json'
            return '', 'flagged_invalid_output'      # valid JSON but not an allowed category
    except (json.JSONDecodeError, TypeError):
        pass
    words = set(re.findall(r'[a-z]+', (text or '').lower()))
    found = words & ALLOWED
    if len(found) == 1:
        return found.pop(), 'classified_keyword'
    return '', 'flagged_invalid_output'


def call_with_retry(messages, ticket_id, rng, log):
    """Call the LLM. On a transient error wait (exponential backoff + jitter) and retry, max 3 retries."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if attempt == MAX_RETRIES:
                log.append(f'ticket_id={ticket_id} gave up after {MAX_RETRIES} retries')
                return None
            wait = BASE_DELAY * (2 ** attempt) * rng.uniform(0.9, 1.15)
            log.append(f'ticket_id={ticket_id} retry={attempt + 1} wait={wait:.4f}')
            time.sleep(wait)


def run_batch(tickets, system_prompt, budget, label):
    mock_llm.reset()                  # every run starts with the same mock state
    rng = random.Random(SEED)
    rows, log = [], []
    total_tokens = calls = 0
    pii = {'email': 0, 'phone': 0, 'order_id': 0}

    for t in tickets.itertuples():
        row = {'ticket_id': t.ticket_id, 'customer_id': t.customer_id, 'status': '', 'category': '',
               'segment_name': '', 'priority': '', 'route': '', 'tokens_used': 0}
        text = str(t.text)

        if not text.strip():
            row['status'] = 'skipped_empty'
        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            row['status'] = 'flagged_long'
        else:
            clean_text, counts = redact(text)
            for key in pii:
                pii[key] += counts[key]
            # make sure no PII is left
            assert not re.search(EMAIL_RE, clean_text) and not re.search(PHONE_RE, clean_text) \
                and not re.search(ORDER_RE, clean_text)

            if total_tokens >= budget:
                row['status'] = 'budget_exhausted'
            else:
                # the user message is only the redacted ticket text
                messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': clean_text}]
                response = call_with_retry(messages, t.ticket_id, rng, log)
                if response is None:
                    row['status'] = 'flagged_llm_error'
                else:
                    calls += 1
                    row['tokens_used'] = response.usage.total_tokens
                    total_tokens += row['tokens_used']
                    row['category'], row['status'] = parse_output(response.choices[0].message.content)
        rows.append(row)

    results = pd.DataFrame(rows)
    classified = results[results['status'].str.startswith('classified')]
    true_cat = tickets.set_index('ticket_id').loc[classified['ticket_id'], 'true_category'].values
    accuracy = (classified['category'].values == true_cat).mean()
    wrong = pd.Series([f'{a}->{b}' for a, b in zip(true_cat, classified['category']) if a != b]).value_counts()

    print(f'{label}: classified={len(classified)} accuracy={accuracy:.1%} tokens={total_tokens} calls={calls} '
          f'tokens/classified={total_tokens / len(classified):.2f} retries={len([l for l in log if "retry=" in l])}')
    print('   status:', results['status'].value_counts().to_dict())
    print('   wrong (true->predicted):', wrong.to_dict())
    return results, log, pii


def final_route(results, segments):
    """Fill priority and route. Fraud -> security_team, premium -> premium_desk, bad rows -> human_review."""
    segments = segments.set_index('customer_id')
    for i, r in results.iterrows():
        if r['status'] in NOT_SENT:
            results.loc[i, ['priority', 'route']] = ['NORMAL', 'human_review']
            continue
        known = r['customer_id'] in segments.index        # customers with no completed order are not premium
        premium = bool(segments.loc[r['customer_id'], 'is_premium']) if known else False
        results.loc[i, 'segment_name'] = segments.loc[r['customer_id'], 'segment_name'] if known else ''
        if r['category'] == 'fraud':
            results.loc[i, ['priority', 'route']] = ['HIGH', 'security_team']
        elif premium:
            results.loc[i, ['priority', 'route']] = ['HIGH', 'premium_desk']
        else:
            results.loc[i, ['priority', 'route']] = ['NORMAL', 'support_queue']
    return results


def run():
    OUT.mkdir(exist_ok=True)
    tickets = pd.read_csv(RAW / 'tickets.csv', keep_default_na=False)
    segments = pd.read_csv(OUT / 'customer_segments.csv')
    print('tickets:', len(tickets), '| empty:', int((tickets['text'].str.strip() == '').sum()),
          '| long:', sum(mock_llm.est_tokens(t) > MAX_INPUT_TOKENS for t in tickets['text']))

    zero, log0, _ = run_batch(tickets, ZERO_SYSTEM, 100000, 'zero-shot')
    few, log1, pii = run_batch(tickets, FEW_SYSTEM, 100000, 'few-shot')
    demo, log2, _ = run_batch(tickets, ZERO_SYSTEM, 5000, 'budget demo (5000)')
    print('PII replaced:', pii, 'total', sum(pii.values()))

    final_route(few, segments).to_csv(OUT / 'triage_results.csv', index=False)
    final_route(demo, segments).to_csv(OUT / 'triage_budget_demo.csv', index=False)
    print(few.groupby(['route', 'priority']).size())

    # retry log for all three runs
    lines = [f'[zero-shot] {l}' for l in log0] + [f'[few-shot] {l}' for l in log1] + [f'[budget demo] {l}' for l in log2]
    (OUT / 'triage_retries.log').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    run()
