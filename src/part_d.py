# Part D: policy RAG (TF-IDF) and a ReAct refund agent with checkpoints
from pathlib import Path
import sys
import json
import os
import sqlite3
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mock_llm

RAW = ROOT / 'data' / 'raw'
POLICIES = ROOT / 'data' / 'policies'
OUT = ROOT / 'outputs'
STATE = ROOT / 'state'
DB = ROOT / 'data' / 'instamart.db'

CHUNK_SIZE = 40
OVERLAP = 10
MAX_ITERATIONS = 6
TOKEN_LIMIT = 6000
TIME_LIMIT = 10        # seconds
LATE_MINUTES = 45      # delivery longer than this -> full refund
AUTO_LIMIT = 1000      # refunds above this need a human


# ---------------------------------------------------------------- RAG
class PolicyRAG:
    def __init__(self, overlap=OVERLAP):
        self.chunks = []                     # list of {'source', 'text'}
        for path in sorted(POLICIES.glob('*.txt')):
            words = path.read_text(encoding='utf-8').split()
            start = 0
            while start < len(words):
                end = min(start + CHUNK_SIZE, len(words))
                self.chunks.append({'source': path.name, 'text': ' '.join(words[start:end])})
                if end == len(words):        # last chunk reached
                    break
                start = end - overlap
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c['text'] for c in self.chunks])

    def retrieve(self, query, k=3):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = np.argsort(-scores)[:k]
        return [{'source': self.chunks[i]['source'], 'score': float(scores[i]), 'text': self.chunks[i]['text']}
                for i in top]

    def answer(self, question):
        hits = self.retrieve(question, 3)
        context = '\n'.join(f"[{h['source']}] {h['text']}" for h in hits)
        messages = [
            {'role': 'system', 'content': 'POLICY ASSISTANT: Answer only from the supplied policy context. '
                                          'If the answer is not found, say I could not find that information in the policy documents.'},
            {'role': 'user', 'content': f'CONTEXT:\n{context}\nQUESTION:\n{question}'}]
        return mock_llm.chat(messages).choices[0].message.content, hits


def evaluate_rag(rag, questions, label):
    doc_ok = fact_ok = fallback_ok = 0
    failed = []
    for q in questions.itertuples():
        answer, hits = rag.answer(q.question)
        if q.expected_doc == 'NONE':
            fallback_ok += answer.startswith('I could not find')
        else:
            doc_ok += q.expected_doc in [h['source'] for h in hits]
            if q.expected_fact in answer:
                fact_ok += 1
            else:
                failed.append((q.question, answer))
    print(f'{label}: {len(rag.chunks)} chunks | doc in top 3: {doc_ok}/12 | fact in answer: {fact_ok}/12 '
          f'| unsupported -> fallback: {fallback_ok}/2')
    return failed


# ---------------------------------------------------------------- tools
TOOLS_SPEC = [
    {'name': 'get_order', 'description': 'Look up a clean order in SQLite.', 'parameters': {'order_id': 'string'}},
    {'name': 'search_policy', 'description': 'Retrieve the most relevant refund-policy chunks.', 'parameters': {'query': 'string'}},
    {'name': 'calculate_refund', 'description': 'Calculate the refund amount from order amount and delivery time.',
     'parameters': {'order_amount': 'number', 'delivery_minutes': 'number'}},
    {'name': 'issue_refund', 'description': 'Record an approved refund in the ledger.',
     'parameters': {'order_id': 'string', 'refund_inr': 'number'}},
    {'name': 'escalate_to_human', 'description': 'Record a refund escalation in the ledger.',
     'parameters': {'order_id': 'string', 'refund_inr': 'number'}},
]


def get_order(order_id):
    con = sqlite3.connect(DB)
    row = con.execute('SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?',
                      (order_id,)).fetchone()
    con.close()
    if row is None:
        return {'error': 'order not found'}
    return {'order_id': row[0], 'status': row[1], 'order_amount': float(row[2]),
            'delivery_minutes': None if row[3] is None else float(row[3])}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or float(delivery_minutes) <= LATE_MINUTES:
        return {'refund_inr': 0.0}
    return {'refund_inr': round(float(order_amount), 2)}


def write_ledger(ledger, tool_name, order_id, refund_inr):
    """Append one line to the ledger. An order can only appear once, so a repeated call
    (e.g. after a crash and resume) cannot refund the same order twice."""
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            old = json.loads(line)
            if old['order_id'] == order_id:
                return {'refund_inr': old['refund_inr'], 'duplicate': True}
    record = {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'), 'tool_name': tool_name,
              'order_id': order_id, 'refund_inr': round(float(refund_inr), 2)}
    with open(ledger, 'a') as f:
        f.write(json.dumps(record) + '\n')
    return {'refund_inr': record['refund_inr']}


def make_tool_map(rag, ledger):
    def search_policy(query):
        return {'results': rag.retrieve(query, 3)}

    def issue_refund(order_id, refund_inr):
        if float(refund_inr) > AUTO_LIMIT:
            return {'error': 'refund above INR 1000 must be escalated'}
        return write_ledger(ledger, 'issue_refund', order_id, refund_inr)

    def escalate_to_human(order_id, refund_inr):
        return write_ledger(ledger, 'escalate_to_human', order_id, refund_inr)

    return {'get_order': get_order, 'search_policy': search_policy, 'calculate_refund': calculate_refund,
            'issue_refund': issue_refund, 'escalate_to_human': escalate_to_human}


# ---------------------------------------------------------------- agent
def save_checkpoint(state):
    # write to a temp file first so a crash can not leave a half-written checkpoint
    path = STATE / f"{state['run_id']}.json"
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    os.replace(temp, path)


def finish(state, status, answer):
    state['finished'] = True
    state['result'] = {'scenario_id': state['scenario_id'], 'status': status, 'final_answer': answer,
                       'llm_calls': state['llm_calls'], 'tokens_used': state['tokens_used'], 'steps': state['steps']}
    save_checkpoint(state)
    return state['result']


def fallback(state, status):
    return finish(state, status, f"{status}: hand off to a human agent. Completed steps={len(state['steps'])}.")


def react_loop(state, tool_map, crash_after=None):
    """Think (LLM call) -> act (tool) -> observe, until a final answer or a limit is hit.
    The same loop is used for new runs and for resuming from a checkpoint."""
    start = time.time() - state['elapsed']
    while state['llm_calls'] < state['max_iterations']:
        state['elapsed'] = time.time() - start
        if state['elapsed'] >= state['time_limit']:
            return fallback(state, 'FALLBACK_TIMEOUT')

        response = mock_llm.chat(state['messages'], tools=TOOLS_SPEC)
        state['llm_calls'] += 1
        state['tokens_used'] += response.usage.total_tokens
        if state['tokens_used'] > state['token_limit']:
            return fallback(state, 'FALLBACK_TOKEN_LIMIT')

        message = response.choices[0].message
        if not message.tool_calls:                       # final answer
            text = message.content or ''
            status = text.split(':', 1)[0].strip() if ':' in text else 'NEEDS_INFO'
            return finish(state, status, text)

        call = message.tool_calls[0]
        call_dict = {'id': call.id, 'name': call.name, 'arguments': dict(call.arguments)}
        if call.name in tool_map:
            observation = tool_map[call.name](**call.arguments)
        else:
            observation = {'error': 'unknown tool ' + call.name}
        state['messages'].append({'role': 'assistant', 'content': None, 'tool_calls': [call_dict]})
        state['messages'].append({'role': 'tool', 'tool_call_id': call.id, 'name': call.name,
                                  'content': json.dumps(observation)})
        state['steps'].append({'tool_name': call.name, 'arguments': call_dict['arguments'], 'observation': observation})
        save_checkpoint(state)                           # checkpoint after every tool observation
        if crash_after is not None and len(state['steps']) == crash_after:
            raise RuntimeError('simulated crash')
    return fallback(state, 'FALLBACK_MAX_ITERATIONS')


def run_agent(message, scenario_id, tool_map, run_id=None, token_limit=TOKEN_LIMIT, time_limit=TIME_LIMIT,
              crash_after=None):
    state = {'run_id': run_id or scenario_id, 'scenario_id': scenario_id, 'finished': False,
             'max_iterations': MAX_ITERATIONS, 'token_limit': token_limit, 'time_limit': time_limit,
             'messages': [{'role': 'system', 'content': 'AGENT: Use the supplied tools to process refund requests safely.'},
                          {'role': 'user', 'content': message}],
             'steps': [], 'tokens_used': 0, 'llm_calls': 0, 'elapsed': 0.0, 'result': None}
    return react_loop(state, tool_map, crash_after)


def resume_agent(run_id, tool_map):
    """Load the last checkpoint and continue. Tools that already ran are in the saved messages,
    so they are not called again."""
    state = json.loads((STATE / f'{run_id}.json').read_text())
    if state['finished']:
        return state['result']
    return react_loop(state, tool_map)


def refund_amount(result):
    for step in reversed(result['steps']):
        if step['tool_name'] in ('issue_refund', 'escalate_to_human') and 'error' not in step['observation']:
            return step['observation']['refund_inr']
    return 0.0


def run():
    OUT.mkdir(exist_ok=True)
    STATE.mkdir(exist_ok=True)
    for old in list(STATE.glob('*')) + [OUT / 'agent_runs.jsonl', OUT / 'refund_ledger.jsonl']:
        old.unlink(missing_ok=True)

    # --- RAG evaluation
    questions = pd.read_csv(RAW / 'policy_eval.csv', keep_default_na=False)
    rag = PolicyRAG(OVERLAP)
    evaluate_rag(rag, questions, 'overlap 10')
    failed = evaluate_rag(PolicyRAG(0), questions, 'overlap 0 ')
    for question, answer in failed:
        print('   zero-overlap miss:', question, '->', answer[:60])
    print('top 3 for "refund policy late delivery":',
          [(h['source'], round(h['score'], 3)) for h in rag.retrieve('refund policy late delivery')])

    # --- seven scenarios
    tool_map = make_tool_map(rag, OUT / 'refund_ledger.jsonl')
    scenarios = pd.read_csv(RAW / 'agent_scenarios.csv', keep_default_na=False)
    with open(OUT / 'agent_runs.jsonl', 'w') as f:
        for s in scenarios.itertuples():
            result = run_agent(s.message, s.scenario_id, tool_map)
            f.write(json.dumps(result) + '\n')
            amount = refund_amount(result)
            ok = result['status'].startswith(s.expected_outcome) and abs(amount - float(s.expected_refund_inr)) < 0.01
            print(s.scenario_id, result['status'], amount, 'calls =', result['llm_calls'],
                  'tokens =', result['tokens_used'], 'OK' if ok else 'MISMATCH')
            assert ok, 'scenario result does not match the expected outcome'

    # --- stop-safe demos (these use their own ledger so the real one is not touched)
    demo_ledger = STATE / 'demo_ledger.jsonl'
    demo_tools = make_tool_map(rag, demo_ledger)
    first = scenarios.iloc[0].message
    r1 = run_agent(first, 'S1', demo_tools, run_id='S1_TOKEN_LIMIT', token_limit=200)
    r2 = run_agent(first, 'S1', demo_tools, run_id='S1_TIMEOUT', time_limit=0)
    print('token limit 200 ->', r1['status'], '| time limit 0 ->', r2['status'])

    # --- crash and resume
    demo_ledger.unlink(missing_ok=True)
    try:
        run_agent(first, 'S1', demo_tools, run_id='S1_CRASH_RESUME', crash_after=2)
    except RuntimeError:
        saved = json.loads((STATE / 'S1_CRASH_RESUME.json').read_text())
        print('crashed after tools:', [s['tool_name'] for s in saved['steps']])
    resumed = resume_agent('S1_CRASH_RESUME', demo_tools)
    print('resumed ->', resumed['status'], refund_amount(resumed))
    print('tool order:', ' -> '.join(s['tool_name'] for s in resumed['steps']))
    assert len(demo_ledger.read_text().splitlines()) == 1       # only one refund written
    again = demo_tools['issue_refund'](order_id='ORD-100048', refund_inr=128.25)
    print('issuing the same refund again ->', again)
    demo_ledger.unlink()


if __name__ == '__main__':
    run()
