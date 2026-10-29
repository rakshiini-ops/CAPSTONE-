# Checks the Part A results against the assignment checkpoints.
# Run after part_a.py:  python src/validate_a.py
from pathlib import Path
import sqlite3
import pandas as pd
from part_a import raw_audit, load_queries

ROOT = Path(__file__).resolve().parents[1]

# 1. raw audit
customers_raw = pd.read_csv(ROOT / 'data/raw/customers.csv', keep_default_na=False, dtype=str)
orders_raw = pd.read_csv(ROOT / 'data/raw/orders.csv', keep_default_na=False,
                         dtype={'order_amount': str, 'delivery_minutes': str})
audit = raw_audit(customers_raw, orders_raw)
expected_audit = {'raw_orders': 13084, 'exact_duplicates': 119, 'pending': 656, 'distinct_order_id': 12309,
                  'nonnumeric_amount': 120, 'negative_delivery': 106, 'blank_delivery': 2541, 'blank_age': 61,
                  'age_outside_10_100': 6, 'raw_city_spellings': 24, 'raw_gender_spellings': 10, 'blank_gender': 30}
assert audit == expected_audit, audit
print('raw audit ok')

# 2. cleaned data (read back from the database)
con = sqlite3.connect(ROOT / 'data/instamart.db')
orders = pd.read_sql_query('SELECT * FROM orders', con)
customers = pd.read_sql_query('SELECT * FROM customers', con)
products = pd.read_sql_query('SELECT * FROM products', con)

done = orders['status'] == 'completed'
cancelled = orders['status'] == 'cancelled'
assert len(orders) == 12309 and orders['order_id'].is_unique
assert done.sum() == 11110 and cancelled.sum() == 1199
assert set(orders['status']) == {'completed', 'cancelled'}          # no pending left
assert orders['order_amount'].notna().all()
assert round(orders.loc[done, 'order_amount'].sum(), 2) == 2998668.80
assert orders.loc[done, 'delivery_minutes'].notna().all()
assert orders.loc[cancelled, 'delivery_minutes'].isna().all()      # cancelled stay missing
assert not (orders['delivery_minutes'] < 0).any()
assert len(customers) == 1200 and customers['age'].between(10, 100).all()
assert customers['city'].nunique() == 6
assert customers['gender'].value_counts().to_dict() == {'female': 610, 'male': 560, 'unknown': 30}
print('cleaned data ok')

# 3. database tables and column types
tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert tables == {'customers', 'products', 'orders'}
types = {row[1]: row[2] for row in con.execute('PRAGMA table_info(orders)')}
assert types['order_amount'] == 'REAL' and types['quantity'] == 'INTEGER' and types['order_ts'] == 'TEXT'
print('schema ok')

# 4. SQL results
q = {name: pd.read_sql_query(sql, con) for name, sql in load_queries().items()}
assert len(q['Q1']) == 30
assert len(q['Q2']) == 30 and round(q['Q2']['revenue'].sum(), 2) == 2998668.80
top = q['Q2'].sort_values('revenue').iloc[-1]
assert (top['city'], top['category'], top['revenue']) == ('Bengaluru', 'Electronics', 233909.45)
assert len(q['Q3']) == 11
assert len(q['Q4']) == 25
assert len(q['Q5']) == 550
assert len(q['Q6']) == 1175
leaders = q['Q6'][q['Q6']['rank_pos'] == 1].set_index('city')
expected_leaders = {'Bengaluru': ('C0082', 12856.20), 'Chennai': ('C0930', 10151.40), 'Delhi': ('C0291', 13742.60),
                    'Hyderabad': ('C0743', 12730.05), 'Mumbai': ('C0693', 14766.55), 'Pune': ('C0426', 10377.10)}
for city, (cust, spend) in expected_leaders.items():
    assert leaders.loc[city, 'customer_id'] == cust and leaders.loc[city, 'total_spend'] == spend, city
assert (q['Q6']['rank_pos'] != q['Q6']['row_number_pos']).sum() == 8   # ties
assert len(q['Q7']) == 183 and q['Q7']['cumulative_revenue'].iloc[-1] == 2998668.80
print('SQL results ok')

# 5. pandas gives the same answers as SQL
paid = orders[done].merge(customers[['customer_id', 'city']], on='customer_id').merge(
    products[['product_id', 'category']], on='product_id')
by_city_cat = paid.groupby(['city', 'category'])['order_amount'].sum().round(2).reset_index()
merged = by_city_cat.merge(q['Q2'], on=['city', 'category'])
assert len(merged) == 30 and (merged['order_amount'] - merged['revenue']).abs().max() < 0.01   # Q2

spend = paid.groupby('customer_id')['order_amount'].sum()
assert set(spend[spend > spend.mean()].index) == set(q['Q5']['customer_id'])                 # Q5
assert round(spend.mean(), 2) == 2552.06

big = orders[done & (orders['order_amount'] > 500)].groupby('customer_id')['order_amount'].sum()
assert set(big[big > 8000].index) == set(q['Q1']['customer_id'])                              # Q1

no_orders = set(customers['customer_id']) - set(orders.loc[done, 'customer_id'])
assert no_orders == set(q['Q4']['customer_id'])                                               # Q4

daily = orders[done].groupby(orders['order_ts'].str[:10])['order_amount'].sum().cumsum()
assert abs(daily.iloc[-1] - q['Q7']['cumulative_revenue'].iloc[-1]) < 0.01                    # Q7
print('pandas cross-check ok')

con.close()
print('ALL CHECKPOINTS PASSED')
