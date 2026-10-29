# Part A: clean the raw data, build the SQLite database, run the SQL queries
from pathlib import Path
import re
import sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
CLEAN = ROOT / 'data' / 'clean'
OUT = ROOT / 'outputs'
DB = ROOT / 'data' / 'instamart.db'

GENDER_MAP = {'m': 'male', 'male': 'male', 'f': 'female', 'female': 'female'}

SCHEMA = """
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY, city TEXT, signup_date TEXT, age REAL, gender TEXT);
CREATE TABLE products (
    product_id TEXT PRIMARY KEY, product_name TEXT, category TEXT, price_inr INTEGER);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY, customer_id TEXT, product_id TEXT, quantity INTEGER,
    order_ts TEXT, status TEXT, discount_pct INTEGER, order_amount REAL, delivery_minutes REAL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id));
"""


def raw_audit(customers, orders):
    """Counts of the problems in the raw files."""
    age = pd.to_numeric(customers['age'], errors='coerce')
    gender = customers['gender'].str.strip()
    return {
        'raw_orders': len(orders),
        'exact_duplicates': int(orders.duplicated().sum()),
        'pending': int((orders['status'] == 'pending').sum()),
        'distinct_order_id': int(orders['order_id'].nunique()),
        'nonnumeric_amount': int(pd.to_numeric(orders['order_amount'], errors='coerce').isna().sum()),
        'negative_delivery': int((pd.to_numeric(orders['delivery_minutes'], errors='coerce') < 0).sum()),
        'blank_delivery': int((orders['delivery_minutes'].str.strip() == '').sum()),
        'blank_age': int((customers['age'].str.strip() == '').sum()),
        'age_outside_10_100': int((age.notna() & ~age.between(10, 100)).sum()),
        'raw_city_spellings': int(customers['city'].nunique()),
        'raw_gender_spellings': int(gender[gender != ''].nunique()),
        'blank_gender': int((gender == '').sum()),
    }


def clean_customers(df):
    df = df.copy()
    df['city'] = df['city'].str.strip().str.title()
    df['gender'] = df['gender'].str.strip().str.lower().map(GENDER_MAP).fillna('unknown')

    # ages outside 10-100 are treated as missing, then filled with the city median
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df.loc[~df['age'].between(10, 100), 'age'] = np.nan
    df['age'] = df['age'].fillna(df.groupby('city')['age'].transform('median'))

    df['signup_date'] = pd.to_datetime(df['signup_date'])
    return df


def clean_orders(df, products):
    df = df.copy()
    df['order_ts'] = pd.to_datetime(df['order_ts'])

    rows_before = len(df)
    df = df.drop_duplicates()
    exact_removed = rows_before - len(df)

    # same order_id appears as 'pending' and later as completed/cancelled -> keep the latest
    df = df.sort_values(['order_id', 'order_ts']).drop_duplicates('order_id', keep='last')

    # 'N/A' and 'error' become NaN; refill them from price * quantity * (1 - discount)
    df['order_amount'] = pd.to_numeric(df['order_amount'], errors='coerce')
    df = df.merge(products[['product_id', 'price_inr', 'category']], on='product_id', how='left')
    expected = (df['price_inr'] * df['quantity'] * (1 - df['discount_pct'] / 100)).round(2)
    df['order_amount'] = df['order_amount'].fillna(expected)

    df['delivery_minutes'] = pd.to_numeric(df['delivery_minutes'], errors='coerce')
    df.loc[df['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan

    # fill missing delivery time with the category median, completed orders only
    # (cancelled orders were never delivered so they stay missing)
    done = df['status'] == 'completed'
    cat_median = df[done].groupby('category')['delivery_minutes'].transform('median')
    df.loc[done, 'delivery_minutes'] = df.loc[done, 'delivery_minutes'].fillna(cat_median)

    df = df.drop(columns=['price_inr', 'category']).sort_values('order_id').reset_index(drop=True)
    return df, exact_removed


def load_queries():
    """Read sql/queries.sql and return {'Q1': sql, ...}."""
    queries = {}
    for block in (ROOT / 'sql' / 'queries.sql').read_text().split(';'):
        match = re.search(r'--\s*(Q\d+):', block)
        if match:
            queries[match.group(1)] = block.strip()
    return queries


def run():
    CLEAN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)

    # read everything as text so the dirty values are not changed by pandas
    customers = pd.read_csv(RAW / 'customers.csv', keep_default_na=False, dtype=str)
    products = pd.read_csv(RAW / 'products.csv')
    orders = pd.read_csv(RAW / 'orders.csv', keep_default_na=False,
                         dtype={'order_amount': str, 'delivery_minutes': str})

    print('RAW AUDIT', raw_audit(customers, orders))

    customers = clean_customers(customers)
    orders, exact_removed = clean_orders(orders, products)
    print('exact duplicates removed:', exact_removed)
    print('cities:', customers['city'].value_counts().to_dict())
    print('gender:', customers['gender'].value_counts().to_dict())
    print('orders:', len(orders), orders['status'].value_counts().to_dict())
    print('completed revenue:', round(orders.loc[orders['status'] == 'completed', 'order_amount'].sum(), 2))

    orders['order_ts'] = orders['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    customers['signup_date'] = customers['signup_date'].dt.strftime('%Y-%m-%d')
    customers.to_csv(CLEAN / 'customers.csv', index=False)
    products.to_csv(CLEAN / 'products.csv', index=False)
    orders.to_csv(CLEAN / 'orders.csv', index=False)

    # SQLite
    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    customers.to_sql('customers', con, index=False, if_exists='append')
    products.to_sql('products', con, index=False, if_exists='append')
    orders.to_sql('orders', con, index=False, if_exists='append')
    for name, sql in load_queries().items():
        print(name, 'rows =', len(pd.read_sql_query(sql, con)))
    con.close()

    # pandas version of the same analysis (cross-check for the SQL)
    done = orders[orders['status'] == 'completed'].merge(customers[['customer_id', 'city']], on='customer_id')
    done = done.merge(products[['product_id', 'category']], on='product_id')
    pivot = pd.pivot_table(done, index='city', columns='category', values='order_amount',
                           aggfunc='sum', fill_value=0, margins=True)
    print('pivot grand total:', round(pivot.loc['All', 'All'], 2))
    print(done.groupby('city')['order_amount'].agg(['sum', 'mean', 'count']).loc[['Bengaluru', 'Mumbai']].round(2))

    amount = done['order_amount']
    q1, q3 = amount.quantile(0.25), amount.quantile(0.75)
    fence = q3 + 1.5 * (q3 - q1)
    print('amount: mean', round(amount.mean(), 2), 'median', amount.median(), 'skew', round(amount.skew(), 2))
    print('IQR upper fence', round(fence, 2), 'outliers', int((amount > fence).sum()))
    print(done[['order_amount', 'discount_pct', 'quantity', 'delivery_minutes']].corr().round(3))


if __name__ == '__main__':
    run()
