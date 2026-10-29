# Part B: RFM customer segments + lapse prediction model
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / 'data' / 'clean'
OUT = ROOT / 'outputs'

SEED = 42
SNAPSHOT = pd.Timestamp('2025-10-01')   # RFM "today"
CUTOFF = pd.Timestamp('2025-09-01')     # features use orders before this date only

NUM_COLS = ['recency', 'frequency', 'monetary', 'avg_discount_pct',
            'cancellation_rate', 'avg_delivery_minutes', 'age', 'tenure_days']
CAT_COLS = ['city', 'gender', 'favourite_category']


def build_rfm(orders):
    done = orders[orders['status'] == 'completed']
    rfm = done.groupby('customer_id').agg(last_order=('order_ts', 'max'),
                                          frequency=('order_id', 'nunique'),
                                          monetary=('order_amount', 'sum')).reset_index()
    rfm['recency'] = (SNAPSHOT - rfm['last_order']).dt.days
    return rfm[['customer_id', 'recency', 'frequency', 'monetary']]


def make_segments(rfm):
    scaled = StandardScaler().fit_transform(rfm[['recency', 'frequency', 'monetary']])

    print('K  inertia  silhouette')
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = km.fit_predict(scaled)
        print(k, round(km.inertia_, 1), round(silhouette_score(scaled, labels), 3))

    # K=4 has the best silhouette
    km = KMeans(n_clusters=4, n_init=10, random_state=SEED)
    rfm = rfm.copy()
    rfm['segment_id'] = km.fit_predict(scaled)

    profile = rfm.groupby('segment_id').agg(size=('customer_id', 'size'), recency=('recency', 'mean'),
                                            frequency=('frequency', 'mean'), monetary=('monetary', 'mean'))
    # cluster numbers are arbitrary, so name them from the cluster averages
    premium = profile['monetary'].idxmax()
    at_risk = profile.drop(premium)['recency'].idxmax()
    loyal = profile.drop([premium, at_risk])['frequency'].idxmax()
    names = {sid: 'Regular' for sid in profile.index}
    names.update({premium: 'Premium', at_risk: 'At Risk', loyal: 'Loyal'})
    rfm['segment_name'] = rfm['segment_id'].map(names)
    rfm['is_premium'] = rfm['segment_id'] == premium
    print(profile.assign(name=pd.Series(names)).round(2))

    pca = PCA(n_components=2, random_state=SEED).fit(scaled)
    print('PCA explained variance:', pca.explained_variance_ratio_.round(3), 'total', pca.explained_variance_ratio_.sum().round(3))
    return rfm


def build_lapse_dataset(customers, orders, products):
    """One row per customer. Features come from orders BEFORE the cutoff only;
    the label looks at September (after the cutoff)."""
    before = orders[orders['order_ts'] < CUTOFF]
    done = before[before['status'] == 'completed']
    grouped = done.groupby('customer_id')

    data = pd.DataFrame({
        'last_order': grouped['order_ts'].max(),
        'frequency': grouped['order_id'].nunique(),
        'monetary': grouped['order_amount'].sum(),
        'avg_discount_pct': grouped['discount_pct'].mean(),
        'avg_delivery_minutes': grouped['delivery_minutes'].mean(),
    })
    data['recency'] = (CUTOFF - data['last_order']).dt.days

    final = before[before['status'].isin(['completed', 'cancelled'])]
    cancel_rate = (final['status'] == 'cancelled').groupby(final['customer_id']).mean()
    data['cancellation_rate'] = cancel_rate.reindex(data.index).fillna(0)

    with_cat = done.merge(products[['product_id', 'category']], on='product_id')
    counts = with_cat.groupby(['customer_id', 'category']).size().reset_index(name='n')
    counts = counts.sort_values(['customer_id', 'n', 'category'], ascending=[True, False, True])
    data['favourite_category'] = counts.drop_duplicates('customer_id').set_index('customer_id')['category']

    cust = customers.set_index('customer_id').loc[data.index]
    data['city'], data['age'], data['gender'] = cust['city'], cust['age'], cust['gender']
    data['tenure_days'] = (CUTOFF - cust['signup_date']).dt.days

    # lapsed = no completed order in September 2025
    september = orders[(orders['order_ts'] >= CUTOFF) & (orders['order_ts'] < SNAPSHOT)
                       & (orders['status'] == 'completed')]
    data['lapsed'] = (~data.index.isin(september['customer_id'])).astype(int)
    return data[NUM_COLS + CAT_COLS + ['lapsed']]


def build_preprocessor():
    # k-1 dummy columns for categories; scaler/encoder get fitted inside the pipeline (train data only)
    return ColumnTransformer([('num', StandardScaler(), NUM_COLS),
                              ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CAT_COLS)])


def train_logistic_model(X_train, y_train):
    pipe = Pipeline([('pre', build_preprocessor()), ('model', LogisticRegression(max_iter=1000, random_state=SEED))])
    grid = {'model__C': [0.01, 0.1, 1, 10, 100]}
    return GridSearchCV(pipe, grid, cv=5, scoring='f1').fit(X_train, y_train)


def train_random_forest(X_train, y_train):
    pipe = Pipeline([('pre', build_preprocessor()), ('model', RandomForestClassifier(n_estimators=100, random_state=SEED))])
    grid = {'model__max_depth': [3, 5, 8], 'model__min_samples_leaf': [5, 10, 20]}
    return GridSearchCV(pipe, grid, cv=5, scoring='f1').fit(X_train, y_train)


def balance_training_data(X_train, y_train):
    """Oversample the lapsed class (training data only) until both classes have the same size."""
    n_extra = (y_train == 0).sum() - (y_train == 1).sum()
    extra = X_train[y_train == 1].sample(n=n_extra, replace=True, random_state=SEED)
    X_bal = pd.concat([X_train, extra])
    y_bal = pd.concat([y_train, pd.Series(1, index=extra.index)])
    return X_bal, y_bal


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    print(f'{name:28s} precision={precision_score(y_test, predictions, zero_division=0):.3f} '
          f'recall={recall_score(y_test, predictions, zero_division=0):.3f} '
          f'F1={f1_score(y_test, predictions, zero_division=0):.3f}',
          confusion_matrix(y_test, predictions).tolist())


def generate_retention_list(model, X_test, y_test, rfm):
    """Premium customers in the test set with the highest lapse probability."""
    lapse_prob = model.predict_proba(X_test)[:, 1]
    results = pd.DataFrame({'customer_id': X_test.index, 'lapse_probability': lapse_prob,
                            'actually_lapsed': y_test.values})
    premium_ids = rfm.loc[rfm['is_premium'], 'customer_id']
    results = results[results['customer_id'].isin(premium_ids)]
    return results.sort_values('lapse_probability', ascending=False).head(10)


def run():
    OUT.mkdir(exist_ok=True)
    customers = pd.read_csv(CLEAN / 'customers.csv', parse_dates=['signup_date'])
    orders = pd.read_csv(CLEAN / 'orders.csv', parse_dates=['order_ts'])
    products = pd.read_csv(CLEAN / 'products.csv')

    # --- segmentation
    rfm = build_rfm(orders)
    print('RFM customers:', len(rfm), '| recency median', rfm['recency'].median(),
          '| frequency mean', rfm['frequency'].mean().round(2), '| monetary mean', rfm['monetary'].mean().round(2))
    rfm = make_segments(rfm)
    rfm[['customer_id', 'recency', 'frequency', 'monetary', 'segment_id', 'segment_name', 'is_premium']].to_csv(
        OUT / 'customer_segments.csv', index=False)

    # --- lapse model
    data = build_lapse_dataset(customers, orders, products)
    X, y = data[NUM_COLS + CAT_COLS], data['lapsed']
    print('cohort:', len(data), '| lapsed:', y.sum(), f'({y.mean():.2%})')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print('train/test:', len(X_train), len(X_test))

    baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    log_search = train_logistic_model(X_train, y_train)
    rf_search = train_random_forest(X_train, y_train)
    print('logistic best params', log_search.best_params_, 'cv F1', round(log_search.best_score_, 4))
    print('logistic cv F1 for each C:', log_search.cv_results_['mean_test_score'].round(3))
    print('forest best params', rf_search.best_params_, 'cv F1', round(rf_search.best_score_, 4))

    evaluate_model('baseline (always not lapsed)', baseline, X_test, y_test)
    evaluate_model('logistic regression', log_search, X_test, y_test)
    evaluate_model('random forest', rf_search, X_test, y_test)

    # class balancing: only the training set is oversampled, test set stays untouched
    X_bal, y_bal = balance_training_data(X_train, y_train)
    print('balanced train size:', len(X_bal), 'lapsed =', y_bal.sum())
    balanced_pipe = Pipeline([('pre', build_preprocessor()),
                              ('model', LogisticRegression(C=log_search.best_params_['model__C'],
                                                           max_iter=1000, random_state=SEED))])
    balanced_pipe.fit(X_bal, y_bal)
    evaluate_model('logistic (balanced train)', balanced_pipe, X_test, y_test)

    # feature importance / coefficients
    rf_best = rf_search.best_estimator_
    importances = pd.Series(rf_best['model'].feature_importances_, index=rf_best['pre'].get_feature_names_out())
    print('top 5 RF features:\n', importances.sort_values(ascending=False).head(5).round(3))
    log_best = log_search.best_estimator_
    coefs = pd.Series(log_best['model'].coef_[0], index=log_best['pre'].get_feature_names_out()).sort_values()
    print('logistic coefficients (lowest 3 / highest 3):\n', pd.concat([coefs.head(3), coefs.tail(3)]).round(3))

    # pick the final model with CV score (not test score)
    if rf_search.best_score_ >= log_search.best_score_:
        best_model, best_name = rf_best, 'random forest'
    else:
        best_model, best_name = log_best, 'logistic regression'
    print('selected model (best CV F1):', best_name)

    joblib.dump(best_model, OUT / 'lapse_model.joblib')
    reloaded = joblib.load(OUT / 'lapse_model.joblib')
    assert np.array_equal(best_model.predict(X_test), reloaded.predict(X_test))
    print('model saved and reloaded: predictions identical')

    retention = generate_retention_list(best_model, X_test, y_test, rfm)
    retention.to_csv(OUT / 'retention_list.csv', index=False)
    print('retention list:', len(retention), 'premium customers,', retention['actually_lapsed'].sum(), 'really lapsed')
    # NOTE: is_premium comes from RFM over the whole period (including September),
    # so this list is only illustrative - see README.


if __name__ == '__main__':
    run()
