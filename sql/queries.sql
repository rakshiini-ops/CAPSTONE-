-- Q1: completed orders above INR 500, customers whose total exceeds INR 8,000
SELECT customer_id, ROUND(SUM(order_amount), 2) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;

-- Q2: city/category revenue
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: products priced above catalogue average
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: customers with no completed order
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: above-average completed spend using two CTEs
WITH spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
), avg_spend AS (
    SELECT AVG(total_spend) AS mean_spend FROM spend
)
SELECT s.customer_id, ROUND(s.total_spend, 2) AS total_spend
FROM spend s CROSS JOIN avg_spend a
WHERE s.total_spend > a.mean_spend
ORDER BY total_spend DESC;

-- Q6: city-partitioned window ranks
WITH spend AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM customers c JOIN orders o ON c.customer_id=o.customer_id
    WHERE o.status='completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, ROUND(total_spend,2) AS total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rank_pos,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank_pos,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_number_pos
FROM spend
ORDER BY city, row_number_pos;

-- Q7: daily completed revenue and running cumulative total
WITH daily AS (
    SELECT DATE(order_ts) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status='completed'
    GROUP BY DATE(order_ts)
)
SELECT day, ROUND(revenue,2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day),2) AS cumulative_revenue
FROM daily ORDER BY day;
