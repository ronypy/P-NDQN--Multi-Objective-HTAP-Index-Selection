-- CH-Benchmark HTAP Test Workload (Unseen queries for evaluation)
-- Mix of OLTP and OLAP queries different from training set
SELECT c_id, c_first, c_last, c_balance FROM customer WHERE c_w_id = 1 AND c_d_id = 4 AND c_id = 50
SELECT c_id, c_first, c_last, c_balance FROM customer WHERE c_w_id = 1 AND c_d_id = 5 AND c_id = 75
SELECT c_id, c_first, c_last FROM customer WHERE c_w_id = 1 AND c_d_id = 6 AND c_balance > 100
SELECT d_next_o_id, d_tax FROM district WHERE d_id = 4 AND d_w_id = 1
SELECT d_next_o_id, d_tax FROM district WHERE d_id = 5 AND d_w_id = 1
SELECT i_price, i_name FROM item WHERE i_id = 1500
SELECT i_price, i_name FROM item WHERE i_id = 3000
SELECT i_price, i_name FROM item WHERE i_id BETWEEN 2000 AND 2100
SELECT s_quantity, s_ytd FROM stock WHERE s_w_id = 1 AND s_i_id = 300
SELECT s_quantity, s_ytd FROM stock WHERE s_w_id = 1 AND s_i_id = 400
SELECT o_id, o_entry_d, o_ol_cnt FROM oorder WHERE o_w_id = 1 AND o_d_id = 4 AND o_c_id = 50 ORDER BY o_id DESC LIMIT 5
SELECT o_id, o_entry_d, o_ol_cnt FROM oorder WHERE o_w_id = 1 AND o_d_id = 5 AND o_c_id = 75 ORDER BY o_id DESC LIMIT 5
SELECT ol_i_id, ol_amount FROM order_line WHERE ol_w_id = 1 AND ol_d_id = 4 AND ol_o_id = 200
SELECT ol_i_id, ol_amount FROM order_line WHERE ol_w_id = 1 AND ol_d_id = 5 AND ol_o_id = 300
SELECT no_o_id FROM new_order WHERE no_w_id = 1 AND no_d_id = 4 ORDER BY no_o_id LIMIT 5
SELECT COUNT(*) FROM history WHERE h_w_id = 1 AND h_d_id = 1
SELECT COUNT(*) FROM new_order WHERE no_w_id = 1
SELECT AVG(c_balance) FROM customer WHERE c_w_id = 1 AND c_credit = 'GC'
SELECT AVG(c_balance) FROM customer WHERE c_w_id = 1 AND c_credit = 'BC'
UPDATE customer SET c_balance = c_balance - 25.00, c_ytd_payment = c_ytd_payment + 25.00, c_payment_cnt = c_payment_cnt + 1 WHERE c_w_id = 1 AND c_d_id = 4 AND c_id = 50
UPDATE customer SET c_balance = c_balance - 30.00, c_ytd_payment = c_ytd_payment + 30.00, c_payment_cnt = c_payment_cnt + 1 WHERE c_w_id = 1 AND c_d_id = 5 AND c_id = 75
UPDATE district SET d_ytd = d_ytd + 25.00 WHERE d_w_id = 1 AND d_id = 4
UPDATE district SET d_ytd = d_ytd + 30.00 WHERE d_w_id = 1 AND d_id = 5
UPDATE stock SET s_quantity = s_quantity - 7, s_ytd = s_ytd + 7, s_order_cnt = s_order_cnt + 1 WHERE s_w_id = 1 AND s_i_id = 300
UPDATE stock SET s_quantity = s_quantity - 4, s_ytd = s_ytd + 4, s_order_cnt = s_order_cnt + 1 WHERE s_w_id = 1 AND s_i_id = 400
SELECT ol_number, SUM(ol_quantity), SUM(ol_amount), COUNT(*) FROM order_line WHERE ol_delivery_d > '2008-01-01' GROUP BY ol_number ORDER BY ol_number
SELECT ol_number, SUM(ol_quantity), AVG(ol_amount) FROM order_line WHERE ol_delivery_d > '2009-06-01' GROUP BY ol_number ORDER BY ol_number
SELECT su_suppkey, su_name, n_name, su_acctbal FROM supplier, nation WHERE su_nationkey = n_nationkey AND n_regionkey = 3 ORDER BY su_acctbal DESC LIMIT 50
SELECT o_id, o_w_id, SUM(ol_amount) AS revenue FROM oorder, order_line WHERE ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND o_entry_d > '2008-06-01' AND ol_delivery_d IS NOT NULL GROUP BY o_id, o_w_id ORDER BY revenue DESC LIMIT 15
SELECT o_ol_cnt, COUNT(*) AS order_count FROM oorder WHERE o_entry_d >= '2008-01-01' AND o_entry_d < '2011-01-01' GROUP BY o_ol_cnt ORDER BY o_ol_cnt
SELECT n_name, SUM(ol_amount) AS revenue FROM oorder, order_line, stock, supplier, nation WHERE ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND ol_i_id = s_i_id AND ol_supply_w_id = s_w_id AND (s_w_id * s_i_id) % 10000 = su_suppkey AND su_nationkey = n_nationkey AND o_entry_d >= '2008-01-01' GROUP BY n_name ORDER BY revenue DESC
SELECT SUM(ol_amount) AS revenue FROM order_line WHERE ol_delivery_d >= '2008-06-01' AND ol_delivery_d < '2009-06-01' AND ol_quantity >= 5 AND ol_quantity <= 50000
SELECT o_ol_cnt, SUM(CASE WHEN o_carrier_id = 1 OR o_carrier_id = 2 THEN 1 ELSE 0 END), SUM(CASE WHEN o_carrier_id <> 1 AND o_carrier_id <> 2 THEN 1 ELSE 0 END) FROM oorder, order_line WHERE ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND o_entry_d >= '2008-01-01' AND o_entry_d < '2011-01-01' AND ol_delivery_d IS NOT NULL GROUP BY o_ol_cnt ORDER BY o_ol_cnt
SELECT c_credit, COUNT(*), AVG(c_balance) FROM customer WHERE c_w_id = 1 GROUP BY c_credit
SELECT d_id, SUM(d_ytd) AS total_ytd FROM district WHERE d_w_id = 1 GROUP BY d_id ORDER BY d_id
SELECT EXTRACT(YEAR FROM o_entry_d), COUNT(*), SUM(o_ol_cnt) FROM oorder WHERE o_w_id = 1 GROUP BY EXTRACT(YEAR FROM o_entry_d) ORDER BY EXTRACT(YEAR FROM o_entry_d)
SELECT s_w_id, COUNT(*), AVG(s_quantity), MIN(s_quantity), MAX(s_quantity) FROM stock WHERE s_quantity < 20 GROUP BY s_w_id
SELECT i_price / 20 * 20 AS price_range, COUNT(*) FROM item GROUP BY i_price / 20 * 20 ORDER BY price_range
SELECT n_name, COUNT(su_suppkey), SUM(su_acctbal) FROM supplier, nation WHERE su_nationkey = n_nationkey GROUP BY n_name ORDER BY n_name
SELECT c_id, c_last, c_balance, c_city FROM customer WHERE c_w_id = 1 AND c_d_id = 7 AND c_balance > 0 ORDER BY c_balance DESC LIMIT 20
SELECT ol_i_id, SUM(ol_quantity), SUM(ol_amount) FROM order_line WHERE ol_w_id = 1 AND ol_d_id = 4 GROUP BY ol_i_id ORDER BY SUM(ol_amount) DESC LIMIT 20
SELECT o_c_id, COUNT(*), SUM(o_ol_cnt) FROM oorder WHERE o_w_id = 1 AND o_d_id = 4 GROUP BY o_c_id ORDER BY COUNT(*) DESC LIMIT 20
SELECT h_d_id, SUM(h_amount), COUNT(*) FROM history WHERE h_w_id = 1 GROUP BY h_d_id ORDER BY h_d_id
SELECT s_i_id, s_quantity FROM stock WHERE s_w_id = 1 AND s_quantity < 15 ORDER BY s_quantity LIMIT 50
SELECT c_last, COUNT(*) FROM customer WHERE c_w_id = 1 GROUP BY c_last ORDER BY COUNT(*) DESC LIMIT 20
