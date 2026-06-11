SELECT c_discount, c_last, c_credit, w_tax FROM customer, warehouse WHERE w_id = 1 AND c_w_id = w_id AND c_d_id = 1 AND c_id = 1
SELECT d_next_o_id, d_tax FROM district WHERE d_id = 1 AND d_w_id = 1
SELECT i_price, i_name, i_data FROM item WHERE i_id = 100
SELECT s_quantity, s_data, s_dist_01 FROM stock WHERE s_i_id = 100 AND s_w_id = 1
SELECT c_discount, c_last, c_credit, w_tax FROM customer, warehouse WHERE w_id = 1 AND c_w_id = w_id AND c_d_id = 2 AND c_id = 5
SELECT d_next_o_id, d_tax FROM district WHERE d_id = 2 AND d_w_id = 1
SELECT i_price, i_name, i_data FROM item WHERE i_id = 500
SELECT s_quantity, s_data, s_dist_02 FROM stock WHERE s_i_id = 500 AND s_w_id = 1
SELECT c_discount, c_last, c_credit, w_tax FROM customer, warehouse WHERE w_id = 1 AND c_w_id = w_id AND c_d_id = 3 AND c_id = 10
SELECT d_next_o_id, d_tax FROM district WHERE d_id = 3 AND d_w_id = 1
SELECT c_first, c_middle, c_last, c_street_1, c_street_2, c_city, c_state, c_zip, c_phone, c_credit, c_credit_lim, c_discount, c_balance, c_since FROM customer WHERE c_w_id = 1 AND c_d_id = 1 AND c_id = 1
SELECT w_street_1, w_street_2, w_city, w_state, w_zip, w_name FROM warehouse WHERE w_id = 1
SELECT d_street_1, d_street_2, d_city, d_state, d_zip, d_name FROM district WHERE d_w_id = 1 AND d_id = 1
SELECT c_first, c_middle, c_last, c_street_1, c_street_2, c_city, c_state, c_zip, c_phone, c_credit, c_credit_lim, c_discount, c_balance, c_since FROM customer WHERE c_w_id = 1 AND c_d_id = 2 AND c_id = 5
SELECT w_street_1, w_street_2, w_city, w_state, w_zip, w_name FROM warehouse WHERE w_id = 1
SELECT d_street_1, d_street_2, d_city, d_state, d_zip, d_name FROM district WHERE d_w_id = 1 AND d_id = 2
SELECT c_id FROM customer WHERE c_w_id = 1 AND c_d_id = 1 AND c_last = 'BARBARBAR'
SELECT c_balance, c_first, c_middle, c_last FROM customer WHERE c_w_id = 1 AND c_d_id = 1 AND c_id = 1
SELECT o_id, o_carrier_id, o_entry_d FROM oorder WHERE o_w_id = 1 AND o_d_id = 1 AND o_c_id = 1 ORDER BY o_id DESC LIMIT 1
SELECT ol_i_id, ol_supply_w_id, ol_quantity, ol_amount, ol_delivery_d FROM order_line WHERE ol_w_id = 1 AND ol_d_id = 1 AND ol_o_id = 1
SELECT c_balance, c_first, c_middle, c_last FROM customer WHERE c_w_id = 1 AND c_d_id = 2 AND c_id = 5
SELECT o_id, o_carrier_id, o_entry_d FROM oorder WHERE o_w_id = 1 AND o_d_id = 2 AND o_c_id = 5 ORDER BY o_id DESC LIMIT 1
SELECT ol_i_id, ol_supply_w_id, ol_quantity, ol_amount, ol_delivery_d FROM order_line WHERE ol_w_id = 1 AND ol_d_id = 2 AND ol_o_id = 100
SELECT c_balance, c_first, c_middle, c_last FROM customer WHERE c_w_id = 1 AND c_d_id = 3 AND c_id = 10
SELECT o_id, o_carrier_id, o_entry_d FROM oorder WHERE o_w_id = 1 AND o_d_id = 3 AND o_c_id = 10 ORDER BY o_id DESC LIMIT 1
SELECT no_o_id FROM new_order WHERE no_d_id = 1 AND no_w_id = 1 ORDER BY no_o_id LIMIT 1
SELECT o_c_id FROM oorder WHERE o_id = 1 AND o_d_id = 1 AND o_w_id = 1
SELECT SUM(ol_amount) FROM order_line WHERE ol_o_id = 1 AND ol_d_id = 1 AND ol_w_id = 1
SELECT no_o_id FROM new_order WHERE no_d_id = 2 AND no_w_id = 1 ORDER BY no_o_id LIMIT 1
SELECT o_c_id FROM oorder WHERE o_id = 100 AND o_d_id = 2 AND o_w_id = 1
SELECT SUM(ol_amount) FROM order_line WHERE ol_o_id = 100 AND ol_d_id = 2 AND ol_w_id = 1
SELECT no_o_id FROM new_order WHERE no_d_id = 3 AND no_w_id = 1 ORDER BY no_o_id LIMIT 1
SELECT d_next_o_id FROM district WHERE d_id = 1 AND d_w_id = 1
SELECT COUNT(DISTINCT s_i_id) FROM order_line, stock WHERE ol_w_id = 1 AND ol_d_id = 1 AND ol_o_id < 1000 AND ol_o_id >= 980 AND s_w_id = 1 AND s_i_id = ol_i_id AND s_quantity < 15
SELECT d_next_o_id FROM district WHERE d_id = 2 AND d_w_id = 1
SELECT COUNT(DISTINCT s_i_id) FROM order_line, stock WHERE ol_w_id = 1 AND ol_d_id = 2 AND ol_o_id < 2000 AND ol_o_id >= 1980 AND s_w_id = 1 AND s_i_id = ol_i_id AND s_quantity < 10
SELECT d_next_o_id FROM district WHERE d_id = 3 AND d_w_id = 1
SELECT COUNT(DISTINCT s_i_id) FROM order_line, stock WHERE ol_w_id = 1 AND ol_d_id = 3 AND ol_o_id < 500 AND ol_o_id >= 480 AND s_w_id = 1 AND s_i_id = ol_i_id AND s_quantity < 20
SELECT * FROM customer WHERE c_w_id = 1 AND c_d_id = 1 AND c_id = 100
SELECT * FROM customer WHERE c_w_id = 1 AND c_d_id = 2 AND c_id = 200
SELECT * FROM customer WHERE c_w_id = 1 AND c_d_id = 3 AND c_id = 300
SELECT * FROM item WHERE i_id = 1000
SELECT * FROM item WHERE i_id = 2000
SELECT * FROM item WHERE i_id = 5000
SELECT * FROM stock WHERE s_w_id = 1 AND s_i_id = 100
SELECT * FROM stock WHERE s_w_id = 1 AND s_i_id = 200
SELECT * FROM stock WHERE s_w_id = 1 AND s_i_id = 500
SELECT * FROM warehouse WHERE w_id = 1
SELECT * FROM district WHERE d_w_id = 1 AND d_id = 1
SELECT * FROM district WHERE d_w_id = 1 AND d_id = 5
SELECT * FROM oorder WHERE o_w_id = 1 AND o_d_id = 1 AND o_id = 1
SELECT * FROM oorder WHERE o_w_id = 1 AND o_d_id = 2 AND o_id = 100
SELECT * FROM order_line WHERE ol_w_id = 1 AND ol_d_id = 1 AND ol_o_id = 1
SELECT * FROM order_line WHERE ol_w_id = 1 AND ol_d_id = 2 AND ol_o_id = 100
SELECT * FROM new_order WHERE no_w_id = 1 AND no_d_id = 1
SELECT * FROM history WHERE h_c_id = 1 AND h_c_w_id = 1 AND h_c_d_id = 1
UPDATE customer SET c_balance = c_balance - 10.00, c_ytd_payment = c_ytd_payment + 10.00, c_payment_cnt = c_payment_cnt + 1 WHERE c_w_id = 1 AND c_d_id = 1 AND c_id = 1
UPDATE customer SET c_balance = c_balance - 15.00, c_ytd_payment = c_ytd_payment + 15.00, c_payment_cnt = c_payment_cnt + 1 WHERE c_w_id = 1 AND c_d_id = 2 AND c_id = 5
UPDATE customer SET c_balance = c_balance + 100.00, c_delivery_cnt = c_delivery_cnt + 1 WHERE c_w_id = 1 AND c_d_id = 1 AND c_id = 1
UPDATE district SET d_ytd = d_ytd + 10.00 WHERE d_w_id = 1 AND d_id = 1
UPDATE district SET d_ytd = d_ytd + 15.00 WHERE d_w_id = 1 AND d_id = 2
UPDATE district SET d_next_o_id = d_next_o_id + 1 WHERE d_w_id = 1 AND d_id = 1
UPDATE warehouse SET w_ytd = w_ytd + 10.00 WHERE w_id = 1
UPDATE warehouse SET w_ytd = w_ytd + 15.00 WHERE w_id = 1
UPDATE stock SET s_quantity = s_quantity - 5, s_ytd = s_ytd + 5, s_order_cnt = s_order_cnt + 1 WHERE s_w_id = 1 AND s_i_id = 100
UPDATE stock SET s_quantity = s_quantity - 3, s_ytd = s_ytd + 3, s_order_cnt = s_order_cnt + 1 WHERE s_w_id = 1 AND s_i_id = 200
UPDATE stock SET s_quantity = s_quantity + 91 WHERE s_w_id = 1 AND s_i_id = 100 AND s_quantity - 5 < 10
UPDATE oorder SET o_carrier_id = 1 WHERE o_id = 1 AND o_d_id = 1 AND o_w_id = 1
UPDATE oorder SET o_carrier_id = 2 WHERE o_id = 100 AND o_d_id = 2 AND o_w_id = 1
UPDATE order_line SET ol_delivery_d = NOW() WHERE ol_o_id = 1 AND ol_d_id = 1 AND ol_w_id = 1
SELECT c_id, c_first, c_last, c_balance FROM customer WHERE c_w_id = 1 AND c_d_id = 1 AND c_balance > 0
SELECT c_id, c_first, c_last, c_balance FROM customer WHERE c_w_id = 1 AND c_d_id = 1 AND c_credit = 'GC'
SELECT COUNT(*) FROM oorder WHERE o_w_id = 1 AND o_d_id = 1 AND o_carrier_id IS NULL
SELECT COUNT(*) FROM new_order WHERE no_w_id = 1 AND no_d_id = 1
SELECT SUM(ol_amount), COUNT(*) FROM order_line WHERE ol_w_id = 1 AND ol_d_id = 1
SELECT AVG(s_quantity) FROM stock WHERE s_w_id = 1 AND s_quantity < 10
SELECT i_id, i_name, i_price FROM item WHERE i_price > 50 AND i_price < 100
SELECT i_id, i_name, i_price FROM item WHERE i_name LIKE 'A%'
SELECT c_id, c_last FROM customer WHERE c_w_id = 1 AND c_last LIKE 'BAR%'
SELECT * FROM supplier WHERE su_nationkey = 1
SELECT * FROM nation WHERE n_regionkey = 1
SELECT * FROM region WHERE r_regionkey = 1
SELECT ol_number, SUM(ol_quantity) AS sum_qty, SUM(ol_amount) AS sum_amount, AVG(ol_quantity) AS avg_qty, AVG(ol_amount) AS avg_amount, COUNT(*) AS count_order FROM order_line WHERE ol_delivery_d > '2007-01-01' GROUP BY ol_number ORDER BY ol_number
SELECT ol_number, SUM(ol_quantity) AS sum_qty, SUM(ol_amount) AS sum_amount, AVG(ol_quantity) AS avg_qty, AVG(ol_amount) AS avg_amount, COUNT(*) AS count_order FROM order_line WHERE ol_delivery_d > '2010-01-01' GROUP BY ol_number ORDER BY ol_number
SELECT ol_number, SUM(ol_quantity) AS sum_qty, SUM(ol_amount) AS sum_amount FROM order_line WHERE ol_delivery_d IS NOT NULL GROUP BY ol_number
SELECT su_suppkey, su_name, n_name, su_address, su_phone, su_acctbal FROM supplier, nation WHERE su_nationkey = n_nationkey AND n_regionkey = 1 ORDER BY su_acctbal DESC LIMIT 100
SELECT su_suppkey, su_name, n_name, su_address, su_phone, su_acctbal FROM supplier, nation WHERE su_nationkey = n_nationkey AND n_regionkey = 2 ORDER BY su_acctbal DESC LIMIT 100
SELECT o_id, o_w_id, o_d_id, o_entry_d, SUM(ol_amount) AS revenue FROM customer, oorder, order_line WHERE c_id = o_c_id AND c_w_id = o_w_id AND c_d_id = o_d_id AND ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND o_entry_d > '2007-01-01' AND ol_delivery_d IS NULL GROUP BY o_id, o_w_id, o_d_id, o_entry_d ORDER BY revenue DESC, o_entry_d LIMIT 10
SELECT o_id, o_w_id, o_d_id, o_entry_d, SUM(ol_amount) AS revenue FROM customer, oorder, order_line WHERE c_id = o_c_id AND c_w_id = o_w_id AND c_d_id = o_d_id AND ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND o_entry_d > '2010-01-01' GROUP BY o_id, o_w_id, o_d_id, o_entry_d ORDER BY revenue DESC LIMIT 10
SELECT o_ol_cnt, COUNT(*) AS order_count FROM oorder WHERE o_entry_d >= '2007-01-01' AND o_entry_d < '2012-01-01' AND EXISTS (SELECT * FROM order_line WHERE ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND ol_delivery_d >= o_entry_d) GROUP BY o_ol_cnt ORDER BY o_ol_cnt
SELECT o_ol_cnt, COUNT(*) AS order_count FROM oorder WHERE o_entry_d >= '2010-01-01' AND EXISTS (SELECT * FROM order_line WHERE ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id) GROUP BY o_ol_cnt ORDER BY o_ol_cnt
SELECT n_name, SUM(ol_amount) AS revenue FROM customer, oorder, order_line, stock, supplier, nation, region WHERE c_id = o_c_id AND c_w_id = o_w_id AND c_d_id = o_d_id AND ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND ol_i_id = s_i_id AND ol_supply_w_id = s_w_id AND (s_w_id * s_i_id) % 10000 = su_suppkey AND su_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'EUROPE' AND o_entry_d >= '2007-01-01' GROUP BY n_name ORDER BY revenue DESC
SELECT n_name, SUM(ol_amount) AS revenue FROM customer, oorder, order_line, stock, supplier, nation, region WHERE c_id = o_c_id AND c_w_id = o_w_id AND c_d_id = o_d_id AND ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND ol_i_id = s_i_id AND ol_supply_w_id = s_w_id AND (s_w_id * s_i_id) % 10000 = su_suppkey AND su_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'ASIA' AND o_entry_d >= '2010-01-01' GROUP BY n_name ORDER BY revenue DESC
SELECT SUM(ol_amount) AS revenue FROM order_line WHERE ol_delivery_d >= '2009-01-01' AND ol_delivery_d < '2010-01-01' AND ol_quantity >= 1 AND ol_quantity <= 100000
SELECT SUM(ol_amount) AS revenue FROM order_line WHERE ol_delivery_d >= '2010-01-01' AND ol_delivery_d < '2011-01-01' AND ol_quantity >= 1 AND ol_quantity <= 100000
SELECT SUM(ol_amount) AS revenue FROM order_line WHERE ol_delivery_d >= '2011-01-01' AND ol_delivery_d < '2012-01-01' AND ol_quantity BETWEEN 10 AND 1000
SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation, EXTRACT(YEAR FROM o_entry_d) AS l_year, SUM(ol_amount) AS revenue FROM supplier, stock, order_line, oorder, customer, nation n1, nation n2 WHERE ol_supply_w_id = s_w_id AND ol_i_id = s_i_id AND (s_w_id * s_i_id) % 10000 = su_suppkey AND ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND c_id = o_c_id AND c_w_id = o_w_id AND c_d_id = o_d_id AND su_nationkey = n1.n_nationkey AND c_w_id % 10 = n2.n_nationkey AND o_entry_d >= '2007-01-01' AND o_entry_d <= '2012-12-31' GROUP BY n1.n_name, n2.n_name, EXTRACT(YEAR FROM o_entry_d) ORDER BY supp_nation, cust_nation, l_year
SELECT EXTRACT(YEAR FROM o_entry_d) AS l_year, SUM(CASE WHEN n2.n_name = 'GERMANY' THEN ol_amount ELSE 0 END) / NULLIF(SUM(ol_amount), 0) AS mkt_share FROM item, supplier, stock, order_line, oorder, customer, nation n1, nation n2, region WHERE i_id = s_i_id AND ol_i_id = s_i_id AND ol_supply_w_id = s_w_id AND (s_w_id * s_i_id) % 10000 = su_suppkey AND ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND c_id = o_c_id AND c_w_id = o_w_id AND c_d_id = o_d_id AND c_w_id % 10 = n1.n_nationkey AND n1.n_regionkey = r_regionkey AND su_nationkey = n2.n_nationkey AND o_entry_d >= '2007-01-01' AND o_entry_d <= '2012-12-31' AND i_id % 10000 > 1 GROUP BY EXTRACT(YEAR FROM o_entry_d) ORDER BY l_year
SELECT n_name, EXTRACT(YEAR FROM o_entry_d) AS l_year, SUM(ol_amount) AS sum_profit FROM item, stock, supplier, order_line, oorder, nation WHERE ol_i_id = s_i_id AND ol_supply_w_id = s_w_id AND s_i_id = i_id AND (s_w_id * s_i_id) % 10000 = su_suppkey AND ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND su_nationkey = n_nationkey GROUP BY n_name, EXTRACT(YEAR FROM o_entry_d) ORDER BY n_name, l_year DESC
SELECT c_id, c_last, SUM(ol_amount) AS revenue, c_city, c_phone FROM customer, oorder, order_line WHERE c_id = o_c_id AND c_w_id = o_w_id AND c_d_id = o_d_id AND ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND o_entry_d >= '2011-01-01' AND o_entry_d < '2012-01-01' GROUP BY c_id, c_last, c_city, c_phone ORDER BY revenue DESC LIMIT 20
SELECT c_id, c_last, SUM(ol_amount) AS revenue, c_city, c_phone FROM customer, oorder, order_line WHERE c_id = o_c_id AND c_w_id = o_w_id AND c_d_id = o_d_id AND ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND o_entry_d >= '2010-01-01' AND o_entry_d < '2011-01-01' GROUP BY c_id, c_last, c_city, c_phone ORDER BY revenue DESC LIMIT 20
SELECT s_i_id, SUM(s_order_cnt) AS ordercount FROM stock, supplier, nation WHERE (s_w_id * s_i_id) % 10000 = su_suppkey AND su_nationkey = n_nationkey AND n_name = 'GERMANY' GROUP BY s_i_id HAVING SUM(s_order_cnt) > (SELECT SUM(s_order_cnt) * 0.005 FROM stock, supplier, nation WHERE (s_w_id * s_i_id) % 10000 = su_suppkey AND su_nationkey = n_nationkey AND n_name = 'GERMANY') ORDER BY ordercount DESC
SELECT o_ol_cnt, SUM(CASE WHEN o_carrier_id = 1 OR o_carrier_id = 2 THEN 1 ELSE 0 END) AS high_line_count, SUM(CASE WHEN o_carrier_id <> 1 AND o_carrier_id <> 2 THEN 1 ELSE 0 END) AS low_line_count FROM oorder, order_line WHERE ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND o_entry_d >= '2007-01-01' AND o_entry_d < '2012-01-01' AND ol_delivery_d IS NOT NULL GROUP BY o_ol_cnt ORDER BY o_ol_cnt
SELECT o_ol_cnt, SUM(CASE WHEN o_carrier_id = 1 OR o_carrier_id = 2 THEN 1 ELSE 0 END) AS high_line_count, SUM(CASE WHEN o_carrier_id <> 1 AND o_carrier_id <> 2 THEN 1 ELSE 0 END) AS low_line_count FROM oorder, order_line WHERE ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id AND o_entry_d >= '2010-01-01' AND ol_delivery_d IS NOT NULL GROUP BY o_ol_cnt ORDER BY o_ol_cnt
SELECT c_count, COUNT(*) AS custdist FROM (SELECT c_id, COUNT(o_id) AS c_count FROM customer LEFT OUTER JOIN oorder ON (c_w_id = o_w_id AND c_d_id = o_d_id AND c_id = o_c_id AND o_carrier_id > 8) GROUP BY c_id) AS c_orders GROUP BY c_count ORDER BY custdist DESC, c_count DESC
SELECT c_count, COUNT(*) AS custdist FROM (SELECT c_id, COUNT(o_id) AS c_count FROM customer LEFT OUTER JOIN oorder ON (c_w_id = o_w_id AND c_d_id = o_d_id AND c_id = o_c_id AND o_carrier_id > 5) GROUP BY c_id) AS c_orders GROUP BY c_count ORDER BY custdist DESC, c_count DESC
SELECT 100.00 * SUM(CASE WHEN i_data LIKE 'PR%' THEN ol_amount ELSE 0 END) / NULLIF(SUM(ol_amount), 0) AS promo_revenue FROM order_line, item WHERE ol_i_id = i_id AND ol_delivery_d >= '2007-01-01' AND ol_delivery_d < '2008-01-01'
SELECT 100.00 * SUM(CASE WHEN i_data LIKE 'PR%' THEN ol_amount ELSE 0 END) / NULLIF(SUM(ol_amount), 0) AS promo_revenue FROM order_line, item WHERE ol_i_id = i_id AND ol_delivery_d >= '2010-01-01' AND ol_delivery_d < '2011-01-01'
SELECT su_suppkey, su_name, su_address, su_phone, total_revenue FROM supplier, (SELECT (s_w_id * s_i_id) % 10000 AS supplier_no, SUM(ol_amount) AS total_revenue FROM order_line, stock WHERE ol_i_id = s_i_id AND ol_supply_w_id = s_w_id AND ol_delivery_d >= '2007-01-01' AND ol_delivery_d < '2008-01-01' GROUP BY (s_w_id * s_i_id) % 10000) AS revenue WHERE su_suppkey = supplier_no ORDER BY total_revenue DESC LIMIT 1
SELECT i_id, COUNT(DISTINCT su_suppkey) AS supplier_cnt FROM item, stock, supplier WHERE i_id = s_i_id AND (s_w_id * s_i_id) % 10000 = su_suppkey AND i_price >= 1 AND i_price <= 100 AND su_comment NOT LIKE '%Customer%Complaints%' GROUP BY i_id ORDER BY supplier_cnt DESC, i_id
SELECT SUM(ol_amount) / 2.0 AS avg_yearly FROM order_line, (SELECT i_id, AVG(ol_quantity) AS avg_qty FROM item, order_line WHERE i_id = ol_i_id GROUP BY i_id) AS sub WHERE ol_i_id = sub.i_id AND ol_quantity < sub.avg_qty
SELECT c_last, c_id, o_id, o_entry_d, o_ol_cnt, SUM(ol_amount) AS amount FROM customer, oorder, order_line WHERE c_id = o_c_id AND c_w_id = o_w_id AND c_d_id = o_d_id AND ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_o_id = o_id GROUP BY c_id, c_last, o_id, o_entry_d, o_ol_cnt HAVING SUM(ol_amount) > 200 ORDER BY amount DESC, o_entry_d LIMIT 100
SELECT SUM(ol_amount) AS revenue FROM order_line, item WHERE ol_i_id = i_id AND ((i_price >= 1 AND i_price <= 5 AND ol_quantity >= 1 AND ol_quantity <= 10) OR (i_price >= 10 AND i_price <= 20 AND ol_quantity >= 10 AND ol_quantity <= 20) OR (i_price >= 20 AND i_price <= 30 AND ol_quantity >= 20 AND ol_quantity <= 30))
SELECT su_name, su_address FROM supplier, nation WHERE su_suppkey IN (SELECT (s_w_id * s_i_id) % 10000 FROM stock, order_line, item WHERE s_i_id IN (SELECT i_id FROM item WHERE i_data LIKE 'co%') AND ol_i_id = s_i_id AND ol_delivery_d >= '2010-01-01' AND ol_delivery_d < '2011-01-01' GROUP BY s_i_id, s_w_id HAVING 2 * SUM(ol_quantity) > MAX(s_quantity)) AND su_nationkey = n_nationkey AND n_name = 'GERMANY' ORDER BY su_name
SELECT su_name, COUNT(*) AS numwait FROM supplier, order_line l1, oorder, stock, nation WHERE ol_o_id = o_id AND ol_w_id = o_w_id AND ol_d_id = o_d_id AND ol_w_id = s_w_id AND ol_i_id = s_i_id AND (s_w_id * s_i_id) % 10000 = su_suppkey AND l1.ol_delivery_d > o_entry_d AND NOT EXISTS (SELECT * FROM order_line l2 WHERE l2.ol_o_id = l1.ol_o_id AND l2.ol_w_id = l1.ol_w_id AND l2.ol_d_id = l1.ol_d_id AND l2.ol_delivery_d > l1.ol_delivery_d) AND su_nationkey = n_nationkey AND n_name = 'GERMANY' GROUP BY su_name ORDER BY numwait DESC, su_name LIMIT 100
SELECT c_state AS cntrycode, COUNT(*) AS numcust, SUM(c_balance) AS totacctbal FROM customer WHERE SUBSTRING(c_state FROM 1 FOR 1) IN ('1', '2', '3', '4', '5', '6', '7') AND c_balance > (SELECT AVG(c_balance) FROM customer WHERE c_balance > 0.00 AND SUBSTRING(c_state FROM 1 FOR 1) IN ('1', '2', '3', '4', '5', '6', '7')) AND NOT EXISTS (SELECT * FROM oorder WHERE o_c_id = c_id AND o_w_id = c_w_id AND o_d_id = c_d_id) GROUP BY c_state ORDER BY c_state
SELECT d_id, SUM(d_ytd) AS total_ytd FROM district WHERE d_w_id = 1 GROUP BY d_id ORDER BY total_ytd DESC
SELECT c_credit, COUNT(*) AS count, AVG(c_balance) AS avg_balance FROM customer WHERE c_w_id = 1 GROUP BY c_credit
SELECT o_d_id, COUNT(*) AS order_count, SUM(o_ol_cnt) AS total_items FROM oorder WHERE o_w_id = 1 AND o_entry_d >= '2010-01-01' GROUP BY o_d_id ORDER BY order_count DESC
SELECT EXTRACT(YEAR FROM o_entry_d) AS year, EXTRACT(MONTH FROM o_entry_d) AS month, COUNT(*) AS orders, SUM(ol_amount) AS revenue FROM oorder, order_line WHERE o_w_id = ol_w_id AND o_d_id = ol_d_id AND o_id = ol_o_id GROUP BY EXTRACT(YEAR FROM o_entry_d), EXTRACT(MONTH FROM o_entry_d) ORDER BY year, month
SELECT s_w_id, AVG(s_quantity) AS avg_qty, MIN(s_quantity) AS min_qty, MAX(s_quantity) AS max_qty FROM stock GROUP BY s_w_id
SELECT i_price / 10 * 10 AS price_range, COUNT(*) AS item_count FROM item GROUP BY i_price / 10 * 10 ORDER BY price_range
SELECT n_name, COUNT(su_suppkey) AS supplier_count, AVG(su_acctbal) AS avg_balance FROM supplier, nation WHERE su_nationkey = n_nationkey GROUP BY n_name ORDER BY supplier_count DESC
