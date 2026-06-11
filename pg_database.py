import os
import psycopg2
import json
import time
import re

class PG_Database():
    def __init__(self, hypo=True, analyze=False):
        # Load credentials: environment variables take priority over the JSON file.
        # Copy data/db_credentials_template.json to data/db_credentials_pg.json and
        # fill in your values, OR export PG_USER / PG_PASSWORD / PG_HOST / PG_PORT /
        # PG_DATABASE before running.
        creds_file = 'data/db_credentials_pg.json'
        if os.path.exists(creds_file):
            with open(creds_file, 'r') as f:
                file_creds = json.load(f)
        else:
            file_creds = {}

        self.credentials = {
            'user':     os.environ.get('PG_USER',     file_creds.get('user',     'postgres')),
            'password': os.environ.get('PG_PASSWORD', file_creds.get('password', '')),
            'host':     os.environ.get('PG_HOST',     file_creds.get('host',     'localhost')),
            'port':     int(os.environ.get('PG_PORT', file_creds.get('port',     5432))),
            'database': os.environ.get('PG_DATABASE', file_creds.get('database', 'chbench')),
        }

        try:
            self.conn = psycopg2.connect(user = self.credentials['user'],
                                         password = self.credentials['password'],
                                         host = self.credentials['host'],
                                         port = self.credentials['port'],
                                         database = self.credentials['database'])
            self.conn.autocommit = True
            with self.conn.cursor() as cur:
                cur.execute("SET statement_timeout TO '30s';")
            if hypo:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT hypopg_reset();")
                    cur.execute("SET hypopg.use_real_oids = true;")
                self.conn.commit()

        except psycopg2.Error as err:
            raise RuntimeError(f"Database connection error: {err}")

        self.hypo = hypo
        self.analyze = analyze
        self.tables = self.get_tables()
        self.columns = [col for cols in self.tables.values() for col in cols]

    def get_query_cost(self, query):
        output = self.execute_fetchall(f"EXPLAIN (FORMAT JSON) {query}")
        if not output:
            print(f"Query execution returned None: {query}")
            return float('inf')  
        try:
            explain = output[0][0][0]
            cost = explain['Plan']['Total Cost']
            return cost
        except (IndexError, KeyError, TypeError) as e:
            print(f"Error processing query cost: {e}")
            print(f"Output: {output}")
            return float('inf')
    

    
    def get_query_use(self, query, column):
        # Get explain plan
        command = "EXPLAIN {}".format(query)
        output = self.execute_fetchall(command)
        # Verify
        for row in output:
            if 'pkey' in row[0]: continue
            if 'fkey' in row[0]: continue
            if 'Index Scan on' in row[0] and column in row[0]: return 1
            elif 'Index Scan using' in row[0] and column in row[0]: return 1
        return 0

    def get_tables(self):
        # Fetch constraints
        command = "SELECT kcu.column_name FROM information_schema.table_constraints tco JOIN information_schema.key_column_usage kcu ON kcu.constraint_name = tco.constraint_name AND kcu.constraint_schema = tco.constraint_schema AND kcu.constraint_name = tco.constraint_name WHERE tco.constraint_type = 'PRIMARY KEY' OR tco.constraint_type = 'FOREIGN KEY' ORDER BY kcu.table_name;"
        output = self.execute_fetchall(command)
        constraints = [row[0] for row in output]

        # Fetch all tables and columns
        command = "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema='public' AND is_updatable='YES';"
        output = self.execute_fetchall(command)
        tables = dict()
        for row in output:
            table, column = row
            if column not in constraints:
                if table not in tables.keys():
                    tables[table] = list()
                tables[table].append(column)
        
        # Return dict with valid columns for indexing
        return tables

   
    def get_indexes(self):
        """Return a dict mapping each column in self.columns to 1 if indexed, else 0."""
        if self.hypo:
            # hypopg() returns one row per hypothetical index with its OID
            sql = """
                SELECT idx.indexrelid,
                    hypopg_get_indexdef(idx.indexrelid) AS idxdef
                FROM hypopg() AS idx;
            """
        else:
            # real indexes: pull the column_name field directly
            sql = """
                SELECT 
                    a.attname AS column_name
                FROM pg_class t
                JOIN pg_index ix    ON t.oid = ix.indrelid
                JOIN pg_class i     ON i.oid = ix.indexrelid
                JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
                JOIN pg_indexes ix2 ON ix2.schemaname = 'public' AND ix2.indexname = i.relname
                ;
            """

        with self.conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        # build set of indexed columns
        indexed_cols = set()
        if self.hypo:
            # rows = [(indexrelid, idxdef), ...]
            for _, idxdef in rows:
                m = re.search(r'\((.*?)\)', idxdef)
                if not m:
                    continue
                for col in m.group(1).split(','):
                    indexed_cols.add(col.strip())
        else:
            # rows = [(column_name,), ...]
            for (colname,) in rows:
                indexed_cols.add(colname)

        # return a 0/1 map over your full column list
        return { col: (1 if col in indexed_cols else 0)
                for col in self.columns }
    

    def drop_index(self, table, column, verbose=False):
        if self.hypo:
            # List all hypothetical indexes via the hypopg() function
            sql = """
                SELECT idx.indexrelid,
                    hypopg_get_indexdef(idx.indexrelid) AS idxdef
                FROM hypopg() AS idx;
            """
            rows = self.execute_fetchall(sql)
            if not rows:
                return
            # Drop the one matching our table & column
            for oid, idxdef in rows:
                if f"ON {table}" in idxdef and f"({column}" in idxdef:
                    self.execute(f"SELECT hypopg_drop_index({oid});", verbose)
        else:
            if 'smartix_' in column or '_idx' in column:
                command = ("DROP INDEX %s;" % (column))
            else:
                command = ("DROP INDEX smartix_%s;" % (column))
            self.execute(command, verbose)


    def create_index(self, table, column, verbose=False):
        if self.hypo:
            # Using hypothetical indexes, significantly faster
            command = f"SELECT * FROM hypopg_create_index('CREATE INDEX smartix_{column} ON {table} ({column})');"
            self.execute(command, verbose)
        else:
            # Avoid blocking by creating indexes concurrently and only if they do not exist
            command = f"CREATE INDEX CONCURRENTLY IF NOT EXISTS smartix_{column} ON {table} ({column});"
            self.execute(command, verbose)
            
        if self.analyze:
            # Avoid analyzing all tables repeatedly; analyze only the affected table
            command = f"ANALYZE {table};"
            self.execute(command, verbose)

    def reset_indexes(self):
        if self.hypo:
            command = "SELECT hypopg_reset();"
            self.execute(command)
            command = "SET hypopg.use_real_oids = true;"
            self.execute(command)
        else:
            command = "SELECT t.relname AS table_name, i.relname AS index_name, a.attname AS column_name FROM pg_class t, pg_class i, pg_index ix, pg_indexes ixs, pg_attribute a WHERE t.oid = ix.indrelid AND i.oid = ix.indexrelid AND a.attrelid = t.oid AND a.attnum = ANY(ix.indkey) AND ixs.schemaname = 'public' AND i.relname = ixs.indexname ORDER BY t.relname, i.relname;"
            output = self.execute_fetchall(command)
            print(output)
            for index in output:
                index_name = index[1]
                if "smartix_" in index_name or "_idx" in index_name:
                    print("Drop", index_name)
                    self.drop_index(None, index_name)

    def close_connection(self):
        try:
            self.conn.close()
            return True
        except psycopg2.DatabaseError as err:
            print('ERROR: {}'.format(err))
            return False

   

    def execute(self, command, verbose=False):
        try:
            with self.conn.cursor() as cur:
                cur.execute(command)
                if verbose:
                    print(f'OK: {command}')
        except psycopg2.DatabaseError as err:
            print(f'Database execution error: {err}')
            print(f'Failed SQL command: {command}')

    def execute_fetchall(self, command, verbose=False):
        try:
            with self.conn.cursor() as cur:
                cur.execute(command)
                output = cur.fetchall()
                if verbose:
                    print(f'OK: {command}')
                return output
        except psycopg2.DatabaseError as err:
            print(f'Database fetch error: {err}')
            print(f'Failed SQL command: {command}')
            return []

    def get_query_execution_time(self, query: str) -> float:
        """
        Execute query with EXPLAIN ANALYZE and return actual execution time in ms.
        This provides wall-clock timing for actual query execution.

        Returns:
            Execution time in milliseconds, or -1 if execution fails.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"EXPLAIN ANALYZE {query}")
                output = cur.fetchall()

                # Parse the EXPLAIN ANALYZE output to find execution time
                # Format: "Execution Time: X.XXX ms"
                for row in output:
                    line = row[0]
                    if 'Execution Time:' in line:
                        # Extract the time value
                        import re
                        match = re.search(r'Execution Time:\s*([\d.]+)\s*ms', line)
                        if match:
                            return float(match.group(1))

                return -1  # Execution time not found in output
        except psycopg2.DatabaseError as err:
            print(f'EXPLAIN ANALYZE error: {err}')
            return -1

    def get_query_planning_time(self, query: str) -> float:
        """
        Execute query with EXPLAIN ANALYZE and return planning time in ms.

        Returns:
            Planning time in milliseconds, or -1 if execution fails.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"EXPLAIN ANALYZE {query}")
                output = cur.fetchall()

                
                # Format: "Planning Time: X.XXX ms"
                for row in output:
                    line = row[0]
                    if 'Planning Time:' in line:
                        import re
                        match = re.search(r'Planning Time:\s*([\d.]+)\s*ms', line)
                        if match:
                            return float(match.group(1))

                return -1
        except psycopg2.DatabaseError as err:
            print(f'EXPLAIN ANALYZE error: {err}')
            return -1

    def get_index_size_mb(self) -> float:
        """
        Get total size of hypothetical or real indexes in MB.
        For hypothetical indexes, estimates based on column statistics.

        Returns:
            Estimated index size in megabytes.
        """
        if self.hypo:
            # For hypothetical indexes, estimate based on table/column stats
            sql = """
                SELECT idx.indexrelid,
                       hypopg_get_indexdef(idx.indexrelid) AS idxdef
                FROM hypopg() AS idx;
            """
            rows = self.execute_fetchall(sql)
            if not rows:
                return 0.0

            total_size = 0.0
            for _, idxdef in rows:
                # Extract table name from index definition
                import re
                match = re.search(r'ON\s+(\w+)', idxdef)
                if match:
                    table_name = match.group(1)
                    # Get row count estimate for the table
                    size_sql = f"""
                        SELECT reltuples::bigint, relpages
                        FROM pg_class
                        WHERE relname = '{table_name}';
                    """
                    size_result = self.execute_fetchall(size_sql)
                    if size_result:
                        rows_estimate, pages = size_result[0]
                        # Rough estimate: ~20 bytes per index entry
                        # (8 bytes key + 6 bytes tuple ID + overhead)
                        total_size += (rows_estimate * 20) / (1024 * 1024)

            return total_size
        else:
            # For real indexes, query pg_relation_size
            sql = """
                SELECT SUM(pg_relation_size(i.indexrelid)) / (1024.0 * 1024.0) AS size_mb
                FROM pg_indexes ix
                JOIN pg_class i ON i.relname = ix.indexname
                WHERE ix.schemaname = 'public'
                AND (ix.indexname LIKE 'smartix_%' OR ix.indexname LIKE '%_idx');
            """
            result = self.execute_fetchall(sql)
            if result and result[0][0]:
                return float(result[0][0])
            return 0.0

    def get_index_count(self) -> int:
        """Get the number of currently active indexes (hypothetical or real)."""
        indexes = self.get_indexes()
        return sum(indexes.values())



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pprint import pprint


  


    db = PG_Database(hypo=True)

    # Get workload
    with open('data/workload/tpch_shift.sql', 'r') as f:
        data = f.read()
    workload = data.split('\n')

    db.create_index('lineitem', 'l_shipdate')
    db.create_index('part', 'p_size')
    db.create_index('part', 'p_container')
    db.create_index('part', 'p_brand')
    db.create_index('orders', 'o_orderdate')
    db.create_index('customer', 'c_acctbal')

    # Count uses
    for col in ['l_shipdate', 'p_size', 'p_container', 'p_brand', 'o_orderdate', 'c_acctbal']:
        total_count = 0
        for i, q in enumerate(workload):
            count = db.get_query_use(q, col)
            print(col, i, count)
            total_count += count
        print("Total count:", total_count, col)

    db.reset_indexes()

    db.close_connection()
