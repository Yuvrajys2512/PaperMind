"""
One-off: estimate aggregate query capacity from real usage_events.

Reads the live Neon usage_events table and prints, for kind='query':
  - total query events
  - avg/median/p90 llm_calls, tokens_in, tokens_out, cost_usd per query
Then projects how many such queries the free upstream quotas can sustain.
"""
import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

load_dotenv()
pool = ConnectionPool(os.environ["DATABASE_URL"], min_size=1, max_size=2, open=True)

with pool.connection() as conn:
    row = conn.execute(
        """
        SELECT
            COUNT(*)                                              AS n,
            AVG(llm_calls)                                        AS avg_calls,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY llm_calls)  AS med_calls,
            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY llm_calls)  AS p90_calls,
            AVG(tokens_in)                                        AS avg_tin,
            AVG(tokens_out)                                       AS avg_tout,
            AVG(tokens_in + tokens_out)                           AS avg_ttot,
            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY tokens_in+tokens_out) AS p90_ttot,
            AVG(cost_usd)                                         AS avg_cost,
            SUM(tokens_in + tokens_out)                           AS sum_tot,
            SUM(cost_usd)                                         AS sum_cost
        FROM usage_events
        WHERE kind = 'query'
        """
    ).fetchone()

    distinct_users = conn.execute(
        "SELECT COUNT(DISTINCT user_id) FROM usage_events WHERE kind='query'"
    ).fetchone()[0]

    by_kind = conn.execute(
        "SELECT kind, COUNT(*) FROM usage_events GROUP BY kind ORDER BY 2 DESC"
    ).fetchall()

(n, avg_calls, med_calls, p90_calls, avg_tin, avg_tout, avg_ttot,
 p90_ttot, avg_cost, sum_tot, sum_cost) = row

print("=== usage_events row counts by kind ===")
for k, c in by_kind:
    print(f"  {k:12} {c}")
print(f"\ndistinct users with >=1 query: {distinct_users}")

print("\n=== per-QUERY stats (kind='query') ===")
print(f"  query events (n)      : {n}")
if not n:
    print("  No query events recorded yet — cannot estimate from data.")
    raise SystemExit

def f(x): return float(x) if x is not None else 0.0
avg_calls, med_calls, p90_calls = f(avg_calls), f(med_calls), f(p90_calls)
avg_tin, avg_tout, avg_ttot = f(avg_tin), f(avg_tout), f(avg_ttot)
p90_ttot, avg_cost, sum_tot, sum_cost = f(p90_ttot), f(avg_cost), f(sum_tot), f(sum_cost)

print(f"  llm_calls  avg/med/p90 : {avg_calls:.2f} / {med_calls:.1f} / {p90_calls:.1f}")
print(f"  tokens_in  avg          : {avg_tin:,.0f}")
print(f"  tokens_out avg          : {avg_tout:,.0f}")
print(f"  tokens_tot avg / p90    : {avg_ttot:,.0f} / {p90_ttot:,.0f}")
print(f"  cost_usd   avg          : ${avg_cost:.6f}")
print(f"  lifetime totals         : {sum_tot:,.0f} tokens, ${sum_cost:.4f} projected")

# ---- capacity projection from free upstream quotas ----
GROQ_DAY = 100_000          # tokens/day per Groq key
GROQ_KEYS = 2
MISTRAL_MONTH = 1_000_000_000  # tokens/month

print("\n=== aggregate query capacity (avg tokens/query) ===")
if avg_ttot > 0:
    groq_daily_q = (GROQ_DAY * GROQ_KEYS) / avg_ttot
    mistral_monthly_q = MISTRAL_MONTH / avg_ttot
    print(f"  Groq alone   : {groq_daily_q:,.0f} queries/day  (~{groq_daily_q*30:,.0f}/mo)")
    print(f"  Mistral alone: {mistral_monthly_q:,.0f} queries/month")
    print(f"  free users supportable @20 q/mo on Groq: {groq_daily_q*30/20:,.0f}")
print("\n  (p90 tokens/query gives the conservative floor)")
if p90_ttot > 0:
    print(f"  Groq p90     : {(GROQ_DAY*GROQ_KEYS)/p90_ttot:,.0f} queries/day")
