# narrative.py
import os, json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def _has_key():
    """Check if OpenAI key is available"""
    return bool(os.getenv("OPENAI_API_KEY"))

def _safe_client():
    """Get OpenAI client or None"""
    if not _has_key():
        return None
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = """You are a concise business analyst. Use only the provided context. 
Do not invent numbers or facts. Write crisp, action-oriented bullets.
"""

MICRO_SYSTEM = """You are a senior retail analyst advising a CEO. 

Your insights must be:
1. Actionable - what specific decision should be made?
2. Strategic - focus on business impact, not chart description
3. Quantified - always include dollar amounts or percentages
4. Forward-looking - what will happen if no action is taken?

Never:
- Describe what's visible in the chart
- Use phrases like "the data shows" or "we can see"
- Give generic advice

Always:
- Start with the business implication
- Suggest a specific next action
- Quantify the risk or opportunity
"""

def micro_insight(context: dict, topic: str) -> str:
    """Generate strategic business insights, not chart descriptions"""
    
    client = _safe_client()
    if not client:
        return _strategic_fallback(context, topic)
    
    try:
        # Enhanced prompt with strategic framing
        prompt = f"""
BUSINESS CONTEXT:
You're analyzing {topic} for a retail executive who needs to make decisions THIS WEEK.

DATA:
{json.dumps(context, indent=2)}

TASK:
Write 1-2 SHORT bullets (max 25 words each) that:
1. Identify the biggest RISK or OPPORTUNITY (with $ impact)
2. Recommend ONE specific action to take this week

Example good insight:
"• Revenue dropped 15% ($124K→$105K) - investigate Store_1 staffing issues before Q4"

Example bad insight:
"• Weekly sales decreased, indicating a downward trend"

YOUR INSIGHT (be specific, actionable, quantified):
"""
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": MICRO_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100  # Force brevity
        )
        
        return resp.choices[0].message.content.strip()
        
    except Exception as e:
        return _strategic_fallback(context, topic)

def _strategic_fallback(context: dict, topic: str) -> str:
    """Generate strategic insights without AI"""
    
    # Sales Trend Analysis
    if "current_value" in context and "previous_value" in context:
        current = context["current_value"]
        previous = context["previous_value"]
        change = current - previous
        pct_change = (change / previous * 100) if previous > 0 else 0
        
        if abs(pct_change) > 10:
            direction = "jumped" if pct_change > 0 else "dropped"
            return f"• Revenue {direction} {abs(pct_change):.1f}% (${abs(change):,.0f}) - {'Investigate cause and replicate success' if pct_change > 0 else 'Urgent: Identify and fix root cause before end of quarter'}"
        else:
            return f"• Stable performance ({pct_change:+.1f}%) - Maintain current strategy but monitor for early warning signs"
    
    # WoW Growth Analysis
    if "last_wow_pct" in context:
        wow = context["last_wow_pct"]
        if wow > 10:
            return f"• Strong momentum (+{wow:.1f}% WoW) - Increase inventory by 15% to capitalize on demand surge"
        elif wow < -10:
            return f"• Sharp decline ({wow:.1f}% WoW) - Review pricing, competition, and marketing spend immediately"
        else:
            return f"• Growth flatlined ({wow:+.1f}%) - Test promotional campaigns in underperforming regions this week"
    
    # Store Performance
    if "leaders" in context and "values" in context:
        top_store = context["leaders"][0]
        top_value = context["values"][0]
        return f"• {top_store} dominates (${top_value:,.0f}) - Document their best practices and train other stores by EOQ"
    
    # Regional Analysis
    if "top_region" in context and "bottom_region" in context:
        top = context["top_region"]
        bottom = context["bottom_region"]
        return f"• {top} outperforms {bottom} - Shift 20% of {bottom}'s marketing budget to {top} for Q4 push"
    
    # Department Concentration
    if "top_dept_share" in context:
        share = context["top_dept_share"]
        if share > 30:
            return f"• Over-concentrated: {share:.1f}% in one department - Diversify product mix to reduce risk"
        else:
            return f"• Balanced portfolio ({share:.1f}% top dept) - Invest in top 3 departments for economies of scale"
    
    # Default
    return f"• {topic}: Analyze weekly trends and set performance targets for next quarter"

# ========= Page-Level Summary (Momentum + Growth Pulse + Benchmarks) =========

SYSTEM_PAGE_SUMMARY = """
You are a senior retail operator. Produce a concise executive summary for THIS WEEK.
Use only the provided context. Do not invent numbers. Be specific and action-oriented.
Structure must be compact bullets, not paragraphs.
"""

def _fallback_page_summary(ctx: dict) -> str:
    """Deterministic, model-free page summary if no API key or API failure."""
    m   = ctx.get("momentum", {})
    gp  = ctx.get("growth_pulse", {})
    sp  = ctx.get("store_pulse", {})
    bks = ctx.get("benchmarks", {})
    s_b = bks.get("store", {}) if isinstance(bks, dict) else {}
    d_b = bks.get("department", {}) if isinstance(bks, dict) else {}

    lines = ["### AI Page Summary", ""]

    # 1) Momentum snapshot
    last_wow = gp.get("last_wow_pct")
    trend_4w = ctx.get("trend_4wk_pct")
    if trend_4w is not None:
        lines.append(f"**Momentum**")
        lines.append(f"• 4-week trend: {trend_4w:+.1f}% — {'acceleration' if trend_4w>0 else 'softness'} detected.")
    if last_wow is not None:
        lines.append(f"• Last WoW: {last_wow:+.1f}% — {'capitalize on demand' if last_wow>0 else 'triage decline fast'}.")

    # 2) Store pulse
    if sp:
        g, d = sp.get("stores_growing", 0), sp.get("stores_declining", 0)
        lines.append("")
        lines.append("**Store Pulse**")
        lines.append(f"• {g} stores improving vs {d} declining (4-week view).")
        if sp.get("best_store"):
            lines.append(f"• Best: {sp['best_store']} ({sp.get('best_growth_pct', 0):+.1f}%).")
        if sp.get("worst_store"):
            lines.append(f"• Fix: {sp['worst_store']} ({sp.get('worst_decline_pct', 0):+.1f}%).")

    # 3) Benchmarks
    lines.append("")
    lines.append("**Ranking & Benchmarks**")
    if s_b.get("bottom") and s_b.get("bottom_gap_pct"):
        lines.append(f"• Store gap: {s_b['bottom']} {s_b['bottom_gap_pct']:.1f}% under benchmark — run promo + conversion audit.")
    if d_b.get("bottom") and d_b.get("bottom_gap_pct"):
        lines.append(f"• Dept gap: {d_b['bottom']} {d_b['bottom_gap_pct']:.1f}% under avg — fix price/placement; 2-week test.")
    if s_b.get("top"):
        lines.append(f"• Replicate top store: {s_b['top']} (roll best-practice playbook).")
    if d_b.get("top"):
        lines.append(f"• Double-down on {d_b['top']} (inventory + promo allocation).")

    # 4) Next-7-days
    lines.append("")
    lines.append("**Next 7 Days — Do This**")
    act = []
    if s_b.get("bottom"): act.append(f"1) Triage **{s_b['bottom']}**: quick promo + staffing/upsell coaching.")
    if sp.get("worst_store"): act.append(f"2) Audit **{sp['worst_store']}** funnel vs {sp.get('best_store','top performer')}.")
    if d_b.get("bottom"): act.append(f"3) Fix **{d_b['bottom']}** price/space in 2-week A/B test.")
    if not act: act.append("1) Monitor momentum; prep playbook replication to underperformers.")
    lines += act[:3]

    return "\n".join(lines)

def draft_page_summary(page_ctx: dict) -> str:
    """
    Compact page summary that blends:
      - Momentum Analysis
      - Weekly Momentum & Store Pulse
      - Ranking & Benchmarks
    Returns markdown with bullets (always short).
    """
    client = _safe_client()
    if not client:
        return _fallback_page_summary(page_ctx)

    try:
        prompt = f"""
Context (JSON):
{json.dumps(page_ctx, indent=2)}

Write a compact page summary for the COO for THIS WEEK.

Rules:
- Output markdown bullets only.
- Sections (exact order & titles):
  1) **Momentum**
  2) **Store Pulse**
  3) **Ranking & Benchmarks**
  4) **Next 7 Days — Do This**
- 2–3 bullets max per section, 8–12 words each where possible.
- Be specific: name stores/departments; include % or $ where present.
- No chart description. No filler.
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PAGE_SUMMARY},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=220,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or _fallback_page_summary(page_ctx)
    except Exception:
        return _fallback_page_summary(page_ctx)
    return resp.choices[0].message.content