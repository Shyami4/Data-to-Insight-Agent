# narrative.py - REFINED VERSION
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

# ========= EXECUTIVE PAGE SUMMARY (Momentum + Growth Pulse + Benchmarks) =========

SYSTEM_PAGE_SUMMARY = """You are a senior retail strategist writing for the CEO/COO.

Your summary must be:
- CRISP: 8-15 words per bullet maximum
- SPECIFIC: Always name stores/departments with numbers
- ACTIONABLE: Every section ends with what to DO
- FORWARD-LOOKING: Focus on impact and risk, not history

Format as markdown with exactly these sections:
1. **Momentum** (2 bullets: trend + interpretation)
2. **Store Pulse** (2-3 bullets: what's working/breaking)
3. **Ranking & Benchmarks** (2-3 bullets: gaps + opportunities)
4. **Next 7 Days — Do This** (3 specific actions)

Never use phrases like "the data shows" or "we can see". Start every bullet with impact."""

def _fallback_page_summary(ctx: dict) -> str:
    """Enhanced deterministic summary for Executive page when API unavailable."""
    m   = ctx.get("momentum", {})
    gp  = ctx.get("growth_pulse", {})
    sp  = ctx.get("store_pulse", {})
    bks = ctx.get("benchmarks", {})
    s_b = bks.get("store", {}) if isinstance(bks, dict) else {}
    d_b = bks.get("department", {}) if isinstance(bks, dict) else {}

    lines = []

    # 1) Momentum - Focus on velocity and direction
    trend_4w = ctx.get("trend_4wk_pct")
    last_wow = gp.get("last_wow_pct")
    
    lines.append("**Momentum**")
    if trend_4w is not None:
        if abs(trend_4w) > 5:
            direction = "Accelerating" if trend_4w > 0 else "Slowing"
            urgency = "capitalize fast" if trend_4w > 0 else "intervention needed"
            lines.append(f"• {direction} {abs(trend_4w):.1f}% over 4 weeks — {urgency}")
        else:
            lines.append(f"• Flat trend ({trend_4w:+.1f}%) — test growth initiatives now")
    
    if last_wow is not None:
        if abs(last_wow) > 3:
            action = "scale winners" if last_wow > 0 else "diagnose blockers"
            lines.append(f"• Last week {last_wow:+.1f}% — {action} immediately")
        else:
            lines.append(f"• Steady state ({last_wow:+.1f}%) — proactive testing required")

    # 2) Store Pulse - Winners and losers
    lines.append("")
    lines.append("**Store Pulse**")
    
    if sp:
        g = sp.get("stores_growing", 0)
        d = sp.get("stores_declining", 0)
        total = g + d
        
        if g > d:
            lines.append(f"• {g}/{total} stores growing — replicate playbook to laggards")
        elif d > g:
            lines.append(f"• {d}/{total} stores declining — urgent triage required")
        else:
            lines.append(f"• Mixed performance ({g}↑ {d}↓) — segment and prioritize")
        
        best = sp.get("best_store")
        best_pct = sp.get("best_growth_pct", 0)
        if best and abs(best_pct) > 5:
            lines.append(f"• {best} +{best_pct:.1f}% — document and scale tactics")
        
        worst = sp.get("worst_store")
        worst_pct = sp.get("worst_decline_pct", 0)
        if worst and abs(worst_pct) > 5:
            lines.append(f"• {worst} {worst_pct:.1f}% — diagnose and fix this week")

    # 3) Ranking & Benchmarks - Specific gaps and opportunities
    lines.append("")
    lines.append("**Ranking & Benchmarks**")
    
    # Store gaps
    if s_b.get("bottom") and s_b.get("bottom_gap_pct"):
        gap = s_b["bottom_gap_pct"]
        store = s_b["bottom"]
        if abs(gap) > 10:
            lines.append(f"• {store} {gap:.1f}% below benchmark — recovery plan needed")
    
    # Department gaps
    if d_b.get("bottom") and d_b.get("bottom_gap_pct"):
        gap = d_b["bottom_gap_pct"]
        dept = d_b["bottom"]
        if abs(gap) > 10:
            lines.append(f"• {dept} {gap:.1f}% underperforming — fix pricing/placement")
    
    # Winners to scale
    if s_b.get("top"):
        lines.append(f"• {s_b['top']} leads — extract and deploy playbook")
    elif d_b.get("top"):
        lines.append(f"• {d_b['top']} strongest — increase inventory 20%")

    # 4) Next 7 Days - Specific, sequenced actions
    lines.append("")
    lines.append("**Next 7 Days — Do This**")
    
    actions = []
    
    # Priority 1: Fix critical issues
    if worst and abs(worst_pct) > 5:
        actions.append(f"1. **{worst}**: Emergency audit — staffing, inventory, merchandising")
    elif s_b.get("bottom"):
        actions.append(f"1. **{s_b['bottom']}**: Run flash promo + conversion analysis")
    
    # Priority 2: Scale winners
    if best and abs(best_pct) > 5:
        actions.append(f"2. **{best}**: Document tactics, start rollout to 3 stores")
    elif s_b.get("top"):
        actions.append(f"2. **{s_b['top']}**: Video best practices, train team leads")
    
    # Priority 3: Fix category/department issues
    if d_b.get("bottom") and d_b.get("bottom_gap_pct") and abs(d_b["bottom_gap_pct"]) > 10:
        actions.append(f"3. **{d_b['bottom']}**: 2-week price test + endcap reset")
    elif d_b.get("top"):
        actions.append(f"3. **{d_b['top']}**: Increase allocation 15%, extend promotions")
    
    # Default actions if none above triggered
    if len(actions) == 0:
        actions = [
            "1. **Performance review**: Compare top vs bottom quartile stores",
            "2. **Quick wins**: Test weekend promotions in flat stores",
            "3. **Playbook**: Document and share top store tactics"
        ]
    
    lines.extend(actions[:3])
    
    return "\n".join(lines)

def draft_page_summary(page_ctx: dict) -> str:
    """
    Executive page summary with crisp, actionable insights.
    Returns markdown optimized for CEO/COO quick-scan.
    """
    client = _safe_client()
    if not client:
        return _fallback_page_summary(page_ctx)

    try:
        prompt = f"""
CONTEXT (analyze this data):
{json.dumps(page_ctx, indent=2)}

Write an executive summary for the CEO to read in 60 seconds.

CRITICAL RULES:
1. Use ONLY these section headers (in order):
   - **Momentum**
   - **Store Pulse**
   - **Ranking & Benchmarks**
   - **Next 7 Days — Do This**

2. Bullet style (8-15 words max):
   ✓ "Store_3 -18% — urgent staffing audit required"
   ✗ "We can see that Store_3 has experienced a decline"

3. Every bullet must include:
   - Specific store/department NAME
   - Actual NUMBER (%, $, count)
   - ACTION verb or consequence

4. "Next 7 Days" must be 3 numbered actions:
   - Start each with number: "1. **Entity**: Action"
   - Be concrete: "Run price test" not "Consider options"

5. NO filler words: "shows", "indicates", "appears", "we can see"

OUTPUT (markdown only):
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PAGE_SUMMARY},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,  # Lower for consistency
            max_tokens=300,   # Increased slightly for better actions
        )
        text = (resp.choices[0].message.content or "").strip()
        
        # Validate we got proper structure
        if "**Momentum**" in text and "**Next 7 Days" in text:
            return text
        else:
            return _fallback_page_summary(page_ctx)
            
    except Exception as e:
        return _fallback_page_summary(page_ctx)

# ========= DRIVERS PAGE SUMMARY (Store/Dept/Region Performance) =========

SYSTEM_DRIVERS_SUMMARY = """You are a retail operations analyst writing for regional managers.

Your summary must identify:
- WHO is winning and losing (specific stores/departments)
- BY HOW MUCH (exact percentages and dollar amounts)
- WHAT TO DO (tactical actions for this week)

Be ruthlessly specific. Every bullet must name an entity and include a number.

Format: Markdown bullets only, grouped into 3 sections."""

def _fallback_drivers_summary(ctx: dict) -> str:
    """
    Enhanced deterministic summary for Drivers page when API unavailable.
    Focus on store/department/region performance details.
    """
    def money(x):
        """Format currency safely."""
        if isinstance(x, (int, float)):
            if x >= 1_000_000:
                return f"${x/1_000_000:.1f}M"
            elif x >= 1_000:
                return f"${x/1_000:.0f}K"
            else:
                return f"${x:,.0f}"
        return "—"
    
    lines = []
    
    # 1) Store Performance - Winner and loser spotlight
    lines.append("**Store Performance**")
    
    top_store = ctx.get("top_store_name")
    top_sales = ctx.get("top_store_sales")
    bottom_store = ctx.get("bottom_store_name")
    bottom_sales = ctx.get("bottom_store_sales")
    
    if top_store and top_sales:
        lines.append(f"• {top_store} leads at {money(top_sales)} — capture and scale their tactics")
    
    if bottom_store and bottom_sales:
        gap_desc = ""
        if top_sales and bottom_sales:
            gap_pct = ((top_sales - bottom_sales) / top_sales * 100)
            if gap_pct > 30:
                gap_desc = f" ({gap_pct:.0f}% gap)"
        lines.append(f"• {bottom_store} trails at {money(bottom_sales)}{gap_desc} — immediate intervention")
    
    if not (top_store or bottom_store):
        lines.append("• Store performance data — analyze variance and set targets")
    
    # 2) Category & Regional Insights
    lines.append("")
    lines.append("**Category & Regional Insights**")
    
    # Department concentration
    top_dept = ctx.get("top_dept_name")
    top_dept_sales = ctx.get("top_dept_sales")
    concentration = ctx.get("dept_concentration_pct", 0)
    
    if top_dept:
        conc_note = ""
        if concentration > 35:
            conc_note = f" ({concentration:.0f}% concentration risk)"
        elif concentration > 25:
            conc_note = f" ({concentration:.0f}% share — balanced)"
        
        lines.append(f"• {top_dept} dominates at {money(top_dept_sales)}{conc_note}")
    
    # Bottom department
    bottom_dept = ctx.get("bottom_dept_name")
    bottom_dept_sales = ctx.get("bottom_dept_sales")
    
    if bottom_dept and bottom_dept_sales:
        lines.append(f"• {bottom_dept} underperforms at {money(bottom_dept_sales)} — test pricing/merchandising")
    
    # Regional performance
    top_region = ctx.get("top_region_name")
    top_region_sales = ctx.get("top_region_sales")
    bottom_region = ctx.get("bottom_region_name")
    bottom_region_sales = ctx.get("bottom_region_sales")
    
    if top_region and bottom_region:
        lines.append(f"• {top_region} ({money(top_region_sales)}) vs {bottom_region} ({money(bottom_region_sales)}) — reallocate resources")
    elif top_region:
        lines.append(f"• {top_region} leads at {money(top_region_sales)} — expand presence")
    
    # 3) Tactical Priorities
    lines.append("")
    lines.append("**Tactical Priorities This Week**")
    
    priorities = []
    
    # Fix bottom performers
    if bottom_store:
        priorities.append(f"1. **{bottom_store}**: Staff audit + inventory check + mystery shop")
    
    # Address department gaps
    if bottom_dept and concentration > 30:
        priorities.append(f"2. **{bottom_dept}**: Launch 2-week promo to diversify revenue")
    elif bottom_dept:
        priorities.append(f"2. **{bottom_dept}**: Price audit + placement test in 3 stores")
    
    # Scale winners
    if top_store:
        priorities.append(f"3. **{top_store}**: Film best practices video for team training")
    elif top_dept:
        priorities.append(f"3. **{top_dept}**: Increase inventory 20% + extend promotions")
    
    # Regional reallocation
    if top_region and bottom_region:
        priorities.append(f"4. **{bottom_region}**: Shift 15% marketing budget to {top_region}")
    
    # Add defaults if needed
    if len(priorities) == 0:
        priorities = [
            "1. **Performance gaps**: Identify root causes in bottom quartile",
            "2. **Quick wins**: Test weekend promotions in flat locations",
            "3. **Best practices**: Document and share top performer tactics"
        ]
    
    lines.extend(priorities[:4])
    
    return "\n".join(lines)

def draft_drivers_summary(ctx: dict) -> str:
    """
    Drivers page summary focusing on store/department/regional performance.
    Returns tactical, operator-focused insights.
    """
    client = _safe_client()
    if not client:
        return _fallback_drivers_summary(ctx)
    
    try:
        prompt = f"""
PERFORMANCE DATA:
{json.dumps(ctx, indent=2)}

Write a tactical summary for regional managers who need to act THIS WEEK.

SECTIONS (use these exact headers):
1. **Store Performance**
2. **Category & Regional Insights**
3. **Tactical Priorities This Week**

BULLET REQUIREMENTS:
- 8-15 words maximum per bullet
- MUST include: entity name + number + action
- Money format: Use K for thousands, M for millions

EXAMPLES:
✓ "Store_1 leads at $2.3M — capture their playbook by Friday"
✓ "Electronics $890K (42% share) — diversify to reduce risk"
✗ "The top store is performing well and showing strong results"

TACTICAL PRIORITIES format:
- Numbered list: "1. **Entity**: Specific action with deadline"
- Example: "1. **Store_3**: Staff audit + inventory check by Thursday"

OUTPUT (markdown only):
"""
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_DRIVERS_SUMMARY},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        
        text = (resp.choices[0].message.content or "").strip()
        
        # Validate structure
        if "**Store Performance**" in text and "**Tactical Priorities" in text:
            return text
        else:
            return _fallback_drivers_summary(ctx)
            
    except Exception as e:
        return _fallback_drivers_summary(ctx)