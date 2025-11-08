# narrative.py - REFINED VERSION
import os, json
import numpy as np
import pandas as pd
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
    lines.append("")
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
    
    # Initialize variables to avoid scope issues
    best = None
    best_pct = 0
    worst = None
    worst_pct = 0
    
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
# ========= ENHANCED AI INSIGHT GENERATORS =========

def _generate_store_insights(ctx: dict, df, result: dict) -> str:
    """Generate detailed store performance insights"""
    client = _safe_client()
    
    # Extract store data
    stores_data = result.get("stores", {})
    store_pulse = ctx.get("store_pulse", {})
    
    if not client:
        return _fallback_store_insights(store_pulse, stores_data)
    
    try:
        prompt = f"""
Analyze store performance data for strategic insights.

STORE DATA:
{json.dumps({
    "store_pulse": store_pulse,
    "stores_summary": stores_data,
    "total_stores": df['store'].nunique() if hasattr(df, 'columns') and 'store' in df.columns else 0
}, indent=2)}

Provide 3-4 bullet points covering:
1. Top performer analysis (what they're doing right)
2. Underperformer diagnosis (specific issues)
3. Performance spread (concentration risk)
4. Immediate action required

Format: markdown bullets, be specific with store names and numbers.
"""
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a retail operations analyst. Focus on actionable store insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        return resp.choices[0].message.content.strip()
        
    except Exception:
        return _fallback_store_insights(store_pulse, stores_data)

def _fallback_store_insights(store_pulse: dict, stores_data: dict) -> str:
    """Fallback store insights without AI"""
    insights = []
    
    if store_pulse:
        best = store_pulse.get("best_store")
        best_pct = store_pulse.get("best_growth_pct", 0)
        worst = store_pulse.get("worst_store") 
        worst_pct = store_pulse.get("worst_decline_pct", 0)
        growing = store_pulse.get("stores_growing", 0)
        declining = store_pulse.get("stores_declining", 0)
        
        if best and best_pct > 5:
            insights.append(f"• **{best}** leads with +{best_pct:.1f}% growth - Document their operational playbook")
            
        if worst and worst_pct < -5:
            insights.append(f"• **{worst}** declining {worst_pct:.1f}% - Requires immediate intervention this week")
            
        if growing and declining:
            total = growing + declining
            if declining > growing:
                insights.append(f"• **Performance Alert**: {declining}/{total} stores declining - Systematic issue analysis needed")
            else:
                insights.append(f"• **Balanced Performance**: {growing}/{total} stores growing - Scale winning practices")
    
    if stores_data:
        top_sales = stores_data.get("top_store_sales", 0)
        bottom_sales = stores_data.get("bottom_store_sales", 0)
        if top_sales and bottom_sales:
            gap = (top_sales - bottom_sales) / top_sales * 100
            insights.append(f"• **Performance Gap**: {gap:.0f}% between top and bottom performers - Address capacity constraints")
    
    if not insights:
        insights = ["• Store performance analysis - Enable individual store tracking for detailed insights"]
    
    return "\n".join(insights)

def _generate_risk_assessment(ctx: dict, df, result: dict) -> str:
    """Generate revenue risk assessment"""
    client = _safe_client()
    
    # Extract risk factors
    momentum = ctx.get("momentum", {})
    growth_pulse = ctx.get("growth_pulse", {})
    departments = result.get("departments", {})
    
    if not client:
        return _fallback_risk_assessment(momentum, growth_pulse, departments)
    
    try:
        prompt = f"""
Assess revenue risks based on performance data.

RISK FACTORS:
{json.dumps({
    "momentum": momentum,
    "growth_pulse": growth_pulse,
    "department_concentration": _calculate_concentration(departments) if departments else 0
}, indent=2)}

Identify 3-4 risk factors:
1. Momentum/trend risks
2. Volatility/stability concerns  
3. Concentration risks
4. External vulnerability

Rate each risk as HIGH/MEDIUM/LOW and suggest mitigation.
Format: markdown bullets with risk levels.
"""
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial risk analyst for retail operations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        return resp.choices[0].message.content.strip()
        
    except Exception:
        return _fallback_risk_assessment(momentum, growth_pulse, departments)

def _fallback_risk_assessment(momentum: dict, growth_pulse: dict, departments: dict) -> str:
    """Fallback risk assessment without AI"""
    risks = []
    
    # Momentum risk
    current = momentum.get("current_sales", 0)
    ma4 = momentum.get("ma4", 0)
    if ma4 and current:
        trend = (current - ma4) / ma4 * 100
        if trend < -10:
            risks.append("• **HIGH RISK**: Revenue momentum declining >10% - Immediate action required")
        elif trend < -5:
            risks.append("• **MEDIUM RISK**: Revenue softening - Monitor weekly and prepare contingencies")
    
    # Volatility risk
    last_wow = growth_pulse.get("last_wow_pct", 0)
    if abs(last_wow) > 15:
        risks.append("• **HIGH RISK**: High volatility (±15% WoW) - Stabilize operations and demand planning")
    
    # Concentration risk
    if departments:
        concentration = _calculate_concentration(departments)
        if concentration > 40:
            risks.append(f"• **MEDIUM RISK**: Revenue concentration {concentration:.0f}% in top department - Diversify portfolio")
    
    # Default
    if not risks:
        risks.append("• **LOW RISK**: Performance appears stable - Continue monitoring key metrics")
    
    return "\n".join(risks)

def _generate_growth_opportunities(ctx: dict, df, result: dict) -> str:
    """Generate growth opportunity insights"""
    client = _safe_client()
    
    # Extract opportunity data
    benchmarks = ctx.get("benchmarks", {})
    regional = result.get("regional", {})
    departments = result.get("departments", {})
    
    if not client:
        return _fallback_growth_opportunities(benchmarks, regional, departments)
    
    try:
        prompt = f"""
Identify growth opportunities from performance gaps and benchmarks.

OPPORTUNITY DATA:
{json.dumps({
    "benchmarks": benchmarks,
    "regional_performance": regional,
    "department_performance": departments
}, indent=2, default=str)}

Find 3-4 opportunities:
1. Underperforming units with high potential
2. Successful practices to scale
3. Market expansion possibilities
4. Product/category opportunities

Quantify potential impact where possible.
Format: markdown bullets with opportunity sizing.
"""
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a growth strategy consultant for retail."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        return resp.choices[0].message.content.strip()
        
    except Exception:
        return _fallback_growth_opportunities(benchmarks, regional, departments)

def _fallback_growth_opportunities(benchmarks: dict, regional: dict, departments: dict) -> str:
    """Fallback growth opportunities without AI"""
    opportunities = []
    
    # Store opportunities
    store_bench = benchmarks.get("store", {})
    if store_bench:
        below_avg = store_bench.get("below_avg_count", 0)
        gap_pct = store_bench.get("bottom_gap_pct", 0)
        if below_avg > 0 and gap_pct > 20:
            opportunities.append(f"• **Store Uplift**: {below_avg} stores below average - {gap_pct:.0f}% gap represents significant opportunity")
    
    # Regional expansion
    if regional:
        region_sales = {k: v.get("sales", 0) for k, v in regional.items()}
        if region_sales:
            top_region = max(region_sales, key=region_sales.get)
            bottom_region = min(region_sales, key=region_sales.get)
            gap = region_sales[top_region] - region_sales[bottom_region]
            opportunities.append(f"• **Regional Growth**: {bottom_region} trails {top_region} by ${gap:,.0f} - Scale best practices")
    
    # Department opportunities  
    if departments:
        dept_sales = {k: v.get("sales", 0) for k, v in departments.items()}
        if dept_sales:
            total_sales = sum(dept_sales.values())
            top_dept = max(dept_sales, key=dept_sales.get)
            opportunities.append(f"• **Category Expansion**: {top_dept} success model - Apply to underperforming categories")
    
    if not opportunities:
        opportunities.append("• **Market Analysis**: Conduct deeper analysis to identify growth opportunities")
    
    return "\n".join(opportunities)

def _generate_efficiency_insights(ctx: dict, df, result: dict) -> str:
    """Generate operational efficiency insights"""
    client = _safe_client()
    
    # Calculate efficiency metrics
    efficiency_data = _calculate_efficiency_metrics(df, result)
    
    if not client:
        return _fallback_efficiency_insights(efficiency_data)
    
    try:
        prompt = f"""
Analyze operational efficiency metrics for improvement opportunities.

EFFICIENCY DATA:
{json.dumps(efficiency_data, indent=2)}

Provide 3-4 efficiency insights:
1. Transaction efficiency (ATV, conversion)
2. Resource utilization gaps
3. Process optimization opportunities  
4. Cost reduction potential

Focus on actionable improvements with ROI estimates.
Format: markdown bullets with efficiency gains.
"""
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an operations efficiency consultant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        return resp.choices[0].message.content.strip()
        
    except Exception:
        return _fallback_efficiency_insights(efficiency_data)

def _fallback_efficiency_insights(efficiency_data: dict) -> str:
    """Fallback efficiency insights without AI"""
    insights = []
    
    avg_transaction = efficiency_data.get("avg_transaction_value", 0)
    if avg_transaction:
        insights.append(f"• **Transaction Optimization**: Avg transaction ${avg_transaction:.0f} - Test upselling strategies")
    
    stores_count = efficiency_data.get("active_stores", 0)
    if stores_count:
        sales_per_store = efficiency_data.get("sales_per_store", 0)
        insights.append(f"• **Store Productivity**: ${sales_per_store:,.0f} per store - Benchmark against top quartile")
    
    utilization = efficiency_data.get("capacity_utilization", 0)
    if utilization and utilization < 80:
        insights.append(f"• **Capacity Gap**: {utilization:.0f}% utilization - Opportunity for volume growth")
    
    if not insights:
        insights.append("• **Efficiency Review**: Implement detailed operational metrics tracking")
    
    return "\n".join(insights)

def _generate_strategic_actions(ctx: dict, df, result: dict) -> str:
    """Generate comprehensive strategic recommendations with 5 bullets per priority level"""
    client = _safe_client()
    
    if not client:
        return _fallback_strategic_actions(ctx, result)
    
    try:
        # Extract comprehensive context for detailed recommendations
        prompt = f"""
Create strategic recommendations with exactly 5 detailed actions per priority level.

COMPLETE CONTEXT:
{json.dumps(ctx, indent=2, default=str)}

Generate strategic actions organized by priority level:

**IMMEDIATE ACTIONS (Next 7 Days)**
1. **[Entity]**: Specific action with timeline and expected impact (15-20 words max)
2. **[Entity]**: Specific action with timeline and expected impact (15-20 words max)
3. **[Entity]**: Specific action with timeline and expected impact (15-20 words max)
4. **[Entity]**: Specific action with timeline and expected impact (15-20 words max)
5. **[Entity]**: Specific action with timeline and expected impact (15-20 words max)

**HIGH PRIORITY (Next 30 Days)**
1. **[Entity]**: Strategic initiative with implementation steps (15-20 words max)
2. **[Entity]**: Strategic initiative with implementation steps (15-20 words max)
3. **[Entity]**: Strategic initiative with implementation steps (15-20 words max)
4. **[Entity]**: Strategic initiative with implementation steps (15-20 words max)
5. **[Entity]**: Strategic initiative with implementation steps (15-20 words max)

**MEDIUM PRIORITY (Next 90 Days)**
1. **[Entity]**: Long-term strategic initiative with ROI focus (15-20 words max)
2. **[Entity]**: Long-term strategic initiative with ROI focus (15-20 words max)
3. **[Entity]**: Long-term strategic initiative with ROI focus (15-20 words max)
4. **[Entity]**: Long-term strategic initiative with ROI focus (15-20 words max)
5. **[Entity]**: Long-term strategic initiative with ROI focus (15-20 words max)

Requirements:
- Reference specific entities from the data (stores, departments, regions)
- Include quantified targets (percentages, dollar amounts)
- Keep each bullet to 15-20 words maximum
- Focus on actionable steps with clear outcomes
- Use actual data from context for concrete recommendations
"""
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior retail strategy consultant. Create concise, actionable strategic recommendations with exactly 5 bullets per priority level."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=400
        )
        
        return resp.choices[0].message.content.strip()
        
    except Exception:
        return _fallback_strategic_actions(ctx, result)

def _fallback_strategic_actions(ctx: dict, result: dict) -> str:
    """Fallback strategic actions with exactly 5 bullets per priority level"""
    actions = []
    
    # Get key entities and metrics from context
    drivers = ctx.get("drivers", {})
    growth_pulse = ctx.get("growth_pulse", {})
    momentum = ctx.get("momentum", {})
    store_pulse = ctx.get("store_pulse", {})
    
    last_wow = growth_pulse.get("last_wow_pct", 0)
    top_region = drivers.get("regional", {}).get("top")
    bottom_region = drivers.get("regional", {}).get("bottom")
    top_dept = drivers.get("departments", {}).get("top")
    bottom_dept = drivers.get("departments", {}).get("bottom")
    worst_store = store_pulse.get("worst_store")
    best_store = store_pulse.get("best_store")
    declining = store_pulse.get("stores_declining", 0)
    
    # IMMEDIATE ACTIONS (7 Days)
    actions.append("**IMMEDIATE ACTIONS (Next 7 Days)**")
    
    if last_wow < -10:
        actions.append("1. **Crisis Response**: Form emergency task force to investigate revenue decline and implement recovery measures.")
    elif worst_store:
        actions.append(f"1. **{worst_store}**: Deploy management team for immediate intervention and promotional campaign launch.")
    else:
        actions.append("1. **Performance Alert**: Identify bottom quartile locations and implement targeted promotional campaigns.")
    
    if declining > 2:
        actions.append(f"2. **Multi-Store Recovery**: Launch coordinated intervention across {declining} declining locations with standardized approach.")
    elif bottom_region:
        actions.append(f"2. **{bottom_region}**: Deploy regional marketing blitz with enhanced promotional partnerships and customer engagement.")
    else:
        actions.append("2. **Market Push**: Implement proactive marketing campaigns in underperforming segments.")
    
    if bottom_dept:
        actions.append(f"3. **{bottom_dept}**: Conduct emergency inventory review and implement pricing optimization strategies.")
    else:
        actions.append("3. **Category Focus**: Review underperforming categories and implement placement optimization initiatives.")
    
    actions.append("4. **Staff Training**: Deploy immediate training sessions on sales techniques and customer engagement best practices.")
    actions.append("5. **Daily Monitoring**: Establish enhanced daily performance tracking and rapid response protocols.")
    
    # HIGH PRIORITY (30 Days)
    actions.append("")
    actions.append("**HIGH PRIORITY (Next 30 Days)**")
    
    if best_store:
        actions.append(f"1. **{best_store} Playbook**: Document success strategies and create implementation guide for network rollout.")
    else:
        actions.append("1. **Best Practices**: Identify top performers and systematize their operational excellence strategies.")
    
    if top_region and bottom_region:
        actions.append(f"2. **Regional Alignment**: Transfer {top_region} strategies to {bottom_region} with quarterly performance targets.")
    else:
        actions.append("2. **Market Optimization**: Implement data-driven market strategies across all regions.")
    
    actions.append("3. **Technology Upgrade**: Deploy advanced POS analytics and real-time performance monitoring systems.")
    actions.append("4. **Inventory Optimization**: Implement demand forecasting and automated replenishment across all locations.")
    actions.append("5. **Customer Experience**: Launch comprehensive customer satisfaction improvement program with loyalty initiatives.")
    
    # MEDIUM PRIORITY (90 Days)
    actions.append("")
    actions.append("**MEDIUM PRIORITY (Next 90 Days)**")
    
    actions.append("1. **Analytics Platform**: Implement predictive analytics and AI-driven demand forecasting system.")
    actions.append("2. **Market Expansion**: Evaluate new market opportunities and develop strategic expansion roadmap.")
    actions.append("3. **Product Innovation**: Launch new product categories based on market analysis and customer insights.")
    actions.append("4. **Operational Excellence**: Establish lean operations framework and continuous improvement processes.")
    actions.append("5. **Strategic Partnerships**: Develop key vendor relationships and explore joint venture opportunities.")
    
    return "\n".join(actions)

# Helper functions
def _calculate_concentration(departments: dict) -> float:
    """Calculate revenue concentration in top department"""
    if not departments:
        return 0
    
    sales_values = [d.get("sales", 0) for d in departments.values()]
    if not sales_values:
        return 0
        
    total_sales = sum(sales_values)
    max_sales = max(sales_values)
    
    return (max_sales / total_sales * 100) if total_sales > 0 else 0

def _calculate_efficiency_metrics(df, result: dict) -> dict:
    """Calculate operational efficiency metrics"""
    metrics = {}
    
    if hasattr(df, 'columns') and not df.empty:
        # Basic metrics
        if 'weekly_sales' in df.columns:
            metrics["total_sales"] = df['weekly_sales'].sum()
            metrics["avg_transaction_value"] = df['weekly_sales'].mean()
        
        if 'store' in df.columns:
            metrics["active_stores"] = df['store'].nunique()
            if metrics.get("total_sales"):
                metrics["sales_per_store"] = metrics["total_sales"] / metrics["active_stores"]
        
        # Capacity utilization (simplified)
        if 'transactions' in df.columns:
            total_transactions = df['transactions'].sum() if 'transactions' in df.columns else df.shape[0]
            max_possible = df.shape[0] * 100  # Simplified assumption
            metrics["capacity_utilization"] = (total_transactions / max_possible * 100) if max_possible else 0
    
    return metrics

def _build_enhanced_ai_context(df, result: dict) -> dict:
    """Build enhanced context for AI insights with additional metrics"""
    # Import pandas here to avoid circular imports
    import pandas as pd
    
    # Start with base context - we'll use the existing function from app.py
    # For now, create a simplified context
    ctx = {}
    
    # Build basic momentum context if we have weekly data
    if hasattr(df, 'columns') and 'weekly_sales' in df.columns:
        kpis_weekly = result.get("kpis_weekly", {})
        if kpis_weekly:
            w = pd.DataFrame(kpis_weekly)
            if not w.empty and "date" in w and "weekly_sales_sum" in w:
                w["date"] = pd.to_datetime(w["date"], errors="coerce")
                w = w.dropna(subset=["date"]).sort_values("date")
                y = w["weekly_sales_sum"]
                ma4 = y.rolling(4, min_periods=1).mean()
                wow = y.pct_change() * 100
                ctx["momentum"] = {
                    "current_sales": float(y.iloc[-1]) if len(y) > 0 else None,
                    "ma4": float(ma4.iloc[-1]) if len(ma4) > 0 else None,
                }
                ctx["growth_pulse"] = {
                    "last_wow_pct": float(wow.iloc[-1]) if len(wow) > 0 else 0.0,
                }
    
    # Add enhanced metrics
    ctx["efficiency_metrics"] = _calculate_efficiency_metrics(df, result)
    
    # Add data quality indicators
    if hasattr(df, 'columns') and not df.empty:
        ctx["data_quality"] = {
            "completeness": (df.notna().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
            "date_range_days": (df['date'].max() - df['date'].min()).days if 'date' in df.columns else 0,
            "record_count": len(df)
        }
    else:
        ctx["data_quality"] = {"completeness": 0, "date_range_days": 0, "record_count": 0}
    
    # Add driver context
    regional = result.get("regional", {})
    departments = result.get("departments", {})
    
    ctx["drivers"] = {
        "regional": {
            "regions": list(regional.keys()) if regional else [],
            "sales": {k: regional[k].get("sales", 0) for k in regional} if regional else {},
            "top": max(regional, key=lambda k: regional[k].get("sales", 0)) if regional else None,
            "bottom": min(regional, key=lambda k: regional[k].get("sales", 0)) if regional else None,
        },
        "departments": {
            "sales": {k: departments[k].get("sales", 0) for k in departments} if departments else {},
            "top": max(departments, key=lambda k: departments[k].get("sales", 0)) if departments else None,
            "bottom": min(departments, key=lambda k: departments[k].get("sales", 0)) if departments else None,
        }
    }
    
    return ctx

def _generate_strategic_actions_concise(ctx: dict, df, result: dict) -> str:
    """Generate concise strategic actions matching the formatting style of existing summaries"""
    client = _safe_client()
    
    if not client:
        return _fallback_strategic_actions_concise(ctx, result)
    
    try:
        # Extract key data points for accurate recommendations
        momentum = ctx.get("momentum", {})
        growth_pulse = ctx.get("growth_pulse", {})
        drivers = ctx.get("drivers", {})
        benchmarks = ctx.get("benchmarks", {})
        
        prompt = f"""
Create strategic actions matching this exact format style:

**Next 7 Days — Do This**
1. **Entity**: Action with specific metric target.
2. **Entity**: Action with specific metric target.
3. **Entity**: Action with specific metric target.

CONTEXT DATA:
{json.dumps({
    "momentum": momentum,
    "growth_pulse": growth_pulse,
    "top_region": drivers.get("regional", {}).get("top"),
    "bottom_region": drivers.get("regional", {}).get("bottom"),
    "top_dept": drivers.get("departments", {}).get("top"),
    "bottom_dept": drivers.get("departments", {}).get("bottom"),
    "benchmarks": benchmarks
}, indent=2)}

Requirements:
- Use exact format: "**Entity**: Action with percentage target."
- Reference actual entities from the data (stores, departments, regions)
- Include specific percentage targets (10%, 15%, 20%)
- Keep each action to 8-12 words maximum
- Focus on immediate tactical actions

OUTPUT (markdown only):
"""
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a retail strategist creating tactical action plans. Match the exact format provided."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )
        
        return resp.choices[0].message.content.strip()
        
    except Exception:
        return _fallback_strategic_actions_concise(ctx, result)

def _fallback_strategic_actions_concise(ctx: dict, result: dict) -> str:
    """Fallback strategic actions matching existing format style"""
    actions = []
    
    # Get key entities from context
    drivers = ctx.get("drivers", {})
    growth_pulse = ctx.get("growth_pulse", {})
    last_wow = growth_pulse.get("last_wow_pct", 0)
    
    top_region = drivers.get("regional", {}).get("top")
    bottom_region = drivers.get("regional", {}).get("bottom")
    top_dept = drivers.get("departments", {}).get("top")
    bottom_dept = drivers.get("departments", {}).get("bottom")
    
    # Store pulse data
    store_pulse = ctx.get("store_pulse", {})
    worst_store = store_pulse.get("worst_store")
    best_store = store_pulse.get("best_store")
    
    actions.append("**Next 7 Days — Do This**")
    
    # Priority actions based on actual data
    if worst_store:
        actions.append(f"1. **{worst_store}**: Implement targeted promotions to increase sales by 15%.")
    elif bottom_region:
        actions.append(f"1. **{bottom_region}**: Launch promotional campaigns to boost performance by 10%.")
    else:
        actions.append("1. **Underperformers**: Identify and address bottom quartile locations by 15%.")
    
    if best_store:
        actions.append(f"2. **{best_store}**: Document best practices for replication across network.")
    elif top_region:
        actions.append(f"2. **{top_region}**: Scale successful strategies to other regions by 20%.")
    else:
        actions.append("2. **Top Performers**: Capture and scale winning practices network-wide.")
    
    if bottom_dept:
        actions.append(f"3. **{bottom_dept}**: Optimize inventory and placement to improve sales by 15%.")
    elif last_wow < -5:
        actions.append("3. **Revenue Decline**: Implement immediate recovery plan to reverse trend.")
    else:
        actions.append("3. **Growth Focus**: Test new promotional strategies in key locations.")
    
    return "\n".join(actions)