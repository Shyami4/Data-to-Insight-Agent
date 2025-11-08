# narrative.py - FIXED VERSION
import os, json
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import logging

load_dotenv()

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _has_key():
    """Check if OpenAI key is available"""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logger.warning("OpenAI API key not found in environment variables")
        return False
    if len(key) < 10:
        logger.warning(f"OpenAI API key appears invalid (length: {len(key)})")
        return False
    return True

def _safe_client():
    """Get OpenAI client or None with better error handling"""
    if not _has_key():
        logger.warning("Cannot create OpenAI client - API key not available")
        return None
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Test the client with a simple call
        return client
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {str(e)}")
        return None

def _test_openai_connection(client):
    """Test if OpenAI connection is working"""
    if not client:
        return False
    try:
        # Simple test call
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        return True
    except Exception as e:
        logger.error(f"OpenAI connection test failed: {str(e)}")
        return False

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
    if not client or not _test_openai_connection(client):
        logger.info(f"Using fallback for micro insight: {topic}")
        return _strategic_fallback(context, topic)
    
    try:
        # Enhanced prompt with strategic framing
        prompt = f"""
BUSINESS CONTEXT:
You're analyzing {topic} for a retail executive who needs to make decisions THIS WEEK.

DATA:
{json.dumps(context, indent=2, default=str)}

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
            max_tokens=100,  # Force brevity
            timeout=10  # Add timeout
        )
        
        result = resp.choices[0].message.content.strip()
        logger.info(f"AI micro insight generated successfully for {topic}")
        return result
        
    except Exception as e:
        logger.error(f"Error generating micro insight for {topic}: {str(e)}")
        return _strategic_fallback(context, topic)

def _strategic_fallback(context: dict, topic: str) -> str:
    """Generate strategic insights without AI"""
    
    # Sales Trend Analysis
    if "current_value" in context and "previous_value" in context:
        current = context["current_value"]
        previous = context["previous_value"]
        if previous and previous != 0:
            change = current - previous
            pct_change = (change / previous * 100)
            
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
        top_store = context["leaders"][0] if context["leaders"] else "Top Store"
        top_value = context["values"][0] if context["values"] else 0
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

def draft_page_summary(ctx: dict) -> str:
    """Main page summary function with improved error handling"""
    client = _safe_client()
    
    if not client or not _test_openai_connection(client):
        logger.info("Using fallback page summary - AI not available")
        return _fallback_page_summary(ctx)
    
    try:
        # Clean and prepare context
        clean_ctx = _clean_context_for_api(ctx)
        
        prompt = f"""
EXECUTIVE DASHBOARD CONTEXT:
{json.dumps(clean_ctx, indent=2, default=str)}

Generate an executive summary following this EXACT format:

**Momentum**
• [Trend insight with percentage] — [what this means for business]
• [Growth pulse insight] — [immediate action needed]

**Store Pulse**
• [X/Y stores growing/declining] — [specific insight]
• [Best/worst performer with numbers] — [action item]
• [Store-level insight] — [tactical recommendation]

**Ranking & Benchmarks**
• [Top performer insight] — [scale/leverage opportunity]
• [Bottom performer insight] — [improvement focus]
• [Competitive positioning] — [strategic move]

**Next 7 Days — Do This**
1. [Specific entity]: [Specific action with target percentage]
2. [Specific entity]: [Specific action with target percentage]
3. [Specific entity]: [Specific action with target percentage]

Requirements:
- Use exact numbers from the data
- Name specific stores/departments from context
- Keep bullets 25-30 words maximum
- Include percentage targets in actions
- Be actionable, not descriptive
"""
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PAGE_SUMMARY},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=400,
            timeout=15
        )
        
        result = resp.choices[0].message.content.strip()
        logger.info("AI page summary generated successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error generating AI page summary: {str(e)}")
        return _fallback_page_summary(ctx)

def _clean_context_for_api(ctx: dict) -> dict:
    """Clean context data for API calls"""
    clean_ctx = {}
    
    for key, value in ctx.items():
        if value is None:
            continue
        
        if isinstance(value, (np.integer, np.floating)):
            clean_ctx[key] = float(value)
        elif isinstance(value, dict):
            clean_dict = {}
            for k, v in value.items():
                if v is not None:
                    if isinstance(v, (np.integer, np.floating)):
                        clean_dict[k] = float(v)
                    elif not isinstance(v, (list, dict)):
                        clean_dict[k] = str(v)
                    else:
                        clean_dict[k] = v
            if clean_dict:
                clean_ctx[key] = clean_dict
        elif isinstance(value, list):
            clean_ctx[key] = [str(v) for v in value if v is not None]
        else:
            clean_ctx[key] = value
    
    return clean_ctx

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
        else:
            lines.append(f"• {d}/{total} stores declining — urgent triage required")
            
        # Top performers
        best = sp.get("best_store", "Store_A")
        best_pct = sp.get("best_store_pct", 0)
        if best_pct > 10:
            lines.append(f"• {best} +{best_pct:.1f}% — extract and deploy playbook")
        
        # Bottom performers  
        worst = sp.get("worst_store", "Store_Z")
        worst_pct = sp.get("worst_store_pct", 0)
        if worst_pct < -10:
            lines.append(f"• {worst} {worst_pct:.1f}% — diagnose and fix this week")

    # 3) Benchmarks - Competitive positioning
    lines.append("")
    lines.append("**Ranking & Benchmarks**")
    
    if s_b:
        top_store = s_b.get("top", "Unknown")
        bottom_store = s_b.get("bottom", "Unknown")
        lines.append(f"• {top_store} leads — extract and deploy playbook")
        if bottom_store != top_store:
            lines.append(f"• {bottom_store} lags — implement recovery plan")
    
    if d_b:
        top_dept = d_b.get("top", "Unknown")
        lines.append(f"• {top_dept} department drives growth — increase allocation 15%")

    # 4) Next 7 Days Actions
    lines.append("")
    lines.append("**Next 7 Days — Do This**")
    
    # Use the best available entity names
    action_entity_1 = worst or bottom_store or "Underperformers"
    action_entity_2 = best or top_store or "Top Performers" 
    action_entity_3 = d_b.get("top", "Revenue Drivers") if d_b else "Growth Focus"
    
    lines.append(f"1. **{action_entity_1}**: Implement recovery plan to improve sales by 15%")
    lines.append(f"2. **{action_entity_2}**: Document best practices for network rollout")
    lines.append(f"3. **{action_entity_3}**: Increase promotional focus to boost performance 10%")

    return "\n".join(lines)

def draft_drivers_summary(ctx: dict) -> str:
    """Drivers page summary with improved error handling"""
    client = _safe_client()
    
    if not client or not _test_openai_connection(client):
        logger.info("Using fallback drivers summary - AI not available")
        return _fallback_drivers_summary(ctx)
    
    try:
        clean_ctx = _clean_context_for_api(ctx)
        
        prompt = f"""
DRIVERS & PERFORMANCE ANALYSIS CONTEXT:
{json.dumps(clean_ctx, indent=2, default=str)}

Generate a drivers analysis following this format:

**Momentum**
• [Trend with specific percentage] — [action required]
• [Secondary trend insight] — [strategic implication]

**Store Pulse**  
• [X stores performing above/below target] — [intervention needed]
• [Top performer with metrics] — [scale opportunity]
• [Bottom performer with metrics] — [recovery action]

**Ranking & Benchmarks**
• [Regional insight with specifics] — [resource reallocation]
• [Department concentration insight] — [portfolio action]

**Next 7 Days — Do This**
1. **[Entity]**: [Specific action with percentage target]
2. **[Entity]**: [Specific action with percentage target] 
3. **[Entity]**: [Specific action with percentage target]

Requirements:
- Use exact numbers from the data
- Name specific stores/departments from context
- Keep bullets 25-30 words maximum
- Include percentage targets in actions
- Be actionable, not descriptive
"""
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a retail performance analyst. Focus on drivers, not just descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=400,
            timeout=15
        )
        
        result = resp.choices[0].message.content.strip()
        logger.info("AI drivers summary generated successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error generating AI drivers summary: {str(e)}")
        return _fallback_drivers_summary(ctx)

def _fallback_drivers_summary(ctx: dict) -> str:
    """Fallback drivers summary"""
    lines = []
    
    # Extract key metrics
    momentum = ctx.get("momentum", {})
    drivers = ctx.get("drivers", {})
    
    # Momentum section
    lines.append("**Momentum**")
    lines.append("")
    
    current_sales = momentum.get("current_sales")
    if current_sales:
        lines.append(f"• Current sales: ${current_sales:,.0f} — monitor weekly trends")
    else:
        lines.append("• Sales trend analysis required — establish baseline metrics")
    
    lines.append("• Growth initiatives needed — test new promotional strategies")
    
    # Store Pulse
    lines.append("")
    lines.append("**Store Pulse**")
    lines.append("• Performance review required — identify top and bottom quartiles")
    lines.append("• Operational efficiency analysis needed — optimize underperformers")
    
    # Regional insights
    regional = drivers.get("regional", {}) if drivers else {}
    if regional.get("top") and regional.get("bottom"):
        top_region = regional["top"]
        bottom_region = regional["bottom"]
        lines.append(f"• {top_region} outperforms {bottom_region} — reallocate resources")
    
    # Benchmarks
    lines.append("")
    lines.append("**Ranking & Benchmarks**")
    
    # Department insights
    departments = drivers.get("departments", {}) if drivers else {}
    if departments.get("top"):
        top_dept = departments["top"]
        lines.append(f"• {top_dept} leads performance — increase investment 15%")
    else:
        lines.append("• Department analysis needed — identify growth categories")
    
    lines.append("• Competitive benchmarking required — establish market position")
    
    # Actions
    lines.append("")
    lines.append("**Next 7 Days — Do This**")
    
    entity1 = regional.get("bottom", "Underperforming Region") if regional else "Low Performers"
    entity2 = departments.get("top", "Top Department") if departments else "Growth Drivers"
    entity3 = regional.get("top", "Leading Region") if regional else "Best Practices"
    
    lines.append(f"1. **{entity1}**: Launch targeted improvement initiatives for 10% uplift")
    lines.append(f"2. **{entity2}**: Expand successful strategies to increase share 15%")
    lines.append(f"3. **{entity3}**: Document and replicate winning approaches")
    
    return "\n".join(lines)

# Enhanced strategic actions generation
def _generate_strategic_actions(ctx: dict, df, result: dict) -> str:
    """Generate strategic actions with better error handling"""
    client = _safe_client()
    
    if not client or not _test_openai_connection(client):
        logger.info("Using fallback strategic actions - AI not available")
        return _fallback_strategic_actions(ctx, result)
    
    try:
        clean_ctx = _clean_context_for_api(ctx)
        
        prompt = f"""
Create strategic recommendations with exactly 5 detailed actions per priority level.

COMPLETE CONTEXT:
{json.dumps(clean_ctx, indent=2, default=str)}

Generate strategic actions organized by priority level:

**IMMEDIATE ACTIONS (Next 7 Days)**
1. **[Entity]**: Specific action with timeline and expected impact (25-30 words max)
2. **[Entity]**: Specific action with timeline and expected impact (25-30 words max)
3. **[Entity]**: Specific action with timeline and expected impact (25-30 words max)
4. **[Entity]**: Specific action with timeline and expected impact (25-30 words max)
5. **[Entity]**: Specific action with timeline and expected impact (25-30 words max)

**HIGH PRIORITY (Next 30 Days)**
1. **[Entity]**: Strategic initiative with implementation steps (25-30 words max)
2. **[Entity]**: Strategic initiative with implementation steps (25-30 words max)
3. **[Entity]**: Strategic initiative with implementation steps (25-30 words max)

Requirements:
- Reference specific entities from the data (stores, departments, regions)
- Include quantified targets (percentages, dollar amounts)
- Keep each bullet to 25-30 words maximum
- Focus on actionable steps with clear outcomes
- Use actual data from context for concrete recommendations
"""
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior retail strategy consultant. Create concise, actionable strategic recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=400,
            timeout=15
        )
        
        result = resp.choices[0].message.content.strip()
        logger.info("AI strategic actions generated successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error generating AI strategic actions: {str(e)}")
        return _fallback_strategic_actions(ctx, result)

def _fallback_strategic_actions(ctx: dict, result: dict) -> str:
    """Fallback strategic actions"""
    actions = []
    
    # Get key entities and metrics from context
    drivers = ctx.get("drivers", {})
    growth_pulse = ctx.get("growth_pulse", {})
    momentum = ctx.get("momentum", {})
    store_pulse = ctx.get("store_pulse", {})
    
    # Extract regional and department data
    regional = drivers.get("regional", {}) if drivers else {}
    departments = drivers.get("departments", {}) if drivers else {}
    
    top_region = regional.get("top", "Leading Region")
    bottom_region = regional.get("bottom", "Underperforming Region")
    top_dept = departments.get("top", "Top Department")
    bottom_dept = departments.get("bottom", "Underperforming Department")
    
    # IMMEDIATE ACTIONS
    actions.append("**IMMEDIATE ACTIONS (Next 7 Days)**")
    actions.append("")
    actions.append(f"1. **Sales Team**: Conduct a sales performance review to identify top-performing departments and regions for targeted promotions.")
    actions.append(f"2. **Marketing**: Launch a targeted campaign for Dept_3 to increase allocation 15%, extend promotions")
    actions.append(f"3. **Inventory Management**: Optimize stock levels in the East region to address low sales, aiming for a 5% increase.")
    actions.append(f"4. **Customer Engagement**: Implement a customer feedback survey in all active stores to enhance service quality and satisfaction.")
    actions.append(f"5. **Data Analysis**: Analyze sales data for the last month to identify trends and adjust strategies accordingly.")
    
    # HIGH PRIORITY
    actions.append("")
    actions.append("**HIGH PRIORITY (Next 30 Days)**")
    actions.append("")
    actions.append(f"1. **Regional Strategy**: Develop a tailored marketing strategy for the East region to boost sales by 15% within 30 days.")
    actions.append(f"2. **Department Focus**: Increase promotional efforts for Dept_1, aiming for a 20% sales uplift through targeted discounts.")
    actions.append(f"3. **Sales Training**: Implement a training program for store staff focused on upselling techniques to improve average transaction value by 5%.")
    
    return "\n".join(actions)

# Additional utility functions
def _calculate_dept_concentration(departments: dict) -> float:
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