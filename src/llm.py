"""
LLM layer — generates plain-English insights from forecast outputs.

Three functions, one Gemini API call each:
  1. explain_forecast()   — what the numbers mean and why
  2. interpret_anomalies() — what each flagged anomaly likely means
  3. flag_risks()          — operational risks and mitigations

All functions accept pre-built context dicts so they can be called
independently or together via get_insights().
"""

import json
import os
from google import genai
from google.genai import types

# Default model; override with NETEXLIR_MODEL env var
_DEFAULT_MODEL = os.getenv("NETEXLIR_MODEL", "gemini-3.5-flash")
_MAX_TOKENS = 600


def _client() -> genai.Client:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Export it before running: export GOOGLE_API_KEY=..."
        )
    return genai.Client(api_key=key)


def _call(system: str, user: str, model: str = _DEFAULT_MODEL) -> str:
    response = _client().models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=_MAX_TOKENS,
        ),
    )
    return response.text.strip()


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_forecast_context(
    forecast_result: dict,
    trailing: dict,
    channel_results: dict | None = None,
) -> str:
    """Serialize forecast data to a compact JSON string for the LLM."""
    rev = forecast_result["revenue_forecasts"]
    spd = forecast_result["spend_forecasts"]
    roas = forecast_result["roas_forecasts"]

    payload = {
        "trailing_30d": {
            "revenue": round(trailing["revenue"]),
            "spend": round(trailing["spend"]),
            "roas": round(trailing["roas"], 2),
            "period": f"{trailing['start'].date()} to {trailing['end'].date()}",
        },
        "forecast_windows": [
            {
                "days": rev[i]["days"],
                "revenue_point": round(rev[i]["point"]),
                "revenue_lower": round(rev[i]["lower"]),
                "revenue_upper": round(rev[i]["upper"]),
                "spend_point": round(spd[i]["point"]),
                "roas_point": round(roas[i]["point"], 2),
                "roas_lower": round(roas[i]["lower"], 2),
                "roas_upper": round(roas[i]["upper"], 2),
                "interval_width_pct": round(forecast_result["interval_width"] * 100),
            }
            for i in range(len(rev))
        ],
    }

    if channel_results:
        payload["channel_30d"] = {
            ch: {
                "revenue": round(r["revenue_forecasts"][0]["point"]),
                "roas": round(r["roas_forecasts"][0]["point"], 2),
            }
            for ch, r in channel_results.items()
        }

    return json.dumps(payload, indent=2)


def _build_budget_context(sim_result: dict) -> str:
    payload = {
        "daily_budgets_usd": sim_result["budget_by_channel"],
        "total_daily_budget": sim_result["total_daily_budget"],
        "portfolio_forecast": {
            days: {
                "revenue_point": round(p["revenue_point"]),
                "revenue_lower": round(p["revenue_lower"]),
                "revenue_upper": round(p["revenue_upper"]),
                "roas_point": round(p["roas_point"], 2),
            }
            for days, p in sim_result["portfolio"].items()
        },
        "channel_30d": {
            ch: {
                "revenue": round(r["revenue_forecasts"][0]["point"]),
                "budget_30d": round(sim_result["budget_by_channel"].get(ch, 0) * 30),
            }
            for ch, r in sim_result["channel_results"].items()
        },
    }
    return json.dumps(payload, indent=2)


# ── Three core LLM functions ───────────────────────────────────────────────────

def explain_forecast(
    forecast_result: dict,
    trailing: dict,
    channel_results: dict | None = None,
    model: str = _DEFAULT_MODEL,
) -> str:
    """
    Generate a plain-English explanation of the 30/60/90-day forecast
    for a non-technical marketing manager.
    """
    context = _build_forecast_context(forecast_result, trailing, channel_results)

    system = (
        "You are a senior digital marketing analyst. "
        "You receive structured ad performance forecast data and explain it clearly "
        "to a non-technical marketing manager. Be concise, direct, and avoid jargon. "
        "Focus on what the numbers mean for business decisions. "
        "Do not repeat raw numbers from the JSON — summarize and interpret them."
    )

    user = f"""Here is the 30/60/90-day revenue and ROAS forecast for this account:

{context}

Write a 4–6 sentence plain-English explanation covering:
1. Whether revenue is expected to grow, decline, or hold steady vs the trailing 30 days, and by roughly how much.
2. The confidence in the forecast (reference the prediction interval width).
3. Which channel is driving the most revenue and whether that concentration is a concern.
4. One key takeaway for the marketing manager.

Keep it under 150 words."""

    return _call(system, user, model)


def explain_budget_simulation(
    sim_result: dict,
    baseline_forecast: dict,
    model: str = _DEFAULT_MODEL,
) -> str:
    """
    Explain what the budget simulation predicts, including the diminishing-returns effect.
    """
    sim_ctx = _build_budget_context(sim_result)
    base_rev_30 = baseline_forecast["revenue_forecasts"][0]["point"]
    base_roas_30 = baseline_forecast["roas_forecasts"][0]["point"]

    system = (
        "You are a digital marketing analyst specialising in paid media ROI. "
        "Explain budget simulation results simply. Be honest about diminishing returns."
    )

    user = f"""A user has entered the following future media budget and requested a revenue simulation:

{sim_ctx}

Baseline forecast (current spend trajectory): 30d revenue ≈ ${base_rev_30:,.0f}, ROAS ≈ {base_roas_30:.2f}x.

Write 3–5 sentences explaining:
1. How the proposed budget compares to current spend trajectory.
2. What revenue uplift (or decline) is expected, referencing the prediction range.
3. Whether the ROAS is improving or deteriorating, and what that implies about diminishing returns.
Keep it under 120 words."""

    return _call(system, user, model)


def interpret_anomalies(
    anomaly_list: list[dict],
    trailing: dict,
    model: str = _DEFAULT_MODEL,
) -> str:
    """
    For each detected anomaly, provide a plain-English likely cause and whether it warrants action.
    Returns empty string if no anomalies.
    """
    if not anomaly_list:
        return "No significant anomalies detected in the recent data."

    top = anomaly_list[:6]  # cap to avoid token overflow
    anomaly_json = json.dumps(top, indent=2)

    system = (
        "You are a digital marketing data analyst. "
        "You identify likely causes of unusual patterns in ad performance data "
        "and give clear, actionable commentary."
    )

    user = f"""The following anomalies were detected in recent ad performance data
(trailing context: 30d revenue ${trailing['revenue']:,.0f}, ROAS {trailing['roas']:.2f}x):

{anomaly_json}

For each anomaly:
- Give 1–2 sentences with the most likely cause (e.g. seasonality, budget change, tracking issue, promo event).
- State whether it needs investigation or is likely benign.

Format as a bullet list. Under 180 words total."""

    return _call(system, user, model)


def flag_risks(
    forecast_result: dict,
    trailing: dict,
    anomaly_list: list[dict],
    channel_results: dict | None = None,
    budget_by_channel: dict | None = None,
    model: str = _DEFAULT_MODEL,
) -> str:
    """
    Identify 3–5 operational risks and suggest mitigations.
    """
    context = _build_forecast_context(forecast_result, trailing, channel_results)

    risk_context = {
        "forecast_summary": json.loads(context),
        "high_severity_anomalies": [a for a in anomaly_list if a["severity"] == "high"][:4],
        "channel_concentration": {
            ch: round(r["revenue_forecasts"][0]["point"])
            for ch, r in (channel_results or {}).items()
        },
        "budget_by_channel": budget_by_channel or {},
    }

    system = (
        "You are a marketing risk analyst. You review paid media forecasts and flag "
        "operational risks with clear, prioritised mitigations. Be concise and specific."
    )

    user = f"""Review the following forecast and context for operational risks:

{json.dumps(risk_context, indent=2)}

List exactly 4 risks. For each:
- Risk title (5 words max)
- 1-sentence description
- 1-sentence mitigation

Format:
**Risk 1: [title]**
[description] → [mitigation]

Under 200 words total."""

    return _call(system, user, model)


# ── Unified entry point ────────────────────────────────────────────────────────

def get_insights(
    forecast_result: dict,
    trailing: dict,
    anomaly_list: list[dict],
    channel_results: dict | None = None,
    sim_result: dict | None = None,
    budget_by_channel: dict | None = None,
    model: str = _DEFAULT_MODEL,
) -> dict:
    """
    Run all three LLM calls and return a dict with keys:
      forecast_explanation, anomaly_interpretation, risk_flags,
      and optionally budget_explanation (if sim_result provided).

    Makes 3–4 API calls sequentially. Each is independent — partial
    results are returned even if one call fails.
    """
    results = {}

    try:
        results["forecast_explanation"] = explain_forecast(
            forecast_result, trailing, channel_results, model=model
        )
    except Exception as e:
        results["forecast_explanation"] = f"[Error generating forecast explanation: {e}]"

    try:
        results["anomaly_interpretation"] = interpret_anomalies(
            anomaly_list, trailing, model=model
        )
    except Exception as e:
        results["anomaly_interpretation"] = f"[Error generating anomaly interpretation: {e}]"

    try:
        results["risk_flags"] = flag_risks(
            forecast_result, trailing, anomaly_list,
            channel_results=channel_results,
            budget_by_channel=budget_by_channel,
            model=model,
        )
    except Exception as e:
        results["risk_flags"] = f"[Error generating risk flags: {e}]"

    if sim_result:
        try:
            results["budget_explanation"] = explain_budget_simulation(
                sim_result, forecast_result, model=model
            )
        except Exception as e:
            results["budget_explanation"] = f"[Error generating budget explanation: {e}]"

    return results


def get_insights_dry_run(
    forecast_result: dict,
    trailing: dict,
    anomaly_list: list[dict],
    channel_results: dict | None = None,
    sim_result: dict | None = None,
    budget_by_channel: dict | None = None,
) -> dict:
    """
    Return the exact prompts that would be sent to the LLM, without making API calls.
    Useful for testing and UI previewing when no API key is available.
    """
    context = _build_forecast_context(forecast_result, trailing, channel_results)
    return {
        "forecast_explanation_prompt": context,
        "anomaly_count": len(anomaly_list),
        "top_anomalies": anomaly_list[:4],
        "model": _DEFAULT_MODEL,
        "note": "Dry run — set GOOGLE_API_KEY and call get_insights() for real output.",
    }
