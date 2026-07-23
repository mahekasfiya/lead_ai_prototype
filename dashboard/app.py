from __future__ import annotations

from datetime import datetime

import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"

READINESS_URL = f"{API_BASE_URL}/readiness"
DISCOVER_LEADS_URL = f"{API_BASE_URL}/discover-leads"


st.set_page_config(
    page_title="Triway Lead Intelligence",
    page_icon="🔎",
    layout="wide",
)

def generate_email_for_lead(lead_data: dict) -> str:
    """Call the /generate-email endpoint and return the draft."""
    try:
        payload = {
            "lead": lead_data,
            "matched_services": lead_data.get("matched_services", [])
        }
        response = requests.post(
            f"{API_BASE_URL}/generate-email",
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get("email_draft", "No draft returned.")
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Failed to generate email: {str(e)}"

def check_api() -> tuple[bool, dict]:
    try:
        response = requests.get(READINESS_URL, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, {"status_code": response.status_code, "error": response.text}
    except requests.RequestException as exc:
        return False, {"error": str(exc)}

def discover_leads(payload: dict) -> dict:
    response = requests.post(
        DISCOVER_LEADS_URL,
        json=payload,
        timeout=(15, 1200),
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Lead discovery failed with status {response.status_code}: {response.text}"
        )
    return response.json()

def confidence_badge(confidence: str | None) -> str:
    normalized = str(confidence or "").lower()
    if normalized == "high":
        return "🟢 High"
    if normalized == "medium":
        return "🟡 Medium"
    return "🔴 Low"

st.title("Triway Lead Intelligence")
st.caption("Search the web for organizations showing demand for services offered by Triway Technologies.")

api_ready, readiness = check_api()
if not api_ready:
    st.error("The FastAPI backend is not available.")
    st.code("python -m uvicorn module_3.main:app --reload")
    with st.expander("Connection details"):
        st.json(readiness)
    st.stop()

st.success(f"Backend connected successfully. {readiness.get('service_count', 0)} services loaded.")

# Sidebar
with st.sidebar:
    st.header("Dashboard")
    developer_mode = st.toggle("Developer Mode", value=False)
    if developer_mode:
        st.divider()
        st.subheader("Discovery Settings")
        max_queries = st.slider("Maximum queries", 1, 50, 5)
        results_per_query = st.slider("Results per query", 1, 10, 5)
        max_leads = st.slider("Maximum leads", 1, 50, 20)
        minimum_similarity = st.slider("Minimum similarity", 0.0, 1.0, 0.25, 0.05)
    else:
        max_queries = 10
        results_per_query = 5
        max_leads = 20
        minimum_similarity = 0.25

    st.divider()
    st.subheader("System Status")
    st.write(f"**Provider:** {readiness.get('provider', '-')}")
    st.write(f"**Model:** {readiness.get('model', '-')}")
    st.write(f"**Services loaded:** {readiness.get('service_count', '-')}")
    if developer_mode:
        st.write(f"**Embedding version:** {readiness.get('embedding_version', '-')}")

st.subheader("Sales Opportunities")
st.write("Run a fresh scan for organizations currently showing buying intent for Triway services.")

last_scan = st.session_state.get("last_scan_time")
st.caption(f"Last scan: {last_scan}" if last_scan else "No scan has been run in this session.")

generate_clicked = st.button("Generate New Opportunities", type="primary", use_container_width=True)

# --- Run a new scan only when the button is clicked. This block ONLY
# stores the result in session_state; it never renders results itself.
if generate_clicked:
    payload = {
        "max_queries": max_queries,
        "results_per_query": results_per_query,
        "max_leads": max_leads,
        "minimum_similarity": minimum_similarity,
        "selected_service_ids": [],
    }
    with st.spinner("Scanning for current buying opportunities, validating sources, and matching Triway services..."):
        try:
            result = discover_leads(payload)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        st.session_state["latest_result"] = result
        st.session_state["last_scan_time"] = datetime.now().strftime("%d %b %Y, %I:%M %p")

# --- Render results from session_state on EVERY rerun (including the
# rerun triggered by clicking "Generate Email"), not just right after a scan.
if "latest_result" in st.session_state:
    result = st.session_state["latest_result"]

    st.success(f"Discovery complete. {result.get('leads_found', 0)} validated opportunities found.")

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Queries Executed", len(result.get("queries_executed", [])))
    col2.metric("Sources Collected", result.get("sources_collected", 0))
    col3.metric("Sources Analyzed", result.get("sources_analyzed", 0))
    col4.metric("Leads Found", result.get("leads_found", 0))
    col5.metric("Manual Review", result.get("manual_review_count", 0))

    # Queries
    with st.expander("Search Queries Executed"):
        for query in result.get("queries_executed", []):
            st.write(f"- {query}")

    leads = result.get("leads", [])
    st.subheader("Validated Opportunities")
    if not leads:
        st.info("No currently valid opportunities passed final validation in this scan.")

    # Display each lead
    for idx, lead in enumerate(leads, start=1):
        top_percentage = lead.get("top_service_match_percentage") or 0.0
        draft_key = f"email_draft_{lead.get('source_url')}"
        # Keep the expander open if it's the first one, or if a draft
        # already exists for this lead.
        is_expanded = (idx == 1) or (draft_key in st.session_state and st.session_state[draft_key])

        title = (
            f"#{idx} "
            f"{lead.get('company_name') or lead.get('source_title', 'Untitled opportunity')} "
            f"— {top_percentage:.2f}% match"
        )

        with st.expander(title, expanded=is_expanded):
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Top Service Match", f"{top_percentage:.2f}%")
            col2.metric("Matched Service", lead.get("top_service_name") or "Unknown")
            confidence = lead.get("qualification", {}).get("confidence", 0.0)
            col3.metric("Confidence", confidence_badge(confidence))

            if developer_mode:
                st.write(f"**Search Query:** {lead.get('search_query', '-')}")

            st.write(f"**Company:** {lead.get('company_name') or 'Unknown'}")
            st.write(f"**Industry:** {lead.get('industry') or 'Unknown'}")
            st.write(f"**Country:** {lead.get('country') or 'Unknown'}")
            snippet = lead.get("source_snippet")
            if snippet:
                st.write(f"**Search Snippet:** {snippet}")
            source_url = lead.get("source_url")
            if source_url:
                st.link_button("Open Source", source_url)

            # Email generation
            if st.button("📧 Generate Email", key=f"email_btn_{idx}"):
                with st.spinner("Generating email draft..."):
                    draft = generate_email_for_lead(lead)
                    st.session_state[draft_key] = draft

            if draft_key in st.session_state:
                st.markdown("### 📧 Email Draft")
                st.text_area(
                    "Draft",
                    st.session_state[draft_key],
                    height=300,
                    key=f"email_text_{idx}"
                )

            # Matched services
            matched_services = lead.get("matched_services", [])
            if matched_services:
                st.markdown("### Matched Triway Services")
                for match in matched_services:
                    service_match = match.get("service_match_percentage") or match.get("similarity_percentage", 0.0)
                    semantic = match.get("similarity_percentage", 0.0)

                    st.markdown(f"#### #{match.get('rank')} {match.get('service_name')}")
                    sc1, sc2 = st.columns(2)
                    sc1.metric("Service Match Score", f"{service_match:.2f}%")
                    sc2.metric("Semantic Similarity", f"{semantic:.2f}%")
                    st.write(f"**Category:** {match.get('category', '-')}")
                    st.write(f"**Explanation:** {match.get('explanation', '-')}")

                    evidence = match.get("evidence", {})
                    evidence_values = []
                    for field_name, values in evidence.items():
                        if not values:
                            continue
                        readable = field_name.replace("_", " ").title()
                        evidence_values.append((readable, values))
                    if evidence_values:
                        with st.expander("View Match Evidence"):
                            for readable, values in evidence_values:
                                st.write(f"**{readable}:**")
                                for v in values:
                                    st.write(f"- {v}")
                    st.divider()

    # Manual review section
    manual_review = result.get("manual_review", [])
    st.subheader("Manual Review")
    if not manual_review:
        st.info("No opportunities require manual review in this scan.")
    else:
        for idx, item in enumerate(manual_review, start=1):
            review_type = item.get("review_type", "manual").replace("_", " ").title()
            review_title = item.get("company_name") or item.get("source_title") or "Untitled opportunity"
            suggested_sim = item.get("suggested_similarity")
            if suggested_sim is not None:
                expander_title = f"#{idx} {review_title} — {review_type} ({suggested_sim:.2f}% suggested match)"
            else:
                expander_title = f"#{idx} {review_title} — {review_type}"
            with st.expander(expander_title, expanded=False):
                st.warning(item.get("reason", "This opportunity requires manual review."))
                rc1, rc2 = st.columns(2)
                rc1.write(f"**Review Type:** {review_type}")
                rc2.write(f"**Suggested Service:** {item.get('suggested_service_name') or 'Not available'}")
                if developer_mode:
                    st.write(f"**Search Query:** {item.get('search_query', '-')}")
                st.write(f"**Company:** {item.get('company_name') or 'Unknown'}")
                st.write(f"**Industry:** {item.get('industry') or 'Unknown'}")
                st.write(f"**Country:** {item.get('country') or 'Unknown'}")
                if item.get("source_snippet"):
                    st.write(f"**Search Snippet:** {item.get('source_snippet')}")
                if item.get("suggested_service_id"):
                    st.write(f"**Suggested Service ID:** {item.get('suggested_service_id')}")
                if suggested_sim is not None:
                    st.metric("Suggested Similarity", f"{suggested_sim:.2f}%")
                if item.get("source_url"):
                    st.link_button("Open Source for Review", item.get("source_url"))

    # Raw response
    with st.expander("Raw Discovery Response"):
        st.json(result)