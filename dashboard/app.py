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


def check_api() -> tuple[bool, dict]:
    """Check whether the FastAPI backend is ready."""
    try:
        response = requests.get(
            READINESS_URL,
            timeout=10,
        )

        if response.status_code == 200:
            return True, response.json()

        return False, {
            "status_code": response.status_code,
            "error": response.text,
        }

    except requests.RequestException as exc:
        return False, {
            "error": str(exc),
        }


def discover_leads(payload: dict) -> dict:
    """Call the FastAPI lead-discovery endpoint."""
    response = requests.post(
        DISCOVER_LEADS_URL,
        json=payload,
        timeout=(15, 1800),
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Lead discovery failed with status "
            f"{response.status_code}: {response.text}"
        )

    return response.json()


def confidence_badge(confidence: str | None) -> str:
    """Format confidence for display."""
    normalized = str(confidence or "").lower()

    if normalized == "high":
        return "🟢 High"

    if normalized == "medium":
        return "🟡 Medium"

    return "🔴 Low"


def display_metrics(result: dict) -> None:
    """Display top-level discovery metrics."""
    (
        metric_col1,
        metric_col2,
        metric_col3,
        metric_col4,
        metric_col5,
    ) = st.columns(5)

    metric_col1.metric(
        "Queries Executed",
        len(result.get("queries_executed", [])),
    )

    metric_col2.metric(
        "Sources Collected",
        result.get("sources_collected", 0),
    )

    metric_col3.metric(
        "Sources Analyzed",
        result.get("sources_analyzed", 0),
    )

    metric_col4.metric(
        "Validated Leads",
        result.get("leads_found", 0),
    )

    metric_col5.metric(
        "Manual Review",
        result.get("manual_review_count", 0),
    )


def display_validated_leads(
    leads: list[dict],
    developer_mode: bool,
) -> None:
    """Render validated opportunities."""
    if not leads:
        st.info(
            "No currently valid opportunities passed final validation "
            "in this scan."
        )
        return

    st.caption(
        "These opportunities passed source validation, qualification, "
        "service matching, and final validation."
    )

    for index, lead in enumerate(leads, start=1):
        top_percentage = (
            lead.get("top_service_match_percentage")
            or 0.0
        )

        title = (
            f"#{index} "
            f"{lead.get('company_name') or lead.get('source_title', 'Untitled opportunity')} "
            f"— {top_percentage:.2f}% match"
        )

        with st.expander(
            title,
            expanded=index == 1,
        ):
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Top Service Match",
                f"{top_percentage:.2f}%",
            )

            col2.metric(
                "Matched Service",
                lead.get("top_service_name") or "Unknown",
            )

            matched_services = lead.get(
                "matched_services",
                [],
            )

            top_match = (
                matched_services[0]
                if matched_services
                else {}
            )

            confidence = top_match.get(
                "service_match_confidence",
                top_match.get("confidence", "Low"),
            )

            col3.metric(
                "Confidence",
                confidence_badge(confidence),
            )

            info_col1, info_col2, info_col3 = st.columns(3)

            info_col1.write(
                f"**Company:** "
                f"{lead.get('company_name') or 'Unknown'}"
            )

            info_col2.write(
                f"**Industry:** "
                f"{lead.get('industry') or 'Unknown'}"
            )

            info_col3.write(
                f"**Country:** "
                f"{lead.get('country') or 'Unknown'}"
            )

            if developer_mode:
                st.write(
                    f"**Search Query:** "
                    f"{lead.get('search_query', '-')}"
                )

            snippet = lead.get("source_snippet")

            if snippet:
                st.markdown("**Opportunity Summary**")
                st.write(snippet)

            source_url = lead.get("source_url")

            if source_url:
                st.link_button(
                    "Open Source",
                    source_url,
                )

            if matched_services:
                st.markdown("### Matched Triway Services")

            for match in matched_services:
                service_match_percentage = (
                    match.get("service_match_percentage")
                    or match.get("similarity_percentage", 0.0)
                )

                semantic_percentage = match.get(
                    "similarity_percentage",
                    0.0,
                )

                st.markdown(
                    f"#### #{match.get('rank')} "
                    f"{match.get('service_name')}"
                )

                service_col1, service_col2, service_col3 = st.columns(3)

                service_col1.metric(
                    "Service Match",
                    f"{service_match_percentage:.2f}%",
                )

                service_col2.metric(
                    "Semantic Similarity",
                    f"{semantic_percentage:.2f}%",
                )

                service_col3.metric(
                    "Confidence",
                    confidence_badge(
                        match.get(
                            "service_match_confidence",
                            match.get("confidence"),
                        )
                    ),
                )

                st.write(
                    f"**Category:** "
                    f"{match.get('category', '-')}"
                )

                st.write(
                    f"**Explanation:** "
                    f"{match.get('explanation', '-')}"
                )

                evidence = match.get("evidence", {})
                evidence_values = []

                for field_name, values in evidence.items():
                    if not values:
                        continue

                    readable_name = (
                        field_name
                        .replace("_", " ")
                        .title()
                    )

                    evidence_values.append(
                        (readable_name, values)
                    )

                if evidence_values:
                    with st.expander("View Match Evidence"):
                        for readable_name, values in evidence_values:
                            st.write(f"**{readable_name}:**")

                            for value in values:
                                st.write(f"- {value}")

                if developer_mode:
                    score_breakdown = match.get("score_breakdown")

                    if score_breakdown:
                        with st.expander("View Score Breakdown"):
                            st.json(score_breakdown)

                st.divider()


def display_manual_review(
    manual_review: list[dict],
    developer_mode: bool,
) -> None:
    """Render opportunities requiring human validation."""
    if not manual_review:
        st.info(
            "No opportunities require manual review in this scan."
        )
        return

    st.caption(
        "These candidates passed part of the pipeline but need a person "
        "to confirm relevance, service fit, or final lead validity."
    )

    similarity_items = [
        item
        for item in manual_review
        if item.get("review_type") == "similarity"
    ]
    gemini_items = [
        item
        for item in manual_review
        if item.get("review_type") == "gemini"
    ]
    other_items = [
        item
        for item in manual_review
        if item.get("review_type") not in {"similarity", "gemini"}
    ]

    review_summary_col1, review_summary_col2, review_summary_col3 = st.columns(3)

    review_summary_col1.metric(
        "Similarity Review",
        len(similarity_items),
    )
    review_summary_col2.metric(
        "Gemini Review",
        len(gemini_items),
    )
    review_summary_col3.metric(
        "Other Review",
        len(other_items),
    )

    for index, review_item in enumerate(
        manual_review,
        start=1,
    ):
        review_type = (
            review_item.get("review_type")
            or "manual"
        ).replace("_", " ").title()

        review_title = (
            review_item.get("company_name")
            or review_item.get("source_title")
            or "Untitled opportunity"
        )

        suggested_similarity = review_item.get(
            "suggested_similarity"
        )

        if suggested_similarity is not None:
            expander_title = (
                f"#{index} {review_title} "
                f"— {review_type} "
                f"({suggested_similarity:.2f}% suggested match)"
            )
        else:
            expander_title = (
                f"#{index} {review_title} "
                f"— {review_type}"
            )

        with st.expander(
            expander_title,
            expanded=index == 1,
        ):
            st.warning(
                review_item.get(
                    "reason",
                    "This opportunity requires manual review.",
                )
            )

            review_col1, review_col2, review_col3 = st.columns(3)

            review_col1.write(
                f"**Review Type:** {review_type}"
            )

            review_col2.write(
                f"**Suggested Service:** "
                f"{review_item.get('suggested_service_name') or 'Not available'}"
            )

            if suggested_similarity is not None:
                review_col3.metric(
                    "Suggested Similarity",
                    f"{suggested_similarity:.2f}%",
                )
            else:
                review_col3.write(
                    "**Suggested Similarity:** Not available"
                )

            detail_col1, detail_col2, detail_col3 = st.columns(3)

            detail_col1.write(
                f"**Company:** "
                f"{review_item.get('company_name') or 'Unknown'}"
            )

            detail_col2.write(
                f"**Industry:** "
                f"{review_item.get('industry') or 'Unknown'}"
            )

            detail_col3.write(
                f"**Country:** "
                f"{review_item.get('country') or 'Unknown'}"
            )

            if developer_mode:
                st.write(
                    f"**Search Query:** "
                    f"{review_item.get('search_query', '-')}"
                )

                suggested_service_id = review_item.get(
                    "suggested_service_id"
                )

                if suggested_service_id:
                    st.write(
                        f"**Suggested Service ID:** "
                        f"{suggested_service_id}"
                    )

            source_snippet = review_item.get(
                "source_snippet"
            )

            if source_snippet:
                st.markdown("**Opportunity Summary**")
                st.write(source_snippet)

            review_source_url = review_item.get(
                "source_url"
            )

            if review_source_url:
                st.link_button(
                    "Open Source for Review",
                    review_source_url,
                )


st.title("Triway Lead Intelligence")

st.caption(
    "Search the web for organizations showing demand for "
    "services offered by Triway Technologies."
)

api_ready, readiness = check_api()

if not api_ready:
    st.error(
        "The FastAPI backend is not available."
    )

    st.code(
        "python -m uvicorn module_3.main:app --reload"
    )

    with st.expander("Connection details"):
        st.json(readiness)

    st.stop()


st.success(
    "Backend connected successfully. "
    f"{readiness.get('service_count', 0)} services loaded."
)

service_count = int(
    readiness.get("service_count", 0) or 0
)


with st.sidebar:
    st.header("Dashboard")

    developer_mode = st.toggle(
        "Developer Mode",
        value=False,
        help=(
            "Shows technical discovery controls, query details, "
            "score breakdowns, and raw API output."
        ),
    )

    if developer_mode:
        st.divider()
        st.subheader("Discovery Settings")

        queries_per_service = st.slider(
            "Queries per service",
            min_value=1,
            max_value=3,
            value=2,
            help=(
                "Number of deterministic search strategies generated "
                "for each eligible Triway service."
            ),
        )

        max_total_queries = st.slider(
            "Maximum total queries",
            min_value=1,
            max_value=100,
            value=50,
            help=(
                "Hard cap for the entire scan. Round-robin allocation "
                "prevents early services from consuming the full budget."
            ),
        )

        results_per_query = st.slider(
            "Results per query",
            min_value=1,
            max_value=10,
            value=5,
            help=(
                "Maximum number of search results collected "
                "for each query."
            ),
        )

        max_leads = st.slider(
            "Maximum validated leads",
            min_value=1,
            max_value=100,
            value=20,
            help="Maximum number of validated leads returned.",
        )

        minimum_similarity = st.slider(
            "Minimum similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help=(
                "Qualified candidates below this service similarity "
                "threshold are routed to manual review."
            ),
        )
    else:
        queries_per_service = 2
        max_total_queries = 50
        results_per_query = 5
        max_leads = 20
        minimum_similarity = 0.25

    requested_queries = service_count * queries_per_service
    expected_queries = min(
        requested_queries,
        max_total_queries,
    )

    st.divider()
    st.subheader("Scan Estimate")

    st.write(
        f"**Services:** {service_count}"
    )
    st.write(
        f"**Queries per service:** {queries_per_service}"
    )
    st.write(
        f"**Requested queries:** {requested_queries}"
    )
    st.write(
        f"**Run limit:** {max_total_queries}"
    )
    st.info(
        f"Expected execution: up to {expected_queries} queries."
    )

    st.divider()
    st.subheader("System Status")

    st.write(
        f"**Provider:** "
        f"{readiness.get('provider', '-')}"
    )

    st.write(
        f"**Model:** "
        f"{readiness.get('model', '-')}"
    )

    st.write(
        f"**Services loaded:** "
        f"{readiness.get('service_count', '-')}"
    )

    if developer_mode:
        st.write(
            f"**Embedding version:** "
            f"{readiness.get('embedding_version', '-')}"
        )


st.subheader("Sales Opportunities")

st.write(
    "Run a fresh scan for organizations currently showing "
    "buying intent for Triway services."
)

last_scan = st.session_state.get("last_scan_time")

if last_scan:
    st.caption(
        f"Last scan: {last_scan}"
    )
else:
    st.caption("No scan has been run in this session.")

generate_clicked = st.button(
    "Generate New Opportunities",
    type="primary",
    use_container_width=True,
)


if generate_clicked:
    payload = {
        "queries_per_service": queries_per_service,
        "max_total_queries": max_total_queries,
        "results_per_query": results_per_query,
        "max_leads": max_leads,
        "minimum_similarity": minimum_similarity,
        "selected_service_ids": [],
    }

    with st.spinner(
        "Scanning for current buying opportunities, validating sources, "
        "matching Triway services, and completing final validation..."
    ):
        try:
            result = discover_leads(payload)

        except Exception as exc:
            st.error(str(exc))

        else:
            st.session_state["latest_result"] = result
            st.session_state["last_scan_time"] = (
                datetime.now().strftime("%d %b %Y, %I:%M %p")
            )

            st.success(
                f"Discovery complete. "
                f"{result.get('leads_found', 0)} validated opportunities "
                f"and {result.get('manual_review_count', 0)} manual-review "
                f"items found."
            )


result = st.session_state.get("latest_result")

if result:
    st.divider()

    display_metrics(result)

    queries = result.get(
        "queries_executed",
        [],
    )
    leads = result.get(
        "leads",
        [],
    )
    manual_review = result.get(
        "manual_review",
        [],
    )

    validated_tab, manual_review_tab = st.tabs(
        [
            f"✅ Validated Leads ({len(leads)})",
            f"🟠 Manual Review ({len(manual_review)})",
        ]
    )

    with validated_tab:
        display_validated_leads(
            leads=leads,
            developer_mode=developer_mode,
        )

    with manual_review_tab:
        display_manual_review(
            manual_review=manual_review,
            developer_mode=developer_mode,
        )

    if developer_mode:
        with st.expander("Search Queries Executed"):
            if queries:
                for index, query in enumerate(
                    queries,
                    start=1,
                ):
                    st.write(f"{index}. {query}")
            else:
                st.write("No queries were executed.")

        with st.expander("Raw Discovery Response"):
            st.json(result)