from __future__ import annotations

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
    """
    Check whether the FastAPI backend is ready.
    """
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
    """
    Call the FastAPI lead-discovery endpoint.
    """
    response = requests.post(
        DISCOVER_LEADS_URL,
        json=payload,
        timeout=300,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Lead discovery failed with status "
            f"{response.status_code}: {response.text}"
        )

    return response.json()


def confidence_badge(confidence: str | None) -> str:
    """
    Format confidence for display.
    """
    normalized = str(confidence or "").lower()

    if normalized == "high":
        return "🟢 High"

    if normalized == "medium":
        return "🟡 Medium"

    return "🔴 Low"


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


with st.sidebar:
    st.header("Discovery Settings")

    max_queries = st.slider(
        "Maximum queries",
        min_value=1,
        max_value=20,
        value=5,
        help=(
            "Maximum number of buying-intent search queries "
            "sent to SerpAPI."
        ),
    )

    results_per_query = st.slider(
        "Results per query",
        min_value=1,
        max_value=10,
        value=5,
        help=(
            "Maximum number of SerpAPI results collected "
            "for each query."
        ),
    )

    max_leads = st.slider(
        "Maximum leads",
        min_value=1,
        max_value=50,
        value=20,
        help="Maximum number of ranked leads shown.",
    )

    minimum_similarity = st.slider(
        "Minimum similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help=(
            "Candidates below this semantic similarity "
            "threshold are excluded."
        ),
    )

    st.divider()

    st.subheader("Model")

    st.write(
        f"**Provider:** "
        f"{readiness.get('provider', '-')}"
    )

    st.write(
        f"**Model:** "
        f"{readiness.get('model', '-')}"
    )

    st.write(
        f"**Services:** "
        f"{readiness.get('service_count', '-')}"
    )

    st.write(
        f"**Embedding version:** "
        f"{readiness.get('embedding_version', '-')}"
    )


st.subheader("Automatic Lead Discovery")

st.write(
    "Click the button below to generate buying-intent queries "
    "from the Triway knowledge base, search them through SerpAPI, "
    "analyze the resulting webpages, and rank potential leads."
)

generate_clicked = st.button(
    "Generate Leads",
    type="primary",
    use_container_width=True,
)


if generate_clicked:
    payload = {
        "max_queries": max_queries,
        "results_per_query": results_per_query,
        "max_leads": max_leads,
        "minimum_similarity": minimum_similarity,
        "selected_service_ids": [],
    }

    with st.spinner(
        "Searching SerpAPI, collecting webpages, "
        "matching services, and ranking leads..."
    ):
        try:
            result = discover_leads(payload)

        except Exception as exc:
            st.error(str(exc))

        else:
            st.success(
                f"Discovery complete. "
                f"{result.get('leads_found', 0)} leads found."
            )

            metric_col1, metric_col2, metric_col3, metric_col4 = (
                st.columns(4)
            )

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
                "Leads Found",
                result.get("leads_found", 0),
            )

            queries = result.get(
                "queries_executed",
                [],
            )

            with st.expander(
                "Search Queries Executed"
            ):
                if queries:
                    for query in queries:
                        st.write(f"- {query}")
                else:
                    st.write("No queries were executed.")

            leads = result.get(
                "leads",
                [],
            )

            st.subheader("Ranked Leads")

            if not leads:
                st.info(
                    "No leads met the selected threshold."
                )

            for index, lead in enumerate(
                leads,
                start=1,
            ):
                top_percentage = (
                    lead.get(
                        "top_service_match_percentage"
                    )
                    or 0.0
                )

                title = (
                    f"#{index} "
                    f"{lead.get('source_title', 'Untitled source')} "
                    f"— {top_percentage:.2f}%"
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
                        lead.get(
                            "top_service_name"
                        )
                        or "Unknown",
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
                        top_match.get(
                            "confidence",
                            "Low",
                        ),
                    )

                    col3.metric(
                        "Confidence",
                        confidence_badge(confidence),
                    )

                    st.write(
                        f"**Search Query:** "
                        f"{lead.get('search_query', '-')}"
                    )

                    st.write(
                        f"**Company:** "
                        f"{lead.get('company_name') or 'Unknown'}"
                    )

                    st.write(
                        f"**Industry:** "
                        f"{lead.get('industry') or 'Unknown'}"
                    )

                    st.write(
                        f"**Country:** "
                        f"{lead.get('country') or 'Unknown'}"
                    )

                    snippet = lead.get(
                        "source_snippet"
                    )

                    if snippet:
                        st.write(
                            f"**Search Snippet:** {snippet}"
                        )

                    source_url = lead.get(
                        "source_url"
                    )

                    if source_url:
                        st.link_button(
                            "Open Source",
                            source_url,
                        )

                    if matched_services:
                        st.markdown(
                            "### Matched Triway Services"
                        )

                    for match in matched_services:
                        service_match_percentage = (
                            match.get(
                                "service_match_percentage"
                            )
                            or match.get(
                                "similarity_percentage",
                                0.0,
                            )
                        )

                        semantic_percentage = match.get(
                            "similarity_percentage",
                            0.0,
                        )

                        st.markdown(
                            f"#### #{match.get('rank')} "
                            f"{match.get('service_name')}"
                        )

                        service_col1, service_col2 = (
                            st.columns(2)
                        )

                        service_col1.metric(
                            "Service Match Score",
                            (
                                f"{service_match_percentage:.2f}%"
                            ),
                        )

                        service_col2.metric(
                            "Semantic Similarity",
                            f"{semantic_percentage:.2f}%",
                        )

                        st.write(
                            f"**Category:** "
                            f"{match.get('category', '-')}"
                        )

                        st.write(
                            f"**Explanation:** "
                            f"{match.get('explanation', '-')}"
                        )

                        evidence = match.get(
                            "evidence",
                            {},
                        )

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
                                (
                                    readable_name,
                                    values,
                                )
                            )

                        if evidence_values:
                            with st.expander(
                                "View Match Evidence"
                            ):
                                for (
                                    readable_name,
                                    values,
                                ) in evidence_values:
                                    st.write(
                                        f"**{readable_name}:**"
                                    )

                                    for value in values:
                                        st.write(
                                            f"- {value}"
                                        )

                        st.divider()

            with st.expander(
                "Raw Discovery Response"
            ):
                st.json(result)