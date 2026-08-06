from __future__ import annotations

import csv
import io
from datetime import datetime

import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"

READINESS_URL = f"{API_BASE_URL}/readiness"
DISCOVER_LEADS_URL = f"{API_BASE_URL}/discover-leads"
GENERATE_EMAIL_URL = f"{API_BASE_URL}/generate-email"
DISCOVER_POTENTIAL_LEADS_URL = f"{API_BASE_URL}/discover-potential-leads"
GENERATE_LINKEDIN_MESSAGE_URL = f"{API_BASE_URL}/generate-linkedin-message"
DISCOVER_COMPANIES_URL = f"{API_BASE_URL}/discover-companies"
ANALYZE_LINKEDIN_URL = f"{API_BASE_URL}/analyze-linkedin-profile"


st.set_page_config(
    page_title="Triway Lead Intelligence",
    page_icon="🔎",
    layout="wide",
)


def check_api() -> tuple[bool, dict]:
    """Check whether the FastAPI backend is ready."""
    try:
        response = requests.get(READINESS_URL, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, {"status_code": response.status_code, "error": response.text}
    except requests.RequestException as exc:
        return False, {"error": str(exc)}


def discover_leads(payload: dict) -> dict:
    """Call the FastAPI lead-discovery endpoint."""
    response = requests.post(DISCOVER_LEADS_URL, json=payload, timeout=(15, 1800))
    if response.status_code != 200:
        raise RuntimeError(f"Lead discovery failed: {response.status_code} {response.text}")
    return response.json()


def discover_potential_leads(payload: dict) -> dict:
    """Call the FastAPI potential-lead discovery endpoint."""
    response = requests.post(DISCOVER_POTENTIAL_LEADS_URL, json=payload, timeout=300)
    if response.status_code != 200:
        raise RuntimeError(f"Potential leads discovery failed: {response.text}")
    return response.json()


def discover_companies(payload: dict) -> dict:
    """Call the FastAPI company discovery endpoint."""
    response = requests.post(DISCOVER_COMPANIES_URL, json=payload, timeout=1800)
    if response.status_code != 200:
        raise RuntimeError(f"Company discovery failed: {response.text}")
    return response.json()


def analyze_linkedin(payload: dict) -> dict:
    """Call the FastAPI LinkedIn profile analyzer endpoint."""
    response = requests.post(ANALYZE_LINKEDIN_URL, json=payload, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"LinkedIn analysis failed: {response.text}")
    return response.json()


def generate_email_for_lead(lead_data: dict) -> str:
    """Call the /generate-email endpoint and return the draft."""
    try:
        payload = {
            "lead": lead_data,
            "matched_services": lead_data.get("matched_services", []),
        }
        response = requests.post(GENERATE_EMAIL_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("email_draft", "No draft returned.")
        return f"Error {response.status_code}: {response.text}"
    except Exception as exc:
        return f"Failed to generate email: {str(exc)}"


def generate_linkedin_message(lead: dict) -> str:
    """Call the /generate-linkedin-message endpoint and return the draft."""
    try:
        response = requests.post(GENERATE_LINKEDIN_MESSAGE_URL, json={"lead": lead}, timeout=30)
        if response.status_code == 200:
            return response.json().get("message", "No message returned.")
        return f"Error {response.status_code}: {response.text}"
    except Exception as exc:
        return f"Failed to generate message: {str(exc)}"


def confidence_badge(confidence: str | None) -> str:
    normalized = str(confidence or "").lower()
    if normalized == "high":
        return "🟢 High"
    if normalized == "medium":
        return "🟡 Medium"
    return "🔴 Low"


def display_metrics(result: dict) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Queries Executed", len(result.get("queries_executed", [])))
    col2.metric("Sources Collected", result.get("sources_collected", 0))
    col3.metric("Sources Analyzed", result.get("sources_analyzed", 0))
    col4.metric("Validated Leads", result.get("leads_found", 0))
    col5.metric("Manual Review", result.get("manual_review_count", 0))


def display_validated_leads(leads: list[dict], developer_mode: bool) -> None:
    if not leads:
        st.info("No currently valid opportunities passed final validation in this scan.")
        return
    st.caption("These opportunities passed source validation, qualification, service matching, and final validation.")
    for index, lead in enumerate(leads, start=1):
        top_percentage = lead.get("top_service_match_percentage") or 0.0
        title = (
            f"#{index} {lead.get('company_name') or lead.get('source_title', 'Untitled opportunity')} "
            f"— {top_percentage:.2f}% match"
        )
        draft_key = f"email_draft_{lead.get('source_url') or index}"
        is_expanded = index == 1 or bool(st.session_state.get(draft_key))
        with st.expander(title, expanded=is_expanded):
            col1, col2, col3 = st.columns(3)
            col1.metric("Top Service Match", f"{top_percentage:.2f}%")
            col2.metric("Matched Service", lead.get("top_service_name") or "Unknown")
            matched_services = lead.get("matched_services", [])
            top_match = matched_services[0] if matched_services else {}
            confidence = top_match.get("service_match_confidence", top_match.get("confidence", "Low"))
            col3.metric("Confidence", confidence_badge(confidence))

            info_col1, info_col2, info_col3 = st.columns(3)
            info_col1.write(f"**Company:** {lead.get('company_name') or 'Unknown'}")
            info_col2.write(f"**Industry:** {lead.get('industry') or 'Unknown'}")
            info_col3.write(f"**Country:** {lead.get('country') or 'Unknown'}")

            if developer_mode:
                st.write(f"**Search Query:** {lead.get('search_query', '-')}")

            snippet = lead.get("source_snippet")
            if snippet:
                st.markdown("**Opportunity Summary**")
                st.write(snippet)

            source_url = lead.get("source_url")
            if source_url:
                st.link_button("Open Source", source_url)

            # Email generation
            if st.button("📧 Generate Email", key=f"email_btn_{index}"):
                with st.spinner("Generating email draft..."):
                    draft = generate_email_for_lead(lead)
                    st.session_state[draft_key] = draft
            if st.session_state.get(draft_key):
                st.markdown("### 📧 Email Draft")
                st.text_area("Draft", st.session_state[draft_key], height=300, key=f"email_text_{index}")

            if matched_services:
                st.markdown("### Matched Triway Services")
                for match in matched_services:
                    service_match = match.get("service_match_percentage") or match.get("similarity_percentage", 0.0)
                    semantic = match.get("similarity_percentage", 0.0)
                    st.markdown(f"#### #{match.get('rank')} {match.get('service_name')}")
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Service Match", f"{service_match:.2f}%")
                    sc2.metric("Semantic Similarity", f"{semantic:.2f}%")
                    sc3.metric("Confidence", confidence_badge(match.get("service_match_confidence", match.get("confidence"))))
                    st.write(f"**Category:** {match.get('category', '-')}")
                    st.write(f"**Explanation:** {match.get('explanation', '-')}")
                    evidence = match.get("evidence", {})
                    evidence_values = []
                    for field_name, values in evidence.items():
                        if values:
                            readable = field_name.replace("_", " ").title()
                            evidence_values.append((readable, values))
                    if evidence_values:
                        with st.expander("View Match Evidence"):
                            for readable, values in evidence_values:
                                st.write(f"**{readable}:**")
                                for v in values:
                                    st.write(f"- {v}")
                    if developer_mode and match.get("score_breakdown"):
                        with st.expander("View Score Breakdown"):
                            st.json(match["score_breakdown"])
                    st.divider()


def display_manual_review(manual_review: list[dict], developer_mode: bool) -> None:
    if not manual_review:
        st.info("No opportunities require manual review in this scan.")
        return
    st.caption("These candidates passed part of the pipeline but need a person to confirm relevance, service fit, or final lead validity.")
    similarity_items = [i for i in manual_review if i.get("review_type") == "similarity"]
    gemini_items = [i for i in manual_review if i.get("review_type") == "gemini"]
    other_items = [i for i in manual_review if i.get("review_type") not in {"similarity", "gemini"}]
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Similarity Review", len(similarity_items))
    rc2.metric("Gemini Review", len(gemini_items))
    rc3.metric("Other Review", len(other_items))

    for idx, item in enumerate(manual_review, start=1):
        review_type = item.get("review_type", "manual").replace("_", " ").title()
        review_title = item.get("company_name") or item.get("source_title") or "Untitled opportunity"
        suggested_sim = item.get("suggested_similarity")
        if suggested_sim is not None:
            expander_title = f"#{idx} {review_title} — {review_type} ({suggested_sim:.2f}% suggested match)"
        else:
            expander_title = f"#{idx} {review_title} — {review_type}"
        with st.expander(expander_title, expanded=idx == 1):
            st.warning(item.get("reason", "This opportunity requires manual review."))
            rc1, rc2, rc3 = st.columns(3)
            rc1.write(f"**Review Type:** {review_type}")
            rc2.write(f"**Suggested Service:** {item.get('suggested_service_name') or 'Not available'}")
            if suggested_sim is not None:
                rc3.metric("Suggested Similarity", f"{suggested_sim:.2f}%")
            else:
                rc3.write("**Suggested Similarity:** Not available")
            dc1, dc2, dc3 = st.columns(3)
            dc1.write(f"**Company:** {item.get('company_name') or 'Unknown'}")
            dc2.write(f"**Industry:** {item.get('industry') or 'Unknown'}")
            dc3.write(f"**Country:** {item.get('country') or 'Unknown'}")
            if developer_mode:
                st.write(f"**Search Query:** {item.get('search_query', '-')}")
                if item.get("suggested_service_id"):
                    st.write(f"**Suggested Service ID:** {item.get('suggested_service_id')}")
            snippet = item.get("source_snippet")
            if snippet:
                st.markdown("**Opportunity Summary**")
                st.write(snippet)
            if item.get("source_url"):
                st.link_button("Open Source for Review", item.get("source_url"))


# ------------------------------
# Main UI
# ------------------------------
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
service_count = int(readiness.get("service_count", 0) or 0)

# Sidebar
with st.sidebar:
    st.header("Dashboard")
    developer_mode = st.toggle("Developer Mode", value=False)
    if developer_mode:
        st.divider()
        st.subheader("Discovery Settings")
        queries_per_service = st.slider("Queries per service", 1, 3, 2)
        max_total_queries = st.slider("Maximum total queries", 1, 100, 50)
        results_per_query = st.slider("Results per query", 1, 10, 5)
        max_leads = st.slider("Maximum validated leads", 1, 100, 20)
        minimum_similarity = st.slider("Minimum similarity", 0.0, 1.0, 0.25, 0.05)
    else:
        queries_per_service = 2
        max_total_queries = 50
        results_per_query = 5
        max_leads = 20
        minimum_similarity = 0.25

    requested_queries = service_count * queries_per_service
    expected_queries = min(requested_queries, max_total_queries)

    st.divider()
    st.subheader("Scan Estimate")
    st.write(f"**Services:** {service_count}")
    st.write(f"**Queries per service:** {queries_per_service}")
    st.write(f"**Requested queries:** {requested_queries}")
    st.write(f"**Run limit:** {max_total_queries}")
    st.info(f"Expected execution: up to {expected_queries} queries.")

    st.divider()
    st.subheader("System Status")
    st.write(f"**Provider:** {readiness.get('provider', '-')}")
    st.write(f"**Model:** {readiness.get('model', '-')}")
    st.write(f"**Services loaded:** {readiness.get('service_count', '-')}")
    if developer_mode:
        st.write(f"**Embedding version:** {readiness.get('embedding_version', '-')}")

# ------------------------------
# Section 1: Sales Opportunities (existing pipeline)
# ------------------------------
st.subheader("Sales Opportunities")
st.write("Run a fresh scan for organizations currently showing buying intent for Triway services.")
last_scan = st.session_state.get("last_scan_time")
st.caption(f"Last scan: {last_scan}" if last_scan else "No scan has been run in this session.")

if st.button("Generate New Opportunities", type="primary", use_container_width=True):
    payload = {
        "queries_per_service": queries_per_service,
        "max_total_queries": max_total_queries,
        "results_per_query": results_per_query,
        "max_leads": max_leads,
        "minimum_similarity": minimum_similarity,
        "selected_service_ids": [],
    }
    with st.spinner("Scanning for current buying opportunities, validating sources, matching Triway services, and completing final validation..."):
        try:
            result = discover_leads(payload)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state["latest_result"] = result
            st.session_state["last_scan_time"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
            st.success(f"Discovery complete. {result.get('leads_found', 0)} validated opportunities and {result.get('manual_review_count', 0)} manual-review items found.")

result = st.session_state.get("latest_result")
if result:
    st.divider()
    display_metrics(result)
    leads = result.get("leads", [])
    manual_review = result.get("manual_review", [])
    validated_tab, manual_review_tab = st.tabs([f"✅ Validated Leads ({len(leads)})", f"🟠 Manual Review ({len(manual_review)})"])
    with validated_tab:
        display_validated_leads(leads, developer_mode)
    with manual_review_tab:
        display_manual_review(manual_review, developer_mode)
    if developer_mode:
        with st.expander("Search Queries Executed"):
            for idx, q in enumerate(result.get("queries_executed", []), 1):
                st.write(f"{idx}. {q}")
        with st.expander("Raw Discovery Response"):
            st.json(result)

# ------------------------------
# Section 2: Potential Leads (updated with free-text inputs)
# ------------------------------
st.divider()
st.subheader("🎯 Potential Leads")
st.caption("Discover decision-makers on LinkedIn using flexible criteria. Enter any industries, job titles, and countries you want.")

with st.form(key="potential_leads_form"):
    col1, col2 = st.columns(2)
    with col1:
        industries_input = st.text_input(
            "Industries (comma separated)",
            value="Finance, Banking, Technology",
            placeholder="e.g., Finance, Healthcare, Manufacturing"
        )
        countries_input = st.text_input(
            "Countries (comma separated)",
            value="UAE, Saudi Arabia, Oman",
            placeholder="e.g., UAE, Saudi Arabia, United Kingdom"
        )
        titles_input = st.text_input(
            "Job Titles (comma separated)",
            value="CIO, CTO, IT Director, Head of IT",
            placeholder="e.g., CIO, CTO, IT Director, VP Engineering"
        )
    with col2:
        min_employees = st.number_input("Min Employees", min_value=0, value=50, step=10)
        max_employees = st.number_input("Max Employees", min_value=0, value=500, step=10)
        revenue = st.text_input("Revenue Range (optional)", placeholder="e.g., $10M-$50M")
        technologies = st.text_input("Technologies (comma separated)", placeholder="e.g., AWS, Azure, Salesforce")
        recent_funding = st.checkbox("Recent Funding", value=False)

    submitted = st.form_submit_button("🔍 Discover Potential Leads")

    if submitted:
        # Parse comma-separated inputs
        industries = [i.strip() for i in industries_input.split(",") if i.strip()]
        countries = [c.strip() for c in countries_input.split(",") if c.strip()]
        titles = [t.strip() for t in titles_input.split(",") if t.strip()]
        tech_list = [t.strip() for t in technologies.split(",") if t.strip()] if technologies else []

        payload = {
            "industries": industries,
            "countries": countries,
            "titles": titles,
            "min_employees": min_employees if min_employees > 0 else None,
            "max_employees": max_employees if max_employees > 0 else None,
            "revenue": revenue if revenue else None,
            "technologies": tech_list,
            "recent_funding": recent_funding,
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None and v != []}
        with st.spinner("Searching for potential leads..."):
            try:
                potential_result = discover_potential_leads(payload)
                st.session_state["potential_leads_result"] = potential_result
            except Exception as exc:
                st.error(str(exc))

potential_result = st.session_state.get("potential_leads_result")
if potential_result:
    leads = potential_result.get("leads", [])
    st.success(f"Found {len(leads)} potential leads.")
    if leads:
        for idx, person in enumerate(leads, start=1):
            with st.expander(f"{idx}. {person.get('name')} — {person.get('job_title', '')} at {person.get('company', '')}", expanded=idx==1):
                col1, col2, col3, col4 = st.columns(4)
                col1.write(f"**LinkedIn:** [Link]({person.get('linkedin_url')})" if person.get("linkedin_url") else "**LinkedIn:** N/A")
                col2.write(f"**Company:** {person.get('company') or 'N/A'}")
                col3.write(f"**Industry:** {person.get('industry') or 'N/A'}")
                col4.write(f"**Country:** {person.get('country') or 'N/A'}")
                st.write(f"**Score:** {person.get('score', 0)}")

                # LinkedIn message generation
                msg_key = f"linkedin_msg_{idx}"
                if st.button("💬 Generate LinkedIn Message", key=f"msg_btn_{idx}"):
                    with st.spinner("Generating personalized message..."):
                        msg = generate_linkedin_message(person)
                        st.session_state[msg_key] = msg
                if st.session_state.get(msg_key):
                    st.markdown("### 📨 LinkedIn Message Draft")
                    st.text_area("Message", st.session_state[msg_key], height=200, key=f"msg_text_{idx}")
                    st.caption("Copy this message and send it as a connection note or InMail.")

        # CSV export option
        if st.button("📥 Export CSV"):
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["name", "linkedin_url", "company", "job_title", "industry", "country", "score"])
            writer.writeheader()
            for lead in leads:
                writer.writerow({
                    "name": lead.get("name"),
                    "linkedin_url": lead.get("linkedin_url"),
                    "company": lead.get("company"),
                    "job_title": lead.get("job_title"),
                    "industry": lead.get("industry"),
                    "country": lead.get("country"),
                    "score": lead.get("score", 0),
                })
            st.download_button("Download CSV", output.getvalue(), "potential_leads.csv", "text/csv")


# ------------------------------
# Section 3: Company Discovery
# ------------------------------
st.divider()
st.subheader("🏢 Company Discovery")
st.caption("Find companies on specific websites or directories, enriched with industry, country, and IT service signals.")

with st.form(key="company_discovery_form"):
    col1, col2 = st.columns(2)
    with col1:
        websites_input = st.text_input(
            "Websites to search (comma separated)",
            value="dubaichamber.com, yellowpages-uae.com, ae.kompass.com",
            placeholder="e.g., dubaichamber.com, yellowpages-uae.com, ded.ae"
        )
        company_industries = st.text_input(
            "Industries (comma separated)",
            value="Technology, Finance, Healthcare, Manufacturing",
            placeholder="e.g., Technology, Finance, Healthcare"
        )
    with col2:
        company_countries = st.text_input(
            "Countries (comma separated)",
            value="UAE, Saudi Arabia, Oman",
            placeholder="e.g., UAE, Saudi Arabia, Oman"
        )
        min_company_employees = st.number_input("Min Employees", min_value=0, value=50, step=10)
        max_company_employees = st.number_input("Max Employees", min_value=0, value=500, step=10)
    submitted_companies = st.form_submit_button("🔍 Discover Companies")

if submitted_companies:
    # Parse inputs into lists (strip spaces, remove empty strings)
    selected_websites = [w.strip() for w in websites_input.split(",") if w.strip()]
    industry_list = [i.strip() for i in company_industries.split(",") if i.strip()]
    country_list = [c.strip() for c in company_countries.split(",") if c.strip()]

    # Build payload
    payload = {
        "websites": selected_websites if selected_websites else None,
        "industries": industry_list if industry_list else None,
        "countries": country_list if country_list else None,
        "min_employees": min_company_employees if min_company_employees > 0 else None,
        "max_employees": max_company_employees if max_company_employees > 0 else None,
    }
    # Remove None values
    payload = {k: v for k, v in payload.items() if v not in ([], None)}

    with st.spinner("Searching companies..."):
        try:
            comp_result = discover_companies(payload)
            st.session_state["company_discovery_result"] = comp_result
            # Force rerun to display results immediately
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

# Display results if they exist
if "company_discovery_result" in st.session_state:
    comp_result = st.session_state["company_discovery_result"]
    comps = comp_result.get("companies", [])

    st.success(f"Found {len(comps)} companies.")
    if comps:
        for idx, comp in enumerate(comps, start=1):
            with st.expander(f"{idx}. {comp.get('name')} — {comp.get('industry', 'N/A')}", expanded=idx==1):
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Domain:** {comp.get('domain', 'N/A')}")
                c2.write(f"**Country:** {comp.get('country', 'N/A')}")
                c3.write(f"**Score:** {comp.get('score', 0)}")
                st.write(f"**Source:** {comp.get('source_site', 'N/A')}")
                if comp.get("url"):
                    st.write(f"**URL:** {comp['url']}")
                if comp.get("buying_signals"):
                    st.write("**Buying Signals:** " + ", ".join(comp["buying_signals"]))
                if comp.get("requires_it_services"):
                    st.write("**Requires IT Services:** ✅")
                if comp.get("contacts"):
                    st.write("**Contacts:**")
                    for contact in comp["contacts"]:
                        st.write(f"- {contact.get('title', 'N/A')}: {contact.get('email', '')} {contact.get('phone', '')}")
        # CSV export for companies
        if st.button("📥 Export Companies CSV"):
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["name", "domain", "industry", "country", "score", "contacts", "source_site"])
            writer.writeheader()
            for comp in comps:
                writer.writerow({
                    "name": comp.get("name"),
                    "domain": comp.get("domain"),
                    "industry": comp.get("industry"),
                    "country": comp.get("country"),
                    "score": comp.get("score", 0),
                    "contacts": comp.get("contacts", []),
                    "source_site": comp.get("source_site"),
                })
            st.download_button("Download", output.getvalue(), "companies.csv", "text/csv")
    else:
        st.info("No companies found matching your criteria.")

    # Debug: show raw response in developer mode
    if developer_mode:
        with st.expander("Raw Company Discovery Response"):
            st.json(comp_result)
            
# ------------------------------
# Section 4: LinkedIn Profile Analyzer
# ------------------------------
st.divider()
st.subheader("🔍 LinkedIn Profile Analyzer")
st.caption("Paste a LinkedIn profile URL (and optional text) to get a professional summary and a draft outreach message.")

with st.form(key="linkedin_analyze_form"):
    linkedin_url = st.text_input("LinkedIn Profile URL", placeholder="https://www.linkedin.com/in/username/")
    linkedin_text = st.text_area("Additional Context (optional)", placeholder="Paste any notes or profile text you have (e.g., About section, summary).")
    submitted_analyze = st.form_submit_button("Analyze Profile")

if submitted_analyze and linkedin_url:
    with st.spinner("Analyzing profile..."):
        try:
            payload = {
                "url": linkedin_url,
                "text": linkedin_text if linkedin_text and linkedin_text.strip() else None
            }
            analysis_result = analyze_linkedin(payload)
            st.session_state["linkedin_analysis"] = analysis_result
        except Exception as exc:
            st.error(str(exc))

if "linkedin_analysis" in st.session_state:
    data = st.session_state["linkedin_analysis"]
    st.markdown("### 📋 Summary")
    st.write(data.get("summary", "N/A"))
    st.markdown("### ✉️ Draft Message")
    st.text_area("Message", data.get("message", "N/A"), height=200)
    st.caption("Copy this message and send it as a connection note or InMail.")