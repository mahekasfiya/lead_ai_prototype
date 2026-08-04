# Lead AI Prototype

The platform combines traditional information retrieval, semantic search, rule-based qualification, Large Language Models, and business intelligence to identify high-value commercial opportunities from across the web.

---

## Features

- Automated lead discovery
- Procurement and RFP detection
- Intelligent document fetching and extraction
- Buyer vs Provider classification
- Deadline and procurement status detection
- Semantic service matching using embeddings
- AI-powered lead validation
- Opportunity scoring
- Contact extraction
- Sales intelligence generation
- Manual review workflow for uncertain opportunities

---

# Architecture

## 1. Sales Opportunity Discovery Pipeline

```text
                Service Portfolio
                      │
                      ▼
          Claude Query Planning Agent
                      │
                      ▼
              Search Queries
                      │
                      ▼
          SerpAPI Search Execution
                      │
                      ▼
             Candidate URLs
                      │
                      ▼
             Document Fetcher
                      │
                      ▼
            Source Validation
                      │
                      ▼
         Listing Page Detection
                      │
                      ▼
          Deadline Assessment
                      │
                      ▼
      Requirement Qualification
                      │
                      ▼
        Contradiction Detection
                      │
                      ▼
        Embedding Similarity Engine
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
   Similarity Review      Claude Lead Validator
                                 │
                                 ▼
                      Lead Intelligence Engine
                                 │
                                 ▼
                         Final Lead Report
```
---

## 2. Potential Lead and Decision-Maker Discovery Pipeline
```text
             User-Defined Criteria
       Industry, Country, Role, Technology
                      │
                      ▼
          Claude Query Planning Agent
                      │
                      ▼
      LinkedIn-Focused Search Queries
                      │
                      ▼
          SerpAPI Search Execution
                      │
                      ▼
        Public Search Result Profiles
                      │
                      ▼
      Profile Extraction and Deduplication
                      │
                      ▼
        Claude Relevance Enrichment
                      │
                      ▼
         Decision-Maker Qualification
                      │
                      ▼
            Lead Scoring and Ranking
                      │
                      ▼
       LinkedIn Message Draft Generation
                      │
                      ▼
       Potential Lead Dashboard / Export

```

# Pipeline
1. Sales Opportunity Discovery Pipeline
## Module 1 — AI Query Planning

Generates intelligent, service-specific search queries using Claude based on the organization's service catalogue.

Search strategies include:

- Government procurement
- RFPs, RFQs and RFIs
- Public tenders
- Enterprise modernization initiatives
- Digital transformation projects
- Hiring signals
- Technology adoption
- Marketplace opportunities

**Output:**

- Optimized search queries
- Search intent classification
- Query diversification

---

## Module 2 — Search & Document Collection

Executes search queries using SerpAPI and retrieves candidate opportunity sources from the web.

Supported sources include:

- Government procurement portals
- Tender websites
- Enterprise websites
- Public announcements
- PDF documents
- HTML webpages
- Marketplace listings

Features:

- Search execution
- URL normalization
- Duplicate removal
- Content extraction
- Metadata extraction

---

## Module 3 — Rule-Based Opportunity Qualification

Determines whether a retrieved document represents a genuine business opportunity before invoking AI validation.

Checks include:

- Buyer intent detection
- Service requirement identification
- External supplier requirement
- Procurement document classification
- Organization relevance
- Technology requirement extraction

Only candidates that satisfy deterministic qualification proceed further.

---

## Module 4 — Source Verification & Contradiction Detection

Removes false positives and validates source credibility.

Detects:

- Vendor marketing pages
- Service advertisements
- Product landing pages
- Consulting company promotions
- Educational content
- Generic industry articles
- Contradictory buying signals

---

## Module 5 — Procurement Timeline Assessment

Determines whether an opportunity is still actionable.

Supported states:

- Active
- Expired
- Unknown

Detects:

- Submission deadlines
- Closing dates
- Procurement status
- Temporal validity
- Opportunity freshness

---

## Module 6 — Semantic Service Matching

Matches qualified opportunities against the organization's service catalogue using Sentence Transformers embeddings.

Evaluates:

- Semantic similarity
- Business capability alignment
- Technology alignment
- Industry relevance
- Problem–solution fit

Outputs:

- Best matching service
- Similarity score
- Candidate ranking

---

## Module 7 — Claude Lead Validation

Performs evidence-based AI validation only after deterministic qualification and semantic matching.

Responsibilities:

- Validate genuine buying intent
- Eliminate remaining false positives
- Assess opportunity quality
- Recommend manual review when evidence is inconclusive

Possible outcomes:

- Valid Lead
- Rejected
- Manual Review

---

## Module 8 — Lead Intelligence & Sales Insights

Transforms validated opportunities into business-ready intelligence for sales teams.

Outputs include:

- Opportunity score
- Buying stage
- Priority
- Urgency
- Risks
- Business need summary
- Recommended services
- Sales talking points
- Suggested next actions
- Executive summary
---

2. Potential Lead Generation Pipeline

## Module 1 — AI Query Planning

Generates intelligent search queries using Claude based on user-defined targeting criteria.

Supported criteria include:

- Industry
- Country
- Job titles
- Company size
- Technologies
- Revenue range
- Funding status

Output:

- LinkedIn-focused search queries
- Executive search queries
- Industry-specific search strategies

---

## Module 2 — Search & Profile Discovery

Executes generated queries using SerpAPI to discover publicly available professional profiles.

Supported sources include:

- LinkedIn public profiles
- Company leadership pages
- Executive directories
- Professional websites

Features:

- Search execution
- URL normalization
- Duplicate removal
- Profile extraction

---

## Module 3 — Candidate Extraction

Extracts candidate information from search results.

Captured attributes include:

- Name
- Job title
- Company
- Profile URL
- Search evidence
- Public metadata

Features:

- Profile deduplication
- Entity extraction
- Candidate normalization

---

## Module 4 — AI Candidate Qualification

Uses Claude to determine whether a discovered professional matches the requested targeting criteria.

Evaluates:

- Job role relevance
- Decision-making authority
- Industry relevance
- Geographic relevance
- Evidence quality

Possible outcomes:

- Qualified Lead
- Not Relevant
- Manual Review

---

## Module 5 — Lead Scoring & Ranking

Ranks qualified professionals according to business relevance.

Scoring considers:

- Job seniority
- Role relevance
- Industry alignment
- Geographic alignment
- Evidence confidence

Outputs:

- Ranked lead list
- Qualification score
- Supporting evidence

---

## Module 6 — AI Outreach Generation

Generates personalized outreach content for qualified prospects.

Supported outputs include:

- LinkedIn connection requests
- LinkedIn InMail drafts
- Cold email drafts
- Personalized introductions

Features:

- Evidence-based personalization
- Professional tone
- Context-aware messaging

# Installation

## Prerequisites

Before running the project, ensure the following dependencies are available:

- Python 3.10 – 3.13
- Git
- Anthropic API Key (Claude)
- SerpAPI Key (for web search)
- Internet connection for LLM and search services
---

## Clone the Repository

```bash
git clone https://github.com/mahekasfiya/lead_ai_prototype.git

cd lead_ai_prototype
```

---

## Create a Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file in the project root.

Example:

```env
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOCAL_EMBEDDING_BATCH_SIZE=16
NORMALIZE_EMBEDDINGS=true

KNOWLEDGE_BASE_PATH=data/triway_knowledge_base_v0_2.json
OUTPUT_DIRECTORY=data/embeddings

EMBEDDING_VERSION=triway-services-local-v1
APP_ENV=development
LOG_LEVEL=INFO

SERPAPI_KEY= YOUR_SERP_API_KEY
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY
USE_CLAUDE = true
CLAUDE_MODEL= claude-sonnet-5

LLM_BATCH_SIZE=8
LLM_MAX_CANDIDATES=20
LLM_MAX_EXCERPT_CHARS=3000


```

Additional configuration can be adjusted in:

```
config.py
```

---

## Knowledge Base Configuration

The repository includes a sample knowledge base used **only for testing and demonstration purposes**.

The provided services, technologies, business problems, keywords, and mappings are intended to validate the discovery pipeline and **do not represent a production-ready service catalog**.

For real deployments, replace the knowledge base with your organization's:

- Service catalog
- Technologies
- Industries
- Capabilities
- Business problems
- Buying signals
- Keywords
- Regions

---
## Generate Embeddings
```bash
python -m module_2.generate_embeddings
```
NOTE: Requires Internet Connection
(GPU can be used to speed up this process)

## Run the API

```bash
uvicorn module_3.main:app
```

The API will start at:

```
http://127.0.0.1:8000
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```
NOTE: This location is for debugging

---
## Run the Streamlit App
```bash
streamlit run dashboard/app.py
```

# API Endpoints

## Health Check

### Service Readiness

```http
GET /readiness
```

Verifies that the application is running and confirms the availability of core services, including the LLM, embedding model, and search provider.

---

## Opportunity Discovery

### Discover Business Opportunities

```http
POST /discover-leads
```

Discovers publicly available service opportunities by:

- Generating AI-powered search queries
- Searching procurement and public web sources
- Collecting and extracting documents
- Qualifying buying signals
- Performing semantic service matching
- Validating opportunities using Claude
- Producing sales-ready lead intelligence

---

### Analyze a Single Opportunity

```http
POST /analyze-lead
```

Analyzes an individual document or opportunity and generates a comprehensive qualification report including:

- Buying signal analysis
- Procurement status
- Service matching
- Similarity scoring
- Business insights
- Recommended actions

---

## Potential Lead Discovery

### Discover Decision Makers

```http
POST /discover-potential-leads
```

Identifies potential decision-makers based on user-defined targeting criteria such as:

- Industry
- Country
- Job title
- Company size
- Technologies
- Revenue range
- Funding status

Returns a ranked list of qualified professionals together with supporting evidence and relevance scores.

---

### Generate LinkedIn Outreach

```http
POST /generate-linkedin-message
```

Generates a personalized LinkedIn connection request or InMail draft for a selected prospect using evidence gathered during the lead discovery process.

---

## Interactive API Documentation

Once the application is running, the complete documentation is available at:

```text
http://127.0.0.1:8000/docs
```
---
## Notes

- Internet connectivity is required for search providers and LLM validation.
- Search results depend on the configured search provider and API quotas.
- LLM validation is used as the final validation layer after deterministic qualification.
- The sample knowledge base included in this repository is intended for testing and evaluation only. Replace it with your organization's own knowledge base before using the system in a production environment.
---

# License

This project is intended for research and prototype purposes.
