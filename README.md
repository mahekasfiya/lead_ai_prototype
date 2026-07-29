# Lead AI Prototype

The platform combines traditional information retrieval, semantic search, rule-based qualification, Large Language Models (Gemini), and business intelligence to identify high-value commercial opportunities from across the web.

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

```
                Search Queries
                      │
                      ▼
           Search Engine Collection
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
                      ▼
          Gemini Lead Validator
                      │
                      ▼
        Lead Intelligence Engine
                      │
                      ▼
            Final Lead Report
```

---

# Pipeline

## Module 1 — Query Generation

Generates optimized search queries for each supported service using multiple search strategies including:

- Procurement
- Government tenders
- RFPs
- RFQs
- Enterprise modernization
- Hiring signals
- Marketplace opportunities

---

## Module 2 — Document Collection

Retrieves and normalizes content from:

- PDF
- HTML
- Procurement portals
- Government websites

Features:

- URL normalization
- Duplicate removal
- Content extraction
- Metadata extraction

---

## Module 3 — Qualification

Identifies whether a document represents a genuine buying opportunity.

Checks include:

- Buyer intent
- Service requirement
- External supplier requirement
- Organization role
- Procurement document type

---

## Module 4 — Contradiction Detection

Prevents false positives by detecting provider-generated content.

Examples:

- Service advertisements
- Consulting firms
- Vendor marketing
- Product pages

---

## Module 5 — Deadline Assessment

Determines procurement status.

Supported states:

- Active
- Expired
- Unknown

Detects:

- Submission deadlines
- Closing dates
- Procurement status
- Temporal validity

---

## Module 6 — Service Matching

Uses Sentence Transformers embeddings to match discovered opportunities against the service catalogue.

Outputs include:

- Semantic similarity
- Capability matching
- Technology matching
- Business problem matching
- Industry matching

---

## Module 7 — Gemini Lead Validation

Performs AI validation only after deterministic qualification.

Possible outcomes:

- Valid Lead
- Not a Lead
- Manual Review

---

## Module 8 — Lead Intelligence

Produces business-ready recommendations.

Outputs include:

- Opportunity score
- Buying stage
- Priority
- Urgency
- Risks
- Recommended actions
- Talking points
- Suggested services

# Installation

## Prerequisites

- Python 3.10+
- Git
- Google Gemini API Key
- SerpAPI Key (or another supported search provider)

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
GEMINI_API_KEY=your_gemini_api_key
SERPAPI_KEY=your_serpapi_key

USE_GEMINI = true

QUERY_WORKERS=5 # no. of queries fetched at the same time through serpapi(can be any no.)
FETCH_WORKERS=10 # 

LOG_LEVEL=INFO
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOCAL_EMBEDDING_BATCH_SIZE=16
NORMALIZE_EMBEDDINGS=true
KNOWLEDGE_BASE_PATH=data/triway_knowledge_base_v0_2.json
OUTPUT_DIRECTORY=data/embeddings

EMBEDDING_VERSION=triway-services-local-v1
APP_ENV=development
LOG_LEVEL=INFO


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
uvicorn module_3.main:app --reload
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

## API Endpoints

### Health Check

```
GET /readiness
```

---

### Lead Discovery

```
POST /discover-leads
```

Discovers procurement opportunities, validates them, matches services, and generates business intelligence.

---

### Lead Analysis

```
POST /analyze-lead
```

Analyzes a single document or opportunity and produces a detailed qualification report.

---

## Notes

- Internet connectivity is required for search providers and Gemini validation.
- Search results depend on the configured search provider and API quotas.
- Gemini validation is used as the final validation layer after deterministic qualification.
- The sample knowledge base included in this repository is intended for testing and evaluation only. Replace it with your organization's own knowledge base before using the system in a production environment.
---

# License

This project is intended for research and prototype purposes.
