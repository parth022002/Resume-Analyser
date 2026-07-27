# TalentForge v2 — AI Multi-Agent Career Intelligence Platform

<p align="center">
  <img src="frontend/public/logo.png" alt="TalentForge Emblem" width="220"/>
</p>

<p align="center">
  <strong>A 10-Agent Multi-Stage AI Career Intelligence Platform powered by LangGraph, Neon PostgreSQL + pgvector, LiteLLM 3-Tier Multi-Model Router, Model Context Protocol (MCP), and Ragas Quality Gates.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LangGraph-Multi--Agent%20Pipeline-blueviolet?style=for-the-badge&logo=python" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Neon-PostgreSQL%20%2B%20pgvector-00e599?style=for-the-badge&logo=postgresql" alt="Neon Database"/>
  <img src="https://img.shields.io/badge/LiteLLM-3--Tier%20Model%20Router-ff69b4?style=for-the-badge" alt="LiteLLM Router"/>
  <img src="https://img.shields.io/badge/OpenRouter-Multi--LLM%20Ecosystem-6f42c1?style=for-the-badge" alt="OpenRouter"/>
  <img src="https://img.shields.io/badge/Vite%20%2B%20React-Tailwind%20Dashboard-61dafb?style=for-the-badge&logo=react" alt="React UI"/>
  <img src="https://img.shields.io/badge/Vercel-Production%20Deployment-black?style=for-the-badge&logo=vercel" alt="Vercel Ready"/>
</p>

---

## 📌 Table of Contents

- [🌟 Overview](#-overview)
- [🤖 OpenRouter & Multi-LLM Model Matrix](#-openrouter--multi-llm-model-matrix)
- [🧩 Deep-Dive Breakdown of All 10 AI Agents](#-deep-dive-breakdown-of-all-10-ai-agents)
- [📊 System Architecture & Flowcharts](#-system-architecture--flowcharts)
  - [1. LangGraph 4-Stage Multi-Agent Orchestration Flowchart](#1-langgraph-4-stage-multi-agent-orchestration-flowchart)
  - [2. Multi-LLM 3-Tier Router Policy Flowchart](#2-multi-llm-3-tier-router-policy-flowchart)
  - [3. GitHub MCP Code Review Sequence Diagram](#3-github-mcp-code-review-sequence-diagram)
  - [4. Quality Gate & Ragas Evaluation Loop](#4-quality-gate--ragas-evaluation-loop)
- [✨ Key Platform Features](#-key-platform-features)
- [📁 Project Folder Structure](#-project-folder-structure)
- [⚡ Quick Start & Local Installation](#-quick-start--local-installation)
- [🔑 Environment Variables Setup (`.env`)](#-environment-variables-setup-env)
- [🧪 Automated Test Verification](#-automated-test-verification)
- [🚀 GitHub Push & Vercel Deployment Guide](#-github-push--vercel-deployment-guide)

---

## 🌟 Overview

Standard ATS resume checkers rely on basic keyword frequency without evaluating single-column screener rules, external GitHub code quality, or long-term career growth. **TalentForge v2** transforms resume checks into actionable career strategy using a **10-Agent LangGraph Pipeline**.

It evaluates candidate ATS readability, inspects public repositories via **Model Context Protocol (MCP)**, generates **3 tailored resume variants**, constructs a **30/90/180-day milestone roadmap**, and provides an **interactive interview readiness coach**.

---

## 🤖 OpenRouter & Multi-LLM Model Matrix

TalentForge v2 uses a 3-Tier Multi-LLM Router routing calls dynamically across specialized models:

| Task / Stage | Primary OpenRouter / Provider Model | Specialized Agent Responsibilities |
|---|---|---|
| **Intake & Quality Gate** | `google/gemini-2.5-flash:free` | Fast PDF parsing, text extraction, PII security redaction, and Ragas groundedness evaluation. |
| **Match & Skill Matrix** | `google/gemma-2-27b-it` | High-precision entity extraction, keyword gap identification, and ATS layout scoring. |
| **GitHub MCP Code Review** | `deepseek/deepseek-chat` / `deepseek-r1` | Reasoning engine for public repository architecture, documentation score, and test coverage grading. |
| **Resume & Safety Synthesis** | `nvidia/nemotron-4-340b-instruct` | Content safety verification & 3-variant executive resume rewriting (**ATS**, **Tech**, **Executive**). |
| **Interview Coach & Roadmaps** | `groq/llama-3.3-70b-versatile` | High-throughput technical Q&A hints, STAR behavioral guidance, and 30/90/180-day career roadmaps. |

---

## 🧩 Deep-Dive Breakdown of All 10 AI Agents

```
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                      STAGE 1: INTAKE & PARSING LAYER                         │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ Agent 1: Intake & Parsing Agent                                              │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │                      STAGE 2: TECHNICAL ANALYSIS LAYER                       │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ Agent 2: Match & ATS Compatibility Agent                                     │
 │ Agent 3: Code & Portfolio Agent (GitHub MCP)                                 │
 │ Agent 4: Skill & Requirement Insight Agent                                   │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │                    STAGE 3: STRATEGIC CAREER GUIDANCE LAYER                  │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ Agent 5: Optimizer Agent (Resume Variants)                                   │
 │ Agent 6: Career Trajectory Agent (30/90/180 Roadmap)                         │
 │ Agent 7: Interview Coach Agent                                               │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │                     STAGE 4: QUALITY GATE & DELIVERY LAYER                   │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ Agent 8: Explainability Agent                                                │
 │ Agent 9: Quality Gate Agent (Ragas Evaluation & Retry Loop)                  │
 │ Agent 10: Report Generator Agent                                             │
 └──────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Intake & Parsing Layer

#### 1. Intake & Parsing Agent
* **Core Function**: Collects candidate PDF resumes and target Job Specifications (text or uploaded PDF/TXT). Extracts text, redacts sensitive PII (emails/phone numbers) for security, and builds structured candidate & job entity graphs.
* **Inputs**: PDF Resume, Job Description (Text/File), Optional GitHub/Portfolio URLs.
* **Outputs**: `CandidateKnowledgeGraph` & `JobKnowledgeGraph` JSON entities.
* **Model Tier**: Tier 0 (*Gemini 2.5 Flash*).

---

### Stage 2: Technical Analysis Layer

#### 2. Match & ATS Compatibility Agent
* **Core Function**: Analyzes semantic similarity and keyword density between Candidate and Job entity graphs. Evaluates single-column ATS formatting compliance and calculates ATS Readability Score (0-100%).
* **Inputs**: `CandidateKnowledgeGraph` & `JobKnowledgeGraph`.
* **Outputs**: ATS Readability Score, Semantic Similarity Score, Missing Keyword List.
* **Model Tier**: Tier 1 (*Google Gemma 2 27B / DeepSeek V3*).

#### 3. Code & Portfolio Agent (GitHub MCP Gateway)
* **Core Function**: Interacts with the **GitHub Model Context Protocol (MCP)** gateway to inspect public repositories, commit hygiene, code modularity, documentation hygiene, and unit test coverage signals.
* **Inputs**: Candidate GitHub Profile URL.
* **Outputs**: Code Quality Grade (`A+`), Documentation Score, Unit Test Verification, MCP Code Findings.
* **Model Tier**: Tier 1 (*DeepSeek V3 / DeepSeek R1*).

#### 4. Skill & Requirement Insight Agent
* **Core Function**: Identifies critical missing technical skills. For every missing skill, explains its real-world engineering application, interview expectations, and current market demand trends.
* **Inputs**: Extracted Candidate Skills vs Target Job Requirements.
* **Outputs**: Priority Skill Gaps, Real-World Engineering Context, Interview Expectation Notes.
* **Model Tier**: Tier 1 (*DeepSeek V3 / Google Gemma 2*).

---

### Stage 3: Strategic Career Guidance Layer

#### 5. Optimizer Agent (Resume Variants Generator)
* **Core Function**: Synthesizes 3 tailored resume variants for different hiring decision-makers:
  1. **ATS-Optimized**: Maximum keyword density for automated screener software.
  2. **Technical Deep-Dive**: In-depth architecture, API design, and concurrency details for engineering leads.
  3. **Executive & Leadership**: ROI, team leadership, and strategic execution metrics for directors/C-Suite.
* **Inputs**: Candidate Resume Graph, Target Job Spec, Verified Skill Matrix.
* **Outputs**: 3 Tailored Resume Variants with targeted bullet points.
* **Model Tier**: Tier 2 (*NVIDIA Nemotron 4 340B / Claude 3.5 Sonnet*).

#### 6. Career Trajectory Agent
* **Core Function**: Analyzes industry hiring trends and constructs an actionable, phased **30-Day, 90-Day, and 180-Day Progression Roadmap**.
* **Inputs**: Skill Gaps, Target Job Role & Industry Domain.
* **Outputs**: 30-Day (Gap Closure), 90-Day (Advanced Systems), 180-Day (Role Placement) Milestones.
* **Model Tier**: Tier 1 (*Groq Llama 3.3 70B*).

#### 7. Interview Coach Agent
* **Core Function**: Generates technical Q&A with answer hints, behavioral questions formatted using the STAR framework, company-specific preparation tips, coding challenge prompts, and calculates an **Interview Readiness Score**.
* **Inputs**: Target Role, Job Description, Skill Matrix.
* **Outputs**: Technical Q&A, Behavioral STAR Hints, Coding Challenges, Interview Readiness Score (0-100%).
* **Model Tier**: Tier 1 (*Groq Llama 3.3 70B*).

---

### Stage 4: Quality Gate & Delivery Layer

#### 8. Explainability Agent
* **Core Function**: Attaches explicit reasoning metadata (`Problem -> Evidence -> Reason -> Expected Improvement -> Confidence Score`) to every insight and score produced across the pipeline.
* **Inputs**: Analysis Scores & Skill Matrix.
* **Outputs**: Explainability Metadata & Confidence Calibration Cards.
* **Model Tier**: Tier 2 (*NVIDIA Nemotron 4 340B*).

#### 9. Quality Gate Agent (Ragas Evaluation & Retry Loop)
* **Core Function**: Performs automated proxy evaluation (scoring groundedness, context relevance, and answer completeness). If the quality score is below `0.75`, triggers an automated LangGraph loop-back edge to re-run Stage 2 analysis.
* **Inputs**: Full Synthesized Report Object.
* **Outputs**: Quality Score (0–1.0), Pass/Retry Edge Decision.
* **Model Tier**: Tier 0 (*Gemini 2.5 Flash*).

#### 10. Report Generator Agent
* **Core Function**: Compiles validated outputs from all 9 previous agents into the final unified JSON Career Intelligence Report and HTML printable export format.
* **Inputs**: LangGraph StateGraph Final Dictionary.
* **Outputs**: Final JSON Report & HTML Printable Export Document.
* **Model Tier**: Tier 2 (*NVIDIA Nemotron 4 340B*).

---

## 📊 System Architecture & Flowcharts

### 1. LangGraph 4-Stage Multi-Agent Orchestration Flowchart

```mermaid
flowchart TD
    User([Candidate / Recruiter]) -->|Upload PDF Resume + Job Spec| Stage1[Stage 1: Intake & Parsing Agent]
    Stage1 -->|Candidate & Job Entity Graphs| Stage2
    
    subgraph Stage2 [Stage 2: Technical Analysis Layer]
        MatchATS[Match & ATS Agent]
        CodeMCP[Code & Portfolio Agent - GitHub MCP]
        SkillInsight[Skill & Requirement Insight Agent]
    end
    
    Stage2 --> Stage3
    
    subgraph Stage3 [Stage 3: Strategic Career Guidance Layer]
        Optimizer[Optimizer Agent - 3 Resume Variants]
        Trajectory[Career Trajectory Agent - 30/90/180 Roadmap]
        InterviewCoach[Interview Coach & Readiness Score]
    end
    
    Stage3 --> Stage4
    
    subgraph Stage4 [Stage 4: Quality Gate & Delivery Layer]
        Explainability[Explainability Agent]
        QualityGate{Quality Gate Agent - Ragas Score >= 0.75?}
        ReportGen[Report Generator Agent]
    end
    
    QualityGate -->|Quality Score < 0.75| RetryLoop[Trigger LangGraph Retry Loop #1]
    RetryLoop --> MatchATS
    
    QualityGate -->|Quality Score >= 0.75| ReportGen
    ReportGen --> FinalOutput([Validated Career Intelligence Report & HTML Export])
```

---

### 2. Multi-LLM 3-Tier Router Policy Flowchart

```mermaid
flowchart LR
    AgentTask[Agent LLM Request] --> Router{LiteLLM Model Router}
    
    Router -->|Stage 1 Intake & Stage 4 Quality Check| Tier0[Tier 0: Fast / Free]
    Tier0 --> Gemini[Google Gemini 2.5 Flash]
    
    Router -->|Stage 2 Match & Code Review| Tier1[Tier 1: High Efficiency]
    Tier1 --> DeepSeek[DeepSeek V3 / Google Gemma 2]
    
    Router -->|Stage 3 Resume & Career Synthesis| Tier2[Tier 2: Premium Guarded]
    Tier2 --> Nemotron[NVIDIA Nemotron 4 340B]
    
    Gemini --> Monitor[Token & Cost Observability Logger]
    DeepSeek --> Monitor
    Nemotron --> Monitor
```

---

### 3. GitHub MCP Code Review Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant App as App Workspace
    participant Agent as Code Agent (Agent 3)
    participant MCP as GitHub MCP Gateway
    participant GitHub as GitHub API
    
    App->>Agent: Candidate GitHub Profile URL
    Agent->>MCP: Request Repository Quality Signals
    MCP->>GitHub: Fetch Repositories, Readme & Commits
    GitHub-->>MCP: Raw Repos & Public Commit Signals
    MCP->>Agent: Compute Modularity, Docs Score & Unit Test Presence
    Agent-->>App: Return Code Quality Grade (A+) & MCP Findings
```

---

### 4. Quality Gate & Ragas Evaluation Loop

```mermaid
stateDiagram-v2
    [*] --> Stage1_Intake
    Stage1_Intake --> Stage2_Analysis
    Stage2_Analysis --> Stage3_Guidance
    Stage3_Guidance --> Quality_Gate_Check
    
    state Quality_Gate_Check {
        [*] --> Evaluate_Ragas
        Evaluate_Ragas --> Score_Check
        Score_Check --> Passed : Quality Score >= 0.75
        Score_Check --> Failed : Quality Score < 0.75
    }
    
    Failed --> Stage2_Analysis : Re-run Analysis Pipeline
    Passed --> Stage4_Report_Generator
    Stage4_Report_Generator --> [*]
```

---

## ✨ Key Platform Features

1. **Compulsory Steps 1 & 2 Form Validation**:
   - **Step 1 (Resume PDF)** and **Step 2 (Target Job Description)** are strictly required with clear `* COMPULSORY` indicators.
2. **Dual Mode Job Description Input**:
   - Toggle seamlessly between **Paste Text** (textarea) and **Upload File** (PDF / TXT / DOCX).
3. **Optional Step 3 with Hover Info Popovers**:
   - Step 3 (GitHub & Portfolio) is optional. Interactive hover `HelpCircle` popovers explain what the Code Agent inspects.
4. **Dark 🌙 & Light ☀️ Mode Theme Engine**:
   - Toggle between dark glassmorphism and crisp, high-contrast light mode with automatic `localStorage` persistence.
5. **Interactive 5-Tab Report View**:
   - 📊 **Career Overview** (Match scores, verified skills matrix, missing skill tags, action items)
   - 🗺️ **30/90/180-Day Roadmap** (Milestone cards)
   - 🎤 **Interview Coach** (Technical Q&A, STAR behavioral hints, coding challenges)
   - 📄 **Resume Variants** (ATS-Optimized, Technical Deep-Dive, Executive Leadership)
   - 🐙 **Code Review** (GitHub MCP repository findings)

---

## 📁 Project Folder Structure

```
TalentForge/
├── backend/
│   ├── app/
│   │   ├── api/v1/             # FastAPI routers (analyze, reports, usage)
│   │   ├── core/               # Settings, Guardrails, LiteLLM Router, Observability
│   │   ├── db/                 # Neon PostgreSQL DB models & sessions
│   │   ├── mcp/                # Model Context Protocol GitHub client gateway
│   │   ├── rag/                # Hybrid RAG retriever & curated benchmark corpus
│   │   └── agents/             # 10 LangGraph Agents (Stage 1 through Stage 4)
│   ├── tests/                  # Automated test suite (run_tests.py)
│   ├── .env                    # Environment variables (Live API keys)
│   ├── .env.example            # Environment template
│   ├── requirements.txt        # Python backend dependencies
│   └── main.py                 # FastAPI application entrypoint
│
├── frontend/                   # React + Vite + Tailwind CSS Dashboard
│   ├── public/                 # Favicon & 3D Logo emblem asset
│   ├── src/                    # App.jsx, index.css, main.jsx
│   ├── index.html
│   ├── vite.config.js          # Vite config with silent proxy error handler
│   └── package.json
│
├── .gitignore                  # Production gitignore rules
├── vercel.json                 # Vercel static & serverless routing rules
├── checklist.md                # Task tracking checklist
└── README.md                   # Complete documentation
```

---

## ⚡ Quick Start & Local Installation

### 1. Prerequisites
* Python 3.9+
* Node.js 18+

### 2. Backend Setup & Run

```powershell
# Navigate to backend directory
cd backend

# Create & activate Python virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Mac / Linux:
# source venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Launch FastAPI Backend Server
uvicorn main:app --reload --port 8000
```
API Documentation: `http://localhost:8000/docs`

### 3. Frontend Setup & Run

```powershell
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Launch Vite Frontend Dashboard
npm run dev
```
Frontend UI: `http://localhost:3000`

---

## 🔑 Environment Variables Setup (`.env`)

Create `backend/.env` with your connection settings and API keys:

```env
ENVIRONMENT=production

# Neon PostgreSQL Database Connection URL
DATABASE_URL=postgresql://neondb_owner:npg_lrHvxfj6d5Ba@ep-bold-dream-aztdx0m2-pooler.c-3.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require

# Live API Keys
GEMINI_API_KEY=AIzaSyCaPmnwMFXjCEoroH4BmsJ4gO39gbO78hE
GROQ_API_KEY=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
OPENROUTER_API_KEY=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

# Multi-Model Router Defaults
TIER_0_MODEL=openrouter/google/gemini-2.5-flash:free
TIER_1_MODEL=openrouter/deepseek/deepseek-chat
TIER_2_MODEL=openrouter/nvidia/nemotron-4-340b-instruct
```

---

## 🧪 Automated Test Verification

Run the full automated backend test runner to verify PII redaction, prompt injection scanning, Neon RAG retrieval, rate limiting, and 10-agent pipeline execution:

```powershell
backend\venv\Scripts\python backend/tests/run_tests.py
```

Expected Output:
```text
============================================================
TALENTFORGE V2 PLATFORM -- COMPREHENSIVE TEST SUITE
============================================================

[1/5] Testing PII Redaction Guardrail...
  [SUCCESS] PII Redaction Filter passed cleanly.

[2/5] Testing Prompt Injection Detector...
  [SUCCESS] Prompt Injection Scanner passed cleanly.

[3/5] Testing Neon Hybrid RAG Retrieval Engine...
  [SUCCESS] RAG Retriever passed (Retrieved 'Single-Column Standard Layout', Score: 2.733).

[4/5] Testing Per-Session Rate Limiter...
  [SUCCESS] Rate Limiter enforcement passed.

[5/5] Testing 10-Agent Multi-Stage Orchestrated Pipeline...
  [SUCCESS] 10-Agent Pipeline Executed Successfully!
    - Report ID: rpt_8827f238
    - Match Score: 55.0%
    - ATS Score: 88.0%
    - Quality Gate Score: 93%
    - GitHub MCP Quality Grade: A

============================================================
ALL 5 TEST SUITES PASSED! TALENTFORGE V2 IS 100% HEALTHY.
============================================================
```

---

## 🚀 GitHub Push & Vercel Deployment Guide

### Step 1: Push Project to GitHub

1. Initialize Git in the root directory:
   ```bash
   git init
   git add .
   git commit -m "feat: complete TalentForge v2 AI Career Intelligence Platform"
   ```

2. Link to your GitHub repository and push:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/TalentForge.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy to Vercel

1. Log in to [Vercel](https://vercel.com).
2. Import your `TalentForge` GitHub repository.
3. Vercel reads `vercel.json` and automatically deploys the **React Frontend** and **FastAPI Serverless Backend**.
4. Add Environment Variable `DATABASE_URL` in Vercel settings.
5. Click **Deploy**. Your platform is live!

---

## 📜 License & Acknowledgements
Built with ❤️ for the **TalentForge v2 AI Career Intelligence Platform**.
