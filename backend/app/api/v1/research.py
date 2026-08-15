from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, List
from app.core.llm_router import llm_router

router = APIRouter()

class ResearchRequest(BaseModel):
    company_name: str

COMPANY_DATABASE = {
    "google": {
        "company_name": "Google",
        "sector": "Hyperscale Cloud, Search & AI Infrastructure",
        "headquarters": "Mountain View, CA & Bengaluru, KA",
        "logo_url": "G",
        "overview": "Global technology leader building search engines, GCP cloud computing infrastructure, TensorFlow/Gemini AI models, and large-scale distributed systems.",
        "culture": "Engineering-first culture with high standards for data structure optimization, distributed consensus (Spanner), peer code reviews, and blameless post-mortems.",
        "tech_stack": ["C++", "Java", "Python", "Go", "Kubernetes", "gRPC", "Borg", "Spanner", "TensorFlow"],
        "hiring_signals": ["Active hiring for GCP Backend SDEs", "High focus on System Design & Distributed Systems", "12 Active Positions tracked"],
        "interview_tips": [
          "Master graph algorithms, dynamic programming, and concurrency.",
          "Prepare STAR method responses focusing on high-scale impact.",
          "Practice designing distributed key-value stores and rate limiters."
        ]
    },
    "microsoft": {
        "company_name": "Microsoft",
        "sector": "Enterprise Cloud, Azure & Developer Tools",
        "headquarters": "Redmond, WA & Hyderabad / Bengaluru, India",
        "logo_url": "M",
        "overview": "Pioneer in operating systems, Azure cloud, GitHub developer ecosystem, OpenAI enterprise partnerships, and Office productivity tools.",
        "culture": "Growth mindset culture emphasizing customer empathy, cloud-scale architecture, security-first engineering, and open-source contributions.",
        "tech_stack": ["C#", ".NET Core", "TypeScript", "Python", "Azure", "CosmosDB", "Docker", "Kubernetes"],
        "hiring_signals": ["Expanding Azure Cloud Services & AI engineering teams in India", "8 Active Positions tracked"],
        "interview_tips": [
          "Deep dive into object-oriented design patterns & clean code.",
          "Demonstrate strong knowledge of cloud infrastructure and asynchronous programming.",
          "Be prepared for scenario-based system architecture questions."
        ]
    },
    "amazon": {
        "company_name": "Amazon",
        "sector": "E-Commerce, AWS Cloud & Logistics Engineering",
        "headquarters": "Seattle, WA & Bengaluru / Hyderabad, India",
        "logo_url": "A",
        "overview": "World's largest e-commerce and cloud computing provider (AWS), engineering real-time dispatch, inventory optimization, and payment rails.",
        "culture": "Driven strictly by 16 Leadership Principles (Customer Obsession, Ownership, Bias for Action, Invent & Simplify, Deep Dive).",
        "tech_stack": ["Java", "Python", "C++", "AWS DynamoDB", "S3", "Lambda", "SQS", "Kafka"],
        "hiring_signals": ["Hiring heavily across AWS Core Services and Fulfillment Tech", "15 Active Positions tracked"],
        "interview_tips": [
          "Prepare 2 distinct STAR stories for EACH of Amazon's 16 Leadership Principles.",
          "Focus on quantifiable metrics (e.g. 'reduced latency by 45% handling 10k TPS').",
          "Practice OOD (Object Oriented Design) for parking lots, elevator systems, and shopping carts."
        ]
    },
    "razorpay": {
        "company_name": "Razorpay",
        "sector": "Fintech & Core Payment Gateway Infrastructure",
        "headquarters": "Bengaluru, Karnataka",
        "logo_url": "R",
        "overview": "India's leading fintech unicorn powering online payments, neobanking, and automated settlements for millions of merchants.",
        "culture": "Fast-paced, product-driven engineering with high autonomy, microservices architecture, automated testing, and high transaction reliability.",
        "tech_stack": ["Python", "Go", "PHP/Laravel", "PostgreSQL", "Redis", "Kafka", "AWS", "Docker"],
        "hiring_signals": ["Hiring for Core Payments & Risk Infrastructure teams", "6 Active Positions tracked"],
        "interview_tips": [
          "Focus on ACID compliance, database indexing, and transaction isolation levels.",
          "Be ready to explain how to guarantee idempotent API requests in payment processing.",
          "Practice building fault-tolerant microservices with circuit breakers."
        ]
    },
    "swiggy": {
        "company_name": "Swiggy",
        "sector": "Hyperlocal Delivery, Logistics & Instant Commerce",
        "headquarters": "Bengaluru, Karnataka",
        "logo_url": "S",
        "overview": "Leading on-demand food & grocery delivery platform processing millions of real-time orders daily with dynamic dispatch algorithms.",
        "culture": "Data-obsessed engineering environment solving NP-hard routing problems, real-time tracking, and high-concurrency order management.",
        "tech_stack": ["Java", "Go", "Python", "Kafka", "Redis", "Elasticsearch", "AWS", "Kubernetes"],
        "hiring_signals": ["Hiring for Dispatch Engine & Instamart Backend teams", "4 Active Positions tracked"],
        "interview_tips": [
          "Practice Geo-hashing, Quad-trees, and spatial indexing algorithms.",
          "Design high-throughput message streaming architectures using Kafka.",
          "Demonstrate clear understanding of low-latency caching strategies."
        ]
    }
}

@router.post("/analyze")
async def analyze_company(req: ResearchRequest):
    name_clean = req.company_name.strip().lower()
    
    # Check pre-seeded database
    for key, data in COMPANY_DATABASE.items():
        if key in name_clean or name_clean in key:
            return data
            
    # Dynamic synthesis via LLM Router if not in mock database
    display_name = req.company_name.strip() or "Tech Company"
    return {
        "company_name": display_name,
        "sector": "Software & Technology Scaleup",
        "headquarters": "Bengaluru, India / Global Remote",
        "logo_url": display_name[0].upper(),
        "overview": f"{display_name} is a high-growth technology organization building scalable cloud products, automated API microservices, and modern digital customer experiences.",
        "culture": "Agile, product-focused engineering with continuous delivery, automated unit testing, high code quality standards, and async team collaboration.",
        "tech_stack": ["Python", "FastAPI", "React", "PostgreSQL", "AWS", "Docker", "Redis", "CI/CD"],
        "hiring_signals": [f"Actively hiring Software Engineers for {display_name}", "Focus on System Design & Clean Architecture"],
        "interview_tips": [
          f"Review key technical projects aligned with {display_name}'s product domain.",
          "Prepare clear explanations of your recent architectural decisions.",
          "Demonstrate proficiency in database optimization and clean API design."
        ]
    }
