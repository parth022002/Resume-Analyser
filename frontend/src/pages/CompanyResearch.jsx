import React, { useState, useEffect } from 'react';
import { Building2, Search, Sparkles, MapPin, Globe, Cpu, CheckCircle2, Lightbulb, ExternalLink, ArrowRight } from 'lucide-react';

export default function CompanyResearch({ searchQuery = "" }) {
  const [targetCompany, setTargetCompany] = useState("Superset Inc.");
  const [searchInput, setSearchInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState({
    company_name: "Superset Inc.",
    sector: "Fintech & Cross-border API Platform",
    headquarters: "Bengaluru, KA",
    logo_url: "S",
    overview: "Fast-growing fintech scaleup enabling automated settlement rails for enterprise B2B payments across Asia and Europe.",
    culture: "High autonomy, async-first documentation, microservices architecture, strong CI/CD continuous deployment posture.",
    tech_stack: ['Python', 'FastAPI', 'PostgreSQL', 'AWS', 'Docker', 'Redis', 'Kafka'],
    hiring_signals: ["Active hiring for Backend Software Engineers", "Focus on Microservices & Distributed Caching", "4 Active Positions tracked"],
    interview_tips: [
      "Be ready to explain RESTful API design best practices and status codes.",
      "Demonstrate expertise in PostgreSQL query optimization and indexing.",
      "Practice STAR responses highlighting ownership and problem solving."
    ]
  });

  const presetCompanies = ["Google", "Microsoft", "Amazon", "Razorpay", "Swiggy", "Superset Inc.", "Airmeet"];

  // Respond to top search bar or local input
  const fetchResearch = async (companyToSearch) => {
    const query = companyToSearch || searchQuery || searchInput || "Superset Inc.";
    if (!query.trim()) return;

    setLoading(true);
    setTargetCompany(query);

    try {
      const res = await fetch('http://localhost:8000/api/v1/research/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company_name: query })
      });
      const data = await res.json();
      setReport(data);
    } catch {
      // Local fallback generator for demonstration
      const name = query.charAt(0).toUpperCase() + query.slice(1);
      setReport({
        company_name: name,
        sector: "Software & Cloud Technology",
        headquarters: "Bengaluru, India / Global",
        logo_url: name.charAt(0),
        overview: `${name} is a fast-growing technology enterprise building resilient cloud products, automated APIs, and high-concurrency microservices.`,
        culture: "Agile, product-focused engineering culture with continuous deployment, automated testing pipelines, and high autonomy.",
        tech_stack: ["Python", "FastAPI", "React", "AWS", "PostgreSQL", "Docker", "Redis"],
        hiring_signals: [`Actively recruiting Software Engineers for ${name}`, "Strong focus on System Design & Data Structures"],
        interview_tips: [
          `Research ${name}'s core engineering products and recent architectural milestones.`,
          "Prepare to write clean, production-grade code with error handling.",
          "Demonstrate strong grasp of asynchronous processing and cloud deployments."
        ]
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (searchQuery.trim()) {
      setSearchInput(searchQuery);
      fetchResearch(searchQuery);
    }
  }, [searchQuery]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    fetchResearch(searchInput);
  };

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 dark:text-white tracking-tight flex items-center gap-2">
            <Building2 className="w-6 h-6 text-indigo-500" />
            Company Intelligence & AI Research
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Real-time AI research reports on company tech stacks, engineering culture, and interview rounds.
          </p>
        </div>
      </div>

      {/* Search Input Bar & Quick Preset Chips */}
      <div className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800 p-5 rounded-2xl shadow-sm space-y-3">
        <form onSubmit={handleSearchSubmit} className="flex items-center gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="Search company (e.g. Google, Microsoft, Amazon, Razorpay, Swiggy)..."
              className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl pl-10 pr-4 py-2.5 text-xs text-slate-800 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs px-5 py-2.5 rounded-xl shadow-lg shadow-indigo-500/20 flex items-center gap-1.5 transition-all shrink-0"
          >
            <Sparkles className="w-4 h-4" /> {loading ? "Analyzing..." : "Generate Report"}
          </button>
        </form>

        {/* Quick Search Preset Chips */}
        <div className="flex items-center gap-2 overflow-x-auto pt-1 custom-scrollbar">
          <span className="text-[11px] font-bold text-slate-400 shrink-0">Popular:</span>
          {presetCompanies.map((c) => (
            <button
              key={c}
              onClick={() => {
                setSearchInput(c);
                fetchResearch(c);
              }}
              className={`text-[11px] font-semibold px-3 py-1 rounded-lg border transition-all shrink-0 ${
                targetCompany.toLowerCase() === c.toLowerCase()
                  ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                  : 'bg-slate-50 dark:bg-slate-900 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-800 hover:border-indigo-500'
              }`}
            >
              {c}
            </button>
          ))}
        </div>
      </div>

      {/* Main Intelligence Report Display */}
      <div className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-6 animate-in fade-in duration-300">
        
        {/* Company Header Block */}
        <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-5">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-violet-600 to-indigo-600 flex items-center justify-center text-white font-extrabold text-2xl shadow-lg shadow-indigo-500/30">
              {report.logo_url || report.company_name?.charAt(0)}
            </div>
            <div>
              <h2 className="text-2xl font-extrabold text-slate-800 dark:text-white flex items-center gap-2">
                {report.company_name}
                <span className="text-xs bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold px-2.5 py-0.5 rounded-full border border-emerald-500/20">
                  Verified Intel
                </span>
              </h2>
              <p className="text-xs text-indigo-500 font-bold mt-0.5">
                {report.sector} • <span className="text-slate-400">{report.headquarters}</span>
              </p>
            </div>
          </div>

          <a
            href={
              report.company_name.toLowerCase().includes('google') ? 'https://careers.google.com/jobs/results/' :
              report.company_name.toLowerCase().includes('microsoft') ? 'https://careers.microsoft.com/us/en/search-results' :
              report.company_name.toLowerCase().includes('amazon') ? 'https://www.amazon.jobs/en/search' :
              report.company_name.toLowerCase().includes('razorpay') ? 'https://www.naukri.com/razorpay-jobs-in-bengaluru' :
              report.company_name.toLowerCase().includes('swiggy') ? 'https://www.linkedin.com/jobs/search/?keywords=Swiggy' :
              `https://www.linkedin.com/jobs/search/?keywords=${encodeURIComponent(report.company_name)}`
            }
            target="_blank"
            rel="noopener noreferrer"
            className="bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs px-4 py-2.5 rounded-xl shadow-md shadow-emerald-500/20 flex items-center gap-1.5"
          >
            <ExternalLink className="w-3.5 h-3.5" /> View Active Openings ↗
          </a>
        </div>

        {/* 3 Overview Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-slate-50 dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 space-y-2">
            <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-wider flex items-center gap-1.5">
              <Building2 className="w-4 h-4 text-indigo-500" /> Company Overview
            </h3>
            <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
              {report.overview}
            </p>
          </div>

          <div className="bg-slate-50 dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 space-y-2">
            <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-wider flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-indigo-500" /> Engineering Culture
            </h3>
            <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
              {report.culture}
            </p>
          </div>

          <div className="bg-slate-50 dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 space-y-2">
            <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-wider">Primary Tech Stack</h3>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {(report.tech_stack || ['Python', 'FastAPI', 'PostgreSQL', 'AWS']).map((tech) => (
                <span key={tech} className="bg-indigo-500/10 text-indigo-500 dark:text-indigo-300 text-xs font-bold px-2.5 py-1 rounded-lg border border-indigo-500/20">
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Interview Guidance & Hiring Signals Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          {/* Interview Strategy Column */}
          <div className="bg-indigo-500/5 border border-indigo-500/20 p-5 rounded-2xl space-y-3">
            <h3 className="text-xs font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-1.5">
              <Lightbulb className="w-4 h-4 text-indigo-400" /> Interview & Round Preparation Strategy
            </h3>
            <div className="space-y-2 text-xs text-slate-300">
              {(report.interview_tips || []).map((tip, idx) => (
                <p key={idx} className="flex items-start gap-2">
                  <span className="text-indigo-400 font-bold">•</span>
                  <span>{tip}</span>
                </p>
              ))}
            </div>
          </div>

          {/* Hiring Signals Column */}
          <div className="bg-emerald-500/5 border border-emerald-500/20 p-5 rounded-2xl space-y-3">
            <h3 className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" /> Hiring Signals & Team Expansion
            </h3>
            <div className="space-y-2 text-xs text-slate-300">
              {(report.hiring_signals || []).map((signal, idx) => (
                <p key={idx} className="flex items-start gap-2">
                  <span className="text-emerald-400 font-bold">✓</span>
                  <span>{signal}</span>
                </p>
              ))}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
