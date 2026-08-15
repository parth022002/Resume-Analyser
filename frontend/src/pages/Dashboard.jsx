import React, { useState, useEffect } from 'react';
import { Sparkles, Briefcase, Award, ChevronRight, Bookmark, Target, Brain, TrendingUp, CheckCircle2, ExternalLink, Globe } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import MatchScoreRing from '../components/MatchScoreRing';
import JobDetailModal from '../components/JobDetailModal';

export default function Dashboard({ searchQuery = "", onSelectJob, user }) {
  const [jobs, setJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:8000/api/v1/jobs/')
      .then(res => res.json())
      .then(data => {
        setJobs(data);
        setLoading(false);
      })
      .catch(() => {
        setJobs([
          {
            id: "job-1",
            title: "Software Engineer - Backend",
            company: "Superset Inc.",
            logo_url: "S",
            location: "Bengaluru, KA (Hybrid)",
            work_mode: "Hybrid",
            salary_range: "₹ 15 - 22 LPA",
            posted_date: "2 days ago",
            overall_score: 92,
            grade_label: "Great Match",
            is_target_company: true,
            source_platform: "LinkedIn",
            source_url: "https://www.linkedin.com/jobs/search/?keywords=Software%20Engineer%20Backend",
            description: "We are looking for a Backend Engineer to build scalable APIs and microservices using Python, FastAPI, and AWS.",
            required_skills: ["Python", "FastAPI", "AWS", "PostgreSQL", "Docker", "CI/CD", "REST APIs", "Microservices", "System Design"],
            nice_to_have_skills: ["Redis", "Kubernetes", "Kafka", "Terraform"]
          },
          {
            id: "job-2",
            title: "SDE II - Full Stack",
            company: "Airmeet",
            logo_url: "A",
            location: "Bengaluru, KA (Remote)",
            work_mode: "Remote",
            salary_range: "₹ 18 - 25 LPA",
            posted_date: "1 day ago",
            overall_score: 88,
            grade_label: "Great Match",
            is_target_company: false,
            source_platform: "Naukri.com",
            source_url: "https://www.naukri.com/software-engineer-jobs-in-bengaluru",
            description: "Seeking an SDE II to build real-time virtual event features using React, Node.js, and Python backend services.",
            required_skills: ["React", "Node.js", "Python", "WebSockets", "AWS"],
            nice_to_have_skills: ["GraphQL", "Redis", "TypeScript"]
          },
          {
            id: "job-3",
            title: "Backend Developer",
            company: "Razorpay",
            logo_url: "R",
            location: "Bengaluru, KA (On-site)",
            work_mode: "On-site",
            salary_range: "₹ 16 - 20 LPA",
            posted_date: "3 days ago",
            overall_score: 85,
            grade_label: "Great Match",
            is_target_company: true,
            source_platform: "Naukri.com",
            source_url: "https://www.naukri.com/razorpay-jobs-in-bengaluru",
            description: "Join Razorpay's Core Payments team to build resilient payment gateway rails handling millions of transactions.",
            required_skills: ["Python", "Go", "PostgreSQL", "Redis", "Kafka"],
            nice_to_have_skills: ["AWS", "Kubernetes"]
          },
          {
            id: "job-4",
            title: "Staff Software Engineer",
            company: "Swiggy",
            logo_url: "S",
            location: "Bengaluru, KA (Hybrid)",
            work_mode: "Hybrid",
            salary_range: "₹ 25 - 35 LPA",
            posted_date: "5 days ago",
            overall_score: 84,
            grade_label: "Great Match",
            is_target_company: false,
            source_platform: "LinkedIn",
            source_url: "https://www.linkedin.com/jobs/search/?keywords=Swiggy%20Software%20Engineer",
            description: "Lead architectural decisions for Swiggy's dispatch engine with distributed systems expertise.",
            required_skills: ["Java", "Python", "Distributed Systems", "Kafka"],
            nice_to_have_skills: ["Go", "Kubernetes"]
          }
        ]);
        setLoading(false);
      });
  }, []);

  const filteredJobs = jobs.filter(j => 
    j.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    j.company.toLowerCase().includes(searchQuery.toLowerCase()) ||
    j.location.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (j.required_skills && j.required_skills.some(s => s.toLowerCase().includes(searchQuery.toLowerCase())))
  );

  return (
    <div className="space-y-6">
      {/* Top Hero Banner with Official Logo */}
      <div className="bg-gradient-to-br from-slate-900 via-indigo-950/60 to-slate-900 border border-slate-800 p-6 rounded-3xl shadow-xl flex flex-col md:flex-row items-center justify-between gap-6 relative overflow-hidden backdrop-blur-xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none"></div>

        <div className="flex items-center gap-4 relative z-10">
          <div className="w-16 h-16 rounded-2xl bg-slate-900 border border-amber-500/40 p-1.5 shadow-2xl shadow-amber-500/20 shrink-0 hover:scale-105 transition-transform overflow-hidden">
            <img src="/logo.png" alt="TalentForge Emblem" className="w-full h-full object-contain drop-shadow" />
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[10px] font-black uppercase gold-badge px-2.5 py-0.5 rounded-full">
                AI CAREER INTELLIGENCE PLATFORM
              </span>
              <span className="text-[10px] font-extrabold text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-full border border-cyan-500/20">
                100% Free Open Access
              </span>
            </div>
            <h1 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
              Welcome back, <span className="gold-gradient-text">{user ? user.full_name : "Arjun B."}</span>! 👋
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Active Job Discovery & Placement Optimization Engine. Discovered <span className="font-bold text-white">32 live matched positions</span> today.
            </p>
          </div>
        </div>
      </div>

      {/* 4 Summary Stat Cards - Focused on Position Discovery & Skill Fit */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard icon={Briefcase} label="Active Positions" value="32" trend="12% this week" trendType="up" color="indigo" />
        <MetricCard icon={Target} label="Target Company Hits" value="18" trend="8% this week" trendType="up" color="emerald" />
        <MetricCard icon={Sparkles} label="High Match (>80%)" value="14" trend="15% this week" trendType="up" color="violet" />
        <MetricCard icon={Award} label="Skill Fit Score" value="88%" trend="5% this week" trendType="up" color="rose" />
      </div>

      {/* Main Grid: Top Matches Feed + Positions Overview & AI Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column (2/3 width): Top Matches for You */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-bold text-slate-800 dark:text-white flex items-center gap-2">
              Top Active Positions for You
              {searchQuery && <span className="text-xs font-normal text-indigo-400">Filtering for "{searchQuery}"</span>}
            </h2>
            <button className="text-xs font-bold text-indigo-500 hover:text-indigo-600 flex items-center gap-1">
              See all <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Job Feed List */}
          <div className="space-y-3">
            {filteredJobs.length === 0 ? (
              <div className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800 rounded-2xl p-8 text-center">
                <p className="text-xs text-slate-400">No active positions matched your search criteria.</p>
              </div>
            ) : (
              filteredJobs.map((job) => {
                const platform = job.source_platform || (job.company === 'Swiggy' ? 'LinkedIn' : job.company === 'Razorpay' ? 'Naukri.com' : 'LinkedIn');
                
                const getLiveSourceUrl = () => {
                  if (job.source_url && (job.source_url.includes('linkedin.com') || job.source_url.includes('naukri.com') || job.source_url.includes('indeed.com'))) {
                    return job.source_url;
                  }
                  const query = encodeURIComponent(`${job.title} ${job.company}`);
                  if (platform === 'Naukri.com') {
                    return `https://www.naukri.com/${job.company.toLowerCase().replace(/[^a-z0-9]/g, '')}-jobs-in-bengaluru`;
                  }
                  return `https://www.linkedin.com/jobs/search/?keywords=${query}`;
                };

                const sourceUrl = getLiveSourceUrl();

                return (
                  <div
                    key={job.id}
                    className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800/80 rounded-2xl p-4 shadow-sm hover:shadow-md transition-all flex items-center justify-between group"
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 rounded-xl bg-slate-900 text-white font-extrabold text-lg flex items-center justify-center shadow-md">
                        {job.logo_url || job.company.charAt(0)}
                      </div>
                      <div>
                        <div className="flex items-center gap-2 mb-0.5">
                          <h3 className="text-sm font-bold text-slate-800 dark:text-white group-hover:text-indigo-500 transition-colors">
                            {job.title}
                          </h3>
                          {job.is_target_company && (
                            <span className="bg-indigo-500/10 text-indigo-400 text-[10px] font-bold px-2 py-0.5 rounded-full border border-indigo-500/20">
                              🎯 Target Company
                            </span>
                          )}
                          <span className="bg-emerald-500/10 text-emerald-400 text-[10px] font-bold px-2 py-0.5 rounded-full border border-emerald-500/20 flex items-center gap-1">
                            <Globe className="w-2.5 h-2.5" /> {platform}
                          </span>
                        </div>
                        <p className="text-xs text-slate-500 dark:text-slate-400 font-semibold mb-1">
                          {job.company} • <span className="text-slate-400">{job.location}</span>
                        </p>
                        <div className="flex items-center gap-2 text-[11px] font-semibold text-slate-500 dark:text-slate-400">
                          <span className="bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded text-slate-700 dark:text-slate-300">
                            {job.salary_range}
                          </span>
                          <span>• {job.posted_date}</span>
                        </div>
                      </div>
                    </div>

                    {/* Score Gauge & View Details / Apply on Source Buttons */}
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <MatchScoreRing score={job.overall_score || 85} size={54} strokeWidth={5} />
                      </div>
                      <div className="flex flex-col gap-1.5 text-right">
                        <button
                          onClick={() => setSelectedJob(job)}
                          className="bg-slate-100 dark:bg-slate-800 hover:bg-indigo-600 hover:text-white text-slate-700 dark:text-slate-200 font-semibold text-xs px-3.5 py-1.5 rounded-xl border border-slate-200 dark:border-slate-700 transition-all"
                        >
                          View Details
                        </button>
                        <a
                          href={sourceUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-[11px] font-bold text-emerald-500 hover:text-emerald-400 flex items-center justify-end gap-1"
                        >
                          Apply on {platform} <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Right Column (1/3 width): Positions Overview & AI Insights */}
        <div className="space-y-6">
          {/* Sourced Positions Overview Card */}
          <div className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm space-y-4">
            <h3 className="text-sm font-bold text-slate-800 dark:text-white">Positions Sourcing Overview</h3>
            <div className="flex items-center justify-around py-2">
              <div className="relative w-24 h-24 rounded-full border-8 border-indigo-500/20 flex items-center justify-center border-t-indigo-500 border-r-purple-500">
                <div className="text-center">
                  <span className="text-xl font-extrabold text-slate-800 dark:text-white">45</span>
                  <p className="text-[10px] font-semibold text-slate-400">Total</p>
                </div>
              </div>
              <div className="space-y-1.5 text-xs font-semibold">
                <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full bg-indigo-500" /> Active: 32</div>
                <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Target Hits: 18</div>
                <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full bg-amber-500" /> High Match: 14</div>
                <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full bg-violet-500" /> Saved: 8</div>
              </div>
            </div>
          </div>

          {/* AI Insights Banner */}
          <div className="bg-gradient-to-br from-indigo-900/40 to-purple-900/40 border border-indigo-500/30 rounded-2xl p-5 space-y-3">
            <div className="flex items-center gap-2 text-indigo-300 font-bold text-xs">
              <Brain className="w-4 h-4 text-indigo-400" />
              <span>AI Insights</span>
            </div>
            <p className="text-xs text-slate-200 leading-relaxed">
              You are a <strong className="text-emerald-400">92% top match</strong> for Backend roles in Product companies.
            </p>
            <div className="space-y-1.5 text-[11px] text-slate-300">
              <p className="flex items-center gap-1.5">✓ Add AWS deployment projects to your CV</p>
              <p className="flex items-center gap-1.5">✓ Practice System Design & Distributed Caching</p>
            </div>
            <button className="text-xs font-bold text-indigo-400 hover:text-indigo-300 flex items-center gap-1 pt-1">
              Get AI Recommendations →
            </button>
          </div>

          {/* Target Companies Widget */}
          <div className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800/80 rounded-2xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-slate-800 dark:text-white flex items-center gap-1.5">
                <Target className="w-4 h-4 text-indigo-500" /> Target Companies
              </h3>
              <span className="text-xs text-indigo-500 font-semibold cursor-pointer">+ Add</span>
            </div>
            <div className="space-y-2">
              {[
                { name: "Google", jobs: "12 active jobs" },
                { name: "Microsoft", jobs: "8 active jobs" },
                { name: "Amazon", jobs: "15 active jobs" },
                { name: "Razorpay", jobs: "6 active jobs" },
              ].map((c) => (
                <div key={c.name} className="flex items-center justify-between text-xs py-1.5 border-b border-slate-100 dark:border-slate-800/60 last:border-none">
                  <span className="font-semibold text-slate-700 dark:text-slate-300">{c.name}</span>
                  <span className="text-[11px] text-slate-400 font-medium">{c.jobs}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Selected Job Detail Modal */}
      {selectedJob && (
        <JobDetailModal
          job={selectedJob}
          onClose={() => setSelectedJob(null)}
          onGeneratePackage={async (jobId) => {
            await fetch(`http://localhost:8000/api/v1/applications/package/${jobId}`, { method: 'POST' });
          }}
        />
      )}
    </div>
  );
}
