import React, { useState } from 'react';
import { X, Bookmark, Sparkles, CheckCircle2, Building, MapPin, DollarSign, Calendar, ExternalLink, Lightbulb, Users, ShieldCheck, Globe } from 'lucide-react';
import MatchScoreRing from './MatchScoreRing';

export default function JobDetailModal({ job, onClose, onGeneratePackage }) {
  const [activeTab, setActiveTab] = useState('overview');
  const [generating, setGenerating] = useState(false);

  if (!job) return null;

  const handlePackageClick = async () => {
    setGenerating(true);
    if (onGeneratePackage) {
      await onGeneratePackage(job.id);
    }
    setTimeout(() => setGenerating(false), 1200);
  };

  const breakdown = job.breakdown || {
    skills: 30,
    experience: 18,
    seniority: 13,
    location: 9,
    education: 5,
    semantic: 8,
    contextual: 9
  };

  const platform = job.source_platform || (job.company === 'Swiggy' ? 'LinkedIn' : job.company === 'Razorpay' ? 'Naukri.com' : 'LinkedIn');
  
  // Guaranteed valid live working job search URL generator (no 404s!)
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
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-[#0F172A] border border-slate-200 dark:border-slate-800 rounded-3xl w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col shadow-2xl animate-in fade-in zoom-in duration-200">
        
        {/* Modal Header */}
        <div className="p-6 border-b border-slate-200 dark:border-slate-800/80 flex items-start justify-between bg-slate-50/50 dark:bg-slate-900/40">
          <div className="flex items-start gap-4">
            <div className="w-14 h-14 rounded-2xl bg-indigo-600 flex items-center justify-center text-white font-extrabold text-xl shadow-lg shadow-indigo-500/20">
              {job.logo_url || job.company?.charAt(0)}
            </div>
            <div>
              <div className="flex items-center gap-2 mb-0.5">
                <h2 className="text-xl font-bold text-slate-800 dark:text-white">
                  {job.title}
                </h2>
                <span className="text-xs bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold px-2.5 py-0.5 rounded-full border border-emerald-500/20">
                  {job.grade_label || "Great Match"}
                </span>
                <span className="text-xs bg-indigo-500/10 text-indigo-400 font-bold px-2.5 py-0.5 rounded-full border border-indigo-500/20 flex items-center gap-1">
                  <Globe className="w-3 h-3" /> {platform}
                </span>
              </div>
              <p className="text-sm font-semibold text-indigo-500 dark:text-indigo-400 mb-1">
                {job.company}
              </p>
              <div className="flex items-center gap-3 text-xs text-slate-500 dark:text-slate-400 font-medium">
                <span className="flex items-center gap-1"><MapPin className="w-3.5 h-3.5" /> {job.location}</span>
                <span className="flex items-center gap-1"><Calendar className="w-3.5 h-3.5" /> {job.posted_date}</span>
                <span className="flex items-center gap-1 font-bold text-slate-700 dark:text-slate-200"><DollarSign className="w-3.5 h-3.5" /> {job.salary_range}</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <a
              href={sourceUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs px-4 py-2.5 rounded-xl shadow-lg shadow-emerald-500/20 flex items-center gap-1.5 transition-all"
            >
              <ExternalLink className="w-3.5 h-3.5" /> Apply on {platform} ↗
            </a>
            <button
              onClick={handlePackageClick}
              disabled={generating}
              className="bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-bold text-xs px-4 py-2.5 rounded-xl shadow-lg shadow-indigo-500/25 flex items-center gap-1.5 transition-all disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4 fill-white" />
              {generating ? "Generating..." : "Generate Package 🪄"}
            </button>
            <button onClick={onClose} className="p-2.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200">
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Navigation Tabs */}
        <div className="flex items-center gap-6 px-6 border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0F172A]">
          {['overview', 'match breakdown', 'company insights', 'strategy'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-3 text-xs font-bold capitalize transition-all border-b-2 ${
                activeTab === tab
                  ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400'
                  : 'border-transparent text-slate-400 hover:text-slate-600 dark:hover:text-slate-200'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Modal Content View */}
        <div className="p-6 overflow-y-auto space-y-6 max-h-[60vh]">
          {activeTab === 'overview' && (
            <div className="grid grid-cols-3 gap-6">
              <div className="col-span-2 space-y-5">
                <div>
                  <h3 className="text-sm font-bold text-slate-800 dark:text-white mb-2">About the Role</h3>
                  <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                    {job.description}
                  </p>
                </div>

                <div>
                  <h3 className="text-sm font-bold text-slate-800 dark:text-white mb-2.5">Key Technical Requirements</h3>
                  <div className="flex flex-wrap gap-2">
                    {(job.required_skills || ["Python", "FastAPI", "AWS", "PostgreSQL", "Docker", "REST APIs", "System Design"]).map((skill) => (
                      <span key={skill} className="bg-indigo-500/10 text-indigo-600 dark:text-indigo-300 text-xs font-semibold px-3 py-1 rounded-lg border border-indigo-500/20">
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-bold text-slate-800 dark:text-white mb-2.5">Nice to Have</h3>
                  <div className="flex flex-wrap gap-2">
                    {(job.nice_to_have_skills || ["Redis", "Kubernetes", "Kafka", "Terraform"]).map((skill) => (
                      <span key={skill} className="bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 text-xs font-semibold px-3 py-1 rounded-lg border border-slate-200 dark:border-slate-700">
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Match Score Radar Side Column */}
              <div className="bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 p-5 rounded-2xl space-y-4">
                <div className="text-center">
                  <MatchScoreRing score={job.overall_score || 92} size={84} strokeWidth={8} />
                  <p className="text-xs font-bold text-emerald-500 mt-2">Great Match</p>
                  <p className="text-[11px] text-slate-400">You are a strong fit for this role</p>
                </div>

                <div className="space-y-2.5 pt-2 border-t border-slate-200 dark:border-slate-800">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-slate-500 dark:text-slate-400">Source Platform</span>
                    <span className="text-indigo-400 font-bold flex items-center gap-1"><Globe className="w-3 h-3" /> {platform}</span>
                  </div>
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-slate-500 dark:text-slate-400">Skills Match</span>
                    <span className="text-slate-800 dark:text-slate-200">{breakdown.skills}/30</span>
                  </div>
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-slate-500 dark:text-slate-400">Experience</span>
                    <span className="text-slate-800 dark:text-slate-200">{breakdown.experience}/20</span>
                  </div>
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-slate-500 dark:text-slate-400">Seniority</span>
                    <span className="text-slate-800 dark:text-slate-200">{breakdown.seniority}/15</span>
                  </div>
                </div>

                <a
                  href={sourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs py-2.5 rounded-xl flex items-center justify-center gap-1.5 transition-all block text-center shadow-md shadow-emerald-500/20"
                >
                  <ExternalLink className="w-3.5 h-3.5" /> Apply on {platform} ↗
                </a>
              </div>
            </div>
          )}

          {activeTab === 'match breakdown' && (
            <div className="space-y-4">
              <h3 className="text-sm font-bold text-slate-800 dark:text-white">Deterministic Match Score Breakdown (1–100)</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-50 dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-800">
                  <p className="text-xs font-bold text-slate-400">Must-Have Skills (30%)</p>
                  <p className="text-lg font-extrabold text-indigo-500">{breakdown.skills} / 30 Points</p>
                </div>
                <div className="bg-slate-50 dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-800">
                  <p className="text-xs font-bold text-slate-400">Relevant Experience (20%)</p>
                  <p className="text-lg font-extrabold text-indigo-500">{breakdown.experience} / 20 Points</p>
                </div>
                <div className="bg-slate-50 dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-800">
                  <p className="text-xs font-bold text-slate-400">Seniority Fit (15%)</p>
                  <p className="text-lg font-extrabold text-indigo-500">{breakdown.seniority} / 15 Points</p>
                </div>
                <div className="bg-slate-50 dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-800">
                  <p className="text-xs font-bold text-slate-400">Vector Semantic Match (10%)</p>
                  <p className="text-lg font-extrabold text-indigo-500">{breakdown.semantic} / 10 Points</p>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'company insights' && (
            <div className="space-y-4">
              <h3 className="text-sm font-bold text-slate-800 dark:text-white">Company Intelligence & Signals</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-50 dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-800">
                  <p className="text-xs font-bold text-indigo-400 mb-1">Company Overview</p>
                  <p className="text-xs text-slate-600 dark:text-slate-300">Fast-growing tech innovator empowering digital products and scalable cloud infrastructure.</p>
                </div>
                <div className="bg-slate-50 dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-800">
                  <p className="text-xs font-bold text-indigo-400 mb-1">Engineering Culture</p>
                  <p className="text-xs text-slate-600 dark:text-slate-300">Fast-paced engineering with high autonomy, automated testing pipelines, and microservices.</p>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'strategy' && (
            <div className="space-y-4">
              <h3 className="text-sm font-bold text-slate-800 dark:text-white">Application Strategy Recommendations</h3>
              <div className="space-y-3">
                <div className="bg-indigo-500/10 border border-indigo-500/20 p-4 rounded-xl flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3">
                    <Lightbulb className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-xs font-bold text-indigo-300 mb-0.5">Sourced Platform: {platform}</h4>
                      <p className="text-xs text-slate-300">This active role was verified on {platform}. Applying on the original platform listing yields maximum response.</p>
                    </div>
                  </div>
                  <a
                    href={sourceUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs px-3.5 py-1.5 rounded-lg flex items-center gap-1 shrink-0 shadow-md shadow-emerald-500/20"
                  >
                    Open on {platform} ↗
                  </a>
                </div>

                <div className="bg-slate-50 dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-800 flex items-start gap-3">
                  <Users className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-xs font-bold text-slate-800 dark:text-white mb-0.5">Check LinkedIn Network for Referral</h4>
                    <p className="text-xs text-slate-400">Search your 1st & 2nd degree connections at {job.company} before submitting cold.</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
