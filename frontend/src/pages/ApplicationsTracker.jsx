import React, { useState, useEffect } from 'react';
import { ExternalLink, Sparkles, CheckCircle2, Clock, FileText } from 'lucide-react';

export default function ApplicationsTracker() {
  const [apps, setApps] = useState([]);

  useEffect(() => {
    fetch('http://localhost:8000/api/v1/applications/')
      .then(res => res.json())
      .then(data => setApps(data))
      .catch(() => {
        setApps([
          { id: 1, job_title: "Software Engineer - Backend", company: "Superset Inc.", status: "Interviewing", template_used: "Jake's Resume", updated_at: "Today, 12:46PM", overleaf_url: "https://www.overleaf.com" },
          { id: 2, job_title: "Backend Developer", company: "Razorpay", status: "Applied", template_used: "Jake's Resume", updated_at: "Yesterday", overleaf_url: "https://www.overleaf.com" },
          { id: 3, job_title: "SDE II - Full Stack", company: "Airmeet", status: "Shortlisted", template_used: "FAANGPath Simple", updated_at: "3 days ago", overleaf_url: "https://www.overleaf.com" }
        ]);
      });
  }, []);

  const columns = ["Shortlisted", "Applied", "Interviewing", "Offer", "Rejected"];

  const handleOverleaf = (app) => {
    if (app.overleaf_url) {
      window.open(app.overleaf_url, '_blank');
    } else {
      window.open('https://www.overleaf.com/docs', '_blank');
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 dark:text-white tracking-tight">
            Applications Tracker (Kanban Board)
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Track submission packages, tailored LaTeX variants, and live interview stages.
          </p>
        </div>
      </div>

      {/* Kanban Columns */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 overflow-x-auto pb-4">
        {columns.map((col) => {
          const colApps = apps.filter(a => a.status.toLowerCase() === col.toLowerCase());

          return (
            <div key={col} className="bg-slate-100/70 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 min-h-[500px] flex flex-col">
              <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-200 dark:border-slate-800">
                <h3 className="text-xs font-extrabold text-slate-700 dark:text-slate-200 uppercase tracking-wider">{col}</h3>
                <span className="bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 text-[10px] font-bold px-2 py-0.5 rounded-full">
                  {colApps.length}
                </span>
              </div>

              <div className="space-y-3 flex-1">
                {colApps.map((app) => (
                  <div key={app.id} className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800 rounded-xl p-3.5 shadow-sm space-y-2.5">
                    <div>
                      <h4 className="text-xs font-bold text-slate-800 dark:text-white">{app.job_title}</h4>
                      <p className="text-[11px] font-semibold text-indigo-500">{app.company}</p>
                    </div>

                    <div className="flex items-center gap-1.5 text-[10px] text-slate-400 bg-slate-50 dark:bg-slate-900/80 p-1.5 rounded-lg border border-slate-100 dark:border-slate-800">
                      <FileText className="w-3 h-3 text-indigo-400" />
                      <span>{app.template_used || "Jake's Resume"}</span>
                    </div>

                    <button
                      onClick={() => handleOverleaf(app)}
                      className="w-full bg-slate-100 dark:bg-slate-800 hover:bg-indigo-600 hover:text-white text-slate-700 dark:text-slate-300 text-[11px] font-semibold py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 transition-colors flex items-center justify-center gap-1"
                    >
                      <ExternalLink className="w-3 h-3" /> Open in Overleaf
                    </button>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
