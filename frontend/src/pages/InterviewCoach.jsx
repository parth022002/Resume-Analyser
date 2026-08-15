import React, { useState } from 'react';
import { GraduationCap, Sparkles, CheckCircle2, MessageSquare } from 'lucide-react';

export default function InterviewCoach() {
  const [selectedRole, setSelectedRole] = useState("Software Engineer - Backend (Superset Inc.)");

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 dark:text-white tracking-tight">
            AI Interview Coach & STAR Prep
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Simulate technical questions and STAR behavioral scenarios tailored to shortlisted jobs.
          </p>
        </div>
      </div>

      <div className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-6">
        <div>
          <label className="text-xs font-bold text-slate-700 dark:text-slate-300 mb-2 block">
            Target Job Role for Interview Prep:
          </label>
          <select
            value={selectedRole}
            onChange={(e) => setSelectedRole(e.target.value)}
            className="w-full max-w-md bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl px-4 py-2 text-xs font-semibold text-slate-800 dark:text-slate-200 focus:outline-none"
          >
            <option>Software Engineer - Backend (Superset Inc.)</option>
            <option>Backend Developer (Razorpay)</option>
            <option>SDE II - Full Stack (Airmeet)</option>
          </select>
        </div>

        {/* STAR Scenario Simulator Card */}
        <div className="space-y-4 pt-2">
          <h3 className="text-sm font-bold text-slate-800 dark:text-white flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-indigo-500" /> Behavioral & STAR Questions
          </h3>

          <div className="bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 space-y-4">
            <h4 className="text-xs font-bold text-indigo-400">
              Q: Tell me about a time you had to optimize a critical API endpoint under heavy load.
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
              <div className="bg-white dark:bg-slate-800 p-3 rounded-lg border border-slate-200 dark:border-slate-700">
                <span className="font-extrabold text-indigo-500 block mb-1">Situation</span>
                Superset Inc. payment API endpoint latency spiked to 450ms during peak checkout traffic.
              </div>
              <div className="bg-white dark:bg-slate-800 p-3 rounded-lg border border-slate-200 dark:border-slate-700">
                <span className="font-extrabold text-indigo-500 block mb-1">Task</span>
                Reduce p99 latency to under 100ms without introducing cache inconsistency risks.
              </div>
              <div className="bg-white dark:bg-slate-800 p-3 rounded-lg border border-slate-200 dark:border-slate-700">
                <span className="font-extrabold text-indigo-500 block mb-1">Action</span>
                Implemented Redis cache-aside strategy and indexed PostgreSQL foreign key lookups in FastAPI.
              </div>
              <div className="bg-white dark:bg-slate-800 p-3 rounded-lg border border-slate-200 dark:border-slate-700">
                <span className="font-extrabold text-emerald-500 block mb-1">Result</span>
                Cut p99 response time from 450ms to 65ms, handling 3x concurrent throughput cleanly.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
