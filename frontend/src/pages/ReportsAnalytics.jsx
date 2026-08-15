import React, { useState, useEffect } from 'react';
import { UserCheck, Award, Target, GraduationCap, ShieldCheck, Brain, CheckCircle2, TrendingUp, Sparkles, BookOpen } from 'lucide-react';
import MetricCard from '../components/MetricCard';

export default function ReportsAnalytics() {
  const [data, setData] = useState({
    candidate_name: "Arjun B.",
    headline: "Software Engineer - Backend & Systems",
    resume_readiness_score: 88,
    target_role_fit_score: 92,
    skill_mastery_coverage: 85,
    interview_readiness_score: 84,
    skills_breakdown: [
      { skill: "Python & Backend Architecture", proficiency: 95, level: "Advanced" },
      { skill: "Cloud & AWS Infrastructure", proficiency: 88, level: "High" },
      { skill: "Database & System Design", proficiency: 85, level: "High" },
      { skill: "Microservices & Docker", proficiency: 90, level: "Advanced" },
      { skill: "Kubernetes & DevOps", proficiency: 60, level: "Medium" }
    ],
    career_roadmap: [
      { phase: "Short-term (30 Days)", action: "Add AWS deployment projects & Docker containerization to CV." },
      { phase: "Mid-term (90 Days)", action: "Practice System Design, Redis caching & Kafka event messaging." },
      { phase: "Long-term (180 Days)", action: "Build Kubernetes orchestration & Terraform IaC portfolio repos." }
    ],
    target_roles: [
      { role: "Software Engineer - Backend", match: 92 },
      { role: "SDE II - Full Stack", match: 88 },
      { role: "Backend Developer", match: 85 },
      { role: "Staff Software Engineer", match: 84 }
    ]
  });

  useEffect(() => {
    fetch('http://localhost:8000/api/v1/analytics/metrics')
      .then(res => res.json())
      .then(resData => setData(resData))
      .catch(() => {});
  }, []);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 dark:text-white tracking-tight flex items-center gap-2">
            Student Career Analytics & Readiness Report
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Personalized candidate performance report, skill mastery analysis, and 30/90/180-day trajectory.
          </p>
        </div>
      </div>

      {/* 4 Student Stat Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          icon={UserCheck}
          label="Resume Readiness"
          value={`${data.resume_readiness_score}%`}
          trend="ATS Tailored"
          trendType="up"
          color="indigo"
        />
        <MetricCard
          icon={Target}
          label="Target Role Fit"
          value={`${data.target_role_fit_score}%`}
          trend="Top Match"
          trendType="up"
          color="emerald"
        />
        <MetricCard
          icon={Award}
          label="Skill Mastery Coverage"
          value={`${data.skill_mastery_coverage}%`}
          trend="Strong Profile"
          trendType="up"
          color="violet"
        />
        <MetricCard
          icon={GraduationCap}
          label="Interview Readiness"
          value={`${data.interview_readiness_score}%`}
          trend="STAR Prepared"
          trendType="up"
          color="rose"
        />
      </div>

      {/* Main Grid: Student Skill Analysis + Target Role Readiness */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column (2/3 width): Skill Analysis & Progress Bars */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Skill Analysis Card */}
          <div className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-slate-800 dark:text-white flex items-center gap-2">
                  <Brain className="w-5 h-5 text-indigo-500" /> Student Skill Analysis
                </h3>
                <p className="text-xs text-slate-400">Evaluated against current active job requirements</p>
              </div>
            </div>

            <div className="space-y-4">
              {data.skills_breakdown.map((item) => (
                <div key={item.skill} className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs font-semibold">
                    <span className="text-slate-800 dark:text-slate-200">{item.skill}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold uppercase bg-indigo-500/10 text-indigo-400 px-2 py-0.5 rounded">
                        {item.level}
                      </span>
                      <span className="text-slate-500 dark:text-slate-400">{item.proficiency}%</span>
                    </div>
                  </div>
                  <div className="w-full bg-slate-100 dark:bg-slate-800 h-2.5 rounded-full overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-violet-600 to-indigo-500 h-full rounded-full transition-all duration-1000"
                      style={{ width: `${item.proficiency}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Actionable Career Improvement Roadmap */}
          <div className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
            <h3 className="text-base font-bold text-slate-800 dark:text-white flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-emerald-500" /> 30 / 90 / 180-Day Growth Roadmap
            </h3>

            <div className="space-y-3">
              {data.career_roadmap.map((step) => (
                <div key={step.phase} className="bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 rounded-xl flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 shrink-0">
                    <BookOpen className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-indigo-400 mb-0.5">{step.phase}</h4>
                    <p className="text-xs text-slate-700 dark:text-slate-300">{step.action}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column (1/3 width): Target Roles Match Breakdown */}
        <div className="space-y-6">
          
          {/* Target Role Matches */}
          <div className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
            <h3 className="text-sm font-bold text-slate-800 dark:text-white flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-amber-400 fill-amber-400" /> Career Role Matches
            </h3>

            <div className="space-y-3">
              {data.target_roles.map((r) => (
                <div key={r.role} className="bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-3.5 rounded-xl flex items-center justify-between">
                  <div>
                    <h4 className="text-xs font-bold text-slate-800 dark:text-white">{r.role}</h4>
                    <p className="text-[10px] text-slate-400 font-semibold">Active Industry Demand</p>
                  </div>
                  <div className="text-right">
                    <span className="text-sm font-extrabold text-indigo-500">{r.match}%</span>
                    <p className="text-[9px] font-bold text-emerald-400">Match</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Student Profile Card */}
          <div className="bg-gradient-to-br from-indigo-900/40 to-purple-900/40 border border-indigo-500/30 rounded-2xl p-5 space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-600 text-white font-bold text-sm flex items-center justify-center">
                AB
              </div>
              <div>
                <h4 className="text-xs font-bold text-white">{data.candidate_name}</h4>
                <p className="text-[10px] text-indigo-300 font-medium">{data.headline}</p>
              </div>
            </div>
            <div className="pt-2 border-t border-indigo-500/20 text-xs text-indigo-200 font-semibold flex items-center justify-between">
              <span>Status: Placement Ready</span>
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
