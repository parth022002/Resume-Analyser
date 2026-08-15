import React from 'react';

export default function MetricCard({ icon: Icon, label, value, trend, trendType = "up", color = "violet" }) {
  const isUp = trendType === "up";
  const isNeutral = trendType === "neutral";

  const colorStyles = {
    violet: "bg-violet-500/10 text-violet-500 border-violet-500/20",
    indigo: "bg-indigo-500/10 text-indigo-500 border-indigo-500/20",
    rose: "bg-rose-500/10 text-rose-500 border-rose-500/20",
    emerald: "bg-emerald-500/10 text-emerald-500 border-emerald-500/20",
  };

  return (
    <div className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800/80 rounded-2xl p-4 shadow-sm hover:shadow-md transition-all">
      <div className="flex items-center justify-between mb-3">
        <div className={`p-2.5 rounded-xl border ${colorStyles[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
        {trend && (
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full flex items-center gap-1 ${
            isUp 
              ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20" 
              : isNeutral
              ? "bg-slate-500/10 text-slate-500 dark:text-slate-400 border border-slate-500/20"
              : "bg-rose-500/10 text-rose-500 border border-rose-500/20"
          }`}>
            {isUp && "↑"} {trend}
          </span>
        )}
      </div>
      <div className="text-2xl font-bold text-slate-800 dark:text-white mb-0.5">
        {value}
      </div>
      <div className="text-xs font-medium text-slate-500 dark:text-slate-400">
        {label}
      </div>
    </div>
  );
}
