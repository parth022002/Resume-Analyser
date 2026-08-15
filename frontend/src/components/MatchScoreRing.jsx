import React from 'react';

export default function MatchScoreRing({ score = 85, size = 60, strokeWidth = 5, showLabel = true }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  let strokeColor = "#6366f1"; // Indigo
  let badgeBg = "bg-indigo-500/10 text-indigo-400 border-indigo-500/30";

  if (score >= 90) {
    strokeColor = "#10b981"; // Emerald
    badgeBg = "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";
  } else if (score >= 80) {
    strokeColor = "#8b5cf6"; // Violet
    badgeBg = "bg-violet-500/10 text-violet-400 border-violet-500/30";
  } else if (score < 70) {
    strokeColor = "#f59e0b"; // Amber
    badgeBg = "bg-amber-500/10 text-amber-400 border-amber-500/30";
  }

  return (
    <div className="relative inline-flex items-center justify-center shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background track circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="rgba(255, 255, 255, 0.08)"
          strokeWidth={strokeWidth}
          fill="transparent"
        />
        {/* Animated glow circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          fill="transparent"
          className="transition-all duration-1000 ease-out drop-shadow-[0_0_8px_rgba(99,102,241,0.5)]"
        />
      </svg>
      
      {/* Inner score label */}
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span className="font-black text-white leading-none tracking-tighter" style={{ fontSize: Math.max(size * 0.32, 13) }}>
          {score}%
        </span>
        {showLabel && (
          <span className="text-[8px] font-bold text-slate-400 tracking-wider uppercase mt-0.5">
            MATCH
          </span>
        )}
      </div>
    </div>
  );
}
