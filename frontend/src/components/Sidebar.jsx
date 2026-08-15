import React from 'react';
import { 
  LayoutDashboard, 
  Briefcase, 
  Target, 
  Building2, 
  FileEdit, 
  LayoutTemplate, 
  Compass, 
  TrendingUp, 
  User,
  CheckCircle2,
  Zap
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab }) {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'jobs', label: 'Active Job Feed', icon: Briefcase },
    { id: 'target-companies', label: 'Target Companies', icon: Target },
    { id: 'company-research', label: 'Company Research', icon: Building2 },
    { id: 'resume-builder', label: 'Resume & ATS Templates', icon: FileEdit },
    { id: 'action-plan', label: 'Student Action Plan', icon: Compass },
    { id: 'reports', label: 'Reports & Analytics', icon: TrendingUp },
    { id: 'settings', label: 'Student Profile', icon: User },
  ];

  return (
    <aside className="w-64 bg-[#0B0F19] border-r border-slate-800/80 flex flex-col justify-between h-screen sticky top-0 left-0 z-30 select-none text-slate-300">
      {/* Brand Header */}
      <div>
        <div className="p-4 flex items-center gap-3 border-b border-slate-800/80 bg-slate-950/40">
          <div className="w-11 h-11 rounded-2xl bg-slate-900 border border-amber-500/40 p-1 shadow-lg shadow-amber-500/20 shrink-0 hover:scale-105 transition-transform overflow-hidden">
            <img src="/logo.png" alt="TalentForge Logo" className="w-full h-full object-contain drop-shadow" />
          </div>
          <div>
            <h1 className="font-black text-white text-base tracking-tight flex items-center gap-1">
              <span className="gold-gradient-text font-extrabold tracking-tight">TALENTFORGE</span>
            </h1>
            <p className="text-[9px] font-extrabold text-cyan-400 tracking-wider uppercase leading-tight mt-0.5">
              AI CAREER INTELLIGENCE
            </p>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="p-3 space-y-1 max-h-[calc(100vh-320px)] overflow-y-auto custom-scrollbar">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id || (item.id === 'settings' && activeTab === 'profile') || (item.id === 'action-plan' && activeTab === 'interview-coach');

            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-xs font-semibold transition-all duration-200 ${
                  isActive
                    ? 'bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-md shadow-indigo-500/25'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </div>
                {item.badge && (
                  <span className="bg-indigo-500/20 text-indigo-300 text-[10px] font-bold px-2 py-0.5 rounded-full border border-indigo-500/30">
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Sidebar Footer Cards */}
      <div className="p-4 space-y-3 border-t border-slate-800/60 bg-[#070A12]/50">
        {/* 100% Free Platform Status */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 text-center">
          <div className="flex items-center justify-center gap-1.5 text-xs font-bold text-emerald-400 mb-0.5">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>100% Free Platform</span>
          </div>
          <p className="text-[10px] text-slate-400">Open Access for All Students</p>
        </div>

        {/* AI Co-pilot Promo Banner */}
        <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-indigo-900/60 to-purple-900/60 p-3 border border-indigo-500/30 text-center">
          <p className="text-xs font-extrabold text-white mb-0.5">Your AI Co-pilot</p>
          <p className="text-[10px] text-indigo-200">Find Active Positions Faster</p>
        </div>
      </div>
    </aside>
  );
}
