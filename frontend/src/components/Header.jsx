import React, { useState, useRef, useEffect } from 'react';
import { Search, Bell, Command, ChevronDown, CheckCircle2, Briefcase, Sparkles, X, User, LogOut, Settings as SettingsIcon, ShieldCheck } from 'lucide-react';

export default function Header({ onSearch, user, onOpenAuth, onNavigateToSettings, onLogout }) {
  const [showNotifications, setShowNotifications] = useState(false);
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setShowProfileMenu(false);
        setShowNotifications(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const [notifications, setNotifications] = useState([
    {
      id: 1,
      title: "New High Match Position 🎯",
      desc: "Superset Inc. posted Software Engineer - Backend (92% Match Score)",
      time: "10 mins ago",
      unread: true
    },
    {
      id: 2,
      title: "Student Action Plan Updated 📈",
      desc: "Your Resume Readiness Score improved to 88% for Product Companies",
      time: "1 hour ago",
      unread: true
    },
    {
      id: 3,
      title: "Target Company Alert 🏢",
      desc: "Razorpay posted 2 new Backend Developer listings on Naukri.com",
      time: "3 hours ago",
      unread: true
    },
    {
      id: 4,
      title: "Overleaf Resume Package Ready 📄",
      desc: "Jake's Resume ATS template compiled and ready for one-click export",
      time: "Yesterday",
      unread: false
    }
  ]);

  const unreadCount = notifications.filter(n => n.unread).length;

  const handleMarkAllRead = () => {
    setNotifications(notifications.map(n => ({ ...n, unread: false })));
  };

  return (
    <header className="h-16 bg-white/80 dark:bg-[#0F172A]/80 backdrop-blur-md border-b border-slate-200 dark:border-slate-800/80 px-6 flex items-center justify-between sticky top-0 z-20 transition-colors">
      {/* Search Input Bar */}
      <div className="relative w-96">
        <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
        <input
          type="text"
          placeholder="Search jobs, companies, roles, skills..."
          onChange={(e) => onSearch && onSearch(e.target.value)}
          className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl pl-9 pr-12 py-2 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all placeholder:text-slate-400"
        />
        <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-0.5 text-[10px] font-semibold text-slate-400 bg-slate-200 dark:bg-slate-800 px-1.5 py-0.5 rounded border border-slate-300 dark:border-slate-700">
          <Command className="w-2.5 h-2.5" /> K
        </div>
      </div>

      {/* Header Actions */}
      <div className="flex items-center gap-4 relative" ref={menuRef}>

        {/* Notifications Bell & Dropdown */}
        <div className="relative">
          <button
            onClick={() => {
              setShowNotifications(!showNotifications);
              setShowProfileMenu(false);
            }}
            className="relative p-2.5 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 bg-slate-100 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 transition-colors focus:outline-none"
          >
            <Bell className="w-4 h-4" />
            {unreadCount > 0 && (
              <span className="absolute top-1.5 right-1.5 w-2.5 h-2.5 bg-rose-500 rounded-full ring-2 ring-white dark:ring-[#0F172A] animate-pulse" />
            )}
          </button>

          {/* Interactive Notifications Drawer */}
          {showNotifications && (
            <div className="absolute right-0 mt-3 w-80 sm:w-96 bg-white dark:bg-[#0F172A] border border-slate-200 dark:border-slate-800 rounded-2xl shadow-2xl overflow-hidden z-50 animate-in fade-in zoom-in duration-150">
              <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between bg-slate-50 dark:bg-slate-900/60">
                <div className="flex items-center gap-2">
                  <h4 className="text-xs font-bold text-slate-800 dark:text-white">Notifications</h4>
                  {unreadCount > 0 && (
                    <span className="bg-rose-500/10 text-rose-500 text-[10px] font-extrabold px-2 py-0.5 rounded-full border border-rose-500/20">
                      {unreadCount} new
                    </span>
                  )}
                </div>
                {unreadCount > 0 && (
                  <button
                    onClick={handleMarkAllRead}
                    className="text-[11px] font-bold text-indigo-500 hover:text-indigo-600"
                  >
                    Mark all read
                  </button>
                )}
              </div>

              <div className="max-h-80 overflow-y-auto divide-y divide-slate-100 dark:divide-slate-800/60 custom-scrollbar">
                {notifications.length === 0 ? (
                  <div className="p-6 text-center text-xs text-slate-400">
                    No notifications right now.
                  </div>
                ) : (
                  notifications.map((n) => (
                    <div
                      key={n.id}
                      className={`p-3.5 text-left transition-colors flex items-start justify-between gap-3 ${
                        n.unread ? 'bg-indigo-500/5 dark:bg-indigo-500/10' : 'hover:bg-slate-50 dark:hover:bg-slate-900/40'
                      }`}
                    >
                      <div className="space-y-0.5">
                        <div className="flex items-center gap-2">
                          <h5 className="text-xs font-bold text-slate-800 dark:text-white">{n.title}</h5>
                          {n.unread && <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />}
                        </div>
                        <p className="text-xs text-slate-600 dark:text-slate-300 leading-snug">{n.desc}</p>
                        <p className="text-[10px] text-slate-400 font-medium pt-0.5">{n.time}</p>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>

        {/* User Profile Menu & Dropdown Container */}
        <div className="relative">
          <div
            onClick={() => {
              setShowProfileMenu(!showProfileMenu);
              setShowNotifications(false);
            }}
            className="flex items-center gap-2.5 pl-2 border-l border-slate-200 dark:border-slate-800 cursor-pointer group select-none"
          >
            {user && user.avatar_url ? (
              <img
                src={user.avatar_url}
                alt="Profile Photo"
                className="w-8 h-8 rounded-full object-cover ring-2 ring-indigo-500/50 group-hover:ring-indigo-400 transition-all shadow-md"
              />
            ) : (
              <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center text-white font-bold text-xs ring-2 ring-indigo-500/30">
                {user ? user.full_name.charAt(0) : "AB"}
              </div>
            )}
            <div className="hidden sm:block text-left">
              <p className="text-xs font-bold text-slate-800 dark:text-white leading-tight group-hover:text-indigo-400 transition-colors">
                {user ? user.full_name : "Arjun B."}
              </p>
              <p className="text-[10px] text-emerald-500 font-semibold flex items-center gap-1">
                {user?.plan || "Free Student Account"}
              </p>
            </div>
            <ChevronDown className={`w-3.5 h-3.5 text-slate-400 transition-transform duration-200 ${showProfileMenu ? 'rotate-180 text-indigo-400' : ''}`} />
          </div>

          {/* Interactive Profile Dropdown Menu */}
          {showProfileMenu && (
            <div className="absolute right-0 mt-3 w-72 bg-white dark:bg-[#0F172A] border border-slate-200 dark:border-slate-800 rounded-2xl shadow-2xl p-2 z-50 animate-in fade-in zoom-in duration-150 space-y-1">
              
              {/* Header Profile Identity Card */}
              <div className="p-3 bg-slate-50 dark:bg-slate-900/80 rounded-xl border border-slate-200/80 dark:border-slate-800/80 flex items-center gap-3">
                <img
                  src={user?.avatar_url || "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150"}
                  alt="Avatar"
                  className="w-10 h-10 rounded-full object-cover ring-2 ring-indigo-500/40 shrink-0"
                />
                <div className="min-w-0 flex-1">
                  <h4 className="text-xs font-bold text-slate-800 dark:text-white truncate">
                    {user?.full_name || "Arjun B."}
                  </h4>
                  <p className="text-[10px] text-slate-400 truncate">
                    {user?.email || "arjun.b@talentforge.ai"}
                  </p>
                  <span className="inline-block text-[9px] font-extrabold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-md mt-1 border border-emerald-500/20">
                    {user?.plan || "Free Student Account"}
                  </span>
                </div>
              </div>

              <div className="pt-1 space-y-1">
                {/* 🎓 Student Profile Button */}
                <button
                  onClick={() => {
                    setShowProfileMenu(false);
                    if (onNavigateToSettings) onNavigateToSettings();
                  }}
                  className="w-full text-left px-3 py-2.5 rounded-xl text-xs font-bold text-slate-700 dark:text-slate-200 hover:bg-indigo-500/10 hover:text-indigo-400 flex items-center gap-2.5 transition-all group"
                >
                  <div className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 group-hover:bg-indigo-500 group-hover:text-white transition-all">
                    <User className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="block text-xs font-bold">Student Profile</span>
                    <span className="block text-[10px] font-normal text-slate-400">View & edit mandatory education & skills</span>
                  </div>
                </button>

                {/* ⚙️ Account Preferences */}
                <button
                  onClick={() => {
                    setShowProfileMenu(false);
                    if (onNavigateToSettings) onNavigateToSettings();
                  }}
                  className="w-full text-left px-3 py-2 rounded-xl text-xs font-bold text-slate-700 dark:text-slate-200 hover:bg-indigo-500/10 hover:text-indigo-400 flex items-center gap-2.5 transition-all group"
                >
                  <div className="p-1.5 rounded-lg bg-slate-500/10 text-slate-400 group-hover:bg-indigo-500 group-hover:text-white transition-all">
                    <SettingsIcon className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="block text-xs font-bold">Account Settings</span>
                    <span className="block text-[10px] font-normal text-slate-400">Manage login credentials & notifications</span>
                  </div>
                </button>

                {/* 🔑 Sign In / Switch Account */}
                <button
                  onClick={() => {
                    setShowProfileMenu(false);
                    if (onOpenAuth) onOpenAuth();
                  }}
                  className="w-full text-left px-3 py-2 rounded-xl text-xs font-bold text-slate-700 dark:text-slate-200 hover:bg-indigo-500/10 hover:text-indigo-400 flex items-center gap-2.5 transition-all group"
                >
                  <div className="p-1.5 rounded-lg bg-purple-500/10 text-purple-400 group-hover:bg-purple-500 group-hover:text-white transition-all">
                    <ShieldCheck className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="block text-xs font-bold">Switch Account / Sign In</span>
                    <span className="block text-[10px] font-normal text-slate-400">Log into another student profile</span>
                  </div>
                </button>
              </div>

              <div className="border-t border-slate-200 dark:border-slate-800 pt-1 mt-1">
                {/* 🚪 Log Out Button */}
                <button
                  onClick={() => {
                    setShowProfileMenu(false);
                    if (onLogout) onLogout();
                  }}
                  className="w-full text-left px-3 py-2.5 rounded-xl text-xs font-bold text-rose-500 hover:bg-rose-500/10 flex items-center gap-2.5 transition-all group"
                >
                  <div className="p-1.5 rounded-lg bg-rose-500/10 text-rose-500 group-hover:bg-rose-500 group-hover:text-white transition-all">
                    <LogOut className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="block text-xs font-bold">Log Out</span>
                    <span className="block text-[10px] font-normal text-rose-400/80">Sign out of TalentForge</span>
                  </div>
                </button>
              </div>

            </div>
          )}
        </div>
      </div>
    </header>
  );
}
