import React, { useState } from 'react';
import { Compass, CheckSquare, Target, ArrowRight, Zap, Code, ShieldCheck, FileText, Sparkles, CheckCircle2, Circle, Plus, Trash2, X, Download, RefreshCw, BarChart2, BookOpen, Video, Award, MessageSquare, Terminal } from 'lucide-react';
import MatchScoreRing from '../components/MatchScoreRing';

export default function ActionPlan({ showToast }) {
  const [selectedRole, setSelectedRole] = useState('job-1');
  const [completedItems, setCompletedItems] = useState({ 'item-1': true });
  const [filterStatus, setFilterStatus] = useState('all'); // 'all' | 'pending' | 'completed' | 'high'
  const [showAddModal, setShowAddModal] = useState(false);

  // Custom User Tasks State
  const [customCategory, setCustomCategory] = useState('');
  const [customGap, setCustomGap] = useState('');
  const [customRecommendation, setCustomRecommendation] = useState('');
  const [customTimeframe, setCustomTimeframe] = useState('3 Days');

  const roles = [
    { id: 'job-1', title: 'Software Engineer - Backend', company: 'Superset Inc.', score: 92 },
    { id: 'job-2', title: 'SDE II - Full Stack', company: 'Airmeet', score: 88 },
    { id: 'job-3', title: 'Backend Developer', company: 'Razorpay', score: 85 },
    { id: 'job-4', title: 'Senior Software Engineer', company: 'Google Cloud', score: 95 },
  ];

  const [actionItems, setActionItems] = useState([
    {
      id: 'item-1',
      category: 'Technical Skill Up-skilling',
      priority: 'High Priority',
      priorityColor: 'bg-rose-500/15 text-rose-400 border-rose-500/30',
      icon: Code,
      gap: 'Missing hands-on experience with Redis caching & Kafka message queues.',
      recommendation: 'Build a lightweight FastAPI microservice featuring Redis cache-aside & PostgreSQL indexing.',
      timeframe: '5 Days',
      linkText: 'View FastAPI + Redis Guide'
    },
    {
      id: 'item-2',
      category: 'Resume & Quantifiable Impact',
      priority: 'High Priority',
      priorityColor: 'bg-rose-500/15 text-rose-400 border-rose-500/30',
      icon: FileText,
      gap: 'Resume project descriptions lack quantifiable performance metrics.',
      recommendation: 'Rewrite bullet points: "Reduced API latency by 40% (450ms → 68ms) handling 2M+ daily active requests."',
      timeframe: '2 Days',
      linkText: 'Open Resume Editor'
    },
    {
      id: 'item-3',
      category: 'System Design & Distributed Rate Limiting',
      priority: 'High Priority',
      priorityColor: 'bg-rose-500/15 text-rose-400 border-rose-500/30',
      icon: Target,
      gap: 'Need deeper practice on distributed API Rate Limiting & WebSockets.',
      recommendation: 'Implement a Token Bucket rate-limiter middleware in Python and document architecture in GitHub README.',
      timeframe: '4 Days',
      linkText: 'System Design Patterns'
    },
    {
      id: 'item-4',
      category: 'GitHub Portfolio & CI/CD Pipelines',
      priority: 'Medium Priority',
      priorityColor: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
      icon: ShieldCheck,
      gap: 'Public repos lack Pytest unit test coverage and GitHub Actions workflows.',
      recommendation: 'Add .github/workflows/ci.yml with automated Pytest testing and Ruff linting rules.',
      timeframe: '3 Days',
      linkText: 'GitHub Actions Template'
    },
    {
      id: 'item-5',
      category: 'Mock Interview Prep & Behavioral Q&A',
      priority: 'Medium Priority',
      priorityColor: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
      icon: MessageSquare,
      gap: 'Practice STAR format answers for behavioral leadership rounds at top tech companies.',
      recommendation: 'Prepare 3 STAR stories highlighting conflict resolution, production outage recovery, and cross-team execution.',
      timeframe: '3 Days',
      linkText: 'Practice STAR Simulator'
    },
    {
      id: 'item-6',
      category: 'DSA & High-Frequency LeetCode Patterns',
      priority: 'Medium Priority',
      priorityColor: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
      icon: Terminal,
      gap: 'Solve 15 high-frequency Graph & Dynamic Programming problems for technical screening.',
      recommendation: 'Complete Top 15 DP & BFS/DFS problems on LeetCode with space-time complexity analysis.',
      timeframe: '4 Days',
      linkText: 'LeetCode Problem List'
    }
  ]);

  const toggleComplete = (id) => {
    setCompletedItems(prev => {
      const updated = { ...prev, [id]: !prev[id] };
      if (showToast) {
        showToast(
          updated[id] ? 'Action Task Completed! 🎉' : 'Task Status Updated',
          updated[id] ? 'Great progress! Added +5% towards your placement readiness target.' : 'Marked task as pending.'
        );
      }
      return updated;
    });
  };

  const handleAddCustomTask = (e) => {
    e.preventDefault();
    if (!customCategory.trim() || !customRecommendation.trim()) return;

    const newTask = {
      id: `custom-${Date.now()}`,
      category: customCategory.trim(),
      priority: 'High Priority',
      priorityColor: 'bg-rose-500/15 text-rose-400 border-rose-500/30',
      icon: Sparkles,
      gap: customGap.trim() || 'Custom candidate skill enhancement target.',
      recommendation: customRecommendation.trim(),
      timeframe: customTimeframe,
      linkText: 'View Custom Details'
    };

    setActionItems([newTask, ...actionItems]);
    setShowAddModal(false);
    setCustomCategory('');
    setCustomGap('');
    setCustomRecommendation('');

    if (showToast) {
      showToast('Custom Task Added! 🎯', `Added "${newTask.category}" to your placement roadmap.`);
    }
  };

  const handleRemoveTask = (e, id) => {
    e.stopPropagation();
    setActionItems(actionItems.filter(item => item.id !== id));
    if (showToast) showToast('Task Removed 🗑️', 'Goal removed from your action plan.');
  };

  const currentRole = roles.find(r => r.id === selectedRole) || roles[0];
  const completedCount = Object.values(completedItems).filter(Boolean).length;
  const totalTasks = actionItems.length;
  const progressPercent = Math.round((completedCount / totalTasks) * 100);

  const filteredItems = actionItems.filter(item => {
    const isDone = !!completedItems[item.id];
    if (filterStatus === 'completed') return isDone;
    if (filterStatus === 'pending') return !isDone;
    if (filterStatus === 'high') return item.priority.includes('High');
    return true;
  });

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      
      {/* Title Header & Target Role Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-slate-800 dark:text-white tracking-tight flex items-center gap-2.5">
            Student Action Plan & Placement Roadmap
            <span className="text-xs bg-indigo-500/15 text-indigo-400 font-extrabold px-3 py-1 rounded-full border border-indigo-500/30 shadow-sm">
              {progressPercent}% Placement Ready
            </span>
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Personalized step-by-step improvement strategy to bridge technical gaps and elevate your candidate tier.
          </p>
        </div>

        {/* Target Position Dropdown Selector */}
        <div className="flex items-center gap-3">
          <label className="text-xs font-extrabold text-slate-400 shrink-0">Target Position:</label>
          <select
            value={selectedRole}
            onChange={(e) => setSelectedRole(e.target.value)}
            className="bg-slate-900 border border-slate-800 text-xs font-black text-white px-4 py-2.5 rounded-2xl focus:outline-none focus:ring-2 focus:ring-indigo-500 cursor-pointer shadow-md"
          >
            {roles.map(r => (
              <option key={r.id} value={r.id} className="bg-slate-900">
                {r.title} ({r.company}) — {r.score}% Match
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Hero Overview Card with Live Progress Engine */}
      <div className="bg-gradient-to-br from-slate-900 via-indigo-950/60 to-slate-900 border border-slate-800 p-6 rounded-3xl shadow-xl flex flex-col md:flex-row items-center justify-between gap-6 relative overflow-hidden backdrop-blur-xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none"></div>

        <div className="space-y-3 relative z-10 w-full md:w-2/3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-extrabold bg-indigo-500/20 text-indigo-300 px-3 py-1 rounded-full border border-indigo-500/30 shadow-sm">
              Target: {currentRole.title} @ {currentRole.company}
            </span>
            <span className="text-xs font-bold text-amber-400 bg-amber-500/10 px-2.5 py-0.5 rounded-full border border-amber-500/20">
              Current Fit: {currentRole.score}%
            </span>
          </div>

          <h2 className="text-xl font-black text-white">
            Action Strategy for Placement Excellence
          </h2>
          <p className="text-xs text-indigo-200 leading-relaxed">
            Completing the remaining <span className="font-extrabold text-amber-400">{totalTasks - completedCount} action items</span> will elevate your match score from <span className="font-extrabold text-amber-400">{currentRole.score}%</span> to <span className="font-extrabold text-emerald-400">98% (Top Candidate Tier)</span>.
          </p>

          {/* Animated Progress Bar */}
          <div className="space-y-1.5 pt-2">
            <div className="flex justify-between text-xs font-extrabold text-slate-300">
              <span>Overall Roadmap Completion</span>
              <span className="text-emerald-400">{completedCount} of {totalTasks} Tasks Done ({progressPercent}%)</span>
            </div>
            <div className="h-3 bg-slate-950 rounded-full overflow-hidden border border-slate-800 p-0.5">
              <div
                className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-emerald-400 rounded-full transition-all duration-500 shadow-md"
                style={{ width: `${progressPercent}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* Right Metric Box */}
        <div className="bg-slate-950/80 border border-slate-800 p-5 rounded-2xl flex flex-col items-center justify-center text-center shrink-0 w-full md:w-auto relative z-10 space-y-2">
          <MatchScoreRing score={Math.min(currentRole.score + Math.round(progressPercent * 0.08), 98)} size={72} strokeWidth={6} />
          <span className="text-xs font-black text-emerald-400">Predicted Score: {Math.min(currentRole.score + Math.round(progressPercent * 0.08), 98)}%</span>
          <p className="text-[10px] text-slate-400">Top 1% Candidate Pool</p>
        </div>
      </div>

      {/* Control Panel: Filters & Add Custom Goal */}
      <div className="bg-white dark:bg-[#0F172A] border border-slate-200 dark:border-slate-800 p-3 rounded-2xl flex flex-col sm:flex-row items-center justify-between gap-3 shadow-sm">
        
        {/* Status Filter Buttons */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-extrabold text-slate-400 ml-1">Filter Tasks:</span>
          
          <button
            onClick={() => setFilterStatus('all')}
            className={`px-3 py-1.5 text-xs font-extrabold rounded-xl transition-all cursor-pointer ${
              filterStatus === 'all' ? 'bg-indigo-600 text-white shadow-md' : 'bg-slate-900 text-slate-400 hover:text-white'
            }`}
          >
            All Tasks ({totalTasks})
          </button>

          <button
            onClick={() => setFilterStatus('pending')}
            className={`px-3 py-1.5 text-xs font-extrabold rounded-xl transition-all cursor-pointer ${
              filterStatus === 'pending' ? 'bg-indigo-600 text-white shadow-md' : 'bg-slate-900 text-slate-400 hover:text-white'
            }`}
          >
            Pending ({totalTasks - completedCount})
          </button>

          <button
            onClick={() => setFilterStatus('completed')}
            className={`px-3 py-1.5 text-xs font-extrabold rounded-xl transition-all cursor-pointer ${
              filterStatus === 'completed' ? 'bg-indigo-600 text-white shadow-md' : 'bg-slate-900 text-slate-400 hover:text-white'
            }`}
          >
            Completed ({completedCount})
          </button>

          <button
            onClick={() => setFilterStatus('high')}
            className={`px-3 py-1.5 text-xs font-extrabold rounded-xl transition-all cursor-pointer ${
              filterStatus === 'high' ? 'bg-indigo-600 text-white shadow-md' : 'bg-slate-900 text-slate-400 hover:text-white'
            }`}
          >
            High Priority
          </button>
        </div>

        {/* Add Goal Button */}
        <button
          onClick={() => setShowAddModal(true)}
          className="bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-extrabold text-xs px-4 py-2 rounded-xl transition-all shadow-md flex items-center gap-1.5 cursor-pointer shrink-0"
        >
          <Plus className="w-4 h-4" /> Add Custom Goal
        </button>
      </div>

      {/* Action Plan Grid Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {filteredItems.map((item) => {
          const Icon = item.icon || Sparkles;
          const isDone = !!completedItems[item.id];

          return (
            <div
              key={item.id}
              onClick={() => toggleComplete(item.id)}
              className={`group relative bg-white dark:bg-[#0F172A] border rounded-3xl p-6 shadow-sm transition-all duration-300 flex flex-col justify-between space-y-4 cursor-pointer hover:shadow-xl ${
                isDone
                  ? 'border-emerald-500/40 opacity-85 bg-emerald-950/10'
                  : 'border-slate-200 dark:border-slate-800/90 hover:border-indigo-500/50'
              }`}
            >
              {/* Top Glowing Border Line */}
              <div className={`absolute top-0 left-0 right-0 h-1 rounded-t-3xl transition-all ${
                isDone ? 'bg-emerald-500' : 'bg-gradient-to-r from-indigo-500 to-purple-500 opacity-40 group-hover:opacity-100'
              }`}></div>

              <div className="space-y-3">
                {/* Header Row: Category Badge & Status Indicator */}
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <div className="w-9 h-9 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 flex items-center justify-center shrink-0">
                      <Icon className="w-4 h-4" />
                    </div>
                    <span className={`text-[10px] font-black uppercase px-2.5 py-1 rounded-full border ${item.priorityColor}`}>
                      {item.priority}
                    </span>
                  </div>

                  <div className="flex items-center gap-1">
                    {/* Delete Task Button */}
                    <button
                      onClick={(e) => handleRemoveTask(e, item.id)}
                      className="p-1 text-slate-500 hover:text-rose-400 transition-colors"
                      title="Remove task"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                    
                    {isDone ? (
                      <CheckCircle2 className="w-6 h-6 text-emerald-400 fill-emerald-400/20" />
                    ) : (
                      <Circle className="w-6 h-6 text-slate-600 hover:text-indigo-400 transition-colors" />
                    )}
                  </div>
                </div>

                <h3 className="text-base font-black text-slate-900 dark:text-white group-hover:text-indigo-400 transition-colors leading-snug">
                  {item.category}
                </h3>

                {/* Gap & Recommendation Boxes */}
                <div className="space-y-2 text-xs">
                  <div className="bg-slate-50 dark:bg-slate-900/80 p-3 rounded-2xl border border-slate-200 dark:border-slate-800/80">
                    <span className="text-[10px] font-extrabold text-rose-400 uppercase tracking-wider block mb-0.5">
                      Identified Gap:
                    </span>
                    <p className="text-slate-700 dark:text-slate-300">{item.gap}</p>
                  </div>

                  <div className="bg-indigo-500/10 border border-indigo-500/20 p-3 rounded-2xl">
                    <span className="text-[10px] font-extrabold text-indigo-400 uppercase tracking-wider block mb-0.5">
                      Recommended Action:
                    </span>
                    <p className="text-slate-800 dark:text-slate-200 font-medium leading-relaxed">{item.recommendation}</p>
                  </div>
                </div>
              </div>

              {/* Card Footer Bar */}
              <div className="pt-3 border-t border-slate-100 dark:border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400 font-semibold">
                <span>Timeframe: <span className="text-slate-300 font-bold">{item.timeframe}</span></span>
                <span className={`font-bold flex items-center gap-1 ${isDone ? 'text-emerald-400' : 'text-indigo-400'}`}>
                  {isDone ? 'Completed ✅' : 'Click to complete'} <ArrowRight className="w-3.5 h-3.5" />
                </span>
              </div>

            </div>
          );
        })}
      </div>

      {/* Add Custom Goal Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 bg-slate-950/85 backdrop-blur-xl flex items-center justify-center p-4 animate-in fade-in duration-200">
          <div className="bg-white dark:bg-[#0F172A] border border-slate-200 dark:border-slate-800 w-full max-w-md rounded-3xl shadow-2xl p-6 relative space-y-4">
            <button
              onClick={() => setShowAddModal(false)}
              className="absolute top-4 right-4 p-2 rounded-full text-slate-400 hover:text-white hover:bg-slate-800 transition-all cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
              <Plus className="w-5 h-5 text-indigo-400" />
              <h3 className="text-base font-black text-white">Add Custom Up-skilling Goal</h3>
            </div>

            <form onSubmit={handleAddCustomTask} className="space-y-3 text-xs">
              <div>
                <label className="font-bold text-slate-300 block mb-1">Goal Category / Title</label>
                <input
                  type="text"
                  required
                  value={customCategory}
                  onChange={(e) => setCustomCategory(e.target.value)}
                  placeholder="e.g., Learn Redis Caching & Cache-Aside Strategy"
                  className="w-full bg-slate-900 border border-slate-800 text-white px-3 py-2 rounded-xl"
                />
              </div>

              <div>
                <label className="font-bold text-slate-300 block mb-1">Identified Skill Gap</label>
                <input
                  type="text"
                  value={customGap}
                  onChange={(e) => setCustomGap(e.target.value)}
                  placeholder="e.g., Need hands-on experience with in-memory caching"
                  className="w-full bg-slate-900 border border-slate-800 text-white px-3 py-2 rounded-xl"
                />
              </div>

              <div>
                <label className="font-bold text-slate-300 block mb-1">Recommended Action Step</label>
                <textarea
                  rows={3}
                  required
                  value={customRecommendation}
                  onChange={(e) => setCustomRecommendation(e.target.value)}
                  placeholder="e.g., Build FastAPI microservice with Redis cache and measure 50% latency drop..."
                  className="w-full bg-slate-900 border border-slate-800 text-white px-3 py-2 rounded-xl"
                />
              </div>

              <div>
                <label className="font-bold text-slate-300 block mb-1">Target Timeframe</label>
                <select
                  value={customTimeframe}
                  onChange={(e) => setCustomTimeframe(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 text-white px-3 py-2 rounded-xl"
                >
                  <option value="2 Days">2 Days</option>
                  <option value="3 Days">3 Days</option>
                  <option value="5 Days">5 Days</option>
                  <option value="1 Week">1 Week</option>
                  <option value="2 Weeks">2 Weeks</option>
                </select>
              </div>

              <div className="pt-2 flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold text-xs px-4 py-2.5 rounded-xl cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs px-5 py-2.5 rounded-xl shadow-md cursor-pointer"
                >
                  Add Task to Roadmap
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

    </div>
  );
}
