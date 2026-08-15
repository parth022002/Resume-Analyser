import React, { useEffect } from 'react';
import { CheckCircle2, Sparkles, X, AlertCircle, Info } from 'lucide-react';

export default function Toast({ toast, onClose }) {
  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => {
        onClose();
      }, 3500);
      return () => clearTimeout(timer);
    }
  }, [toast, onClose]);

  if (!toast) return null;

  return (
    <div className="fixed top-5 right-5 z-50 animate-in fade-in slide-in-from-top-4 duration-300">
      <div className="bg-slate-900/90 dark:bg-slate-900/95 backdrop-blur-md border border-indigo-500/30 text-white rounded-2xl p-4 shadow-2xl flex items-center gap-3 max-w-sm">
        <div className="p-2 rounded-xl bg-gradient-to-tr from-violet-600 to-indigo-600 text-white">
          <Sparkles className="w-4 h-4 fill-white" />
        </div>
        <div className="flex-1">
          <h4 className="text-xs font-bold text-white">{toast.title || "Notification"}</h4>
          <p className="text-[11px] text-slate-300 font-medium">{toast.message}</p>
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-white p-1">
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
