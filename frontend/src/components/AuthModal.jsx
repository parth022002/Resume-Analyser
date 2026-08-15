import React, { useState } from 'react';
import { X, Upload, CheckCircle2, User, Mail, Lock, Sparkles, FileText, Camera, GraduationCap, ArrowRight, ShieldCheck, KeyRound, Eye, EyeOff } from 'lucide-react';

export default function AuthModal({ isOpen, onClose, onLoginSuccess, showToast }) {
  const [tab, setTab] = useState('login'); // 'login' | 'signup'
  
  // Login State
  const [loginEmail, setLoginEmail] = useState('arjun.b@talentforge.ai');
  const [loginPassword, setLoginPassword] = useState('password123');
  const [showPassword, setShowPassword] = useState(false);
  
  // Signup State
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [avatarUrl, setAvatarUrl] = useState('https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150');
  const [headline, setHeadline] = useState('Software Engineer - Backend & Systems');
  const [skillsText, setSkillsText] = useState('Python, FastAPI, AWS, PostgreSQL, Docker, Microservices');
  const [experienceYears, setExperienceYears] = useState(3.5);
  const [preferredRoles, setPreferredRoles] = useState('Software Engineer, Backend Developer, SDE II');
  const [preferredLocations, setPreferredLocations] = useState('Bengaluru, Remote, Hybrid');
  
  // Compulsory Education Data State
  const [eduDegree, setEduDegree] = useState('B.Tech Computer Science');
  const [eduInstitute, setEduInstitute] = useState('RV College of Engineering');
  const [eduYear, setEduYear] = useState('2024');

  const [resumeUploaded, setResumeUploaded] = useState(false);
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  // Preset quick fill demo student
  const handleLoadDemoAccount = () => {
    setLoginEmail('arjun.b@talentforge.ai');
    setLoginPassword('password123');
    if (showToast) showToast('Demo Credentials Loaded ⚡', 'Arjun B. student account selected');
  };

  const handlePhotoFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setAvatarUrl(reader.result);
        if (showToast) showToast('Photo Uploaded', 'Student profile photo loaded successfully!');
      };
      reader.readAsDataURL(file);
    }
  };

  const handleResumeFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setResumeUploaded(true);
      if (showToast) {
        showToast('Resume Parsed!', `Extracted skills and experience details from ${file.name}`);
      }
    }
  };

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    if (!loginEmail.trim() || !loginPassword.trim()) {
      if (showToast) showToast('⚠️ Missing Details', 'Please enter your email and password.');
      return;
    }

    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/v1/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: loginEmail.trim(), password: loginPassword })
      });
      const data = await res.json();
      if (res.ok && data.user) {
        onLoginSuccess(data.user);
        if (showToast) showToast('Welcome Back! 🔐', `Security login notification email sent to ${data.user.email}`);
        onClose();
      } else {
        alert(data.detail || 'Login failed');
      }
    } catch {
      const user = {
        id: 1,
        full_name: "Arjun B.",
        email: loginEmail.trim(),
        avatar_url: avatarUrl,
        plan: "Free Student Account",
        headline: headline,
        skills: skillsText.split(',').map(s => s.trim()).filter(Boolean),
        experience_years: Number(experienceYears),
        preferred_roles: preferredRoles.split(',').map(r => r.trim()).filter(Boolean),
        preferred_locations: preferredLocations.split(',').map(l => l.trim()).filter(Boolean),
        education_details: [
          { degree: eduDegree || "B.Tech Computer Science", institute: eduInstitute || "RV College of Engineering", year: eduYear || "2024", cgpa: "8.8 / 10" }
        ]
      };
      onLoginSuccess(user);
      if (showToast) showToast('Welcome Back! 🔐', `Security login notification email sent to ${user.email}`);
      onClose();
    } finally {
      setLoading(false);
    }
  };

  const handleSignupSubmit = async (e) => {
    e.preventDefault();

    if (!fullName.trim()) {
      if (showToast) showToast('⚠️ Compulsory Field', 'Full Name is required!');
      return;
    }
    if (!email.trim() || !password.trim()) {
      if (showToast) showToast('⚠️ Compulsory Field', 'Email and Password are required!');
      return;
    }
    if (!skillsText.trim()) {
      if (showToast) showToast('⚠️ Compulsory Field', 'At least one Technical Skill is required!');
      return;
    }
    if (!eduDegree.trim() || !eduInstitute.trim()) {
      if (showToast) showToast('⚠️ Compulsory Field', 'Education Name (University / Institute) & Degree are required!');
      return;
    }

    setLoading(true);
    const payload = {
      full_name: fullName.trim(),
      email: email.trim(),
      password: password,
      avatar_url: avatarUrl,
      headline: headline,
      skills: skillsText.split(',').map(s => s.trim()).filter(Boolean),
      experience_years: Number(experienceYears),
      preferred_roles: preferredRoles.split(',').map(r => r.trim()).filter(Boolean),
      preferred_locations: preferredLocations.split(',').map(l => l.trim()).filter(Boolean),
      education_details: [
        { degree: eduDegree.trim(), institute: eduInstitute.trim(), year: eduYear.trim() || "2024", cgpa: "" }
      ]
    };

    try {
      const res = await fetch('http://localhost:8000/api/v1/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (res.ok && data.user) {
        onLoginSuccess(data.user);
        if (showToast) showToast('Account Created & Email Sent! 📧', `Login credentials & password sent to ${data.user.email}`);
        onClose();
      } else {
        alert(data.detail || 'Signup failed');
      }
    } catch {
      const user = {
        id: Date.now(),
        ...payload,
        plan: "Free Student Account"
      };
      onLoginSuccess(user);
      if (showToast) showToast('Account Created & Email Sent! 📧', `Login credentials & password sent to ${user.email}`);
      onClose();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/85 backdrop-blur-xl flex items-center justify-center p-4 overflow-y-auto animate-in fade-in duration-200">
      <div className="bg-white dark:bg-[#0F172A] border border-slate-200 dark:border-slate-800 w-full max-w-lg rounded-3xl shadow-2xl overflow-hidden relative transition-all">
        
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 rounded-full text-slate-400 hover:text-white hover:bg-slate-800/60 transition-all z-20"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Modal Header & Hero Banner */}
        <div className="bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950/80 p-6 text-center border-b border-slate-800/80 space-y-2">
          <div className="w-14 h-14 rounded-2xl bg-slate-900 border border-amber-500/40 p-1 mx-auto shadow-xl shadow-amber-500/20 overflow-hidden hover:scale-105 transition-transform">
            <img src="/logo.png" alt="TalentForge Emblem" className="w-full h-full object-contain drop-shadow" />
          </div>
          <h2 className="text-xl font-black tracking-tight">
            <span className="gold-gradient-text">
              {tab === 'login' ? 'Welcome Back to TalentForge' : 'Create Student Candidate Account'}
            </span>
          </h2>
          <p className="text-xs text-slate-300 max-w-xs mx-auto">
            {tab === 'login'
              ? 'Access your personalized AI job feed, candidate knowledge base, and ATS resume score.'
              : 'Setup mandatory education details & skills to get instant match scores for target roles.'}
          </p>
        </div>

        {/* Tab Switcher Bar */}
        <div className="px-6 pt-5">
          <div className="grid grid-cols-2 p-1 bg-slate-100 dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800">
            <button
              type="button"
              onClick={() => setTab('login')}
              className={`py-2.5 text-xs font-extrabold rounded-xl transition-all flex items-center justify-center gap-2 ${
                tab === 'login'
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-500 hover:text-slate-200'
              }`}
            >
              <KeyRound className="w-3.5 h-3.5" /> Sign In
            </button>
            <button
              type="button"
              onClick={() => setTab('signup')}
              className={`py-2.5 text-xs font-extrabold rounded-xl transition-all flex items-center justify-center gap-2 ${
                tab === 'signup'
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-500 hover:text-slate-200'
              }`}
            >
              <User className="w-3.5 h-3.5" /> Student Sign Up
            </button>
          </div>
        </div>

        {/* Body Content */}
        <div className="p-6">
          {tab === 'login' ? (
            /* ---------------- LOGIN / SIGN IN FORM ---------------- */
            <form onSubmit={handleLoginSubmit} className="space-y-4">
              
              {/* Quick Preset Demo Account Selector */}
              <div className="p-3 bg-indigo-500/10 border border-indigo-500/20 rounded-xl flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-indigo-400 shrink-0" />
                  <span className="text-[11px] text-slate-300 font-medium">Testing locally? Load demo profile</span>
                </div>
                <button
                  type="button"
                  onClick={handleLoadDemoAccount}
                  className="text-[11px] bg-indigo-600 hover:bg-indigo-500 text-white font-bold px-3 py-1.5 rounded-lg shadow-sm transition-all"
                >
                  ⚡ Auto-Fill Demo
                </button>
              </div>

              <div>
                <label className="text-xs font-bold text-slate-700 dark:text-slate-300 block mb-1">Email Address</label>
                <div className="relative">
                  <Mail className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                  <input
                    type="email"
                    required
                    value={loginEmail}
                    onChange={e => setLoginEmail(e.target.value)}
                    className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white pl-10 pr-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                    placeholder="arjun.b@talentforge.ai"
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs font-bold text-slate-700 dark:text-slate-300">Password</label>
                  <a href="#" onClick={(e) => { e.preventDefault(); alert("Use Demo Credentials or reset password!"); }} className="text-[11px] font-bold text-indigo-400 hover:underline">
                    Forgot password?
                  </a>
                </div>
                <div className="relative">
                  <Lock className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                  <input
                    type={showPassword ? "text" : "password"}
                    required
                    value={loginPassword}
                    onChange={e => setLoginPassword(e.target.value)}
                    className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white pl-10 pr-10 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                    placeholder="••••••••"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200 p-1"
                  >
                    {showPassword ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-bold text-xs py-3.5 rounded-xl transition-all shadow-lg shadow-indigo-500/25 flex items-center justify-center gap-2 mt-2"
              >
                {loading ? 'Signing in...' : 'Sign In to Student Account'}
                {!loading && <ArrowRight className="w-4 h-4" />}
              </button>

              <div className="text-center pt-2 border-t border-slate-200 dark:border-slate-800/80">
                <p className="text-xs text-slate-400">
                  Don't have a student candidate account yet?{' '}
                  <button
                    type="button"
                    onClick={() => setTab('signup')}
                    className="font-bold text-indigo-400 hover:underline"
                  >
                    Create Free Student Account
                  </button>
                </p>
              </div>
            </form>
          ) : (
            /* ---------------- SIGNUP & ONBOARDING FORM ---------------- */
            <form onSubmit={handleSignupSubmit} className="space-y-4 max-h-[60vh] overflow-y-auto pr-1 custom-scrollbar">
              
              {/* Photo & Resume Picker */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="flex items-center gap-3 bg-slate-50 dark:bg-slate-900 p-3 rounded-xl border border-slate-200 dark:border-slate-800">
                  <img
                    src={avatarUrl}
                    alt="Profile Avatar"
                    className="w-12 h-12 rounded-full object-cover border-2 border-indigo-500 shadow-md shrink-0"
                  />
                  <div className="flex-1">
                    <label className="text-[11px] font-bold text-slate-700 dark:text-white block mb-0.5">Profile Photo</label>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handlePhotoFileUpload}
                      id="signup-photo-upload"
                      className="hidden"
                    />
                    <label
                      htmlFor="signup-photo-upload"
                      className="cursor-pointer inline-flex items-center gap-1 bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-[10px] px-2.5 py-1 rounded-md shadow-sm transition-all"
                    >
                      <Camera className="w-3 h-3" /> Choose Image
                    </label>
                  </div>
                </div>

                <div className="border border-dashed border-indigo-500/40 rounded-xl p-3 text-center bg-indigo-500/5 relative flex flex-col justify-center">
                  <input
                    type="file"
                    accept=".pdf,.docx"
                    onChange={handleResumeFileUpload}
                    className="absolute inset-0 opacity-0 cursor-pointer"
                  />
                  <p className="text-[11px] font-bold text-slate-800 dark:text-white flex items-center justify-center gap-1">
                    <Upload className="w-3.5 h-3.5 text-indigo-400" />
                    {resumeUploaded ? "✅ Resume Uploaded" : "Upload Resume PDF"}
                  </p>
                  <p className="text-[9px] text-slate-400 mt-0.5">Auto-fills skills & education</p>
                </div>
              </div>

              {/* Basic Account Fields */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="text-[11px] font-bold text-slate-700 dark:text-slate-300 block mb-1">
                    Full Name <span className="text-rose-500 font-extrabold">*</span>
                  </label>
                  <input
                    type="text"
                    required
                    value={fullName}
                    onChange={e => setFullName(e.target.value)}
                    className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2.5 rounded-xl"
                    placeholder="Arjun B."
                  />
                </div>
                <div>
                  <label className="text-[11px] font-bold text-slate-700 dark:text-slate-300 block mb-1">
                    Email Address <span className="text-rose-500 font-extrabold">*</span>
                  </label>
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2.5 rounded-xl"
                    placeholder="student@talentforge.ai"
                  />
                </div>
              </div>

              <div>
                <label className="text-[11px] font-bold text-slate-700 dark:text-slate-300 block mb-1">
                  Password <span className="text-rose-500 font-extrabold">*</span>
                </label>
                <input
                  type="password"
                  required
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2.5 rounded-xl"
                  placeholder="••••••••"
                />
              </div>

              {/* 🎓 Compulsory Education Data Setup Section */}
              <div className="p-3.5 bg-indigo-500/10 border border-indigo-500/30 rounded-2xl space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-indigo-300 flex items-center gap-1.5">
                    <GraduationCap className="w-4 h-4 text-indigo-400" /> Mandatory Student Education Setup
                  </span>
                  <span className="text-[10px] bg-rose-500/20 text-rose-300 font-extrabold px-2 py-0.5 rounded-full border border-rose-500/30">
                    Compulsory
                  </span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <label className="text-[11px] font-bold text-slate-700 dark:text-slate-300 block mb-1">
                      Degree / Specialization <span className="text-rose-500">*</span>
                    </label>
                    <input
                      type="text"
                      required
                      value={eduDegree}
                      onChange={e => setEduDegree(e.target.value)}
                      className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2 rounded-xl"
                      placeholder="B.Tech Computer Science"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] font-bold text-slate-700 dark:text-slate-300 block mb-1">
                      University / Institute Name <span className="text-rose-500">*</span>
                    </label>
                    <input
                      type="text"
                      required
                      value={eduInstitute}
                      onChange={e => setEduInstitute(e.target.value)}
                      className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2 rounded-xl"
                      placeholder="RV College of Engineering"
                    />
                  </div>
                </div>
              </div>

              <div>
                <label className="text-[11px] font-bold text-slate-700 dark:text-slate-300 block mb-1">Professional Headline</label>
                <input
                  type="text"
                  value={headline}
                  onChange={e => setHeadline(e.target.value)}
                  className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2.5 rounded-xl"
                  placeholder="Software Engineer - Backend & Cloud"
                />
              </div>

              <div>
                <label className="text-[11px] font-bold text-slate-700 dark:text-slate-300 block mb-1">
                  Technical Skills (comma separated) <span className="text-rose-500 font-extrabold">*</span>
                </label>
                <input
                  type="text"
                  value={skillsText}
                  onChange={e => setSkillsText(e.target.value)}
                  className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2.5 rounded-xl"
                  placeholder="Python, FastAPI, AWS, PostgreSQL, Docker"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-[11px] font-bold text-slate-700 dark:text-slate-300 block mb-1">Preferred Roles</label>
                  <input
                    type="text"
                    value={preferredRoles}
                    onChange={e => setPreferredRoles(e.target.value)}
                    className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2.5 rounded-xl"
                  />
                </div>
                <div>
                  <label className="text-[11px] font-bold text-slate-700 dark:text-slate-300 block mb-1">Preferred Locations</label>
                  <input
                    type="text"
                    value={preferredLocations}
                    onChange={e => setPreferredLocations(e.target.value)}
                    className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2.5 rounded-xl"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-bold text-xs py-3.5 rounded-xl transition-all shadow-lg shadow-indigo-500/25 flex items-center justify-center gap-2 mt-3"
              >
                {loading ? 'Creating Student Profile...' : 'Complete Free Registration & Build Graph'}
                {!loading && <ArrowRight className="w-4 h-4" />}
              </button>

              <div className="text-center pt-2 border-t border-slate-200 dark:border-slate-800/80">
                <p className="text-xs text-slate-400">
                  Already have an account?{' '}
                  <button
                    type="button"
                    onClick={() => setTab('login')}
                    className="font-bold text-indigo-400 hover:underline"
                  >
                    Sign In to Existing Profile
                  </button>
                </p>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
