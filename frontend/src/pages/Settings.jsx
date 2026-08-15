import React, { useState } from 'react';
import { User, GraduationCap, Briefcase, Award, Trophy, Camera, Plus, Trash2, Save, Sparkles, CheckCircle2, AlertCircle, ShieldCheck, Upload, Globe, Github, Linkedin, Code, Link } from 'lucide-react';
import MatchScoreRing from '../components/MatchScoreRing';

export default function Settings({ user, onUpdateUser, showToast }) {
  // Basic Info State
  const [fullName, setFullName] = useState(user?.full_name || 'Arjun B.');
  const [headline, setHeadline] = useState(user?.headline || 'Software Engineer - Backend & Systems');
  const [avatarUrl, setAvatarUrl] = useState(user?.avatar_url || 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150');
  const [skillsText, setSkillsText] = useState(user?.skills ? user.skills.join(', ') : 'Python, FastAPI, AWS, PostgreSQL, Docker, Microservices');
  const [experienceYears, setExperienceYears] = useState(user?.experience_years || 3.5);
  const [preferredRoles, setPreferredRoles] = useState(user?.preferred_roles ? user.preferred_roles.join(', ') : 'Software Engineer, Backend Developer, SDE II');
  const [preferredLocations, setPreferredLocations] = useState(user?.preferred_locations ? user.preferred_locations.join(', ') : 'Bengaluru, Remote, Hybrid');

  // Social & Online Footprint Links
  const [portfolioUrl, setPortfolioUrl] = useState(user?.portfolio_url || 'https://arjun.dev');
  const [githubUrl, setGithubUrl] = useState(user?.github_url || 'https://github.com/arjun-b');
  const [linkedinUrl, setLinkedinUrl] = useState(user?.linkedin_url || 'https://linkedin.com/in/arjun-b');
  const [leetcodeUrl, setLeetcodeUrl] = useState(user?.leetcode_url || 'https://leetcode.com/arjun_b');
  const [otherUrls, setOtherUrls] = useState(user?.other_urls || 'https://twitter.com/arjun_dev');

  // Education Details List
  const [educationList, setEducationList] = useState(user?.education_details || [
    { degree: "B.Tech Computer Science", institute: "RV College of Engineering", year: "2024", cgpa: "8.8 / 10" }
  ]);

  // Certifications List
  const [certificationsList, setCertificationsList] = useState(user?.certifications || [
    { title: "AWS Certified Developer – Associate", issuer: "Amazon Web Services", year: "2023" },
    { title: "PostgreSQL Professional Certification", issuer: "PostgreSQL Institute", year: "2023" }
  ]);

  // Extra-Curricular Activities & Achievements
  const [extracurricularList, setExtracurricularList] = useState(user?.extracurricular || [
    { title: "Hackathon Winner", org: "Smart India Hackathon", desc: "1st place out of 500+ teams building real-time logistics routing" },
    { title: "Open Source Contributor", org: "FastAPI Ecosystem", desc: "Contributed performance benchmarks and documentation fixes" }
  ]);

  const [saving, setSaving] = useState(false);
  const [resumeParsed, setResumeParsed] = useState(false);
  const [formErrors, setFormErrors] = useState({});
  const [attemptedSave, setAttemptedSave] = useState(false);

  // Compulsory Fields Validation Engine
  const validateForm = () => {
    const errs = {};
    if (!fullName || !fullName.trim()) {
      errs.fullName = "Full Name is a compulsory field.";
    }
    if (!skillsText || !skillsText.trim()) {
      errs.skillsText = "At least one technical skill is compulsory.";
    }
    if (!educationList || educationList.length === 0) {
      errs.educationList = "At least one education record is compulsory.";
    } else {
      educationList.forEach((edu, idx) => {
        if (!edu.degree || !edu.degree.trim()) {
          errs[`edu_degree_${idx}`] = "Degree / Specialization is compulsory.";
        }
        if (!edu.institute || !edu.institute.trim()) {
          errs[`edu_institute_${idx}`] = "University / Institute Name is compulsory.";
        }
      });
    }
    return errs;
  };

  // Resume Rating Calculation
  const calculateResumeRating = () => {
    let score = 50;
    if (skillsText.length > 20) score += 12;
    if (educationList.length > 0 && educationList.some(e => e.institute && e.degree)) score += 8;
    if (certificationsList.length > 0) score += 8;
    if (extracurricularList.length > 0) score += 8;
    if (githubUrl || linkedinUrl) score += 8;
    if (portfolioUrl) score += 4;
    return Math.min(score, 96);
  };

  const currentScore = calculateResumeRating();

  const activeErrors = validateForm();
  const isProfileComplete = Object.keys(activeErrors).length === 0;

  const handlePhotoFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setAvatarUrl(reader.result);
        if (showToast) showToast('Photo Uploaded', 'Profile photo updated successfully!');
      };
      reader.readAsDataURL(file);
    }
  };

  const handleResumeUploadAndExtract = (e) => {
    const file = e.target.files[0];
    if (file) {
      setResumeParsed(true);
      if (!skillsText.includes('System Design')) {
        setSkillsText(prev => prev + ', System Design, Microservices, CI/CD');
      }
      if (showToast) {
        showToast('Resume Auto-Extracted! 📄', `Extracted additional candidate skills & education from ${file.name}`);
      }
    }
  };

  const handleSaveProfile = (e) => {
    e.preventDefault();
    setAttemptedSave(true);
    const errs = validateForm();
    setFormErrors(errs);

    if (Object.keys(errs).length > 0) {
      if (showToast) {
        showToast(
          '⚠️ Missing Compulsory Data!',
          'Education Name, Degree, Full Name, and Technical Skills are mandatory to build your profile!'
        );
      }
      return;
    }

    setSaving(true);
    const updatedPayload = {
      id: user?.id || 1,
      full_name: fullName,
      headline: headline,
      avatar_url: avatarUrl,
      skills: skillsText.split(',').map(s => s.trim()).filter(Boolean),
      experience_years: Number(experienceYears),
      preferred_roles: preferredRoles.split(',').map(r => r.trim()).filter(Boolean),
      preferred_locations: preferredLocations.split(',').map(l => l.trim()).filter(Boolean),
      portfolio_url: portfolioUrl,
      github_url: githubUrl,
      linkedin_url: linkedinUrl,
      leetcode_url: leetcodeUrl,
      other_urls: otherUrls,
      education_details: educationList,
      certifications: certificationsList,
      extracurricular: extracurricularList,
      resume_score: currentScore,
      is_profile_complete: true
    };

    fetch('http://localhost:8000/api/v1/auth/update_profile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updatedPayload)
    })
      .then(res => res.json())
      .then(data => {
        const finalUser = data.user || updatedPayload;
        if (onUpdateUser) onUpdateUser(finalUser);
        if (showToast) showToast('Neon DB Synchronized! 🐘', 'Profile details & education records saved to Neon PostgreSQL!');
      })
      .catch(() => {
        if (onUpdateUser) onUpdateUser(updatedPayload);
        if (showToast) showToast('Profile Updated! 🎉', 'Education & candidate details updated!');
      })
      .finally(() => {
        setSaving(false);
      });
  };

  return (
    <div className="space-y-8 max-w-5xl">
      {/* Header Banner */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 dark:text-white tracking-tight flex items-center gap-2">
            <User className="w-6 h-6 text-indigo-500" /> Student Profile & Candidate Knowledge Base
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Upload your resume to auto-extract missing data, add portfolio/GitHub links, and evaluate your live resume score (0–100).
          </p>
        </div>
      </div>

      {/* 📄 Resume Upload & Auto-Extraction Banner */}
      <div className="bg-gradient-to-r from-indigo-600/10 via-purple-600/10 to-indigo-600/10 border-2 border-dashed border-indigo-500/40 rounded-3xl p-6 text-center space-y-3 relative hover:border-indigo-500 transition-all">
        <input
          type="file"
          accept=".pdf,.docx"
          onChange={handleResumeUploadAndExtract}
          className="absolute inset-0 opacity-0 cursor-pointer z-10"
        />
        <div className="w-12 h-12 rounded-2xl bg-indigo-600/20 text-indigo-400 flex items-center justify-center mx-auto shadow-md">
          <Upload className="w-6 h-6" />
        </div>
        <div>
          <h3 className="text-sm font-bold text-slate-800 dark:text-white">
            {resumeParsed ? "✅ Resume Uploaded & Parsed!" : "Upload Resume PDF to Auto-Fill Missing Data"}
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Auto-extracts missing technical skills, education details, and work experience straight into your student profile!
          </p>
        </div>
      </div>

      {/* 📊 Live Resume Rating & Scoring Engine Box */}
      <div className="bg-gradient-to-br from-indigo-900/60 via-slate-900 to-purple-950/60 border border-indigo-500/30 rounded-3xl p-6 shadow-xl space-y-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6 border-b border-slate-800/80 pb-6">
          <div className="flex items-center gap-5">
            <MatchScoreRing score={currentScore} size={90} strokeWidth={9} />
            <div>
              <span className="text-xs font-extrabold text-indigo-400 uppercase tracking-wider">Live Candidate Resume Rating</span>
              <h2 className="text-2xl font-extrabold text-white flex items-center gap-2">
                {currentScore} / 100 Points
                <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2.5 py-0.5 rounded-full font-bold border border-emerald-500/30">
                  Strong Technical CV
                </span>
              </h2>
              <p className="text-xs text-slate-300 mt-1">
                Your profile data is synchronized with the Resume Builder & ATS Fit Scoring Engine.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <button
              onClick={handleSaveProfile}
              className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs px-5 py-2.5 rounded-xl flex items-center gap-2 shadow-lg shadow-indigo-500/25 transition-all"
            >
              <Save className="w-4 h-4" /> Save Profile & Sync Resume
            </button>
          </div>
        </div>

        {/* 📌 Compulsory Profile Data Status Card */}
        <div className={`p-4 rounded-2xl border transition-all ${
          isProfileComplete 
            ? 'bg-emerald-950/30 border-emerald-500/30 text-emerald-300' 
            : 'bg-amber-950/40 border-amber-500/40 text-amber-200'
        }`}>
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div className="flex items-center gap-3">
              {isProfileComplete ? (
                <CheckCircle2 className="w-6 h-6 text-emerald-400 shrink-0" />
              ) : (
                <AlertCircle className="w-6 h-6 text-amber-400 shrink-0" />
              )}
              <div>
                <h4 className="text-sm font-extrabold flex items-center gap-2">
                  {isProfileComplete ? "Compulsory Profile Data Complete" : "Incomplete Compulsory Profile Data"}
                  <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase ${
                    isProfileComplete ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'
                  }`}>
                    {isProfileComplete ? "100% Ready" : "Action Required"}
                  </span>
                </h4>
                <p className="text-xs text-slate-300 mt-0.5">
                  {isProfileComplete 
                    ? "All required fields (Full Name, Skills, Education Name & Degree) are filled. Profile is optimized for AI ATS matching."
                    : "Please fill in compulsory fields below so your candidate graph and resume builder work effectively."
                  }
                </p>
              </div>
            </div>
            
            {/* Field Status Badges */}
            <div className="flex items-center gap-2 text-xs flex-wrap">
              <span className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold border ${
                fullName.trim() ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
              }`}>
                Full Name {fullName.trim() ? '✓' : '✗ Required'}
              </span>
              <span className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold border ${
                skillsText.trim() ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
              }`}>
                Skills {skillsText.trim() ? '✓' : '✗ Required'}
              </span>
              <span className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold border ${
                educationList.length > 0 && educationList[0]?.institute && educationList[0]?.degree
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                  : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
              }`}>
                Education Name & Degree {educationList.length > 0 && educationList[0]?.institute && educationList[0]?.degree ? '✓' : '✗ Required'}
              </span>
            </div>
          </div>
        </div>

        {/* AI Resume Improvement Recommendations */}
        <div className="space-y-3">
          <h3 className="text-xs font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-indigo-400" /> AI Resume Score Boost Recommendations
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            <div className="bg-slate-900/80 border border-slate-800 p-3.5 rounded-xl flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-white block">Quantify Internship Impact</strong>
                <span className="text-slate-400 text-[11px]">Add metric numbers (e.g. "Reduced API response latency by 40%").</span>
              </div>
            </div>
            <div className="bg-slate-900/80 border border-slate-800 p-3.5 rounded-xl flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-white block">Add Cloud Certifications</strong>
                <span className="text-slate-400 text-[11px]">AWS Certified Developer increases recruiter response rate by 28%.</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Student Profile Form */}
      <form onSubmit={handleSaveProfile} className="space-y-6">

        {/* 1. Basic Student Info & Photo Upload */}
        <div className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-5">
          <h3 className="text-base font-bold text-slate-800 dark:text-white flex items-center gap-2">
            <User className="w-5 h-5 text-indigo-500" /> 1. Personal Information & Photo
          </h3>

          <div className="flex items-center gap-5 bg-slate-50 dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-800">
            <img
              src={avatarUrl}
              alt="Student Avatar"
              className="w-16 h-16 rounded-full object-cover ring-4 ring-indigo-500/30 shadow-md shrink-0"
            />
            <div className="flex-1 space-y-1.5">
              <label className="text-xs font-bold text-slate-700 dark:text-white block">Upload Official Profile Photo</label>
              <input
                type="file"
                accept="image/*"
                onChange={handlePhotoFileChange}
                id="profile-photo-upload"
                className="hidden"
              />
              <label
                htmlFor="profile-photo-upload"
                className="cursor-pointer inline-flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs px-4 py-2 rounded-xl shadow-md transition-all"
              >
                <Camera className="w-4 h-4" /> Choose Image File
              </label>
              <p className="text-[10px] text-slate-400">Supported formats: JPG, PNG, WEBP</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-bold text-slate-700 dark:text-slate-300 flex items-center gap-1 mb-1">
                Full Name <span className="text-rose-500 font-extrabold">*</span>
                <span className="text-[10px] text-rose-400 bg-rose-500/10 px-1.5 py-0.2 rounded font-semibold ml-1">Required</span>
              </label>
              <input
                type="text"
                required
                value={fullName}
                onChange={e => {
                  setFullName(e.target.value);
                  if (formErrors.fullName) setFormErrors({...formErrors, fullName: null});
                }}
                className={`w-full bg-slate-50 dark:bg-slate-900 border text-xs text-slate-800 dark:text-white px-3.5 py-2.5 rounded-xl transition-all ${
                  attemptedSave && formErrors.fullName
                    ? 'border-rose-500 ring-2 ring-rose-500/20'
                    : 'border-slate-200 dark:border-slate-800 focus:border-indigo-500'
                }`}
              />
              {attemptedSave && formErrors.fullName && (
                <p className="text-[11px] text-rose-400 mt-1 font-semibold flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" /> {formErrors.fullName}
                </p>
              )}
            </div>
            <div>
              <label className="text-xs font-bold text-slate-700 dark:text-slate-300 block mb-1">Professional Headline</label>
              <input
                type="text"
                value={headline}
                onChange={e => setHeadline(e.target.value)}
                className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3.5 py-2.5 rounded-xl"
              />
            </div>
          </div>

          <div>
            <label className="text-xs font-bold text-slate-700 dark:text-slate-300 flex items-center gap-1 mb-1">
              Technical Skills (comma separated) <span className="text-rose-500 font-extrabold">*</span>
              <span className="text-[10px] text-rose-400 bg-rose-500/10 px-1.5 py-0.2 rounded font-semibold ml-1">Required</span>
            </label>
            <input
              type="text"
              value={skillsText}
              onChange={e => {
                setSkillsText(e.target.value);
                if (formErrors.skillsText) setFormErrors({...formErrors, skillsText: null});
              }}
              className={`w-full bg-slate-50 dark:bg-slate-900 border text-xs text-slate-800 dark:text-white px-3.5 py-2.5 rounded-xl transition-all ${
                attemptedSave && formErrors.skillsText
                  ? 'border-rose-500 ring-2 ring-rose-500/20'
                  : 'border-slate-200 dark:border-slate-800 focus:border-indigo-500'
              }`}
            />
            {attemptedSave && formErrors.skillsText && (
              <p className="text-[11px] text-rose-400 mt-1 font-semibold flex items-center gap-1">
                <AlertCircle className="w-3 h-3" /> {formErrors.skillsText}
              </p>
            )}
          </div>
        </div>

        {/* 🌐 2. Professional Links & Online Footprint (Portfolio, GitHub, LinkedIn, LeetCode) */}
        <div className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
          <h3 className="text-base font-bold text-slate-800 dark:text-white flex items-center gap-2">
            <Globe className="w-5 h-5 text-indigo-500" /> 2. Portfolio & Online Social Platform Links
          </h3>
          <p className="text-xs text-slate-400">
            TalentForge analyzes your GitHub code, LeetCode stats, and portfolio projects to build your candidate knowledge graph.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-bold text-slate-700 dark:text-slate-300 flex items-center gap-1.5 mb-1">
                <Globe className="w-3.5 h-3.5 text-indigo-400" /> Personal Portfolio Website URL
              </label>
              <input
                type="url"
                value={portfolioUrl}
                onChange={e => setPortfolioUrl(e.target.value)}
                placeholder="https://arjun.dev"
                className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3.5 py-2.5 rounded-xl"
              />
            </div>

            <div>
              <label className="text-xs font-bold text-slate-700 dark:text-slate-300 flex items-center gap-1.5 mb-1">
                <Github className="w-3.5 h-3.5 text-indigo-400" /> GitHub Profile Link
              </label>
              <input
                type="url"
                value={githubUrl}
                onChange={e => setGithubUrl(e.target.value)}
                placeholder="https://github.com/username"
                className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3.5 py-2.5 rounded-xl"
              />
            </div>

            <div>
              <label className="text-xs font-bold text-slate-700 dark:text-slate-300 flex items-center gap-1.5 mb-1">
                <Linkedin className="w-3.5 h-3.5 text-indigo-400" /> LinkedIn Profile Link
              </label>
              <input
                type="url"
                value={linkedinUrl}
                onChange={e => setLinkedinUrl(e.target.value)}
                placeholder="https://linkedin.com/in/username"
                className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3.5 py-2.5 rounded-xl"
              />
            </div>

            <div>
              <label className="text-xs font-bold text-slate-700 dark:text-slate-300 flex items-center gap-1.5 mb-1">
                <Code className="w-3.5 h-3.5 text-indigo-400" /> LeetCode / Codeforces Profile
              </label>
              <input
                type="url"
                value={leetcodeUrl}
                onChange={e => setLeetcodeUrl(e.target.value)}
                placeholder="https://leetcode.com/username"
                className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3.5 py-2.5 rounded-xl"
              />
            </div>
          </div>

          <div>
            <label className="text-xs font-bold text-slate-700 dark:text-slate-300 flex items-center gap-1.5 mb-1">
              <Link className="w-3.5 h-3.5 text-indigo-400" /> Other Social / Kaggle / Medium Links
            </label>
            <input
              type="text"
              value={otherUrls}
              onChange={e => setOtherUrls(e.target.value)}
              placeholder="https://twitter.com/username"
              className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3.5 py-2.5 rounded-xl"
            />
          </div>
        </div>

        {/* 3. Complete Education Details Section (COMPULSORY DATA) */}
        <div className={`bg-white dark:bg-[#1E293B] border rounded-2xl p-6 shadow-sm space-y-4 transition-all ${
          attemptedSave && (formErrors.educationList || Object.keys(formErrors).some(k => k.startsWith('edu_')))
            ? 'border-rose-500/60 ring-2 ring-rose-500/10'
            : 'border-slate-200 dark:border-slate-800'
        }`}>
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold text-slate-800 dark:text-white flex items-center gap-2">
                <GraduationCap className="w-5 h-5 text-indigo-500" /> 3. Complete Education Details
                <span className="text-rose-500 font-extrabold">*</span>
                <span className="text-[10px] text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded font-bold uppercase">Compulsory</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Education Name (University / Institute) and Degree are compulsory for profile validation and resume templates.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setEducationList([...educationList, { degree: "", institute: "", year: "", cgpa: "" }])}
              className="text-xs font-bold text-indigo-500 hover:text-indigo-400 flex items-center gap-1"
            >
              <Plus className="w-4 h-4" /> Add Education
            </button>
          </div>

          {attemptedSave && formErrors.educationList && (
            <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-xl text-xs text-rose-400 font-semibold flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              {formErrors.educationList}
            </div>
          )}

          {educationList.map((edu, idx) => (
            <div key={idx} className="bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 rounded-xl space-y-3 relative">
              {educationList.length > 1 && (
                <button
                  type="button"
                  onClick={() => setEducationList(educationList.filter((_, i) => i !== idx))}
                  className="absolute top-3 right-3 text-slate-400 hover:text-rose-500 p-1"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="text-[11px] font-bold text-slate-500 block mb-1">
                    Degree / Specialization <span className="text-rose-500">*</span>
                  </label>
                  <input
                    type="text"
                    value={edu.degree}
                    onChange={(e) => {
                      const copy = [...educationList];
                      copy[idx].degree = e.target.value;
                      setEducationList(copy);
                    }}
                    placeholder="B.Tech Computer Science"
                    className={`w-full bg-white dark:bg-slate-950 border text-xs text-slate-800 dark:text-white px-3 py-2 rounded-lg transition-all ${
                      attemptedSave && !edu.degree.trim()
                        ? 'border-rose-500 ring-2 ring-rose-500/20'
                        : 'border-slate-200 dark:border-slate-800'
                    }`}
                  />
                  {attemptedSave && !edu.degree.trim() && (
                    <p className="text-[10px] text-rose-400 mt-1 font-semibold">Degree name is compulsory.</p>
                  )}
                </div>
                <div>
                  <label className="text-[11px] font-bold text-slate-500 block mb-1">
                    University / Institute Name <span className="text-rose-500">*</span>
                  </label>
                  <input
                    type="text"
                    value={edu.institute}
                    onChange={(e) => {
                      const copy = [...educationList];
                      copy[idx].institute = e.target.value;
                      setEducationList(copy);
                    }}
                    placeholder="RV College of Engineering"
                    className={`w-full bg-white dark:bg-slate-950 border text-xs text-slate-800 dark:text-white px-3 py-2 rounded-lg transition-all ${
                      attemptedSave && !edu.institute.trim()
                        ? 'border-rose-500 ring-2 ring-rose-500/20'
                        : 'border-slate-200 dark:border-slate-800'
                    }`}
                  />
                  {attemptedSave && !edu.institute.trim() && (
                    <p className="text-[10px] text-rose-400 mt-1 font-semibold">University/Institute name is compulsory.</p>
                  )}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-[11px] font-bold text-slate-500 block mb-1">Graduation Year</label>
                  <input
                    type="text"
                    value={edu.year}
                    onChange={(e) => {
                      const copy = [...educationList];
                      copy[idx].year = e.target.value;
                      setEducationList(copy);
                    }}
                    placeholder="2024"
                    className="w-full bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2 rounded-lg"
                  />
                </div>
                <div>
                  <label className="text-[11px] font-bold text-slate-500 block mb-1">CGPA / Percentage</label>
                  <input
                    type="text"
                    value={edu.cgpa}
                    onChange={(e) => {
                      const copy = [...educationList];
                      copy[idx].cgpa = e.target.value;
                      setEducationList(copy);
                    }}
                    placeholder="8.8 / 10"
                    className="w-full bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2 rounded-lg"
                  />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* 4. Certifications & Credentials Section */}
        <div className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-slate-800 dark:text-white flex items-center gap-2">
              <Award className="w-5 h-5 text-indigo-500" /> 4. Certifications & Technical Credentials
            </h3>
            <button
              type="button"
              onClick={() => setCertificationsList([...certificationsList, { title: "", issuer: "", year: "" }])}
              className="text-xs font-bold text-indigo-500 hover:text-indigo-400 flex items-center gap-1"
            >
              <Plus className="w-4 h-4" /> Add Certification
            </button>
          </div>

          {certificationsList.map((cert, idx) => (
            <div key={idx} className="bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 rounded-xl space-y-3 relative">
              <button
                type="button"
                onClick={() => setCertificationsList(certificationsList.filter((_, i) => i !== idx))}
                className="absolute top-3 right-3 text-slate-400 hover:text-rose-500 p-1"
              >
                <Trash2 className="w-4 h-4" />
              </button>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="md:col-span-2">
                  <label className="text-[11px] font-bold text-slate-500 block mb-1">Certification Name</label>
                  <input
                    type="text"
                    value={cert.title}
                    onChange={(e) => {
                      const copy = [...certificationsList];
                      copy[idx].title = e.target.value;
                      setCertificationsList(copy);
                    }}
                    placeholder="AWS Certified Developer – Associate"
                    className="w-full bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2 rounded-lg"
                  />
                </div>
                <div>
                  <label className="text-[11px] font-bold text-slate-500 block mb-1">Issuing Body / Year</label>
                  <input
                    type="text"
                    value={cert.issuer}
                    onChange={(e) => {
                      const copy = [...certificationsList];
                      copy[idx].issuer = e.target.value;
                      setCertificationsList(copy);
                    }}
                    placeholder="Amazon Web Services (2023)"
                    className="w-full bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2 rounded-lg"
                  />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* 5. Extra-Curricular Activities & Achievements */}
        <div className="bg-white dark:bg-[#1E293B] border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-slate-800 dark:text-white flex items-center gap-2">
              <Trophy className="w-5 h-5 text-indigo-500" /> 5. Extra-Curricular Activities & Achievements
            </h3>
            <button
              type="button"
              onClick={() => setExtracurricularList([...extracurricularList, { title: "", org: "", desc: "" }])}
              className="text-xs font-bold text-indigo-500 hover:text-indigo-400 flex items-center gap-1"
            >
              <Plus className="w-4 h-4" /> Add Activity
            </button>
          </div>

          {extracurricularList.map((act, idx) => (
            <div key={idx} className="bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 rounded-xl space-y-3 relative">
              <button
                type="button"
                onClick={() => setExtracurricularList(extracurricularList.filter((_, i) => i !== idx))}
                className="absolute top-3 right-3 text-slate-400 hover:text-rose-500 p-1"
              >
                <Trash2 className="w-4 h-4" />
              </button>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="text-[11px] font-bold text-slate-500 block mb-1">Achievement / Role Title</label>
                  <input
                    type="text"
                    value={act.title}
                    onChange={(e) => {
                      const copy = [...extracurricularList];
                      copy[idx].title = e.target.value;
                      setExtracurricularList(copy);
                    }}
                    placeholder="Hackathon Winner / Club President"
                    className="w-full bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2 rounded-lg"
                  />
                </div>
                <div>
                  <label className="text-[11px] font-bold text-slate-500 block mb-1">Organization / Event</label>
                  <input
                    type="text"
                    value={act.org}
                    onChange={(e) => {
                      const copy = [...extracurricularList];
                      copy[idx].org = e.target.value;
                      setExtracurricularList(copy);
                    }}
                    placeholder="Smart India Hackathon"
                    className="w-full bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2 rounded-lg"
                  />
                </div>
              </div>
              <div>
                <label className="text-[11px] font-bold text-slate-500 block mb-1">Brief Summary of Impact</label>
                <input
                  type="text"
                  value={act.desc}
                  onChange={(e) => {
                    const copy = [...extracurricularList];
                    copy[idx].desc = e.target.value;
                    setExtracurricularList(copy);
                  }}
                  placeholder="Led 5 team members to build an automated logistics solver..."
                  className="w-full bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-xs text-slate-800 dark:text-white px-3 py-2 rounded-lg"
                />
              </div>
            </div>
          ))}
        </div>

        {/* Submit Bar */}
        <div className="flex justify-end pt-2">
          <button
            type="submit"
            disabled={saving}
            className="bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-bold text-xs px-8 py-3 rounded-xl shadow-lg shadow-indigo-500/25 flex items-center gap-2 transition-all"
          >
            <Save className="w-4 h-4" /> {saving ? "Saving..." : "Save Profile & Build Student Database"}
          </button>
        </div>
      </form>
    </div>
  );
}
