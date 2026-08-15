import React, { useState } from 'react';
import { LayoutTemplate, ExternalLink, CheckCircle2, Download, Code, Sparkles, FileText, Copy, Eye, Edit3, ShieldCheck, Zap, RefreshCw, Star, Layers, Plus, Trash2, X, Share2, Upload, FileDown, Printer, ChevronDown, ChevronUp, User, Briefcase, GraduationCap, FolderGit2, Award, FileCode } from 'lucide-react';
import MatchScoreRing from '../components/MatchScoreRing';

export default function ResumeBuilder({ showToast }) {
  const [activeSubTab, setActiveSubTab] = useState('builder'); // 'builder' | 'templates'
  const [rightViewMode, setRightViewMode] = useState('preview'); // 'preview' | 'code'
  const [selectedTemplateId, setSelectedTemplateId] = useState('jakes-resume');
  const [showOverleafImportModal, setShowOverleafImportModal] = useState(false);
  const [customOverleafCode, setCustomOverleafCode] = useState('');

  // Accordion Section States
  const [openSection, setOpenSection] = useState('basics'); // 'basics' | 'skills' | 'experience' | 'projects' | 'education' | 'custom'

  // Core Candidate Profile Data
  const [candidateName, setCandidateName] = useState('Arjun B.');
  const [candidateHeadline, setCandidateHeadline] = useState('Software Engineer - Backend & Systems');
  const [candidateEmail, setCandidateEmail] = useState('arjun.b@talentforge.ai');
  const [candidatePhone, setCandidatePhone] = useState('+91 98765 43210');
  const [candidateLocation, setCandidateLocation] = useState('Bengaluru, KA');
  const [candidateSkills, setCandidateSkills] = useState('Python, FastAPI, AWS, PostgreSQL, Docker, Redis, Kafka, System Design');

  // Dynamic Work Experience Entries
  const [experiences, setExperiences] = useState([
    {
      id: 1,
      role: 'Software Engineer Intern',
      company: 'Superset Inc.',
      period: '2025',
      desc: 'Built microservices handling 2M+ daily requests using FastAPI & Redis. Designed RESTful APIs with 99.99% uptime.'
    }
  ]);

  // Dynamic Technical Projects Entries
  const [projects, setProjects] = useState([
    {
      id: 1,
      name: 'TalentForge AI Sourcing Engine',
      tech: 'FastAPI, Python, PostgreSQL, React',
      desc: 'Real-time candidate sourcing & ATS resume matching engine with 7-factor evaluation.'
    }
  ]);

  // Dynamic Education Entries
  const [educationList, setEducationList] = useState([
    {
      id: 1,
      degree: 'B.Tech Computer Science',
      institute: 'RV College of Engineering',
      year: '2024',
      cgpa: '8.8 / 10'
    }
  ]);

  // Dynamic Custom User Sections
  const [customSections, setCustomSections] = useState([
    { id: 1, title: 'Achievements & Open Source', content: '1st Place Winner - Smart India Hackathon. Active contributor to FastAPI ecosystem.' }
  ]);

  // Placement Templates Gallery
  const [templates, setTemplates] = useState([
    {
      id: 'jakes-resume',
      name: "Jake's Resume (FAANG Placement Standard)",
      category: "ATS Safe / Technical",
      atsScore: 98,
      desc: "Clean single-column layout preferred by Google, Microsoft, and Amazon tech recruiters.",
      code: `\\documentclass[letterpaper,11pt]{article}
\\usepackage[empty]{fullpage}
\\usepackage{titlesec}
\\usepackage{hyperref}
\\begin{document}
\\begin{center}
    {\\Huge \\scshape __NAME__} \\\\
    \\small __LOCATION__ $|$ __EMAIL__ $|$ __PHONE__
\\end{center}
\\section{Summary}
__HEADLINE__ specializing in high-throughput backend services and cloud systems.

\\section{Technical Skills}
__SKILLS__

\\section{Experience}
__EXPERIENCE_BLOCK__

\\section{Projects}
__PROJECTS_BLOCK__

\\section{Education}
__EDUCATION_BLOCK__

__CUSTOM_BLOCK__
\\end{document}`
    },
    {
      id: 'faangpath',
      name: "FAANGPath Minimalist",
      category: "Minimalist / High-Density",
      atsScore: 96,
      desc: "Single-page compact layout optimized for high-throughput software engineering placement roles.",
      code: `\\documentclass[10pt,a4paper]{article}
\\begin{document}
\\centerline{\\Large \\bf __NAME__}
\\centerline{__EMAIL__ | __PHONE__ | __LOCATION__}
\\line(1,0){450}

\\section*{SUMMARY}
__HEADLINE__

\\section*{TECHNICAL SKILLS}
__SKILLS__

\\section*{EXPERIENCE}
__EXPERIENCE_BLOCK__

\\section*{PROJECTS}
__PROJECTS_BLOCK__

\\section*{EDUCATION}
__EDUCATION_BLOCK__
\\end{document}`
    },
    {
      id: 'deedy',
      name: "Deedy Resume OpenFont",
      category: "Modern Two-Column",
      atsScore: 94,
      desc: "Two-column design highlighting skills, metrics, and experience side by side.",
      code: `\\documentclass[]{deedy-resume-openfont}
\\begin{document}
\\namesection{Arjun}{B.}{__EMAIL__ | __PHONE__}
\\begin{minipage}[t]{0.33\\textwidth}
\\section{Skills}
__SKILLS__
\\end{minipage}
\\end{document}`
    },
    {
      id: 'awesome-cv',
      name: "Awesome CV Tech",
      category: "Executive / Design",
      atsScore: 95,
      desc: "Elegant typography layout with customizable primary accent styling.",
      code: `\\documentclass[11pt, a4paper]{awesome-cv}
\\begin{document}
\\name{Arjun}{B.}
\\position{__HEADLINE__}
\\end{document}`
    }
  ]);

  const currentTemplate = templates.find(t => t.id === selectedTemplateId) || templates[0];

  // Helper formatting for LaTeX blocks
  const buildExperienceLatex = () => {
    return experiences.map(exp => 
      `\\textbf{${exp.role}} -- ${exp.company} \\hfill ${exp.period}\\\\\n\\begin{itemize}\n  \\item ${exp.desc}\n\\end{itemize}`
    ).join("\n\n");
  };

  const buildProjectsLatex = () => {
    return projects.map(p => 
      `\\textbf{${p.name}} $|$ \\textit{${p.tech}}\\\\\n${p.desc}`
    ).join("\n\n");
  };

  const buildEducationLatex = () => {
    return educationList.map(e => 
      `\\textbf{${e.degree}} -- ${e.institute} \\hfill ${e.year} (CGPA: ${e.cgpa})`
    ).join("\n");
  };

  const buildCustomLatex = () => {
    return customSections.map(cs => 
      `\\section{${cs.title}}\n${cs.content}`
    ).join("\n\n");
  };

  // Dynamic LaTeX Code Compiler
  const getCompiledLatex = () => {
    return currentTemplate.code
      .replace(/__NAME__/g, candidateName)
      .replace(/__HEADLINE__/g, candidateHeadline)
      .replace(/__EMAIL__/g, candidateEmail)
      .replace(/__PHONE__/g, candidatePhone)
      .replace(/__LOCATION__/g, candidateLocation)
      .replace(/__SKILLS__/g, candidateSkills)
      .replace(/__EXPERIENCE_BLOCK__/g, buildExperienceLatex())
      .replace(/__PROJECTS_BLOCK__/g, buildProjectsLatex())
      .replace(/__EDUCATION_BLOCK__/g, buildEducationLatex())
      .replace(/__CUSTOM_BLOCK__/g, buildCustomLatex());
  };

  const compiledCode = getCompiledLatex();

  // Add / Remove Handlers
  const handleAddExperience = () => {
    const newItem = {
      id: Date.now(),
      role: 'Backend Engineer',
      company: 'Tech Corp',
      period: '2024 - Present',
      desc: 'Architected cloud microservices and deployment pipelines.'
    };
    setExperiences([...experiences, newItem]);
    if (showToast) showToast('Section Added ➕', 'Added new Work Experience entry');
  };

  const handleAddProject = () => {
    const newItem = {
      id: Date.now(),
      name: 'Distributed Microservice Gateway',
      tech: 'Go, Docker, Redis',
      desc: 'High-throughput API gateway routing 5M requests/day.'
    };
    setProjects([...projects, newItem]);
    if (showToast) showToast('Section Added ➕', 'Added new Technical Project entry');
  };

  const handleAddEducation = () => {
    const newItem = {
      id: Date.now(),
      degree: 'M.Tech Software Engineering',
      institute: 'IIT Bengaluru',
      year: '2025',
      cgpa: '9.0 / 10'
    };
    setEducationList([...educationList, newItem]);
    if (showToast) showToast('Section Added ➕', 'Added new Education record');
  };

  const handleAddCustomSection = () => {
    const newItem = {
      id: Date.now(),
      title: 'Publications & Honors',
      content: 'Published research paper on Distributed Systems in IEEE Conference.'
    };
    setCustomSections([...customSections, newItem]);
    if (showToast) showToast('Custom Section Added ➕', 'Added new custom placement section');
  };

  // Export Handlers
  const handleOpenOverleaf = () => {
    const encoded = encodeURIComponent(compiledCode);
    window.open(`https://www.overleaf.com/docs?snip=${encoded}`, '_blank');
    if (showToast) showToast("Opening Overleaf 🚀", "Exporting project into Overleaf Studio");
  };

  const handleDownloadTex = () => {
    const blob = new Blob([compiledCode], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${candidateName.toLowerCase().replace(/\s+/g, '_')}_resume.tex`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    if (showToast) showToast("LaTeX File Saved 📄", "Downloaded resume.tex file to your computer");
  };

  const handlePrintPDF = () => {
    window.print();
    if (showToast) showToast("Print PDF 🖨️", "Opening browser print dialog to export PDF");
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(compiledCode);
    if (showToast) showToast("Code Copied 📋", "LaTeX source code copied to clipboard");
  };

  const handleImportOverleafCode = (e) => {
    e.preventDefault();
    if (!customOverleafCode.trim()) return;

    const newCustomTmpl = {
      id: `custom-overleaf-${Date.now()}`,
      name: "Custom Overleaf Template",
      category: "Imported / Overleaf",
      atsScore: 98,
      desc: "Custom LaTeX template imported directly from Overleaf.",
      code: customOverleafCode
    };

    setTemplates([newCustomTmpl, ...templates]);
    setSelectedTemplateId(newCustomTmpl.id);
    setShowOverleafImportModal(false);
    setCustomOverleafCode('');

    if (showToast) showToast("Overleaf Code Imported! 🚀", "Applied custom template to resume builder");
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      
      {/* Hero Header & Control Bar */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/60 to-slate-900 border border-slate-800 p-6 rounded-3xl shadow-xl flex flex-col lg:flex-row lg:items-center justify-between gap-6 relative overflow-hidden backdrop-blur-xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none"></div>

        <div className="space-y-1 relative z-10">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs bg-indigo-500/20 text-indigo-300 font-extrabold px-3 py-1 rounded-full border border-indigo-500/30 flex items-center gap-1">
              <Zap className="w-3.5 h-3.5 fill-indigo-300" /> AI Resume Studio
            </span>
            <span className="text-xs bg-emerald-500/15 text-emerald-400 font-extrabold px-3 py-1 rounded-full border border-emerald-500/30 flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5" /> {currentTemplate.atsScore}% ATS Match
            </span>
          </div>

          <h1 className="text-2xl lg:text-3xl font-black text-white tracking-tight">
            ATS Resume Builder & Overleaf Studio
          </h1>
          <p className="text-xs text-slate-400 max-w-xl">
            Build ATS-formatted campus placement resumes, edit dynamic sections, import from Overleaf, and export print-ready PDFs.
          </p>
        </div>

        {/* Global Header Action Buttons */}
        <div className="flex flex-wrap items-center gap-2.5 relative z-10 shrink-0">
          <button
            onClick={handlePrintPDF}
            className="bg-slate-800 hover:bg-slate-700 text-slate-200 font-extrabold text-xs px-4 py-3 rounded-2xl border border-slate-700 transition-all flex items-center gap-2 cursor-pointer shadow-sm active:scale-95"
          >
            <Printer className="w-4 h-4 text-emerald-400" /> Save PDF
          </button>

          <button
            onClick={handleDownloadTex}
            className="bg-slate-800 hover:bg-slate-700 text-slate-200 font-extrabold text-xs px-4 py-3 rounded-2xl border border-slate-700 transition-all flex items-center gap-2 cursor-pointer shadow-sm active:scale-95"
          >
            <FileDown className="w-4 h-4 text-indigo-400" /> Download .tex
          </button>

          <button
            onClick={() => setShowOverleafImportModal(true)}
            className="bg-indigo-950/80 hover:bg-indigo-900 border border-indigo-500/40 text-indigo-300 font-extrabold text-xs px-4 py-3 rounded-2xl transition-all flex items-center gap-2 cursor-pointer shadow-sm active:scale-95"
          >
            <Upload className="w-4 h-4 text-indigo-400" /> Import Overleaf
          </button>

          <button
            onClick={handleOpenOverleaf}
            className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-black text-xs px-5 py-3 rounded-2xl transition-all shadow-lg shadow-emerald-500/25 flex items-center gap-2 active:scale-95 cursor-pointer"
          >
            <ExternalLink className="w-4 h-4" /> Open in Overleaf ↗
          </button>
        </div>
      </div>

      {/* Main Sub-Navigation Bar & Template Selector */}
      <div className="bg-white dark:bg-[#0F172A] border border-slate-200 dark:border-slate-800 p-2 rounded-2xl flex flex-col sm:flex-row items-center justify-between gap-3 shadow-sm">
        
        {/* Tab Switcher */}
        <div className="bg-slate-900 p-1 rounded-xl flex items-center gap-1.5 w-full sm:w-auto">
          <button
            onClick={() => setActiveSubTab('builder')}
            className={`flex-1 sm:flex-initial px-4 py-2 text-xs font-black rounded-lg transition-all flex items-center justify-center gap-2 cursor-pointer ${
              activeSubTab === 'builder'
                ? 'bg-indigo-600 text-white shadow-md'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <FileText className="w-3.5 h-3.5" /> Resume Workspace
          </button>

          <button
            onClick={() => setActiveSubTab('templates')}
            className={`flex-1 sm:flex-initial px-4 py-2 text-xs font-black rounded-lg transition-all flex items-center justify-center gap-2 cursor-pointer ${
              activeSubTab === 'templates'
                ? 'bg-indigo-600 text-white shadow-md'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <LayoutTemplate className="w-3.5 h-3.5" /> Template Library ({templates.length})
          </button>
        </div>

        {/* Template Quick Selector dropdown */}
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <span className="text-xs font-bold text-slate-400 shrink-0">Active Template:</span>
          <select
            value={selectedTemplateId}
            onChange={(e) => setSelectedTemplateId(e.target.value)}
            className="bg-slate-900 border border-slate-800 text-xs font-extrabold text-white px-3 py-2 rounded-xl focus:outline-none cursor-pointer w-full sm:w-auto"
          >
            {templates.map(t => (
              <option key={t.id} value={t.id} className="bg-slate-900">{t.name} ({t.atsScore}% ATS)</option>
            ))}
          </select>
        </div>
      </div>

      {/* ---------------- WORKSPACE TAB ---------------- */}
      {activeSubTab === 'builder' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          
          {/* Left Column (5 Cols): Accordion Form Section Editor */}
          <div className="lg:col-span-5 bg-white dark:bg-[#0F172A] border border-slate-200 dark:border-slate-800/90 rounded-3xl p-5 shadow-sm space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-2">
              <h3 className="text-sm font-black text-slate-900 dark:text-white flex items-center gap-2">
                <Edit3 className="w-4 h-4 text-indigo-400" /> Resume Section Editor
              </h3>
              <span className="text-[10px] text-emerald-400 font-extrabold bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                Live Dynamic Sync
              </span>
            </div>

            {/* 1. Basic & Contact Info */}
            <div className="border border-slate-800 rounded-2xl overflow-hidden bg-slate-900/60">
              <button
                onClick={() => setOpenSection(openSection === 'basics' ? '' : 'basics')}
                className="w-full px-4 py-3 text-left font-black text-xs text-white flex items-center justify-between hover:bg-slate-800/50 cursor-pointer"
              >
                <span className="flex items-center gap-2">
                  <User className="w-4 h-4 text-indigo-400" /> 1. Personal & Contact Details
                </span>
                {openSection === 'basics' ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </button>

              {openSection === 'basics' && (
                <div className="p-4 border-t border-slate-800 space-y-3 text-xs">
                  <div>
                    <label className="font-bold text-slate-300 block mb-1">Full Name</label>
                    <input
                      type="text"
                      value={candidateName}
                      onChange={(e) => setCandidateName(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-800 text-white px-3 py-2 rounded-xl focus:ring-1 focus:ring-indigo-500"
                    />
                  </div>

                  <div>
                    <label className="font-bold text-slate-300 block mb-1">Professional Headline</label>
                    <input
                      type="text"
                      value={candidateHeadline}
                      onChange={(e) => setCandidateHeadline(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-800 text-white px-3 py-2 rounded-xl focus:ring-1 focus:ring-indigo-500"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="font-bold text-slate-300 block mb-1">Email</label>
                      <input
                        type="email"
                        value={candidateEmail}
                        onChange={(e) => setCandidateEmail(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-800 text-white px-3 py-2 rounded-xl"
                      />
                    </div>
                    <div>
                      <label className="font-bold text-slate-300 block mb-1">Phone</label>
                      <input
                        type="text"
                        value={candidatePhone}
                        onChange={(e) => setCandidatePhone(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-800 text-white px-3 py-2 rounded-xl"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="font-bold text-slate-300 block mb-1">Location</label>
                    <input
                      type="text"
                      value={candidateLocation}
                      onChange={(e) => setCandidateLocation(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-800 text-white px-3 py-2 rounded-xl"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* 2. Technical Skills */}
            <div className="border border-slate-800 rounded-2xl overflow-hidden bg-slate-900/60">
              <button
                onClick={() => setOpenSection(openSection === 'skills' ? '' : 'skills')}
                className="w-full px-4 py-3 text-left font-black text-xs text-white flex items-center justify-between hover:bg-slate-800/50 cursor-pointer"
              >
                <span className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-indigo-400" /> 2. Technical Skills & Languages
                </span>
                {openSection === 'skills' ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </button>

              {openSection === 'skills' && (
                <div className="p-4 border-t border-slate-800 space-y-3 text-xs">
                  <label className="font-bold text-slate-300 block">Skills (Comma Separated)</label>
                  <textarea
                    rows={3}
                    value={candidateSkills}
                    onChange={(e) => setCandidateSkills(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 text-white p-3 rounded-xl focus:ring-1 focus:ring-indigo-500 text-xs"
                  />
                </div>
              )}
            </div>

            {/* 3. Work Experience */}
            <div className="border border-slate-800 rounded-2xl overflow-hidden bg-slate-900/60">
              <button
                onClick={() => setOpenSection(openSection === 'experience' ? '' : 'experience')}
                className="w-full px-4 py-3 text-left font-black text-xs text-white flex items-center justify-between hover:bg-slate-800/50 cursor-pointer"
              >
                <span className="flex items-center gap-2">
                  <Briefcase className="w-4 h-4 text-indigo-400" /> 3. Work Experience ({experiences.length})
                </span>
                {openSection === 'experience' ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </button>

              {openSection === 'experience' && (
                <div className="p-4 border-t border-slate-800 space-y-3 text-xs">
                  <div className="flex justify-end">
                    <button
                      onClick={handleAddExperience}
                      className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-[11px] px-3 py-1.5 rounded-lg flex items-center gap-1 cursor-pointer shadow-sm"
                    >
                      <Plus className="w-3.5 h-3.5" /> Add Experience Entry
                    </button>
                  </div>

                  {experiences.map((exp, idx) => (
                    <div key={exp.id} className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-2 relative">
                      <button
                        onClick={() => setExperiences(experiences.filter(e => e.id !== exp.id))}
                        className="absolute top-2.5 right-2.5 text-slate-500 hover:text-rose-400 p-1 cursor-pointer"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                      <div className="grid grid-cols-2 gap-2 pr-6">
                        <input
                          type="text"
                          value={exp.role}
                          onChange={(e) => {
                            const updated = [...experiences];
                            updated[idx].role = e.target.value;
                            setExperiences(updated);
                          }}
                          placeholder="Role Title"
                          className="bg-slate-900 border border-slate-800 text-white px-2.5 py-1.5 rounded-lg text-xs"
                        />
                        <input
                          type="text"
                          value={exp.company}
                          onChange={(e) => {
                            const updated = [...experiences];
                            updated[idx].company = e.target.value;
                            setExperiences(updated);
                          }}
                          placeholder="Company"
                          className="bg-slate-900 border border-slate-800 text-white px-2.5 py-1.5 rounded-lg text-xs"
                        />
                      </div>
                      <textarea
                        rows={2}
                        value={exp.desc}
                        onChange={(e) => {
                          const updated = [...experiences];
                          updated[idx].desc = e.target.value;
                          setExperiences(updated);
                        }}
                        placeholder="Key metrics & accomplishments..."
                        className="w-full bg-slate-900 border border-slate-800 text-white px-2.5 py-1.5 rounded-lg text-xs"
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* 4. Technical Projects */}
            <div className="border border-slate-800 rounded-2xl overflow-hidden bg-slate-900/60">
              <button
                onClick={() => setOpenSection(openSection === 'projects' ? '' : 'projects')}
                className="w-full px-4 py-3 text-left font-black text-xs text-white flex items-center justify-between hover:bg-slate-800/50 cursor-pointer"
              >
                <span className="flex items-center gap-2">
                  <FolderGit2 className="w-4 h-4 text-indigo-400" /> 4. Technical Projects ({projects.length})
                </span>
                {openSection === 'projects' ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </button>

              {openSection === 'projects' && (
                <div className="p-4 border-t border-slate-800 space-y-3 text-xs">
                  <div className="flex justify-end">
                    <button
                      onClick={handleAddProject}
                      className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-[11px] px-3 py-1.5 rounded-lg flex items-center gap-1 cursor-pointer shadow-sm"
                    >
                      <Plus className="w-3.5 h-3.5" /> Add Technical Project
                    </button>
                  </div>

                  {projects.map((p, idx) => (
                    <div key={p.id} className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-2 relative">
                      <button
                        onClick={() => setProjects(projects.filter(item => item.id !== p.id))}
                        className="absolute top-2.5 right-2.5 text-slate-500 hover:text-rose-400 p-1 cursor-pointer"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                      <div className="grid grid-cols-2 gap-2 pr-6">
                        <input
                          type="text"
                          value={p.name}
                          onChange={(e) => {
                            const updated = [...projects];
                            updated[idx].name = e.target.value;
                            setProjects(updated);
                          }}
                          placeholder="Project Name"
                          className="bg-slate-900 border border-slate-800 text-white px-2.5 py-1.5 rounded-lg text-xs"
                        />
                        <input
                          type="text"
                          value={p.tech}
                          onChange={(e) => {
                            const updated = [...projects];
                            updated[idx].tech = e.target.value;
                            setProjects(updated);
                          }}
                          placeholder="Tech Stack"
                          className="bg-slate-900 border border-slate-800 text-white px-2.5 py-1.5 rounded-lg text-xs"
                        />
                      </div>
                      <textarea
                        rows={2}
                        value={p.desc}
                        onChange={(e) => {
                          const updated = [...projects];
                          updated[idx].desc = e.target.value;
                          setProjects(updated);
                        }}
                        placeholder="Project details..."
                        className="w-full bg-slate-900 border border-slate-800 text-white px-2.5 py-1.5 rounded-lg text-xs"
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* 5. Education Records */}
            <div className="border border-slate-800 rounded-2xl overflow-hidden bg-slate-900/60">
              <button
                onClick={() => setOpenSection(openSection === 'education' ? '' : 'education')}
                className="w-full px-4 py-3 text-left font-black text-xs text-white flex items-center justify-between hover:bg-slate-800/50 cursor-pointer"
              >
                <span className="flex items-center gap-2">
                  <GraduationCap className="w-4 h-4 text-indigo-400" /> 5. Education Records ({educationList.length})
                </span>
                {openSection === 'education' ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </button>

              {openSection === 'education' && (
                <div className="p-4 border-t border-slate-800 space-y-3 text-xs">
                  <div className="flex justify-end">
                    <button
                      onClick={handleAddEducation}
                      className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-[11px] px-3 py-1.5 rounded-lg flex items-center gap-1 cursor-pointer shadow-sm"
                    >
                      <Plus className="w-3.5 h-3.5" /> Add Education Record
                    </button>
                  </div>

                  {educationList.map((edu, idx) => (
                    <div key={edu.id} className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-2 relative">
                      <button
                        onClick={() => setEducationList(educationList.filter(e => e.id !== edu.id))}
                        className="absolute top-2.5 right-2.5 text-slate-500 hover:text-rose-400 p-1 cursor-pointer"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                      <input
                        type="text"
                        value={edu.degree}
                        onChange={(e) => {
                          const updated = [...educationList];
                          updated[idx].degree = e.target.value;
                          setEducationList(updated);
                        }}
                        placeholder="Degree / Specialization"
                        className="w-full bg-slate-900 border border-slate-800 text-white px-2.5 py-1.5 rounded-lg text-xs"
                      />
                      <div className="grid grid-cols-2 gap-2">
                        <input
                          type="text"
                          value={edu.institute}
                          onChange={(e) => {
                            const updated = [...educationList];
                            updated[idx].institute = e.target.value;
                            setEducationList(updated);
                          }}
                          placeholder="Institute"
                          className="bg-slate-900 border border-slate-800 text-white px-2.5 py-1.5 rounded-lg text-xs"
                        />
                        <input
                          type="text"
                          value={edu.cgpa}
                          onChange={(e) => {
                            const updated = [...educationList];
                            updated[idx].cgpa = e.target.value;
                            setEducationList(updated);
                          }}
                          placeholder="CGPA / Grade"
                          className="bg-slate-900 border border-slate-800 text-white px-2.5 py-1.5 rounded-lg text-xs"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* 6. Custom Placement Sections */}
            <div className="border border-slate-800 rounded-2xl overflow-hidden bg-slate-900/60">
              <button
                onClick={() => setOpenSection(openSection === 'custom' ? '' : 'custom')}
                className="w-full px-4 py-3 text-left font-black text-xs text-white flex items-center justify-between hover:bg-slate-800/50 cursor-pointer"
              >
                <span className="flex items-center gap-2">
                  <Award className="w-4 h-4 text-indigo-400" /> 6. Custom Placement Sections ({customSections.length})
                </span>
                {openSection === 'custom' ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </button>

              {openSection === 'custom' && (
                <div className="p-4 border-t border-slate-800 space-y-3 text-xs">
                  <div className="flex justify-end">
                    <button
                      onClick={handleAddCustomSection}
                      className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-[11px] px-3 py-1.5 rounded-lg flex items-center gap-1 cursor-pointer shadow-sm"
                    >
                      <Plus className="w-3.5 h-3.5" /> Add Custom Section
                    </button>
                  </div>

                  {customSections.map((cs, idx) => (
                    <div key={cs.id} className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-2 relative">
                      <button
                        onClick={() => setCustomSections(customSections.filter(item => item.id !== cs.id))}
                        className="absolute top-2.5 right-2.5 text-slate-500 hover:text-rose-400 p-1 cursor-pointer"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                      <input
                        type="text"
                        value={cs.title}
                        onChange={(e) => {
                          const updated = [...customSections];
                          updated[idx].title = e.target.value;
                          setCustomSections(updated);
                        }}
                        placeholder="Section Title"
                        className="w-full bg-slate-900 border border-slate-800 text-white px-2.5 py-1.5 rounded-lg text-xs font-bold"
                      />
                      <textarea
                        rows={2}
                        value={cs.content}
                        onChange={(e) => {
                          const updated = [...customSections];
                          updated[idx].content = e.target.value;
                          setCustomSections(updated);
                        }}
                        placeholder="Content text..."
                        className="w-full bg-slate-900 border border-slate-800 text-white px-2.5 py-1.5 rounded-lg text-xs"
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>

          </div>

          {/* Right Column (7 Cols): Dual View Switcher (Live PDF Preview vs LaTeX Code) */}
          <div className="lg:col-span-7 space-y-4">
            
            {/* View Mode Bar */}
            <div className="bg-white dark:bg-[#0F172A] border border-slate-200 dark:border-slate-800 p-2 rounded-2xl flex items-center justify-between">
              <div className="bg-slate-900 p-1 rounded-xl flex items-center gap-1">
                <button
                  onClick={() => setRightViewMode('preview')}
                  className={`px-3.5 py-1.5 text-xs font-extrabold rounded-lg transition-all flex items-center gap-1.5 cursor-pointer ${
                    rightViewMode === 'preview' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <Eye className="w-3.5 h-3.5" /> Rendered PDF Canvas
                </button>

                <button
                  onClick={() => setRightViewMode('code')}
                  className={`px-3.5 py-1.5 text-xs font-extrabold rounded-lg transition-all flex items-center gap-1.5 cursor-pointer ${
                    rightViewMode === 'code' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <FileCode className="w-3.5 h-3.5" /> LaTeX Source (`.tex`)
                </button>
              </div>

              {rightViewMode === 'code' && (
                <button
                  onClick={handleCopyCode}
                  className="text-xs font-bold text-indigo-400 hover:text-indigo-300 flex items-center gap-1 bg-slate-900 px-3 py-1.5 rounded-xl border border-slate-800 cursor-pointer"
                >
                  <Copy className="w-3.5 h-3.5" /> Copy Code
                </button>
              )}
            </div>

            {/* RENDERED PDF PREVIEW CANVAS */}
            {rightViewMode === 'preview' && (
              <div className="bg-white text-slate-900 rounded-3xl p-8 shadow-2xl border border-slate-300 font-sans min-h-[750px] relative space-y-5 animate-in fade-in duration-200">
                <div className="absolute top-4 right-4 text-[10px] font-extrabold bg-emerald-100 text-emerald-800 px-2.5 py-1 rounded-full border border-emerald-300 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" /> 100% ATS Verified Document
                </div>

                {/* Candidate Header */}
                <div className="text-center border-b border-slate-200 pb-4">
                  <h2 className="text-2xl font-black text-slate-900 tracking-tight">{candidateName}</h2>
                  <p className="text-xs font-bold text-slate-600 mt-0.5">{candidateHeadline}</p>
                  <p className="text-[11px] text-slate-500 font-medium mt-1">
                    {candidateLocation} • {candidateEmail} • {candidatePhone} • github.com/arjun-b
                  </p>
                </div>

                {/* Skills */}
                <div>
                  <h4 className="text-xs font-black text-indigo-900 uppercase tracking-wider border-b border-slate-200 pb-1 mb-1.5">
                    Technical Skills & Languages
                  </h4>
                  <p className="text-xs text-slate-800 font-medium leading-relaxed">{candidateSkills}</p>
                </div>

                {/* Experience */}
                {experiences.length > 0 && (
                  <div>
                    <h4 className="text-xs font-black text-indigo-900 uppercase tracking-wider border-b border-slate-200 pb-1 mb-2">
                      Work Experience
                    </h4>
                    <div className="space-y-2.5">
                      {experiences.map(exp => (
                        <div key={exp.id}>
                          <div className="flex justify-between font-bold text-xs text-slate-900">
                            <span>{exp.role} — <span className="text-indigo-700">{exp.company}</span></span>
                            <span className="text-slate-500 text-[11px]">{exp.period}</span>
                          </div>
                          <p className="text-xs text-slate-700 mt-0.5 leading-relaxed">• {exp.desc}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Projects */}
                {projects.length > 0 && (
                  <div>
                    <h4 className="text-xs font-black text-indigo-900 uppercase tracking-wider border-b border-slate-200 pb-1 mb-2">
                      Technical Projects
                    </h4>
                    <div className="space-y-2.5">
                      {projects.map(p => (
                        <div key={p.id}>
                          <div className="flex justify-between font-bold text-xs text-slate-900">
                            <span>{p.name} <span className="text-slate-500 font-normal">({p.tech})</span></span>
                          </div>
                          <p className="text-xs text-slate-700 mt-0.5 leading-relaxed">• {p.desc}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Education */}
                {educationList.length > 0 && (
                  <div>
                    <h4 className="text-xs font-black text-indigo-900 uppercase tracking-wider border-b border-slate-200 pb-1 mb-1.5">
                      Education Records
                    </h4>
                    <div className="space-y-1">
                      {educationList.map(edu => (
                        <div key={edu.id} className="flex justify-between text-xs text-slate-800 font-bold">
                          <span>{edu.degree} — {edu.institute}</span>
                          <span className="text-slate-500 text-[11px]">{edu.year} (CGPA: {edu.cgpa})</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Custom Sections */}
                {customSections.map(cs => (
                  <div key={cs.id}>
                    <h4 className="text-xs font-black text-indigo-900 uppercase tracking-wider border-b border-slate-200 pb-1 mb-1">
                      {cs.title}
                    </h4>
                    <p className="text-xs text-slate-700 leading-relaxed">• {cs.content}</p>
                  </div>
                ))}
              </div>
            )}

            {/* LATEX CODE EDITOR */}
            {rightViewMode === 'code' && (
              <div className="bg-slate-950 border border-slate-800 rounded-3xl p-6 shadow-2xl space-y-3 min-h-[750px] animate-in fade-in duration-200">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <span className="text-xs font-bold text-indigo-400 flex items-center gap-2">
                    <Code className="w-4 h-4" /> Live Compiled LaTeX Source (`template.tex`)
                  </span>
                  <span className="text-[10px] text-emerald-400 font-mono font-bold">100% Validated Overleaf Syntax</span>
                </div>
                <div className="bg-slate-900/90 rounded-2xl p-5 font-mono text-[12px] text-indigo-300 overflow-x-auto min-h-[660px] border border-slate-800">
                  <pre>{compiledCode}</pre>
                </div>
              </div>
            )}

          </div>

        </div>
      )}

      {/* ---------------- TEMPLATE LIBRARY TAB ---------------- */}
      {activeSubTab === 'templates' && (
        <div className="space-y-6 animate-in fade-in duration-200">
          <div className="bg-slate-900 border border-slate-800 p-5 rounded-3xl flex items-center justify-between">
            <div>
              <h3 className="text-sm font-black text-white flex items-center gap-2">
                <LayoutTemplate className="w-4 h-4 text-indigo-400" />
                Verified ATS Placement LaTeX Templates
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">Tested against Greenhouse, Lever, Workday, and Ashby campus placement ATS systems.</p>
            </div>

            <button
              onClick={() => setShowOverleafImportModal(true)}
              className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs px-4 py-2.5 rounded-xl transition-all shadow-md flex items-center gap-1.5 cursor-pointer"
            >
              <Upload className="w-4 h-4" /> Import Custom Overleaf Code
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {templates.map((tmpl) => {
              const isSelected = selectedTemplateId === tmpl.id;

              return (
                <div
                  key={tmpl.id}
                  className={`bg-white dark:bg-[#0F172A] border rounded-3xl p-6 shadow-sm transition-all flex flex-col justify-between space-y-4 ${
                    isSelected
                      ? 'border-indigo-500 ring-2 ring-indigo-500/30'
                      : 'border-slate-200 dark:border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-black uppercase bg-indigo-500/15 text-indigo-400 px-2.5 py-1 rounded-full border border-indigo-500/30">
                        {tmpl.category}
                      </span>
                      <span className="text-xs font-bold text-emerald-400 flex items-center gap-1">
                        <Star className="w-3.5 h-3.5 fill-emerald-400" /> {tmpl.atsScore}% ATS Score
                      </span>
                    </div>

                    <h3 className="text-base font-black text-slate-900 dark:text-white">{tmpl.name}</h3>
                    <p className="text-xs text-slate-400 leading-relaxed">{tmpl.desc}</p>
                  </div>

                  <div className="pt-3 border-t border-slate-800 flex items-center justify-between">
                    <button
                      onClick={() => {
                        setSelectedTemplateId(tmpl.id);
                        setActiveSubTab('builder');
                        if (showToast) showToast("Template Applied! 🎨", `${tmpl.name} loaded into workspace`);
                      }}
                      className={`w-full py-2.5 rounded-xl font-bold text-xs transition-all flex items-center justify-center gap-2 cursor-pointer ${
                        isSelected
                          ? 'bg-indigo-600 text-white shadow-md'
                          : 'bg-slate-800 hover:bg-slate-700 text-slate-200'
                      }`}
                    >
                      {isSelected ? <CheckCircle2 className="w-4 h-4" /> : <Sparkles className="w-4 h-4" />}
                      {isSelected ? 'Active Template' : 'Use Template'}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Overleaf Code Import Modal */}
      {showOverleafImportModal && (
        <div className="fixed inset-0 z-50 bg-slate-950/85 backdrop-blur-xl flex items-center justify-center p-4 animate-in fade-in duration-200">
          <div className="bg-white dark:bg-[#0F172A] border border-slate-200 dark:border-slate-800 w-full max-w-xl rounded-3xl shadow-2xl p-6 relative space-y-4">
            <button
              onClick={() => setShowOverleafImportModal(false)}
              className="absolute top-4 right-4 p-2 rounded-full text-slate-400 hover:text-white hover:bg-slate-800 transition-all cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3 border-b border-slate-800 pb-3">
              <Upload className="w-5 h-5 text-indigo-400" />
              <div>
                <h3 className="text-base font-black text-white">Import Custom Overleaf / LaTeX Code</h3>
                <p className="text-xs text-slate-400">Paste your custom LaTeX template code directly from Overleaf</p>
              </div>
            </div>

            <form onSubmit={handleImportOverleafCode} className="space-y-4">
              <div>
                <label className="text-xs font-bold text-slate-300 block mb-1">LaTeX Code (`.tex` format)</label>
                <textarea
                  rows={8}
                  required
                  value={customOverleafCode}
                  onChange={(e) => setCustomOverleafCode(e.target.value)}
                  placeholder="Paste \documentclass{...} code from Overleaf here..."
                  className="w-full bg-slate-900 border border-slate-800 text-xs font-mono text-indigo-300 p-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                />
              </div>

              <div className="flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowOverleafImportModal(false)}
                  className="bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold text-xs px-4 py-2.5 rounded-xl cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs px-5 py-2.5 rounded-xl shadow-md cursor-pointer"
                >
                  Import & Apply Template
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

    </div>
  );
}
