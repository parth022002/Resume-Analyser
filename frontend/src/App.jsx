import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import FloatingAssistant from './components/FloatingAssistant';
import Toast from './components/Toast';
import AuthModal from './components/AuthModal';

import Dashboard from './pages/Dashboard';
import JobFeed from './pages/JobFeed';
import ApplicationsTracker from './pages/ApplicationsTracker';
import TargetCompanies from './pages/TargetCompanies';
import CompanyResearch from './pages/CompanyResearch';
import ResumeBuilder from './pages/ResumeBuilder';
import ActionPlan from './pages/ActionPlan';
import ReportsAnalytics from './pages/ReportsAnalytics';
import Settings from './pages/Settings';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [theme, setTheme] = useState(() => localStorage.getItem('talentforge_theme') || 'dark');
  const [searchQuery, setSearchQuery] = useState('');
  const [toast, setToast] = useState(null);
  
  // Student Auth State
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('talentforge_student_user');
    return saved ? JSON.parse(saved) : {
      id: 1,
      full_name: "Arjun B.",
      email: "arjun.b@talentforge.ai",
      avatar_url: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150",
      plan: "Free Student Account",
      headline: "Software Engineer - Backend & Systems",
      skills: ["Python", "FastAPI", "AWS", "PostgreSQL", "Docker", "Microservices"],
      experience_years: 3.5,
      preferred_roles: ["Software Engineer", "Backend Developer", "SDE II"],
      preferred_locations: ["Bengaluru", "Remote", "Hybrid"],
      education_details: [
        { degree: "B.Tech Computer Science", institute: "RV College of Engineering", year: "2024", cgpa: "8.8 / 10" }
      ]
    };
  });

  useEffect(() => {
    localStorage.setItem('talentforge_theme', theme);
    if (theme === 'dark') {
      document.body.classList.add('dark');
      document.body.classList.remove('light');
    } else {
      document.body.classList.add('light');
      document.body.classList.remove('dark');
    }
  }, [theme]);

  // Sync fresh student profile directly from Neon PostgreSQL database
  useEffect(() => {
    if (user && user.id) {
      fetch(`http://localhost:8000/api/v1/auth/user/${user.id}`)
        .then(res => res.json())
        .then(data => {
          if (data && data.full_name) {
            setUser(data);
            localStorage.setItem('talentforge_student_user', JSON.stringify(data));
          }
        })
        .catch(err => console.log('Neon PostgreSQL profile sync notice:', err));
    }
  }, [user?.id]);

  const handleUpdateUser = (updatedUser) => {
    setUser(updatedUser);
    localStorage.setItem('talentforge_student_user', JSON.stringify(updatedUser));
  };

  const showToast = (title, message) => {
    setToast({ title, message });
  };

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard user={user} searchQuery={searchQuery} onSelectJob={() => setActiveTab('jobs')} showToast={showToast} />;
      case 'jobs':
        return <JobFeed searchQuery={searchQuery} showToast={showToast} />;
      case 'applications':
      case 'tracker':
        return <ApplicationsTracker searchQuery={searchQuery} showToast={showToast} />;
      case 'target-companies':
        return <TargetCompanies user={user} searchQuery={searchQuery} showToast={showToast} />;
      case 'company-research':
        return <CompanyResearch searchQuery={searchQuery} />;
      case 'resume-builder':
      case 'templates':
        return <ResumeBuilder showToast={showToast} />;
      case 'action-plan':
      case 'interview-coach':
        return <ActionPlan showToast={showToast} />;
      case 'assistant':
        return <Dashboard user={user} searchQuery={searchQuery} showToast={showToast} />;
      case 'reports':
        return <ReportsAnalytics user={user} />;
      case 'settings':
        return <Settings user={user} onUpdateUser={handleUpdateUser} showToast={showToast} />;
      default:
        return <Dashboard user={user} searchQuery={searchQuery} showToast={showToast} />;
    }
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('talentforge_student_user');
    showToast('Logged Out 👋', 'Signed out of student session.');
    setIsAuthOpen(true);
  };

  return (
    <div className={`min-h-screen flex bg-slate-50 dark:bg-[#030712] text-slate-800 dark:text-slate-100 font-sans transition-colors duration-300`}>
      {/* Dark Navy Sidebar */}
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Workspace Area */}
      <div className="flex-1 flex flex-col min-w-0 pb-20">
        {/* Header Bar */}
        <Header
          theme={theme}
          setTheme={setTheme}
          onSearch={setSearchQuery}
          user={user}
          onOpenAuth={() => setIsAuthOpen(true)}
          onNavigateToSettings={() => setActiveTab('settings')}
          onLogout={handleLogout}
        />

        {/* View Content Body */}
        <main className="p-6 md:p-8 flex-1">
          {renderActiveTab()}
        </main>

        {/* Floating Bottom-Right Corner AI Assistant */}
        <FloatingAssistant user={user} />

        {/* Toast Notification Container */}
        <Toast toast={toast} onClose={() => setToast(null)} />

        {/* Student Auth & Onboarding Modal */}
        <AuthModal
          isOpen={isAuthOpen}
          onClose={() => setIsAuthOpen(false)}
          onLoginSuccess={(loggedInUser) => {
            handleUpdateUser(loggedInUser);
            setIsAuthOpen(false);
          }}
          showToast={showToast}
        />
      </div>
    </div>
  );
}
