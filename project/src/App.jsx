import React, { useState } from 'react';
import { Brain } from 'lucide-react';
import Navigation from './components/Navigation';
import UploadPage from './components/UploadPage';
import Viewer from './components/Viewer';
import Results from './components/Results';
import Dashboard from './components/Dashboard';
import History from './components/History';
import Home from './pages/Home';

function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [currentScan, setCurrentScan] = useState(null);
  const [viewMode, setViewMode] = useState('upload'); // upload, viewer, results

  const handleNavigate = (page) => {
    setCurrentPage(page);
    if (page !== 'upload') {
      setViewMode('upload');
      setCurrentScan(null);
    }
  };

  const handleUploadComplete = (scan) => {
    setCurrentScan(scan);
    setViewMode('viewer');
  };

  const handleProceedToResults = () => {
    setViewMode('results');
  };

  const handleBackFromViewer = () => {
    setViewMode('upload');
  };

  const handleBackFromResults = () => {
    setViewMode('viewer');
  };

  const handleViewScan = (scan) => {
    setCurrentScan(scan);
    setCurrentPage('upload');
    setViewMode('results');
  };

  const handleViewHistory = () => {
    setCurrentPage('history');
  };

  const renderContent = () => {
    if (currentPage === 'home') {
      return <Home onNavigate={handleNavigate} />;
    }

    if (currentPage === 'upload') {
      if (viewMode === 'results' && currentScan) {
        return (
          <Results
            scan={currentScan}
            onBack={handleBackFromResults}
            onViewHistory={handleViewHistory}
          />
        );
      }
      
      if (viewMode === 'viewer' && currentScan) {
        return (
          <Viewer
            scan={currentScan}
            onBack={handleBackFromViewer}
            onProceedToResults={handleProceedToResults}
          />
        );
      }
      
      return <UploadPage onUploadComplete={handleUploadComplete} />;
    }

    if (currentPage === 'history') {
      return <History onViewScan={handleViewScan} />;
    }

    if (currentPage === 'dashboard') {
      return <Dashboard />;
    }

    return <Home onNavigate={handleNavigate} />;
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation currentPage={currentPage} onNavigate={handleNavigate} />
      <main>
        {renderContent()}
      </main>

      {/* Footer */}
      <footer className="bg-secondary-500 text-white py-8 mt-12">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <Brain className="h-6 w-6" />
            <span className="text-lg font-semibold">MRI Analyzer</span>
          </div>
          <p className="text-secondary-100 mb-2">
            Advanced AI-powered brain tumor detection for medical professionals
          </p>
          <p className="text-sm text-secondary-200">
            © 2025 MRI Analyzer. For research and educational purposes.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;