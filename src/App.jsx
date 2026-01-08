import React, { useState } from 'react';
import { BookOpen, Code, Database, AlertCircle } from 'lucide-react';
import SingleValueColumn from './component/SingleValueColumn';
import VeryFewUniqueValues from './component/VeryFewUniqueValues';
import LowVariance from './component/LowVariance';
import DuplicateRows from './component/DuplicateRows';
import MissingSpaces from './component/MissingSpaces';
import MissingNA from './component/MissingNA';
import MissingQuestionMark from './component/MissingQuestionMark';
import Outliers from './component/Outliers';
import InconsistentCategories from './component/InconsistentCategories';
import LeadingTrailingSpaces from './component/LeadingTrailingSpaces';
import ImpossibleValues from './component/ImpossibleValues'; 
import RareCategories from './component/RareCategories'; 
import SkewedNumericValues from './component/SkewedNumericValues'; 
import CorruptedText from './component/CorruptedText'; 

// ============= Main App Component =============

const App = () => {
  const [activeTab, setActiveTab] = useState(1);

  const problems = [
    { id: 1, name: 'Single-value columns', completed: true, component: SingleValueColumn },
    { id: 2, name: 'Very few unique values', completed: false, component: VeryFewUniqueValues },
    { id: 3, name: 'Low variance', completed: false, component: LowVariance },
    { id: 4, name: 'Duplicate rows', completed: false, component: DuplicateRows },
    { id: 5, name: 'Missing " "', completed: false, component: MissingSpaces },
    { id: 6, name: 'Missing "na"', completed: false, component: MissingNA },
    { id: 7, name: 'Missing "?"', completed: false, component: MissingQuestionMark },
    { id: 8, name: 'Outliers', completed: false, component: Outliers },
    { id: 9, name: 'Inconsistent categories', completed: false, component: InconsistentCategories },
    { id: 10, name: 'Wrong data types', completed: false, component: null },
    { id: 11, name: 'Leading / trailing spaces', completed: false, component: LeadingTrailingSpaces },
    { id: 12, name: 'Impossible values', completed: false, component: ImpossibleValues },
    { id: 13, name: 'Mixed units', completed: false, component: null },
    { id: 14, name: 'Rare categories', completed: false, component: RareCategories },
    { id: 15, name: 'Skewed numeric values', completed: false, component: SkewedNumericValues },
    { id: 16, name: 'Corrupted text', completed: false, component: CorruptedText },
    { id: 17, name: 'Duplicate meaning columns', completed: false, component: null },
    { id: 18, name: 'Target leakage', completed: false, component: null },
  ];

  const currentProblem = problems.find(p => p.id === activeTab);
  const ProblemComponent = currentProblem?.component;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto p-6">
        {/* Header */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 mb-6 border border-white/20">
          <div className="flex items-center gap-4 mb-4">
            <div className="bg-purple-500 p-3 rounded-xl">
              <Database className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold text-white">Data Cleaning Guide</h1>
              <p className="text-purple-200 mt-1">Master the art of preparing data for Machine Learning</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-12 gap-6">
          {/* Sidebar */}
          <div className="col-span-3">
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 border border-white/20 sticky top-6">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <BookOpen className="w-5 h-5" />
                Problems
              </h2>
              <div className="space-y-2 max-h-[calc(100vh-200px)] overflow-y-auto">
                {problems.map((problem) => (
                  <button
                    key={problem.id}
                    onClick={() => setActiveTab(problem.id)}
                    className={`w-full text-left px-4 py-3 rounded-lg transition-all duration-200 ${
                      activeTab === problem.id
                        ? 'bg-purple-500 text-white shadow-lg shadow-purple-500/50'
                        : problem.completed
                        ? 'bg-white/10 text-purple-200 hover:bg-white/20'
                        : 'bg-white/5 text-gray-400 hover:bg-white/10'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-semibold">{problem.id}.</span>
                      <span className="text-sm">{problem.name}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="col-span-9">
            {ProblemComponent ? (
              <ProblemComponent />
            ) : (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-12 border border-white/20 text-center">
                <AlertCircle className="w-16 h-16 text-purple-400 mx-auto mb-4" />
                <h3 className="text-2xl font-bold text-white mb-2">Coming Soon</h3>
                <p className="text-purple-200">This problem's solution will be added shortly.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;