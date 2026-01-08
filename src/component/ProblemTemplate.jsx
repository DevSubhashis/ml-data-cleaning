// ============= Reusable Template Component =============
import React from 'react';
import { BookOpen, Code, Database, AlertCircle } from 'lucide-react';

const ProblemTemplate = ({ data, problemNumber }) => {
  return (
    <div className="space-y-6">
      {/* Problem Header */}
      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
        <div className="flex items-start gap-4">
          <div className="bg-purple-500 p-3 rounded-xl">
            <AlertCircle className="w-6 h-6 text-white" />
          </div>
          <div className="flex-1">
            <h2 className="text-3xl font-bold text-white mb-2">
              {problemNumber}. {data.title}
            </h2>
            <p className="text-purple-200 leading-relaxed">
              {data.description}
            </p>
          </div>
        </div>
      </div>

      {/* Original Dataset Table */}
      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
        <div className="flex items-center gap-2 mb-4">
          <Database className="w-5 h-5 text-red-400" />
          <h3 className="text-xl font-bold text-white">Original Dataset (With Problem)</h3>
          <span className="ml-auto bg-red-500/20 text-red-300 px-3 py-1 rounded-full text-sm">
            {data.originalData.length} rows × {Object.keys(data.originalData[0]).length} cols
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-800/50 border-b border-white/10">
                {Object.keys(data.originalData[0]).map((key) => (
                  <th key={key} className={`px-4 py-3 text-left font-semibold ${
                    data.removedColumns?.includes(key) 
                      ? 'text-red-300 bg-red-500/10' 
                      : 'text-purple-300'
                  }`}>
                    {key}
                    {data.removedColumns?.includes(key) && (
                      <span className="ml-2 text-xs bg-red-500 text-white px-2 py-0.5 rounded">Issue</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.originalData.map((row, idx) => (
                <tr key={idx} className="border-b border-white/5 hover:bg-white/5">
                  {Object.entries(row).map(([key, value]) => (
                    <td key={key} className={`px-4 py-3 ${
                      data.removedColumns?.includes(key)
                        ? 'text-red-200 bg-red-500/5'
                        : 'text-gray-200'
                    }`}>
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Cleaned Dataset Table */}
      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
        <div className="flex items-center gap-2 mb-4">
          <Database className="w-5 h-5 text-green-400" />
          <h3 className="text-xl font-bold text-white">Cleaned Dataset (After Processing)</h3>
          <span className="ml-auto bg-green-500/20 text-green-300 px-3 py-1 rounded-full text-sm">
            {data.cleanedData.length} rows × {Object.keys(data.cleanedData[0]).length} cols
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-800/50 border-b border-white/10">
                {Object.keys(data.cleanedData[0]).map((key) => (
                  <th key={key} className="px-4 py-3 text-left text-green-300 font-semibold">
                    {key}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.cleanedData.map((row, idx) => (
                <tr key={idx} className="border-b border-white/5 hover:bg-white/5">
                  {Object.values(row).map((value, i) => (
                    <td key={i} className="px-4 py-3 text-gray-200">
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {data.removedColumns && data.removedColumns.length > 0 && (
          <div className="mt-4 bg-green-500/10 border border-green-500/20 rounded-lg p-4">
            <p className="text-green-300 text-sm">
              ✓ Removed {data.removedColumns.length} column(s): <span className="font-semibold">{data.removedColumns.join(', ')}</span>
            </p>
          </div>
        )}
      </div>

      {/* Python Code - Test Dataset */}
      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
        <div className="flex items-center gap-2 mb-4">
          <Code className="w-5 h-5 text-purple-400" />
          <h3 className="text-xl font-bold text-white">Python Code - Generate Test Dataset</h3>
        </div>
        <div className="bg-slate-900/50 rounded-xl p-4 border border-white/10">
          <pre className="text-green-300 text-sm overflow-x-auto">
            <code>{data.testDataset}</code>
          </pre>
        </div>
      </div>

      {/* Solution */}
      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
        <div className="flex items-center gap-2 mb-4">
          <Code className="w-5 h-5 text-green-400" />
          <h3 className="text-xl font-bold text-white">Solution Code</h3>
        </div>
        <div className="bg-slate-900/50 rounded-xl p-4 border border-white/10">
          <pre className="text-blue-300 text-sm overflow-x-auto">
            <code>{data.solution}</code>
          </pre>
        </div>
      </div>

      {/* Explanation */}
      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
        <div className="flex items-center gap-2 mb-4">
          <BookOpen className="w-5 h-5 text-yellow-400" />
          <h3 className="text-xl font-bold text-white">Explanation</h3>
        </div>
        <div className="text-purple-100 leading-relaxed">
          {data.explanation.split('\n').map((line, i) => (
            <p key={i} className="mb-2">{line}</p>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ProblemTemplate;