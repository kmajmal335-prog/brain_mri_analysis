import React, { useState, useEffect } from 'react';
import { Calendar, Filter, Search, Download, Eye } from 'lucide-react';
import { motion } from 'framer-motion';
import { apiService } from '../utils/apiService';

const History = ({ onViewScan }) => {
  const [scans, setScans] = useState([]);
  const [filteredScans, setFilteredScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');

  useEffect(() => {
    const loadScans = async () => {
      try {
        const scanData = await apiService.getScans();
        setScans(scanData);
        setFilteredScans(scanData);
      } catch (error) {
        console.error('Error loading scans:', error);
      } finally {
        setLoading(false);
      }
    };

    loadScans();
  }, []);

  useEffect(() => {
    let filtered = scans;

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(scan =>
        scan.filename.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply type filter
    if (filterType !== 'all') {
      filtered = filtered.filter(scan => 
        scan.prediction.toLowerCase() === filterType
      );
    }

    setFilteredScans(filtered);
  }, [scans, searchTerm, filterType]);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-neutral-200 rounded w-1/3"></div>
          <div className="space-y-4">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-20 bg-neutral-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-secondary-500 mb-2">Scan History</h1>
        <p className="text-neutral-500">View and manage all your MRI scan analyses</p>
      </div>

      {/* Filters and Search */}
      <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-neutral-400" />
            <input
              type="text"
              placeholder="Search by filename..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-neutral-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
          </div>

          {/* Filter */}
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-neutral-400" />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="pl-10 pr-8 py-2 border border-neutral-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 bg-white"
            >
              <option value="all">All Results</option>
              <option value="normal">Normal Only</option>
              <option value="tumor">Tumor Only</option>
            </select>
          </div>
        </div>
      </div>

      {/* Results Table */}
      <div className="bg-white rounded-xl shadow-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-neutral-100">
          <h3 className="text-lg font-semibold text-secondary-500">
            {filteredScans.length} Scan{filteredScans.length !== 1 ? 's' : ''} Found
          </h3>
        </div>

        {filteredScans.length === 0 ? (
          <div className="text-center py-12">
            <Calendar className="h-12 w-12 text-neutral-300 mx-auto mb-4" />
            <p className="text-neutral-500">No scans match your criteria</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-neutral-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    Scan
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    Result
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    Confidence
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-neutral-100">
                {filteredScans.map((scan, index) => (
                  <motion.tr
                    key={scan.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.05 * index }}
                    className="hover:bg-neutral-50 transition-colors"
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-3">
                        <img
                          src={scan.imageUrl}
                          alt="MRI Preview"
                          className="w-10 h-10 object-cover rounded-lg border"
                        />
                        <div>
                          <p className="font-semibold text-secondary-500">{scan.filename}</p>
                          <p className="text-sm text-neutral-500">{scan.metadata.size}</p>
                        </div>
                      </div>
                    </td>
                    
                    <td className="px-6 py-4 whitespace-nowrap">
                      <p className="text-secondary-500">{new Date(scan.uploadDate).toLocaleDateString()}</p>
                      <p className="text-sm text-neutral-500">{new Date(scan.uploadDate).toLocaleTimeString()}</p>
                    </td>
                    
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                        scan.prediction === 'Tumor'
                          ? 'bg-red-100 text-medical-error'
                          : 'bg-green-100 text-medical-success'
                      }`}>
                        {scan.prediction}
                      </span>
                    </td>
                    
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-2">
                        <div className="w-16 bg-neutral-100 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              scan.prediction === 'Tumor' ? 'bg-medical-error' : 'bg-medical-success'
                            }`}
                            style={{ width: `${scan.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-neutral-600">
                          {Math.round(scan.confidence * 100)}%
                        </span>
                      </div>
                    </td>
                    
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => onViewScan(scan)}
                          className="p-2 text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
                          title="View Scan"
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                        <button
                          className="p-2 text-neutral-600 hover:bg-neutral-50 rounded-lg transition-colors"
                          title="Download Report"
                        >
                          <Download className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default History;