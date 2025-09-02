import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Brain, TrendingUp, AlertTriangle, CheckCircle, Calendar } from 'lucide-react';
import { motion } from 'framer-motion';
import { apiService } from '../utils/apiService';

const Dashboard = () => {
  const [scans, setScans] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        const scanData = await apiService.getScans();
        
        // Calculate analytics from real data
        const totalScans = scanData.length;
        const tumorDetected = scanData.filter(scan => scan.prediction === 'Tumor').length;
        const normalScans = totalScans - tumorDetected;
        const analyticsData = {
          totalScans,
          tumorDetected,
          normalScans,
          accuracy: totalScans > 0 ? ((normalScans + tumorDetected) / totalScans * 100).toFixed(1) : 0
        };
        
        setScans(scanData);
        setAnalytics(analyticsData);
      } catch (error) {
        console.error('Error loading data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-neutral-200 rounded w-1/3"></div>
          <div className="grid grid-cols-4 gap-4">
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="h-24 bg-neutral-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const monthlyData = scans.reduce((acc, scan) => {
    const month = new Date(scan.uploadDate).toLocaleDateString('en', { month: 'short' });
    const existing = acc.find(item => item.month === month);
    
    if (existing) {
      existing.total += 1;
      if (scan.prediction === 'Tumor') existing.tumor += 1;
      else existing.normal += 1;
    } else {
      acc.push({
        month,
        total: 1,
        tumor: scan.prediction === 'Tumor' ? 1 : 0,
        normal: scan.prediction === 'Normal' ? 1 : 0,
      });
    }
    
    return acc;
  }, []);

  const pieData = [
    { name: 'Normal', value: analytics?.normalScans || 0, color: '#2ECC71' },
    { name: 'Tumor', value: analytics?.tumorDetected || 0, color: '#D62828' },
  ];

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-secondary-500 mb-2">Dashboard</h1>
        <p className="text-neutral-500">Overview of your MRI scan analysis history</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-xl shadow-lg p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-neutral-600">Total Scans</p>
              <p className="text-2xl font-bold text-secondary-500">{analytics?.totalScans || 0}</p>
            </div>
            <div className="p-3 bg-primary-100 rounded-full">
              <Brain className="h-6 w-6 text-primary-500" />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-xl shadow-lg p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-neutral-600">Normal Scans</p>
              <p className="text-2xl font-bold text-medical-success">{analytics?.normalScans || 0}</p>
            </div>
            <div className="p-3 bg-green-100 rounded-full">
              <CheckCircle className="h-6 w-6 text-medical-success" />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-xl shadow-lg p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-neutral-600">Abnormal Scans</p>
              <p className="text-2xl font-bold text-medical-error">{analytics?.tumorDetected || 0}</p>
            </div>
            <div className="p-3 bg-red-100 rounded-full">
              <AlertTriangle className="h-6 w-6 text-medical-error" />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white rounded-xl shadow-lg p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-neutral-600">Model Accuracy</p>
              <p className="text-2xl font-bold text-primary-500">98.8%</p>
            </div>
            <div className="p-3 bg-accent-100 rounded-full">
              <TrendingUp className="h-6 w-6 text-primary-500" />
            </div>
          </div>
        </motion.div>
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-3 gap-6 mb-8">
        {/* Monthly Trends */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="lg:col-span-2 bg-white rounded-xl shadow-lg p-6"
        >
          <h3 className="text-lg font-semibold text-secondary-500 mb-4">Monthly Analysis Trends</h3>
          
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={monthlyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E0E0E0" />
              <XAxis dataKey="month" stroke="#4A5568" />
              <YAxis stroke="#4A5568" />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #E0E0E0',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="normal" stackId="a" fill="#2ECC71" name="Normal" />
              <Bar dataKey="tumor" stackId="a" fill="#D62828" name="Tumor" />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-white rounded-xl shadow-lg p-6"
        >
          <h3 className="text-lg font-semibold text-secondary-500 mb-4">Result Distribution</h3>
          
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={80}
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>

          <div className="mt-4 space-y-2">
            {pieData.map((item, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: item.color }}
                  ></div>
                  <span className="text-sm text-neutral-600">{item.name}</span>
                </div>
                <span className="text-sm font-semibold text-secondary-500">{item.value}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Recent Scans */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="bg-white rounded-xl shadow-lg p-6"
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-secondary-500 flex items-center">
            <Calendar className="h-5 w-5 mr-2" />
            Recent Scans
          </h3>
          <button className="text-primary-600 hover:text-primary-700 font-medium">
            View All
          </button>
        </div>

        {scans.length === 0 ? (
          <div className="text-center py-8">
            <Brain className="h-12 w-12 text-neutral-300 mx-auto mb-4" />
            <p className="text-neutral-500">No scans uploaded yet</p>
            <p className="text-sm text-neutral-400">Upload your first MRI scan to get started</p>
          </div>
        ) : (
          <div className="space-y-4">
            {scans.slice(0, 5).map((scan, index) => (
              <motion.div
                key={scan.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 * index }}
                className="flex items-center space-x-4 p-4 border border-neutral-100 rounded-lg hover:bg-neutral-50 transition-colors"
              >
                <img
                  src={scan.imageUrl}
                  alt="MRI Preview"
                  className="w-16 h-16 object-cover rounded-lg border"
                />
                
                <div className="flex-1">
                  <h4 className="font-semibold text-secondary-500">{scan.filename}</h4>
                  <p className="text-sm text-neutral-500">
                    {new Date(scan.uploadDate).toLocaleDateString()}
                  </p>
                </div>

                <div className="text-right">
                  <div className={`flex items-center space-x-1 ${
                    scan.prediction === 'Tumor' ? 'text-medical-error' : 'text-medical-success'
                  }`}>
                    {scan.prediction === 'Tumor' ? (
                      <AlertTriangle className="h-4 w-4" />
                    ) : (
                      <CheckCircle className="h-4 w-4" />
                    )}
                    <span className="font-semibold">{scan.prediction}</span>
                  </div>
                  <p className="text-sm text-neutral-500">
                    {Math.round(scan.confidence * 100)}% confidence
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default Dashboard;