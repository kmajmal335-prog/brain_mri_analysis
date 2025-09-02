import React, { useState, useEffect } from 'react';
import { apiService } from '../utils/apiService';

const ModelLoader = () => {
  const [modelPath, setModelPath] = useState('');
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    checkBackendStatus();
    checkModelStatus();
  }, []);

  const checkBackendStatus = async () => {
    try {
      await apiService.healthCheck();
      setBackendStatus('connected');
    } catch (error) {
      setBackendStatus('disconnected');
      setError('Backend is not running. Please start the FastAPI backend.');
    }
  };

  const checkModelStatus = async () => {
    try {
      const info = await apiService.getModelInfo();
      setModelInfo(info);
    } catch (error) {
      // Model not loaded yet, which is fine
      setModelInfo(null);
    }
  };

  const handleLoadModel = async (e) => {
    e.preventDefault();
    if (!modelPath.trim()) {
      setError('Please enter a model path');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const result = await apiService.loadModel(modelPath);
      setModelInfo(result.model_info);
      setError('');
      alert('Model loaded successfully!');
    } catch (error) {
      setError(`Failed to load model: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'connected': return 'text-green-600';
      case 'disconnected': return 'text-red-600';
      default: return 'text-yellow-600';
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'connected': return 'Connected';
      case 'disconnected': return 'Disconnected';
      default: return 'Checking...';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Model Configuration</h2>
      
      {/* Backend Status */}
      <div className="mb-4 p-3 bg-gray-50 rounded-lg">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700">Backend Status:</span>
          <span className={`text-sm font-semibold ${getStatusColor(backendStatus)}`}>
            {getStatusText(backendStatus)}
          </span>
        </div>
        {backendStatus === 'disconnected' && (
          <div className="mt-2 text-sm text-red-600">
            Make sure to start the FastAPI backend on port 8000
          </div>
        )}
      </div>

      {/* Model Status */}
      {modelInfo ? (
        <div className="mb-4 p-4 bg-green-50 border border-green-200 rounded-lg">
          <h3 className="text-lg font-medium text-green-800 mb-2">✅ Model Loaded</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium text-gray-700">Type:</span>
              <span className="ml-2 text-gray-600">{modelInfo.model_type}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Device:</span>
              <span className="ml-2 text-gray-600">{modelInfo.device}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Classes:</span>
              <span className="ml-2 text-gray-600">{modelInfo.num_classes}</span>
            </div>
            {modelInfo.accuracy && (
              <div>
                <span className="font-medium text-gray-700">Accuracy:</span>
                <span className="ml-2 text-gray-600">{(modelInfo.accuracy * 100).toFixed(2)}%</span>
              </div>
            )}
            <div className="col-span-2">
              <span className="font-medium text-gray-700">Loaded:</span>
              <span className="ml-2 text-gray-600">
                {new Date(modelInfo.loaded_at).toLocaleString()}
              </span>
            </div>
            <div className="col-span-2">
              <span className="font-medium text-gray-700">Path:</span>
              <span className="ml-2 text-gray-600 text-xs break-all">{modelInfo.model_path}</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="mb-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <h3 className="text-lg font-medium text-yellow-800 mb-2">⚠️ No Model Loaded</h3>
          <p className="text-sm text-yellow-700">
            Please load a trained model (.pth file) to start making predictions.
          </p>
        </div>
      )}

      {/* Model Loading Form */}
      <form onSubmit={handleLoadModel} className="space-y-4">
        <div>
          <label htmlFor="modelPath" className="block text-sm font-medium text-gray-700 mb-2">
            Model Path (.pth file)
          </label>
          <input
            type="text"
            id="modelPath"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            placeholder="/path/to/your/model.pth"
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            disabled={loading || backendStatus !== 'connected'}
          />
          <p className="mt-1 text-xs text-gray-500">
            Enter the full path to your trained model file (e.g., /workspace/project/brain_mri_analysis/models/best_model.pth)
          </p>
        </div>

        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-md">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        <button
          type="submit"
          disabled={loading || backendStatus !== 'connected'}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Loading Model...' : 'Load Model'}
        </button>
      </form>

      {/* Instructions */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h4 className="text-sm font-medium text-blue-800 mb-2">Instructions:</h4>
        <ol className="text-xs text-blue-700 space-y-1 list-decimal list-inside">
          <li>Make sure your FastAPI backend is running on port 8000</li>
          <li>Train your model using the training scripts</li>
          <li>Enter the full path to your saved .pth model file</li>
          <li>Click "Load Model" to initialize the model for predictions</li>
          <li>Once loaded, you can upload MRI images for analysis</li>
        </ol>
      </div>

      {/* Model Classes Info */}
      {modelInfo && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Supported Classes:</h4>
          <div className="flex flex-wrap gap-2">
            {modelInfo.class_names.map((className, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
              >
                {className}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelLoader;