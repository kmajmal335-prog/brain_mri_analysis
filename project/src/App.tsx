import React, { useState, useEffect } from 'react';
import { apiService } from './utils/apiService';

function App() {
  const [scans, setScans] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchScans();
  }, []);

  const fetchScans = async () => {
    try {
      setLoading(true);
      const data = await apiService.getScans();
      setScans(data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch scans. Make sure the backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    try {
      setLoading(true);
      await apiService.uploadMRI(file);
      await fetchScans(); // Refresh the scans list
      setError(null);
    } catch (err) {
      setError('Failed to upload file. Make sure the backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8">Brain Tumor Detection</h1>
        
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}
        
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Upload MRI Scan</h2>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            disabled={loading}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100"
          />
          {loading && <p className="mt-2 text-blue-600">Processing...</p>}
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Scan History</h2>
          {loading && scans.length === 0 ? (
            <p>Loading scans...</p>
          ) : scans.length === 0 ? (
            <p>No scans found. Upload an MRI scan to get started.</p>
          ) : (
            <div className="space-y-4">
              {scans.map((scan) => (
                <div key={scan.id} className="border rounded-lg p-4">
                  <div className="flex justify-between items-center">
                    <h3 className="font-medium">{scan.filename}</h3>
                    <span className={`px-2 py-1 rounded text-sm font-medium ${
                      scan.prediction === 'Tumor' 
                        ? 'bg-red-100 text-red-800' 
                        : 'bg-green-100 text-green-800'
                    }`}>
                      {scan.prediction} ({(scan.confidence * 100).toFixed(1)}%)
                    </span>
                  </div>
                  <p className="text-sm text-gray-500 mt-1">
                    Uploaded: {new Date(scan.uploadDate).toLocaleString()}
                  </p>
                  {scan.imageUrl && (
                    <img 
                      src={`http://localhost:12000${scan.imageUrl}`} 
                      alt="MRI Scan" 
                      className="mt-2 max-w-xs h-auto"
                    />
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
