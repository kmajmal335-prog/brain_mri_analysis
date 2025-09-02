import React, { useState } from 'react';
import { ZoomIn, ZoomOut, RotateCw, Download, ArrowLeft, CheckCircle } from 'lucide-react';
import { motion } from 'framer-motion';

const Viewer = ({ scan, onBack, onProceedToResults }) => {
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.2, 3));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.2, 0.5));
  const handleRotate = () => setRotation(prev => prev + 90);

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-4">
          <button
            onClick={onBack}
            className="p-2 text-neutral-500 hover:text-primary-500 transition-colors"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          
          <div>
            <h1 className="text-2xl font-bold text-secondary-500">MRI Viewer</h1>
            <p className="text-neutral-500">{scan.filename}</p>
          </div>
        </div>

        <button
          onClick={onProceedToResults}
          className="px-6 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors font-semibold"
        >
          View Results
        </button>
      </div>

      <div className="grid lg:grid-cols-4 gap-6">
        {/* Image Viewer */}
        <div className="lg:col-span-3">
          <div className="bg-white rounded-xl shadow-lg overflow-hidden">
            {/* Controls */}
            <div className="p-4 border-b border-neutral-100 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <button
                  onClick={handleZoomOut}
                  className="p-2 text-neutral-500 hover:text-primary-500 hover:bg-accent-50 rounded-lg transition-all"
                >
                  <ZoomOut className="h-4 w-4" />
                </button>
                
                <span className="text-sm text-neutral-600 font-mono">
                  {Math.round(zoom * 100)}%
                </span>
                
                <button
                  onClick={handleZoomIn}
                  className="p-2 text-neutral-500 hover:text-primary-500 hover:bg-accent-50 rounded-lg transition-all"
                >
                  <ZoomIn className="h-4 w-4" />
                </button>
              </div>

              <div className="flex items-center space-x-2">
                <button
                  onClick={handleRotate}
                  className="p-2 text-neutral-500 hover:text-primary-500 hover:bg-accent-50 rounded-lg transition-all"
                >
                  <RotateCw className="h-4 w-4" />
                </button>
                
                <button className="p-2 text-neutral-500 hover:text-primary-500 hover:bg-accent-50 rounded-lg transition-all">
                  <Download className="h-4 w-4" />
                </button>
              </div>
            </div>

            {/* Image Display */}
            <div className="p-6 bg-neutral-50 min-h-[500px] flex items-center justify-center overflow-hidden">
              <motion.img
                src={scan.imageUrl}
                alt="MRI Scan"
                className="max-w-full max-h-full object-contain border border-neutral-200 rounded-lg shadow-sm"
                style={{
                  transform: `scale(${zoom}) rotate(${rotation}deg)`,
                  transition: 'transform 0.3s ease-in-out'
                }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              />
            </div>
          </div>
        </div>

        {/* Metadata Panel */}
        <div className="space-y-6">
          {/* File Information */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold text-secondary-500 mb-4">File Information</h3>
            
            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium text-neutral-600">Filename</label>
                <p className="text-secondary-500 font-mono text-sm break-all">{scan.filename}</p>
              </div>
              
              <div>
                <label className="text-sm font-medium text-neutral-600">Upload Date</label>
                <p className="text-secondary-500">{new Date(scan.uploadDate).toLocaleDateString()}</p>
              </div>
              
              <div>
                <label className="text-sm font-medium text-neutral-600">File Size</label>
                <p className="text-secondary-500">{scan.metadata.size}</p>
              </div>
              
              <div>
                <label className="text-sm font-medium text-neutral-600">Format</label>
                <p className="text-secondary-500">{scan.metadata.format}</p>
              </div>
              
              <div>
                <label className="text-sm font-medium text-neutral-600">Dimensions</label>
                <p className="text-secondary-500">{scan.metadata.dimensions}</p>
              </div>
            </div>
          </div>

          {/* Processing Status */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold text-secondary-500 mb-4">Processing Status</h3>
            
            <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
              <CheckCircle className="h-5 w-5 text-medical-success" />
              <div>
                <p className="font-semibold text-medical-success">Analysis Complete</p>
                <p className="text-sm text-neutral-600">AI model has processed the scan</p>
              </div>
            </div>

            <button
              onClick={onProceedToResults}
              className="w-full mt-4 px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors font-semibold"
            >
              View Analysis Results
            </button>
          </div>

          {/* Quick Actions */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold text-secondary-500 mb-4">Quick Actions</h3>
            
            <div className="space-y-2">
              <button className="w-full text-left p-3 text-primary-600 hover:bg-accent-50 rounded-lg transition-colors">
                Download Original
              </button>
              <button className="w-full text-left p-3 text-primary-600 hover:bg-accent-50 rounded-lg transition-colors">
                Share with Doctor
              </button>
              <button className="w-full text-left p-3 text-primary-600 hover:bg-accent-50 rounded-lg transition-colors">
                Add to Report
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Viewer;