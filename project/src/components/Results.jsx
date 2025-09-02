import React, { useState } from 'react';
import { AlertTriangle, CheckCircle, Brain, Calendar, FileText, ArrowLeft, ZoomIn, ZoomOut, RotateCw } from 'lucide-react';
import { motion } from 'framer-motion';

const Results = ({ scan, onBack, onViewHistory }) => {
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.2, 3));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.2, 0.5));
  const handleRotate = () => setRotation(prev => prev + 90);

  const isTumor = scan.prediction === 'Tumor';
  const confidencePercentage = Math.round(scan.confidence * 100);

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center space-x-4">
          <button
            onClick={onBack}
            className="p-2 text-neutral-500 hover:text-primary-500 transition-colors"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          
          <div>
            <h1 className="text-3xl font-bold text-secondary-500">Analysis Results</h1>
            <p className="text-neutral-500">AI-powered brain tumor detection analysis</p>
          </div>
        </div>

        <button
          onClick={onViewHistory}
          className="px-4 py-2 text-primary-600 border border-primary-200 rounded-lg hover:bg-primary-50 transition-colors"
        >
          View All Results
        </button>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Main Result */}
        <div className="lg:col-span-2 space-y-6">
          {/* Primary Result Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`bg-white rounded-xl shadow-lg p-6 border-l-4 ${
              isTumor ? 'border-medical-error' : 'border-medical-success'
            }`}
          >
            <div className="flex items-center space-x-4 mb-4">
              {isTumor ? (
                <div className="p-3 bg-red-100 rounded-full">
                  <AlertTriangle className="h-6 w-6 text-medical-error" />
                </div>
              ) : (
                <div className="p-3 bg-green-100 rounded-full">
                  <CheckCircle className="h-6 w-6 text-medical-success" />
                </div>
              )}
              
              <div>
                <h2 className="text-2xl font-bold text-secondary-500">
                  {scan.prediction}
                  {isTumor ? ' Detected' : ' Scan'}
                </h2>
                <p className="text-neutral-500">
                  Confidence: {confidencePercentage}%
                </p>
              </div>
            </div>

            {/* Confidence Bar */}
            <div className="mb-4">
              <div className="flex justify-between text-sm text-neutral-600 mb-2">
                <span>Confidence Level</span>
                <span>{confidencePercentage}%</span>
              </div>
              <div className="w-full bg-neutral-100 rounded-full h-3">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${confidencePercentage}%` }}
                  transition={{ duration: 1, delay: 0.5 }}
                  className={`h-3 rounded-full ${
                    isTumor ? 'bg-medical-error' : 'bg-medical-success'
                  }`}
                ></motion.div>
              </div>
            </div>

            {/* Interpretation */}
            <div className={`p-4 rounded-lg ${
              isTumor ? 'bg-red-50 border border-red-200' : 'bg-green-50 border border-green-200'
            }`}>
              <h4 className="font-semibold text-secondary-500 mb-2">Clinical Interpretation</h4>
              <p className={`text-sm ${
                isTumor ? 'text-red-700' : 'text-green-700'
              }`}>
                {isTumor 
                  ? 'The AI analysis has detected potential abnormal tissue that may indicate the presence of a brain tumor. Please consult with a radiologist for professional diagnosis.'
                  : 'The AI analysis shows no significant abnormalities detected in this brain MRI scan. Continue regular monitoring as recommended by your healthcare provider.'
                }
              </p>
            </div>
          </motion.div>

          {/* Technical Details */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white rounded-xl shadow-lg p-6"
          >
            <h3 className="text-lg font-semibold text-secondary-500 mb-4 flex items-center">
              <Brain className="h-5 w-5 mr-2" />
              Technical Analysis
            </h3>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="p-4 bg-neutral-50 rounded-lg">
                <h4 className="font-semibold text-secondary-500 mb-2">Model Information</h4>
                <div className="space-y-1 text-sm">
                  <p><span className="text-neutral-600">Model:</span> ResNet-50 CNN</p>
                  <p><span className="text-neutral-600">Training Data:</span> 15,000+ MRI scans</p>
                  <p><span className="text-neutral-600">Accuracy:</span> 96.8%</p>
                </div>
              </div>

              <div className="p-4 bg-neutral-50 rounded-lg">
                <h4 className="font-semibold text-secondary-500 mb-2">Processing Details</h4>
                <div className="space-y-1 text-sm">
                  <p><span className="text-neutral-600">Processing Time:</span> 2.3 seconds</p>
                  <p><span className="text-neutral-600">Image Quality:</span> High</p>
                  <p><span className="text-neutral-600">Analysis Date:</span> {new Date(scan.uploadDate).toLocaleString()}</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Scan Preview */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white rounded-xl shadow-lg p-6"
          >
            <h3 className="text-lg font-semibold text-secondary-500 mb-4">Current Scan</h3>
            
            <img
              src={scan.imageUrl}
              alt="MRI Scan"
              className="w-full h-48 object-contain bg-neutral-50 rounded-lg border mb-4"
              style={{
                transform: `scale(${zoom}) rotate(${rotation}deg)`,
                transition: 'transform 0.3s ease-in-out'
              }}
            />

            <div className="flex space-x-2 mb-4">
              <button
                onClick={handleZoomOut}
                className="flex-1 p-2 text-neutral-500 hover:text-primary-500 hover:bg-accent-50 rounded-lg transition-all"
              >
                <ZoomOut className="h-4 w-4 mx-auto" />
              </button>
              <button
                onClick={handleZoomIn}
                className="flex-1 p-2 text-neutral-500 hover:text-primary-500 hover:bg-accent-50 rounded-lg transition-all"
              >
                <ZoomIn className="h-4 w-4 mx-auto" />
              </button>
              <button
                onClick={handleRotate}
                className="flex-1 p-2 text-neutral-500 hover:text-primary-500 hover:bg-accent-50 rounded-lg transition-all"
              >
                <RotateCw className="h-4 w-4 mx-auto" />
              </button>
            </div>

            <div className="text-sm space-y-2">
              <p><span className="text-neutral-600">File:</span> {scan.filename}</p>
              <p><span className="text-neutral-600">Size:</span> {scan.metadata.size}</p>
              <p><span className="text-neutral-600">Format:</span> {scan.metadata.format}</p>
            </div>
          </motion.div>

          {/* Next Steps */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-white rounded-xl shadow-lg p-6"
          >
            <h3 className="text-lg font-semibold text-secondary-500 mb-4 flex items-center">
              <FileText className="h-5 w-5 mr-2" />
              Recommended Actions
            </h3>

            <div className="space-y-3">
              {isTumor ? (
                <>
                  <div className="p-3 bg-red-50 rounded-lg">
                    <p className="text-sm text-red-700 font-medium">Urgent: Consult Specialist</p>
                  </div>
                  <div className="p-3 bg-yellow-50 rounded-lg">
                    <p className="text-sm text-yellow-700">Schedule follow-up imaging</p>
                  </div>
                  <div className="p-3 bg-blue-50 rounded-lg">
                    <p className="text-sm text-blue-700">Prepare medical history</p>
                  </div>
                </>
              ) : (
                <>
                  <div className="p-3 bg-green-50 rounded-lg">
                    <p className="text-sm text-green-700 font-medium">Continue routine monitoring</p>
                  </div>
                  <div className="p-3 bg-blue-50 rounded-lg">
                    <p className="text-sm text-blue-700">Schedule next scan as advised</p>
                  </div>
                  <div className="p-3 bg-neutral-50 rounded-lg">
                    <p className="text-sm text-neutral-600">Maintain healthy lifestyle</p>
                  </div>
                </>
              )}
            </div>
          </motion.div>

          {/* Export Options */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
            className="bg-white rounded-xl shadow-lg p-6"
          >
            <h3 className="text-lg font-semibold text-secondary-500 mb-4">Export & Share</h3>
            
            <div className="space-y-2">
              <button className="w-full text-left p-3 text-primary-600 hover:bg-accent-50 rounded-lg transition-colors">
                Download PDF Report
              </button>
              <button className="w-full text-left p-3 text-primary-600 hover:bg-accent-50 rounded-lg transition-colors">
                Share with Doctor
              </button>
              <button className="w-full text-left p-3 text-primary-600 hover:bg-accent-50 rounded-lg transition-colors">
                Export DICOM
              </button>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Results;