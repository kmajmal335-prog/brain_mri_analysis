import React, { useState, useCallback } from 'react';
import { Upload, FileImage, AlertCircle, CheckCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import { apiService } from '../utils/apiService';

const UploadPage = ({ onUploadComplete }) => {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (selectedFile) => {
    setError('');
    
    // Validate file type
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please select a valid image file');
      return;
    }

    // Validate file size (max 10MB)
    if (selectedFile.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }

    setFile(selectedFile);
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError('');

    try {
      // Real API call
      const result = await apiService.uploadMRI(file);
      onUploadComplete(result);
    } catch (err) {
      setError('Upload failed. Please try again.');
      console.error('Upload error:', err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-secondary-500 mb-2">
          Upload MRI Scan
        </h1>
        <p className="text-neutral-500">
          Upload your brain MRI scan for AI-powered tumor detection analysis
        </p>
      </div>

      {/* Upload Area */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div
          className={`relative border-2 border-dashed rounded-xl p-8 transition-all duration-200 ${
            dragActive
              ? 'border-primary-500 bg-primary-50'
              : file
              ? 'border-medical-success bg-green-50'
              : 'border-neutral-200 hover:border-primary-300'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept="image/*"
            onChange={handleChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          
          <div className="text-center">
            {file ? (
              <CheckCircle className="h-12 w-12 text-medical-success mx-auto mb-4" />
            ) : (
              <Upload className="h-12 w-12 text-neutral-400 mx-auto mb-4" />
            )}
            
            <h3 className="text-lg font-semibold text-secondary-500 mb-2">
              {file ? 'File Selected' : 'Drop your MRI scan here'}
            </h3>
            
            <p className="text-neutral-500 mb-4">
              {file ? file.name : 'or click to browse your files'}
            </p>
            
            <div className="text-sm text-neutral-400">
              <p>Supported formats: JPEG, PNG, DICOM</p>
              <p>Maximum file size: 10MB</p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Error Message */}
      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center space-x-2"
        >
          <AlertCircle className="h-5 w-5 text-medical-error" />
          <span className="text-medical-error">{error}</span>
        </motion.div>
      )}

      {/* Preview */}
      {preview && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="mb-6 bg-white rounded-xl shadow-lg p-6"
        >
          <h3 className="text-lg font-semibold text-secondary-500 mb-4 flex items-center">
            <FileImage className="h-5 w-5 mr-2" />
            Preview
          </h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <img
                src={preview}
                alt="MRI Preview"
                className="w-full h-64 object-contain bg-neutral-50 rounded-lg border"
              />
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium text-neutral-600">Filename</label>
                <p className="text-secondary-500 font-mono text-sm">{file?.name}</p>
              </div>
              
              <div>
                <label className="text-sm font-medium text-neutral-600">File Size</label>
                <p className="text-secondary-500">{(file?.size / 1024).toFixed(1)} KB</p>
              </div>
              
              <div>
                <label className="text-sm font-medium text-neutral-600">Type</label>
                <p className="text-secondary-500">{file?.type}</p>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Upload Button */}
      {file && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center"
        >
          <button
            onClick={handleUpload}
            disabled={uploading}
            className={`px-8 py-3 rounded-lg font-semibold transition-all duration-200 ${
              uploading
                ? 'bg-neutral-300 text-neutral-500 cursor-not-allowed'
                : 'bg-primary-500 hover:bg-primary-600 text-white shadow-lg hover:shadow-xl transform hover:-translate-y-0.5'
            }`}
          >
            {uploading ? (
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                <span>Analyzing MRI...</span>
              </div>
            ) : (
              'Start AI Analysis'
            )}
          </button>
        </motion.div>
      )}
    </div>
  );
};

export default UploadPage;