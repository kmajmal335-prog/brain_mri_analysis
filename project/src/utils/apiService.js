
// Real API service to communicate with FastAPI backend
class APIService {
  constructor() {
    this.baseURL = 'http://localhost:8000';
    this.scans = []; // Local storage for scans
  }

  // Health check endpoint
  async healthCheck() {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }

  // Load model endpoint
  async loadModel(modelPath) {
    try {
      const response = await fetch(`${this.baseURL}/load-model?model_path=${encodeURIComponent(modelPath)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Load model error:', error);
      throw error;
    }
  }

  // Get model info
  async getModelInfo() {
    try {
      const response = await fetch(`${this.baseURL}/model-info`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Get model info error:', error);
      throw error;
    }
  }

  // Get available classes
  async getClasses() {
    try {
      const response = await fetch(`${this.baseURL}/classes`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Get classes error:', error);
      throw error;
    }
  }

  // Upload and analyze MRI scan
  async uploadMRI(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${this.baseURL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      // Create scan object for local storage
      const scan = {
        id: String(this.scans.length + 1),
        filename: result.file_info.filename,
        fileSize: result.file_info.size,
        uploadDate: result.timestamp,
        imageUrl: URL.createObjectURL(file),
        prediction: result.predicted_class,
        confidence: result.confidence,
        probabilities: result.probabilities,
        processed: true,
        metadata: {
          dimensions: '224x224', // Model input size
          format: result.file_info.content_type.split('/')[1].toUpperCase(),
          size: `${Math.round(result.file_info.size / 1024)} KB`
        }
      };

      this.scans.push(scan);
      return scan;

    } catch (error) {
      console.error('Upload error:', error);
      throw error;
    }
  }

  // Batch upload multiple files
  async uploadMultipleMRI(files) {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch(`${this.baseURL}/batch-predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      // Process results and add to local storage
      const newScans = result.results.map((res, index) => {
        if (res.error) {
          return { error: res.error, filename: res.filename };
        }
        
        const scan = {
          id: String(this.scans.length + index + 1),
          filename: res.file_info.filename,
          fileSize: res.file_info.size,
          uploadDate: res.timestamp,
          imageUrl: URL.createObjectURL(files[index]),
          prediction: res.predicted_class,
          confidence: res.confidence,
          probabilities: res.probabilities,
          processed: true,
          metadata: {
            dimensions: '224x224',
            format: res.file_info.content_type.split('/')[1].toUpperCase(),
            size: `${Math.round(res.file_info.size / 1024)} KB`
          }
        };
        
        this.scans.push(scan);
        return scan;
      });

      return {
        results: newScans,
        total_files: result.total_files,
        successful_predictions: result.successful_predictions
      };

    } catch (error) {
      console.error('Batch upload error:', error);
      throw error;
    }
  }

  // Get all scans (from local storage)
  async getScans() {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve(this.scans);
      }, 100);
    });
  }

  // Get scan by ID (from local storage)
  async getScanById(scanId) {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        const scan = this.scans.find(s => s.id === scanId);
        if (scan) {
          resolve(scan);
        } else {
          reject(new Error('Scan not found'));
        }
      }, 100);
    });
  }

  // Get analytics based on local scans
  async getAnalytics() {
    return new Promise((resolve) => {
      setTimeout(() => {
        const total = this.scans.length;
        const tumorScans = this.scans.filter(s => s.prediction !== 'notumor').length;
        const normalScans = this.scans.filter(s => s.prediction === 'notumor').length;
        
        // Calculate class distribution
        const classDistribution = {};
        this.scans.forEach(scan => {
          classDistribution[scan.prediction] = (classDistribution[scan.prediction] || 0) + 1;
        });
        
        // Calculate average confidence
        const avgConfidence = total > 0 
          ? this.scans.reduce((sum, scan) => sum + scan.confidence, 0) / total 
          : 0;
        
        resolve({
          totalScans: total,
          tumorDetected: tumorScans,
          normalScans: normalScans,
          classDistribution: classDistribution,
          averageConfidence: avgConfidence,
          accuracy: Math.round(avgConfidence * 100) // Approximate accuracy based on confidence
        });
      }, 100);
    });
  }

  // Analyze existing scan (re-run prediction)
  async analyzeScan(scanId) {
    return this.getScanById(scanId);
  }
}

export const apiService = new APIService();

// Mock API service for frontend development without a backend
class MockAPIService {
  constructor() {
    this.scans = [
      {
        id: '1',
        filename: 'mri_scan_1.jpg',
        fileSize: 123456,
        uploadDate: new Date().toISOString(),
        imageUrl: 'https://via.placeholder.com/150',
        prediction: 'Tumor',
        confidence: 0.95,
        processed: true,
        metadata: {
          dimensions: '512x512',
          format: 'JPEG',
          size: '121 KB'
        }
      },
      {
        id: '2',
        filename: 'mri_scan_2.png',
        fileSize: 234567,
        uploadDate: new Date().toISOString(),
        imageUrl: 'https://via.placeholder.com/150',
        prediction: 'No Tumor',
        confidence: 0.99,
        processed: true,
        metadata: {
          dimensions: '512x512',
          format: 'PNG',
          size: '229 KB'
        }
      }
    ];
  }

  async uploadMRI(file) {
    return new Promise((resolve) => {
      setTimeout(() => {
        const newScan = {
          id: String(this.scans.length + 1),
          filename: file.name,
          fileSize: file.size,
          uploadDate: new Date().toISOString(),
          imageUrl: URL.createObjectURL(file),
          prediction: Math.random() > 0.5 ? 'Tumor' : 'No Tumor',
          confidence: Math.random() * (0.99 - 0.90) + 0.90,
          processed: true,
          metadata: {
            dimensions: '512x512',
            format: file.type.split('/')[1].toUpperCase(),
            size: `${Math.round(file.size / 1024)} KB`
          }
        };
        this.scans.push(newScan);
        resolve(newScan);
      }, 1000);
    });
  }

  async analyzeScan(scanId) {
    return this.getScanById(scanId);
  }

  async getScans() {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve(this.scans);
      }, 500);
    });
  }

  async getScanById(scanId) {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        const scan = this.scans.find(s => s.id === scanId);
        if (scan) {
          resolve(scan);
        } else {
          reject(new Error('Scan not found'));
        }
      }, 500);
    });
  }

  async getAnalytics() {
    return new Promise((resolve) => {
      setTimeout(() => {
        const total = this.scans.length;
        const tumor = this.scans.filter(s => s.prediction === 'Tumor').length;
        const normal = total - tumor;
        const accuracy = total > 0 ? Math.round((normal + tumor) / total * 100) : 0;
        
        resolve({
          totalScans: total,
          tumorDetected: tumor,
          normalScans: normal,
          accuracy: accuracy
        });
      }, 500);
    });
  }
}

export const mockApiService = new MockAPIService();
