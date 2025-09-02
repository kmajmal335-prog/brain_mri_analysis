
// Real API service to communicate with FastAPI backend
class APIService {
  constructor() {
    this.baseURL = 'http://localhost:12000/api';
  }

  async uploadMRI(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${this.baseURL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Upload error:', error);
      throw error;
    }
  }

  async getScans() {
    try {
      const response = await fetch(`${this.baseURL}/scans`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get scans error:', error);
      throw error;
    }
  }

  async getScanById(scanId) {
    try {
      const response = await fetch(`${this.baseURL}/scans/${scanId}` , {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get scan by ID error:', error);
      throw error;
    }
  }

  async getAnalytics() {
    try {
      const response = await fetch(`${this.baseURL}/analytics`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Get analytics error:', error);
      throw error;
    }
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
