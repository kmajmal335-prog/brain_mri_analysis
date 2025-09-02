// Mock API to simulate backend functionality
class MockAPI {
  constructor() {
    this.scans = this.loadScansFromStorage();
  }

  loadScansFromStorage() {
    const stored = localStorage.getItem('mri_scans');
    return stored ? JSON.parse(stored) : [];
  }

  saveScansToStorage() {
    localStorage.setItem('mri_scans', JSON.stringify(this.scans));
  }

  async uploadMRI(file) {
    // Simulate upload delay
    await new Promise(resolve => setTimeout(resolve, 2000));

    const fileUrl = URL.createObjectURL(file);
    const scanId = Math.random().toString(36).substr(2, 9);
    
    // Mock AI analysis
    const predictions = ['Normal', 'Tumor'];
    const prediction = predictions[Math.floor(Math.random() * predictions.length)];
    const confidence = (Math.random() * 0.3 + 0.7).toFixed(3); // 0.7 - 1.0

    const scan = {
      id: scanId,
      filename: file.name,
      fileSize: file.size,
      uploadDate: new Date().toISOString(),
      imageUrl: fileUrl,
      prediction,
      confidence: parseFloat(confidence),
      processed: true,
      metadata: {
        dimensions: '256x256',
        format: file.type,
        size: `${(file.size / 1024).toFixed(1)} KB`
      }
    };

    this.scans.unshift(scan);
    this.saveScansToStorage();

    return scan;
  }

  async getScans() {
    await new Promise(resolve => setTimeout(resolve, 500));
    return this.scans;
  }

  async getScanById(id) {
    await new Promise(resolve => setTimeout(resolve, 300));
    return this.scans.find(scan => scan.id === id);
  }

  getAnalytics() {
    const totalScans = this.scans.length;
    const tumorDetected = this.scans.filter(scan => scan.prediction === 'Tumor').length;
    const normalScans = totalScans - tumorDetected;
    
    return {
      totalScans,
      tumorDetected,
      normalScans,
      accuracy: totalScans > 0 ? ((normalScans + tumorDetected) / totalScans * 100).toFixed(1) : 0
    };
  }
}

export const mockApi = new MockAPI();