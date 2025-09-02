import React from 'react';
import { Brain, Upload, Shield, Zap, Users, Award } from 'lucide-react';
import { motion } from 'framer-motion';

const Home = ({ onNavigate }) => {
  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Analysis',
      description: 'Advanced deep learning models trained on thousands of MRI scans'
    },
    {
      icon: Zap,
      title: 'Fast Results',
      description: 'Get analysis results in seconds, not hours'
    },
    {
      icon: Shield,
      title: 'Medical Grade Security',
      description: 'HIPAA compliant with enterprise-level data protection'
    },
    {
      icon: Award,
      title: '96.8% Accuracy',
      description: 'Clinically validated with leading medical institutions'
    }
  ];

  const stats = [
    { label: 'Scans Analyzed', value: '50,000+' },
    { label: 'Medical Centers', value: '150+' },
    { label: 'Countries', value: '25' },
    { label: 'Accuracy Rate', value: '96.8%' }
  ];

  return (
    <div>
      {/* Hero Section */}
      <div className="bg-gradient-to-br from-primary-500 via-primary-600 to-secondary-500 text-white">
        <div className="max-w-7xl mx-auto px-6 py-20">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center"
          >
            <div className="mb-6">
              <div className="inline-flex p-4 bg-white/10 rounded-full mb-4">
                <Brain className="h-12 w-12" />
              </div>
              <h1 className="text-5xl font-bold mb-4">
                AI Brain Tumor Detection
              </h1>
              <p className="text-xl text-primary-100 mb-8 max-w-3xl mx-auto">
                Advanced artificial intelligence for rapid, accurate brain tumor detection 
                from MRI scans. Trusted by medical professionals worldwide.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={() => onNavigate('upload')}
                className="px-8 py-3 bg-white text-primary-600 rounded-lg font-semibold hover:bg-accent-50 transition-all transform hover:-translate-y-0.5 shadow-lg"
              >
                Upload MRI Scan
              </button>
              <button
                onClick={() => onNavigate('dashboard')}
                className="px-8 py-3 border-2 border-white text-white rounded-lg font-semibold hover:bg-white hover:text-primary-600 transition-all"
              >
                View Dashboard
              </button>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="bg-white py-16">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * index }}
                className="text-center"
              >
                <p className="text-3xl font-bold text-primary-500 mb-2">{stat.value}</p>
                <p className="text-neutral-600">{stat.label}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="bg-background py-20">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl font-bold text-secondary-500 mb-4">
              Why Choose Our AI Analysis?
            </h2>
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto">
              Our advanced AI system provides fast, accurate, and reliable brain tumor detection 
              to support medical professionals in diagnosis and treatment planning.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * index }}
                className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow"
              >
                <div className="p-3 bg-primary-100 rounded-full w-fit mb-4">
                  <feature.icon className="h-6 w-6 text-primary-500" />
                </div>
                <h3 className="text-lg font-semibold text-secondary-500 mb-2">
                  {feature.title}
                </h3>
                <p className="text-neutral-600">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-accent-100 py-16">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Upload className="h-16 w-16 text-primary-500 mx-auto mb-6" />
            <h2 className="text-3xl font-bold text-secondary-500 mb-4">
              Ready to Analyze Your MRI?
            </h2>
            <p className="text-xl text-neutral-600 mb-8">
              Upload your brain MRI scan and get AI-powered analysis results in seconds.
            </p>
            <button
              onClick={() => onNavigate('upload')}
              className="px-8 py-3 bg-primary-500 text-white rounded-lg font-semibold hover:bg-primary-600 transition-all transform hover:-translate-y-0.5 shadow-lg"
            >
              Start Analysis Now
            </button>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Home;