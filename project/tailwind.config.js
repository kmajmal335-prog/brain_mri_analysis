/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#E6F3FF',
          100: '#B3DBFF',
          500: '#0077B6',
          600: '#005A8A',
          700: '#003D5C',
        },
        secondary: {
          50: '#F0F2F5',
          100: '#D4D8DD',
          500: '#1B263B',
          600: '#151D2A',
          700: '#0F1419',
        },
        accent: {
          50: '#F0FBFD',
          100: '#ADE8F4',
          500: '#7BC7DB',
          600: '#4AA5C2',
          700: '#2E7D9A',
        },
        neutral: {
          50: '#F9F9F9',
          100: '#E0E0E0',
          200: '#C7C7C7',
          300: '#AEAEAE',
          500: '#808080',
          700: '#4A4A4A',
        },
        medical: {
          success: '#2ECC71',
          error: '#D62828',
          warning: '#F39C12',
          info: '#3498DB',
        },
        background: '#F8FAFC',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-subtle': 'pulseSubtle 2s infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        pulseSubtle: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.8' },
        },
      },
    },
  },
  plugins: [],
};