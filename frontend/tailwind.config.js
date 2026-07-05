/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#0b0d14',
        panel: '#11141d',
        line: '#1e2230',
        brand: '#7c8cff',
        cs: '#3ec6ff',
        math: '#c58bff',
        food: '#ffb15c',
        sport: '#5fe6a5',
        doc: '#5fe6a5',
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
      },
    },
  },
  plugins: [],
}
