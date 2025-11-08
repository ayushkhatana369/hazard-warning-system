/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  safelist: [
    'bg-blue-500', 'bg-blue-600',
    'bg-green-600', 'bg-green-700',
    'bg-red-600', 'bg-red-700',
    'shadow-red-500/40', 'shadow-green-500/40', 'shadow-blue-500/30'
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
