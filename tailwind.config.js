/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        editorial: {
          primary: '#0f172a',
          muted: 'hsl(215, 25%, 32%)',
          btnBg: '#000000',
          btnText: '#ffffff',
          fallback: 'hsl(201, 100%, 13%)',
          paper: '#fafaf9',
          offwhite: '#f5f5f4',
          subtleBorder: 'rgba(15, 23, 42, 0.08)',
          cardBg: '#fbfbfb',
        }
      },
      fontFamily: {
        serif: ['"Instrument Serif"', 'Georgia', 'serif'],
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      letterSpacing: {
        'hero': '-2.46px',
        'tight-editorial': '-0.02em',
        'wide-editorial': '0.12em',
      },
      keyframes: {
        'fade-rise': {
          '0%': { opacity: '0', transform: 'translateY(24px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'fade-rise': 'fade-rise 0.8s ease-out forwards',
        'fade-rise-delayed': 'fade-rise 0.8s ease-out 0.2s forwards',
        'fade-rise-delayed-2': 'fade-rise 0.8s ease-out 0.4s forwards',
      },
    },
  },
  plugins: [],
}
