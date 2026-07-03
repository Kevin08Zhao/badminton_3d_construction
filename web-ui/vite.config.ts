import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
  // `npm run preview` also proxies /api so the built UI can talk to a local backend without CORS setup.
  preview: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
