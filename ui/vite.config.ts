import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// When running in Docker, API_TARGET=http://api:8000 is injected by docker-compose.
// Outside Docker, falls back to localhost.
const API_TARGET = process.env.API_TARGET ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/health': { target: API_TARGET, changeOrigin: true },
      '/api': { target: API_TARGET, changeOrigin: true },
    },
  },
})
