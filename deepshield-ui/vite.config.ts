import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // forward /analyze to your backend
      '/analyze': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // also proxy static heatmaps if you reference /static/heatmaps/*
      '/static': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
