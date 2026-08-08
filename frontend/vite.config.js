import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  // Use the automatic JSX runtime everywhere, so files don't need `import React`.
  // The app's own .jsx already gets this from @vitejs/plugin-react; stating it
  // here extends the same behaviour to the Vitest transform, which otherwise
  // falls back to the classic runtime and fails every render() with
  // "React is not defined".
  esbuild: { jsx: 'automatic' },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      '/papers': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
  test: {
    // jsdom so component tests can render; node would do for the pure-logic
    // files but splitting the config in two is not worth it at this size.
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/setupTests.js',
    include: ['src/**/*.test.{js,jsx}'],
  },
})