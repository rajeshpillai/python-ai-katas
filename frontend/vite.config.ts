import { defineConfig } from "vite";
import solidPlugin from "vite-plugin-solid";

export default defineConfig({
  plugins: [solidPlugin()],
  server: {
    port: 3000,
    proxy: {
      "/api/python": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace("/api/python", "/api"),
      },
      "/api/rust": {
        target: "http://localhost:8001",
        changeOrigin: true,
        rewrite: (path) => path.replace("/api/rust", "/api"),
      },
    },
  },
  build: {
    target: "esnext",
  },
});
