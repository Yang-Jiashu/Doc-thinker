import { defineConfig } from 'vite';
import uniModule from '@dcloudio/vite-plugin-uni';
const uni = uniModule.default || uniModule;
export default defineConfig({ plugins: [uni()], server: { host: '127.0.0.1', port: 5174, strictPort: true } });
