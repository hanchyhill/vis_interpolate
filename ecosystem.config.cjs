const path = require('node:path');

const root = __dirname;
const python = process.platform === 'win32'
  ? path.join(root, '.venv', 'Scripts', 'python.exe')
  : path.join(root, '.venv', 'bin', 'python');

module.exports = {
  apps: [
    {
      name: 'vis-interpolate-business',
      cwd: root,
      script: python,
      args: '-m src.business serve',
      interpreter: 'none',
      autorestart: true,
      watch: false,
      restart_delay: 10000,
      max_restarts: 20,
      min_uptime: 10000,
      kill_timeout: 10000,
      time: true,
      merge_logs: true,
      out_file: path.join(root, 'output', 'pm2-business-out.log'),
      error_file: path.join(root, 'output', 'pm2-business-error.log'),
      env: {
        PYTHONUTF8: '1',
        VIS_BUSINESS_CONFIG: process.env.VIS_BUSINESS_CONFIG
          || path.join(root, 'src', 'config', 'local.config.json'),
      },
    },
  ],
};
