// ================================================================================
// AgriCrop — Express.js Backend
// REST API for crop recommendation using Python ML inference
// ================================================================================

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const { execFile } = require('child_process');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 5000;
const upload = multer({ storage: multer.memoryStorage() });

// Paths
const ML_DIR = path.join(__dirname, '..', 'ml');
const PREDICT_SCRIPT = path.join(ML_DIR, 'predict.py');
const METRICS_FILE = path.join(ML_DIR, 'artifacts', 'metrics.json');
const CLIENT_BUILD = path.join(__dirname, '..', 'client', 'dist');

// Middleware
app.use(cors());
app.use(express.json());

// Serve React build in production
if (fs.existsSync(CLIENT_BUILD)) {
  app.use(express.static(CLIENT_BUILD));
}

// ── Helper: Call Python predict script ────────────────────────────────────────
function callPython(inputData) {
  return new Promise((resolve, reject) => {
    const { spawn } = require('child_process');
    const py = spawn('python', [PREDICT_SCRIPT], {
      cwd: ML_DIR,
      timeout: 30000,
      shell: true,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    });

    let stdout = '';
    let stderr = '';

    py.stdout.on('data', (data) => { stdout += data.toString(); });
    py.stderr.on('data', (data) => { stderr += data.toString(); });

    py.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`Python exited with code ${code}: ${stderr}`));
      }
      try {
        const result = JSON.parse(stdout);
        if (result.error) {
          return reject(new Error(result.error));
        }
        resolve(result);
      } catch (parseErr) {
        reject(new Error(`Failed to parse Python output: ${stdout} | stderr: ${stderr}`));
      }
    });

    py.on('error', (err) => {
      reject(new Error(`Failed to start Python: ${err.message}`));
    });

    // Write input to stdin
    py.stdin.write(JSON.stringify(inputData));
    py.stdin.end();
  });
}

// ── ROUTES ───────────────────────────────────────────────────────────────────

// POST /api/predict — Single prediction
app.post('/api/predict', async (req, res) => {
  try {
    const data = req.body;
    const required = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'];
    const missing = required.filter(f => data[f] === undefined || data[f] === '');

    if (missing.length > 0) {
      return res.status(400).json({ error: `Missing fields: ${missing.join(', ')}` });
    }

    const result = await callPython(data);
    res.json(result);
  } catch (err) {
    console.error('Prediction error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/predict-csv — Batch prediction from CSV upload
app.post('/api/predict-csv', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const content = req.file.buffer.toString('utf-8');
    const lines = content.trim().split('\n');

    if (lines.length < 2) {
      return res.status(400).json({ error: 'CSV file is empty or has no data rows' });
    }

    // Parse CSV manually (simple, no external dep needed for this)
    const headers = lines[0].split(',').map(h => h.trim().toLowerCase());

    const colMap = {
      'n': 'N', 'p': 'P', 'k': 'K',
      'temperature': 'temperature', 'temp': 'temperature',
      'humidity': 'humidity', 'hum': 'humidity',
      'ph': 'ph',
      'rainfall': 'rainfall', 'rain': 'rainfall',
    };

    const mappedHeaders = headers.map(h => colMap[h] || h);
    const required = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'];
    const missing = required.filter(f => !mappedHeaders.includes(f));

    if (missing.length > 0) {
      return res.status(400).json({
        error: `CSV missing columns: ${missing.join(', ')}. Required: ${required.join(', ')}`
      });
    }

    // Parse rows into array of objects
    const dataRows = [];
    for (let i = 1; i < Math.min(lines.length, 101); i++) {
      const vals = lines[i].split(',').map(v => v.trim());
      if (vals.length < headers.length) continue;
      const row = {};
      mappedHeaders.forEach((h, idx) => {
        row[h] = parseFloat(vals[idx]);
      });
      if (!required.some(f => isNaN(row[f]))) {
        dataRows.push(row);
      }
    }

    if (dataRows.length === 0) {
      return res.status(400).json({ error: 'No valid data rows found in CSV' });
    }

    // Send batch to Python
    const results = await callPython(dataRows);
    res.json({ results: Array.isArray(results) ? results : [results], count: dataRows.length });
  } catch (err) {
    console.error('CSV prediction error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/model-info — Model metrics and crop list
app.get('/api/model-info', (req, res) => {
  try {
    if (!fs.existsSync(METRICS_FILE)) {
      return res.status(503).json({ error: 'Model not trained yet. Run: cd ml && python train_model.py' });
    }
    const metrics = JSON.parse(fs.readFileSync(METRICS_FILE, 'utf-8'));
    res.json(metrics);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Catch-all: serve React app for client-side routing
if (fs.existsSync(CLIENT_BUILD)) {
  app.get('*', (req, res) => {
    res.sendFile(path.join(CLIENT_BUILD, 'index.html'));
  });
}

// ── START SERVER ─────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🌾 AgriCrop API server running at http://localhost:${PORT}`);
  console.log(`   POST /api/predict      — single prediction`);
  console.log(`   POST /api/predict-csv  — batch CSV prediction`);
  console.log(`   GET  /api/model-info   — model metrics\n`);
});
