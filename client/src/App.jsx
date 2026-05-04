import { useState, useEffect } from 'react';

const API = '/api';

// ── Presets ─────────────────────────────────────────────────────────────────────
const PRESETS = {
  rice:     { N: 90, P: 42, K: 43, temperature: 20.88, humidity: 82.0, ph: 6.5, rainfall: 202.94, label: '🍚 Rice Field' },
  banana:   { N: 100, P: 82, K: 50, temperature: 27.0, humidity: 80.0, ph: 6.0, rainfall: 150.0, label: '🍌 Banana Farm' },
  chickpea: { N: 40, P: 68, K: 80, temperature: 18.5, humidity: 17.0, ph: 7.2, rainfall: 80.0, label: '🫘 Chickpea Plot' },
  deficient:{ N: 10, P: 7, K: 55, temperature: 28.5, humidity: 32.0, ph: 4.6, rainfall: 35.0, label: '⚠️ Deficient Soil' },
};

const ADVISORY_META = {
  nitrogen:   { icon: '🧪', title: 'Nitrogen (N)', unit: 'kg/ha' },
  phosphorus: { icon: '🧫', title: 'Phosphorus (P)', unit: 'kg/ha' },
  potassium:  { icon: '⚗️', title: 'Potassium (K)', unit: 'kg/ha' },
  ph:         { icon: '📊', title: 'Soil pH', unit: 'pH' },
  irrigation: { icon: '💧', title: 'Irrigation', unit: '' },
};

function App() {
  const [form, setForm] = useState({ N: '', P: '', K: '', temperature: '', humidity: '', ph: '', rainfall: '' });
  const [result, setResult] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [csvLoading, setCsvLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [activeTab, setActiveTab] = useState('manual');
  const [error, setError] = useState('');
  const [expandedRow, setExpandedRow] = useState(null);

  // Fetch model info on load
  useEffect(() => {
    fetch(`${API}/model-info`)
      .then(r => r.json())
      .then(data => setModelInfo(data))
      .catch(() => {});
  }, []);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const fillPreset = (key) => {
    const { label, ...vals } = PRESETS[key];
    setForm(vals);
    setError('');
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);
    setBatchResults(null);

    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
      setTimeout(() => document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' }), 100);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCsvUpload = async (file) => {
    if (!file) return;
    setCsvLoading(true);
    setError('');
    setResult(null);
    setBatchResults(null);
    setExpandedRow(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API}/predict-csv`, { method: 'POST', body: formData });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setBatchResults(data);
      setTimeout(() => document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' }), 100);
    } catch (err) {
      setError(err.message);
    } finally {
      setCsvLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) handleCsvUpload(file);
    else setError('Please upload a .csv file');
  };

  const handleDragOver = (e) => { e.preventDefault(); e.currentTarget.classList.add('drag-over'); };
  const handleDragLeave = (e) => { e.currentTarget.classList.remove('drag-over'); };

  // Reusable advisory report component — same format for single + batch
  const AdvisoryReport = ({ data }) => (
    <div className="advisory-report">
      {/* Report Header — crop + health */}
      <div className="report-header">
        <div className="report-crop">
          <span className="report-crop-icon">{data.cropEmoji}</span>
          <div>
            <span className="report-label">Recommended Crop</span>
            <h3 className="report-crop-name">{data.crop}</h3>
          </div>
        </div>
        <div className={`health-badge health-${data.healthLevel}`}>
          {data.healthIcon} {data.health}
        </div>
      </div>

      {/* Input params */}
      <div className="report-params">
        {Object.entries(data.input).map(([key, val]) => (
          <div className="param-chip" key={key}>
            <span className="param-key">{key}</span>
            <span className="param-val">{val}</span>
          </div>
        ))}
      </div>

      {/* Terminal-style advisory box — matches notebook output */}
      <div className="report-terminal">
        <div className="terminal-titlebar">
          <span className="terminal-dot red"></span>
          <span className="terminal-dot yellow"></span>
          <span className="terminal-dot green"></span>
          <span className="terminal-title">AgriCrop Advisory Report — ICCSC 2026</span>
        </div>
        <div className="terminal-body">
          <div className="terminal-line"><span className="t-key">Recommended Crop</span><span className="t-sep">:</span><span className="t-val t-crop">{data.crop}</span></div>
          <div className="terminal-line"><span className="t-key">Crop Health Status</span><span className="t-sep">:</span><span className={`t-val t-health-${data.healthLevel}`}>{data.health} {data.healthIcon}</span></div>
          <div className="terminal-divider"></div>
          {Object.entries(data.advisories).map(([key, adv]) => (
            <div className="terminal-advisory" key={key}>
              <div className="terminal-line">
                <span className="t-key">{ADVISORY_META[key]?.icon} {ADVISORY_META[key]?.title}</span>
                <span className="t-sep">:</span>
                <span className={`t-val t-status-${adv.level}`}>{adv.status}</span>
              </div>
              <div className="terminal-advice">{adv.advice}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Advisory cards grid */}
      <div className="advisory-grid">
        {Object.entries(data.advisories).map(([key, adv]) => (
          <div className={`advisory-card advisory-${adv.level}`} key={key}>
            <div className="advisory-header">
              <span className="advisory-icon">{ADVISORY_META[key]?.icon}</span>
              <span className="advisory-title">{ADVISORY_META[key]?.title}</span>
              <span className={`advisory-badge badge-${adv.level}`}>{adv.status}</span>
            </div>
            <p className="advisory-text">{adv.advice}</p>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="app">
      {/* Nav */}
      <nav className="navbar">
        <div className="nav-brand"><span className="nav-icon">🌾</span><span className="nav-title">AgriCrop</span></div>
        <div className="nav-links">
          <a href="#predict" className="nav-link">Predict</a>
          <a href="#results" className="nav-link">Results</a>
          <a href="#dashboard" className="nav-link">Dashboard</a>
        </div>
      </nav>

      {/* Hero */}
      <header className="hero">
        <div className="hero-content">
          <div className="hero-badge">ICCSC 2026 • AKGEC Ghaziabad</div>
          <h1 className="hero-title"><span className="hero-emoji">🌾</span> AgriCrop</h1>
          <p className="hero-subtitle">Health Monitoring & Crop Recommendation System</p>
          <p className="hero-desc">AI-powered crop recommendations with actionable soil health advisories — powered by XGBoost with 98.6% accuracy.</p>
          <div className="hero-stats">
            <div className="stat-pill"><span className="stat-number">12</span><span className="stat-label">Crops</span></div>
            <div className="stat-pill"><span className="stat-number">98.6%</span><span className="stat-label">Accuracy</span></div>
            <div className="stat-pill"><span className="stat-number">4</span><span className="stat-label">Models</span></div>
          </div>
          <a href="#predict" className="hero-cta">Get Recommendation →</a>
        </div>
      </header>

      {/* Prediction Section */}
      <section className="section" id="predict">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">🔬 Crop Prediction</h2>
            <p className="section-desc">Enter soil & weather parameters or upload a CSV</p>
          </div>

          <div className="tab-bar">
            <button className={`tab ${activeTab === 'manual' ? 'active' : ''}`} onClick={() => setActiveTab('manual')}>📝 Manual Input</button>
            <button className={`tab ${activeTab === 'csv' ? 'active' : ''}`} onClick={() => setActiveTab('csv')}>📄 CSV Upload</button>
          </div>

          {activeTab === 'manual' && (
            <div className="tab-content fade-in">
              <div className="presets">
                <span className="preset-label">Quick Presets:</span>
                {Object.entries(PRESETS).map(([key, val]) => (
                  <button key={key} className="preset-btn" onClick={() => fillPreset(key)}>{val.label}</button>
                ))}
              </div>

              <form className="predict-form" onSubmit={handlePredict}>
                <div className="form-grid">
                  {[
                    { name: 'N', label: 'Nitrogen (N)', icon: '🧪', placeholder: '0 – 140', hint: 'kg/ha' },
                    { name: 'P', label: 'Phosphorus (P)', icon: '🧫', placeholder: '5 – 145', hint: 'kg/ha' },
                    { name: 'K', label: 'Potassium (K)', icon: '⚗️', placeholder: '5 – 205', hint: 'kg/ha' },
                    { name: 'temperature', label: 'Temperature', icon: '🌡️', placeholder: '8 – 44 °C', hint: '°C' },
                    { name: 'humidity', label: 'Humidity', icon: '💧', placeholder: '14 – 100 %', hint: '%' },
                    { name: 'ph', label: 'Soil pH', icon: '📊', placeholder: '3.5 – 10', hint: 'pH scale' },
                    { name: 'rainfall', label: 'Rainfall', icon: '🌧️', placeholder: '20 – 300 mm', hint: 'mm' },
                  ].map(field => (
                    <div className={`form-group ${field.name === 'rainfall' ? 'full-width' : ''}`} key={field.name}>
                      <label className="form-label"><span className="label-icon">{field.icon}</span> {field.label}</label>
                      <input type="number" name={field.name} className="form-input" placeholder={field.placeholder}
                        value={form[field.name]} onChange={handleChange} step="any" required />
                      <span className="form-hint">{field.hint}</span>
                    </div>
                  ))}
                </div>
                <button type="submit" className="submit-btn" disabled={loading}>
                  {loading ? <><span className="spinner"></span> Analyzing...</> : '🌱 Get Recommendation'}
                </button>
              </form>
            </div>
          )}

          {activeTab === 'csv' && (
            <div className="tab-content fade-in">
              <div className="csv-info">
                <p>Upload a CSV file with columns:</p>
                <code className="csv-cols">N, P, K, temperature, humidity, ph, rainfall</code>
                <p className="csv-note">Each row analyzed independently. Max 100 rows.</p>
              </div>
              <div className={`drop-zone ${csvLoading ? 'loading' : ''}`}
                onDrop={handleDrop} onDragOver={handleDragOver} onDragLeave={handleDragLeave}>
                <span className="drop-icon">📄</span>
                <p className="drop-text">{csvLoading ? 'Processing...' : 'Drag & drop your CSV file here'}</p>
                <p className="drop-or">or</p>
                <label className="file-btn">
                  Browse Files
                  <input type="file" accept=".csv" hidden onChange={(e) => handleCsvUpload(e.target.files[0])} />
                </label>
              </div>
            </div>
          )}

          {error && <div className="error-msg">⚠️ {error}</div>}
        </div>
      </section>

      {/* Results Section */}
      <section className="section section-dark" id="results">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">📋 Results</h2>
            <p className="section-desc">Your crop recommendation and soil health advisory</p>
          </div>

          {/* Single Prediction Result */}
          {result && (
            <div className="result-panel fade-in">
              <AdvisoryReport data={result} />
            </div>
          )}

          {/* Batch CSV Results */}
          {batchResults && (
            <div className="batch-panel fade-in">
              <div className="batch-header">
                <h3 className="batch-title">📊 Batch Prediction Results</h3>
                <span className="batch-count">{batchResults.count} rows analyzed</span>
              </div>

              {/* Summary Table */}
              <div className="batch-table-wrap">
                <table className="batch-table">
                  <thead>
                    <tr><th>#</th><th>N</th><th>P</th><th>K</th><th>Temp</th><th>Hum.</th><th>pH</th><th>Rain</th><th>Crop</th><th>Health</th><th></th></tr>
                  </thead>
                  <tbody>
                    {batchResults.results.map((r, i) => (
                      <tr key={i} className={`batch-row ${expandedRow === i ? 'expanded' : ''}`}
                        onClick={() => setExpandedRow(expandedRow === i ? null : i)}
                        style={{ cursor: 'pointer' }}>
                        <td>{i + 1}</td>
                        <td>{r.input?.N}</td><td>{r.input?.P}</td><td>{r.input?.K}</td>
                        <td>{r.input?.temperature}</td><td>{r.input?.humidity}</td>
                        <td>{r.input?.ph}</td><td>{r.input?.rainfall}</td>
                        <td><span className="crop-cell">{r.cropEmoji} {r.crop}</span></td>
                        <td><span className={`health-badge-sm health-${r.healthLevel}`}>{r.healthIcon} {r.health}</span></td>
                        <td><span className="expand-icon">{expandedRow === i ? '▲' : '▼'}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Expanded Detail for selected row */}
              {expandedRow !== null && batchResults.results[expandedRow] && (
                <div className="batch-detail fade-in">
                  <div className="batch-detail-header">
                    <span>📄 Detailed Advisory Report — Row #{expandedRow + 1}</span>
                    <button className="close-detail" onClick={() => setExpandedRow(null)}>✕</button>
                  </div>
                  <AdvisoryReport data={batchResults.results[expandedRow]} />
                </div>
              )}

              <p className="batch-hint">💡 Click any row to view its full advisory report</p>
            </div>
          )}

          {!result && !batchResults && (
            <div className="empty-state">
              <span className="empty-icon">🌿</span>
              <p className="empty-text">Enter your soil parameters above to get a crop recommendation</p>
            </div>
          )}
        </div>
      </section>

      {/* Dashboard Section */}
      <section className="section" id="dashboard">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">📊 Model Performance Dashboard</h2>
            <p className="section-desc">Comparing all 4 ML models trained on the AgriCrop pipeline</p>
          </div>

          {modelInfo && (
            <>
              <div className="model-grid">
                {modelInfo.models.map((m, i) => (
                  <div className={`model-card ${i === 0 ? 'best' : ''}`} key={m.model}>
                    {i === 0 && <div className="best-badge">⭐ Best Model</div>}
                    <h4 className="model-name">{m.model}</h4>
                    <div className="metric-bars">
                      {['accuracy', 'precision', 'recall', 'f1_score'].map(metric => (
                        <div className="metric-row" key={metric}>
                          <span className="metric-label">{metric === 'f1_score' ? 'F1-Score' : metric.charAt(0).toUpperCase() + metric.slice(1)}</span>
                          <div className="metric-bar-bg">
                            <div className="metric-bar-fill" style={{ width: `${m[metric] * 100}%` }}></div>
                          </div>
                          <span className="metric-value">{(m[metric] * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              <div className="crops-section">
                <h3 className="crops-title">🌿 Supported Crops</h3>
                <div className="crops-grid">
                  {modelInfo.crops.map(c => (
                    <div className="crop-pill" key={c.name}>
                      <span className="crop-emoji-sm">{c.emoji || '🌱'}</span> {c.name}
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {!modelInfo && <div className="empty-state"><span className="empty-icon">⏳</span><p className="empty-text">Loading model information...</p></div>}
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <p className="footer-text">🌾 AgriCrop — ICCSC 2026 | AKGEC, Ghaziabad</p>
        <p className="footer-sub">Built with 🌱 for smarter agriculture in Uttar Pradesh</p>
      </footer>
    </div>
  );
}

export default App;
