import React, { useState } from 'react';


function App() {
  const [spectrogram, setSpectrogram] = useState('');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e) => {
    setSpectrogram(e.target.value);
  };

  const handlePredict = async () => {
    setResult('');
    let data;
    try {
      data = JSON.parse(spectrogram);
    } catch {
      alert('Invalid JSON format');
      return;
    }
    setLoading(true);
    try {
      // Call Flask API, update to your deployed address if needed
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ spectrogram: data })
      });
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      const resData = await response.json();
      setResult(
        resData.probability !== undefined
          ? `Prediction Probability: ${Number(resData.probability).toFixed(4)}`
          : resData.error
            ? `Server error: ${resData.error}`
            : 'Unknown error or response'
      );
    } catch (err) {
      setResult(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      background: '#222',
      color: '#eee',
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '30px'
    }}>
      <h2 style={{marginBottom: '8px'}}>Prediction from Seismic/Cyclone Model</h2>
      <label htmlFor="spectro-input" style={{marginBottom: '4px'}}>
        Enter your 2D array below:
      </label>
      <textarea
        id="spectro-input"
        value={spectrogram}
        onChange={handleInputChange}
        rows={12}
        cols={50}
        placeholder='Paste your spectrogram/time series JSON'
        style={{marginBottom: '10px', resize: 'vertical'}}
      />
      <br />
      <button
        onClick={handlePredict}
        disabled={loading}
        style={{
          padding: '10px 24px',
          borderRadius: '8px',
          fontSize: '16px',
          background: loading ? '#444' : '#2d9bf0',
          color: '#fff',
          border: 'none',
          cursor: loading ? 'not-allowed' : 'pointer'
        }}>
        {loading ? 'Predicting...' : 'Predict'}
      </button>
      <div style={{marginTop: '20px', fontSize: '18px'}}>
        {result}
      </div>
      <div style={{marginTop: '20px', fontSize: '0.9em', color: '#aaa'}}>
        Example input format: <br />
        <code>
          [[0.1,0.2,0.3,0.4,0.5,0.6],<br />
          [0.2,0.3,0.4,0.5,0.6,0.7], ...]
        </code>
      </div>
    </div>
  );
}

export default App;
