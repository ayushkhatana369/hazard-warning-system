import React, { useState } from "react";

function App() {
  const [hazardType, setHazardType] = useState("cyclone"); // 'cyclone' or 'earthquake'
  const [inputData, setInputData] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleHazardChange = (e) => {
    setHazardType(e.target.value);
    setInputData("");
    setResult("");
  };

  const handleInputChange = (e) => {
    setInputData(e.target.value);
  };

  const validateInputShape = (data) => {
  if (!Array.isArray(data) || data.length < 1) return false;
  if (!Array.isArray(data[0])) {
    // single row input: make it a 2D array
    data = [data];
  }

  const rows = data.length;
  const cols = data[0].length;

  let expectedRows = 64;
  let expectedCols = hazardType === 'cyclone' ? 6 : 129;

  if (cols !== expectedCols) return false;
  if (rows !== 1 && rows !== expectedRows) return false;

  return true;
};

    

  const handlePredict = async () => {
    setResult("");
    let data;
    try {
      data = JSON.parse(inputData);
    } catch {
      alert("Invalid JSON format");
      return;
    }

    if (!validateInputShape(data)) {
      const expectedCols = hazardType === "cyclone" ? 6 : 129;
      alert(
        `Invalid input shape for ${hazardType.toUpperCase()} input. ` +
          `Columns must be exactly ${expectedCols}, and rows must be 1 or 64.`
      );
      return;
    }

    setLoading(true);
    try {
      const backendIP = "192.168.1.2"; // Your Flask server LAN IP

      const endpoint =
        hazardType === "cyclone"
          ? `http://${backendIP}:5000/predict/cyclone`
          : `http://${backendIP}:5000/predict/earthquake`;

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spectrogram: data }),
      });

      if (!response.ok) {
        throw new Error(
          `Server error: ${response.status} ${response.statusText}`
        );
      }

      const resData = await response.json();

      setResult(
        resData.probability !== undefined
          ? `Prediction Probability: ${Number(resData.probability).toFixed(4)}`
          : resData.error
          ? `Server error: ${resData.error}`
          : "Unknown error or response"
      );
    } catch (err) {
      setResult(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        background: "#222",
        color: "#eee",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "30px",
      }}
    >
      <h2 style={{ marginBottom: "8px" }}>
        Hazard Prediction: Cyclone / Earthquake
      </h2>

      <div style={{ marginBottom: "20px" }}>
        <label>
          Select Hazard Type:{" "}
          <select value={hazardType} onChange={handleHazardChange}>
            <option value="cyclone">Cyclone (64 rows × 6 columns)</option>
            <option value="earthquake">
              Earthquake (64 rows × 129 columns)
            </option>
          </select>
        </label>
      </div>

      <label htmlFor="input-data" style={{ marginBottom: "4px" }}>
        Enter your 2D array JSON (single row or 64 rows):
      </label>
      <textarea
        id="input-data"
        value={inputData}
        onChange={handleInputChange}
        rows={20}
        cols={60}
        placeholder={`Paste your ${hazardType} input JSON here`}
        style={{
          marginBottom: "10px",
          resize: "vertical",
          fontFamily: "monospace",
        }}
      />
      <br />

      <button
        onClick={handlePredict}
        disabled={loading}
        style={{
          padding: "10px 24px",
          borderRadius: "8px",
          fontSize: "16px",
          background: loading ? "#444" : "#2d9bf0",
          color: "#fff",
          border: "none",
          cursor: loading ? "not-allowed" : "pointer",
        }}
      >
        {loading ? "Predicting..." : "Predict"}
      </button>

      <div style={{ marginTop: "20px", fontSize: "18px" }}>{result}</div>

      <div
        style={{
          marginTop: "20px",
          fontSize: "0.9em",
          color: "#aaa",
          maxWidth: "700px",
          whiteSpace: "pre-wrap",
        }}
      >
        <b>Example input format for {hazardType} (single row or 64 rows):</b>
        <br />
        {hazardType === "cyclone" ? (
          <>
            {"[ [0.6, 0.4, 0.3, 0.2, 0.1, 0.0],\n".repeat(2)}
            (or just a single row: [0.6, 0.4, 0.3, 0.2, 0.1, 0.0])
          </>
        ) : (
          <>
            {"[ [0.01, 0.02, ..., 0.12, ... , 0.15],\n".repeat(2)}
            (or just a single row with 129 cols)
          </>
        )}
      </div>
    </div>
  );
}

export default App;
