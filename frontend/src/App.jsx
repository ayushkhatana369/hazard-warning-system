import React, { useState } from "react";

function App() {
  const [hazardType, setHazardType] = useState("cyclone");
  const [inputData, setInputData] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [probability, setProbability] = useState(null);

  const handleHazardChange = (e) => {
    setHazardType(e.target.value);
    setInputData("");
    setResult("");
    setProbability(null);
  };

  const handleInputChange = (e) => setInputData(e.target.value);

  const validateInputShape = (data) => {
    if (!Array.isArray(data) || data.length < 1) return false;
    if (!Array.isArray(data[0])) data = [data];
    const rows = data.length;
    const cols = data[0].length;
    let expectedRows = 64;
    let expectedCols = hazardType === "cyclone" ? 6 : 129;
    if (cols !== expectedCols) return false;
    if (rows !== 1 && rows !== expectedRows) return false;
    return true;
  };

  const handlePredict = async () => {
    setResult("");
    setProbability(null);
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
        `Invalid input shape for ${hazardType.toUpperCase()} input. Columns must be exactly ${expectedCols}, and rows must be 1 or 64.`
      );
      return;
    }

    setLoading(true);
    try {
      const backendIP = "192.168.1.2"; // your Flask server IP
      const endpoint =
        hazardType === "cyclone"
          ? `http://${backendIP}:5000/predict/cyclone`
          : `http://${backendIP}:5000/predict/earthquake`;

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spectrogram: data }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const resData = await response.json();
      const prob = resData.probability;
      setProbability(prob);
      setResult(
        prob !== undefined
          ? `Prediction Probability: ${Number(prob).toFixed(4)}`
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

  // Background and heading colors
  let bgClass =
    probability === null
      ? "from-gray-900 via-gray-800 to-black"
      : probability > 0.5
      ? "from-red-900 via-black to-purple-900 animate-gradient-danger"
      : "from-green-800 via-emerald-700 to-teal-800 animate-gradient-safe";

  const headingGradient =
    probability === null
      ? "from-blue-400 via-purple-400 to-pink-400"
      : probability > 0.5
      ? "from-red-400 via-orange-400 to-yellow-400"
      : "from-emerald-300 via-green-400 to-lime-300";

  return (
    <div
      className={`flex flex-col min-h-screen w-full bg-gradient-to-br ${bgClass} text-gray-100 transition-all duration-700`}
    >
      {/* Main Content */}
      <main className="flex flex-col justify-center items-center flex-grow text-center px-6 py-12">
        <h1
          className={`text-6xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r ${headingGradient} mb-3`}
        >
          Hazard Prediction
        </h1>
        <p className="text-lg text-gray-300 mb-10">
          Cyclone 🌪️ & Earthquake 🌋 Detection Model
        </p>

        <div className="bg-white/10 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/20 w-full max-w-4xl p-10 space-y-8">
          {/* Hazard Type Selector */}
          <div>
            <label className="block text-lg font-semibold mb-2">
              Select Hazard Type:
            </label>
            <select
              value={hazardType}
              onChange={handleHazardChange}
              className="p-3 rounded-lg bg-gray-900 border border-gray-600 focus:ring-2 focus:ring-blue-400 w-64 text-center"
            >
              <option value="cyclone">🌪️ Cyclone (64×6)</option>
              <option value="earthquake">🌋 Earthquake (64×129)</option>
            </select>
          </div>

          {/* Input Area */}
          <div>
            <label htmlFor="input-data" className="block mb-2 font-medium">
              Enter your 2D array JSON:
            </label>
            <textarea
              id="input-data"
              value={inputData}
              onChange={handleInputChange}
              rows={10}
              placeholder={`Paste your ${hazardType} input JSON here`}
              className="bg-gray-900/70 border border-gray-600 rounded-xl p-3 text-sm w-full font-mono focus:ring-2 focus:ring-blue-400"
            />
          </div>

          {/* Predict Button */}
          <div className="flex justify-center">
            <button
              onClick={handlePredict}
              disabled={loading}
              className={`px-12 py-3 text-lg rounded-xl font-semibold shadow-lg transition-all duration-300 ${
                loading
                  ? "bg-gray-600 cursor-not-allowed"
                  : probability === null
                  ? "bg-blue-500 hover:bg-blue-600 shadow-blue-500/30"
                  : probability > 0.5
                  ? "bg-red-600 hover:bg-red-700 shadow-red-500/40 animate-pulse"
                  : "bg-green-600 hover:bg-green-700 shadow-green-500/40"
              }`}
            >
              {loading ? (
                <div className="flex items-center space-x-2">
                  <div className="w-5 h-5 border-2 border-t-transparent border-white rounded-full animate-spin"></div>
                  <span>Predicting...</span>
                </div>
              ) : (
                "Predict"
              )}
            </button>
          </div>

          {/* Result Display */}
          <div className="text-center text-xl font-semibold text-yellow-300">
            {result}
          </div>

          {/* Example Format */}
          <div className="text-sm text-gray-300 whitespace-pre-wrap text-center">
            <b>Example input format for {hazardType}:</b>
            <br />
            {hazardType === "cyclone" ? (
              <>
                [ [0.6, 0.4, 0.3, 0.2, 0.1, 0.0], ... ]
                {"\n(or a single row: [0.6, 0.4, 0.3, 0.2, 0.1, 0.0])"}
              </>
            ) : (
              <>
                [ [0.01, 0.02, ..., 0.12, ..., 0.15], ... ]
                {"\n(or a single row with 129 values)"}
              </>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-4 text-sm text-center text-gray-400 border-t border-white/10 bg-black/30 mt-auto w-full">
        Made with <span className="text-red-500">❤️</span> by{" "}
        <span className="font-semibold text-gray-200">Ayush</span>
      </footer>
    </div>
  );
}

export default App;
