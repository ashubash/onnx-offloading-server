// src/FundusDemo.jsx
import React, { useEffect, useState } from "react";
import useCppServerInference from "./hooks/useCppServerInference"; // Use the new server hook

const CLASSES = ["Normal", "Glaucoma", "Myopia", "Diabetes"];
const API_URL = 'http://13.211.167.122:8080';

export default function FundusDemo() {
  const { runInference, loading: inferenceLoading, error: inferenceError } = useCppServerInference(API_URL);

  const [testSplit, setTestSplit] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedPreprocPath, setSelectedPreprocPath] = useState(null);
  const [gtLabel, setGtLabel] = useState(null);
  const [predResult, setPredResult] = useState(null);
  const [inferenceTime, setInferenceTime] = useState(null);
  const [serverStatus, setServerStatus] = useState('checking');

  // Fetch the list of available images
  useEffect(() => {
    async function fetchTestSplit() {
      try {
        const res = await fetch("/test_split_preproc.json");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setTestSplit(data);
      } catch (err) {
        console.error("Failed to load test_split_preproc.json:", err);
      }
    }
    fetchTestSplit();
  }, []);

  // Check if the backend server is running
  useEffect(() => {
    async function checkServerStatus() {
      try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
          setServerStatus('ready');
        } else {
          setServerStatus('error');
        }
      } catch (err) {
        console.error('Server health check failed:', err);
        setServerStatus('error');
      }
    }
    checkServerStatus();
  }, [API_URL]);

  // Pick a random image from the list
  const pickRandomImage = () => {
    if (!testSplit.length) return;
    const sample = testSplit[Math.floor(Math.random() * testSplit.length)];
    setSelectedImage(sample.original_path);
    setSelectedPreprocPath(sample.preproc_path);
    setGtLabel(sample.class_label_remapped);
    setPredResult(null); // Clear previous results
    setInferenceTime(null);
  };

  // Compute softmax probabilities from server output
  const computeProbs = (output) => {
    const probs = new Float32Array(output.length);
    let maxLogit = -Infinity;
    for (let i = 0; i < output.length; i++) {
      if (output[i] > maxLogit) maxLogit = output[i];
    }
    let sumExp = 0;
    for (let i = 0; i < output.length; i++) {
      sumExp += Math.exp(output[i] - maxLogit);
    }
    for (let i = 0; i < output.length; i++) {
      probs[i] = Math.exp(output[i] - maxLogit) / sumExp;
    }
    return probs;
  };

  // Handle the inference button click
  const runInferenceHandler = async () => {
    if (!selectedPreprocPath) {
      alert('Please select an image first.');
      return;
    }
    if (serverStatus !== 'ready') {
      alert('Server is not ready. Please try again later.');
      return;
    }

    const start = performance.now();
    try {
      // The magic happens here: we just pass the URL to the server!
      const output = await runInference(selectedPreprocPath);
      
      const probs = computeProbs(output);
      let maxProb = 0; let predIndex = 0;
      for (let i = 0; i < probs.length; i++) {
        if (probs[i] > maxProb) {
          maxProb = probs[i];
          predIndex = i;
        }
      }
      const confidence = maxProb;
      const end = performance.now();
      const time = (end - start).toFixed(2);
      const pred = { index: predIndex, confidence };
      setPredResult(pred);
      setInferenceTime(time);
    } catch (err) {
      console.error("Inference failed:", err);
    }
  };

  return (
    <div className="fundus-demo">
      <h1>Fundus Classification Demo (Optimized)</h1>

      <div style={{ padding: '8px', borderRadius: '4px', backgroundColor: serverStatus === 'ready' ? '#d4edda' : serverStatus === 'error' ? '#f8d7da' : '#fff3cd', marginBottom: '16px' }}>
        C++ Server Status: {serverStatus === 'ready' ? '✅ Ready' : serverStatus === 'error' ? '❌ Error' : '⏳ Checking...'}
      </div>

      {inferenceError && <p style={{ color: "red" }}>Inference Error: {inferenceError}</p>}

      <button onClick={pickRandomImage} disabled={!testSplit.length}>
        Pick Random Image
      </button>

      <button onClick={runInferenceHandler} disabled={inferenceLoading || !selectedPreprocPath || serverStatus !== 'ready'}>
        {inferenceLoading ? "Running Inference..." : "Run Inference"}
      </button>

      {selectedImage && (
        <div>
          <h3>Selected Image:</h3>
          <img
            src={selectedImage}
            alt="Fundus"
            width={224}
            height={224}
            style={{ border: '1px solid #ccc', display: 'block' }}
            onError={(e) => {
              e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjI0IiBoZWlnaHQ9IjIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzY2NiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4=';
            }}
          />
          {selectedPreprocPath && (
            <p style={{ fontSize: "0.8em", color: "#666" }}>
              Using latent: {selectedPreprocPath}
            </p>
          )}
        </div>
      )}

      {gtLabel !== null && (
        <p>Ground Truth: <b>{CLASSES[gtLabel]}</b></p>
      )}

      {predResult && (
        <p>
          Prediction: <b>{CLASSES[predResult.index]}</b> 
          | Confidence: {(predResult.confidence * 100).toFixed(2)}% 
          | Time: {inferenceTime} ms
        </p>
      )}
    </div>
  );
}