// src/hooks/useCppServerInference.js
import { useState } from 'react';

export default function useCppServerInference(serverUrl) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // The function now takes an npy_url instead of a tensor object
  const runInference = async (npyUrl) => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${serverUrl}/inference_from_npy`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          npy_url: npyUrl, // Send the URL
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP ${response.status}`);
      }

      const result = await response.json();
      return result.output;
    } catch (err) {
      console.error('C++ Server inference failed:', err);
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { runInference, loading, error };
}