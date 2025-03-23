import React, { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      console.log("API Response:", data); // Check response

      // Correctly set result
      setResult(`Prediction: ${data.prediction}, Confidence: ${data.confidence}`);
    } catch (error) {
      console.error("Error:", error);
      setResult("Error in prediction.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-blue-500 to-purple-600 flex flex-col items-center justify-center text-white p-4">
      <h1 className="text-4xl font-bold mb-6">🕵️‍♂️ Deepfake Detector</h1>

      <form
        onSubmit={handleSubmit}
        className="bg-white p-6 rounded-xl shadow-xl flex flex-col items-center gap-4 text-gray-800"
      >
        <input
          type="file"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none"
        />
        <button
          type="submit"
          disabled={loading}
          className={`px-6 py-2 rounded-lg font-semibold ${
            loading ? "bg-gray-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
          } text-white`}
        >
          {loading ? "Analyzing..." : "Detect"}
        </button>
      </form>

      {result && (
        <div className="mt-6 p-4 bg-white rounded-lg shadow-lg text-gray-800">
          <h2 className="text-xl font-semibold">Result:</h2>
          <p className={`text-lg mt-2 ${result.includes("FAKE") ? "text-red-500" : "text-green-500"}`}>
            {result}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
