import React, { useState, useEffect } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const EXAMPLES = [
  {
    title: "U.S. Budget Fight (Likely Real)",
    text: "As U.S. budget fight looms, Republicans flip their fiscal script. Washington (Reuters) - The U.S. Congress will return to work next week facing a mountain of unfinished business, including a high-stakes budget fight that will test Republicans' resolve to rein in federal spending. Ahead of the congressional debates, conservative Republicans have started to pivot their rhetoric towards reducing the national deficit, signaling a tough negotiation ahead with Democrats."
  },
  {
    title: "Trump New Year Message (Likely Fake)",
    text: "Donald Trump Sends Out Embarrassing New Year's Message. Donald Trump just couldn't help himself. On New Year's Eve, the President of the United States sent out a bizarre and childish tweet wishing a happy new year to his enemies, haters, and the dishonest fake news media. The tweet was immediately ridiculed across social media for its unprofessional tone and lack of presidential dignity."
  }
];

function App() {
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [activeStep, setActiveStep] = useState('raw');

  // Ensure input is empty on load
  useEffect(() => {
    setInputText('');
  }, []);

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!inputText.trim()) return;

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to make prediction.");
      }

      const data = await response.json();
      setResults(data);
      setActiveStep('raw'); // reset pipeline view to raw input
    } catch (err) {
      console.error(err);
      setError(err.message || "An error occurred while connecting to the backend server.");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setInputText('');
    setResults(null);
    setError(null);
  };

  const handleLoadExample = (example) => {
    setInputText(example.text);
    setResults(null);
    setError(null);
  };

  // Format current date
  const getFormattedDate = () => {
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    return new Date().toLocaleDateString('en-US', options);
  };

  return (
    <div className="app-container">
      {/* Newspaper Header */}
      <header className="newspaper-header">
        <h1 className="newspaper-title">Fake News Prediction</h1>
        <p className="newspaper-subtitle">Natural Language Processing & Machine Learning Fake News Forensic Lab</p>
        <div className="newspaper-meta">
          <span>INVESTIGATION NO. 101</span>
          <span>{getFormattedDate()}</span>
          <span>PRICE: 10 CENTS</span>
        </div>
      </header>

      {/* Error alert banner */}
      {error && (
        <div className="alert-banner">
          <div className="alert-banner-text">
            <div className="alert-banner-title">Forensic Alert</div>
            {error}
          </div>
        </div>
      )}

      {/* Main Grid layout */}
      <main className="dashboard-grid">
        {/* Left Column: Input and Preprocessing */}
        <section className="left-panel">
          <h2 className="section-title">
            <span>I. News Article Under Examination</span>
          </h2>
          
          <div className="input-section">
            <p style={{ fontSize: '0.95rem', marginBottom: '0.5rem', fontStyle: 'italic', color: 'var(--text-muted)' }}>
              Paste the news article text below. The NLP engine will run tokenization, lowercase normalization, and stopword extraction, then feed it to the ensemble classifier.
            </p>
            
            <textarea
              className="news-textarea"
              placeholder="Paste article headline or content here to analyze..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              disabled={loading}
              autoComplete="off"
            />
            
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 'bold', marginRight: '0.5rem' }}>DOCKET TEMPLATES:</span>
              {EXAMPLES.map((ex, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="btn btn-secondary"
                  style={{ fontSize: '0.8rem', padding: '0.3rem 0.6rem' }}
                  onClick={() => handleLoadExample(ex)}
                  disabled={loading}
                >
                  {ex.title}
                </button>
              ))}
            </div>

            <div className="button-container">
              <button
                className="btn btn-primary"
                onClick={handleSubmit}
                disabled={loading || !inputText.trim()}
              >
                {loading ? 'Analyzing...' : 'Run Forensic Analysis'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={handleClear}
                disabled={loading || !inputText}
              >
                Clear Slate
              </button>
            </div>
          </div>

          {/* Interactive NLP Pipeline Stepper */}
          {results && (
            <div className="pipeline-container">
              <h3 className="sub-section-title">NLP Preprocessing Pipeline</h3>
              <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '1.2rem', fontStyle: 'italic' }}>
                Click on each stage to inspect the intermediate text transformations performed by the Natural Language Toolkit:
              </p>
              
              <div className="pipeline-stepper">
                {/* Step 1: Raw Input */}
                <div 
                  className={`pipeline-step ${activeStep === 'raw' ? 'active' : ''}`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setActiveStep('raw')}
                >
                  <div className="step-header">
                    <span className="step-title">Stage 1: Raw News Text</span>
                    <span className="step-badge">INPUT</span>
                  </div>
                  {activeStep === 'raw' && (
                    <div className="step-content-box">
                      {results.preprocessing_steps.raw}
                    </div>
                  )}
                </div>

                {/* Step 2: Lowercasing */}
                <div 
                  className={`pipeline-step ${activeStep === 'lowercase' ? 'active' : ''}`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setActiveStep('lowercase')}
                >
                  <div className="step-header">
                    <span className="step-title">Stage 2: Case Folding</span>
                    <span className="step-badge">LOWERCASE</span>
                  </div>
                  {activeStep === 'lowercase' && (
                    <div className="step-content-box">
                      {results.preprocessing_steps.lowercase}
                    </div>
                  )}
                </div>

                {/* Step 3: Cleaned Text */}
                <div 
                  className={`pipeline-step ${activeStep === 'cleaned' ? 'active' : ''}`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setActiveStep('cleaned')}
                >
                  <div className="step-header">
                    <span className="step-title">Stage 3: Punctuation & Digit Filtering</span>
                    <span className="step-badge">REGEX STRIP</span>
                  </div>
                  {activeStep === 'cleaned' && (
                    <div className="step-content-box">
                      {results.preprocessing_steps.cleaned}
                    </div>
                  )}
                </div>

                {/* Step 4: Tokenization */}
                <div 
                  className={`pipeline-step ${activeStep === 'tokenized' ? 'active' : ''}`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setActiveStep('tokenized')}
                >
                  <div className="step-header">
                    <span className="step-title">Stage 4: Word Tokenization</span>
                    <span className="step-badge">NLTK.PUNKT ({results.preprocessing_steps.tokenized.length} tokens)</span>
                  </div>
                  {activeStep === 'tokenized' && (
                    <div className="step-content-box">
                      <div className="token-list">
                        {results.preprocessing_steps.tokenized.map((t, idx) => (
                          <span key={idx} className="token-badge">{t}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Step 5: Stopwords Removed */}
                <div 
                  className={`pipeline-step ${activeStep === 'stopwords' ? 'active' : ''}`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setActiveStep('stopwords')}
                >
                  <div className="step-header">
                    <span className="step-title">Stage 5: Stopwords Exclusion</span>
                    <span className="step-badge">NLTK.STOPWORDS ({results.preprocessing_steps.stopwords_removed.length} remaining)</span>
                  </div>
                  {activeStep === 'stopwords' && (
                    <div className="step-content-box">
                      <div className="token-list">
                        {results.preprocessing_steps.stopwords_removed.map((t, idx) => (
                          <span key={idx} className="token-badge" style={{ backgroundColor: '#d0c6b1' }}>{t}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Right Column: Prediction results */}
        <section className="right-panel">
          <h2 className="section-title">
            <span>II. Forensic Verdict</span>
          </h2>

          {loading && (
            <div className="loading-overlay">
              <div className="analysis-spinner"></div>
              <p style={{ fontFamily: 'var(--font-serif)', fontSize: '1.2rem', fontStyle: 'italic' }}>
                Executing text preprocessing pipeline...<br />
                Querying Machine Learning, LSTM, and BERT classifiers...
              </p>
            </div>
          )}

          {!loading && !results && (
            <div className="empty-state">
              <div className="empty-state-icon">✍</div>
              <p className="empty-state-text">Ledger is empty.</p>
              <p style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>
                Enter news text on the left and submit it to display the forensic verdict.
              </p>
            </div>
          )}

          {!loading && results && (
            <div style={{ animation: 'fadeIn 0.5s ease-out' }}>
              {/* Ensemble Verdict Card */}
              <div className={`verdict-box ${results.ensemble.prediction_code === 1 ? 'true' : 'fake'}`}>
                <div className="verdict-header">ENSEMBLE CLASSIFICATION</div>
                <div className="verdict-title">{results.ensemble.prediction}</div>
                <p className="verdict-desc">
                  Based on a majority vote of 8 neural networks and statistical ML estimators.
                </p>
                <div className="verdict-stat">
                  VOTES: {results.ensemble.votes_true} True, {results.ensemble.votes_fake} Fake News ({results.ensemble.confidence}% Agreement)
                </div>
              </div>

              {/* Individual Model Predictions */}
              <h3 className="sub-section-title">Estimator Verdict Breakdown</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem', fontStyle: 'italic' }}>
                Verdicts rendered by each individual model based on their unique architecture:
              </p>

              <div className="predictions-list">
                {results.predictions.map((p, idx) => (
                  <div key={idx} className="prediction-item">
                    <div className="model-info">
                      <span className="model-name">{p.model}</span>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {p.model === 'BERT' ? 'Fine-tuned Transformer Sequence Classifier' : 
                         p.model === 'LSTM' ? 'Recurrent Neural Network (Word Sequence)' : 
                         'Statistical Machine Learning Model'}
                      </span>
                    </div>
                    <div className="model-vote">
                      <span className={`vote-badge ${p.prediction_code === 1 ? 'true' : 'fake'}`}>
                        {p.prediction_code === 1 ? 'Real' : 'Fake'}
                      </span>
                      <span className="model-confidence">
                        Conf: {p.confidence}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* Bottom Full-width Section: Model Prediction Voting Map & Consensus */}
        {results && (
          <section className="scoreboard-section" style={{ animation: 'fadeIn 0.5s ease-out' }}>
            <h2 className="section-title">
              <span>III. Ensemble Consensus & Model Voting Map</span>
            </h2>
            
            <p className="scoreboard-intro">
              This section visualizes the classification votes and prediction confidence scores of all 8 models for the article currently under examination. 
              The consensus meter displays the collective agreement of the ensemble, while the diverging chart plots individual model votes (Fake News on the left, Real News on the right) with their corresponding confidence levels (50% to 100%).
            </p>

            {/* 1. Consensus scale */}
            <div className="consensus-scale-container">
              <div className="consensus-labels">
                <span>☠ 100% FAKE CONSENSUS</span>
                <span>NEUTRAL / SPLIT</span>
                <span>✔ 100% REAL CONSENSUS</span>
              </div>
              <div className="consensus-track">
                <div 
                  className="consensus-pointer" 
                  style={{ left: `${(results.ensemble.votes_true / results.ensemble.total_models) * 100}%` }}
                ></div>
              </div>
              <div className="consensus-text">
                ENSEMBLE VERDICT: <span style={{ color: results.ensemble.prediction_code === 1 ? 'var(--verdict-true)' : 'var(--verdict-fake)', textTransform: 'uppercase' }}>{results.ensemble.prediction}</span> 
                <span style={{ fontStyle: 'italic', fontWeight: 'normal', marginLeft: '0.5rem', color: 'var(--text-muted)' }}>
                  ({results.ensemble.votes_true} of {results.ensemble.total_models} models voted Real — {results.ensemble.confidence}% consensus agreement)
                </span>
              </div>
            </div>

            {/* 2. Diverging voting map */}
            <h3 className="sub-section-title">Forensic Model Voting Alignment</h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '1.2rem' }}>
              Detailed breakdown mapping each model's prediction score. Bars extending left indicate a Fake News prediction, while bars extending right indicate a Real News prediction:
            </p>

            <div className="diverging-chart">
              {results.predictions.map((p) => {
                const isTrue = p.prediction_code === 1;
                // Scale confidence from 50%-100% range to 0%-100% width of the side
                const barWidth = Math.max(0, (p.confidence - 50) * 2);
                
                return (
                  <div key={p.model} className="diverging-row">
                    <div className="diverging-label">{p.model}</div>
                    <div className="diverging-bars-container">
                      {/* Left: Fake News */}
                      <div className="diverging-side left">
                        {!isTrue && (
                          <div className="diverging-bar fake" style={{ width: `${barWidth}%` }}>
                            <span style={{ marginRight: '0.3rem' }}>{p.confidence}%</span>
                          </div>
                        )}
                      </div>
                      
                      {/* Divider line */}
                      <div className="diverging-divider"></div>
                      
                      {/* Right: Real News */}
                      <div className="diverging-side right">
                        {isTrue && (
                          <div className="diverging-bar true" style={{ width: `${barWidth}%` }}>
                            <span style={{ marginLeft: '0.3rem' }}>{p.confidence}%</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}
      </main>
      
      {/* Newspaper Footer */}
      <footer style={{ marginTop: '4rem', paddingTop: '1.5rem', borderTop: '1px solid var(--border-color)', textAlign: 'center', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
        <p>© 2026 Fake News Prediction. Published in pair-programming cooperation with Antigravity AI.</p>
        <p style={{ marginTop: '0.3rem', fontStyle: 'italic' }}>
          Disclaimer: This site is a forensic laboratory demonstration. Veracity predictions are generated by statistical approximations of NLP models.
        </p>
      </footer>
    </div>
  );
}

export default App;
