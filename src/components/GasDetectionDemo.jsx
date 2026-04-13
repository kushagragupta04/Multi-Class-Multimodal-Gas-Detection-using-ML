import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, Sliders, RefreshCw, AlertCircle, Zap, Wind } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const SENSOR_COLS = ['MQ2', 'MQ3', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ135'];

const SENSOR_META = {
  MQ2:   { label: 'MQ2',   desc: 'Smoke / LPG / Methane',    color: 'blue' },
  MQ3:   { label: 'MQ3',   desc: 'Alcohol / Perfume',        color: 'violet' },
  MQ5:   { label: 'MQ5',   desc: 'Natural Gas / LPG',        color: 'cyan' },
  MQ6:   { label: 'MQ6',   desc: 'LPG / Butane',             color: 'teal' },
  MQ7:   { label: 'MQ7',   desc: 'Carbon Monoxide',          color: 'orange' },
  MQ8:   { label: 'MQ8',   desc: 'Hydrogen Gas',             color: 'pink' },
  MQ135: { label: 'MQ135', desc: 'Air Quality / NH₃ / CO₂',  color: 'green' },
};

const CLASS_CONFIG = {
  Smoke:   { bg: 'from-orange-900/40 to-slate-900', badge: 'bg-orange-500/20 text-orange-300 border-orange-500/40', icon: '🔥', dot: 'bg-orange-400' },
  Perfume: { bg: 'from-violet-900/40 to-slate-900', badge: 'bg-violet-500/20 text-violet-300 border-violet-500/40', icon: '🌸', dot: 'bg-violet-400' },
  NoGas:   { bg: 'from-emerald-900/40 to-slate-900', badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40', icon: '✅', dot: 'bg-emerald-400' },
  Mixture: { bg: 'from-yellow-900/40 to-slate-900', badge: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40', icon: '⚗️', dot: 'bg-yellow-400' },
};

const TAILWIND_BARS = {
  blue:   'bg-blue-400',
  violet: 'bg-violet-400',
  cyan:   'bg-cyan-400',
  teal:   'bg-teal-400',
  orange: 'bg-orange-400',
  pink:   'bg-pink-400',
  green:  'bg-green-400',
};

function SensorBar({ label, value, color }) {
  const pct = Math.min(100, Math.round((value / 1023) * 100));
  return (
    <div className="flex items-center gap-3">
      <span className="w-14 text-xs text-slate-400 font-mono shrink-0">{label}</span>
      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${TAILWIND_BARS[color]}`}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
        />
      </div>
      <span className="w-12 text-xs text-slate-300 font-mono text-right shrink-0">
        {typeof value === 'number' ? value.toFixed(0) : '—'}
      </span>
    </div>
  );
}

function ConfidenceBar({ cls, pct, isTop }) {
  const cfg = CLASS_CONFIG[cls] || { dot: 'bg-slate-400' };
  return (
    <div className="flex items-center gap-2">
      <span className={`w-2 h-2 rounded-full shrink-0 ${cfg.dot}`} />
      <span className="w-16 text-xs text-slate-400 shrink-0">{cls}</span>
      <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${cfg.dot}`}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>
      <span className={`w-10 text-xs font-mono text-right shrink-0 ${isTop ? 'text-white font-semibold' : 'text-slate-500'}`}>
        {pct.toFixed(1)}%
      </span>
    </div>
  );
}

// ─── Auto Mode ───────────────────────────────────────────────────────────────
function AutoMode() {
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [tick, setTick]           = useState(0);
  const [apiReady, setApiReady]   = useState(null);   // null=checking, true, false
  const intervalRef               = useRef(null);

  const fetchRandom = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/random`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResult(data);
      setApiReady(true);
    } catch (e) {
      setError(e.message);
      setApiReady(false);
    } finally {
      setLoading(false);
    }
  }, []);

  // Wake-up probe + first fetch
  useEffect(() => {
    setApiReady(null);
    fetchRandom();
  }, [fetchRandom]);

  // Poll every 3 s
  useEffect(() => {
    intervalRef.current = setInterval(() => {
      setTick(t => t + 1);
      fetchRandom();
    }, 3000);
    return () => clearInterval(intervalRef.current);
  }, [fetchRandom]);

  const cfg = result ? (CLASS_CONFIG[result.predicted_class] || CLASS_CONFIG.NoGas) : null;

  return (
    <div className="space-y-6">
      {/* Status bar */}
      <div className="flex items-center justify-between text-xs text-slate-500">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${apiReady === null ? 'bg-yellow-400 animate-pulse' : apiReady ? 'bg-emerald-400' : 'bg-red-400'}`} />
          {apiReady === null && 'Waking up API…'}
          {apiReady === true  && 'Live  ·  refreshing every 3s'}
          {apiReady === false && 'API unreachable — check VITE_API_URL'}
        </div>
        {loading && <RefreshCw size={12} className="animate-spin text-blue-400" />}
      </div>

      {error && (
        <div className="flex items-center gap-2 text-red-400 text-sm bg-red-900/20 border border-red-800/40 rounded-xl px-4 py-3">
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      {/* Sensor bars */}
      {result && (
        <AnimatePresence mode="wait">
          <motion.div
            key={tick}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-2"
          >
            <p className="text-xs text-slate-500 mb-3">Sensor readings (raw ADC values, 0–1023)</p>
            {SENSOR_COLS.map(col => (
              <SensorBar
                key={col}
                label={col}
                value={result.sensors[col]}
                color={SENSOR_META[col].color}
              />
            ))}
          </motion.div>
        </AnimatePresence>
      )}

      {/* Prediction card */}
      <AnimatePresence mode="wait">
        {result && (
          <motion.div
            key={`${tick}-pred`}
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`rounded-2xl bg-gradient-to-br ${cfg.bg} border border-slate-700/60 p-5 space-y-4`}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-slate-500 mb-1">Detected gas class</p>
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{cfg.icon}</span>
                  <span className="text-2xl font-bold text-white tracking-tight">
                    {result.predicted_class}
                  </span>
                </div>
              </div>
              <div className={`px-3 py-1.5 rounded-full border text-sm font-semibold ${cfg.badge}`}>
                {result.confidence}%
              </div>
            </div>

            {/* Probability breakdown */}
            <div className="space-y-1.5 pt-2 border-t border-slate-700/40">
              {Object.entries(result.probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([cls, pct]) => (
                  <ConfidenceBar key={cls} cls={cls} pct={pct} isTop={cls === result.predicted_class} />
                ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Manual Mode ─────────────────────────────────────────────────────────────
const DEFAULT_VALUES = {
  MQ2: 620, MQ3: 370, MQ5: 340, MQ6: 368, MQ7: 580, MQ8: 582, MQ135: 308,
};

const PRESETS = {
  Smoke:   { MQ2: 620, MQ3: 370, MQ5: 340, MQ6: 368, MQ7: 580, MQ8: 582, MQ135: 308 },
  Perfume: { MQ2: 753, MQ3: 523, MQ5: 489, MQ6: 461, MQ7: 685, MQ8: 696, MQ135: 495 },
  NoGas:   { MQ2: 748, MQ3: 529, MQ5: 416, MQ6: 423, MQ7: 615, MQ8: 656, MQ135: 458 },
  Mixture: { MQ2: 678, MQ3: 442, MQ5: 456, MQ6: 418, MQ7: 514, MQ8: 371, MQ135: 435 },
};

function ManualMode() {
  const [values, setValues]   = useState(DEFAULT_VALUES);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const handleChange = (col, raw) => {
    const num = parseFloat(raw);
    setValues(v => ({ ...v, [col]: isNaN(num) ? raw : Math.min(1023, Math.max(0, num)) }));
  };

  const handlePreset = (preset) => {
    setValues(PRESETS[preset]);
    setResult(null);
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const body = Object.fromEntries(
        SENSOR_COLS.map(col => [col, parseFloat(values[col]) || 0])
      );
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const cfg = result ? (CLASS_CONFIG[result.predicted_class] || CLASS_CONFIG.NoGas) : null;

  return (
    <div className="space-y-6">
      {/* Presets */}
      <div>
        <p className="text-xs text-slate-500 mb-2">Load a preset</p>
        <div className="flex flex-wrap gap-2">
          {Object.keys(PRESETS).map(preset => {
            const pc = CLASS_CONFIG[preset];
            return (
              <button
                key={preset}
                onClick={() => handlePreset(preset)}
                className={`px-3 py-1 rounded-full text-xs font-medium border transition-all ${pc.badge} hover:opacity-80`}
              >
                {pc.icon} {preset}
              </button>
            );
          })}
        </div>
      </div>

      {/* Sensor inputs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {SENSOR_COLS.map(col => (
          <div key={col}>
            <label className="block text-xs text-slate-400 mb-1">
              <span className="font-mono font-semibold text-slate-200">{SENSOR_META[col].label}</span>
              <span className="ml-2 text-slate-600">{SENSOR_META[col].desc}</span>
            </label>
            <input
              type="number"
              min={0}
              max={1023}
              step={1}
              value={values[col]}
              onChange={e => handleChange(col, e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-sm font-mono text-white
                         focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50
                         transition-colors placeholder-slate-600"
            />
          </div>
        ))}
      </div>

      {/* Detect button */}
      <button
        onClick={handlePredict}
        disabled={loading}
        className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500
                   disabled:opacity-50 disabled:cursor-not-allowed
                   text-white font-semibold rounded-xl px-6 py-3 transition-colors"
      >
        {loading
          ? <><RefreshCw size={16} className="animate-spin" /> Detecting…</>
          : <><Zap size={16} /> Detect Gas Class</>
        }
      </button>

      {error && (
        <div className="flex items-center gap-2 text-red-400 text-sm bg-red-900/20 border border-red-800/40 rounded-xl px-4 py-3">
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      {/* Result */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className={`rounded-2xl bg-gradient-to-br ${cfg.bg} border border-slate-700/60 p-5 space-y-4`}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-slate-500 mb-1">Detected gas class</p>
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{cfg.icon}</span>
                  <span className="text-2xl font-bold text-white tracking-tight">
                    {result.predicted_class}
                  </span>
                </div>
              </div>
              <div className={`px-3 py-1.5 rounded-full border text-sm font-semibold ${cfg.badge}`}>
                {result.confidence}%
              </div>
            </div>

            <div className="space-y-1.5 pt-2 border-t border-slate-700/40">
              {Object.entries(result.probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([cls, pct]) => (
                  <ConfidenceBar key={cls} cls={cls} pct={pct} isTop={cls === result.predicted_class} />
                ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export function GasDetectionDemo() {
  const [mode, setMode] = useState('auto');   // 'auto' | 'manual'

  return (
    <section id="demo" className="py-20 px-6">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-500/20 rounded-full px-4 py-1.5 text-blue-400 text-xs font-medium mb-4">
            <Cpu size={12} />
            SensorTransformer · Live Demo
          </div>
          <h2 className="text-3xl font-bold mb-3">Try the Gas Detection Model</h2>
          <p className="text-slate-400 text-sm max-w-md mx-auto">
            The model runs 7 MQ sensor readings through a Transformer Encoder to classify
            the gas into one of four categories in real time.
          </p>
        </div>

        {/* Mode toggle */}
        <div className="flex gap-2 p-1 bg-slate-900 border border-slate-800 rounded-2xl mb-8">
          <button
            onClick={() => setMode('auto')}
            className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-medium transition-all
              ${mode === 'auto'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/40'
                : 'text-slate-400 hover:text-slate-200'}`}
          >
            <Wind size={14} />
            Auto  <span className="text-xs opacity-60">(random feed)</span>
          </button>
          <button
            onClick={() => setMode('manual')}
            className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-medium transition-all
              ${mode === 'manual'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/40'
                : 'text-slate-400 hover:text-slate-200'}`}
          >
            <Sliders size={14} />
            Manual  <span className="text-xs opacity-60">(enter values)</span>
          </button>
        </div>

        {/* Panel */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-3xl p-6 shadow-2xl">
          <AnimatePresence mode="wait">
            <motion.div
              key={mode}
              initial={{ opacity: 0, x: mode === 'auto' ? -12 : 12 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              {mode === 'auto' ? <AutoMode /> : <ManualMode />}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Footer note */}
        <p className="text-center text-xs text-slate-600 mt-4">
          Model: SensorTransformer · 2-layer Transformer Encoder · ~96% accuracy · Trained on 6,400 samples
        </p>
      </div>
    </section>
  );
}
