import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  Shield, LayoutDashboard, Wifi, WifiOff, Clock,
  Flame, Droplets, Wind, Zap, Activity, ArrowUpDown, AlertTriangle,
  TriangleAlert, CheckCircle2, RefreshCw, ChevronRight, ArrowLeft,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';
const HIGH_THRESHOLD = 550;
const POLL_MS = 2000; // 0.5 Hz

// ── Sensor metadata ───────────────────────────────────────────
const SENSORS = [
  { key: 'MQ2',   label: 'MQ2',   targetGas: 'Smoke / LPG',      Icon: Flame },
  { key: 'MQ3',   label: 'MQ3',   targetGas: 'Alcohol',           Icon: Droplets },
  { key: 'MQ5',   label: 'MQ5',   targetGas: 'LPG / Methane',     Icon: Wind },
  { key: 'MQ6',   label: 'MQ6',   targetGas: 'LPG (Propane)',      Icon: Zap },
  { key: 'MQ7',   label: 'MQ7',   targetGas: 'Carbon Monoxide',   Icon: Activity },
  { key: 'MQ8',   label: 'MQ8',   targetGas: 'Hydrogen',          Icon: ArrowUpDown },
  { key: 'MQ135', label: 'MQ135', targetGas: 'Ammonia / Benzene', Icon: AlertTriangle },
];

const PRESETS = {
  'No Gas':  { MQ2: 748, MQ3: 529, MQ5: 416, MQ6: 423, MQ7: 615, MQ8: 656, MQ135: 458 },
  'Smoke':   { MQ2: 620, MQ3: 370, MQ5: 340, MQ6: 368, MQ7: 580, MQ8: 582, MQ135: 308 },
  'Perfume': { MQ2: 753, MQ3: 523, MQ5: 489, MQ6: 461, MQ7: 685, MQ8: 696, MQ135: 495 },
  'Mixture': { MQ2: 678, MQ3: 442, MQ5: 456, MQ6: 418, MQ7: 514, MQ8: 371, MQ135: 435 },
};

const DEFAULT_VALUES = { MQ2: 400, MQ3: 300, MQ5: 350, MQ6: 330, MQ7: 420, MQ8: 380, MQ135: 320 };

// All hazard classes use blue tones — no purple anywhere
const CLASS_META = {
  Smoke:   { textCls: 'text-amber-600',  bgCls: 'bg-amber-50',  borderCls: 'border-amber-200',  hazard: true  },
  Perfume: { textCls: 'text-blue-600',   bgCls: 'bg-blue-50',   borderCls: 'border-blue-200',   hazard: true  },
  NoGas:   { textCls: 'text-emerald-600',bgCls: 'bg-emerald-50',borderCls: 'border-emerald-200', hazard: false },
  Mixture: { textCls: 'text-sky-700',    bgCls: 'bg-sky-50',    borderCls: 'border-sky-200',    hazard: true  },
};

// ── Helpers ───────────────────────────────────────────────────
const isHigh = (v) => v > HIGH_THRESHOLD;

function fmtUptime(secs) {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h > 0) return `${h}H ${m}M`;
  if (m > 0) return `${m}M ${s}S`;
  return `${s}S`;
}

function fmtTime(d = new Date()) {
  return d.toLocaleTimeString('en-US', { hour12: true, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// ── Level bar ─────────────────────────────────────────────────
function LevelBar({ value }) {
  const pct  = Math.min(100, (value / 1023) * 100);
  const high = isHigh(value);
  return (
    <div className="w-28 h-2 bg-slate-200 rounded-full overflow-hidden shrink-0">
      <div
        className={`h-full rounded-full transition-all duration-500 ${high ? 'bg-red-500' : 'bg-blue-500'}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

// ── Status badge ──────────────────────────────────────────────
function StatusBadge({ value }) {
  const high = isHigh(value);
  return high ? (
    <span className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded bg-red-100 text-red-700 tracking-wide">
      <TriangleAlert size={10} /> HIGH
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded bg-emerald-100 text-emerald-700 tracking-wide">
      <CheckCircle2 size={10} /> SAFE
    </span>
  );
}

// ── Auto mode sensor row ──────────────────────────────────────
function AutoRow({ sensor, value }) {
  const { Icon, label, targetGas } = sensor;
  const high = isHigh(value);
  return (
    <tr className={`border-b border-slate-100 transition-colors ${high ? 'bg-red-50/50' : 'hover:bg-slate-50'}`}>
      <td className="py-3.5 px-5">
        <div className="flex items-center gap-2.5">
          <span className={`p-1.5 rounded-lg ${high ? 'bg-red-100' : 'bg-blue-50'}`}>
            <Icon size={14} className={high ? 'text-red-600' : 'text-blue-500'} />
          </span>
          <span className="font-semibold text-slate-800 text-sm">{label}</span>
        </div>
      </td>
      <td className="py-3.5 px-4 text-sm text-slate-500">{targetGas}</td>
      <td className="py-3.5 px-4">
        <span className={`font-mono font-semibold text-sm ${high ? 'text-red-600' : 'text-slate-800'}`}>
          {typeof value === 'number' ? value.toFixed(1) : '—'}
        </span>
        <span className="text-xs text-slate-400 ml-1">PPM</span>
      </td>
      <td className="py-3.5 px-4"><LevelBar value={value} /></td>
      <td className="py-3.5 px-5"><StatusBadge value={value} /></td>
    </tr>
  );
}

// ── Manual mode sensor row (with slider) ─────────────────────
function ManualRow({ sensor, value, onChange }) {
  const { Icon, label, targetGas } = sensor;
  const high = isHigh(value);
  return (
    <tr className={`border-b border-slate-100 transition-colors ${high ? 'bg-red-50/50' : 'hover:bg-slate-50'}`}>
      <td className="py-3 px-5">
        <div className="flex items-center gap-2.5">
          <span className={`p-1.5 rounded-lg ${high ? 'bg-red-100' : 'bg-blue-50'}`}>
            <Icon size={14} className={high ? 'text-red-600' : 'text-blue-500'} />
          </span>
          <span className="font-semibold text-slate-800 text-sm">{label}</span>
        </div>
      </td>
      <td className="py-3 px-4 text-sm text-slate-500">{targetGas}</td>
      <td className="py-3 px-4">
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={0}
            max={1023}
            step={1}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
            className={`w-36 h-1.5 rounded-full cursor-pointer ${high ? 'accent-red-500' : 'accent-blue-600'}`}
          />
          <span className={`font-mono font-bold text-sm w-10 text-right ${high ? 'text-red-600' : 'text-slate-700'}`}>
            {value}
          </span>
        </div>
      </td>
      <td className="py-3 px-4"><LevelBar value={value} /></td>
      <td className="py-3 px-5"><StatusBadge value={value} /></td>
    </tr>
  );
}

// ── Incident log ──────────────────────────────────────────────
function IncidentLog({ entries }) {
  const bottomRef = useRef(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [entries]);

  return (
    <div className="w-64 shrink-0 border-l border-slate-200 flex flex-col bg-white">
      <div className="px-4 py-3.5 border-b border-slate-200 bg-slate-50">
        <span className="text-[11px] font-semibold text-slate-500 tracking-widest uppercase">Incident Log</span>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-3">
        {entries.length === 0 ? (
          <p className="text-xs text-slate-400 text-center mt-6">No incidents recorded.</p>
        ) : (
          [...entries].reverse().map((e, i) => (
            <div key={i} className="border-b border-slate-100 pb-2.5">
              <p className="text-xs font-semibold text-blue-600 mb-0.5">{e.time}</p>
              <p className="text-xs text-slate-600">
                <span className="font-semibold text-slate-800">High {e.sensor}</span>
                {' '}— {e.value} PPM
              </p>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

// ── Result banner ─────────────────────────────────────────────
function ResultBanner({ result, mode, loading }) {
  if (!result && !loading) return (
    <div className="mx-6 mt-5 rounded-xl border border-slate-200 bg-slate-50 px-6 py-4 flex items-center gap-3">
      <div className="w-2 h-2 rounded-full bg-slate-300 animate-pulse" />
      <span className="text-slate-400 text-sm">Waiting for first reading…</span>
    </div>
  );

  if (loading && !result) return (
    <div className="mx-6 mt-5 rounded-xl border border-slate-200 bg-slate-50 px-6 py-4 flex items-center gap-3">
      <RefreshCw size={14} className="animate-spin text-blue-500" />
      <span className="text-slate-500 text-sm">Running inference…</span>
    </div>
  );

  const cls  = result?.predicted_class;
  const meta = CLASS_META[cls] || CLASS_META.NoGas;
  const conf = result?.confidence;

  return (
    <div className={`mx-6 mt-5 rounded-xl border ${meta.borderCls} ${meta.bgCls} px-6 py-4 flex items-center justify-between`}>
      <div>
        <p className="text-[11px] font-semibold text-slate-400 tracking-widest uppercase mb-1">
          {mode === 'auto' ? '⚡ Live Model Result' : '⚙ Manual Model Inference'}
        </p>
        <p className={`text-3xl font-black tracking-tight ${meta.textCls}`}>{cls}</p>
      </div>
      <div className="flex items-center gap-4">
        <div className="text-right">
          <p className="text-xs text-slate-400 mb-0.5">Confidence</p>
          <p className={`text-xl font-bold ${meta.textCls}`}>{conf?.toFixed(1)}%</p>
        </div>
        {meta.hazard ? (
          <div className="flex items-center gap-2 bg-red-600 text-white text-xs font-bold px-4 py-2.5 rounded-lg">
            <TriangleAlert size={13} />
            HAZARD DETECTED
          </div>
        ) : (
          <div className="flex items-center gap-2 bg-emerald-600 text-white text-xs font-bold px-4 py-2.5 rounded-lg">
            <CheckCircle2 size={13} />
            ALL CLEAR
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sidebar ───────────────────────────────────────────────────
function Sidebar({ mode, connected, loading, onModeSwitch, onPreset, onDetect }) {
  return (
    <aside className="w-52 shrink-0 bg-white border-r border-slate-200 flex flex-col">

      {/* Logo */}
      <div className="px-5 py-5 border-b border-slate-200">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 bg-blue-600 rounded-lg flex items-center justify-center shadow-sm">
            <Shield size={14} className="text-white" />
          </div>
          <div>
            <p className="text-slate-900 font-bold text-sm leading-none">GasSense</p>
            <p className="text-blue-600 text-[10px] font-semibold tracking-widest uppercase">Pro</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="px-3 py-3 border-b border-slate-200">
        <div className="flex items-center gap-2.5 px-3 py-2 rounded-lg bg-blue-50 text-blue-700">
          <LayoutDashboard size={14} />
          <span className="text-sm font-semibold">Dashboard</span>
        </div>
      </nav>

      {/* Control Mode */}
      <div className="px-4 py-4 border-b border-slate-200 space-y-3">
        <p className="text-[10px] font-semibold text-slate-400 tracking-widest uppercase">Control Mode</p>
        <div className="flex gap-1.5">
          <button
            onClick={() => onModeSwitch('auto')}
            className={`flex-1 py-1.5 rounded-lg text-xs font-semibold transition-all
              ${mode === 'auto'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'bg-slate-100 text-slate-500 hover:bg-slate-200 hover:text-slate-700'}`}
          >
            AUTO
          </button>
          <button
            onClick={() => onModeSwitch('manual')}
            className={`flex-1 py-1.5 rounded-lg text-xs font-semibold transition-all
              ${mode === 'manual'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'bg-slate-100 text-slate-500 hover:bg-slate-200 hover:text-slate-700'}`}
          >
            MANUAL
          </button>
        </div>
        <div className={`text-center py-1 rounded text-[10px] font-bold tracking-wide
          ${mode === 'auto' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'}`}>
          {mode === 'auto' ? '● AUTOMATIC' : '⚙  MANUAL'}
        </div>
      </div>

      {/* Quick Scenarios — manual only */}
      {mode === 'manual' && (
        <div className="px-4 py-4 border-b border-slate-200 space-y-2">
          <p className="text-[10px] font-semibold text-slate-400 tracking-widest uppercase">Quick Scenarios</p>
          <div className="grid grid-cols-2 gap-1.5">
            {Object.keys(PRESETS).map(p => (
              <button
                key={p}
                onClick={() => onPreset(p)}
                className="py-1.5 px-2 rounded-lg bg-slate-100 hover:bg-blue-50 hover:text-blue-700
                           text-slate-600 text-[11px] font-medium transition-colors text-left border border-transparent hover:border-blue-200"
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Run Detection — manual only */}
      {mode === 'manual' && (
        <div className="px-4 py-4 border-b border-slate-200">
          <button
            onClick={onDetect}
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700
                       disabled:opacity-50 text-white text-xs font-bold py-2.5 rounded-lg transition-colors shadow-sm"
          >
            {loading
              ? <><RefreshCw size={12} className="animate-spin" /> Detecting…</>
              : <><ChevronRight size={12} /> RUN DETECTION</>}
          </button>
        </div>
      )}

      <div className="flex-1" />

      {/* Connection status */}
      <div className="px-4 py-4 border-t border-slate-200">
        <div className={`flex items-center gap-2 text-[11px] font-semibold
          ${connected === null ? 'text-amber-500' : connected ? 'text-emerald-600' : 'text-red-500'}`}>
          {connected === false ? <WifiOff size={12} /> : <Wifi size={12} />}
          <span>
            {connected === null && 'CONNECTING…'}
            {connected === true  && 'LINK ESTABLISHED'}
            {connected === false && 'DISCONNECTED'}
          </span>
        </div>
      </div>
    </aside>
  );
}

// ── Main Dashboard ────────────────────────────────────────────
export default function Dashboard() {
  const [mode, setMode]           = useState('auto');
  const [sensors, setSensors]     = useState(DEFAULT_VALUES);
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [incidents, setIncidents] = useState([]);
  const [uptime, setUptime]       = useState(0);
  const [connected, setConnected] = useState(null);

  // Uptime counter
  useEffect(() => {
    const t = setInterval(() => setUptime(u => u + 1), 1000);
    return () => clearInterval(t);
  }, []);

  // Log high-threshold sensor hits
  const pushIncidents = useCallback((vals) => {
    const now     = fmtTime();
    const entries = SENSORS
      .filter(s => isHigh(vals[s.key]))
      .map(s => ({ time: now, sensor: s.targetGas, value: Number(vals[s.key]).toFixed(0) }));
    if (entries.length) setIncidents(prev => [...prev, ...entries].slice(-60));
  }, []);

  // Auto polling
  const pollRef = useRef(null);

  const fetchRandom = useCallback(async () => {
    try {
      const res  = await fetch(`${API_BASE}/api/random`);
      const data = await res.json();
      setResult(data);
      setSensors(data.sensors);
      setConnected(true);
      pushIncidents(data.sensors);
    } catch {
      setConnected(false);
    }
  }, [pushIncidents]);

  useEffect(() => {
    if (mode !== 'auto') { clearInterval(pollRef.current); return; }
    setConnected(null);
    fetchRandom();
    pollRef.current = setInterval(fetchRandom, POLL_MS);
    return () => clearInterval(pollRef.current);
  }, [mode, fetchRandom]);

  // Manual detect
  const handleDetect = async () => {
    setLoading(true);
    try {
      const res  = await fetch(`${API_BASE}/api/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(sensors),
      });
      const data = await res.json();
      setResult(data);
      setConnected(true);
      pushIncidents(sensors);
    } catch {
      setConnected(false);
    } finally {
      setLoading(false);
    }
  };

  const switchMode = (m) => { setMode(m); setResult(null); };
  const applyPreset = (p) => { setSensors(PRESETS[p]); setResult(null); };
  const changeSensor = (k, v) => setSensors(s => ({ ...s, [k]: v }));

  return (
    <div className="flex h-screen bg-slate-50 font-sans overflow-hidden">

      <Sidebar
        mode={mode}
        connected={connected}
        loading={loading}
        onModeSwitch={switchMode}
        onPreset={applyPreset}
        onDetect={handleDetect}
      />

      {/* ── Main content ────────────────────────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* Header */}
        <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-4">
            <Link
              to="/"
              className="flex items-center gap-1.5 text-slate-400 hover:text-blue-600 transition-colors text-xs font-medium"
            >
              <ArrowLeft size={13} />
              Back
            </Link>
            <div className="h-4 w-px bg-slate-200" />
            <div>
              <h1 className="text-lg font-bold text-slate-900 leading-none">Atmospheric Intelligence</h1>
              <p className="text-xs text-slate-500 mt-0.5">
                {mode === 'auto' ? 'Real-Time Sensor Monitoring' : 'Manual Simulation Active'}
                {' '}·{' '}Unit GS-8821
              </p>
            </div>
          </div>
          <div className="flex items-center gap-5 text-xs">
            <div className="flex items-center gap-1.5 text-slate-500">
              <Clock size={12} className="text-slate-400" />
              <span className="font-mono font-semibold text-slate-700">UPTIME: {fmtUptime(uptime)}</span>
            </div>
            <div className="h-4 w-px bg-slate-200" />
            <div className="flex items-center gap-1.5">
              <Activity size={12} className={mode === 'auto' ? 'text-blue-500' : 'text-amber-500'} />
              <span className={`font-semibold ${mode === 'auto' ? 'text-blue-600' : 'text-amber-600'}`}>
                {mode === 'auto' ? '0.5 HZ FEED' : 'MANUAL INPUT'}
              </span>
            </div>
          </div>
        </header>

        {/* Result banner */}
        <ResultBanner result={result} mode={mode} loading={loading} />

        {/* Sensor table + Incident log */}
        <div className="flex flex-1 overflow-hidden mt-5 mx-6 mb-5 bg-white border border-slate-200 rounded-xl shadow-sm">

          <div className="flex-1 overflow-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 sticky top-0">
                  <th className="text-left px-5 py-3 text-[11px] font-semibold text-slate-400 tracking-widest uppercase">Sensor</th>
                  <th className="text-left px-4 py-3 text-[11px] font-semibold text-slate-400 tracking-widest uppercase">Target Gas</th>
                  <th className="text-left px-4 py-3 text-[11px] font-semibold text-slate-400 tracking-widest uppercase">
                    {mode === 'auto' ? 'Concentration' : 'Set Value (PPM)'}
                  </th>
                  <th className="text-left px-4 py-3 text-[11px] font-semibold text-slate-400 tracking-widest uppercase">Level Bar</th>
                  <th className="text-left px-5 py-3 text-[11px] font-semibold text-slate-400 tracking-widest uppercase">Status</th>
                </tr>
              </thead>
              <tbody>
                {SENSORS.map(sensor =>
                  mode === 'auto' ? (
                    <AutoRow
                      key={sensor.key}
                      sensor={sensor}
                      value={typeof sensors[sensor.key] === 'number' ? sensors[sensor.key] : 0}
                    />
                  ) : (
                    <ManualRow
                      key={sensor.key}
                      sensor={sensor}
                      value={sensors[sensor.key]}
                      onChange={(v) => changeSensor(sensor.key, v)}
                    />
                  )
                )}
              </tbody>
            </table>
          </div>

          <IncidentLog entries={incidents} />
        </div>
      </div>
    </div>
  );
}
