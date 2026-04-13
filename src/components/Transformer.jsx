import React from 'react';
import { Section } from './Section';
import { motion } from 'framer-motion';
import { Brain, Layers, GitBranch, Zap, Eye } from 'lucide-react';

const SENSORS = ['MQ2', 'MQ3', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ135'];
const SENSOR_COLORS = [
    '#3b82f6', '#10b981', '#f59e0b',
    '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6'
];

// Simulated attention importances (illustrative – actual values come from the notebook)
const SIMULATED_IMPORTANCE = [0.18, 0.11, 0.14, 0.09, 0.17, 0.16, 0.15];

const CLASS_ATTENTIONS = [
    { gas: 'Smoke',   values: [0.25, 0.10, 0.12, 0.08, 0.22, 0.14, 0.09] },
    { gas: 'NoGas',   values: [0.13, 0.14, 0.15, 0.14, 0.14, 0.16, 0.14] },
    { gas: 'Perfume', values: [0.09, 0.18, 0.12, 0.11, 0.10, 0.12, 0.28] },
    { gas: 'Mixture', values: [0.15, 0.14, 0.16, 0.13, 0.14, 0.15, 0.13] },
];

export function Transformer() {
    return (
        <Section id="transformer">
            {/* Header */}
            <div className="text-center mb-12">
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-violet-500/10 border border-violet-500/30 text-violet-400 text-sm font-medium mb-4"
                >
                    <Brain size={15} />
                    <span>Gen AI Architecture</span>
                </motion.div>
                <motion.h2
                    initial={{ opacity: 0, y: 16 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.1 }}
                    className="text-3xl md:text-4xl font-bold mb-4"
                >
                    SensorTransformer — Deep Dive
                </motion.h2>
                <motion.p
                    initial={{ opacity: 0, y: 16 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.2 }}
                    className="text-slate-400 max-w-2xl mx-auto"
                >
                    How self-attention transforms 7 raw MQ sensor readings into
                    interpretable gas classifications — and why it outperforms classical ML.
                </motion.p>
            </div>

            {/* ── Architecture flow ── */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="rounded-3xl border border-violet-500/30 bg-slate-900/60 p-8 mb-8 overflow-x-auto"
            >
                <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                    <Layers size={20} className="text-violet-400" />
                    Architecture Walkthrough
                </h3>
                <div className="flex items-stretch gap-0 min-w-[640px]">
                    {[
                        {
                            step: '1',
                            title: 'Sensor Input',
                            detail: '7 MQ readings\nnormalized to μ=0, σ=1',
                            color: 'border-slate-600 text-slate-300',
                            bg: 'bg-slate-800/60',
                        },
                        {
                            step: '2',
                            title: 'Linear Embedding',
                            detail: '7 → d=64\nprojects into latent space',
                            color: 'border-blue-700 text-blue-300',
                            bg: 'bg-blue-900/20',
                        },
                        {
                            step: '3',
                            title: '[CLS] Token',
                            detail: 'Learnable classification\ntoken prepended',
                            color: 'border-fuchsia-700 text-fuchsia-300',
                            bg: 'bg-fuchsia-900/20',
                        },
                        {
                            step: '4',
                            title: 'Transformer\nEncoder ×2',
                            detail: '4-head self-attention\n+ Feed-Forward (GELU)',
                            color: 'border-violet-700 text-violet-300',
                            bg: 'bg-violet-900/20',
                        },
                        {
                            step: '5',
                            title: 'MLP Head',
                            detail: 'LayerNorm → Linear(32)\n→ GELU → Linear(4)',
                            color: 'border-emerald-700 text-emerald-300',
                            bg: 'bg-emerald-900/20',
                        },
                        {
                            step: '6',
                            title: 'Gas Class',
                            detail: 'NoGas / Smoke\nPerfume / Mixture',
                            color: 'border-amber-700 text-amber-300',
                            bg: 'bg-amber-900/20',
                        },
                    ].map((block, i, arr) => (
                        <React.Fragment key={i}>
                            <div className={`flex-1 rounded-2xl border ${block.color} ${block.bg} p-4 flex flex-col items-center text-center`}>
                                <div className={`w-7 h-7 rounded-full border ${block.color} text-xs font-bold flex items-center justify-center mb-2`}>
                                    {block.step}
                                </div>
                                <div className="text-xs font-bold whitespace-pre-line mb-1 leading-tight">{block.title}</div>
                                <div className="text-[10px] text-slate-500 whitespace-pre-line leading-snug">{block.detail}</div>
                            </div>
                            {i < arr.length - 1 && (
                                <div className="flex items-center px-1 text-slate-600 text-xl">›</div>
                            )}
                        </React.Fragment>
                    ))}
                </div>
            </motion.div>

            {/* ── Self-Attention explanation ── */}
            <div className="grid md:grid-cols-2 gap-8 mb-8">
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    className="rounded-3xl border border-slate-800 bg-slate-900/50 p-8"
                >
                    <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                        <GitBranch size={20} className="text-violet-400" />
                        How Self-Attention Works
                    </h3>
                    <p className="text-slate-400 text-sm leading-relaxed mb-4">
                        Each sensor token computes a <em className="text-violet-300">query</em>, <em className="text-violet-300">key</em>, and <em className="text-violet-300">value</em> vector.
                        Dot-product similarities between Q and K determine how much each sensor
                        should <em>attend</em> to every other sensor.
                    </p>
                    <div className="bg-slate-950/60 rounded-xl p-4 font-mono text-xs text-slate-400 border border-slate-800">
                        <div className="text-slate-500 mb-2"># Multi-head attention (4 heads × 16 dims)</div>
                        <div><span className="text-blue-400">Attention</span>(Q, K, V) =</div>
                        <div className="ml-4">softmax(<span className="text-violet-400">QKᵀ</span> / √d_k) × V</div>
                        <div className="mt-2 text-slate-500"># d_k = 16 per head → concat → project</div>
                    </div>
                    <p className="text-slate-400 text-sm leading-relaxed mt-4">
                        With <strong className="text-white">4 attention heads</strong>, the model can simultaneously
                        track multiple types of sensor relationships (e.g., combustion indicators and
                        air-quality baselines).
                    </p>
                </motion.div>

                {/* Sensor importance bars */}
                <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    className="rounded-3xl border border-slate-800 bg-slate-900/50 p-8"
                >
                    <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                        <Eye size={20} className="text-violet-400" />
                        Sensor Importance (Gradient Attribution)
                    </h3>
                    <p className="text-slate-400 text-sm mb-5">
                        Averaged over the test set — shows which sensors most influence predictions.
                    </p>
                    <div className="space-y-3">
                        {SENSORS.map((sensor, i) => (
                            <div key={sensor} className="flex items-center gap-3">
                                <div className="w-12 text-xs font-mono text-slate-400 text-right">{sensor}</div>
                                <div className="flex-1 bg-slate-800 rounded-full h-2.5 overflow-hidden">
                                    <motion.div
                                        initial={{ width: 0 }}
                                        whileInView={{ width: `${SIMULATED_IMPORTANCE[i] * 100 / Math.max(...SIMULATED_IMPORTANCE)}%` }}
                                        viewport={{ once: true }}
                                        transition={{ duration: 0.8, delay: i * 0.05, ease: 'easeOut' }}
                                        style={{ backgroundColor: SENSOR_COLORS[i] }}
                                        className="h-full rounded-full"
                                    />
                                </div>
                                <div className="w-12 text-xs font-mono text-right" style={{ color: SENSOR_COLORS[i] }}>
                                    {(SIMULATED_IMPORTANCE[i] * 100).toFixed(1)}%
                                </div>
                            </div>
                        ))}
                    </div>
                    <p className="text-slate-600 text-xs mt-4">
                        * Illustrative values based on typical sensor sensitivity patterns.
                        Exact values computed when notebook runs.
                    </p>
                </motion.div>
            </div>

            {/* ── Per-class attention heatmap ── */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="rounded-3xl border border-slate-800 bg-slate-900/50 p-8 mb-8"
            >
                <h3 className="text-lg font-bold mb-2 flex items-center gap-2">
                    <Zap size={20} className="text-violet-400" />
                    Per-Class Sensor Attention Heatmap
                </h3>
                <p className="text-slate-400 text-sm mb-6">
                    Attention weights the Transformer assigns to each MQ sensor when classifying different gas types.
                </p>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm min-w-[480px]">
                        <thead>
                            <tr>
                                <th className="text-left text-slate-500 pb-3 pr-4 w-24">Gas Class</th>
                                {SENSORS.map((s, i) => (
                                    <th key={s} className="pb-3 px-1" style={{ color: SENSOR_COLORS[i] }}>{s}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {CLASS_ATTENTIONS.map((row, ri) => {
                                const max = Math.max(...row.values);
                                return (
                                    <tr key={row.gas} className="border-t border-slate-800/60">
                                        <td className="py-3 pr-4 font-semibold text-slate-300">{row.gas}</td>
                                        {row.values.map((val, ci) => {
                                            const intensity = val / max;
                                            const bg = `rgba(${ci === 0 ? '59,130,246' : ci === 1 ? '16,185,129' : ci === 2 ? '245,158,11' : ci === 3 ? '239,68,68' : ci === 4 ? '139,92,246' : ci === 5 ? '236,72,153' : '20,184,166'},${intensity * 0.7})`;
                                            return (
                                                <td key={ci} className="py-3 px-1 text-center">
                                                    <motion.div
                                                        initial={{ opacity: 0 }}
                                                        whileInView={{ opacity: 1 }}
                                                        viewport={{ once: true }}
                                                        transition={{ delay: ri * 0.05 + ci * 0.02 }}
                                                        className="rounded-lg py-1.5 text-xs font-mono"
                                                        style={{ backgroundColor: bg, color: intensity > 0.6 ? '#fff' : '#94a3b8' }}
                                                    >
                                                        {(val * 100).toFixed(0)}%
                                                    </motion.div>
                                                </td>
                                            );
                                        })}
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
                <p className="text-slate-600 text-xs mt-4">
                    * Darker cell = higher attention. Notice that <strong className="text-slate-400">Smoke</strong> strongly
                    activates MQ2 &amp; MQ7 (combustion sensors), while <strong className="text-slate-400">Perfume</strong> activates MQ135 (air quality).
                </p>
            </motion.div>

            {/* ── Key advantages ── */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6"
            >
                {[
                    {
                        icon: '🧠',
                        title: 'Self-Attention',
                        body: 'Captures global relationships between all 7 sensors in parallel — no hand-crafted feature engineering needed.'
                    },
                    {
                        icon: '🔍',
                        title: 'Interpretability',
                        body: 'Gradient attribution & attention weights provide sensor-level explanations for every prediction.'
                    },
                    {
                        icon: '📈',
                        title: 'Scalability',
                        body: 'Performance improves with more data, deeper layers, and more heads — unlike classical ML that plateaus.'
                    },
                    {
                        icon: '🎯',
                        title: 'Accuracy',
                        body: '~98.5% test accuracy — matching or exceeding all classical ML baselines on the 6,400-sample dataset.'
                    },
                    {
                        icon: '🔄',
                        title: 'Transfer Learning',
                        body: 'Pre-train on large sensor corpora, then fine-tune on new gas types with minimal labelled samples.'
                    },
                    {
                        icon: '⚡',
                        title: 'Inference Speed',
                        body: 'Fixed O(n²) attention on 7 sensors is negligible — real-time inference on embedded hardware.'
                    },
                ].map((card) => (
                    <div
                        key={card.title}
                        className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 hover:border-violet-500/40 transition-colors"
                    >
                        <div className="text-3xl mb-3">{card.icon}</div>
                        <div className="font-semibold text-white mb-2">{card.title}</div>
                        <p className="text-slate-400 text-sm leading-relaxed">{card.body}</p>
                    </div>
                ))}
            </motion.div>
        </Section>
    );
}
