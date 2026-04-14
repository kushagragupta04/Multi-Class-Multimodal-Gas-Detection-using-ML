import React from 'react';
import { Section } from './Section';
import { BarChart, BrainCircuit, Network, Zap, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';

export function Models() {
    const models = [
        {
            id: "transformer",
            name: "SensorTransformer",
            icon: <Sparkles />,
            accuracy: "~96.5%",
            desc: "A custom Transformer Encoder that treats 7 MQ sensor readings as tokens and uses multi-head self-attention to learn inter-sensor correlations. Interpretable via attention & gradient analysis.",
            best: true,
            genAI: true,
            color: "text-violet-400",
            bg: "bg-violet-500/10",
            border: "border-violet-500/50",
            shadow: "shadow-violet-500/10"
        },
        {
            id: "knn",
            name: "K-Nearest Neighbors",
            icon: <Zap />,
            accuracy: "95.08%",
            desc: "A non-parametric metric-based instance learning method. Showed excellent performance on this dataset.",
            confusionMatrixImg: "/assets/KNearestConfusionMatrix.png",
            color: "text-purple-400",
            bg: "bg-purple-500/10",
            border: "border-slate-800",
            shadow: ""
        },
        {
            id: "rf",
            name: "Random Forest",
            icon: <Network />,
            accuracy: "94.93%",
            desc: "An ensemble learning method constructing multiple decision trees. It captures non-linear relationships effectively.",
            confusionMatrixImg: "/assets/RandomForestConfusionMatrix.png",
            color: "text-green-400",
            bg: "bg-green-500/10",
            border: "border-slate-800",
            shadow: ""
        },
        {
            id: "svm",
            name: "Support Vector Machine",
            icon: <BrainCircuit />,
            accuracy: "90.00%",
            desc: "Finds the optimal hyperplane for class separation. Effective but slightly outperformed by KNN and Random Forest here.",
            confusionMatrixImg: "/assets/SVMConfusionMatrix.png",
            color: "text-blue-400",
            bg: "bg-blue-500/10",
            border: "border-slate-800",
            shadow: ""
        },
        {
            id: "lr",
            name: "Logistic Regression",
            icon: <BarChart />,
            accuracy: "83.00%",
            desc: "A statistical linear model using a logistic function. Provides a baseline but struggles with complex non-linear boundaries.",
            confusionMatrixImg: "/assets/LogisticRegressionConfusionMatrix.png",
            color: "text-yellow-400",
            bg: "bg-yellow-500/10",
            border: "border-slate-800",
            shadow: ""
        }
    ];

    const containerVariants = {
        hidden: {},
        visible: { transition: { staggerChildren: 0.1 } }
    };

    const cardVariants = {
        hidden: { opacity: 0, y: 24 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.5 } }
    };

    return (
        <Section id="models">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 text-center">
                Machine Learning Models
            </h2>
            <p className="text-center text-slate-400 mb-12 max-w-2xl mx-auto">
                We experimented with classical ML algorithms and a custom{" "}
                <span className="text-violet-400 font-semibold">Gen AI Transformer</span>{" "}
                to find the best fit for sensor data. Below are the detailed results.
            </p>

            {/* ── Transformer spotlight ── */}
            <motion.div
                variants={cardVariants}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                className="rounded-3xl border border-violet-500/50 bg-gradient-to-br from-violet-900/20 via-slate-900/60 to-slate-950/80
                           shadow-2xl shadow-violet-500/10 overflow-hidden mb-8"
            >
                <div className="p-8 md:p-10">
                    {/* Header row */}
                    <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 mb-6">
                        <div className="flex items-center gap-4">
                            <div className="w-14 h-14 rounded-2xl bg-violet-500/20 text-violet-400 flex items-center justify-center text-2xl">
                                <Sparkles size={28} />
                            </div>
                            <div>
                                <div className="flex items-center gap-2 flex-wrap">
                                    <h3 className="text-2xl font-bold">SensorTransformer</h3>
                                    <span className="text-xs bg-violet-600 text-white px-3 py-1 rounded-full uppercase tracking-wider font-semibold">
                                        Best Model
                                    </span>
                                    <span className="text-xs bg-gradient-to-r from-fuchsia-600 to-violet-600 text-white px-3 py-1 rounded-full uppercase tracking-wider font-semibold animate-pulse">
                                        ✦ Gen AI
                                    </span>
                                </div>
                                <p className="text-slate-400 text-sm mt-1">Custom Transformer Encoder — Self-Attention over MQ Sensors</p>
                            </div>
                        </div>
                        <div className="text-right shrink-0">
                            <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Test Accuracy</div>
                            <div className="text-4xl font-bold text-violet-300 font-mono">~96.5%</div>
                        </div>
                    </div>

                    {/* Architecture diagram */}
                    <div className="bg-slate-950/60 rounded-2xl p-6 border border-slate-800 mb-6 font-mono text-sm overflow-x-auto">
                        <p className="text-slate-500 text-xs uppercase tracking-widest mb-3">Architecture Pipeline</p>
                        <div className="flex flex-wrap items-center gap-2 text-center">
                            {[
                                { label: "7 MQ Sensors", sub: "Input", color: "text-slate-300 bg-slate-800" },
                                { label: "→", color: "text-slate-600", plain: true },
                                { label: "Linear\nEmbedding", sub: "7 → d=64", color: "text-blue-300 bg-blue-900/30" },
                                { label: "→", color: "text-slate-600", plain: true },
                                { label: "Transformer\nEncoder", sub: "2 layers, 4 heads", color: "text-violet-300 bg-violet-900/30" },
                                { label: "→", color: "text-slate-600", plain: true },
                                { label: "[CLS] Token\nExtraction", sub: "Global representation", color: "text-fuchsia-300 bg-fuchsia-900/30" },
                                { label: "→", color: "text-slate-600", plain: true },
                                { label: "MLP\nClassifier", sub: "4 gas classes", color: "text-emerald-300 bg-emerald-900/30" },
                            ].map((block, i) =>
                                block.plain ? (
                                    <span key={i} className={`text-xl ${block.color} hidden sm:inline`}>{block.label}</span>
                                ) : (
                                    <div key={i} className={`px-3 py-2 rounded-xl ${block.color} text-xs whitespace-pre-line`}>
                                        <div className="font-bold">{block.label}</div>
                                        <div className="opacity-60 text-[10px] mt-0.5">{block.sub}</div>
                                    </div>
                                )
                            )}
                        </div>
                    </div>

                    {/* Why Transformer grid */}
                    <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        {[
                            { title: "Self-Attention", body: "Captures global correlations between all 7 MQ sensor readings simultaneously" },
                            { title: "Interpretable", body: "Gradient attribution reveals which sensors drive each prediction" },
                            { title: "Scalable", body: "Performance grows with more data & compute; classical ML plateaus" },
                            { title: "Transfer Ready", body: "Can be fine-tuned for new gas types with minimal labelled data" },
                        ].map((feat) => (
                            <div key={feat.title} className="bg-slate-900/50 rounded-xl p-4 border border-slate-800">
                                <div className="text-violet-400 font-semibold text-sm mb-1">✦ {feat.title}</div>
                                <p className="text-slate-400 text-xs leading-relaxed">{feat.body}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </motion.div>

            {/* ── Classical models grid ── */}
            <motion.div
                variants={containerVariants}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                className="grid lg:grid-cols-2 gap-8"
            >
                {models.filter(m => !m.genAI).map((model) => (
                    <motion.div
                        key={model.id}
                        variants={cardVariants}
                        className={`rounded-3xl border ${model.border} bg-slate-900/50 overflow-hidden flex flex-col`}
                    >
                        {/* Header Section */}
                        <div className="p-8 border-b border-slate-800 bg-slate-900/30">
                            <div className="flex items-start justify-between mb-4">
                                <div className={`w-14 h-14 rounded-2xl ${model.bg} ${model.color} flex items-center justify-center`}>
                                    {model.icon}
                                </div>
                                <div className="text-right">
                                    <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Accuracy</div>
                                    <div className="text-3xl font-bold text-white font-mono">{model.accuracy}</div>
                                </div>
                            </div>

                            <h3 className="text-2xl font-bold flex items-center gap-3 mb-2">
                                {model.name}
                            </h3>
                            <p className="text-slate-400 text-sm leading-relaxed">
                                {model.desc}
                            </p>
                        </div>

                        {/* Matrix Image Section */}
                        {model.confusionMatrixImg && (
                            <div className="p-6 bg-slate-950/30 flex-1 flex flex-col items-center justify-center">
                                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 w-full text-center">
                                    Confusion Matrix
                                </h4>
                                <div className="relative group w-full max-w-[520px] rounded-xl overflow-hidden border border-slate-800 shadow-xl bg-black">
                                    <img
                                        src={model.confusionMatrixImg}
                                        alt={`Confusion Matrix for ${model.name}`}
                                        className="w-full h-auto object-cover"
                                        width="519"
                                        height="416"
                                        onError={(e) => {
                                            e.target.onerror = null;
                                            e.target.src = "https://placehold.co/519x416/1e293b/475569?text=Confusion+Matrix+Not+Found";
                                        }}
                                    />
                                    <div className="absolute top-2 right-2 bg-black/70 text-white text-[10px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                                        {model.confusionMatrixImg}
                                    </div>
                                </div>
                            </div>
                        )}
                    </motion.div>
                ))}
            </motion.div>

            {/* ── Comparison bar ── */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.3 }}
                className="mt-12 rounded-3xl border border-slate-800 bg-slate-900/50 p-8"
            >
                <h3 className="text-xl font-bold mb-6 text-center">Model Accuracy Comparison</h3>
                <div className="space-y-4">
                    {[
                        { name: "SensorTransformer (Gen AI)", acc: 96.5, color: "bg-violet-500", textColor: "text-violet-300" },
                        { name: "K-Nearest Neighbors",        acc: 95.08, color: "bg-purple-500",  textColor: "text-purple-300" },
                        { name: "Random Forest",              acc: 94.93, color: "bg-green-500",   textColor: "text-green-300" },
                        { name: "Support Vector Machine",     acc: 90.0,  color: "bg-blue-500",    textColor: "text-blue-300" },
                        { name: "Logistic Regression",        acc: 83.0,  color: "bg-yellow-500",  textColor: "text-yellow-300" },
                    ].map((row) => (
                        <div key={row.name} className="flex items-center gap-4">
                            <div className="w-52 text-right text-sm text-slate-400 shrink-0">{row.name}</div>
                            <div className="flex-1 bg-slate-800 rounded-full h-3 overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    whileInView={{ width: `${row.acc}%` }}
                                    viewport={{ once: true }}
                                    transition={{ duration: 1, ease: "easeOut" }}
                                    className={`h-full rounded-full ${row.color}`}
                                />
                            </div>
                            <div className={`w-16 text-sm font-mono font-bold ${row.textColor} shrink-0`}>
                                {row.acc}%
                            </div>
                        </div>
                    ))}
                </div>
            </motion.div>
        </Section>
    );
}
