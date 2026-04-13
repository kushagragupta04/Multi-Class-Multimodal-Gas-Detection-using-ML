import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowRight, Database, Activity, Sparkles, LayoutDashboard } from 'lucide-react';

export function Hero() {
    return (
        <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
            {/* Background Gradients */}
            <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl -z-10 animate-pulse" />
            <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl -z-10 animate-pulse delay-1000" />
            <div className="absolute top-1/3 right-1/3 w-80 h-80 bg-violet-500/10 rounded-full blur-3xl -z-10 animate-pulse delay-500" />

            <div className="text-center max-w-4xl mx-auto px-6">
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5 }}
                    className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-gradient-to-r from-violet-500/10 to-fuchsia-500/10 border border-violet-500/30 text-sm text-violet-300 mb-6 backdrop-blur-sm"
                >
                    <Sparkles size={15} />
                    <span>Gen AI · Transformer-Powered · Industrial Safety</span>
                    <Activity size={15} className="text-fuchsia-400" />
                </motion.div>

                <motion.h1
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="text-5xl md:text-7xl font-bold mb-6"
                >
                    <span className="bg-gradient-to-r from-white via-slate-200 to-slate-400 text-transparent bg-clip-text">
                        Intelligent Gas
                    </span>
                    <br />
                    <span className="bg-gradient-to-r from-violet-400 via-fuchsia-400 to-pink-400 text-transparent bg-clip-text">
                        Detection &amp; Classification
                    </span>
                </motion.h1>

                <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="text-lg md:text-xl text-slate-400 mb-4 max-w-2xl mx-auto"
                >
                    A multi-class, multimodal gas detection system powered by a custom
                    <span className="text-violet-400 font-semibold"> SensorTransformer</span> —
                    using self-attention over 7 MQ sensors to identify hazardous gases with
                    interpretable, state-of-the-art accuracy.
                </motion.p>

                {/* Stat pills */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5 }}
                    className="flex flex-wrap justify-center gap-3 mb-8"
                >
                    {[
                        { label: '6,400', sub: 'Sensor Samples' },
                        { label: '7', sub: 'MQ Gas Sensors' },
                        { label: '4', sub: 'Gas Classes' },
                        { label: '~98.5%', sub: 'Transformer Accuracy' },
                    ].map(stat => (
                        <div key={stat.label} className="px-4 py-2 rounded-xl bg-slate-800/60 border border-slate-700/50 text-center">
                            <div className="text-white font-bold text-sm">{stat.label}</div>
                            <div className="text-slate-500 text-xs">{stat.sub}</div>
                        </div>
                    ))}
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.7 }}
                    className="flex flex-col sm:flex-row gap-4 justify-center"
                >
                    <Link
                        to="/dashboard"
                        className="inline-flex items-center justify-center gap-2 px-8 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold transition-colors shadow-lg shadow-indigo-500/25 group"
                    >
                        <LayoutDashboard size={18} />
                        Live Dashboard
                        <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
                    </Link>
                    <a
                        href="https://github.com/kushagragupta04/Multi-Class-Multimodal-Gas-Detection-using-ML/blob/main/Multimodal%20Gas%20Detection%20%26%20Classification%20Dataset/Gas%20Sensors%20Measurements/Gas_Sensors_Measurements.csv"
                        className="inline-flex items-center justify-center gap-2 px-8 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-medium transition-colors shadow-lg shadow-blue-500/20 group"
                    >
                        <Database size={20} />
                        View Dataset
                        <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
                    </a>
                    <a
                        href="https://github.com/kushagragupta04/Multi-Class-Multimodal-Gas-Detection-using-ML/blob/main/Multi_Class_Multimodal_Gas_Detection_using_ML.ipynb"
                        className="inline-flex items-center justify-center gap-2 px-8 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-medium transition-colors border border-slate-700 hover:border-slate-600"
                    >
                        Classical ML Notebook
                    </a>
                    <a
                        href="https://github.com/kushagragupta04/Multi-Class-Multimodal-Gas-Detection-using-ML/blob/main/transformer_gas_detection.py"
                        className="inline-flex items-center justify-center gap-2 px-8 py-3 rounded-xl bg-violet-700 hover:bg-violet-600 text-white font-medium transition-colors border border-violet-600 shadow-lg shadow-violet-500/20 group"
                    >
                        <Sparkles size={16} />
                        Transformer Code
                        <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
                    </a>
                </motion.div>
            </div>

            {/* Scroll Indicator */}
            {/* <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1, duration: 1 }}
                className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-slate-500"
            >
                <span className="text-xs uppercase tracking-widest">Scroll to explore</span>
                <div className="w-[1px] h-12 bg-gradient-to-b from-slate-500 to-transparent" />
            </motion.div> */}
        </section>
    );
}
