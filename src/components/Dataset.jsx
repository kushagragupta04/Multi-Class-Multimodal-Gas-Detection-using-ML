import React from 'react';
import { Section } from './Section';
import { Database, FileSpreadsheet, Users } from 'lucide-react';

export function Dataset() {
    return (
        <Section id="dataset" className="bg-slate-900/30">
            <div className="text-center mb-12">
                <h2 className="text-3xl md:text-4xl font-bold mb-4">Dataset Overview</h2>
                <p className="text-slate-400">
                    The foundation of our model is the Kaggle Multimodal Gas Detection Dataset, providing diverse sensor readings for robust training.
                </p>
            </div>

            <div className="bg-slate-800/40 rounded-3xl border border-slate-700/50 overflow-hidden backdrop-blur-sm">
                <div className="grid md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-slate-700/50">
                    {[
                        { label: "Total Samples", value: "6,400+", icon: <Database /> },
                        { label: "Features", value: "7 Sensors", icon: <FileSpreadsheet /> },
                        { label: "Classes", value: "4 Gas Types", icon: <Users /> }
                    ].map((stat, i) => (
                        <div key={i} className="p-8 flex flex-col items-center text-center hover:bg-slate-700/20 transition-colors">
                            <div className="mb-4 text-blue-400 bg-blue-500/10 p-3 rounded-full">{stat.icon}</div>
                            <div className="text-3xl font-bold text-white mb-1">{stat.value}</div>
                            <div className="text-sm text-slate-400 uppercase tracking-wider">{stat.label}</div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="mt-12 overflow-x-auto">
                <table className="w-full text-left text-sm text-slate-400">
                    <thead className="text-xs uppercase bg-slate-800/50 text-slate-300">
                        <tr>
                            <th className="px-6 py-4 rounded-tl-lg">Feature Type</th>
                            <th className="px-6 py-4">Description</th>
                            <th className="px-6 py-4 rounded-tr-lg">Example Unit</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">MQ2 Sensor</td>
                            <td className="px-6 py-4">Detects Smoke, LPG, Methane</td>
                            <td className="px-6 py-4 font-mono text-xs">Voltage (V)</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">MQ3 Sensor</td>
                            <td className="px-6 py-4">Detects Alcohol Vapors (Perfume, Ethanol)</td>
                            <td className="px-6 py-4 font-mono text-xs">Voltage (V)</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">MQ5 Sensor</td>
                            <td className="px-6 py-4">Sensitive to Natural Gas, LPG</td>
                            <td className="px-6 py-4 font-mono text-xs">Voltage (V)</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">MQ6 Sensor</td>
                            <td className="px-6 py-4">Detects LPG, Butane, Propane</td>
                            <td className="px-6 py-4 font-mono text-xs">Voltage (V)</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">MQ7 Sensor</td>
                            <td className="px-6 py-4">Detects Carbon Monoxide (CO)</td>
                            <td className="px-6 py-4 font-mono text-xs">Voltage (V)</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">MQ8 Sensor</td>
                            <td className="px-6 py-4">Detects Hydrogen Gas (H₂)</td>
                            <td className="px-6 py-4 font-mono text-xs">Voltage (V)</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">MQ135 Sensor</td>
                            <td className="px-6 py-4">
                                Air Quality Monitoring (NH₃, Benzene, Smoke, CO₂)
                            </td>
                            <td className="px-6 py-4 font-mono text-xs">PPM</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Target Label</td>
                            <td className="px-6 py-4">Categorical Gas Class (Smoke, Perfume, Mixture, No Gas)</td>
                            <td className="px-6 py-4 font-mono text-xs">One-Hot Encoded</td>
                        </tr>
                    </tbody>

                </table>
            </div>
        </Section>
    );
}
