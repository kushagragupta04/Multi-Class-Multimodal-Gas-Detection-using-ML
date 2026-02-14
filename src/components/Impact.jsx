import React from 'react';
import { Section } from './Section';
import { Zap, ShieldCheck, TrendingUp, Network, Building2 } from 'lucide-react';

export function Impact() {
    const impacts = [
        {
            icon: <Zap className="text-yellow-400" />,
            text: "Faster and automated gas identification, reducing dependence on manual inspection"
        },
        {
            icon: <ShieldCheck className="text-emerald-400" />,
            text: "Early detection of hazardous gases, helping prevent accidents, explosions, and health risks"
        },
        {
            icon: <TrendingUp className="text-blue-400" />,
            text: "Improved accuracy over conventional threshold-based detectors, especially in multi-gas environments"
        },
        {
            icon: <Network className="text-purple-400" />,
            text: "Scalable monitoring solutions that can be integrated with IoT devices for real-time safety systems"
        },
        {
            icon: <Building2 className="text-orange-400" />,
            text: "Support for insurance, industrial compliance, and emergency response, where timely detection is critical"
        }
    ];

    return (
        <Section id="impact">
            <div className="bg-slate-900/40 rounded-3xl p-8 md:p-12 border border-slate-800">
                <div className="text-center mb-10">
                    <h2 className="text-3xl md:text-4xl font-bold mb-6">Impact of the Project</h2>
                    <p className="text-slate-400 max-w-3xl mx-auto leading-relaxed">
                        This project has significant real-world impact in the domain of safety and industrial monitoring. Gas leakage and hazardous emissions are major risks in environments such as chemical plants, laboratories, mines, and residential areas. Traditional detection systems are often limited, manual, or prone to error.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-10">
                    {impacts.map((item, i) => (
                        <div key={i} className="p-6 rounded-2xl bg-slate-950/50 border border-slate-800 hover:border-slate-700 transition-colors flex flex-col gap-4">
                            <div className="w-10 h-10 rounded-lg bg-slate-900 flex items-center justify-center">
                                {item.icon}
                            </div>
                            <p className="text-slate-300 text-sm leading-relaxed">
                                {item.text}
                            </p>
                        </div>
                    ))}
                </div>

                <div className="text-center border-t border-slate-800/50 pt-8">
                    <p className="text-slate-400 font-medium max-w-2xl mx-auto">
                        Overall, this project demonstrates how AI can enhance environmental monitoring and industrial safety through intelligent classification of gases.
                    </p>
                </div>
            </div>
        </Section>
    );
}
