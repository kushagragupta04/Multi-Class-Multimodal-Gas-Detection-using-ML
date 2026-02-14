import React from 'react';
import { Section } from './Section';
import { AlertTriangle, Fingerprint, ShieldAlert } from 'lucide-react';

export function ProblemStatement() {
    return (
        <Section id="problem" className="relative">
            <div className="grid md:grid-cols-2 gap-12 items-center">
                <div>
                    <h2 className="text-3xl md:text-4xl font-bold mb-6">Why Early Gas Detection Matters</h2>
                    <p className="text-slate-400 mb-6 text-lg">
                        Gas leaks pose catastrophic risks in industrial and residential settings. Traditional detection methods often fail to classify specific gas types or detect mixtures, leading to delayed responses.
                    </p>
                    <ul className="space-y-4">
                        {[
                            "Silent threat: Colorless and odorless gases are often undetected.",
                            "False alarms from generic sensors reduce trust.",
                            "Single-purpose detectors are costly and hard to maintain."
                        ].map((item, i) => (
                            <li key={i} className="flex items-start gap-3 text-slate-300">
                                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-red-500/20 text-red-400 flex items-center justify-center text-sm">!</span>
                                {item}
                            </li>
                        ))}
                    </ul>
                </div>

                <div className="grid gap-6">
                    <div className="p-6 rounded-2xl bg-slate-800/50 border border-slate-700/50 backdrop-blur-sm hover:bg-slate-800/80 transition-colors group">
                        <div className="w-12 h-12 rounded-lg bg-orange-500/20 text-orange-400 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                            <AlertTriangle size={24} />
                        </div>
                        <h3 className="text-xl font-semibold mb-2">High False Positives</h3>
                        <p className="text-slate-400">Conventional sensors struggle to distinguish between hazardous leaks and harmless background interference.</p>
                    </div>

                    <div className="p-6 rounded-2xl bg-slate-800/50 border border-slate-700/50 backdrop-blur-sm hover:bg-slate-800/80 transition-colors group translate-x-4 md:translate-x-8">
                        <div className="w-12 h-12 rounded-lg bg-blue-500/20 text-blue-400 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                            <Fingerprint size={24} />
                        </div>
                        <h3 className="text-xl font-semibold mb-2">Lack of Classification</h3>
                        <p className="text-slate-400">Knowing a gas is present isn't enough; identifying the specific type is crucial for the correct response protocol.</p>
                    </div>
                </div>
            </div>
        </Section>
    );
}
