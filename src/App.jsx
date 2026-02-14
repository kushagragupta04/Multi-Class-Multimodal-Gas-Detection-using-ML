import React from 'react';
import { Hero } from './components/Hero';
import { ProblemStatement } from './components/ProblemStatement';
import { ProjectScope } from './components/ProjectScope';
import { Dataset } from './components/Dataset';
import { Workflow } from './components/Workflow';
import { Models } from './components/Models';
import { Results } from './components/Results';
import { Credits } from './components/Credits';
import { Footer } from './components/Footer';

function App() {
    return (
        <div className="min-h-screen bg-slate-950 text-slate-50 selection:bg-blue-500/30 font-sans">
            <div className="fixed inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none z-50 mix-blend-overlay"></div>

            <Hero />
            <ProblemStatement />
            <ProjectScope />
            <Dataset />
            <Workflow />
            <Models />
            <Results />

            <section className="py-20 text-center px-6">
                <div className="max-w-3xl mx-auto p-10 rounded-3xl bg-gradient-to-b from-slate-900 to-slate-950 border border-slate-800">
                    <h2 className="text-2xl font-bold mb-4">Conclusion</h2>
                    <p className="text-slate-400 mb-8">
                        This project successfully demonstrates that machine learning models can effectively classify gases using sensor signal patterns, significantly improving the reliability of safety systems compared to traditional threshold-based detectors.
                    </p>
                </div>
            </section>

            <Credits />

            <Footer />
        </div>
    );
}

export default App;
