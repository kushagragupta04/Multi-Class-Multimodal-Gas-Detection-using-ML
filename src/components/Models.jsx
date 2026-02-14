import React from 'react';
import { Section } from './Section';
import { BarChart, BrainCircuit, Network, Zap } from 'lucide-react';
import { motion } from 'framer-motion';

export function Models() {
    const models = [
        {
            id: "rf",
            name: "Random Forest",
            icon: <Network />,
            accuracy: "96.93%",
            desc: "An ensemble learning method constructing multiple decision trees. It captures non-linear relationships effectively.",
            confusionMatrixImg: "/assets/RandomForestConfusionMatrix.png",
            color: "text-green-400",
            bg: "bg-green-500/10"
        },
        {
            id: "lr",
            name: "Logistic Regression",
            icon: <BarChart />,
            accuracy: "83.00%",
            desc: "A statistical linear model using a logistic function. Provides a baseline but struggles with complex non-linear boundaries.",
            confusionMatrixImg: "/assets/LogisticRegressionConfusionMatrix.png",
            color: "text-yellow-400",
            bg: "bg-yellow-500/10"
        },
        {
            id: "knn",
            name: "K-Nearest Neighbors",
            icon: <Zap />,
            accuracy: "97.08%",
            desc: "A non-parametric metric-based instance learning method. Showed excellent performance on this dataset.",
            best: true,
            confusionMatrixImg: "/assets/KNearestConfusionMatrix.png",
            color: "text-purple-400",
            bg: "bg-purple-500/10"
        },
        {
            id: "svm",
            name: "Support Vector Machine",
            icon: <BrainCircuit />,
            accuracy: "91.00%",
            desc: "Finds the optimal hyperplane for class separation. Effective but slightly outperformed by KNN and Random Forest here.",
            confusionMatrixImg: "/assets/SVMConfusionMatrix.png",
            color: "text-blue-400",
            bg: "bg-blue-500/10"
        }
    ];

    return (
        <Section id="models">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 text-center">Machine Learning Models</h2>
            <p className="text-center text-slate-400 mb-12 max-w-2xl mx-auto">
                We experimented with several algorithms to find the best fit for our sensor data.
                Below are the detailed results and confusion matrices for each model.
            </p>

            <div className="grid lg:grid-cols-2 gap-8">
                {models.map((model) => (
                    <motion.div
                        key={model.id}
                        className={`rounded-3xl border ${model.best ? 'border-purple-500/50 shadow-lg shadow-purple-500/10' : 'border-slate-800'} bg-slate-900/50 overflow-hidden flex flex-col`}
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
                                {model.best && <span className="text-xs bg-purple-500 text-white px-3 py-1 rounded-full uppercase tracking-wider font-medium">Best Model</span>}
                            </h3>
                            <p className="text-slate-400 text-sm leading-relaxed">
                                {model.desc}
                            </p>
                        </div>

                        {/* Matrix Image Section - Fitting 519x416 approx ratio */}
                        <div className="p-6 bg-slate-950/30 flex-1 flex flex-col items-center justify-center">
                            <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 w-full text-center">Confusion Matrix</h4>

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
                                {/* Tooltip hint for users if image is missing */}
                                <div className="absolute top-2 right-2 bg-black/70 text-white text-[10px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                                    {model.confusionMatrixImg}
                                </div>
                            </div>
                        </div>
                    </motion.div>
                ))}
            </div>
        </Section>
    );
}
