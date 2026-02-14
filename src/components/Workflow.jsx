import React from 'react';
import { Section } from './Section';
import { ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';

export function Workflow() {
    const steps = [
        {
            title: "Data Loading",
            description: "Importing the multimodal dataset containing 6,400+ samples from 8 different gas sensors (MQ2, MQ5, MQ135, etc.) to establish the raw data foundation."
        },
        {
            title: "Profiling and Cleaning",
            description: "Analyzing statistical properties, checking for missing values, and removing noise or outliers to ensure data quality and reliability."
        },
        {
            title: "Feature Engineering",
            description: "Transforming raw sensor voltage readings into meaningful features, potentially including normalization or creating ratio-based features for better separability."
        },
        {
            title: "Train-Test Split",
            description: "Dividing the dataset into training (80%) and testing (20%) sets to evaluate the models on unseen data and prevent overfitting."
        },
        {
            title: "Scaling",
            description: "Standardizing features using StandardScaler to ensure all sensor inputs contribute equally to the model's decision boundary."
        },
        {
            title: "Model Training",
            description: "Fitting multiple algorithms (Random Forest Classifier, Logistic Regression, K-Nearest Neighbors, Support Vector Machine) on the training data to learn the complex patterns mapping sensors to gas types."
        },
        {
            title: "Evaluation",
            description: "Assessing performance using accuracy, confusion matrices, and classification reports to identify the optimal model."
        }
    ];

    return (
        <Section id="workflow">
            <div className="text-center mb-16">
                <h2 className="text-3xl md:text-4xl font-bold mb-4">Implementation Workflow</h2>
                <p className="text-slate-400">A detailed breakdown of our machine learning pipeline.</p>
            </div>

            <div className="relative max-w-5xl mx-auto">
                {/* Central timeline line */}
                <div className="hidden md:block absolute left-[50%] top-0 bottom-0 w-0.5 bg-slate-800 -translate-x-1/2" />

                <div className="space-y-12 relative">
                    {steps.map((step, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true, margin: "-50px" }}
                            transition={{ delay: index * 0.1 }}
                            className={`flex flex-col md:flex-row items-center gap-8 ${index % 2 === 0 ? 'md:flex-row-reverse' : ''}`}
                        >
                            {/* Content Card */}
                            <div className="flex-1 w-full relative group">
                                <div className="p-8 rounded-3xl bg-slate-900/80 border border-slate-800 hover:border-blue-500/30 transition-all relative overflow-hidden">
                                    <h3 className="text-xl font-bold text-slate-100 mb-3">{step.title}</h3>
                                    <p className="text-slate-400 text-sm leading-relaxed border-t border-slate-700/50 pt-3">
                                        {step.description}
                                    </p>
                                </div>

                                {/* Connection Arrow */}
                                <div className={`hidden md:block absolute top-1/2 ${index % 2 === 0 ? '-left-12' : '-right-12'} -translate-y-1/2 text-slate-700`}>
                                    {index < steps.length - 1 && (
                                        index % 2 === 0 ? <ArrowRight className="rotate-180" /> : <ArrowRight />
                                    )}
                                </div>
                            </div>

                            {/* Center Node */}
                            <div className="relative z-10 flex-shrink-0 w-12 h-12 rounded-full bg-slate-950 border-4 border-slate-800 flex items-center justify-center text-slate-300 font-bold text-sm shadow-xl">
                                {index + 1}
                            </div>

                            {/* Spacer for alternate side */}
                            <div className="flex-1 hidden md:block" />
                        </motion.div>
                    ))}
                </div>
            </div>
        </Section>
    );
}
