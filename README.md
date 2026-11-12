# GridPilot — AgentX Smart Message Gate

### Predict. Prioritise. Power the Future.

**Multi-Agent Intelligence for Smarter Grid Operations**

**Live Demo:** https://gridpilot-m2qc.onrender.com  
**GitHub Repository:** https://github.com/pratikpanda2006/GridPilot

---

## Overview

GridPilot is an intelligent, multi-agent decision-support system designed to improve electricity distribution and grid operations.

At its core is **AgentX — Smart Message Gate**, which operates in front of DISCOM power procurement and inspection queues.

Traditional grid monitoring systems generate multiple alerts, predictions, and recommendations. However, not every message requires equal attention. Excessive alerts can overwhelm operators, while critical operational signals may go unnoticed.

AgentX addresses this challenge by coordinating multiple specialised agents, evaluating their recommendations through a machine learning-based decision gate, and intelligently routing messages according to their potential operational value.

**Objective:** Reduce decision noise, prioritise meaningful operational signals, and support more efficient grid management.

---

## System Architecture

AgentX follows a multi-agent and machine learning architecture consisting of five primary components.

### 1. Multi-Agent Intelligence Layer

Four specialised agents independently analyse grid conditions and generate structured recommendations.

| Agent | Responsibility |
|---|---|
| Fixed Limit Agent | Detects threshold-based electricity consumption violations. |
| Weather Adjusted Agent | Evaluates consumption against weather-related conditions. |
| Peer Agent | Identifies unusual consumption by comparing similar buildings. |
| Trend Agent | Analyses historical consumption patterns and identifies emerging trends. |

Each agent generates a structured `AgentMessage` containing relevant findings, confidence scores, signals, and recommendations.

### 2. Orchestration Layer

The orchestrator coordinates the execution and communication of all four agents.

Its responsibilities include:

- Collecting messages from individual agents.
- Validating message structures.
- Filtering invalid building identifiers.
- Handling empty and stale messages.
- Preparing candidate messages for evaluation by the decision gate.

### 3. Machine Learning Decision Gate

AgentX uses a LightGBM classification model to evaluate the potential operational value of incoming agent messages.

The model processes engineered features and predicts three classes:

- **Beneficial:** Messages containing useful operational intelligence.
- **Harmful:** Messages that may lead to misleading or counterproductive decisions.
- **Negligible:** Messages with limited operational value.

The gate uses the model's class probabilities to support downstream routing decisions.

### 4. Intelligent Routing Layer

Based on the evaluation, messages are assigned one of four routing outcomes.

| Decision | Description |
|---|---|
| ACCEPT | Accept and prioritise useful messages. |
| IGNORE | Discard messages with negligible operational value. |
| DISCOUNT | Reduce the priority of less useful or potentially harmful signals. |
| VERIFY | Forward uncertain or important cases for human review. |

The system also generates explanations for routing decisions, identifying relevant conditions, contributing signals, and human approval requirements.

### 5. Human-in-the-Loop Verification

AgentX incorporates human oversight into its decision-making workflow.

Messages requiring verification are presented to an operator, who can review, approve, or reject recommendations.

This ensures that important operational decisions retain an appropriate level of human supervision.

---

## Dataset

The project uses the **Building Data Genome Project 2 (BDG2)** dataset.

The Robin site is used in the current implementation.

| Dataset Attribute | Details |
|---|---|
| Dataset | Building Data Genome Project 2 |
| Site | Robin |
| Number of Buildings | 51 |
| Time Period | January 2016 – December 2017 |
| Data Frequency | Hourly |
| Total Observations | 17,544 |

The historical electricity consumption data provides the foundation for analysing building-level consumption patterns and generating operational intelligence.

---

## Demo Features

The GridPilot dashboard provides an interactive demonstration of the AgentX workflow.

Key features include:

- **Demo Mode:** A curated sequence of scenarios demonstrating different operational conditions.
- **Operator Mode:** An interface for exploring agent outputs and routing decisions.
- **Multi-Agent Visibility:** Displays individual agent findings, confidence scores, signals, and recommendations.
- **Machine Learning Gate Visualisation:** Displays beneficial, harmful, and negligible probability scores.
- **Decision Explanations:** Explains why a message was accepted, ignored, discounted, or forwarded for verification.
- **Human Verification:** Allows operators to approve or reject messages requiring review.
- **Verification Status Tracking:** Displays pending, approved, and rejected verification states.

The dashboard maintains a distinction between the machine learning model's predictions and the final routing decision.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | LightGBM |
| Data Processing | Pandas, NumPy |
| Multi-Agent Framework | Custom Python Agents |
| Backend | Python API |
| Frontend | HTML, CSS, JavaScript |
| Dataset | Building Data Genome Project 2 |
| Version Control | Git, GitHub |
| Deployment | Render |

---

## Project Structure

```text
GridPilot/
│
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── base.py
│   │   │   ├── fixed_limit.py
│   │   │   ├── weather_adjusted.py
│   │   │   ├── peer.py
│   │   │   └── trend.py
│   │   │
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── features.py
│   │   ├── gate.py
│   │   ├── labelling.py
│   │   ├── messages.py
│   │   ├── orchestrator.py
│   │   ├── settings_store.py
│   │   └── main.py
│   │
│   ├── artifacts/
│   │   ├── gate_model.txt
│   │   └── gate_meta.json
│   │
│   ├── scripts/
│   │   ├── train_gate.py
│   │   └── diagnostic scripts
│   │
│   └── static/
│       └── index.html
│
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: This structure highlights the principal backend components. Additional scripts and project files are maintained in the repository.*

---

## Installation and Local Setup

### Prerequisites

- Python 3.x
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/pratikpanda2006/GridPilot.git
cd GridPilot
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Start the Backend

```bash
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8099
```

### 5. Access the Application

Open the following address in your browser:

```text
http://localhost:8099/
```

**Note:** Ensure that the required dataset files and existing model artifacts are available in their configured locations before starting the application.

---

## Deployment

GridPilot is deployed using Render.

**Live Application:** https://gridpilot-m2qc.onrender.com

The deployed application provides access to the interactive AgentX demonstration.

---

## Future Scope

The project can be extended through the following enhancements:

1. **Large Language Model Integration:** Incorporate LLMs to translate complex grid signals into natural-language explanations and assist operators in understanding recommendations.
2. **Real-Time Grid Integration:** Connect the system to live smart-meter data and electricity monitoring infrastructure.
3. **Advanced Forecasting:** Extend consumption forecasting and anomaly detection capabilities.
4. **Adaptive Agent Intelligence:** Improve agent recommendations through historical outcomes and operator feedback.
5. **Explainable AI:** Introduce deeper explanations of machine learning predictions and operational decisions.
6. **Scalability:** Extend the framework to support additional DISCOMs, buildings, and geographical regions.

---

## Why AgentX?

AgentX goes beyond a standalone machine learning prediction model by integrating multiple intelligent components into a unified decision-support pipeline.

The system combines:

- Multiple specialised analytical agents.
- Centralised orchestration and message validation.
- Machine learning-based message evaluation.
- Intelligent decision routing.
- Human-in-the-loop verification.

By combining these capabilities, AgentX aims to help electricity grid operators reduce information overload and focus on operationally significant signals.

---

## Project Information

**Project Name:** GridPilot  
**Core System:** AgentX — Smart Message Gate

**Tagline:** Predict. Prioritise. Power the Future.

Developed to explore the application of multi-agent intelligence and machine learning in building smarter, more efficient, and more reliable electricity grid operations.
