# ITI-MCR: An Effective Infrequent Trace Identification Method Based on Multi-perspective Conformance Rules

This repository contains the official source code implementation for the paper *"An Effective Infrequent Trace Identification Method Based on Multi-perspective Conformance Rules"*.



In process mining, infrequent traces are often misclassified as noise, causing information loss and model distortion. To address this issue, an identification method based on multi-perspective conformance rules (ITI-MCR) is proposed. The method extracts rules from high-frequency traces and evaluates infrequent traces from multiple perspectives. Finally, the accurate identification of effective infrequent traces with business value is achieved based on a weighted combination of the conformance evaluations.



## 📂 Repository Structure

The repository is organized into the following main directories:

- **`code/`**: Contains the core algorithmic implementation of the ITI-MCR method.

  - `Log Splitting.py`: Taking the original event log as input, the original event log is divided into a high-frequency log and a low-frequency log based on K-means clustering.

  - `RuleExtraction&TraceEvaluation.py`: Extracts dependencies, screens core resources, and statistically analyzes time distributions from high-frequency traces to construct conformance rules from the control-flow, organizational, and time perspectives. Subsequently, the method evaluates the deviation of each infrequent trace based on these rules, thereby accurately distinguish effective infrequent traces.

    

- **`dataset/`**: Contains the public event logs used for experimental evaluation.

  - Includes the model-simulated public healthcare treatment process dataset.

  - Includes logs from the BPI Challenge 2020 series. These datasets provide rich event and resource information, suitable for verifying the effectiveness of the multi-perspective infrequent trace identification method in different process environments.

    

## ⚙️ Environment Requirements

The experiments were conducted using the following environment configuration:

- **Operating System:** Windows 10 Home Edition 
- **Python Version:** 3.8 
- **Core Libraries:** PM4PY (version 2.7.15.3) 
- **Other Software:** ProM 6.9 

## 🚀 Usage

1. **Log Partitioning:** Run `code/Log Splitting.py` to cluster the initial event log into high-frequency and low-frequency trace logs.
2. **Rule Extraction & Trace Evaluation:** Run `code/RuleExtraction&TraceEvaluation.py`. This script will process the split logs, construct multi-perspective (control-flow, organizational, time) conformance rules from the high-frequency log, and output a comprehensive score for traces in the low-frequency log to identify effective infrequent behaviors.
