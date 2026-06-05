<h1 align="center">Enterprise Clinical Copilot</h1>

<p align="center">
  <img src="assets/banner.png" alt="Enterprise Clinical Copilot" width="480">
</p>

A machine learning system to predict patient readmission risk at discharge. Alongside the model, this workspace will develop agentic tools that automate the risk-analysis process and provide in-depth analysis of patient status — agents reason over unstructured EHR data and combine it with ML risk scores to help healthcare professionals thoroughly assess readmission risk.

## Architecture

```mermaid
flowchart LR
  BQ["BigQuery<br/>(source tables)"]
  DF["Dataform<br/>(ELT → data representation)"]
  ML["Vertex AI Pipelines<br/>(train + endpoint)"]
  RAG["Agentic RAG<br/>(Vector Search)"]
  AG["Orchestration Agent<br/>(Agent Engine)"]
  DJ["Django + A2UI<br/>(UI / harness)"]

  BQ --> DF --> ML --> AG
  RAG --> AG --> DJ
  DF -.notes.-> RAG
```

## Status

🚧 This repo is under construction and expected to be completed at end of June.
