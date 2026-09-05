# SARF — Project Overview

## Project

**SARF — صَرْف** is a cross-dialect Arabic NLP project that studies Saudi-style Arabic exposure for banking intent classification.

## Problem

Banking intent classifiers are often trained on labelled Modern Standard Arabic (MSA), while Saudi customers may express the same intent using Saudi-style wording. Labelled Saudi banking data are limited and expensive to create, so the project investigates whether unlabelled Saudi-style text can help an AraBERT-based classifier adapt before the supervised banking task.

## Main Research Question

How does increasing synthetic Saudi Arabic exposure affect Macro-F1 when AraBERT is adapted through Continued Pre-Training, fine-tuned on labelled MSA banking queries, and evaluated on a frozen Saudi test split?

## Project Scope

The project focuses on 77-class banking intent classification. It compares two labelled-data baselines, TF-IDF with SVM and TextCNN, with five AraBERT conditions: E0, E1, E2, E3, and EB.

The project does not include a chatbot, RAG, extra dialects, or unrelated NLP tasks.

## Approved Experimental Conditions

| Condition | Saudi CPT exposure |
| --- | --- |
| E0 | No Saudi CPT |
| E1 | 3K general synthetic Saudi-style text |
| E2 | 6K general synthetic Saudi-style text |
| E3 | 12K general synthetic Saudi-style text |
| EB | 12K general synthetic Saudi-style text plus 3K Saudi banking text |

All AraBERT conditions start from the same base checkpoint, use the same MSA fine-tuning and validation protocol, and run with three approved random seeds.

## Data Roles

The labelled MSA training set is used for fine-tuning. The MSA validation set is used for checkpoint and development decisions. The Saudi test split is frozen and is used only once for final evaluation after all model decisions are fixed.

The active synthetic release is **v2**. The v2 files are used for CPT only. Version v1 and draft files are retained for provenance and must not be used for training.

## Current Project Phase

The data-foundation phase is complete. It included EDA, synthetic corpus generation, cleaning, manual sample auditing, leakage checks, and the v2 freeze. The next phase is model implementation: SVM, TextCNN, and AraBERT E0 followed by the remaining AraBERT CPT conditions.

## Folder Guide

| Folder | Purpose |
| --- | --- |
| `00_project_context/` | Short project overview and orientation for new contributors |
| `01_docs/` | Decisions, roles, weekly status, corpus cards, freeze declarations, and report draft |
| `02_configs/` | Prompts, topic plans, schema, and experiment plan |
| `03_notebooks/` | EDA and synthetic-corpus notebooks |
| `04_src/` | Baseline code and shared metrics |
| `05_data_notes/` | Dataset manifest and lightweight data notes |

## Reading Order

Read this overview first. Then read `01_docs/team_roles.md`, `01_docs/weekly_status.md`, `02_configs/experiment_plan.md`, and the active v2 corpus documentation before contributing code or running experiments.
