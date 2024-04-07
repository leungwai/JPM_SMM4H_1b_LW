# 🐤 JPM_SMM4H_1b_LW

**Subtask 1b** for the **7th Social Mining Media for Health (SMM4H)** competition hosted by **International Conference on Computational Logistics (COLING) 2022**

_by Leung Wai Liu_

This repo is training, ensembling and analysis code for the BERT Model used for Subtask 1b of the SMM4H competition that I competed in during my internship at the AI Research team in J.P. Morgan Chase in the Summer of 2022. 

_\#NLP \#BERT \#ML \#Python_

**See Also:** Subtask 1a | Subtask 1c | Subtask 2a | Subtask 2b | Subtask 5

## Premise
The need to use Natural Language Processing \(NLP\) on social media posts is increasingly important as its userbase grows to guage public perception on issues, such as sentiments during the COVID-19 pandemic. 

## Task Description
Task 1 is a pharmacoviligance task, which Subtask 1b consists of given the adverse event (ADE) of a tweet, detect the span of words that pertains to such ADE. 

## Methodology
The datasets were trained on variants top of the BERT language model \(Devlin et al., 2019\) (RoBERTa<sub>BASE</sub>, RoBERTa<sub>LARGE</sub>, BERT<sub>BASE</sub>-uncased, BERT<sub>LARGE</sub>-uncased. 

The model ensembling code is adapted from \(Jayanthi and Gupta, 2021\) method of model ensembling. Various methods of ensembling were experimented, including majority-vote, weighted and unweighted. Ultimately, a majority ensemble of RoBERTa<sub>LARGE</sub> models were used. 

## Results 
**Overlapping Performance Metrics for Subtask 1b** 
| Task | F1-Score | Precision | Recall | 
| ---: | :---: | :---: | :---: |
| Task 1a | 0.568 | 0.671 | 0.492 | 

> Placed **7th** of 14 submissions

## Special Thanks
- **Akshat Gupta**, for being a great project manager and guiding us through NLP from start to finish
- **Saheed Obitayo**, for being a great manager
- **J.P. Morgan AI Research** and **Prep for Prep** for the incredible opportunity for the internship
- The organizers for the 7th SMM4H competition and 2022 COLING conference
