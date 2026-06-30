# Plan — Step-by-Step Revision of COM748 Final Report

Work through these phases in order. Each phase builds on the previous one.

---

## Phase 1 — Fix the Code and Reproduce Results (Day 1–2)

Before touching the paper, make sure the actual experiment is complete and reproducible.

**Step 1.1 — Resolve dataset confusion**
- Confirm whether you used the UCI Cleveland dataset (303 rows) or the combined version (918 rows)
- If 303: document it clearly; if 918: clarify which sources were merged and why

**Step 1.2 — Re-run all 5 models with hyperparameters recorded**
- Log the exact hyperparameters used for each model:
  - Logistic Regression: `C`, `solver`, `max_iter`
  - Random Forest: `n_estimators`, `max_depth`, `min_samples_split`
  - XGBoost: `n_estimators`, `learning_rate`, `max_depth`
  - SVM: `C`, `kernel`, `gamma`
  - KNN: `n_neighbors`, `metric`

**Step 1.3 — Generate all missing outputs**
- Full metrics table (accuracy, precision, recall, F1, AUC) for all 5 models
- Confusion matrix for each model
- 5-fold cross-validation results (mean ± std per metric per model)
- SHAP: summary plot, bar plot, waterfall plot for at least one patient
- LIME: explanation for one positive-prediction patient and one negative-prediction patient

**Step 1.4 — Save all figures**
- Export all plots as `.png` files at 300 DPI
- Name them clearly: `shap_summary.png`, `cm_random_forest.png`, `lime_patient_positive.png`, etc.

---

## Phase 2 — Expand the Literature Review (Day 2–3)

**Step 2.1 — Find 15–20 sources minimum**
Use Google Scholar. Search for:
- "heart disease prediction machine learning" (find 3–4 papers)
- "SHAP explainability clinical decision" (find 3–4 papers)
- "LIME medical prediction" (find 2–3 papers)
- "SMOTE class imbalance medical" (find 2–3 papers)
- "CRISP-DM health informatics" (find 1–2 papers)
- UCI Cleveland dataset papers (find 2–3 papers)

**Step 2.2 — Identify the research gap**
After reading the papers, write 2–3 sentences that answer:
> "Most existing studies do X, but none combine Y and Z together, which is what this study does."

**Step 2.3 — Rewrite the literature review section**
- Organise into sub-sections (same structure as current: A, B, C, D)
- Each sub-section should cite 3–5 papers and end with what is still missing

---

## Phase 3 — Rewrite the Paper Sections (Day 3–5)

Work section by section in this order:

**Step 3.1 — Methodology**
- Fix CRISP-DM description to match what was actually done
- Add hyperparameter table
- Remove results from this section
- Add LIME implementation detail (library used, number of features shown, kernel width if set)
- Remove or properly define "adversarial testing"

**Step 3.2 — Results**
- Insert full model comparison table
- Insert confusion matrices (one per model or a combined figure)
- Insert cross-validation table
- Insert SHAP plots with captions explaining what each shows
- Insert LIME plots with captions
- Remove all "preliminary results" phrases

**Step 3.3 — Discussion**
- Write at least one paragraph per model explaining performance relative to others
- Explain why Logistic Regression and SVM achieved the highest AUC (0.960)
- Explain why Random Forest had the best precision/recall balance
- Connect SHAP findings back to clinical literature (asymptomatic chest pain, thalassemia, vessel count)

**Step 3.4 — Conclusion** (new section — currently missing)
- Summarise what was done (2–3 sentences)
- State the best-performing model and key SHAP/LIME findings (2–3 sentences)
- State 3–4 limitations honestly
- State 2–3 directions for future work

**Step 3.5 — Abstract**
- Rewrite last — once everything else is final
- Structure: problem → dataset → methods → all 5 model results → SHAP/LIME finding → conclusion
- Length: 200–250 words

---

## Phase 4 — Fix Presentation Issues (Day 5–6)

**Step 4.1 — Author name**
- Add your name to the title page / paper header

**Step 4.2 — Section numbering**
- Check for and fix all duplicate section numbers

**Step 4.3 — References**
- Format all references in IEEE or ACM style (match whatever the template uses)
- Check every in-text citation has a corresponding entry in the reference list
- Add citations for SHAP, LIME, SMOTE, scikit-learn, and the UCI dataset itself

**Step 4.4 — Language and tone**
- Search for and remove informal phrases
- Replace overstatements ("proves that", "demonstrates conclusively") with measured language ("suggests", "indicates", "provides evidence that")
- Run a spellcheck pass

---

## Phase 5 — Final Review and Submission (Day 6–7)

**Step 5.1 — Self-review checklist**
- [ ] All 5 models have results reported
- [ ] SHAP and LIME visualisations are in the paper with captions
- [ ] Confusion matrices and cross-validation table are present
- [ ] Literature review has 15+ references
- [ ] A dedicated Conclusion section exists
- [ ] Author name is on the paper
- [ ] No duplicate section numbers
- [ ] No "preliminary results" language

**Step 5.2 — Commit revised paper to GitHub**
```bash
git add .
git commit -m "fix: address teacher review comments — revised report v2"
git push
```

**Step 5.3 — Submit**
- Submit the revised `.doc` / `.pdf` to your course submission portal

---

## Summary Timeline

| Day | Work |
|-----|------|
| 1–2 | Re-run experiments, generate all missing figures and tables |
| 2–3 | Expand literature review to 15–20 sources, identify research gap |
| 3–5 | Rewrite methodology, results, discussion, and add conclusion section |
| 5–6 | Fix presentation: author name, numbering, references, language |
| 6–7 | Final self-review, commit to GitHub, submit |
