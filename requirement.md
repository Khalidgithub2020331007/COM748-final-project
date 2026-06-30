# Requirements — What Needs to Be Fixed (COM748 Final Report)

Based on the teacher's review, the following issues must be addressed before resubmission.

---

## 1. Abstract
- [ ] Rewrite the abstract to reflect a **completed study**, not a proposal
- [ ] Report results for **all 5 models**, not just Random Forest's 0.91 ROC-AUC
- [ ] Include LIME results — it is claimed as a contribution but has no output shown
- [ ] State the novelty clearly and concisely — what makes this different from existing work

---

## 2. Literature Review ✅ DONE in work doc
- [x] Expand references from 5 to a minimum of **15–20 sources** — work doc has 37 IEEE-formatted references
- [x] Add dedicated coverage of: **SHAP**, **LIME**, and **Explainable AI in cardiovascular prediction**
- [x] Identify a clear **research gap** — covered in Section D (Synthesis and Research Gap)
- [x] Justify why SMOTE + multiple models + SHAP + CRISP-DM together is a meaningful contribution

---

## 3. Methodology
- [ ] Resolve the dataset size confusion — clarify whether it is **303 or 918 instances** and why
- [ ] Add **hyperparameters** for all 5 classifiers (e.g., n_estimators, C, kernel, k, max_depth)
- [ ] Move results out of the methodology section — results belong in the Results section only
- [ ] Provide full implementation detail for **LIME** (not just a mention)
- [ ] Clarify or remove the term **"adversarial testing"** — define it or drop it
- [ ] Fix the CRISP-DM description — align it correctly with how it was actually applied

---

## 4. Results
- [ ] Add a **full comparison table** of all 5 models: accuracy, precision, recall, F1, AUC
- [ ] Add **confusion matrices** for each classifier
- [ ] Include **cross-validation output** (mean ± std per model per metric)
- [ ] Add **SHAP visualisations**: summary plot, bar plot, waterfall plot
- [ ] Add **LIME visualisations**: at least one positive and one negative patient explanation
- [ ] Remove all uses of "preliminary results" — present findings as final

---

## 5. Discussion and Conclusion
- [ ] Write a proper **discussion** that explains why certain models outperformed others
- [ ] Discuss all 5 classifiers — not just 1 or 2
- [ ] Write a dedicated **Conclusion section** (currently missing)
- [ ] Acknowledge at least **3–4 limitations**: dataset size, single-centre data, binary target, generalisability
- [ ] Remove or tighten the loosely written legal/ethical/project management sections
- [ ] **Fix stale AUC figure in conclusion:** work doc abstract and conclusion state "Logistic Regression and SVM achieved AUC 0.960" — this contradicts the actual results (RF best test AUC = 0.917, SVM best recall = 0.902); update the conclusion to match real numbers

---

## 6. References and Presentation
- [ ] Add the **author's name** to the paper
- [ ] Fix **duplicate section numbering**
- [ ] Expand references to meet postgraduate standards (15–20 minimum, properly cited)
- [ ] Fix typographical errors and informal phrasing throughout
- [ ] Remove overstatements — soften unsupported claims
- [ ] Ensure all claimed contributions have matching visual or numerical evidence in the paper
