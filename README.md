# Movie Recommendation System

**Adithya Nayak — 251090700070**  


This folder contains one notebook for each phase, shared datasets, a fitted model, the local Flask application, evaluation results, and the report in Word and PDF formats.

| File or folder | Purpose |
|---|---|
| Movie_Recommendation_Phase1.ipynb | Dataset analysis, EDA and model planning |
| Movie_Recommendation_Phase2.ipynb | Training, evaluation and recommendations |
| Movie_Recommendation_Phase3.ipynb | Model selection evidence and deployment demonstration |
| reports/ | Project report; AI declaration and supplied prompts are in Section 10 |
| data/ | Original datasets, checksums and source notices |
| evaluation/ | Measured comparison tables and plots |
| artifacts/ | Fitted model, export audit and test evidence |
| app.py, inference.py, web/ | Flask application and recommendation interface |

The cleaned catalogue contains **4,557 movies**, after removing 11 repeated listings defined by case-insensitive title, release year and description. The matched subset remains **304 movies and 8,125 ratings**. The deployment ranks 295 reference-observed movies.

**Selection:** Popularity wins validation NDCG@10. Ridge is the best supervised validation ranker and provides personalization. Interaction SVD scores highest on test, but does not replace the validation-selected method.

**Deployment:** install requirements-runtime.txt and run `python app.py`; the local address is http://127.0.0.1:5000. See Phase 3 Section 8. No public hosting is claimed.

**Reproduction:** requirements.txt lists the analysis dependencies. Run Phase 2 before train_export.py to regenerate evaluation and export artifacts. training_evaluation.py executes the Phase 2 notebook code rather than keeping a duplicate implementation. test_deployment.py checks the Python API and scores; test_browser_model.cjs checks JavaScript parity using the generated Python fixtures. Saved notebook outputs reflect successful fresh-kernel execution.

**Scope:** historical Hotstar listings and MovieLens ratings; current streaming availability is unverified. Quantitative results apply to the matched known-user setting. Both students must confirm their individual learning reflection and design ownership before submission.
