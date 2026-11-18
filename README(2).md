# AI Study Lab --- ML-Powered Student Learning Assistant

## Project Goal

Build a clean, beginner-friendly, game-like **AI Study Lab** for
students.

The application must use **five different machine-learning models and
five different educational datasets**. The purpose is to help a student
study, discover resources, understand questions, choose learning
material, and organize related courses.

**Important:** This project is a study assistant. It must NOT be
presented as a system that grades, ranks, predicts marks, or evaluates a
student's academic performance.

------------------------------------------------------------------------

# 1. Machine Learning Models

Use the supplied Python training files as the ML backend.

  --------------------------------------------------------------------------------
  Model          Python file                    Dataset folder Student-facing
                                                               purpose
  -------------- ------------------------------ -------------- -------------------
  Logistic       `logistic_regression.py`       `logistic/`    Recommend a
  Regression                                                   suitable
                                                               course/resource

  Decision Tree  `decision_tree.py`             `decision/`    Guide the student
                                                               toward an
                                                               appropriate
                                                               learning level/path

  Naive Bayes    `naive_bayes.py`               `naives/`      Classify a
                                                               student's question
                                                               into a
                                                               chapter/topic

  SVM            `svm_question_classifier.py`   `question/`    Classify the type
                                                               of a student's
                                                               question

  K-Means        `kmeans.py`                    `k means/`     Group similar
                                                               educational
                                                               resources/courses
  --------------------------------------------------------------------------------

Do not replace these models with another algorithm unless absolutely
necessary. The project requirement is five different ML models.

------------------------------------------------------------------------

# 2. Dataset Mapping

## Logistic Regression

Source file inside `logistic/`:

`recomdata.csv`

Columns:

-   `interests`
-   `category`
-   `preferred_difficulty`
-   `past_experience`
-   `preferred_language`
-   `recommended_course`

Use these fields to create a course/resource recommendation experience.

### Required cleaning

-   Remove exact duplicate rows.
-   Remove rows with missing target values.
-   Normalize text fields.
-   Strip leading/trailing whitespace.
-   Normalize inconsistent capitalization where appropriate.
-   Handle missing categorical values safely.
-   Check class distribution.
-   Do not create fake records.
-   Do not randomly invent course names.
-   Encode categorical variables using a reproducible preprocessing
    pipeline.
-   Avoid data leakage: preprocessing must be fitted on training data
    only.

------------------------------------------------------------------------

# 3. Decision Tree

Source files inside `decision/` include:

-   `Courses_and_Learning_Material.csv`
-   `Learning_Pathway_Index.csv`

Use the learning-path dataset as the primary source for the Decision
Tree.

Relevant fields include:

-   `Module_Code`
-   `Course_Learning_Material`
-   `Source`
-   `Course_Level`
-   `Type_Free_Paid`
-   `Module`
-   `Duration`
-   `Difficulty_Level`
-   `Keywords_Tags_Skills_Interests_Categories`
-   `Links`

The model should help a student decide **what level/resource to study
next**, rather than predicting marks.

### Required cleaning

-   Remove duplicate records.
-   Clean text fields.
-   Normalize difficulty/level labels.
-   Handle missing values.
-   Convert duration into a usable numeric representation where
    possible.
-   Clean URLs but do not fabricate missing URLs.
-   Clean keyword/tag fields.
-   Inspect class/target distribution before training.
-   Keep the original dataset values available for displaying resources.
-   Do not manufacture learning resources.

------------------------------------------------------------------------

# 4. Naive Bayes

Source files inside `naives/`:

-   `train.csv`
-   `val.csv`
-   `test.csv`

Columns:

-   `q_id`
-   `eng`
-   `class`
-   `chapter`

The primary NLP input is `eng`.

The model should allow a student to type a question and receive the
predicted **chapter/topic**.

### Required cleaning

-   Combine/use the supplied train/validation/test structure correctly.
-   Never train on the test set.
-   Remove empty questions.
-   Remove duplicate question records where appropriate.
-   Normalize whitespace.
-   Preserve meaningful question words.
-   Handle malformed rows.
-   Use a reproducible text vectorization pipeline.
-   Fit vectorization only on training data.
-   Report validation/test performance separately.
-   Save the complete preprocessing + model pipeline.

### Student experience

Example:

> Student: "What is supervised learning?"

Application:

> Predicted topic/chapter: Machine Learning Basics

The exact output must come from the trained dataset/model, not a
hard-coded answer.

------------------------------------------------------------------------

# 5. SVM

Source files inside `question/`:

-   `train_5500.txt`
-   `test_data.txt`

The records use question labels such as:

-   `ABBR`
-   `DESC`
-   `ENTY`
-   `HUM`
-   `LOC`
-   `NUM`

The text before/after the label must be parsed correctly.

The SVM should classify the **type of question**.

Example:

> "Where is the Taj Mahal located?"

Possible result:

> Question Type: LOC

### Required cleaning

-   Parse the label and question text robustly.
-   Remove malformed rows.
-   Remove empty questions.
-   Normalize whitespace.
-   Preserve meaningful NLP information.
-   Do not leak test data into training.
-   Use a reproducible TF-IDF/text preprocessing pipeline.
-   Evaluate with accuracy, precision, recall, F1-score and confusion
    matrix.
-   Save the complete preprocessing + model pipeline.

------------------------------------------------------------------------

# 6. K-Means

The `k means/` archive contains large educational course datasets,
including:

-   `combined_dataset.json`
-   `edx_courses.json`
-   `edx_degree_programs.json`
-   `edx_executive_education_paidstuff.json`
-   `edx_programs.json`
-   `processed_coursera_data.json`

Use the most appropriate course/resource records for clustering.

Potential useful fields include:

-   course title/name
-   description
-   skills
-   subject
-   level
-   provider/organization
-   duration
-   rating where available

### Required cleaning

Because this dataset is large and contains heterogeneous JSON
structures:

1.  Inspect the schemas before combining data.
2.  Select only compatible educational-resource records.
3.  Standardize column names.
4.  Remove exact duplicates.
5.  Remove records without meaningful course/resource text.
6.  Clean HTML/noisy text if present.
7.  Normalize whitespace.
8.  Handle missing values.
9.  Avoid using meaningless identifiers as clustering features.
10. Convert useful text into TF-IDF features.
11. Scale numeric features only when they are actually used.
12. Choose a defensible number of clusters using an appropriate method
    such as silhouette score/elbow analysis.
13. Save the fitted vectorizer/scaler/model.
14. Preserve the original course title, subject, provider and URL for
    displaying results.

### Student experience

The student can explore groups such as:

-   Python / Programming
-   Data Science
-   Machine Learning
-   Web Development
-   Mathematics
-   Business
-   Other clusters actually discovered from the data

**Do not hard-code these cluster names unless the cluster contents
genuinely support them.**

------------------------------------------------------------------------

# 7. Global Data Cleaning Rules

Every dataset must go through a documented cleaning pipeline.

## Required checks

For every dataset:

-   Shape before cleaning
-   Shape after cleaning
-   Duplicate count
-   Missing-value count
-   Data types
-   Unique target values/classes
-   Class imbalance where applicable
-   Invalid records
-   Empty text records
-   Outliers for relevant numerical fields

## Cleaning principles

### Never fabricate data

Do NOT:

-   invent missing course names
-   invent ratings
-   invent URLs
-   invent question answers
-   create fake student records
-   randomly assign categories
-   replace missing values with made-up information

### Missing values

Use sensible methods depending on the column:

-   categorical → most frequent/explicit unknown category where
    appropriate
-   numeric → median or model-appropriate imputation
-   text → empty string or safe placeholder only where appropriate
-   target → remove rows when target is unavailable

### Duplicates

Remove exact duplicates.

For NLP datasets, also check for duplicated question text that could
cause leakage between train and test.

------------------------------------------------------------------------

# 8. Data Leakage Prevention

This is extremely important.

The project must not accidentally make the model look better than it
really is.

For supervised models:

1.  Split the data first.
2.  Fit preprocessing only on the training data.
3.  Transform validation/test using the fitted preprocessing.
4.  Evaluate only on unseen data.

For NLP:

`TF-IDF -> Model`

must be implemented as a single reproducible pipeline wherever possible.

Do not fit TF-IDF on the complete dataset before splitting.

For K-Means:

-   Fit the vectorizer/scaler and clustering model on the intended
    training/reference dataset.
-   Reuse the fitted preprocessing when assigning a new resource to a
    cluster.

------------------------------------------------------------------------

# 9. Model Saving

Each training script should save its trained model/pipeline as a `.pkl`
file.

Suggested names:

``` text
logistic_regression_model.pkl
decision_tree_model.pkl
naive_bayes_model.pkl
svm_question_classifier_model.pkl
kmeans_model.pkl
```

Also save any required vectorizer/scaler if it is not already included
inside the pipeline.

The Streamlit application must load the saved models instead of
retraining every time a student opens the app.

------------------------------------------------------------------------

# 10. Required Evaluation

Each supervised model must show:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion matrix where practical

For K-Means, show:

-   Number of clusters
-   Cluster sizes
-   Silhouette score if applicable
-   Basic interpretation of clusters

Do not hide poor results. If a dataset naturally produces a lower score,
report it honestly.

------------------------------------------------------------------------

# 11. Student-Friendly Application Design

The UI must be **clean, simple and fun**, while still looking like a
serious college ML project.

The visual style should feel slightly game-like without becoming
childish.

Think:

**Study Dashboard + Learning Quest + AI Assistant**

## Main dashboard

Show:

-   Welcome message
-   "Start Learning" button
-   Progress/XP-style visual elements
-   Five ML feature cards
-   Recommended resources
-   Recently explored topics
-   Small achievement/badge area

Do not require account creation unless needed.

------------------------------------------------------------------------

# 12. Five Main Student Features

## Feature 1 --- Course Finder

Powered by Logistic Regression.

Ask the student simple questions such as:

-   What are you interested in?
-   What category?
-   Preferred difficulty?
-   Previous experience?
-   Preferred language?

Then show recommended learning resources.

Keep the form short and beginner-friendly.

------------------------------------------------------------------------

## Feature 2 --- Learning Path

Powered by Decision Tree.

Ask what the student wants to learn and/or their current level.

Return:

-   Recommended level
-   Suggested learning material
-   Next learning step
-   Resource link when available

Use friendly language such as:

> "Your next learning step"

instead of technical ML terminology.

------------------------------------------------------------------------

## Feature 3 --- Ask the Study AI

Powered by Naive Bayes.

Provide a large question box:

> "Type your study question..."

Return:

-   Predicted chapter/topic
-   Confidence/probability if the model supports it
-   Suggested next action

Do not pretend the model generated an answer if it only classifies the
question.

------------------------------------------------------------------------

## Feature 4 --- Question Detective

Powered by SVM.

Student enters a question.

Show:

> Question Type: Location / Definition / Person / Number / etc.

Make this feature visually engaging.

For example:

> 🔎 Question classified!

The game-like presentation should enhance learning, not hide what the
model is actually doing.

------------------------------------------------------------------------

## Feature 5 --- Explore Learning Worlds

Powered by K-Means.

Display course/resource clusters as learning "worlds" or "zones."

Example:

``` text
🌐 Learning Worlds

🧠 World 1
Machine Learning & AI

💻 World 2
Programming & Development

📊 World 3
Data & Analytics
```

Only use labels that are actually supported by cluster contents.

Clicking a world should display real resources from the dataset.

------------------------------------------------------------------------

# 13. Game-Like Learning System

The game elements should be lightweight.

Possible elements:

-   XP points
-   Learning streak
-   Badges
-   "Quest" cards
-   Progress bar
-   Topic exploration
-   Completed-resource counter
-   Small celebration when a student completes a learning activity

Example:

``` text
🎯 Today's Study Quest

[ ] Find a beginner Python course
[ ] Explore one ML topic
[ ] Classify 3 questions
[ ] Discover a new learning resource

XP: 120
```

Important:

These game elements are for motivation only.

Do not use them to rank students against one another.

------------------------------------------------------------------------

# 14. UI/UX Rules

The application should be:

-   Clean
-   Responsive
-   Easy to understand
-   Beginner-friendly
-   Fast
-   Consistent
-   Not overloaded with text
-   Not overloaded with charts
-   Accessible on laptop and normal desktop screens

Avoid:

-   cluttered dashboards
-   excessive animations
-   complicated ML terminology
-   unnecessary login systems
-   unnecessary forms
-   fake statistics
-   fake AI answers
-   meaningless decorations

Use clear cards, icons, buttons and short explanations.

Every ML feature should tell the student:

1.  What it does
2.  What to enter
3.  What the result means

------------------------------------------------------------------------

# 15. Technical Requirements for the App

Recommended stack:

-   Python
-   Streamlit
-   pandas
-   NumPy
-   scikit-learn
-   matplotlib / Plotly if required
-   joblib or pickle for saved models

The app must load the trained `.pkl` files.

Do not retrain models during normal application use.

If a model file is missing, display a clear setup message instead of
crashing.

------------------------------------------------------------------------

# 16. Explainability

Because this is a college ML project, include a small **"How AI Works"**
section.

Keep it simple.

Example:

### Logistic Regression

"Uses patterns in your selected preferences to recommend suitable
learning resources."

### Decision Tree

"Uses a sequence of learning-related decisions to suggest an appropriate
path."

### Naive Bayes

"Looks at the words in your question and predicts the most likely
chapter/topic."

### SVM

"Finds patterns that separate different types of questions."

### K-Means

"Groups similar learning resources together based on their
characteristics."

Do not claim the models are intelligent beyond what they actually do.

------------------------------------------------------------------------

# 17. Project Structure

Use a clean structure similar to:

``` text
AI-Study-Lab/
│
├── app.py
│
├── models/
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── naive_bayes.py
│   ├── svm_question_classifier.py
│   └── kmeans.py
│
├── trained_models/
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── svm_question_classifier_model.pkl
│   └── kmeans_model.pkl
│
├── data/
│   ├── logistic/
│   ├── decision/
│   ├── naives/
│   ├── question/
│   └── k means/
│
├── notebooks/
│   └── optional EDA notebooks
│
├── assets/
│
├── requirements.txt
└── README.md
```

The exact structure can be adjusted if necessary, but keep the project
organized.

------------------------------------------------------------------------

# 18. Important Integration Rule

The supplied Python model files are the ML foundation.

If Antigravity needs to modify them for integration, preserve:

-   the same five algorithms
-   the same datasets
-   the same intended purposes
-   proper preprocessing
-   proper evaluation
-   proper model saving
-   reproducibility

Do not replace the ML system with hard-coded recommendation rules.

The UI may contain simple presentation logic, but the actual
prediction/classification/clustering must come from the trained ML
models.

------------------------------------------------------------------------

# 19. Data Tinkering / Feature Engineering

Perform meaningful feature engineering where it improves the model.

Examples:

### Course datasets

Possible features:

-   normalized difficulty
-   course level
-   duration
-   free/paid status
-   subject/category
-   keyword/tag features
-   text features

### NLP datasets

Possible features:

-   TF-IDF
-   n-grams where useful
-   normalized text

Do not add random features simply to make the project look complicated.

Every engineered feature should have a reason.

------------------------------------------------------------------------

# 20. Reproducibility

Set random states where supported.

Example:

``` python
random_state=42
```

The same dataset and code should produce reproducible results as far as
the selected algorithms allow.

Print useful training information so the student can understand what
happened.

------------------------------------------------------------------------

# 21. Final Quality Check

Before considering the project finished, verify:

-   [ ] All five datasets load correctly.
-   [ ] All five datasets are cleaned.
-   [ ] Duplicate and missing-value checks are performed.
-   [ ] No fake data is introduced.
-   [ ] No target leakage exists.
-   [ ] All five models train successfully.
-   [ ] All five models are evaluated appropriately.
-   [ ] All five models are saved.
-   [ ] Streamlit can load every model.
-   [ ] Predictions work with real user input.
-   [ ] K-Means returns real resources from the dataset.
-   [ ] Invalid user input does not crash the application.
-   [ ] Missing data is handled gracefully.
-   [ ] The UI is understandable to a beginner.
-   [ ] The UI feels slightly game-like but remains professional.
-   [ ] No student marks/performance evaluation is included.
-   [ ] No fake AI-generated answers are presented as model predictions.
-   [ ] The application works from a clean installation.

------------------------------------------------------------------------

# 22. Overall User Journey

The ideal student experience is:

``` text
                    AI STUDY LAB
                         │
             ┌───────────┴───────────┐
             │                       │
        🎯 Start Learning       🌎 Explore
             │                       │
     ┌───────┼────────┐              │
     │       │        │              │
   Course  Learning  Ask AI       Learning
   Finder   Path     Question      Worlds
     │       │        │              │
     └───────┴────────┴──────────────┘
                         │
                    🎮 Earn XP
                         │
                  🏆 Complete Quests
                         │
                  📚 Keep Learning
```

The goal is to make ML feel useful and approachable to a student.

The application should communicate:

> **"Tell me what you want to learn. I'll help you find where to
> start."**

------------------------------------------------------------------------

# 24. Training, Testing and Accuracy Requirements

This project **must train and test all five ML models** using the
supplied datasets.

Do not only build the UI. The ML pipeline must actually execute:

``` text
Dataset
   ↓
Data cleaning
   ↓
Preprocessing
   ↓
Train/Test split or supplied train/validation/test split
   ↓
Model training
   ↓
Prediction on unseen test data
   ↓
Evaluation
   ↓
Save trained model
   ↓
Streamlit uses saved model
```

## Accuracy must be visible

The training scripts must print the test accuracy after training.

The Streamlit application should also have an **ML Model Performance**
section showing the final test metrics for all models.

Use wording that makes it clear these are **test-set results**, not
guarantees about future predictions.

### Current baseline results from the supplied datasets

These are the results obtained when the supplied Python scripts were run
against the uploaded datasets:

  Model                 Evaluation metric     Current baseline
  --------------------- ------------------- ------------------
  Logistic Regression   Test Accuracy              **100.00%**
  Decision Tree         Test Accuracy               **84.09%**
  Naive Bayes           Test Accuracy               **62.00%**
  SVM                   Test Accuracy               **88.60%**
  K-Means               Silhouette Score            **0.0076**

### Important: K-Means does not have "accuracy"

K-Means is an unsupervised clustering algorithm. There are no class
labels for a normal accuracy calculation.

Therefore, **do not display an invented K-Means accuracy**.

Use a clustering metric such as:

-   Silhouette Score
-   Inertia / Within-Cluster Sum of Squares
-   Cluster sizes

The current baseline silhouette score is **0.0076**, which indicates
that the current course representation produces weakly separated
clusters. Antigravity should improve the data cleaning/feature
representation and validate whether a better clustering configuration is
justified. It must not artificially manipulate the data just to increase
the score.

## Overall accuracy

There is technically **no single scientifically valid "overall accuracy"
for all five models**, because:

-   Logistic Regression, Decision Tree, Naive Bayes and SVM are
    supervised classification models.
-   K-Means is unsupervised clustering and has no accuracy value.

For the project presentation, calculate and display:

### Average classification accuracy

Average the four supervised model test accuracies:

``` text
(100.00 + 84.09 + 62.00 + 88.60) / 4
= 83.67%
```

Therefore the current **average classification accuracy is 83.67%**.

Label this clearly as:

> **Average Classification Accuracy (4 supervised models): 83.67%**

Do NOT label 83.67% as "overall accuracy of the entire ML system",
because that would incorrectly include/exclude K-Means.

The dashboard can instead show:

``` text
ML Model Performance

Logistic Regression     100.00%
Decision Tree            84.09%
Naive Bayes              62.00%
SVM                      88.60%

Average Classification Accuracy
83.67%

K-Means
Silhouette Score: 0.0076
```

The displayed numbers must be generated from the actual evaluation
results rather than hard-coded into the application's prediction logic.

------------------------------------------------------------------------

# 25. Training Mode vs Application Mode

Keep training and application use separate.

## Training mode

Each `.py` file must:

1.  Load the real dataset.
2.  Clean the data.
3.  Perform preprocessing.
4.  Split the data correctly or use the supplied train/validation/test
    files.
5.  Train the model.
6.  Test on unseen data.
7.  Print evaluation metrics.
8.  Save the trained model/pipeline.
9.  Save any metadata needed by the application.

## Application mode

The Streamlit application must:

1.  Load the saved `.pkl` model.
2.  Accept new student input.
3.  Apply the same saved preprocessing.
4.  Generate a real prediction/cluster.
5.  Display the result clearly.

**Do not retrain every time a student uses the application.**

Provide a clearly documented way to retrain when the dataset or training
code changes.

------------------------------------------------------------------------

# 26. Accuracy Reporting for the College Project

The project presentation/report should include a table similar to:

  ------------------------------------------------------------------------
  Algorithm             Dataset Task                           Test Metric
  --------------------- --------------------- ----------------------------
  Logistic Regression   Course recommendation             100.00% accuracy

  Decision Tree         Learning                           84.09% accuracy
                        difficulty/path       
                        classification        

  Naive Bayes           Question → chapter                 62.00% accuracy
                        classification        

  SVM                   Question-type                      88.60% accuracy
                        classification        

  K-Means               Learning-resource          0.0076 silhouette score
                        clustering            
  ------------------------------------------------------------------------

Also mention:

> The four supervised models have an average classification accuracy of
> 83.67%. K-Means is evaluated separately using clustering metrics
> because it is an unsupervised algorithm.

If retraining after improved cleaning/feature engineering changes the
scores, **replace the baseline values with the actual new test
results**. Never claim the old numbers if the final trained models
produce different results.

# 23. Final Instruction to the Builder

Build this as a **real, working ML project**, not a UI mockup.

Use the actual supplied datasets and trained models.

Clean the data properly.

Use real model predictions.

Keep the interface simple enough for a student to understand.

Make the experience engaging enough that a student would actually want
to explore it.

The final product should feel like a combination of:

**Study Assistant + Learning Resource Explorer + Small Learning Game**

while remaining technically credible as a five-model machine-learning
college project.
