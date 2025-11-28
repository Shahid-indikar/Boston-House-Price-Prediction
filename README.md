# Boston-House-Price-Prediction
End-to-end machine learning project using the Boston Housing dataset. Includes EDA, feature importance analysis, model comparison (Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, and XGBoost), and deployment. The final XGBoost model achieves ~0.92 R² and is deployed with Streamlit on Hugging Face Spaces for prediction.

🚀 Live Demo

🔗 Try the App:
👉 https://shahid-indikar2020-boston-house-price-prediction.hf.space

🔗 Hugging Face Space:
👉 https://huggingface.co/spaces/Shahid-indikar2020/Boston-house-price-prediction

📊 Project Overview:
This project walks through a full ML workflow:

✔ Data cleaning & preprocessing
✔ Statistical exploration & visualization
✔ Feature correlation analysis
✔ Model comparison (Linear Regression → XGBoost)
✔ Hyperparameter tuning
✔ Performance evaluation using R² Score
✔ Deployment of final model with UI
The final model (XGBoost Regressor) reaches ~0.92 R² score, indicating strong predictive accuracy on the dataset.

🧬 Dataset
Boston Housing Dataset
Samples: 506
Features: 13 numerical predictors
Target Variable: medv (Median home price in $1000s)
Example features:
| Feature   | Meaning                           |
| --------- | --------------------------------- |
| `crim`    | Per capita crime rate             |
| `rm`      | Avg. number of rooms per dwelling |
| `lstat`   | % lower status population         |
| `ptratio` | Pupil-teacher ratio               |
| `tax`     | Property tax rate                 |

🧠 Model Selection Process
| Model                     | R² Score  |
| ------------------------- | --------- |
| Linear Regression         | ~0.66     |
| Ridge Regression          | ~0.67     |
| Lasso Regression          | ~0.65     |
| Decision Tree             | ~0.85     |
| Random Forest             | ~0.88     |
| **Gradient Boosting**     | **~0.91** |
| **XGBoost (Final Model)** | **~0.92** |
After evaluation, XGBoost was chosen for deployment.

🛠️ Tech Stack
| Category        | Tools                                |
| --------------- | ------------------------------------ |
| Language        | Python                               |
| ML/DS Libraries | scikit-learn, XGBoost, Pandas, NumPy |
| Visualization   | Matplotlib, Seaborn                  |
| Deployment      | Streamlit, Hugging Face Spaces       |


▶️ How to Run Locally
# Clone repository
git clone https://github.com/<your-username>/Boston-House-Price-Prediction.git
cd Boston-House-Price-Prediction
# Install dependencies
pip install -r requirements.txt
# Run app
streamlit run streamlit_app.py

🎯 Key Learning Outcomes
Evaluating multiple ML algorithms to select the best model
Understanding feature importance and model interpretability
Serving predictions via an interactive web interface
Deploying real ML models publicly using Hugging Face Spaces
