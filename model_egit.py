import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "insurance.csv")

df = pd.read_csv(csv_path)


X = df.drop("charges", axis=1)
y = df["charges"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()


numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n================= MODEL RAPORU =================")
print(f"R2 Score : %{r2*100:.2f}")
print(f"MAE      : {mae:.2f} $")
print(f"RMSE     : {rmse:.2f} $")
print("================================================")

model_path = os.path.join(BASE_DIR, "insurance_ai_model.pkl")
joblib.dump(model_pipeline, model_path)

print("Model başarıyla kaydedildi:")
print(model_path)
#modeli öncelikle model_egit.py ile calıştırdıktan sonra modeli eğitme işlemimiz tamamlanıyor
#daha sonrasında bu model  .pkl dosyası şeklinde kaydediliyor ve benim de bu projeyi bir tık daha üst seviye ye
#çıkarmak için belirli bire localhe bağlanan bir websitesine dönüştürme işlemi yaptım 
#bu websitesine bağlanmak için öncelikle python model_egit.py ile dosyası çalıştırıyoruz daha sonrasında websitesine erişim için ise 
#terminal kısmına streamlit run app.py yazmamız yeterlidir bu bizi modelimize yönlendirecektir.