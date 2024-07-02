import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error
import pickle

# Carregar dados e limpeza inicial
file_path = 'desafio_indicium_imdb.csv'
data = pd.read_csv(file_path)

# Limpeza e pré-processamento dos dados
data = data.drop(columns=['Unnamed: 0'], errors='ignore')
data['Gross'] = data['Gross'].str.replace(',', '').astype(float)
data['Certificate'] = data['Certificate'].fillna('Not Rated')
data['Meta_score'] = data['Meta_score'].fillna(data['Meta_score'].mean())
data['Gross'] = data['Gross'].fillna(data['Gross'].mean())

# Verificar a limpeza dos dados
print(data.info())
print(data.head())
summary = data.describe()
display(summary)

# Análise exploratória dos dados (EDA)
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Distribuição das notas do IMDB
sns.histplot(data['IMDB_Rating'], bins=20, kde=True, color='blue', ax=axes[0, 0])
axes[0, 0].set_title('Distribuição das Notas do IMDB')

# Distribuição do faturamento (Gross)
sns.histplot(data['Gross'], bins=20, kde=True, color='green', ax=axes[0, 1])
axes[0, 1].set_title('Distribuição do Faturamento (Gross)')

# Distribuição do número de votos (No_of_Votes)
sns.histplot(data['No_of_Votes'], bins=20, kde=True, color='red', ax=axes[1, 0])
axes[1, 0].set_title('Distribuição do Número de Votos')

# Análise da relação entre as estrelas (Star1), gênero e faturamento
top_stars = data['Star1'].value_counts().nlargest(10).index
sns.boxplot(x='Star1', y='Gross', data=data[data['Star1'].isin(top_stars)], ax=axes[1, 1], palette='Set3', hue='Star1', dodge=False)
axes[1, 1].set_title('Faturamento por Ator/Atriz Principal (Top 10)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

#Recomendações de Filmes
recommended_movies = data[(data['IMDB_Rating'] >= 8.5) & (data['No_of_Votes'] >= 500000)]
recommended_movies = recommended_movies.sort_values(by=['IMDB_Rating', 'No_of_Votes'], ascending=[False, False])
print(recommended_movies[['Series_Title', 'IMDB_Rating', 'No_of_Votes', 'Gross']].head(10))

#  Modelo de regressão linear para previsão de faturamento
filtered_data = data.dropna(subset=['Gross', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Star1'])
X = filtered_data[['IMDB_Rating', 'Meta_score', 'No_of_Votes']]
y = filtered_data['Gross']
model = LinearRegression()
model.fit(X, y)

# Coeficientes do modelo
coefficients = pd.Series(model.coef_, index=X.columns)
r_squared = model.score(X, y)
print("Coeficientes do Modelo:")
print(coefficients)
print("\nR² do Modelo:")
print(r_squared)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Salvar o modelo em um arquivo .pkl
with open('modelo_imdb.pkl', 'wb') as file:
    pickle.dump(model, file)

# Célula 6: Análise de Overview para Inferência de Gênero
data_genre = data[['Overview', 'Genre']].dropna()
X_train, X_test, y_train, y_test = train_test_split(data_genre['Overview'], data_genre['Genre'], test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('classifier', LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

new_overview = ["A thrilling adventure of a young hero in a fantastical world."]
predicted_genre = pipeline.predict(new_overview)
print("Exemplo de overview:", new_overview)
print(f"Predicted Genre: {predicted_genre[0]}")



# Célula 7: Modelo de regressão linear para previsão da nota IMDB
filtered_data = data.dropna(subset=['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross', 'Star1', 'Genre', 'Runtime'])
filtered_data['Runtime'] = filtered_data['Runtime'].str.replace(' min', '').astype(int)
filtered_data = pd.get_dummies(filtered_data, columns=['Genre', 'Certificate'], drop_first=True)

X = filtered_data.drop(columns=['Series_Title', 'Released_Year', 'Overview', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'IMDB_Rating'])
y = filtered_data['IMDB_Rating']
model = LinearRegression()
model.fit(X, y)

new_movie = {
    'Meta_score': 80.0,
    'No_of_Votes': 2343110,
    'Gross': 28341469.0,
    'Runtime': 142,
    'Genre_Drama': 1,
    'Certificate_A': 1,
}

# Preencher 0 para colunas de gênero e certificado não presentes no novo filme
for col in X.columns:
    if col not in new_movie:
        new_movie[col] = 0

# Converter para DataFrame e garantir a mesma ordem das colunas
new_movie_df = pd.DataFrame([new_movie], columns=X.columns)

# Fazer a previsão
imdb_rating_prediction = model.predict(new_movie_df)

print(f"A nota prevista do IMDB para o filme é: {imdb_rating_prediction[0]}")
