import pandas as pd

metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

C = metadata['vote_average'].mean()

# Calcule o número mínimo de votos necessários para estar no gráfico, m
m = metadata['vote_count'].quantile(0.90)

# Filtre todos os filmes qualificados em um novo DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]

# Função que calcula a classificação ponderada de cada filme
# C = mediana dos votos e m = mumero de votos considerando o percentil 90
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Defina um novo recurso 'score' e calcule seu valor com `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

# Classifique os filmes com base na pontuação calculada acima
q_movies = q_movies.sort_values('score', ascending=False)

print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))

