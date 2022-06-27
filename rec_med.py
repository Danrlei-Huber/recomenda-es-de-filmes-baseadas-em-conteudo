import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Defina um Objeto Vetorizador TF-IDF. Remova todas as palavras de parada em inglês, como 'the', 'a'
# tfidf é a lista dos nomes que aparecem nos enredos
tfidf = TfidfVectorizer(stop_words='english')

# Substitua NaN por uma string vazia
metadata['overview'] = metadata['overview'].fillna('')

# Construa a matriz TF-IDF necessária ajustando e transformando os dados
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Calcular a matriz de similaridade de cosseno
# a linear_kernel pegou a matrix de palavras (filmes x palavras) e 
# trasformou uma relação de proximidade entre os filmes usando os valores de 
# cosseno (variando de -1 até 1)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construir um mapa reverso de índices e títulos de filmes
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

# Função que recebe o título do filme como entrada e produz a maioria dos filmes semelhantes
def get_recommendations(title, cosine_sim):
    # Obtenha o índice do filme que corresponde ao título
    idx = indices[title]

    # Obtenha as pontuações de semelhança em pares de todos os filmes com esse filme
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Classifique os filmes com base nas pontuações de semelhança
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtenha as pontuações dos 10 filmes mais parecidos
    sim_scores = sim_scores[1:11]

    # Obtenha os índices de filmes
    movie_indices = [i[0] for i in sim_scores]

    # Retorne os 10 filmes mais parecidos
    return metadata['title'].iloc[movie_indices]
print("")
print('Recomendações para: The Dark Knight Rises')
print(get_recommendations('The Dark Knight Rises',cosine_sim))
print("")
print('Recomendações para: The Godfather')
print(get_recommendations('The Godfather',cosine_sim))
print("")
print('Recomendações para: Fight Club')
print(get_recommendations('Fight Club',cosine_sim))
print("")
print('Recomendações para: The Matrix')
print(get_recommendations('The Matrix',cosine_sim))