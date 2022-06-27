import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

metadata = pd.read_csv('movies_metadata.csv', low_memory=False)
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

# Remova as linhas com IDs inválidos.
metadata = metadata.drop([19730, 29503, 35587])

# manipulação dos dados para melhor uso posterior

# Converter IDs em int. Necessário para mesclar
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# Mescle palavras-chave e créditos em seu dataframe de metadados principal
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Verifique se existem mais de 3 elementos. Se sim, retorne apenas os três primeiros. Se não, retorna a lista inteira.
        if len(names) > 3:
            names = names[:3]
        return names
    return []

# Defina novos recursos de diretor, elenco, gêneros e palavras-chave que estejam de forma adequada.
metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

# Função para converter todas as strings para letras minúsculas e retirar nomes de espaços
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Verifique se o diretório existe. Se não, retorna uma string vazia
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Aplique a função clean_data aos seus recursos
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

# Criar um novo recurso de sopa
metadata['soup'] = metadata.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])

# Calcula a matriz de similaridade de cossenos com base na count_matrix
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Redefina o índice do seu DataFrame principal e construa o mapeamento reverso
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])

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
print(get_recommendations('The Dark Knight Rises',cosine_sim2))
print("")
print('Recomendações para: The Godfather')
print(get_recommendations('The Godfather',cosine_sim2))
print("")
print('Recomendações para: Fight Club')
print(get_recommendations('Fight Club',cosine_sim2))
print("")
print('Recomendações para: The Matrix')
print(get_recommendations('The Matrix',cosine_sim2))
