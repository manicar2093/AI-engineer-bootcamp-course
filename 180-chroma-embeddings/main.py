from chromadb.utils import embedding_functions

default_ef = embedding_functions.DefaultEmbeddingFunction()

name = "Mani"

embedded = default_ef(name)

print(embedded)
