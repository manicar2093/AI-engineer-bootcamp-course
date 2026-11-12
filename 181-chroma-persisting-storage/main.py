import chromadb
from chromadb.utils import embedding_functions

default_ef = embedding_functions.DefaultEmbeddingFunction()

client = chromadb.PersistentClient(path="./181-chroma-persisting-storage/chroma_persisted")

collection = client.get_or_create_collection("test_persisted_collection", embedding_function=default_ef)

documents = [
    {"id": "dc1", "text": "The Milky Way galaxy is estimated to contain 100-400 billion stars."},
    {"id": "dc2", "text": "Black holes are so dense that a sugar cube could be compressed to the size of a proton if placed near one."},
    {"id": "dc3", "text": "The Andromeda galaxy is the closest spiral galaxy to our Milky Way, about 2.5 million light-years away."},
    {"id": "dc4", "text": "Type Ia supernovae are used as standard candles because of their consistent brightness."},
    {"id": "dc5", "text": "Venus is the only planet in our solar system that doesn't have moons."},
    {"id": "dc6", "text": "The Sun is on the edge of the Milky Way's galactic disk and is moving towards the Milky Way's center at about 7.35 km/s."},
    {"id": "dc7", "text": "Astronomers have found water vapor in the atmosphere of Jupiter's moons, like Europa and Ganymede."},
    {"id": "dc8", "text": "The first stars in the universe were formed about 13.8 billion years ago, just a few hundred million years after the Big Bang."},
    {"id": "dc9", "text": "The Large Hadron Collider (LHC) simulates conditions similar to those in the early universe."},
    {"id": "dca", "text": "Hypothetical exoplanets known as 'hot Jupiters' can orbit their stars in just a few days."},
    {"id": "dc10", "text": "The universe's observable space extends about 46.5 billion light-years in diameter."},
    {"id": "dc11", "text": "Neptune's blue color is caused by its atmosphere absorbing sunlight at certain wavelengths."},
    {"id": "dc12", "text": "Galaxies are often categorized by their shapes: elliptical, spiral, or irregular."},
    {"id": "dc13", "text": "The speed of light is approximately 299,792 kilometers per second."},
    {"id": "dc14", "text": "Einstein's theory of relativity predicts that time can slow down near black holes."},
    {"id": "dc15", "text": "The Big Bang theory suggests the universe began with a singularity, a point with infinite density."},
    {"id": "dc16", "text": "The oldest stars in the universe are Population II stars, which were formed from the first generation of elements."},
    {"id": "dc17", "text": "Comet 67P/Churyumov-Gerasimenko is the first comet visited by a space probe (Rosetta) and will be replaced by its own tail."},
    {"id": "dc18", "text": "The Cassini mission studied Saturn for over a decade before its intentional destruction in 2017."},
    {"id": "dc19", "text": "The Crab Nebula is one of the most famous supernova remnants in the galaxy."},
    {"id": "dc20", "text": "The Great Attractor is a massive structure in the universe that influences the motion of galaxies."},
    {"id": "dc21", "text": "The James Webb Space Telescope is designed to observe the first stars and galaxies formed in the universe."},
    {"id": "dc22", "text": "The Milky Way and the Andromeda galaxy are expected to collide and merge into a single galaxy in about 4.5 billion years."},
    {"id": "dc23", "text": "The International Space Station (ISS) can be seen from Earth without binoculars, especially during night passes."},
    {"id": "dc24", "text": "Saturn has over 140 known moons, more than any other planet in the solar system."},
    {"id": "dc25", "text": "The first images from the James Webb Space Telescope showed distant galaxies forming the early universe."},
    {"id": "dc26", "text": "The universe's largest known structure is the Cosmic Web, a network of filaments and voids."},
    {"id": "dc27", "text": "The Hubble Space Telescope has captured images of the universe's most distant galaxies."},
    {"id": "dc28", "text": "Neutrinos are elementary particles with very low mass and are barely interacted with matter."},
    {"id": "dc29", "text": "The Big Bang theory explains the expansion of space, leading to the acceleration of the universe's expansion."},
    {"id": "dc30", "text": "The universe contains about 200 billion galaxies, with the Milky Way being one of them."}
]

for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]])


query = "What is the universe's largest known structure?"

results = collection.query(
    query_texts=[query],
    n_results=2
)

print(f"Query: {query}\n\n")

for i, doc in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][i]
    distance = results["distances"][0][i]
    print(f"Document ID: {doc_id}, Distance: {distance}, Text: {doc}")
