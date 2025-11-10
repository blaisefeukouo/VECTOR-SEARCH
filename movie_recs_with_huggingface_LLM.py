import pymongo
import requests

uri = "mongodb+srv://admin:admin@cluster0.jurxqio.mongodb.net/?appName=Cluster0"

client = pymongo.MongoClient(uri)
db = client.sample_mflix
movies_collection = db.movies

hf_token = "hf_token"
embedding_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

#Generate the embedding
def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        embedding_url,
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": text}
    )
    
    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
    
    return response.json()

""" # Create vector embedding for 50 documents in our dataset that have the field plot
## embedding will be created based on the plot
for doc in movies_collection.find({'plot':{"$exists":True}}).limit(50):
    #print("Creating embedding for plot {doc['plot']}")
    doc['plot_embedding_hf']=generate_embedding(doc['plot'])
    movies_collection.replace_one({'_id':doc['_id']}, doc) """

#Try out and use the embedding
query = "imginary characters from ourter space at war"

#Search in the collection where the plot_embedding_hf field is semantically simailar to the query
result = movies_collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": generate_embedding(query),
            "path": "plot_embedding_hf",
            "numCandidates": 100,
            "limit": 4,
            "index": "PlotSemanticSearch"
        }
    }
]);

for document in result:
    print(f'Movie Name: {document["title"]}, \nMovie Plot: {document["plot"]}\n')