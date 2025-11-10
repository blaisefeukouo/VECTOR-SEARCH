import pymongo
from openai import OpenAI

openAiClient = OpenAI(api_key="token")

mongoClient = pymongo.MongoClient("mongodb+srv://user:passwd@cluster0.jurxqio.mongodb.net/?appName=Cluster0")
db = mongoClient.sample_mflix
movies_collection = db.embedded_movies

def generate_embedding(text: str) -> list[float]:
    response = openAiClient.embeddings.create(
        model="text-embedding-ada-002",
        input=text        
    )
    return response.data[0].embedding

""" 
# Create vector embedding for 50 documents in our dataset that have the field plot
## embedding will be created based on the plot
for doc in movies_collection.find({'plot':{"$exists":True}}).limit(50):
    #print("Creating embedding for plot {doc['plot']}")
    doc['plot_embedding_hf']=generate_embedding(doc['plot'])
    movies_collection.replace_one({'_id':doc['_id']}, doc) """

query = "imaginary characters from outer space at war"

result = movies_collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": generate_embedding(query),
            "path": "plot_embedding",
            "numCandidates": 100,
            "limit": 4,
            "index": "PlotSemanticSearch"
        }
    }
]); 
for document in result:
    print(f'Movie Name: {document["title"]}, \nMovie Plot: {document["plot"]}\n')   