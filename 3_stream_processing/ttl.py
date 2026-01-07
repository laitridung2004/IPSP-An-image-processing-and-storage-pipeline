import os
from pymongo import MongoClient, ASCENDING 

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://root:bigdataproject@localhost:27017") 
client = MongoClient(MONGO_URI)
db = client["traffic_db"]
collection = db["realtime_monitor"]

print("Connected to MongoDB. Creating Indexes...")

collection.create_index([("camera_id", ASCENDING), ("timestamp", ASCENDING)])
print("Index created.")