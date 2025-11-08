import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Kết nối Mongo
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://linhtng22416c_db_user:33y8XJiZhcLlOXf5@cluster0.jbned6v.mongodb.net/?appName=Cluster0")
DB_NAME = os.getenv("DB_NAME", "fruit_detection")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Định nghĩa các collection (rất quan trọng)
Models = db["models"]
Images = db["images"]
Detections = db["detections"]
Predictions = db["predictions"]
