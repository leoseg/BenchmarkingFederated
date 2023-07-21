import yaml
from pymongo import MongoClient
SECRETS_FILE = "../../utils/secrets.yaml"

class MongoDBHandler:
    """A class to handle MongoDB operations."""

    DB_NAME = "masterthesis"
    COLLECTION_NAME = "benchmarks"

    def __init__(self):
        """
        Initialize the MongoDBHandler class.
        """
        self.client = MongoClient(self.load_mongodb_url())
        self.db = self.client[self.DB_NAME]
        self.collection = self.db[self.COLLECTION_NAME]

    def update_benchmark_data(self, data, name):
        """
        Update or insert a document with the specified data and name.

        :param data: The dictionary to be added to the document.
        :type data: dict
        :param name: The name of the document.
        :type name: str
        """
        existing_document = self.collection.find_one({"name": name})

        if existing_document:
            self.collection.update_one({"name": name}, {"$set": {"data": data}})
        else:
            self.collection.insert_one({"data": data, "name": name})

    @staticmethod
    def load_mongodb_url():
        """
        Load the MongoDB URL from the specified YAML file.

        :return: The MongoDB URL.
        :rtype: str
        """
        with open(SECRETS_FILE, "r") as file:
            secrets = yaml.safe_load(file)
            return secrets["mongodb_adress"]

    def get_data_by_name(self, name, calc_total_memory=False) ->list:
        """
        Get data from a document by its name.

        :param name: The name of the document.
        :type name: str
        :return: The data from the document or None if the document doesn't exist.
        :rtype: dict or None
        """
        document = self.collection.find_one({"name": name})
        if calc_total_memory:
            for count, round in enumerate(document["data"]):
                for key, value in round.items():
                    if "total_memory_server" in value.keys() and "total_memory_client" in value.keys():
                        document["data"][count][key]["total_memory"] = [(x + y) for x,y in zip(value["total_memory_server"] ,value["total_memory_client"])]
        return document["data"] if document else None

