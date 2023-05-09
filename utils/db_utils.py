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
                        document["data"][count][key]["total_memory"] = [(x + y)/2 for x,y in zip(value["total_memory_server"] ,value["total_memory_client"])]
        return document["data"] if document else None

#
# class PostgresHandler:
#     """A class to handle PostgreSQL operations."""
#
#     DB_NAME = "masterthesis"
#     TABLE_NAME = "benchmarks"
#
#     def __init__(self):
#         """
#         Initialize the PostgresHandler class.
#         """
#         self.conn = self.connect_to_postgres()
#
#     def connect_to_postgres(self):
#         """
#         Connect to the PostgreSQL database.
#
#         :return: A connection object.
#         :rtype: psycopg2.extensions.connection
#         """
#         with open(SECRETS_FILE, "r") as file:
#             secrets = yaml.safe_load(file)
#             db_config = secrets["postgres"]
#
#         return psycopg2.connect(
#             dbname=db_config["dbname"],
#             user=db_config["user"],
#             password=db_config["password"],
#             host=db_config["host"],
#             port=db_config["port"],
#         )
#
#     def load_dataframe(self, df, name):
#         """
#         Load a pandas DataFrame into a PostgreSQL table.
#
#         :param df: The DataFrame to be loaded.
#         :param name: The name of the data to later load it with this name.
#         """
#         df["name"] = name
#         df.to_sql("benchmarks", con=self.conn, if_exists='append')
#
#     def get_data_by_name(self, name):
#         """
#         Get data from a PostgreSQL table row by its name.
#         :param name: The name of the row.
#         :return: The data from the row or None if the row doesn't exist.
#         """
#         query = sql.SQL(f"SELECT * FROM benchmark WHERE name = %s")
#         with self.conn.cursor() as cursor:
#             cursor.execute(query, (name,))
#             row = cursor.fetchone()
#             if row:
#                 column_names = [desc[0] for desc in cursor.description]
#                 return dict(zip(column_names, row))
#             else:
#                 return None