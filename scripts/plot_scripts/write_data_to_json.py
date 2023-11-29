from datapostprocessing.db_utils import MongoDBHandler

mongodb = MongoDBHandler()

mongodb.store_all_docs_as_json()
