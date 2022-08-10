import logging
import traceback
from pathlib import Path

import google.cloud.storage as storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

class CloudStorageUtilities:
    def __init__(self, config_file_path):
        try:
            self.storage_client = storage.Client.from_service_account_json(config_file_path)
        except Exception as e:
            logger.error("{0}\n{1}".format(e, traceback.format_exc()))

    def get_list_of_buckets(self):
        list_of_buckets = []
        try:
            list_of_buckets = list(self.storage_client.list_buckets())
        except Exception as e:
            logger.error("{0}\n{1}".format(e, traceback.format_exc()))
        return list_of_buckets

    def write_data(self, data, bucket_name, file_key, content_type="text/plain"):
        status = False
        try:
            bucket = self.storage_client.get_bucket(bucket_name)
            file_obj = bucket.blob(file_key)
            print("Writing data into : {0}/{1}".format(bucket_name, file_key))
            file_obj.upload_from_string(data=data, content_type=content_type)
            status = True
        except Exception as e:
            logger.error("{0}\n{1}".format(e, traceback.format_exc()))
        return status

    def read_data(self, bucket_name, file_key):
        data = None
        try:
            bucket_obj = self.storage_client.get_bucket(bucket_name)
            blob_obj = bucket_obj.get_blob(file_key)
            data = blob_obj.download_as_text()
        except Exception as e:
            logger.error("{0}\n{1}".format(e, traceback.format_exc()))
        return data