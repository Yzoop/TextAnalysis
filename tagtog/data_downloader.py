from datetime import datetime
from tqdm import tqdm
import requests
import json
import os

TAGTOGURL = "https://www.tagtog.net/-api/documents/v1"

AUTH = requests.auth.HTTPBasicAuth(username=os.environ["username"],
                                   password=os.environ["password"])

ENTITY_TYPE_ID = {"m_17": "HISTORY", "m_7": "CULINARY", "m_18": "IT", "m_13": "POLITICS", "m_8": "TECHNOLOGY",
                  "m_3": "FINANCE", "m_10": "AGRICULTURE", "m_14": "ANIMALS", "m_9": "FASION", "m_5": "JOKES",
                  "m_21": "SOCIALMEDIA", "m_4": "SCIENCE", "m_11": "CULTURE", "m_15": "MUSIC", "m_20": "CARS",
                  "m_6": "MEDICINE", "m_19": "CRYPTO", "m_16": "ART", "m_12": "GEOGRAPHIC"}


def baseline_request(auth=AUTH):
    params = {"project": os.environ["project"], "owner": os.environ["owner"], "search": "folder:pool"}
    response = requests.get(TAGTOGURL, params=params, auth=auth)
    if response.status_code == 200:
        response_dict = json.loads(response.text)
        n_pages = response_dict["pages"]["numPages"]
        return n_pages
    else:
        raise ConnectionError("could not make a request. Status code=", response.status_code)


def annotation_request(id: str):
    params = {"owner": os.environ["owner"],
              "project": os.environ["project"],
              'ids': id,
              "output": "ann.json"}
    response = requests.get(TAGTOGURL, params=params, auth=AUTH)
    if response.status_code == 200:
        response_dict = json.loads(response.text)
        labels = [ENTITY_TYPE_ID[classid] for classid, data in response_dict["metas"].items() if data["value"]]
        return labels
    else:
        raise ConnectionError(
            f"error while getting annotations.\nStatus Code={response.status_code}; reason={response.text}")


def text_request(id: str):
    params = {"owner": os.environ["owner"],
              "project": os.environ["project"],
              'ids': id,
              "output": "orig"}
    response = requests.get(TAGTOGURL, params=params, auth=AUTH)
    if response.status_code == 200:
        return response.text
    else:
        raise ConnectionError(
            f"error while getting texts.\nStatus Code={response.status_code}; reason={response.text}")


def docdata_request(n_pages, auth=AUTH, filter_nonlabeled=False):
    docs = []
    for page_i in range(n_pages):
        params_docids = {"project": os.environ["project"],
                         "owner": os.environ["owner"],
                         "search": "folder:pool",
                         "page": page_i}
        response = requests.get(TAGTOGURL, params=params_docids, auth=auth)
        response_dict = json.loads(response.text)
        batch_ids = [{"text": text_request(doc["id"]),
                      "annotation": annotation_request(doc["id"])} for doc in tqdm(response_dict["docs"])]
        docs.extend(batch_ids)
    if filter_nonlabeled:
        docs = list(filter(lambda doc: len(doc["annotation"]) > 0, docs))
    return docs



if __name__ == "__main__":
    n_pages = baseline_request(auth=AUTH)
    docs_data = docdata_request(n_pages=n_pages, auth=AUTH, filter_nonlabeled=True)
    print(f"successfully downloaded all the data. Found {len(docs_data)}")
    with open(f"../data/text_classification_dataset_{datetime.today().strftime('%Y-%m-%d')}.json",
              'w+',
              encoding="utf-8") as f:
        json.dump(docs_data, f, indent=4, ensure_ascii=False)

