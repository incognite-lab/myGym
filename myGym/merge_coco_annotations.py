from jsonmerge import Merger
import json, os
import argparse

schema = {"properties": {
    "images": {"mergeStrategy": "append"
               },
    "annotations": {"mergeStrategy": "append"
                    }
}
}

def merge_jsons(dir, clean_files = True):
    json_list = [pos_json for pos_json in os.listdir(dir) if pos_json.endswith('.json')]
    if "annotations.json" in json_list:
        json_list.remove("annotations.json")
    json_list = sorted(json_list, key=lambda i: int(i.split("_")[1])) #not necessary, comment out if names of jsons don't contain numbers
    merger = Merger(schema)
    annot = None
    for i in json_list:
        with open(os.path.join(dir, i), "r") as f:
            if annot is None:
                annot = json.load(f)
            else:
                a = json.load(f)
                annot = merger.merge(annot, a)
            if clean_files:
                os.remove(os.path.join(args.d, i))

    with open(os.path.join(args.d, "annotations.json"), "w") as f:
        json.dump(annot, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "--dir", type=str, help="Paths to the directories with jsons to be merged")
    parser.add_argument("--clean-files", action='store_true', help="Whether to delete individual jsons after merge")
    args = parser.parse_args()
    merge_jsons(args.d, args.clean_files)