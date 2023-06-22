import json


def join_experiments(experiments_paths, output_file_path, keys):

    output_file = dict()

    for key in keys:
        output_file_content = []
        for experiments_path in experiments_paths:
            output_file_content.extend(json.load(open(experiments_path))[key])

        output_file[key] = output_file_content

    with open(output_file_path, "w") as outfile:
        json.dump(output_file, outfile)

    return output_file


if __name__ == '__main__':
    experiments_paths = ['features_selection_and_optimization/DECISION_TREE/experiments_pt1.json',
                         'features_selection_and_optimization/DECISION_TREE/experiments_pt2.json']

    output_file_path = "features_selection_and_optimization/DECISION_TREE/experiments.json"

    keys = ['DECISION_TREE']

    print(join_experiments(experiments_paths, output_file_path, keys))

