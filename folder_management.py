import os

SUBMISSION_FOLDER = r"./data/submission_folder"


def get_name_iteration():
    submission_files = [
        file_name
        for file_name in os.listdir(SUBMISSION_FOLDER)
        if file_name.endswith(".csv")
    ]

    if not submission_files:
        next_file_name = "test_prediction_1.csv"
        return os.path.join(SUBMISSION_FOLDER, next_file_name)

    valid_submission_files = [
        file_name[16:-4].replace("_", "")
        for file_name in submission_files
        if file_name.startswith("test_prediction")
    ]
    valid_submission_files.sort()

    while len(valid_submission_files) > 0 and not valid_submission_files[-1].isnumeric():
        del valid_submission_files[-1]

    if len(valid_submission_files) == 0:
        next_file_name = "test_prediction_1.csv"
        return os.path.join(SUBMISSION_FOLDER, next_file_name)

    next_file_num = int(valid_submission_files[-1]) + 1
    next_file_name = "test_prediction_" + str(next_file_num) + ".csv"
    return os.path.join(SUBMISSION_FOLDER, next_file_name)
