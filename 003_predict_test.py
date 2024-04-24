import pandas as pd

# Manufacture test record dataset

test_info=pd.read_csv(os.getcwd() + "/nfl-player-contact-detection/sample_submission.csv")
test_info["game_play"] = list(map(lambda text: text.split("_")[0] + "_" + text.split("_")[1], test_info["contact_id"]))
test_info["step"] = list(map(lambda text: text.split("_")[2], test_info["contact_id"]))
test_info["nfl_player_id_1"] = list(map(lambda text: text.split("_")[3], test_info["contact_id"]))
test_info["nfl_player_id_2"] = list(map(lambda text: text.split("_")[4], test_info["contact_id"]))
test_info["contact"]=np.NaN
test_info.to_csv(os.getcwd() + "/nfl-player-contact-detection/test_labels.csv")