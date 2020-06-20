# ig_scores = run_models(main_fun(
#     deepcopy(data),
#     rem_stop_words=True,
#     is_ig=True,
#     num_of_wds=2000
#     # pad_len=500
# ), adding="", epoch_s=epochs)
# print(data.iloc[0])


# cprint('MAX Accuracy table', 'green')
# table = BeautifulTable()
# table.set_style(STYLE_BOX)
# lis = ["name", "loss", "Precision", "Recall", "F1 score", "Accuracy", "time", "epoch"]
# table.column_headers = [colored(i, "red") for i in lis]
# for i, j in zip(ig_scores, simple_scores):
#     s = ig_scores[i][0][2]
#     f1_score = 2 * s[1] * s[2] / (s[1] + s[2]) if s[1] + s[2] != 0 else 0
#     table.append_row(
#         [i, s[0], s[1], s[2], f1_score, s[3], human_time(ig_scores[i][1]), ig_scores[i][0][3]])
#     s = simple_scores[j][0][2]
#     f1_score = 2 * s[1] * s[2] / (s[1] + s[2]) if s[1] + s[2] != 0 else 0
#     table.append_row([j, s[0], s[1], s[2], f1_score, s[3], human_time(simple_scores[j][1]),
#                       simple_scores[j][0][3]])
# print(table, end="\n\n\n")

# cprint('Last accuracy table', 'green')
# table = BeautifulTable()
# table.set_style(STYLE_BOX)
# lis = ["name", "loss", "Precision", "Recall", "F1 score", "Accuracy", "time", "epoch"]
# table.column_headers = [colored(i, "red") for i in lis]
# for i, j in zip(ig_scores, simple_scores):
#     s = ig_scores[i][0][0]
#     f1_score = 2 * s[1] * s[2] / (s[1] + s[2]) if s[1] + s[2] != 0 else 0
#     table.append_row(
#         [i, s[0], s[1], s[2], f1_score, s[3], human_time(ig_scores[i][1]), ig_scores[i][0][1]])
#     s = simple_scores[j][0][0]
#     f1_score = 2 * s[1] * s[2] / (s[1] + s[2]) if s[1] + s[2] != 0 else 0
#     table.append_row(
#         [j, s[0], s[1], s[2], f1_score, s[3], human_time(simple_scores[j][1]), simple_scores[j][0][1]])
# print(table)

# print("val-accuracy: ", history.history["val_accuracy"])
# print("loss: ", history.history["loss"])
# print("################################")
# print("Loss:", scores[0])
# print("Precision:", scores[1])
# print("Recall:", scores[2])
# print("F1 score:", (2 * scores[1] * scores[2]) / (scores[1] + scores[2]))

# scores.append(2 * scores[1] * scores[2]) / (scores[1] + scores[2])
# print("################################")
# print("Loss:", scores[0], end=" ")
# print("Precision:", scores[1])
# print("Recall:", scores[2], end=" ")
# print("Accuracy:", scores[3], end=" ")
# print("F1 score:", (2 * scores[1] * scores[2]) / (scores[1] + scores[2]))
# print("Sum of epochs:", epochs)
