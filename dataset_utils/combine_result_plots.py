import pickle

# Deserialize the object from the file
with open(
    "/home/dan/school/6th_semester/bachelor/bachelor_detection_pipeline/results2/balanced_mordiff_w_ubo_test_mipgan/plot_metrics.pkl",
    "rb",
) as f:
    fig1 = pickle.load(f)

with open(
    "/home/dan/school/6th_semester/bachelor/bachelor_detection_pipeline/results2/mordiff_test_ubo/plot_metrics.pkl",
    "rb",
) as f:
    fig2 = pickle.load(f)


accuracyPlotFig1 = fig1.axes[1]
accuracyPlotFig2 = fig2.axes[1]

firstPlotLine = fig.axes[0].get_lines()[0]
firstPlotLine.set_color("red")


fig.savefig("plot_metrics.png")
