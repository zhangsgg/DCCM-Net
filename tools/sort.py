import pandas

filepath="/home/zsg/tmp/segment/runs/MyNet_Resnet34_SDIN/acc.csv"
def sortresult(filepath):
    data=pandas.read_csv(filepath,header=None)
    sorted_data = data.sort_values(by=data.columns[5], ascending=False)
    sorted_data.to_csv(filepath[:-4] + '_sort.csv', header=False, index=False)
# sortresult(filepath)
