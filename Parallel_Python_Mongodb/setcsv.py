import csv

def setcsv(cId_list):
    f = open('./cId.csv','w')
    csvWriter = csv.writer(f)

    csvWriter.writerow(cId_list)
    f.close()
