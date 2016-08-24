import csv
import numpy as np

csv_file_object = csv.reader(open('train.csv', 'rb'))
header = csv_file_object.next()

data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)

fare_ceiling = 40

#modify the data in the Fare column to = 39, if it is greater or equal to the ceiling
data[data[0::, 9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1.0

fare_bucket_size = 10
num_price_brackets = fare_ceiling / fare_bucket_size

num_classes = len(np.unique(data[0::, 2]))

#create a 2x3x4 table
survival_table = np.zeros((2, num_classes, num_price_brackets))

for i in xrange(num_classes):
    for j in xrange(num_price_brackets):
        women_only_stats = data[ \
            #female
            (data[0::, 4] == 'female') \
            #is of class i
            &(data[0::, 2].astype(np.float) == i + 1) \
            #has fare in bucket j
            &(data[0::, 9].astype(np.float) >= j * fare_bucket_size) \
            &(data[0::, 9].astype(np.float) < (j+1) * fare_bucket_size) \
            , 1]

        men_only_stats = data[ \
            #male
            (data[0::, 4] != 'female') \
            #is of class i
            &(data[0::, 2].astype(np.float) == i + 1) \
            #has fare in bucket j
            &(data[0::, 9].astype(np.float) >= j * fare_bucket_size) \
            &(data[0::, 9].astype(np.float) < (j+1) * fare_bucket_size) \
            , 1]

        survival_table[0, i, j] = np.mean(women_only_stats.astype(np.float))
        survival_table[1, i, j] = np.mean(men_only_stats.astype(np.float))

survival_table[ survival_table != survival_table ] = 0.

survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1


print survival_table


#Read in the test file
test_file_object = csv.reader(open('test.csv', 'rb'))
header = test_file_object.next()

#Create new file to store results
prediction_file = open('genderClassFare.csv', 'wb')
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(['PassengerId', 'Survived'])
for row in test_file_object:
    for j in xrange(num_price_brackets):
        try:
            row[8] = float(row[8])
        except:
            bin_fare = 3 - float(row[1])
            break
        if row[8] > fare_ceiling:
            bin_fare = num_price_brackets - 1
            break
        if row[8] >= j * fare_bucket_size and row[8] < (j + 1) * fare_bucket_size:
            bin_fare = j
            break
    prediction_file_object.writerow([row[0], "%d" % int(survival_table[0 if row[3] == 'female' else 1, float(row[1])-1, bin_fare])])

prediction_file.close()



