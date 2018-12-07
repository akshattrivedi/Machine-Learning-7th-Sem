import csv

my_list = list(csv.reader(open("Dataset1.csv","r"),delimiter=","))

h=['0','0','0','0','0','0']

for i in my_list:
    print(i)
    if i[-1]=="Y":
        j=0
        for x in i:
            if x!=h[j] and h[j]=="0":
                h[j] = x;
            elif x!=h[j] and h[j]!="0":
                h[j] = "?"
            j = j + 1

print("The Specific Hypothesis is:")
print(h)


