import csv

def predict(data,model):
    flag = True
    for i,col in enumerate(model):
        if col!=data[i] and col!="?":
            flag = False
            break

    return flag



def train():
    S = ['0','0','0','0','0','0']
    G = [['?','?','?','?','?','?']]

    data = csv.reader(open("Dataset2.csv","r"),delimiter=',')

    for i,row in enumerate(data):
        #positive examples
        print(row)
        if row[-1]=="P":
            for j,col in enumerate(row[:-1]):
                if col!=S[j] and S[j]=="0":
                    S[j] = col
                elif col!=S[j] and S[j]!="0":
                    S[j] = "?"
        
            n = len(G)
            m = 0
            while m<n:
                if predict(row[:-1],G[m])==False:
                    G = G[:m] + G[m+1:]
                    print(G)
                    n = n-1
                m = m + 1

        #negative examples

        print("S",(i+1),":",S)
        
        print("G",(i+1),":",G)
        print("\n")

            

train()
