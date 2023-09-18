a = [{'a':3,'b':4},{'a':4,'b':5}]
c = dict()
for key in a[0].keys():
    c[key] = a[0][key] * 0
print(c)
for key in a[0].keys():
    for i in range(2):
        c[key] += a[i][key]
print(c)