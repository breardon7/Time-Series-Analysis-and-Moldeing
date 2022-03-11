m1 = 3

if m1 % 2 == 0:
    upper_trail = int(m1 / 2)
    lower_trail = int(m1 / 2 - 1)
else:
    upper_trail = int((m1 - 1) / 2)
    lower_trail = int((m1 - 1) / 2)

print(upper_trail, lower_trail)

m1 = 4

if m1 % 2 == 0:
    upper_trail = int(m1 / 2)
    lower_trail = int(m1 / 2 - 1)
else:
    upper_trail = int((m1 - 1) / 2)
    lower_trail = int((m1 - 1) / 2)

print(upper_trail, lower_trail)