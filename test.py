import glob

all_files = glob.glob('assignment5/data/*.wav') #assignment5/data
all_files.sort()

l1st = [['assignment5/data/203654.wav', 'assignment5/data/208652.wav', 'assignment5/data/205878.wav', 'assignment5/data/206037.wav', 'assignment5/data/209864.wav'], ['assignment5/data/204765.wav', 'assignment5/data/203913.wav', 'assignment5/data/205610.wav'], ['assignment5/data/204773.wav', 'assignment5/data/209672.wav', 'assignment5/data/207962.wav', 'assignment5/data/207124.wav', 'assignment5/data/204067.wav'], ['assignment5/data/204408.wav', 'assignment5/data/204240.wav']]

for item in l1st:
    for subitem in item:
        subitem = subitem.replace('assignment5/data/', '')
        subitem = subitem.replace('.wav', '')
        print(subitem)


zipped = 