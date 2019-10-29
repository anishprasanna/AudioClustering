import glob

def main():
    all_files = glob.glob('assignment5/data/*.wav') #assignment5/data
    all_files.sort()
    print(len(all_files))
    
    # count = 0
    # for file in all_files:
    #     count+=1
    # print(count)
if __name__ == "__main__":
    main()