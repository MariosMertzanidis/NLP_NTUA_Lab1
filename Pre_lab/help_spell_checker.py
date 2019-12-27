import sys

def create_T(s):
    for index,char in enumerate(s):
        if index+1==len(s):
            string = str(index+1)+" 0 "+char+" "+char+" 0\n"
        else:
            string = str(index+1)+" "+str(index+2)+" "+char+" "+char+" 0\n"
        print(string)

    print(str(0))

create_T(str(sys.argv[1]))
