# Python 3 code to rename multiple
# files in a directory or folder
 
# importing os module
import os
 
# Function to rename multiple files
def main(emotion):
    first_two = emotion[0:2]
    folder = "C:/Users/User/Project 4/archive/"+emotion
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"{first_two}_{str(count)}.png"
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder}/{dst}"
         
        # rename() function will
        # rename all the files
        os.rename(src, dst)
 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main("anger")
    main("contempt")
    main("disgust")
    main("fear")
    main("happiness")
    main("neutrality")
    main("sadness")
    main("surprise")